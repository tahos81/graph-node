use std::{
    collections::HashMap,
    ops::{Add, Deref},
    sync::Arc,
};

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};

use itertools::Itertools;
use slog::Logger;
use thiserror::Error;

use crate::{
    blockchain::ChainIdentifier, cheap_clone::CheapClone, prelude::error, tokio::sync::RwLock,
};

const VALIDATION_ATTEMPT_TTL_SECONDS: i64 = 60 * 5;

#[derive(Debug, Error)]
pub enum ProviderManagerError {
    #[error("unknown error {0}")]
    Unknown(#[from] anyhow::Error),
    #[error("provider {provider} failed verification, expected ident {expected}, got {actual}")]
    ProviderFailedValidation {
        provider: ProviderName,
        expected: ChainIdentifier,
        actual: ChainIdentifier,
    },
    #[error("no providers available for chain {0}")]
    NoProvidersAvailable(ChainId),
    #[error("all providers for chain_id {0} have failed")]
    AllProvidersFailed(ChainId),
}

#[async_trait]
pub trait NetIdentifiable: Sync + Send {
    async fn net_identifiers(&self) -> Result<ChainIdentifier, anyhow::Error>;
    fn provider_name(&self) -> ProviderName;
}

pub type ProviderName = String;
pub type ChainId = String;

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
struct Ident {
    provider: ProviderName,
    chain_id: ChainId,
}

/// ProviderCorrectness will maintain a list of providers which have had their
/// ChainIdentifiers checked. The first identifier is considered correct, if a later
/// provider for the same chain offers a different ChainIdentifier, this will be considered a
/// failed validation and it will be disabled.
#[derive(Debug)]
pub struct ProviderManager<T: NetIdentifiable + Clone> {
    inner: Arc<Inner<T>>,
}

impl<T: NetIdentifiable + Clone + 'static> ProviderManager<T> {
    pub fn new(
        logger: Logger,
        adapters: impl Iterator<Item = (ChainId, Vec<T>)>,
        expected_idents: impl Iterator<Item = (ChainId, Option<ChainIdentifier>)>,
    ) -> Self {
        let mut status: Vec<(Ident, RwLock<GenesisCheckStatus>)> = Vec::new();
        let expected_idents = HashMap::from_iter(expected_idents);

        let adapters = HashMap::from_iter(adapters.map(|(chain_id, adapters)| {
            let adapters = adapters
                .into_iter()
                .map(|adapter| {
                    let name = adapter.provider_name();

                    // Get status index or add new status.
                    let index = match status
                        .iter()
                        .find_position(|(ident, _)| ident.provider.eq(&name))
                    {
                        Some((index, _)) => index,
                        None => {
                            status.push((
                                Ident {
                                    provider: name,
                                    chain_id: chain_id.clone(),
                                },
                                RwLock::new(GenesisCheckStatus::NotChecked),
                            ));
                            status.len() - 1
                        }
                    };
                    (index, adapter)
                })
                .collect_vec();

            (chain_id, adapters)
        }));

        Self {
            inner: Arc::new(Inner {
                logger,
                adapters,
                status,
                expected_idents,
            }),
        }
    }

    async fn verify(&self, adapters: &Vec<(usize, T)>) -> Result<(), ProviderManagerError> {
        let mut tasks = vec![];

        for (index, adapter) in adapters.into_iter() {
            let inner = self.inner.cheap_clone();
            let adapter = adapter.clone();
            let index = *index;
            tasks.push(inner.verify_provider(index, adapter));
        }

        crate::futures03::future::join_all(tasks)
            .await
            .into_iter()
            .collect::<Result<Vec<()>, ProviderManagerError>>()?;

        Ok(())
    }

    pub async fn get_all(&self, chain_id: ChainId) -> Result<Vec<&T>, ProviderManagerError> {
        let adapters = match self.inner.adapters.get(&chain_id) {
            Some(adapters) if !adapters.is_empty() => adapters,
            _ => return Ok(vec![]),
        };

        // Optimistic check
        if self.inner.is_all_verified(&adapters).await {
            return Ok(adapters.iter().map(|v| &v.1).collect());
        }

        match self.verify(adapters).await {
            Ok(_) => {}
            Err(error) => error!(
                self.inner.logger,
                "unable to verify genesis for adapter: {}",
                error.to_string()
            ),
        }

        self.inner.get_verified_for_chain(&chain_id).await
    }
}

#[derive(Debug)]
struct Inner<T: NetIdentifiable> {
    logger: Logger,
    // Most operations start by getting the value so we keep track of the index to minimize the
    // locked surface.
    adapters: HashMap<ChainId, Vec<(usize, T)>>,
    // Status per (ChainId, ProviderName) pair. The RwLock here helps prevent multiple concurrent
    // checks for the same provider, when one provider is being checked, all other uses will wait,
    // this is correct because no provider should be used until they have been validated.
    // There shouldn't be many values here so Vec is fine even if less ergonomic, because we track
    // the index alongside the adapter it should be O(1) after initialization.
    status: Vec<(Ident, RwLock<GenesisCheckStatus>)>,
    // When an adapter is not available at the start and is a new chain, an identifier will not
    // be available. This will be set to None
    expected_idents: HashMap<ChainId, Option<ChainIdentifier>>,
}

impl<T: NetIdentifiable + 'static> Inner<T> {
    async fn is_all_verified(&self, adapters: &Vec<(usize, T)>) -> bool {
        for (index, _) in adapters.iter() {
            let status = self.status.get(*index).unwrap().1.read().await;
            if *status != GenesisCheckStatus::Valid {
                return false;
            }
        }

        true
    }

    /// Returns any adapters that have been validated, empty if none are defined or an error if
    /// all adapters have failed or are unavailable, returns different errors for these use cases
    /// so that that caller can handle the different situations, as one is permanent and the other
    /// is retryable.
    async fn get_verified_for_chain(
        &self,
        chain_id: &ChainId,
    ) -> Result<Vec<&T>, ProviderManagerError> {
        let mut out = vec![];
        let adapters = match self.adapters.get(chain_id) {
            Some(adapters) if !adapters.is_empty() => adapters,
            _ => return Ok(vec![]),
        };

        let mut failed = 0;
        for (index, adapter) in adapters.iter() {
            let status = self.status.get(*index).unwrap().1.read().await;
            match status.deref() {
                GenesisCheckStatus::Valid => {}
                GenesisCheckStatus::Failed => {
                    failed += 1;
                    continue;
                }
                GenesisCheckStatus::NotChecked | GenesisCheckStatus::TemporaryFailure { .. } => {
                    continue
                }
            }
            out.push(adapter);
        }

        if out.is_empty() {
            if failed == adapters.len() {
                return Err(ProviderManagerError::AllProvidersFailed(
                    chain_id.to_string(),
                ));
            }

            return Err(ProviderManagerError::NoProvidersAvailable(
                chain_id.to_string(),
            ));
        }

        Ok(out)
    }

    async fn get_ident_status(&self, index: usize) -> (Ident, GenesisCheckStatus) {
        match self.status.get(index) {
            Some(status) => (status.0.clone(), status.1.read().await.clone()),
            None => (Ident::default(), GenesisCheckStatus::Failed),
        }
    }

    fn ttl_has_elapsed(checked_at: &DateTime<Utc>) -> bool {
        checked_at.add(Duration::seconds(VALIDATION_ATTEMPT_TTL_SECONDS)) < Utc::now()
    }

    fn should_verify(status: &GenesisCheckStatus) -> bool {
        match status {
            GenesisCheckStatus::TemporaryFailure { checked_at }
                if Self::ttl_has_elapsed(checked_at) =>
            {
                true
            }
            // Let check the provider
            GenesisCheckStatus::NotChecked => true,
            _ => false,
        }
    }

    async fn verify_provider(
        self: Arc<Inner<T>>,
        index: usize,
        adapter: T,
    ) -> Result<(), ProviderManagerError> {
        let (ident, status) = self.get_ident_status(index).await;
        if !Self::should_verify(&status) {
            return Ok(());
        }

        let expected_ident = self.expected_idents.get(&ident.chain_id).unwrap_or(&None);
        // unwrap: If index didn't exist it would have failed the previous check so it's safe
        // to unwrap.
        let mut status = self.status.get(index).unwrap().1.write().await;
        // double check nothing has changed.
        if !Self::should_verify(&status) {
            return Ok(());
        }

        let expected_ident = match expected_ident {
            Some(ident) => ident,
            None => {
                *status = GenesisCheckStatus::Valid;
                return Ok(());
            }
        };

        match adapter.net_identifiers().await {
            Ok(ident) if ident.eq(expected_ident) => {
                *status = GenesisCheckStatus::Valid;
            }
            Ok(ident) => {
                *status = GenesisCheckStatus::Failed;
                return Err(ProviderManagerError::ProviderFailedValidation {
                    provider: adapter.provider_name(),
                    // unwrap: Safe to unwrap because None is caught by the previous condition
                    expected: expected_ident.clone(),
                    actual: ident,
                });
            }
            Err(err) => {
                error!(
                    &self.logger,
                    "failed to get net identifiers: {}",
                    err.to_string()
                );
                *status = GenesisCheckStatus::TemporaryFailure {
                    checked_at: Utc::now(),
                };

                return Err(err.into());
            }
        };

        Ok(())
    }
}

#[derive(Debug)]
struct Item<T> {
    index: u8,
    item: T,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GenesisCheckStatus {
    NotChecked,
    TemporaryFailure { checked_at: DateTime<Utc> },
    Valid,
    Failed,
}

#[cfg(test)]
mod test {
    use std::ops::{Deref, Sub};

    use crate::{
        bail,
        components::adapter::{ChainId, GenesisCheckStatus},
        prelude::lazy_static,
    };
    use async_trait::async_trait;
    use chrono::{DateTime, Duration, Utc};
    use slog::{o, Discard, Logger};
    use sqlparser::keywords::VALID;

    use crate::{
        blockchain::{BlockHash, ChainIdentifier},
        components::adapter::ProviderManagerError,
    };

    use super::{NetIdentifiable, ProviderManager, ProviderName, VALIDATION_ATTEMPT_TTL_SECONDS};

    const FAILED_CHAIN: &str = "failed";
    const VALID_CHAIN: &str = "valid";
    const TESTABLE_CHAIN: &str = "testable";
    const UNTESTABLE_CHAIN: &str = "untestable";
    const EMPTY_CHAIN: &str = "empty";

    lazy_static! {
        static ref VALID_IDENT: ChainIdentifier = ChainIdentifier {
            net_version: VALID_CHAIN.into(),
            genesis_block_hash: BlockHash::default(),
        };
        static ref FAILED_IDENT: ChainIdentifier = ChainIdentifier {
            net_version: FAILED_CHAIN.into(),
            genesis_block_hash: BlockHash::default(),
        };
        static ref UNTESTABLE_ADAPTER: MockAdapter =
                    MockAdapter{
                provider: UNTESTABLE_CHAIN.into(),
                ident: FAILED_IDENT.clone(),
                checked_at: Some(Utc::now()),
            };

                        // way past TTL, ready to check again
        static ref TESTABLE_ADAPTER: MockAdapter =
            MockAdapter{
                provider: TESTABLE_CHAIN.into(),
                ident: VALID_IDENT.clone(),
                checked_at: Some(Utc::now().sub(Duration::seconds(10000000))),
            };
        static ref VALID_ADAPTER: MockAdapter = MockAdapter::valid();
        static ref FAILED_ADAPTER: MockAdapter = MockAdapter::failed();
    }

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct MockAdapter {
        provider: String,
        ident: ChainIdentifier,
        checked_at: Option<DateTime<Utc>>,
    }

    impl MockAdapter {
        fn failed() -> Self {
            Self {
                provider: FAILED_CHAIN.into(),
                ident: FAILED_IDENT.clone(),
                checked_at: None,
            }
        }

        fn valid() -> Self {
            Self {
                provider: VALID_CHAIN.into(),
                ident: VALID_IDENT.clone(),
                checked_at: None,
            }
        }

        fn testable(ttl: DateTime<Utc>) -> Self {
            Self {
                provider: TESTABLE_CHAIN.into(),
                ident: VALID_IDENT.clone(),
                checked_at: Some(ttl),
            }
        }
    }

    #[async_trait]
    impl NetIdentifiable for MockAdapter {
        async fn net_identifiers(&self) -> Result<ChainIdentifier, anyhow::Error> {
            match self.checked_at {
                Some(checked_at)
                    if checked_at
                        > Utc::now().sub(Duration::seconds(VALIDATION_ATTEMPT_TTL_SECONDS)) =>
                {
                    unreachable!("should never check if ttl has not elapsed")
                }
                _ => {}
            }

            Ok(self.ident.clone())
        }
        fn provider_name(&self) -> ProviderName {
            self.provider.clone()
        }
    }

    #[tokio::test]
    async fn test_provider_manager() {
        struct Case<'a> {
            name: &'a str,
            chain_id: &'a str,
            status: Vec<(ProviderName, GenesisCheckStatus)>,
            adapters: Vec<(ChainId, Vec<MockAdapter>)>,
            idents: Vec<(ChainId, Option<ChainIdentifier>)>,
            expected: Result<Vec<&'a MockAdapter>, ProviderManagerError>,
        }

        let cases = vec![
            Case {
                name: "no adapters",
                chain_id: EMPTY_CHAIN,
                status: vec![],
                adapters: vec![],
                idents: vec![(VALID_CHAIN.into(), None)],
                expected: Ok(vec![]),
            },
            Case {
                name: "adapter temporary failure with Ident None",
                chain_id: VALID_CHAIN,
                status: vec![(
                    UNTESTABLE_CHAIN.to_string(),
                    GenesisCheckStatus::TemporaryFailure {
                        checked_at: UNTESTABLE_ADAPTER.checked_at.unwrap(),
                    },
                )],
                // UNTESTABLE_ADAPTER has failed ident, will be valid cause idents has None value
                adapters: vec![(VALID_CHAIN.into(), vec![UNTESTABLE_ADAPTER.clone()])],
                idents: vec![(VALID_CHAIN.into(), None)],
                expected: Err(ProviderManagerError::NoProvidersAvailable(
                    VALID_CHAIN.to_string(),
                )),
            },
            Case {
                name: "adapter temporary failure",
                chain_id: VALID_CHAIN,
                status: vec![(
                    UNTESTABLE_CHAIN.to_string(),
                    GenesisCheckStatus::TemporaryFailure {
                        checked_at: Utc::now(),
                    },
                )],
                adapters: vec![(VALID_CHAIN.into(), vec![UNTESTABLE_ADAPTER.clone()])],
                idents: vec![(VALID_CHAIN.into(), Some(FAILED_IDENT.clone()))],
                expected: Err(ProviderManagerError::NoProvidersAvailable(
                    VALID_CHAIN.to_string(),
                )),
            },
            Case {
                name: "chain ident None",
                chain_id: VALID_CHAIN,
                // Failed adapter has VALID_CHAIN as the ident, which is not validated if
                // the expected ident is None
                status: vec![],
                adapters: vec![(VALID_CHAIN.into(), vec![FAILED_ADAPTER.clone()])],
                idents: vec![(VALID_CHAIN.into(), None)],
                expected: Ok(vec![&FAILED_ADAPTER]),
            },
            Case {
                name: "wrong chain ident",
                chain_id: VALID_CHAIN,
                status: vec![],
                adapters: vec![(VALID_CHAIN.into(), vec![MockAdapter::failed()])],
                idents: vec![(VALID_CHAIN.into(), Some(VALID_IDENT.clone()))],
                expected: Err(ProviderManagerError::AllProvidersFailed(
                    VALID_CHAIN.to_string(),
                )),
            },
            Case {
                name: "all adapters ok or not checkable yet",
                chain_id: VALID_CHAIN,
                status: vec![(
                    FAILED_CHAIN.to_string(),
                    GenesisCheckStatus::TemporaryFailure {
                        checked_at: Utc::now(),
                    },
                )],
                adapters: vec![(
                    VALID_CHAIN.into(),
                    vec![VALID_ADAPTER.clone(), FAILED_ADAPTER.clone()],
                )],
                idents: vec![(VALID_CHAIN.into(), Some(VALID_IDENT.clone()))],
                expected: Ok(vec![&VALID_ADAPTER]),
            },
            Case {
                name: "all adapters ok or checkable",
                chain_id: VALID_CHAIN,
                status: vec![(
                    TESTABLE_CHAIN.to_string(),
                    GenesisCheckStatus::TemporaryFailure {
                        checked_at: TESTABLE_ADAPTER.checked_at.unwrap(),
                    },
                )],
                adapters: vec![(
                    VALID_CHAIN.into(),
                    vec![VALID_ADAPTER.clone(), TESTABLE_ADAPTER.clone()],
                )],
                idents: vec![(VALID_CHAIN.into(), Some(VALID_IDENT.clone()))],
                expected: Ok(vec![&VALID_ADAPTER, &TESTABLE_ADAPTER]),
            },
        ];

        for case in cases.into_iter() {
            let Case {
                name,
                chain_id,
                status,
                adapters,
                idents,
                expected,
            } = case;

            let logger = Logger::root(Discard, o!());

            let manager = ProviderManager::new(logger, adapters.into_iter(), idents.into_iter());

            for (provider, status) in status.iter() {
                let slot = manager
                    .inner
                    .status
                    .iter()
                    .find(|(ident, _)| ident.provider.eq(provider))
                    .expect(&format!(
                        "case: {} - there should be a status for provider \"{}\"",
                        name, provider
                    ));
                let mut s = slot.1.write().await;
                *s = status.clone();
            }

            let result = manager.get_all(chain_id.into()).await;
            match (expected, result) {
                (Ok(expected), Ok(result)) => assert_eq!(
                    expected, result,
                    "case {} failed. Result: {:?}",
                    name, result
                ),
                (Err(expected), Err(result)) => assert_eq!(
                    expected.to_string(),
                    result.to_string(),
                    "case {} failed. Result: {:?}",
                    name,
                    result
                ),
                (Ok(expected), Err(result)) => panic!(
                    "case {} failed. Result: {}, Expected: {:?}",
                    name, result, expected
                ),
                (Err(expected), Ok(result)) => panic!(
                    "case {} failed. Result: {:?}, Expected: {}",
                    name, result, expected
                ),
            }
        }
    }
}
