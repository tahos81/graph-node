//! Helpers for creating relational queries using diesel. A lot of this code
//! is copied from `diesel_dynamic_schema` and adapted to our data
//! structures, especially the `Table` and `Column` types.

use std::marker::PhantomData;

use diesel::backend::Backend;
use diesel::dsl::sql;
use diesel::expression::{expression_types, is_aggregate, TypedExpressionType, ValidGrouping};
use diesel::pg::Pg;
use diesel::query_builder::{
    AsQuery, AstPass, BoxedSelectStatement, FromClause, Query, QueryFragment, QueryId,
    SelectStatement,
};
use diesel::query_dsl::methods::SelectDsl;
use diesel::query_source::QuerySource;

use diesel::sql_types::{
    Array, BigInt, Binary, Bool, Integer, Nullable, Numeric, SingleValue, Text, Timestamptz,
    Untyped,
};
use diesel::{AppearsOnTable, Expression, QueryDsl, QueryResult, SelectableExpression};
use diesel_dynamic_schema::DynamicSelectClause;
use graph::components::store::{AttributeNames, BlockNumber, StoreError, BLOCK_NUMBER_MAX};
use graph::data::store::{Id, IdType, ID};
use graph::data_source::CausalityRegion;
use graph::prelude::{lazy_static, ENV_VARS};

use crate::relational::ColumnType;
use crate::relational_queries::PARENT_ID;

use super::value::FromOidRow;
use super::Column as RelColumn;
use super::SqlName;
use super::{BLOCK_COLUMN, BLOCK_RANGE_COLUMN};

const TYPENAME: &str = "__typename";

lazy_static! {
    pub static ref TYPENAME_SQL: SqlName = TYPENAME.into();
    pub static ref VID_SQL: SqlName = "vid".into();
    pub static ref PARENT_SQL: SqlName = PARENT_ID.into();
    pub static ref TYPENAME_COL: RelColumn = RelColumn::pseudo_column(TYPENAME, ColumnType::String);
    pub static ref VID_COL: RelColumn = RelColumn::pseudo_column("vid", ColumnType::Int8);
    pub static ref BLOCK_COL: RelColumn = RelColumn::pseudo_column(BLOCK_COLUMN, ColumnType::Int8);
    // The column type is a placeholder, we can't deserialize in4range; but
    // we also never try to use it when we get data from the database
    pub static ref BLOCK_RANGE_COL: RelColumn =
        RelColumn::pseudo_column(BLOCK_RANGE_COLUMN, ColumnType::Bytes);
    pub static ref PARENT_STRING_COL: RelColumn = RelColumn::pseudo_column(PARENT_ID, ColumnType::String);
    pub static ref PARENT_BYTES_COL: RelColumn = RelColumn::pseudo_column(PARENT_ID, ColumnType::Bytes);
    pub static ref PARENT_INT_COL: RelColumn = RelColumn::pseudo_column(PARENT_ID, ColumnType::Int8);

    pub static ref META_COLS: [&'static RelColumn; 2] = [&*TYPENAME_COL, &*VID_COL];
}

#[doc(hidden)]
/// A dummy expression.
pub struct DummyExpression;

impl DummyExpression {
    pub(crate) fn new() -> Self {
        DummyExpression
    }
}

impl<QS> SelectableExpression<QS> for DummyExpression {}

impl<QS> AppearsOnTable<QS> for DummyExpression {}

impl Expression for DummyExpression {
    type SqlType = expression_types::NotSelectable;
}

impl ValidGrouping<()> for DummyExpression {
    type IsAggregate = is_aggregate::No;
}

#[derive(Debug, Clone, Copy)]
/// A wrapper around the `super::Table` struct that provides helper
/// functions for generating SQL queries
pub struct Table<'a>(&'a super::Table);

impl<'a> Table<'a> {
    pub(crate) fn new(table: &'a super::Table) -> Self {
        Self(table)
    }

    /// Reference a column in this table and use the correct SQL type `ST`
    fn column<ST>(&self, name: &str) -> Option<Column<ST>> {
        self.0
            .columns
            .iter()
            .chain(META_COLS.into_iter())
            .find(|c| &c.name == name)
            .map(|c| Column::new(self.clone(), c))
    }

    /// Return a filter expression that generates the SQL for `id = $id`
    pub fn id_eq(&'a self, id: &'a Id) -> IdEq<'a> {
        IdEq::new(*self, id)
    }

    /// Return an expression that generates the SQL for `block_range @>
    /// $block` or `block = $block` depending on whether the table is
    /// mutable or not
    pub fn at_block(&'a self, block: BlockNumber) -> AtBlock<'a> {
        AtBlock::new(*self, block)
    }

    /// Return an expression that generates the SQL for `causality_region =
    /// $cr` if the table uses causality regions
    pub fn belongs_to_causality_region(
        &'a self,
        cr: CausalityRegion,
    ) -> BelongsToCausalityRegion<'a> {
        BelongsToCausalityRegion::new(*self, cr)
    }

    /// Produce a list of the columns that should be selected for a query
    /// based on `column_names`. The result needs to be used both to create
    /// the actual select statement with `Self::select_cols` and to decode
    /// query results with `FromOidRow`.
    pub fn selected_columns<T: FromOidRow>(
        &self,
        column_names: &'a AttributeNames,
        parent_type: Option<IdType>,
    ) -> Result<Vec<&'a super::Column>, StoreError> {
        let mut cols = Vec::new();
        if T::WITH_INTERNAL_KEYS {
            cols.push(&*TYPENAME_COL);
        }

        match column_names {
            AttributeNames::All => cols.extend(self.0.columns.iter()),
            AttributeNames::Select(names) => {
                let pk = self.0.primary_key();
                cols.push(pk);
                let mut names: Vec<_> = names.iter().filter(|name| *name != &*ID).collect();
                names.sort();
                for name in names {
                    let column = self.0.column_for_field(&name)?;
                    cols.push(column);
                }
            }
        };

        if T::WITH_INTERNAL_KEYS {
            match parent_type {
                Some(IdType::String) => cols.push(&*PARENT_STRING_COL),
                Some(IdType::Bytes) => cols.push(&*PARENT_BYTES_COL),
                Some(IdType::Int8) => cols.push(&*PARENT_INT_COL),
                None => (),
            }
        }

        if T::WITH_SYSTEM_COLUMNS {
            cols.push(&*VID_COL);
            if self.0.immutable {
                cols.push(&*BLOCK_COL);
            } else {
                // TODO: We can't deserialize in4range
                cols.push(&*BLOCK_RANGE_COL);
            }
        }
        Ok(cols)
    }

    /// Create a Diesel select statement that selects the columns in
    /// `columns`. Use to generate a query via
    /// `table.select_cols(columns).filter(...)`. For a full example, see
    /// `Layout::find`
    pub fn select_cols(
        &'a self,
        columns: &[&'a RelColumn],
    ) -> BoxedSelectStatement<'a, Untyped, FromClause<Table<'a>>, Pg> {
        type SelectClause<'b> = DynamicSelectClause<'b, Pg, Table<'b>>;

        fn add_field<'b, ST: SingleValue + Send>(
            select: &mut SelectClause<'b>,
            table: &'b Table<'b>,
            column: &'b RelColumn,
        ) {
            let name = &column.name;

            match (column.is_list(), column.is_nullable()) {
                (true, true) => {
                    select.add_field(table.column::<Nullable<Array<ST>>>(name).unwrap())
                }
                (true, false) => select.add_field(table.column::<Array<ST>>(name).unwrap()),
                (false, true) => select.add_field(table.column::<Nullable<ST>>(name).unwrap()),
                (false, false) => select.add_field(table.column::<ST>(name).unwrap()),
            }
        }

        fn add_enum_field<'b>(
            select: &mut SelectClause<'b>,
            table: &'b Table<'b>,
            column: &'b RelColumn,
        ) {
            let name = format!("{}.{}::text", table.0.qualified_name, &column.name);

            match (column.is_list(), column.is_nullable()) {
                (true, true) => select.add_field(sql::<Nullable<Array<Text>>>(&name)),
                (true, false) => select.add_field(sql::<Array<Text>>(&name)),
                (false, true) => select.add_field(sql::<Nullable<Text>>(&name)),
                (false, false) => select.add_field(sql::<Text>(&name)),
            }
        }

        let mut selection = DynamicSelectClause::new();
        for column in columns {
            if column.name == TYPENAME_COL.name {
                selection.add_field(sql::<Text>(&format!(
                    "'{}' as __typename",
                    self.0.object.typename()
                )));
                continue;
            }
            match column.column_type {
                ColumnType::Boolean => add_field::<Bool>(&mut selection, self, column),
                ColumnType::BigDecimal => add_field::<Numeric>(&mut selection, self, column),
                ColumnType::BigInt => add_field::<Numeric>(&mut selection, self, column),
                ColumnType::Bytes => add_field::<Binary>(&mut selection, self, column),
                ColumnType::Int => add_field::<Integer>(&mut selection, self, column),
                ColumnType::Int8 => add_field::<BigInt>(&mut selection, self, column),
                ColumnType::Timestamp => add_field::<Timestamptz>(&mut selection, self, column),
                ColumnType::String => add_field::<Text>(&mut selection, self, column),
                ColumnType::TSVector(_) => add_field::<Text>(&mut selection, self, column),
                ColumnType::Enum(_) => add_enum_field(&mut selection, self, column),
            };
        }
        <Self as SelectDsl<SelectClause<'a>>>::select(*self, selection).into_boxed()
    }
}

impl<'a> QuerySource for Table<'a> {
    type FromClause = Self;
    type DefaultSelection = DummyExpression;

    fn from_clause(&self) -> Self {
        self.clone()
    }

    fn default_selection(&self) -> Self::DefaultSelection {
        DummyExpression::new()
    }
}

impl<'a> AsQuery for Table<'a>
where
    SelectStatement<FromClause<Self>>: Query<SqlType = expression_types::NotSelectable>,
{
    type SqlType = expression_types::NotSelectable;
    type Query = SelectStatement<FromClause<Self>>;

    fn as_query(self) -> Self::Query {
        SelectStatement::simple(self)
    }
}

impl<'a> diesel::Table for Table<'a>
where
    Self: QuerySource + AsQuery,
{
    type PrimaryKey = DummyExpression;
    type AllColumns = DummyExpression;

    fn primary_key(&self) -> Self::PrimaryKey {
        DummyExpression::new()
    }

    fn all_columns() -> Self::AllColumns {
        DummyExpression::new()
    }
}

impl<'a, DB> QueryFragment<DB> for Table<'a>
where
    DB: Backend,
{
    fn walk_ast<'b>(&'b self, mut out: AstPass<'_, 'b, DB>) -> QueryResult<()> {
        out.unsafe_to_cache_prepared();

        out.push_identifier(self.0.nsp.as_str())?;
        out.push_sql(".");
        out.push_identifier(&self.0.name)?;
        Ok(())
    }
}

impl<'a> QueryId for Table<'a> {
    type QueryId = ();
    const HAS_STATIC_QUERY_ID: bool = false;
}

pub struct IdEq<'a> {
    table: Table<'a>,
    id: &'a Id,
}

impl<'a> IdEq<'a> {
    pub fn new(table: Table<'a>, id: &'a Id) -> Self {
        IdEq { table, id }
    }
}

impl Expression for IdEq<'_> {
    type SqlType = Bool;
}

impl<'a> QueryFragment<Pg> for IdEq<'a> {
    fn walk_ast<'b>(&'b self, mut out: AstPass<'_, 'b, Pg>) -> QueryResult<()> {
        out.unsafe_to_cache_prepared();
        self.table.walk_ast(out.reborrow())?;
        out.push_sql(".id = ");
        match self.id {
            Id::String(s) => out.push_bind_param::<Text, _>(s.as_str())?,
            Id::Bytes(b) => out.push_bind_param::<Binary, _>(b)?,
            Id::Int8(i) => out.push_bind_param::<BigInt, _>(i)?,
        }
        Ok(())
    }
}

impl ValidGrouping<()> for IdEq<'_> {
    type IsAggregate = is_aggregate::No;
}

impl<'a> AppearsOnTable<Table<'a>> for IdEq<'a> {}

pub struct AtBlock<'a> {
    table: Table<'a>,
    block: BlockNumber,
    filters_by_id: bool,
}

impl<'a> AtBlock<'a> {
    pub fn new(table: Table<'a>, block: BlockNumber) -> Self {
        AtBlock {
            table,
            block,
            filters_by_id: false,
        }
    }

    pub fn filters_by_id(mut self) -> Self {
        self.filters_by_id = true;
        self
    }
}

impl Expression for AtBlock<'_> {
    type SqlType = Bool;
}

impl<'a> QueryFragment<Pg> for AtBlock<'a> {
    fn walk_ast<'b>(&'b self, mut out: AstPass<'_, 'b, Pg>) -> QueryResult<()> {
        out.unsafe_to_cache_prepared();

        if self.table.0.immutable {
            if self.block == BLOCK_NUMBER_MAX {
                // `self.block <= BLOCK_NUMBER_MAX` is always true
                out.push_sql("true");
            } else {
                self.table.walk_ast(out.reborrow())?;
                out.push_sql(".");
                out.push_identifier(BLOCK_COLUMN)?;
                out.push_sql(" <= ");
                out.push_bind_param::<Integer, _>(&self.block)?;
            }
        } else {
            // Table is mutable and has a block_range column
            self.table.walk_ast(out.reborrow())?;
            out.push_sql(".");
            out.push_identifier(BLOCK_RANGE_COLUMN)?;
            out.push_sql(" @> ");
            out.push_bind_param::<Integer, _>(&self.block)?;

            let should_use_brin =
                !self.filters_by_id || ENV_VARS.store.use_brin_for_all_query_types;
            if self.table.0.is_account_like && self.block < BLOCK_NUMBER_MAX && should_use_brin {
                // When block is BLOCK_NUMBER_MAX, these checks would be wrong; we
                // don't worry about adding the equivalent in that case since
                // we generally only see BLOCK_NUMBER_MAX here for metadata
                // queries where block ranges don't matter anyway.
                //
                // We also don't need to add these if the query already filters by ID,
                // because the ideal index is the GiST index on id and block_range.
                out.push_sql(" and coalesce(upper(");
                out.push_identifier(BLOCK_RANGE_COLUMN)?;
                out.push_sql("), 2147483647) > ");
                out.push_bind_param::<Integer, _>(&self.block)?;
                out.push_sql(" and lower(");
                out.push_identifier(BLOCK_RANGE_COLUMN)?;
                out.push_sql(") <= ");
                out.push_bind_param::<Integer, _>(&self.block)?;
            }
        }

        Ok(())
    }
}

impl ValidGrouping<()> for AtBlock<'_> {
    type IsAggregate = is_aggregate::No;
}

impl<'a> AppearsOnTable<Table<'a>> for AtBlock<'a> {}

pub struct BelongsToCausalityRegion<'a> {
    table: Table<'a>,
    cr: CausalityRegion,
}

impl<'a> BelongsToCausalityRegion<'a> {
    pub fn new(table: Table<'a>, cr: CausalityRegion) -> Self {
        BelongsToCausalityRegion { table, cr }
    }
}

impl Expression for BelongsToCausalityRegion<'_> {
    type SqlType = Bool;
}

impl<'a> QueryFragment<Pg> for BelongsToCausalityRegion<'a> {
    fn walk_ast<'b>(&'b self, mut out: AstPass<'_, 'b, Pg>) -> QueryResult<()> {
        out.unsafe_to_cache_prepared();

        if self.table.0.has_causality_region {
            self.table.walk_ast(out.reborrow())?;
            out.push_sql(".causality_region");
            out.push_sql(" = ");
            out.push_bind_param::<Integer, _>(&self.cr)?;
        } else {
            out.push_sql("true");
        }
        Ok(())
    }
}

impl ValidGrouping<()> for BelongsToCausalityRegion<'_> {
    type IsAggregate = is_aggregate::No;
}

impl<'a> AppearsOnTable<Table<'a>> for BelongsToCausalityRegion<'a> {}

#[derive(Debug, Clone, Copy)]
/// A database table column.
pub struct Column<'a, ST> {
    table: Table<'a>,
    column: &'a super::Column,
    _sql_type: PhantomData<ST>,
}

impl<'a, ST> Column<'a, ST> {
    fn new(table: Table<'a>, column: &'a super::Column) -> Self {
        Self {
            table,
            column,
            _sql_type: PhantomData,
        }
    }
}

impl<'a, ST> QueryId for Column<'a, ST> {
    type QueryId = ();
    const HAS_STATIC_QUERY_ID: bool = false;
}

impl<'a, ST, QS> SelectableExpression<QS> for Column<'a, ST> where Self: Expression {}

impl<'a, ST, QS> AppearsOnTable<QS> for Column<'a, ST> where Self: Expression {}

impl<'a, ST> Expression for Column<'a, ST>
where
    ST: TypedExpressionType,
{
    type SqlType = ST;
}

impl<'a, ST> ValidGrouping<()> for Column<'a, ST> {
    type IsAggregate = is_aggregate::No;
}

impl<'a, ST, DB> QueryFragment<DB> for Column<'a, ST>
where
    DB: Backend,
{
    fn walk_ast<'b>(&'b self, mut out: AstPass<'_, 'b, DB>) -> QueryResult<()> {
        out.unsafe_to_cache_prepared();
        self.table.walk_ast(out.reborrow())?;
        out.push_sql(".");
        out.push_identifier(&self.column.name)?;
        Ok(())
    }
}
