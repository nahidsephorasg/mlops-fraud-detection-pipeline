from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64, String
from datetime import timedelta

transaction_entity = Entity(
    name="transaction_id",
    join_keys=["TransactionID"],
    value_type=ValueType.INT64,
    description="Transaction identifier"
)


transaction_source = FileSource(
    path="data/test_transactions.parquet",
    timestamp_field="timestamp"
)


transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="TransactionAmt", dtype=Float64),
        Field(name="card1", dtype=Int64),
        Field(name="card2", dtype=Float64),
        Field(name="card3", dtype=Float64),
    ],
    source=transaction_source,
    online=True
)
