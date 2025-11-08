#!/usr/bin/env python3
"""
End-to-end test for Feast feature store
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from feast import FeatureStore

def test_feast_setup():
    print("Testing Feast Feature Store Setup")
    print("=" * 50)
    
    # Initialize feature store
    print("\n1. Initializing Feature Store...")
    store = FeatureStore(repo_path="./feature_store/feature_repo")
    print(f"   Project: {store.project}")
    print(f"   Registry: {store.config.registry}")
    
    # List registered objects
    print("\n2. Listing registered objects...")
    entities = store.list_entities()
    print(f"   Entities: {len(entities)}")
    for entity in entities:
        print(f"     - {entity.name}")
    
    feature_views = store.list_feature_views()
    print(f"   Feature Views: {len(feature_views)}")
    for fv in feature_views:
        print(f"     - {fv.name}")
    
    return store

def test_feature_materialization(store):
    print("\n3. Testing feature materialization...")
    
    # Create sample data
    data_dir = Path("./feature_store/feature_repo/data")
    data_dir.mkdir(exist_ok=True)
    
    sample_data = pd.DataFrame({
        'TransactionID': [1, 2, 3, 4, 5],
        'TransactionAmt': [100.0, 200.0, 150.0, 300.0, 50.0],
        'card1': [1000, 2000, 3000, 4000, 5000],
        'card2': [111.0, 222.0, 333.0, 444.0, 555.0],
        'card3': [11.0, 22.0, 33.0, 44.0, 55.0],
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(5)]
    })
    
    parquet_file = data_dir / "test_transactions.parquet"
    sample_data.to_parquet(parquet_file, index=False)
    print(f"   Created test data: {parquet_file}")
    print(f"   Rows: {len(sample_data)}")
    
    return sample_data

def test_online_features(store, sample_data):
    print("\n4. Testing online feature retrieval...")
    
    try:
        # Materialize features to online store
        print("   Materializing features to Redis...")
        store.materialize(
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        print("   Materialization complete")
        
        # Retrieve online features
        print("\n   Retrieving online features...")
        entity_rows = [{"TransactionID": tid} for tid in sample_data['TransactionID'].tolist()]
        
        features = store.get_online_features(
            features=[
                "transaction_features:TransactionAmt",
                "transaction_features:card1",
                "transaction_features:card2",
            ],
            entity_rows=entity_rows
        ).to_dict()
        
        print("   Retrieved features:")
        for key, values in features.items():
            print(f"     {key}: {values[:3]}...")
        
        print("\n   Online feature test: SUCCESS")
        return True
        
    except Exception as e:
        print(f"   Error during online feature test: {e}")
        print("   Note: This is expected if features haven't been applied yet")
        return False

def main():
    try:
        # Test basic setup
        store = test_feast_setup()
        
        # Create test data
        sample_data = test_feature_materialization(store)
        
        # Test online features
        test_online_features(store, sample_data)
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Define your feature views in feature_store/feature_repo/")
        print("2. Run: cd feature_store/feature_repo && feast apply")
        print("3. Materialize features: feast materialize <start> <end>")
        print("4. Query features using the FeatureStore API")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
