
import pandas as pd
import os
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_SOURCE_PATH = SCRIPT_DIR / "feature_store" / "feature_repo" / "data" / "data_source.parquet"

def verify_feature_data():
    """Verify that feature data exists and display information"""
    
    print("=" * 60)
    print("Verifying Feature Data")
    print("=" * 60)
    
    # Check if file exists
    print(f"\nChecking file: {DATA_SOURCE_PATH}")
    
    if not DATA_SOURCE_PATH.exists():
        print("\nFile not found!")
        print("You need to store data first.")
        print("\nOptions:")
        print("1. Use the REST API:")
        print("   python start_feast.py")
        print("   curl -X POST http://localhost:8000/store-data -H 'Content-Type: application/json' -d '{...}'")
        print("\n2. Run the end-to-end test:")
        print("   python test_feast_end_to_end.py")
        return False
    
    # Read the parquet file
    print("File found! Reading data...")
    try:
        df = pd.read_parquet(DATA_SOURCE_PATH)
        
        print("\n" + "=" * 60)
        print("Feature Data Summary")
        print("=" * 60)
        
        # Display shape
        print(f"\nShape: {df.shape}")
        print(f"  Rows: {df.shape[0]:,}")
        print(f"  Columns: {df.shape[1]}")
        
        # Display columns
        print(f"\nColumns: {df.columns.tolist()}")
        
        # Display data types
        print("\nData Types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Display basic statistics
        print("\nBasic Statistics:")
        print(df.describe())
        
        # Check for missing values
        print("\nMissing Values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("  No missing values")
        else:
            print(missing[missing > 0])
        
        # Check timestamp column
        if 'timestamp' in df.columns:
            print("\nTimestamp Range:")
            print(f"  Earliest: {df['timestamp'].min()}")
            print(f"  Latest: {df['timestamp'].max()}")
        
        print("\n" + "=" * 60)
        print("Feature data verification complete!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nError reading file: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_feature_data()
