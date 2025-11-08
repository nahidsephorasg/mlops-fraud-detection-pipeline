from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
import os
import requests
import json
import numpy as np

# Configuration
TMP_DIR = '/opt/airflow/fraud_detection_pipeline/tmp'
MONGO_URI = 'mongodb://admin:admin123@fraud-detection-mongodb:27017/'
DB_NAME = 'fraud_detection'
FEAST_SERVER_URL = 'http://fraud-detection-feast-server:6566'  

# Utility Functions
def get_mongo_client():
    """Establish connection to MongoDB"""
    try:
        return MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise

def load_data_from_mongodb(collection_name, limit=1000):
    """Load a limited number of documents from a MongoDB collection"""
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[collection_name]

    cursor = collection.find({}).limit(limit)
    df = pd.DataFrame(list(cursor))
    if '_id' in df.columns:
        df.drop('_id', axis=1, inplace=True)
    return df

def save_data_to_mongodb(data, collection_name):
    """Save data to MongoDB collection"""
    try:
        client = get_mongo_client()
        db = client[DB_NAME]
        collection = db[collection_name]
        collection.drop()
        records = data.to_dict('records')
        collection.insert_many(records, ordered=False)
        print(f"Saved {len(records)} records to MongoDB collection {collection_name}")
    except Exception as e:
        print(f"Error saving data to MongoDB collection {collection_name}: {e}")
        raise

def load_data():
    """Load and merge limited transaction and identity data from MongoDB"""
    print("Loading train and test documents from MongoDB...")
    try:
        # Load limited datasets from the MongoDB instance
        train_transaction = load_data_from_mongodb('train_transaction', limit=1000)
        test_transaction = load_data_from_mongodb('test_transaction', limit=1000)
        train_identity = load_data_from_mongodb('train_identity', limit=1000)
        test_identity = load_data_from_mongodb('test_identity', limit=1000)

        print(f"Loaded train_transaction: {train_transaction.shape}")
        print(f"Loaded test_transaction: {test_transaction.shape}")
        print(f"Loaded train_identity: {train_identity.shape}")
        print(f"Loaded test_identity: {test_identity.shape}")
        
        # Check if data is empty
        if train_transaction.empty or test_transaction.empty:
            raise ValueError("No data found in MongoDB collections. Please import the dataset first using dataset_import.sh")
        
        # Print column names to debug
        print(f"train_transaction columns: {list(train_transaction.columns[:5])}")
        print(f"train_identity columns: {list(train_identity.columns[:5]) if not train_identity.empty else 'Empty'}")
        
        # Check if TransactionID exists
        if 'TransactionID' not in train_transaction.columns:
            raise KeyError(f"TransactionID column not found. Available columns: {list(train_transaction.columns[:10])}")

        # Merge datasets
        X_train = train_transaction.merge(train_identity, on='TransactionID', how='left') if not train_identity.empty else train_transaction
        X_test = test_transaction.merge(test_identity, on='TransactionID', how='left') if not test_identity.empty else test_transaction

        # Display basic info
        print(f"Train data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        if 'isFraud' in X_train.columns:
            print(f"Target distribution: {X_train['isFraud'].value_counts()}")
        else:
            print("Warning: 'isFraud' column not found in training data")

        os.makedirs(TMP_DIR, exist_ok=True)
        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_fe.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_fe.pkl'))
        print("Limited data loading completed successfully")
    except Exception as e:
        print(f"Error in load_data: {e}")
        raise

def normalize_d_columns_task():
    """Normalize D columns (D1-D15)"""
    print("Starting D columns normalization task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_fe.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_fe.pkl'))

        # Get D columns that exist in the dataset
        d_columns = [f'D{i}' for i in range(1, 16) if f'D{i}' in X_train.columns]
        print(f"Found D columns: {d_columns}")
        
        # Check if TransactionDT exists
        if 'TransactionDT' not in X_train.columns:
            print("Warning: TransactionDT column not found. Skipping normalization.")
            return

        # Convert TransactionDT to numeric
        print("Converting TransactionDT to numeric...")
        X_train['TransactionDT'] = pd.to_numeric(X_train['TransactionDT'], errors='coerce')
        X_test['TransactionDT'] = pd.to_numeric(X_test['TransactionDT'], errors='coerce')

        # Process each D column
        for col in d_columns:
            print(f"Processing {col}...")
            
            # Convert D column to numeric
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
            
            # Only subtract where both values are not NaN
            train_mask = X_train[col].notna() & X_train['TransactionDT'].notna()
            test_mask = X_test[col].notna() & X_test['TransactionDT'].notna()
            
            # Apply normalization
            X_train.loc[train_mask, col] = X_train.loc[train_mask, col] - X_train.loc[train_mask, 'TransactionDT']
            X_test.loc[test_mask, col] = X_test.loc[test_mask, col] - X_test.loc[test_mask, 'TransactionDT']

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_fe.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_fe.pkl'))
        print(f"Normalized D columns: {d_columns}")
        
    except Exception as e:
        print(f"Error in normalize_d_columns_task: {e}")
        import traceback
        traceback.print_exc()
        raise

def frequency_encoding_task():
    """Apply frequency encoding to card and address columns"""
    print("Starting frequency encoding task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_fe.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_fe.pkl'))

        card_addr_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2']
        for col in card_addr_cols:
            if col in X_train.columns:
                freq = X_train[col].value_counts().to_dict()
                X_train[f'{col}_freq'] = X_train[col].map(freq)
                X_test[f'{col}_freq'] = X_test[col].map(freq)

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_fe.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_fe.pkl'))
        print("Frequency encoding task completed")
    except Exception as e:
        print(f"Error in frequency_encoding_task: {e}")
        raise

def combine_features_task():
    """Combine features to create new categorical variables"""
    print("Starting combine features task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_fe.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_fe.pkl'))

        for df in [X_train, X_test]:
            if 'card1' in df.columns and 'addr1' in df.columns:
                df['card1_addr1'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
            if 'P_emaildomain' in df.columns:
                df['card1_addr1_P_emaildomain'] = df['card1_addr1'].astype(str) + '_' + df['P_emaildomain'].astype(str)

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_combined.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_combined.pkl'))
        print("Combine features task completed")
    except Exception as e:
        print(f"Error in combine_features_task: {e}")
        raise

def label_encoding_task():
    """Apply label encoding to combined features"""
    print("Starting label encoding task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_combined.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_combined.pkl'))

        cols = ['card1_addr1', 'card1_addr1_P_emaildomain']
        for col in cols:
            if col in X_train.columns:
                le = LabelEncoder()
                train_values = X_train[col].astype(str)
                test_values = X_test[col].astype(str)
                le.fit(pd.concat([train_values, test_values]))
                X_train[f'{col}_encoded'] = le.transform(train_values)
                X_test[f'{col}_encoded'] = le.transform(test_values)
                print(f"Label encoded {col}")

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))
        print("Label encoding task completed")
    except Exception as e:
        print(f"Error in label_encoding_task: {e}")
        raise

def group_aggregation_task():
    """Apply group aggregation features (count)"""
    print("Starting group aggregation task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))

        agg_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2']
        for col in agg_cols:
            if col in X_train.columns:
                agg = X_train.groupby(col)['TransactionID'].count().to_dict()
                X_train[f'{col}_count'] = X_train[col].map(agg)
                X_test[f'{col}_count'] = X_test[col].map(agg)

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))
        print("Group aggregation task completed")
    except Exception as e:
        print(f"Error in group_aggregation_task: {e}")
        raise

def group_aggregation_nunique_task():
    """Apply group aggregation with nunique for TransactionAmt"""
    print("Starting group aggregation nunique task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))

        agg_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2']
        for col in agg_cols:
            if col in X_train.columns and 'TransactionAmt' in X_train.columns:
                agg = X_train.groupby(col)['TransactionAmt'].nunique().to_dict()
                X_train[f'{col}_nunique'] = X_train[col].map(agg)
                X_test[f'{col}_nunique'] = X_test[col].map(agg)

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))
        print("Group aggregation nunique task completed")
    except Exception as e:
        print(f"Error in group_aggregation_nunique_task: {e}")
        raise

def transaction_cents_task():
    """Extract transaction cents and amount features"""
    print("Starting transaction features task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))

        for df in [X_train, X_test]:
            if 'TransactionAmt' in df.columns:
                df['TransactionAmt_to_mean'] = df['TransactionAmt'] / df['TransactionAmt'].mean()
                df['TransactionCents'] = df['TransactionAmt'] - df['TransactionAmt'].astype(int)

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))
        print("Transaction features task completed")
    except Exception as e:
        print(f"Error in transaction_cents_task: {e}")
        raise

def frequency_encoding_combined_task():
    """Apply frequency encoding to combined features"""
    print("Starting frequency encoding combined task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))

        combined_cols = ['card1_addr1', 'card1_addr1_P_emaildomain']
        for col in combined_cols:
            if col in X_train.columns:
                freq = X_train[col].value_counts().to_dict()
                X_train[f'{col}_freq'] = X_train[col].map(freq)
                X_test[f'{col}_freq'] = X_test[col].map(freq)

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))
        print("Frequency encoding combined task completed")
    except Exception as e:
        print(f"Error in frequency_encoding_combined_task: {e}")
        raise

def remove_columns_task():
    """Remove unnecessary columns and finalize preprocessing"""
    print("Starting finalization task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))

        cols_to_drop = [f'D{i}' for i in range(1, 16)] + ['TransactionDT']
        X_train.drop(columns=cols_to_drop, errors='ignore', inplace=True)
        X_test.drop(columns=cols_to_drop, errors='ignore', inplace=True)

        print("FRAUD DETECTION PREPROCESSING PIPELINE COMPLETED")
        print(f"Final training data shape: {X_train.shape}")
        print(f"Final test data shape: {X_test.shape}")
        print(f"Total features: {len(X_train.columns)}")

        train_memory = X_train.memory_usage(deep=True).sum() / 1024**2
        test_memory = X_test.memory_usage(deep=True).sum() / 1024**2
        print(f"Memory usage - Train: {train_memory:.2f} MB")
        print(f"Memory usage - Test: {test_memory:.2f} MB")

        train_missing = X_train.isnull().sum().sum()
        test_missing = X_test.isnull().sum().sum()
        print(f"Missing values - Train: {train_missing}, Test: {test_missing}")

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_final.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_final.pkl'))
        save_data_to_mongodb(X_train, 'X_train_final')
        save_data_to_mongodb(X_test, 'X_test_final')
        print("Final data saved to: X_train_final.pkl, X_test_final.pkl")
    except Exception as e:
        print(f"Error in remove_columns_task: {e}")
        raise

def push_to_feast():
    """Push final processed data to your simple Feast server"""
    print("Starting push to Feast task...")
    try:
        # Load final processed data
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_final.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_final.pkl'))

        print(f"Loaded X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        # Combine train and test for Feast
        data = pd.concat([X_train, X_test], axis=0)
        data = data.reset_index(drop=True)
        
        # Add required columns for your simple server
        if 'TransactionID' not in data.columns:
            data['TransactionID'] = range(1, len(data) + 1)
        
        # Ensure TransactionID is int64
        data['TransactionID'] = data['TransactionID'].astype('int64')
        
        # Add timestamp if not present
        if 'event_timestamp' not in data.columns:
            base_time = datetime.now().replace(microsecond=0)
            data['event_timestamp'] = [base_time + timedelta(seconds=i) for i in range(len(data))]

        print(f"Combined data shape: {data.shape}")
        print(f"Sample data columns: {list(data.columns[:10])}")

        # Call your simple Feast server's store-data endpoint
        success = call_simple_feast_api(data)
        
        if success:
            print("✓ Data pushed to Feast server successfully")
        else:
            print("⚠ Feast push failed, but continuing pipeline...")
        
    except Exception as e:
        print(f"Error in push_to_feast: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing pipeline despite Feast push error...")

def call_simple_feast_api(data):
    """Call your simple Feast server's API endpoints"""
    try:
        # First, check if server is healthy
        health_url = f"{FEAST_SERVER_URL}/health"
        print(f"Checking server health: {health_url}")
        
        health_response = requests.get(health_url, timeout=10)
        if health_response.status_code == 200:
            print("✓ Feast server is healthy")
            print(f"Health response: {health_response.json()}")
        else:
            print(f"⚠ Server health check returned: {health_response.status_code}")
            return False
        
        # Check server status
        status_url = f"{FEAST_SERVER_URL}/status"
        print(f"Checking server status: {status_url}")
        
        status_response = requests.get(status_url, timeout=10)
        if status_response.status_code == 200:
            print("✓ Server status OK")
            print(f"Status response: {status_response.json()}")
        
        # Prepare data for storage (sample first 100 records to avoid timeout)
        sample_data = data.head(100).copy()
        
        # Convert data for JSON serialization
        records = prepare_data_for_api(sample_data)
        
        # Call store-data endpoint
        store_url = f"{FEAST_SERVER_URL}/store-data"
        payload = {"data": records}
        
        print(f"Storing {len(records)} records to: {store_url}")
        
        store_response = requests.post(
            store_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        print(f"Store response status: {store_response.status_code}")
        print(f"Store response: {store_response.text}")
        
        if store_response.status_code == 200:
            response_data = store_response.json()
            print(f"✓ Successfully stored {response_data.get('records_count', 'unknown')} records")
            return True
        else:
            print(f"✗ Store request failed with status: {store_response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to Feast server at {FEAST_SERVER_URL}")
        return False
    except requests.exceptions.Timeout:
        print("✗ Request to Feast server timed out")
        return False
    except Exception as e:
        print(f"✗ Error calling Feast API: {e}")
        return False

def prepare_data_for_api(data):
    """Prepare data for API call - handle NaN values and data types"""
    try:
        data_clean = data.copy()
        
        # Replace NaN with None for JSON compatibility
        data_clean = data_clean.where(pd.notnull(data_clean), None)
        
        # Convert datetime columns to string
        for col in data_clean.columns:
            if pd.api.types.is_datetime64_any_dtype(data_clean[col]):
                data_clean[col] = data_clean[col].astype(str)
        
        # Convert to list of dictionaries
        records = data_clean.to_dict('records')
        
        print(f"Prepared {len(records)} records for API")
        if records:
            print(f"Sample record keys: {list(records[0].keys())[:10]}")
        
        return records
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        return []

def test_feast_connection():
    """Test connection to your simple Feast server"""
    try:
        health_url = f"{FEAST_SERVER_URL}/health"
        print(f"Testing connection to: {health_url}")
        
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            print(f"✓ Successfully connected to Feast server")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"⚠ Server responded with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Failed to connect to Feast server: {e}")
        return False

# Define DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 28),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'preprocessing_pipeline_simple_feast',
    default_args=default_args,
    description='Fraud detection preprocessing pipeline with simple Feast server',
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=['fraud_detection', 'preprocessing', 'ml', 'feast']
) as dag:
    
    # Test Feast connection first
    test_feast_task = PythonOperator(
        task_id='test_feast_connection',
        python_callable=test_feast_connection
    )
    
    # Data loading task
    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )
    
    # Feature engineering tasks
    normalize_d_columns = PythonOperator(
        task_id='normalize_d_columns',
        python_callable=normalize_d_columns_task
    )
    
    frequency_encoding = PythonOperator(
        task_id='frequency_encoding',
        python_callable=frequency_encoding_task
    )
    
    combine_features = PythonOperator(
        task_id='combine_features',
        python_callable=combine_features_task
    )
    
    label_encoding = PythonOperator(
        task_id='label_encoding',
        python_callable=label_encoding_task
    )
    
    group_aggregation = PythonOperator(
        task_id='group_aggregation',
        python_callable=group_aggregation_task
    )
    
    group_aggregation_nunique = PythonOperator(
        task_id='group_aggregation_nunique',
        python_callable=group_aggregation_nunique_task
    )
    
    transaction_cents = PythonOperator(
        task_id='transaction_cents',
        python_callable=transaction_cents_task
    )
    
    frequency_encoding_combined = PythonOperator(
        task_id='frequency_encoding_combined',
        python_callable=frequency_encoding_combined_task
    )
    
    remove_columns = PythonOperator(
        task_id='remove_columns',
        python_callable=remove_columns_task
    )
    
    # Push to simple Feast server
    push_to_feast_task = PythonOperator(
        task_id='push_to_feast',
        python_callable=push_to_feast
    )

    # Define task dependencies
    test_feast_task >> load_data_task >> normalize_d_columns >> frequency_encoding >> combine_features >> label_encoding >> group_aggregation >> group_aggregation_nunique >> transaction_cents >> frequency_encoding_combined >> remove_columns >> push_to_feast_task
