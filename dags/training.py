from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from pymongo import MongoClient
import os
import requests
import json
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import IsolationForest
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available, will skip LightGBM training")
from sklearn.preprocessing import LabelEncoder

# Configuration
TMP_DIR = '/opt/airflow/fraud_detection_pipeline/tmp'
MODEL_DIR = '/opt/airflow/fraud_detection_pipeline/models'
MONGO_URI = 'mongodb://admin:admin123@fraud-detection-mongodb:27017/'
DB_NAME = 'fraud_detection'
FEAST_SERVER_URL = 'http://fraud-detection-feast-server:6566'
MLFLOW_TRACKING_URI = 'http://mlflow-server:5101'

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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

def test_feast_connection():
    """Test connection to Feast server"""
    try:
        health_url = f"{FEAST_SERVER_URL}/health"
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            print(f"✓ Successfully connected to Feast server at {FEAST_SERVER_URL}")
            
            # Also test status endpoint
            status_url = f"{FEAST_SERVER_URL}/status"
            status_response = requests.get(status_url, timeout=10)
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Feast status: {status_data}")
            
            return True
        else:
            print(f"Feast server responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"Failed to connect to Feast server: {e}")
        return False

def test_mlflow_connection():
    """Test connection to MLflow server"""
    try:
        # Test MLflow connection
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        
        # Try to list experiments
        experiments = client.list_experiments()
        print(f"✓ Successfully connected to MLflow server at {MLFLOW_TRACKING_URI}")
        print(f"Found {len(experiments)} experiments")
        
        return True
    except Exception as e:
        print(f"Failed to connect to MLflow server: {e}")
        return False

def load_features_from_feast():
    """Load features from Feast feature store"""
    print("Loading features from Feast...")
    try:
        # Check if Feast server has data
        status_url = f"{FEAST_SERVER_URL}/status"
        status_response = requests.get(status_url, timeout=30)
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            if not status_data.get('data_source_exists', False):
                print("No data in Feast feature store, loading from MongoDB instead...")
                return load_features_from_mongodb()
        
        # If Feast has data, try to load it directly from the parquet file
        feast_data_path = "/home/ubuntu/fraud_detection_pipeline/fraud_detection_features/data/data_source.parquet"
        
        # We'll need to access this via API call or load from MongoDB
        print("Loading processed features from MongoDB (X_train_final and X_test_final)...")
        return load_features_from_mongodb()
        
    except Exception as e:
        print(f"Error loading from Feast: {e}")
        print("Falling back to MongoDB...")
        return load_features_from_mongodb()

def load_features_from_mongodb():
    """Load final processed features from MongoDB"""
    try:
        print("Loading final processed features from MongoDB...")
        
        # Load the final processed data that was saved by preprocessing pipeline
        X_train = load_data_from_mongodb('X_train_final', limit=1000)
        X_test = load_data_from_mongodb('X_test_final', limit=1000)
        
        print(f"Loaded training data shape: {X_train.shape}")
        print(f"Loaded test data shape: {X_test.shape}")
        print(f"Training data columns: {list(X_train.columns)}")
        
        # Ensure we have the target variable
        if 'isFraud' not in X_train.columns:
            print("Warning: isFraud column not found in training data")
            # Create dummy target for testing
            X_train['isFraud'] = np.random.choice([0, 1], size=len(X_train), p=[0.9, 0.1])
        
        # Save to temp directory
        os.makedirs(TMP_DIR, exist_ok=True)
        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_features.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_features.pkl'))
        
        print("✓ Features loaded and saved to temp directory")
        return True
        
    except Exception as e:
        print(f"Error loading features from MongoDB: {e}")
        raise

def prepare_training_data():
    """Prepare data for training"""
    print("Preparing training data...")
    try:
        # Load features
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_features.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_features.pkl'))
        
        print(f"Original training data shape: {X_train.shape}")
        print(f"Original test data shape: {X_test.shape}")
        print(f"Training columns: {list(X_train.columns)[:10]}...")  # Show first 10 columns
        print(f"Test columns: {list(X_test.columns)[:10]}...")  # Show first 10 columns
        
        # Separate features and target
        if 'isFraud' in X_train.columns:
            y_train = X_train['isFraud']
            X_train_features = X_train.drop('isFraud', axis=1)
        else:
            print("Warning: No target variable found, creating dummy target")
            y_train = np.random.choice([0, 1], size=len(X_train), p=[0.9, 0.1])
            X_train_features = X_train.copy()
        
        # Get numeric columns from training data
        train_numeric_columns = X_train_features.select_dtypes(include=[np.number]).columns
        print(f"Training numeric columns: {len(train_numeric_columns)}")
        
        # Get numeric columns from test data
        if len(X_test) > 0:
            test_numeric_columns = X_test.select_dtypes(include=[np.number]).columns
            print(f"Test numeric columns: {len(test_numeric_columns)}")
            
            # Find intersection of columns (only columns that exist in both)
            common_columns = train_numeric_columns.intersection(test_numeric_columns)
            print(f"Common numeric columns: {len(common_columns)}")
            
            # Use only common columns
            X_train_features = X_train_features[common_columns]
            X_test_features = X_test[common_columns]
        else:
            # If no test data, use all training numeric columns
            X_train_features = X_train_features[train_numeric_columns]
            X_test_features = X_train_features.iloc[:0].copy()
        
        print(f"Final training feature matrix shape: {X_train_features.shape}")
        print(f"Final test feature matrix shape: {X_test_features.shape}")
        print(f"Target distribution: {pd.Series(y_train).value_counts()}")
        
        # Handle missing values
        X_train_features = X_train_features.fillna(0)
        X_test_features = X_test_features.fillna(0)
        
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_features, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Training split shape: {X_train_split.shape}")
        print(f"Validation split shape: {X_val_split.shape}")
        
        # Save prepared data
        os.makedirs(TMP_DIR, exist_ok=True)
        X_train_split.to_pickle(os.path.join(TMP_DIR, 'X_train_prepared.pkl'))
        X_val_split.to_pickle(os.path.join(TMP_DIR, 'X_val_prepared.pkl'))
        X_test_features.to_pickle(os.path.join(TMP_DIR, 'X_test_prepared.pkl'))
        
        with open(os.path.join(TMP_DIR, 'y_train_prepared.pkl'), 'wb') as f:
            pickle.dump(y_train_split, f)
        with open(os.path.join(TMP_DIR, 'y_val_prepared.pkl'), 'wb') as f:
            pickle.dump(y_val_split, f)
        
        print("✓ Training data prepared and saved")
        return True
        
    except Exception as e:
        print(f"Error preparing training data: {e}")
        import traceback
        traceback.print_exc()
        raise

def train_xgboost_model():
    """Train XGBoost model"""
    print("Training XGBoost model...")
    try:
        # Load prepared data
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_prepared.pkl'))
        X_val = pd.read_pickle(os.path.join(TMP_DIR, 'X_val_prepared.pkl'))
        
        with open(os.path.join(TMP_DIR, 'y_train_prepared.pkl'), 'rb') as f:
            y_train = pickle.load(f)
        with open(os.path.join(TMP_DIR, 'y_val_prepared.pkl'), 'rb') as f:
            y_val = pickle.load(f)
        
        # Start MLflow experiment
        mlflow.set_experiment("fraud_detection_xgboost")
        
        with mlflow.start_run(run_name="xgboost_fraud_detection"):
            # Model parameters
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            auc_score = roc_auc_score(y_val, y_pred_proba)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            
            # Log metrics
            mlflow.log_metrics({
                'auc': auc_score,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            # Log and register model
            model_info = mlflow.xgboost.log_model(
                model, 
                "model",
                registered_model_name="fraud_detection_xgboost"
            )
            
            print(f"Model registered: {model_info.model_uri}")
            
            # Save model locally
            os.makedirs(MODEL_DIR, exist_ok=True)
            model.save_model(os.path.join(MODEL_DIR, 'xgboost_fraud_model.json'))
            
            print(f"XGBoost Model Performance:")
            print(f"AUC: {auc_score:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            # Save metrics to file
            with open(os.path.join(TMP_DIR, 'xgboost_metrics.json'), 'w') as f:
                json.dump({
                    'model': 'xgboost',
                    'auc': auc_score,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }, f)
        
        print("✓ XGBoost model training completed")
        return True
        
    except Exception as e:
        print(f"Error training XGBoost model: {e}")
        raise

def train_lightgbm_model():
    """Train LightGBM model"""
    if not LIGHTGBM_AVAILABLE:
        print("LightGBM not available, skipping...")
        return True
    
    print("Training LightGBM model...")
    try:
        # Load prepared data
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_prepared.pkl'))
        X_val = pd.read_pickle(os.path.join(TMP_DIR, 'X_val_prepared.pkl'))
        
        with open(os.path.join(TMP_DIR, 'y_train_prepared.pkl'), 'rb') as f:
            y_train = pickle.load(f)
        with open(os.path.join(TMP_DIR, 'y_val_prepared.pkl'), 'rb') as f:
            y_val = pickle.load(f)
        
        # Start MLflow experiment
        mlflow.set_experiment("fraud_detection_lightgbm")
        
        with mlflow.start_run(run_name="lightgbm_fraud_detection"):
            # Model parameters
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': -1
            }
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            auc_score = roc_auc_score(y_val, y_pred_proba)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            
            # Log metrics
            mlflow.log_metrics({
                'auc': auc_score,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            # Log model
            mlflow.lightgbm.log_model(model, "model")
            
            # Save model locally
            os.makedirs(MODEL_DIR, exist_ok=True)
            model.booster_.save_model(os.path.join(MODEL_DIR, 'lightgbm_fraud_model.txt'))
            
            print(f"LightGBM Model Performance:")
            print(f"AUC: {auc_score:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            # Save metrics to file
            with open(os.path.join(TMP_DIR, 'lightgbm_metrics.json'), 'w') as f:
                json.dump({
                    'model': 'lightgbm',
                    'auc': auc_score,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }, f)
        
        print("✓ LightGBM model training completed")
        return True
        
    except Exception as e:
        print(f"Error training LightGBM model: {e}")
        raise

def train_isolation_forest():
    """Train Isolation Forest for anomaly detection"""
    print("Training Isolation Forest model...")
    try:
        # Load prepared data
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_prepared.pkl'))
        X_val = pd.read_pickle(os.path.join(TMP_DIR, 'X_val_prepared.pkl'))
        
        with open(os.path.join(TMP_DIR, 'y_val_prepared.pkl'), 'rb') as f:
            y_val = pickle.load(f)
        
        # Start MLflow experiment
        mlflow.set_experiment("fraud_detection_isolation_forest")
        
        with mlflow.start_run(run_name="isolation_forest_fraud_detection"):
            # Model parameters
            params = {
                'contamination': 0.1,  # Expected proportion of outliers
                'random_state': 42,
                'n_estimators': 100
            }
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model (unsupervised)
            model = IsolationForest(**params)
            model.fit(X_train)
            
            # Make predictions (-1 for outliers, 1 for inliers)
            y_pred_raw = model.predict(X_val)
            # Convert to binary (1 for fraud, 0 for normal)
            y_pred = np.where(y_pred_raw == -1, 1, 0)
            
            # Calculate metrics
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            
            # For AUC, use decision function
            y_scores = model.decision_function(X_val)
            auc_score = roc_auc_score(y_val, -y_scores)  # Negative because lower scores indicate outliers
            
            # Log metrics
            mlflow.log_metrics({
                'auc': auc_score,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            # Log and register model
            model_info = mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="fraud_detection_isolation_forest"
            )
            
            print(f"Model registered: {model_info.model_uri}")
            
            # Save model locally
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(os.path.join(MODEL_DIR, 'isolation_forest_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
            
            print(f"Isolation Forest Model Performance:")
            print(f"AUC: {auc_score:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            # Save metrics to file
            with open(os.path.join(TMP_DIR, 'isolation_forest_metrics.json'), 'w') as f:
                json.dump({
                    'model': 'isolation_forest',
                    'auc': auc_score,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }, f)
        
        print("✓ Isolation Forest model training completed")
        return True
        
    except Exception as e:
        print(f"Error training Isolation Forest model: {e}")
        raise

def select_best_model():
    """Select the best performing model"""
    print("Selecting best model...")
    try:
        models_performance = []
        
        # Load metrics for each model
        for model_name in ['xgboost', 'lightgbm', 'isolation_forest']:
            metrics_file = os.path.join(TMP_DIR, f'{model_name}_metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    models_performance.append(metrics)
        
        if not models_performance:
            print("No model metrics found!")
            return False
        
        # Select best model based on F1-score
        best_model = max(models_performance, key=lambda x: x['f1_score'])
        
        print("Model Performance Comparison:")
        for model in models_performance:
            print(f"{model['model']}: AUC={model['auc']:.4f}, F1={model['f1_score']:.4f}")
        
        print(f"\n✓ Best model: {best_model['model']} (F1-Score: {best_model['f1_score']:.4f})")
        
        # Save best model info
        with open(os.path.join(TMP_DIR, 'best_model.json'), 'w') as f:
            json.dump(best_model, f)
        
        # Log to MLflow
        mlflow.set_experiment("fraud_detection_model_selection")
        with mlflow.start_run(run_name="best_model_selection"):
            mlflow.log_params({"selection_criteria": "f1_score"})
            mlflow.log_metrics({
                "best_f1_score": best_model['f1_score'],
                "best_auc": best_model['auc']
            })
            mlflow.log_artifact(os.path.join(TMP_DIR, 'best_model.json'))
        
        print("✓ Best model selection completed")
        return True
        
    except Exception as e:
        print(f"Error selecting best model: {e}")
        raise

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
    'training_pipeline',
    default_args=default_args,
    description='Fraud detection model training pipeline',
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=['fraud_detection', 'training', 'ml']
) as dag:
    
    # Test connections
    test_feast_task = PythonOperator(
        task_id='test_feast_connection',
        python_callable=test_feast_connection
    )
    
    test_mlflow_task = PythonOperator(
        task_id='test_mlflow_connection',
        python_callable=test_mlflow_connection
    )
    
    # Load features
    load_features_task = PythonOperator(
        task_id='load_features',
        python_callable=load_features_from_feast
    )
    
    # Prepare training data
    prepare_data_task = PythonOperator(
        task_id='prepare_training_data',
        python_callable=prepare_training_data
    )
    
    # Train models
    train_xgboost_task = PythonOperator(
        task_id='train_xgboost',
        python_callable=train_xgboost_model
    )
    
    train_lightgbm_task = PythonOperator(
        task_id='train_lightgbm',
        python_callable=train_lightgbm_model
    )
    
    train_isolation_forest_task = PythonOperator(
        task_id='train_isolation_forest',
        python_callable=train_isolation_forest
    )
    
    # Select best model
    select_best_model_task = PythonOperator(
        task_id='select_best_model',
        python_callable=select_best_model
    )

    # Define task dependencies
    [test_feast_task, test_mlflow_task] >> load_features_task >> prepare_data_task
    prepare_data_task >> [train_xgboost_task, train_lightgbm_task, train_isolation_forest_task]
    [train_xgboost_task, train_lightgbm_task, train_isolation_forest_task] >> select_best_model_task
