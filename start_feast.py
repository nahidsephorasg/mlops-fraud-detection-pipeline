#!/usr/bin/env python3
"""
Fixed Feast Feature Server with better error handling
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
FEAST_REPO_PATH = SCRIPT_DIR / "feature_store" / "feature_repo"
DATA_SOURCE_PATH = FEAST_REPO_PATH / "data" / "data_source.parquet"

def check_environment():
    """Check if we're in the correct environment"""
    logger.info("Checking environment...")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")

    # Check if we're in conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    logger.info(f"Conda environment: {conda_env}")

def test_imports():
    """Test required imports"""
    logger.info("Testing imports...")

    try:
        import feast
        logger.info(f"✓ Feast version: {feast.__version__}")
    except ImportError as e:
        logger.error(f"✗ Cannot import feast: {e}")
        return False

    try:
        import pandas as pd
        logger.info(f"✓ Pandas version: {pd.__version__}")
    except ImportError as e:
        logger.error(f"✗ Cannot import pandas: {e}")
        return False

    try:
        from flask import Flask
        logger.info("✓ Flask import successful")
    except ImportError as e:
        logger.error(f"✗ Cannot import Flask: {e}")
        logger.info("Install with: pip install flask")
        return False

    return True

def test_feast_store():
    """Test Feast store initialization"""
    logger.info("Testing Feast store initialization...")

    try:
        from feast import FeatureStore
        store = FeatureStore(repo_path=str(FEAST_REPO_PATH))
        logger.info("✓ FeatureStore initialized successfully")

        # List entities and feature views
        entities = store.list_entities()
        feature_views = store.list_feature_views()

        logger.info(f"Found {len(entities)} entities and {len(feature_views)} feature views")

        return store

    except Exception as e:
        logger.error(f"✗ FeatureStore initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def start_simple_server():
    """Start a simple Flask server without complex Feast operations"""
    try:
        from flask import Flask, request, jsonify

        app = Flask(__name__)

        @app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "healthy", "message": "Simple Feast server is running"}), 200

        @app.route('/status', methods=['GET'])
        def get_status():
            try:
                # Basic status without complex operations
                return jsonify({
                    "status": "success",
                    "message": "Simple Feast server",
                    "repo_path": str(FEAST_REPO_PATH),
                    "data_source_path": str(DATA_SOURCE_PATH),
                    "data_source_exists": os.path.exists(DATA_SOURCE_PATH)
                }), 200
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500

        @app.route('/store-data', methods=['POST'])
        def store_data():
            try:
                data = request.get_json()
                feature_data = data.get('data', [])

                if not feature_data:
                    return jsonify({"status": "error", "message": "No data provided"}), 400

                # Convert to DataFrame and save
                df = pd.DataFrame(feature_data)
                os.makedirs(os.path.dirname(DATA_SOURCE_PATH), exist_ok=True)
                df.to_parquet(DATA_SOURCE_PATH, index=False)

                logger.info(f"Stored {len(df)} records to {DATA_SOURCE_PATH}")

                return jsonify({
                    "status": "success",
                    "message": f"Stored {len(df)} records",
                    "records_count": len(df)
                }), 200

            except Exception as e:
                logger.error(f"Error storing data: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        # Use different port to avoid conflict with Docker feast-server
        port = int(os.environ.get('PORT', 6567))
        logger.info(f"Starting simple Flask server on port {port}...")
        app.run(host='0.0.0.0', port=port, debug=False)

    except Exception as e:
        logger.error(f"Error starting simple server: {e}")
        return False

def main():
    """Main function"""
    logger.info("Starting Feast Feature Server (Fixed Version)...")

    # Check environment
    check_environment()

    # Test imports
    if not test_imports():
        logger.error("Import tests failed. Please check your environment.")
        sys.exit(1)

    # Check if feast repo directory exists
    if not FEAST_REPO_PATH.exists():
        logger.error(f"Feast repository not found: {FEAST_REPO_PATH}")
        logger.info("Please initialize feast first with: cd feature_store/feature_repo && feast apply")
        sys.exit(1)
    else:
        logger.info(f"Feast repository found: {FEAST_REPO_PATH}")

    # Test feature store
    store = test_feast_store()

    if store:
        logger.info("Full Feast functionality available")
        # Here you could start the full server with all Feast features
        # For now, we'll start the simple server
        start_simple_server()
    else:
        logger.warning("Feast store initialization failed, starting simple server...")
        start_simple_server()

if __name__ == '__main__':
    main()
