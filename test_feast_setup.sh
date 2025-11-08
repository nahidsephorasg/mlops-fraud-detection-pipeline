#!/bin/bash

set -e

echo "=========================================="
echo "Testing Feast Feature Store Setup"
echo "=========================================="

cd /Users/mmondol/Documents/personal/learning-all/MLOps/e2e-real-time-fraud-detection/feature_store/feature_repo

echo ""
echo "1. Testing Feast CLI commands..."
echo "-----------------------------------"

echo "Listing entities:"
feast entities list

echo ""
echo "Listing feature views:"
feast feature-views list

echo ""
echo "Listing feature services:"
feast feature-services list

echo ""
echo "2. Testing Redis connection..."
echo "-----------------------------------"
if redis-cli ping > /dev/null 2>&1; then
    echo "Redis connection: SUCCESS"
else
    echo "Redis connection: FAILED"
    exit 1
fi

echo ""
echo "3. Testing Python imports..."
echo "-----------------------------------"
python -c "import feast; import redis; print('All imports successful')"

echo ""
echo "4. Testing Feast registry..."
echo "-----------------------------------"
python << EOF
from feast import FeatureStore
store = FeatureStore(repo_path=".")
print(f"Project: {store.project}")
print(f"Registry path: {store.config.registry}")
print(f"Online store: {store.config.online_store}")
EOF

echo ""
echo "5. Testing Redis connectivity from Python..."
echo "-----------------------------------"
python << EOF
import redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
r.set('test_key', 'test_value')
value = r.get('test_key')
print(f"Redis write/read test: {value}")
r.delete('test_key')
print("Redis test: SUCCESS")
EOF

echo ""
echo "=========================================="
echo "All tests passed!"
echo "=========================================="
