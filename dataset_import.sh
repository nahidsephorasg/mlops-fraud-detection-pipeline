#!/bin/bash


set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

MONGO_CONTAINER="fraud-detection-mongodb"
MONGO_USER="admin"
MONGO_PASS="admin123"
MONGO_DB="fraud_detection"
MONGO_AUTH_DB="admin"

echo "Checking MongoDB connection..."
if ! docker ps | grep -q "$MONGO_CONTAINER"; then
    echo "MongoDB container is not running!"
    echo "Please start it with: docker-compose up -d mongodb"
    exit 1
fi

# Test MongoDB connection using docker exec
echo "Testing MongoDB connection..."
if ! docker exec "$MONGO_CONTAINER" mongosh -u "$MONGO_USER" -p "$MONGO_PASS" --authenticationDatabase "$MONGO_AUTH_DB" --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
    echo "Cannot connect to MongoDB"
    echo "Waiting for MongoDB to be ready..."
    sleep 5
    if ! docker exec "$MONGO_CONTAINER" mongosh -u "$MONGO_USER" -p "$MONGO_PASS" --authenticationDatabase "$MONGO_AUTH_DB" --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
        echo "Failed to connect to MongoDB. Please check docker-compose logs."
        exit 1
    fi
fi

echo "MongoDB is running and accessible"


mkdir -p "$DATA_DIR"


REQUIRED_FILES=("train_transaction.csv" "test_transaction.csv" "train_identity.csv" "test_identity.csv")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "Missing CSV files:"
    printf '%s\n' "${MISSING_FILES[@]}"
    echo "Please copy all required CSV files to: $DATA_DIR/"
    exit 1
fi

echo "Found all required CSV files"

# Prepare database collections
echo ""
echo "Preparing database collections..."
docker exec "$MONGO_CONTAINER" mongosh -u "$MONGO_USER" -p "$MONGO_PASS" --authenticationDatabase "$MONGO_AUTH_DB" --eval "
use('${MONGO_DB}');
db.train_transaction.drop();
db.test_transaction.drop();
db.train_identity.drop();
db.test_identity.drop();
print('Collections dropped/prepared');
"

# Copy CSV files to container
echo ""
echo "Copying CSV files to MongoDB container..."
docker exec "$MONGO_CONTAINER" mkdir -p /tmp/data
docker cp "$DATA_DIR/train_transaction.csv" "$MONGO_CONTAINER:/tmp/data/"
docker cp "$DATA_DIR/test_transaction.csv" "$MONGO_CONTAINER:/tmp/data/"
docker cp "$DATA_DIR/train_identity.csv" "$MONGO_CONTAINER:/tmp/data/"
docker cp "$DATA_DIR/test_identity.csv" "$MONGO_CONTAINER:/tmp/data/"

# Import CSV files using mongoimport inside container
echo ""
echo "Importing train_transaction.csv..."
docker exec "$MONGO_CONTAINER" mongoimport \
    -u "$MONGO_USER" \
    -p "$MONGO_PASS" \
    --authenticationDatabase "$MONGO_AUTH_DB" \
    --db "$MONGO_DB" \
    --collection train_transaction \
    --type csv \
    --headerline \
    --file /tmp/data/train_transaction.csv

echo "Importing test_transaction.csv..."
docker exec "$MONGO_CONTAINER" mongoimport \
    -u "$MONGO_USER" \
    -p "$MONGO_PASS" \
    --authenticationDatabase "$MONGO_AUTH_DB" \
    --db "$MONGO_DB" \
    --collection test_transaction \
    --type csv \
    --headerline \
    --file /tmp/data/test_transaction.csv

echo "Importing train_identity.csv..."
docker exec "$MONGO_CONTAINER" mongoimport \
    -u "$MONGO_USER" \
    -p "$MONGO_PASS" \
    --authenticationDatabase "$MONGO_AUTH_DB" \
    --db "$MONGO_DB" \
    --collection train_identity \
    --type csv \
    --headerline \
    --file /tmp/data/train_identity.csv

echo "Importing test_identity.csv..."
docker exec "$MONGO_CONTAINER" mongoimport \
    -u "$MONGO_USER" \
    -p "$MONGO_PASS" \
    --authenticationDatabase "$MONGO_AUTH_DB" \
    --db "$MONGO_DB" \
    --collection test_identity \
    --type csv \
    --headerline \
    --file /tmp/data/test_identity.csv

# Clean up temporary files in container
echo ""
echo "🧹 Cleaning up temporary files..."
docker exec "$MONGO_CONTAINER" rm -rf /tmp/data

# Create indexes for performance
echo ""
echo "🔧 Creating indexes..."
docker exec "$MONGO_CONTAINER" mongosh -u "$MONGO_USER" -p "$MONGO_PASS" --authenticationDatabase "$MONGO_AUTH_DB" --eval "
use('${MONGO_DB}');
db.train_transaction.createIndex({'TransactionID': 1});
db.test_transaction.createIndex({'TransactionID': 1});
db.train_identity.createIndex({'TransactionID': 1});
db.test_identity.createIndex({'TransactionID': 1});
print('Indexes created');
" --quiet

# Create collections for processed data
echo "Creating collections for processed data..."
docker exec "$MONGO_CONTAINER" mongosh -u "$MONGO_USER" -p "$MONGO_PASS" --authenticationDatabase "$MONGO_AUTH_DB" --eval "
use('${MONGO_DB}');
db.createCollection('X_train_final');
db.createCollection('X_test_final');
print('Collections created');
" --quiet

# Verify import
echo ""
echo "Verifying import..."
docker exec "$MONGO_CONTAINER" mongosh -u "$MONGO_USER" -p "$MONGO_PASS" --authenticationDatabase "$MONGO_AUTH_DB" --eval "
use('${MONGO_DB}');
print('\\n=== Collection Counts ===');
print('train_transaction: ' + db.train_transaction.countDocuments({}));
print('test_transaction: ' + db.test_transaction.countDocuments({}));
print('train_identity: ' + db.train_identity.countDocuments({}));
print('test_identity: ' + db.test_identity.countDocuments({}));

var stats = db.stats();
print('\\n=== Database Statistics ===');
print('Database size: ' + Math.round(stats.dataSize / (1024*1024)) + ' MB');
print('Total collections: ' + stats.collections);
"

echo ""
echo "============================================"
echo "Data import completed successfully!"
echo "============================================"
echo "Access Mongo Express at: http://localhost:8081"
echo "   Username: admin"
echo "   Password: admin123"
echo ""
echo "MongoDB Connection String:"
echo "   mongodb://admin:admin123@localhost:27017/fraud_detection?authSource=admin"
echo "============================================"
