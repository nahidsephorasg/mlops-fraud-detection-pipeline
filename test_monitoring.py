import requests
import random
import time

API_URL = "http://localhost:8000"

for i in range(20):
    # Make prediction
    transaction = {
        "TransactionAmt": random.uniform(50, 500),
        "card1": random.randint(1000, 20000),
        "card2": random.uniform(50, 500),
        "card3": random.uniform(50, 500),
        "card4": random.choice(["visa", "mastercard", "discover"]),
        "card5": random.uniform(100, 300),
        "card6": random.choice(["credit", "debit"]),
        "addr1": random.uniform(100, 500),
        "addr2": random.uniform(50, 200),
        "ProductCD": random.choice(["W", "C", "R", "H"]),
        "P_emaildomain": random.choice(["gmail.com", "yahoo.com"]),
        "R_emaildomain": random.choice(["gmail.com", "yahoo.com"]),
    }

    response = requests.post(f"{API_URL}/predict", json=transaction)
    result = response.json()
    transaction_id = result["transaction_id"]

    print(f"✓ Prediction {i+1}: {transaction_id} - Fraud: {result['is_fraud']}")

    # Submit ground truth (random for testing)
    ground_truth = {
        "transaction_id": transaction_id,
        "is_fraud": random.choice([True, False]),
    }
    requests.post(f"{API_URL}/monitoring/ground-truth", json=ground_truth)

    time.sleep(0.5)

# Force metric calculation
requests.post(f"{API_URL}/monitoring/performance/calculate")

# Check metrics
metrics = requests.get(f"{API_URL}/monitoring/performance").json()
print(
    f"\n📊 Metrics: Precision={metrics.get('precision', 0):.3f}, Recall={metrics.get('recall', 0):.3f}"
)
