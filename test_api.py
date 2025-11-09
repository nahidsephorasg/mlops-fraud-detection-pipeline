#!/usr/bin/env python3
"""
Test script for Fraud Detection API
"""
import requests
import json
from datetime import datetime

API_URL = "http://localhost:8000"


def print_response(title, response):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    print()


def test_health():
    """Test health check endpoint"""
    response = requests.get(f"{API_URL}/health")
    print_response("Health Check", response)
    return response.status_code == 200


def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{API_URL}/model/info")
    print_response("Model Info", response)
    return response.status_code == 200


def test_single_prediction():
    """Test single transaction prediction"""
    # Sample transaction (normal)
    normal_transaction = {
        "TransactionAmt": 50.00,
        "card1": 13926,
        "card2": 150.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 226.0,
        "card6": "credit",
        "addr1": 315.0,
        "addr2": 87.0,
        "ProductCD": "W",
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
    }

    response = requests.post(f"{API_URL}/predict", json=normal_transaction)
    print_response("Single Prediction (Normal Transaction)", response)

    # Sample suspicious transaction
    suspicious_transaction = {
        "TransactionAmt": 9999.99,
        "card1": 1234,
        "card2": 500.0,
        "card3": 500.0,
        "card4": "visa",
        "card5": 999.0,
        "card6": "credit",
        "addr1": 1.0,
        "addr2": 1.0,
        "ProductCD": "R",
        "P_emaildomain": "tempmail.com",
        "R_emaildomain": "tempmail.com",
    }

    response = requests.post(f"{API_URL}/predict", json=suspicious_transaction)
    print_response("Single Prediction (Suspicious Transaction)", response)

    return response.status_code == 200


def test_batch_prediction():
    """Test batch predictions"""
    batch_request = {
        "transactions": [
            {
                "TransactionAmt": 50.00,
                "card1": 13926,
                "card2": 150.0,
                "card3": 150.0,
                "card4": "visa",
                "card5": 226.0,
                "card6": "credit",
                "addr1": 315.0,
                "addr2": 87.0,
                "ProductCD": "W",
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "gmail.com",
            },
            {
                "TransactionAmt": 5000.00,
                "card1": 5678,
                "card2": 300.0,
                "card3": 300.0,
                "card4": "mastercard",
                "card5": 500.0,
                "card6": "debit",
                "addr1": 100.0,
                "addr2": 50.0,
                "ProductCD": "C",
                "P_emaildomain": "yahoo.com",
                "R_emaildomain": "yahoo.com",
            },
            {
                "TransactionAmt": 9999.99,
                "card1": 1234,
                "card2": 999.0,
                "card3": 999.0,
                "card4": "discover",
                "card5": 999.0,
                "card6": "credit",
                "addr1": 1.0,
                "addr2": 1.0,
                "ProductCD": "R",
                "P_emaildomain": "tempmail.com",
                "R_emaildomain": "tempmail.com",
            },
        ]
    }

    response = requests.post(f"{API_URL}/predict/batch", json=batch_request)
    print_response("Batch Prediction (3 Transactions)", response)

    return response.status_code == 200


def main():
    """Run all tests"""
    print(f"Testing Fraud Detection API at {API_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"{test_name} FAILED: {e}")
            results[test_name] = False

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name:.<40} {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
