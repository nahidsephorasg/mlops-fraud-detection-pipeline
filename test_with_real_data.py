#!/usr/bin/env python3
"""
Test fraud detection API with real transaction data from the test dataset.
This will show both fraud and non-fraud predictions.
"""

import pandas as pd
import requests
import json
import time
from typing import Dict, Any

# Configuration
API_URL = "http://localhost:8000/predict"  # Use regular prediction endpoint with preprocessing
TEST_TRANSACTION_PATH = "data/test_transaction.csv"
TEST_IDENTITY_PATH = "data/test_identity.csv"


def load_test_data(n_samples: int = 20):
    """Load test transactions"""
    print(f"📁 Loading test data from {TEST_TRANSACTION_PATH}...")

    # Load transaction data
    df_trans = pd.read_csv(TEST_TRANSACTION_PATH, nrows=n_samples)

    # Load identity data if available
    try:
        df_identity = pd.read_csv(TEST_IDENTITY_PATH, nrows=n_samples)
        # Merge on TransactionID
        df = pd.merge(df_trans, df_identity, on="TransactionID", how="left")
        print(f"✅ Loaded {len(df)} transactions with identity data")
    except:
        df = df_trans
        print(f"✅ Loaded {len(df)} transactions (without identity data)")

    return df


def prepare_transaction(row: pd.Series) -> Dict[str, Any]:
    """Prepare a transaction for API request - send only the 12 basic fields"""
    transaction = {
        "TransactionAmt": float(row.get("TransactionAmt", 0)),
        "card1": float(row.get("card1", 0)) if pd.notna(row.get("card1")) else None,
        "card2": float(row.get("card2", 0)) if pd.notna(row.get("card2")) else None,
        "card3": float(row.get("card3", 0)) if pd.notna(row.get("card3")) else None,
        "card4": str(row.get("card4", "")) if pd.notna(row.get("card4")) else None,
        "card5": float(row.get("card5", 0)) if pd.notna(row.get("card5")) else None,
        "card6": str(row.get("card6", "")) if pd.notna(row.get("card6")) else None,
        "addr1": float(row.get("addr1", 0)) if pd.notna(row.get("addr1")) else None,
        "addr2": float(row.get("addr2", 0)) if pd.notna(row.get("addr2")) else None,
        "ProductCD": (
            str(row.get("ProductCD", "")) if pd.notna(row.get("ProductCD")) else None
        ),
        "P_emaildomain": (
            str(row.get("P_emaildomain", ""))
            if pd.notna(row.get("P_emaildomain"))
            else None
        ),
        "R_emaildomain": (
            str(row.get("R_emaildomain", ""))
            if pd.notna(row.get("R_emaildomain"))
            else None
        ),
    }

    return transaction


def make_prediction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction via API"""
    try:
        response = requests.post(API_URL, json=transaction, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 70)
    print("   Testing Fraud Detection API with Real Test Data")
    print("=" * 70)

    # Load test data
    df = load_test_data(n_samples=50)

    print(f"\n📊 Sample transaction data:")
    print(df[["TransactionAmt", "card1", "card4", "card6", "ProductCD"]].head())

    # Make predictions
    print(f"\n🔮 Making predictions...")
    print("-" * 70)

    fraud_count = 0
    legitimate_count = 0
    error_count = 0

    results = []

    for idx, row in df.iterrows():
        transaction = prepare_transaction(row)
        result = make_prediction(transaction)

        if "error" in result:
            error_count += 1
            status = "❌ ERROR"
            fraud_status = result["error"][:50]
        else:
            is_fraud = result.get("is_fraud", False)
            prob = result.get("fraud_probability", 0)
            risk = result.get("risk_level", "UNKNOWN")

            if is_fraud:
                fraud_count += 1
                status = "🚨 FRAUD"
            else:
                legitimate_count += 1
                status = "✅ LEGIT"

            fraud_status = f"{status} | Prob: {prob:.2%} | Risk: {risk}"
            results.append(
                {
                    "is_fraud": is_fraud,
                    "probability": prob,
                    "risk_level": risk,
                    "amount": transaction["TransactionAmt"],
                }
            )

        print(f"{idx+1:2d}. ${transaction['TransactionAmt']:8.2f} | {fraud_status}")

        # Small delay to avoid overwhelming the API
        time.sleep(0.1)

    # Summary
    print("\n" + "=" * 70)
    print("   SUMMARY")
    print("=" * 70)
    print(f"\n✅ Legitimate: {legitimate_count}")
    print(f"🚨 Fraud:      {fraud_count}")
    print(f"❌ Errors:     {error_count}")
    print(f"📊 Total:      {len(df)}")

    if legitimate_count > 0:
        print(f"\n🎉 Success! Found {legitimate_count} legitimate transactions")

        # Show some legitimate examples
        legit_results = [r for r in results if not r["is_fraud"]]
        if legit_results:
            print("\n💰 Example Legitimate Transactions:")
            for i, r in enumerate(legit_results[:5], 1):
                print(
                    f"   {i}. ${r['amount']:.2f} - Prob: {r['probability']:.2%} - Risk: {r['risk_level']}"
                )

    if fraud_count > 0:
        print(f"\n🚨 Found {fraud_count} fraudulent transactions")

        # Show some fraud examples
        fraud_results = [r for r in results if r["is_fraud"]]
        if fraud_results:
            print("\n⚠️  Example Fraudulent Transactions:")
            for i, r in enumerate(fraud_results[:5], 1):
                print(
                    f"   {i}. ${r['amount']:.2f} - Prob: {r['probability']:.2%} - Risk: {r['risk_level']}"
                )

    print("\n📊 Check your dashboards:")
    print("   • Grafana: http://localhost:3000")
    print("   • Prometheus: http://localhost:9090")
    print("\nYou should now see a mix of fraud and legitimate predictions!")


if __name__ == "__main__":
    main()
