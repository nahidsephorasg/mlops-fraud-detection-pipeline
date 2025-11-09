"""
Feature engineering module for fraud detection API
Applies the same transformations used during model training
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_transaction_cents_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Extract cents from transaction amount"""
    df = df.copy()
    df["TransactionCents"] = (
        df["TransactionAmt"] - df["TransactionAmt"].astype(int)
    ) * 100
    return df


def create_frequency_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Create frequency encoding for categorical columns"""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[f"{col}_freq"] = df[col].map(freq_map)
    return df


def create_aggregation_features(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Create aggregation features (count, nunique)"""
    df = df.copy()

    for col in group_cols:
        if col in df.columns:
            # Count occurrences
            count_map = df[col].value_counts().to_dict()
            df[f"{col}_count"] = df[col].map(count_map)

            # Number of unique values (nunique) - same as count for single column
            df[f"{col}_nunique"] = df[col].map(count_map)

    return df


def create_combination_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create combination features from multiple columns"""
    df = df.copy()

    # card1_addr1 combination
    if "card1" in df.columns and "addr1" in df.columns:
        df["card1_addr1_encoded"] = (
            (df["card1"].astype(str) + "_" + df["addr1"].astype(str))
            .astype("category")
            .cat.codes
        )

        # Frequency of card1_addr1 combination
        combo = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
        freq_map = combo.value_counts(normalize=True).to_dict()
        df["card1_addr1_freq"] = combo.map(freq_map)

    # card1_addr1_P_emaildomain combination
    if (
        "card1" in df.columns
        and "addr1" in df.columns
        and "P_emaildomain" in df.columns
    ):
        df["card1_addr1_P_emaildomain_encoded"] = (
            (
                df["card1"].astype(str)
                + "_"
                + df["addr1"].astype(str)
                + "_"
                + df["P_emaildomain"].astype(str)
            )
            .astype("category")
            .cat.codes
        )

        # Frequency of card1_addr1_P_emaildomain combination
        combo = (
            df["card1"].astype(str)
            + "_"
            + df["addr1"].astype(str)
            + "_"
            + df["P_emaildomain"].astype(str)
        )
        freq_map = combo.value_counts(normalize=True).to_dict()
        df["card1_addr1_P_emaildomain_freq"] = combo.map(freq_map)

    return df


def create_transaction_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features based on transaction amount"""
    df = df.copy()

    if "TransactionAmt" in df.columns:
        # Ratio to mean transaction amount
        mean_amt = df["TransactionAmt"].mean()
        df["TransactionAmt_to_mean"] = (
            df["TransactionAmt"] / mean_amt if mean_amt > 0 else 0
        )

    return df


def add_missing_features(df: pd.DataFrame, required_features: list) -> pd.DataFrame:
    """Add missing features with default values"""
    df = df.copy()

    for feature in required_features:
        if feature not in df.columns:
            # Add missing feature with default value 0
            df[feature] = 0

    return df


def engineer_features(raw_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline to raw transaction data

    Args:
        raw_data: Dictionary containing raw transaction fields

    Returns:
        DataFrame with all engineered features matching training data
    """
    # Convert to DataFrame
    df = pd.DataFrame([raw_data])

    logger.info(f"Starting feature engineering with {len(df.columns)} raw features")

    # 1. Basic feature engineering
    df = create_transaction_cents_feature(df)
    df = create_transaction_amount_features(df)

    # 2. Frequency encoding for categorical features
    categorical_cols = [
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
        "ProductCD",
    ]
    df = create_frequency_encoding(df, categorical_cols)

    # 3. Aggregation features
    group_cols = [
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
    ]
    df = create_aggregation_features(df, group_cols)

    # 4. Combination features
    df = create_combination_features(df)

    # 5. Add all V-columns and C-columns with default values
    # These are the V-columns from training (V95-V137, V279-V321)
    v_columns = [f"V{i}" for i in range(95, 138)] + [  # V95-V137
        f"V{i}"
        for i in [
            279,
            280,
            284,
            285,
            286,
            287,
            290,
            291,
            292,
            293,
            294,
            295,
            297,
            298,
            299,
            302,
            303,
            304,
            305,
            306,
            307,
            308,
            309,
            310,
            311,
            312,
            316,
            317,
            318,
            319,
            320,
            321,
        ]
    ]

    # C-columns (C1-C14)
    c_columns = [f"C{i}" for i in range(1, 15)]

    # Add TransactionID if missing
    if "TransactionID" not in df.columns:
        df["TransactionID"] = 0

    # Add all missing V and C columns
    for col in v_columns + c_columns:
        if col not in df.columns:
            df[col] = 0

    # 6. Ensure all numeric columns
    for col in df.columns:
        if df[col].dtype == "object":
            # Convert categorical to numeric using category codes
            df[col] = df[col].astype("category").cat.codes

    # 7. Define expected feature list (from training)
    expected_features = [
        "TransactionID",
        "TransactionAmt",
        "card1",
        "card3",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "V95",
        "V96",
        "V97",
        "V98",
        "V99",
        "V100",
        "V101",
        "V102",
        "V103",
        "V104",
        "V105",
        "V106",
        "V107",
        "V108",
        "V109",
        "V110",
        "V111",
        "V112",
        "V113",
        "V114",
        "V115",
        "V116",
        "V117",
        "V118",
        "V119",
        "V120",
        "V121",
        "V122",
        "V123",
        "V124",
        "V125",
        "V126",
        "V127",
        "V128",
        "V129",
        "V130",
        "V131",
        "V132",
        "V133",
        "V134",
        "V135",
        "V136",
        "V137",
        "V279",
        "V280",
        "V284",
        "V285",
        "V286",
        "V287",
        "V290",
        "V291",
        "V292",
        "V293",
        "V294",
        "V295",
        "V297",
        "V298",
        "V299",
        "V302",
        "V303",
        "V304",
        "V305",
        "V306",
        "V307",
        "V308",
        "V309",
        "V310",
        "V311",
        "V312",
        "V316",
        "V317",
        "V318",
        "V319",
        "V320",
        "V321",
        "card1_freq",
        "card2_freq",
        "card3_freq",
        "card4_freq",
        "card5_freq",
        "card6_freq",
        "addr1_freq",
        "addr2_freq",
        "card1_addr1_encoded",
        "card1_addr1_P_emaildomain_encoded",
        "card1_count",
        "card2_count",
        "card3_count",
        "card4_count",
        "card5_count",
        "card6_count",
        "addr1_count",
        "addr2_count",
        "card1_nunique",
        "card2_nunique",
        "card3_nunique",
        "card4_nunique",
        "card5_nunique",
        "card6_nunique",
        "addr1_nunique",
        "addr2_nunique",
        "TransactionAmt_to_mean",
        "TransactionCents",
        "card1_addr1_freq",
        "card1_addr1_P_emaildomain_freq",
    ]

    # 8. Add any still-missing features
    df = add_missing_features(df, expected_features)

    # 9. Select only expected features in correct order
    df = df[expected_features]

    # 10. Fill any remaining NaN values
    df = df.fillna(0)

    logger.info(f"Feature engineering complete: {len(df.columns)} features generated")

    return df
