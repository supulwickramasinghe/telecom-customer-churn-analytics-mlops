import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    return parser.parse_args()


def load_parquet_folder(input_path):
    folder = Path(input_path)
    parquet_files = list(folder.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found under {folder}"
        )

    print(f"Parquet files found: {len(parquet_files)}")

    return pd.concat(
        [pd.read_parquet(file) for file in parquet_files],
        ignore_index=True
    )


def main():
    args = parse_args()
    df = load_parquet_folder(args.input_data)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("Columns:", df.columns.tolist())

    required_columns = {"customer_id", "churn_value"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    duplicate_customers = df["customer_id"].duplicated().sum()
    null_targets = df["churn_value"].isna().sum()
    invalid_targets = (~df["churn_value"].isin([0, 1])).sum()

    print("\nValidation summary:")
    print(f"Duplicate customer IDs: {duplicate_customers}")
    print(f"Null target values: {null_targets}")
    print(f"Invalid target values: {invalid_targets}")

    print("\nTarget distribution:")
    print(
        df["churn_value"]
        .value_counts(dropna=False)
        .sort_index()
    )

    if duplicate_customers > 0:
        raise ValueError("Duplicate customer IDs were found.")

    if null_targets > 0 or invalid_targets > 0:
        raise ValueError(
            "The target contains null or invalid values."
        )

    print("\nAzure ML data validation completed successfully.")


if __name__ == "__main__":
    main()
