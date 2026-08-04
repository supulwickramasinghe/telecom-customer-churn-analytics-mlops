import argparse
import os
import json
import joblib

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to the validated training CSV file",
    )

    parser.add_argument(
        "--model_output",
        type=str,
        required=True,
        help="Directory where the trained model will be saved",
    )

    parser.add_argument(
        "--target_column",
        type=str,
        default="churn_value",
        help="Name of the binary churn target column",
    )

    return parser.parse_args()


def load_data(data_path):
    print(f"Loading training data from: {data_path}")

    if os.path.isdir(data_path):
        csv_files = [
            os.path.join(data_path, file_name)
            for file_name in os.listdir(data_path)
            if file_name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV file was found inside directory: {data_path}"
            )

        data_path = csv_files[0]

    dataframe = pd.read_csv(data_path)

    print(f"Dataset shape: {dataframe.shape}")
    print(f"Columns: {dataframe.columns.tolist()}")

    return dataframe


def normalize_target(target):
    if pd.api.types.is_numeric_dtype(target):
        normalized_target = pd.to_numeric(target, errors="coerce")
    else:
        mapping = {
            "yes": 1,
            "churned": 1,
            "churn": 1,
            "true": 1,
            "1": 1,
            "no": 0,
            "stayed": 0,
            "stay": 0,
            "false": 0,
            "0": 0,
        }

        normalized_target = (
            target.astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )

    if normalized_target.isna().any():
        invalid_values = target[normalized_target.isna()].dropna().unique()

        raise ValueError(
            "The target column contains unsupported values: "
            f"{invalid_values.tolist()}"
        )

    return normalized_target.astype(int)


def main():
    args = parse_args()

    dataframe = load_data(args.train_data)

    if args.target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{args.target_column}' was not found. "
            f"Available columns: {dataframe.columns.tolist()}"
        )

    dataframe = dataframe.dropna(subset=[args.target_column]).copy()

    y_train = normalize_target(dataframe[args.target_column])
    X_train = dataframe.drop(columns=[args.target_column])

    identifier_columns = [
        column
        for column in X_train.columns
        if column.lower() in {
            "customer_id",
            "customerid",
            "customer_key",
        }
    ]

    if identifier_columns:
        print(f"Dropping identifier columns: {identifier_columns}")
        X_train = X_train.drop(columns=identifier_columns)

    numeric_columns = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    print(f"Numeric columns: {numeric_columns}")
    print(f"Categorical columns: {categorical_columns}")
    print(f"Target distribution:\n{y_train.value_counts()}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iterations", 2000)
        mlflow.log_param("target_column", args.target_column)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("feature_count", X_train.shape[1])

        print("Training Logistic Regression model...")
        model_pipeline.fit(X_train, y_train)

        predictions = model_pipeline.predict(X_train)
        probabilities = model_pipeline.predict_proba(X_train)[:, 1]

        metrics = {
            "training_accuracy": accuracy_score(
                y_train,
                predictions,
            ),
            "training_precision": precision_score(
                y_train,
                predictions,
                zero_division=0,
            ),
            "training_recall": recall_score(
                y_train,
                predictions,
                zero_division=0,
            ),
            "training_f1_score": f1_score(
                y_train,
                predictions,
                zero_division=0,
            ),
            "training_roc_auc": roc_auc_score(
                y_train,
                probabilities,
            ),
            "training_pr_auc": average_precision_score(
                y_train,
                probabilities,
            ),
        }

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, float(metric_value))
            print(f"{metric_name}: {metric_value:.4f}")

        os.makedirs(args.model_output, exist_ok=True)

        model_file = os.path.join(
            args.model_output,
            "churn_model.joblib",
        )

        joblib.dump(model_pipeline, model_file)

        metadata = {
            "target_column": args.target_column,
            "feature_columns": X_train.columns.tolist(),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "model_type": "LogisticRegression",
        }

        metadata_file = os.path.join(
            args.model_output,
            "model_metadata.json",
        )

        with open(metadata_file, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        report = classification_report(
            y_train,
            predictions,
            zero_division=0,
        )

        report_file = os.path.join(
            args.model_output,
            "classification_report.txt",
        )

        with open(report_file, "w", encoding="utf-8") as file:
            file.write(report)

        confusion_matrix_values = confusion_matrix(
            y_train,
            predictions,
        ).tolist()

        confusion_matrix_file = os.path.join(
            args.model_output,
            "confusion_matrix.json",
        )

        with open(
            confusion_matrix_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                confusion_matrix_values,
                file,
                indent=4,
            )

        mlflow.log_artifact(report_file)
        mlflow.log_artifact(confusion_matrix_file)
        mlflow.log_artifact(metadata_file)

        signature = mlflow.models.infer_signature(
            X_train,
            predictions,
        )

        input_example = X_train.head(5)

        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        print(f"Model saved to: {model_file}")
        print("MLflow model logged successfully.")
        print("Training completed successfully.")


if __name__ == "__main__":
    main()