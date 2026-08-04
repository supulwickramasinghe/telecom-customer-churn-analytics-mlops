import argparse
import json
import os

import mlflow
import pandas as pd

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Folder containing test_predictions.csv",
    )

    parser.add_argument(
        "--evaluation_output",
        type=str,
        required=True,
        help="Output folder for evaluation artifacts",
    )

    return parser.parse_args()


def find_predictions_file(input_path):
    if os.path.isfile(input_path):
        return input_path

    for root, _, files in os.walk(input_path):
        for file_name in files:
            if file_name == "test_predictions.csv":
                return os.path.join(root, file_name)

    raise FileNotFoundError(
        f"test_predictions.csv was not found inside: {input_path}"
    )


def main():
    args = parse_args()

    predictions_path = find_predictions_file(args.test_data)

    print(f"Reading predictions from: {predictions_path}")

    dataframe = pd.read_csv(predictions_path)

    required_columns = [
        "actual_churn",
        "predicted_churn",
        "churn_probability",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required prediction columns: {missing_columns}"
        )

    y_true = dataframe["actual_churn"].astype(int)
    y_pred = dataframe["predicted_churn"].astype(int)
    y_probability = dataframe["churn_probability"].astype(float)

    metrics = {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "test_recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "test_f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "test_roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
        "test_pr_auc": average_precision_score(
            y_true,
            y_probability,
        ),
    }

    print("\nEvaluation metrics")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
        mlflow.log_metric(
            metric_name,
            float(metric_value),
        )

    os.makedirs(
        args.evaluation_output,
        exist_ok=True,
    )

    metrics_path = os.path.join(
        args.evaluation_output,
        "metrics.json",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    confusion_matrix_values = confusion_matrix(
        y_true,
        y_pred,
    )

    confusion_matrix_data = {
        "true_negative": int(confusion_matrix_values[0][0]),
        "false_positive": int(confusion_matrix_values[0][1]),
        "false_negative": int(confusion_matrix_values[1][0]),
        "true_positive": int(confusion_matrix_values[1][1]),
    }

    confusion_matrix_path = os.path.join(
        args.evaluation_output,
        "confusion_matrix.json",
    )

    with open(
        confusion_matrix_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            confusion_matrix_data,
            file,
            indent=4,
        )

    classification_report_text = classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )

    report_path = os.path.join(
        args.evaluation_output,
        "classification_report.txt",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(classification_report_text)

    mlflow.log_artifact(metrics_path)
    mlflow.log_artifact(confusion_matrix_path)
    mlflow.log_artifact(report_path)

    print("\nClassification report")
    print(classification_report_text)

    print(f"Metrics saved to: {metrics_path}")
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()