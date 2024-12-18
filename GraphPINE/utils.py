import gc
import os
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             fbeta_score, precision_score, recall_score,
                             roc_auc_score, balanced_accuracy_score)


@lru_cache(maxsize=1000)
def fetch_drug_name(nsc_number: str) -> str:
    """
    Fetch the drug name from PubChem using the NSC number, utilizing a cache for performance.

    Args:
        nsc_number (str): The NSC number of the drug.

    Returns:
        str: The drug name if found, otherwise returns "NSC{nsc_number}".
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/NSC{nsc_number}/property/Title/JSON"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "PropertyTable" in data:
            return data["PropertyTable"]["Properties"][0]["Title"]

    return f"NSC{nsc_number}"


def create_nsc_name_mapping(file_path: str) -> Dict[str, str]:
    """
    Create a mapping between NSC numbers and drug names from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing NSC numbers and drug names.

    Returns:
        Dict[str, str]: A dictionary mapping NSC numbers to drug names.
    """
    df = pd.read_csv(file_path, usecols=["NSC", "NAME"])
    return dict(zip(df["NSC"], df["NAME"]))


def free_memory() -> None:
    """
    Free up memory by collecting garbage and clearing CUDA cache if available.

    This function helps manage memory usage, especially useful in machine learning
    pipelines where memory constraints can be an issue.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def calculate_metrics(
    targets,
    predictions,
):
    """
    Calculate various performance metrics using sklearn.

    Args:
        targets (numpy.ndarray): True labels.
        predictions (numpy.ndarray): Predicted probabilities.

    Returns:
        dict: Dictionary containing calculated metrics.
    """

    mask = ~np.isnan(predictions)
    targets = targets[mask]
    predictions = predictions[mask]

    threshold = 0.5
    predictions_binary = (predictions > threshold).astype(int)

    accuracy = accuracy_score(targets, predictions_binary)
    precision = precision_score(targets, predictions_binary)
    recall = recall_score(targets, predictions_binary)
    f1 = f1_score(targets, predictions_binary)
    f2 = fbeta_score(targets, predictions_binary, beta=2)
    specificity = recall_score(targets, predictions_binary, pos_label=0)
    npv = precision_score(targets, predictions_binary, pos_label=0)
    balanced_accuracy = balanced_accuracy_score(targets, predictions_binary)

    # Check if there are more than one class in targets to calculate AUC-ROC and AUC-PR
    if len(np.unique(targets)) > 1:
        auc_roc = roc_auc_score(targets, predictions)
        auc_pr = average_precision_score(targets, predictions)
    else:
        print("Only one class in targets. Cannot calculate AUC-ROC and AUC-PR.")
        auc_roc = None
        auc_pr = None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "AUC-ROC": auc_roc,
        "AUC-PR": auc_pr,
        "Specificity": specificity,
        "NPV": npv,
        "balanced_accuracy": balanced_accuracy,
    }


def predict_by_pretrained_model(config, model_dir="results/model.pt"):
    model = torch.load(model_dir)

    print("Generating predictions and evaluating the model...")
    # Generate predictions and other relevant metrics from the model.
    results, test_preds, y_test, importances = predict(
        model, data_manager, save_results=True
    )

    res = calculate_metrics(y_test, test_preds)

    print("Results:")
    print(f"Accuracy: {res['accuracy']}")
    print(f"Precision: {res['precision']}")
    print(f"Recall: {res['recall']}")
    print(f"F1 Score: {res['f1']}")


@lru_cache(maxsize=1)
def load_genes():
    return pd.read_csv("data/genes.csv").columns.tolist()


def update_importance_csv(pred_importance, batch, data_type):
    """
    Update the CSV file with predicted importance values.
    Creates a new file if it doesn't exist, otherwise appends to the existing file.

    Args:
        pred_importance (torch.Tensor): Predicted importance values.
        batch: The batch of data.
        data_type (str): Type of data being processed ('Train', 'Val', or 'Test').
    """
    file_path = "result/importance.csv.gz"
    new_df = create_importance_dataframe(pred_importance, batch, data_type)

    if not os.path.exists(file_path):
        new_df.to_csv(file_path, index=False, compression="gzip")
    else:
        new_df.to_csv(
            file_path, mode="a", header=False, index=False, compression="gzip"
        )


def create_importance_dataframe(pred_importance, batch, data_type):
    tmp = pred_importance.reshape(batch.num_graphs, -1).numpy()
    nsc, cell_name = batch.nsc, batch.cell_line
    genes = load_genes()

    return pd.DataFrame(
        {
            "nsc": nsc,
            "cell_name": cell_name,
            "type": data_type,
            **{gene: tmp[:, i] for i, gene in enumerate(genes)},
        }
    )


# Optional: Add a function to check and optimize the file if needed
def optimize_importance_csv(file_path, max_size_mb=1000):
    """
    Check the file size and optimize if it exceeds the specified limit.
    This function can be called periodically to manage file size.

    Args:
        file_path (str): Path to the CSV file.
        max_size_mb (int): Maximum allowed file size in MB.
    """
    if (
        os.path.exists(file_path)
        and os.path.getsize(file_path) > max_size_mb * 1024 * 1024
    ):
        df = pd.read_csv(file_path, compression="gzip")
        df = df.drop_duplicates(subset=["nsc", "cell_name", "type"], keep="last")
        df.to_csv(file_path, index=False, compression="gzip")
        print(
            f"File optimized. New size: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB"
        )
