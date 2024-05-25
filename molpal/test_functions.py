import subprocess
import os
import pandas as pd

def run_molpal(model, confid, metrics, init, batches, max_iter, k, BETA):
    """
    Runs molpal with specified parameters.

    Parameters:
    - model (str): The model to be used by molpal.
    - confid (str): The confidence method to be used.
    - metrics (str): The metric for evaluation.
    - init (int): Initial size of the dataset.
    - batches (int): Batch size for iterations.
    - max_iter (int): Maximum number of iterations.
    - k (int): Number of top samples to select.
    - BETA (float): Beta parameter for model configuration.
    """

    command = [
        "\!molpal", "run",
        "--write-intermediate", "--write-final", "--retrain-from-scratch",
        "--library", "/content/molpal/data/Enamine10k_scores.csv.gz",
        "-o", "lookup",
        "--objective-config", "/content/molpal/examples/objective/Enamine10k_lookup.ini",
        "--model", model,
        "--conf-method", confid,
        "--metric", metrics,
        "--init-size", str(init),
        "--batch-size", str(batches),
        "--max-iters", str(max_iter),
        "--fps", "/content/molpal/folder_output/fps_file.h5",
        "--output-dir", "run_output",
        "-k", str(k),
        "--beta", str(BETA)
    ]

    subprocess.run(command)

# Example usage:
# run_molpal("model_value", "confid_value", "metrics_value", 10, 20, 100, 5, 0.1)
