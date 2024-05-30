"""
This files contain a simple function to see if we could run a "!molpal run" command when we clone and install molapl. Saddly it doesn't work, even with subprocess library.
"""
import subprocess
import pandas as pd

def Molpal_run(model, confid, metrics, init, batches, max_iter, dataset, top_x_perc, k, BETA):
    """
    Runs the molpal tool with specified parameters, processes the results, and evaluates the fraction of top X% molecules found.

    Parameters:
    - model (str): The model to be used by molpal.
    - confid (str): The confidence method to be used.
    - metrics (str): The metric for evaluation.
    - init (int): Initial size of the dataset.
    - batches (int): Batch size for iterations.
    - max_iter (int): Maximum number of iterations.
    - dataset (int): Dataset identifier used to specify the dataset file.
    - top_x_perc (float): The top X percentage of molecules to consider.
    - k (int): Number of top samples to select.
    - BETA (float): Beta parameter for model configuration.

    Functionality:
    1. Runs the molpal tool with the specified parameters, generating intermediate and final output files.
    2. Moves and renames the final output file.
    3. Reads the dataset file and selects the top X% of molecules based on their scores.
    4. Saves the selected top X% molecules to a new file.
    5. Reads the final output file from molpal and finds common molecules with the top X% list.
    6. Prints the fraction of top X% molecules found in the final output.
    7. Processes and merges CSV files.
    8. Performs selection using the specified function.

    Outputs:
    - Prints the fraction of top X% molecules found.
    - Moves and renames output files.
    - Calls additional functions to process and select data.
    """

    command = [
        "molpal", "run",
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

    output_filename_csv = f"/content/molpal/folder_output/run_output/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}.csv"
    # Rename the final output file
    subprocess.run(["mv", "/content/molpal/folder_output/run_output/all_explored_final.csv", output_filename_csv])

    output_folder_name = f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}"
    # Move the output directory
    subprocess.run(["mv", "/content/molpal/folder_output/run_output", output_folder_name])

    file_path = f'/content/molpal/data/Enamine{dataset}k_scores.csv.gz'
    df = pd.read_csv(file_path)

    df_sorted = df.sort_values(by='score', ascending=True)

    percentile_index = int(len(df_sorted) * top_x_perc)

    top_x_smiles = df_sorted[['smiles', 'score']].iloc[:percentile_index]

    output_file_path = f'/content/molpal/data/Top_{top_x_perc}_Enamine10k_scores.csv'
    top_x_smiles.to_csv(output_file_path, index=False)

    df_found = pd.read_csv(f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}.csv")
    df_top_x = pd.read_csv(output_file_path)

    smiles_found = set(df_found['smiles'])
    smiles_top_x = set(df_top_x['smiles'])

    common_smiles = smiles_found.intersection(smiles_top_x)

    num_common_smiles = len(common_smiles)
    print("Fraction of top", top_x_perc * 100, " % of smiles found:", num_common_smiles * 100 / len(smiles_top_x), "%")

    process_and_merge_csvs(
        f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}.csv",
        f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_ucb.csv",
        f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_std.csv",
        f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_data"
    )

    selection_fct(
        f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_data",
        f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}.csv",
        f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_selected_data"
    )

