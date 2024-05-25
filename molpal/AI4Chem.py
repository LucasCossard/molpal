import pandas as pd
import os
import subprocess


import subprocess

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
        "!molpal", "run",
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

"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""

def process_and_merge_csvs(csv_score, csv_ucb, csv_std, output_filename):
    df1 = pd.read_csv(csv_score)

    df2 = pd.read_csv(csv_ucb, header=None)
    df2_transposed = df2.T
    df2_transposed.reset_index(drop=True, inplace=True)
    ucb_values = df2_transposed.iloc[:, -1]

    df3 = pd.read_csv(csv_std, header=None)
    df3_transposed = df3.T
    df3_transposed.reset_index(drop=True, inplace=True)
    std_values = df3_transposed.iloc[:, -1]

    df_score_UCB = pd.merge(df1, ucb_values, left_index=True, right_index=True)
    df_score_UCB_dev = pd.merge(df_score_UCB, std_values, left_index=True, right_index=True)

    new_column_names = list(df_score_UCB_dev.columns)
    new_column_names[-2] = 'UCB'
    new_column_names[-1] = 'Std dev'
    df_score_UCB_dev.columns = new_column_names

    df_score_UCB_dev.to_csv(output_filename, index=False)
    print(f"Output saved to {output_filename}")

def selection_fct(csv_all, csv_allselected, output_csv):
    df1 = pd.read_csv(csv_all)
    df2 = pd.read_csv(csv_allselected)

    matches = df2[['smiles']].merge(df1.reset_index(), on='smiles', how='inner')
    matched_rows = df1.loc[matches['index']]
    new_df = matched_rows.copy()
    new_df1 = new_df.sort_values(by='score', ascending=False)

    new_df1.to_csv(output_csv, index=False)
    print(f"Selection output saved to {output_csv}")

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
    os.system(f"molpal run --write-intermediate --write-final --retrain-from-scratch --library /content/molpal/data/Enamine10k_scores.csv.gz -o lookup --objective-config /content/molpal/examples/objective/Enamine10k_lookup.ini \
        --model {model} --conf-method {confid} --metric {metrics} --init-size {init} \
        --batch-size {batches} --max-iters {max_iter} --fps /content/molpal/folder_output/fps_file.h5 \
        --output-dir run_output -k {k} --beta {BETA}")

    output_filename_csv = f"/content/molpal/folder_output/run_output/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}.csv"
    os.system(f"mv /content/molpal/folder_output/run_output/all_explored_final.csv {output_filename_csv}")

    output_folder_name = f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}"
    os.system(f"mv /content/molpal/folder_output/run_output {output_folder_name}")

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

    process_and_merge_csvs(f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}.csv",
                           f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_ucb.csv",
                           f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_std.csv",
                           f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_data.csv"
                           )

    selection_fct(f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_data.csv",
                  f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}.csv",
                  f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_selected_data.csv"
    )

#=============================================================================================================================================================================================================


"""
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

def molpal_run2(model, confid, metrics, init, batches, max_iter, k, BETA):
    """
    Runs the molpal tool with specified parameters and moves the generated output files to appropriate locations.

    Parameters: all listed above molpal_run2

    The function executes molpal with given parameters, renames the final output file, and moves the output directory.
    """

 command = ["!molpal", "run","--write-intermediate", "--write-final", "--retrain-from-scratch","--library", "/content/molpal/data/Enamine10k_scores.csv.gz","-o", "lookup","--objective-config", "/content/molpal/examples/objective/Enamine10k_lookup.ini","--model", model,"--conf-method", confid,"--metric", metrics,"--init-size", str(init),"--batch-size", str(batches),"--max-iters", str(max_iter),"--fps", "/content/molpal/folder_output/fps_file.h5","--output-dir", "run_output","-k", str(k),"--beta", str(BETA)]

    subprocess.run(command, shell=True)

    output_filename_csv = f"/content/molpal/folder_output/run_output/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}.csv"
    os.rename("/content/molpal/folder_output/run_output/all_explored_final.csv", output_filename_csv)

    output_folder_name = f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}"
    os.rename("/content/molpal/folder_output/run_output", output_folder_name)



def molpal_run_random2(model, confid, init, batches, max_iter, k, BETA):
    """
    Runs the molpal tool with the metric set to random and specified parameters, then moves the generated output files to appropriate locations.

    Parameters: all listed above molpal_run2

    The function executes molpal with given parameters and the metric set to random, renames the final output file, and moves the output directory.
    """

    subprocess.run([
        "!molpal", "run",
        "--write-intermediate", "--write-final", "--retrain-from-scratch",
        "--library", "/content/molpal/data/Enamine10k_scores.csv.gz",
        "-o", "lookup",
        "--objective-config", "/content/molpal/examples/objective/Enamine10k_lookup.ini",
        "--model", model,
        "--conf-method", confid,
        "--metric", "random",
        "--init-size", str(init),
        "--batch-size", str(batches),
        "--max-iters", str(max_iter),
        "--fps", "/content/molpal/folder_output/fps_file.h5",
        "--output-dir", "run_output",
        "-k", str(k),
        "--beta", str(BETA)
    ])

    output_filename_csv = f"/content/molpal/folder_output/run_output/all_explored_final_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}.csv"
    os.rename("/content/molpal/folder_output/run_output/all_explored_final.csv", output_filename_csv)

    output_folder_name = f"/content/molpal/folder_output/run_output_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}"
    os.rename("/content/molpal/folder_output/run_output", output_folder_name)


def frac_top_x(top_x_perc, dataset, csv_selectedsmiles):
    """
    Calculates the fraction of molecules from the selected smiles that are in the top X% of the dataset based on scores.

    Parameters:
    - top_x_perc (float): The top X percent of the dataset to consider (e.g., 0.1 for top 10%).
    - dataset (int): Dataset identifier used in the file name.
    - csv_selectedsmiles (str): Path to the CSV file containing the selected smiles.

    Returns:
    - float: The fraction of top X% smiles found in the selected smiles.
    """

    file_path = f'/content/molpal/data/Enamine{dataset}k_scores.csv.gz'
    df = pd.read_csv(file_path)

    df_sorted = df.sort_values(by='score', ascending=False)

    percentile_index = int(len(df_sorted) * top_x_perc)

    top_x_smiles = df_sorted[['smiles', 'score']].iloc[:percentile_index]

    output_file_path = f'/content/molpal/data/Top_{top_x_perc}_Enamine10k_scores.csv'
    top_x_smiles.to_csv(output_file_path, index=False)

    df_found = pd.read_csv(csv_selectedsmiles)
    df_top_x = pd.read_csv(f'/content/molpal/data/Top_{top_x_perc}_Enamine10k_scores.csv')

    smiles_found = set(df_found['smiles'])
    smiles_top_x = set(df_top_x['smiles'])

    common_smiles = smiles_found.intersection(smiles_top_x)

    num_common_smiles = len(common_smiles)
    print("Fraction of top", top_x_perc * 100, " % of smiles found:", num_common_smiles * 100 / len(smiles_top_x), "%")

    return num_common_smiles * 100 / len(smiles_top_x)


def experiment_top_ef(model, confid, metrics, init, batches, max_iter, dataset, top_x_perc, k, BETA):
    """
    Runs the molpal tool with specified and random metrics, evaluates the fraction of top X% molecules found, and calculates the Enrichment Factor (EF).

    Parameters: all listed above molpal_run2

    Functionality:
    1. Runs the molpal tool with specified parameters using the provided metric.
    2. Runs the molpal tool with specified parameters using a random metric.
    3. Reads the output files from both runs.
    4. Calculates the fraction of top X% molecules found in each run.
    5. Prints and returns the fraction for both runs and the Enrichment Factor (EF).

    Outputs:
    - Prints the fraction of top X% molecules found for both specified and random metrics.
    - Prints the calculated Enrichment Factor (EF).
    - Returns the fraction for both runs and the EF.

    """

    molpal_run2(model, confid, metrics, init, batches, max_iter, k, BETA)

    molpal_run_random2(model, confid, init, batches, max_iter, k, BETA)

    csv_file_random = f'/content/molpal/folder_output/run_output_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}/all_explored_final_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}.csv'
    csv_file = f'/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}.csv'

    frac = frac_top_x(top_x_perc, dataset, csv_file)

    frac_random = frac_top_x(top_x_perc, dataset, csv_file_random)

    print("The percentage of molecules recovered from the top", top_x_perc, "% is", frac, "% for", metrics, "and", frac_random, "% for random")
    EF = frac / frac_random

    print("The Enrichment Factor (EF) was calculated to be", EF)

    return frac, EF


import subprocess
import os

def molpal_run3(model, confid, metrics, init, batches, max_iter, k, BETA, n):
    """
    Runs the molpal tool with specified parameters and saves the output with run index.

    Parameters:
    - model (str): The model to be used by molpal.
    - confid (str): The confidence method to be used.
    - metrics (str): The metric for evaluation.
    - init (int): Initial size of the dataset.
    - batches (int): Batch size for iterations.
    - max_iter (int): Maximum number of iterations.
    - k (int): Number of top samples to select.
    - BETA (float): Beta parameter for UCB.
    - n (int): Run index for identifying multiple runs.

    This function executes molpal with the given parameters, renames the final output file with the run index,
    and moves the output directory to a unique location based on the run index.
    """

    subprocess.run([
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
    ])

    output_filename_csv = f"/content/molpal/folder_output/run_output/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}.csv"
    os.rename("/content/molpal/folder_output/run_output/all_explored_final.csv", output_filename_csv)

    output_folder_name = f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}"
    os.rename("/content/molpal/folder_output/run_output", output_folder_name)


def molpal_run_random3(model, confid, init, batches, max_iter, k, BETA, n):
    """
    Runs the molpal tool with random metrics and saves the output with run index.

    Parameters: same as molpal_run3 without metrics, because we only use random here

    This function executes molpal with random metrics, renames the final output file with the run index,
    and moves the output directory to a unique location based on the run index.
    """

    subprocess.run([
        "molpal", "run",
        "--write-intermediate", "--write-final", "--retrain-from-scratch",
        "--library", "/content/molpal/data/Enamine10k_scores.csv.gz",
        "-o", "lookup",
        "--objective-config", "/content/molpal/examples/objective/Enamine10k_lookup.ini",
        "--model", model,
        "--conf-method", confid,
        "--metric", "random",
        "--init-size", str(init),
        "--batch-size", str(batches),
        "--max-iters", str(max_iter),
        "--fps", "/content/molpal/folder_output/fps_file.h5",
        "--output-dir", "run_output",
        "-k", str(k),
        "--beta", str(BETA)
    ])

    output_filename_csv = f"/content/molpal/folder_output/run_output/all_explored_final_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}.csv"
    os.rename("/content/molpal/folder_output/run_output/all_explored_final.csv", output_filename_csv)

    output_folder_name = f"/content/molpal/folder_output/run_output_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}"
    os.rename("/content/molpal/folder_output/run_output", output_folder_name)


def experiment_multirun(model, confid, metrics, init, batches, max_iter, dataset, top_x_perc, k, BETA, n):
    """
    Executes multiple runs of molpal with specified and random metrics, evaluates the fraction of top X% molecules found, and calculates the Enrichment Factor (EF).

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
    - n (int): Run index for identifying multiple runs.

    Functionality:
    1. Runs molpal with specified metrics and saves the output with the run index.
    2. Runs molpal with random metrics and saves the output with the run index.
    3. Reads the output files from both runs.
    4. Calculates the fraction of top X% molecules found in each run.
    5. Prints and returns the fraction for both runs and the Enrichment Factor (EF).

    Outputs:
    - Prints the fraction of top X% molecules found for both specified and random metrics.
    - Prints the calculated Enrichment Factor (EF).
    - Returns the fraction for both runs and the EF.
    """

    # Run molpal with the specified metrics
    molpal_run3(model, confid, metrics, init, batches, max_iter, k, BETA, n)

    # Run molpal with random metrics
    molpal_run_random3(model, confid, init, batches, max_iter, k, BETA, n)

    # Define file paths for the output CSV files
    csv_file_random = f'/content/molpal/folder_output/run_output_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}/all_explored_final_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}.csv'
    csv_file = f'/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}.csv'

    # Calculate the fraction of top X% molecules found for the specified metrics
    frac = frac_top_x(top_x_perc, dataset, csv_file)

    # Calculate the fraction of top X% molecules found for the random metrics
    frac_random = frac_top_x(top_x_perc, dataset, csv_file_random)

    # Print the results
    print("The percentage of molecules recovered from the top", top_x_perc, "% is", frac, "% for", metrics, "and", frac_random, "% for random")

    # Calculate the Enrichment Factor (EF)
    EF = frac / frac_random

    # Print the Enrichment Factor (EF)
    print("The Enrichment Factor (EF) was calculated to be", EF)

    # Return the fractions and the Enrichment Factor (EF)
    return frac, EF


def experiment_multirun2(model, confid, metrics, init, batches, max_iter, dataset, top_x_perc, k, BETA, n):
    """
    Executes a single run of molpal with specified metrics and evaluates the fraction of top X% molecules found.

    Parameters: same as experiment_multirun

    Functionality:
    1. Runs molpal with the specified metrics and saves the output with the run index.
    2. Reads the output file from the run.
    3. Calculates the fraction of top X% molecules found.
    4. Prints and returns the fraction of top X% molecules found.

    Outputs:
    - Prints the fraction of top X% molecules found.
    - Returns the fraction of top X% molecules found.
    """

    # Run molpal with the specified metrics
    molpal_run3(model, confid, metrics, init, batches, max_iter, k, BETA, n)

    # Define the file path for the output CSV file
    csv_file = f'/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}.csv'

    # Calculate the fraction of top X% molecules found
    frac = frac_top_x(top_x_perc, dataset, csv_file)

    # Print the result
    print("The percentage of molecules recovered from the top", top_x_perc * 100, "% is", frac, "% for", metrics)

    # Return the fraction
    return frac
