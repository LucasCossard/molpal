import subprocess
import os
import pandas as pd

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

    df1 = pd.read_csv(f"{csv_all}")

    df2 = pd.read_csv(f"{csv_allselected}")

    matches = df2[['smiles']].merge(df1.reset_index(), on='smiles', how='inner')

    matched_rows = df1.loc[matches['index']]

    new_df = matched_rows.copy()

    new_df1 = new_df.sort_values(by = 'score', ascending = False)

    return new_df1.to_csv(output_csv, index=False)


def molpal_run(model, confid, metrics, init, batches, max_iter, k, BETA, dataset, top_x_perc):
    """
    Runs the molpal tool with specified parameters, processes the results, and evaluates the fraction of top X% molecules found.

    Parameters:
    - model (str): The model to be used by molpal.
    - confid (str): The confidence method to be used.
    - metrics (str): The metric for evaluation.
    - init (int): Initial size of the dataset.
    - batches (int): Batch size for iterations.
    - max_iter (int): Maximum number of iterations.
    - k (int): Number of top samples to select.
    - BETA (float): Beta parameter for model configuration.
    - dataset (int): Dataset identifier used to specify the dataset file.
    - top_x_perc (float): The top X percentage of molecules to consider.
    """

    command = [
        "molpal", "run",
        "--write-intermediate", "--write-final", "--retrain-from-scratch",
        "--library", f"/content/molpal/data/Enamine{dataset}k_scores.csv.gz",
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
    os.rename("/content/molpal/folder_output/run_output/all_explored_final.csv", output_filename_csv)

    output_folder_name = f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}"
    os.rename("/content/molpal/folder_output/run_output", output_folder_name)

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
                           f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_data"
                           )

    selection_fct(f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_data",
                  f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}.csv",
                  f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}/all_selected_data"
    )
