'''
This file contains all the functions defined and used in this project. 
As some of these functions use molpal with ‘!molpal run’, this file is not usable and is a showcase of our functions.
'''
#=======================================================================================================================================================================================================================================

import pandas as pd
import numpy as npf
import gzip
import shutil
import csv
import gzip
import umap
import matplotlib.pyplot as plt
import more_itertools as mit
import seaborn as sns
import random
import torch

from io import StringIO
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rdkit.DataStructs import BulkTanimotoSimilarity

#=======================================================================================================================================================================================================================================

'''
# Let's define a function to merge all the data generated from the run
'''

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

    import pandas as pd

    df1 = pd.read_csv(f"{csv_all}")

    df2 = pd.read_csv(f"{csv_allselected}")

    matches = df2[['smiles']].merge(df1.reset_index(), on='smiles', how='inner')

    matched_rows = df1.loc[matches['index']]

    new_df = matched_rows.copy()

    new_df1 = new_df.sort_values(by = 'score', ascending = False)

    return new_df1.to_csv(output_csv, index=False)

#=======================================================================================================================================================================================================================================

'''
# UMAP functions: Create a file to have all the molecules on the map and integrates the UMAP visualization.
'''

def create_combined_csv(smiles_csv_gz_file, fps_h5_file, output_csv_file):
      """
    Reads SMILES and scores from a compressed CSV file, reads fingerprints from an HDF5 file,
    combines the data, and writes the combined data into a new CSV file.

    Parameters:
    - smiles_csv_gz_file (str): Path to the compressed CSV file containing SMILES and scores.
    - fps_h5_file (str): Path to the HDF5 file containing fingerprints.
    - output_csv_file (str): Path to the output CSV file.

    Returns:
    - None
    """

    # Read smiles and scores from the compressed CSV file
    with gzip.open(smiles_csv_gz_file, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Read the headers
        if headers != ['smiles', 'score']:
            print("Error: Incorrect headers. The headers must be 'smiles' and 'score'.")
            return
        smiles_scores = [(row[0], row[1]) for row in reader]

    # Read fingerprints from the HDF5 file
    with h5py.File(fps_h5_file, 'r') as h5file:
        fps = h5file['fps'][:]

    # Check if the number of smiles matches the number of fingerprints
    if len(smiles_scores) != len(fps):
        print("Error: Number of smiles and fingerprints don't match")
        return

    # Combine smiles, fps, and scores into a list of rows
    combined_data = [(smiles, *fps_values, score) for (smiles, score), fps_values in zip(smiles_scores, fps)]

    # Write the combined data to a new CSV file
    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Smiles'] + [f'FP{i}' for i in range(len(fps[0]))] + ['Score'])
        writer.writerows(combined_data)

# Usage example
smiles_score = '/content/molpal/data/Enamine10k_scores.csv.gz'
fps = '/content/molpal/folder_output/fps_file.h5'
create_combined_csv(smiles_score, fps, '/content/molpal/folder_output/smiles_fps_score.csv')


def smiles_difference(dataset1_path, dataset2_path, output_path):
    df1 = pd.read_csv(dataset1_path)
    df2 = pd.read_csv(dataset2_path)

    df1.rename(columns={df1.columns[0]: 'smiles'}, inplace=True)
    df2.rename(columns={df2.columns[0]: 'smiles'}, inplace=True)

    df1['Mol'] = df1['smiles'].apply(Chem.MolFromSmiles)
    df2['Mol'] = df2['smiles'].apply(Chem.MolFromSmiles)

    df1['Canonical_SMILES'] = df1['Mol'].apply(Chem.MolToSmiles)
    df2['Canonical_SMILES'] = df2['Mol'].apply(Chem.MolToSmiles)

    common_smiles = set(df1['Canonical_SMILES']).intersection(set(df2['Canonical_SMILES']))

    df2_unique = df2[~df2['Canonical_SMILES'].isin(common_smiles)]

    df2_unique = df2_unique.drop(columns=['Mol', 'Canonical_SMILES'])

    df2_unique.to_csv(output_path, index=False)

def umap_visualization(csv_file, csv_explore):
    data = pd.read_csv(csv_file)
    explore = pd.read_csv(csv_explore)

    data_smiles = set(data['Smiles'])
    explore_smiles = set(explore['smiles'])
    common_smiles = data_smiles.intersection(explore_smiles)

    common_data = data[data['Smiles'].isin(common_smiles)]

    fingerprint_cols = [col for col in data.columns if 'FP' in col]
    common_fingerprint_cols = [col for col in common_data.columns if 'FP' in col]

    score_col = data.columns[-1]
    common_score_col = common_data.columns[-1]

    features = data[fingerprint_cols].values
    common_features = common_data[common_fingerprint_cols].values
    labels = data[score_col].values
    common_labels = common_data[common_score_col].values

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(features)
    embedding_2 = reducer.transform(common_features)

    return embedding, embedding_2, labels

base_path = "/content/molpal/data/beta_50k/"
dataset_paths = glob.glob(os.path.join(base_path, "top_*_explored_iter_*_beta_4.csv"))

def extract_num(path):
    match = re.search(r'top_(\d+)_explored_iter_(\d+)_beta_4', path)
    return int(match.group(2)), int(match.group(1))

dataset_paths = sorted(dataset_paths, key=extract_num)
print(dataset_paths)

og_database = '/content/molpal/folder_output/smiles_fps_score.csv'

cmap = cm.get_cmap('plasma', len(dataset_paths))

all_embeddings = []
all_labels = []

for i in range(1, len(dataset_paths)):
    dataset1_path = dataset_paths[i-1]
    dataset2_path = dataset_paths[i]
    output_path = f'unique_dataset_{i}.csv'

    smiles_difference(dataset1_path, dataset2_path, output_path)

    embedding, embedding_2, labels = umap_visualization(og_database, output_path)

    all_embeddings.append(embedding_2)
    all_labels.append(labels)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Greys', s=10, vmin=min(labels), vmax=max(labels))
    plt.scatter(embedding_2[:, 0], embedding_2[:, 1], color=cmap(i), s=10, label=f'iter {i}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    plt.legend()
    plt.savefig(f'umap_beta_4_iter_{i}.png')
    plt.show()


plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Greys', s=10, vmin=min(labels), vmax=max(labels))
for i, c_embedding in enumerate(all_embeddings):
    plt.scatter(c_embedding[:, 0], c_embedding[:, 1], color=cmap(i), s=10)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
#plt.title('UMAP Visualization for All Unique Datasets')
plt.legend()
plt.savefig('umap_beta_4_all_iter.png')
plt.show()

#=======================================================================================================================================================================================================================================

def Molpal_run(model, confid, metrics, init, batches, max_iter, k, BETA):
    """
    Runs the MolPAL algorithm with specified parameters and saves the results.

    Parameters:
    - model (str): Model to use for the MolPAL run.
    - confid (str): Confidence method.
    - metrics (str): Evaluation metric.
    - init (int): Initial dataset size.
    - batches (int): Batch size for each iteration.
    - max_iter (int): Maximum number of iterations.
    - k (int): Number of neighbors for k-nearest neighbors.
    - BETA (float): Beta parameter for the MolPAL run.

    Returns:
    - None
    """
    
    !molpal run --write-intermediate --write-final --retrain-from-scratch --library /content/molpal/data/Enamine10k_scores.csv.gz -o lookup --objective-config /content/molpal/examples/objective/Enamine10k_lookup.ini \
        --model {model} --conf-method {confid} --metric {metrics} --init-size {init} \
        --batch-size {batches} --max-iters {max_iter} --fps /content/molpal/folder_output/fps_file.h5 \
        --output-dir run_output -k {k} --beta {BETA}
    !mv /content/molpal/folder_output/run_output/all_explored_final.csv /content/molpal/folder_output/run_output/all_explored_final_beta_{BETA}.csv
    !mv /content/molpal/folder_output/run_output /content/molpal/folder_output/run_output_beta_{BETA}

def frac_top_x(top_x, csv_file):
    """
    Calculates the fraction of molecules from the selected smiles that are in the top X% of the dataset based on scores.

    Parameters:
    - top_x (int): Number of top molecules to consider.
    - csv_file (str): Path to the CSV file containing the selected smiles.

    Returns:
    - None
    """

    file_path = '/content/molpal/data/Enamine10k_scores.csv.gz'
    df = pd.read_csv(file_path)

    df_sorted = df.sort_values(by='score', ascending=False)

    top_x_smiles = df_sorted[['smiles', 'score']].iloc[:top_x]

    output_file_path = f'/content/molpal/data/Top_{top_x}_Enamine10k_scores.csv'
    top_x_smiles.to_csv(output_file_path, index=False)

    df_found = pd.read_csv(csv_file)
    df_top_x = pd.read_csv(output_file_path)

    smiles_found = set(df_found['smiles'])
    smiles_top_x = set(df_top_x['smiles'])

    common_smiles = smiles_found.intersection(smiles_top_x)

    num_common_smiles = len(common_smiles)
    print("Fraction of top", top_x, "smiles found:", num_common_smiles / len(smiles_top_x))

#=======================================================================================================================================================================================================================================

'''
# Let's define others functions to run molpal with specified parameters, processe the results, 
and evaluate the fraction of top X% molecules found, and also calculating the Enrichment Factor (EF).
'''

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

    !molpal run --write-intermediate --write-final --retrain-from-scratch --library /content/molpal/data/Enamine10k_scores.csv.gz -o lookup --objective-config /content/molpal/examples/objective/Enamine10k_lookup.ini \
        --model {model} --conf-method {confid} --metric {metrics} --init-size {init} \
        --batch-size {batches} --max-iters {max_iter} --fps /content/molpal/folder_output/fps_file.h5 \
        --output-dir run_output -k {k} --beta {BETA}

    output_filename_csv = f"/content/molpal/folder_output/run_output/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}.csv"
    !mv /content/molpal/folder_output/run_output/all_explored_final.csv {output_filename_csv}

    output_folder_name = f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}"
    !mv /content/molpal/folder_output/run_output {output_folder_name}


def molpal_run_random2(model, confid, init, batches, max_iter, k, BETA):
    """
    Runs the molpal tool with the metric set to random and specified parameters, then moves the generated output files to appropriate locations.

    Parameters: all listed above molpal_run2

    The function executes molpal with given parameters and the metric set to random, renames the final output file, and moves the output directory.
    """

    !molpal run --write-intermediate --write-final --retrain-from-scratch --library /content/molpal/data/Enamine10k_scores.csv.gz -o lookup --objective-config /content/molpal/examples/objective/Enamine10k_lookup.ini \
        --model {model} --conf-method {confid} --metric random --init-size {init} \
        --batch-size {batches} --max-iters {max_iter} --fps /content/molpal/folder_output/fps_file.h5 \
        --output-dir run_output -k {k} --beta {BETA}

    output_filename_csv = f"/content/molpal/folder_output/run_output/all_explored_final_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}.csv"
    !mv /content/molpal/folder_output/run_output/all_explored_final.csv {output_filename_csv}

    output_folder_name = f"/content/molpal/folder_output/run_output_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}"
    !mv /content/molpal/folder_output/run_output {output_folder_name}


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

#=======================================================================================================================================================================================================================================

'''
# The main function here is experiment_multirun wich can executes multiple runs of molapal 
if placed in a loop because we added n, a run index. 
It was used to calculate averages and standard deviations to generate more significant results. 
'''

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

    !molpal run --write-intermediate --write-final --retrain-from-scratch --library /content/molpal/data/Enamine10k_scores.csv.gz -o lookup --objective-config /content/molpal/examples/objective/Enamine10k_lookup.ini \
        --model {model} --conf-method {confid} --metric {metrics} --init-size {init} \
        --batch-size {batches} --max-iters {max_iter} --fps /content/molpal/folder_output/fps_file.h5 \
        --output-dir run_output -k {k} --beta {BETA}

    output_filename_csv = f"/content/molpal/folder_output/run_output/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}.csv"
    !mv /content/molpal/folder_output/run_output/all_explored_final.csv {output_filename_csv}

    output_folder_name = f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}"
    !mv /content/molpal/folder_output/run_output {output_folder_name}


def molpal_run_random3(model, confid, init, batches, max_iter, k, BETA, n):
    """
    Runs the molpal tool with random metrics and saves the output with run index.

    Parameters: same as molpal_run3 without metrics, beacasuse we only use random here

    This function executes molpal with random metrics, renames the final output file with the run index,
    and moves the output directory to a unique location based on the run index.
    """

    !molpal run --write-intermediate --write-final --retrain-from-scratch --library /content/molpal/data/Enamine10k_scores.csv.gz -o lookup --objective-config /content/molpal/examples/objective/Enamine10k_lookup.ini \
        --model {model} --conf-method {confid} --metric random --init-size {init} \
        --batch-size {batches} --max-iters {max_iter} --fps /content/molpal/folder_output/fps_file.h5 \
        --output-dir run_output -k {k} --beta {BETA}

    output_filename_csv = f"/content/molpal/folder_output/run_output/all_explored_final_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}.csv"
    !mv /content/molpal/folder_output/run_output/all_explored_final.csv {output_filename_csv}

    output_folder_name = f"/content/molpal/folder_output/run_output_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}"
    !mv /content/molpal/folder_output/run_output {output_folder_name}


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

#=======================================================================================================================================================================================================================================

'''
# Experiment_multirun2 is the same as ecperiment_multirun2, without the EF calculation. 
It has been defined specifically for the study of acquisition functions
'''
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

#=======================================================================================================================================================================================================================================

'''
# Experiment_multirun50k is the same as before, it just use Enamine50k instead of Enamine10k.
'''

def molpal_run50k(model, confid, metrics, init, batches, max_iter, k, BETA, n):
    !molpal run --write-intermediate --write-final --retrain-from-scratch --library /content/molpal/data/Enamine50k_scores.csv.gz -o lookup --objective-config /content/molpal/examples/objective/Enamine50k_lookup.ini \
        --model {model} --conf-method {confid} --metric {metrics} --init-size {init} \
        --batch-size {batches} --max-iters {max_iter} --fps /content/molpal/folder_output/fps_file.h5 \
        --output-dir run_output -k {k} --beta {BETA}

    output_filename_csv = f"/content/molpal/folder_output/run_output/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}.csv"
    !mv /content/molpal/folder_output/run_output/all_explored_final.csv {output_filename_csv}

    output_folder_name = f"/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}"
    !mv /content/molpal/folder_output/run_output {output_folder_name}


def molpal_run_random50k(model, confid, init, batches, max_iter, k, BETA, n):
    !molpal run --write-intermediate --write-final --retrain-from-scratch --library /content/molpal/data/Enamine50k_scores.csv.gz -o lookup --objective-config /content/molpal/examples/objective/Enamine50k_lookup.ini \
        --model {model} --conf-method {confid} --metric random --init-size {init} \
        --batch-size {batches} --max-iters {max_iter} --fps /content/molpal/folder_output/fps_file.h5 \
        --output-dir run_output -k {k} --beta {BETA}

    output_filename_csv = f"/content/molpal/folder_output/run_output/all_explored_final_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}.csv"
    !mv /content/molpal/folder_output/run_output/all_explored_final.csv {output_filename_csv}

    output_folder_name = f"/content/molpal/folder_output/run_output_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}"
    !mv /content/molpal/folder_output/run_output {output_folder_name}


def experiment_multirun50k(model, confid, metrics, init, batches, max_iter, dataset, top_x_perc, k, BETA, n):

      molpal_run50k(model, confid, metrics, init, batches, max_iter, k, BETA, n)
      molpal_run_random50k(model, confid, init, batches, max_iter, k, BETA, n)

      csv_file_random = f'/content/molpal/folder_output/run_output_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}/all_explored_final_{model}_random_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}.csv'
      csv_file = f'/content/molpal/folder_output/run_output_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}/all_explored_final_{model}_{metrics}_{init}_{batches}_{max_iter}_beta_{BETA}_run{n}.csv'

      frac = frac_top_x(top_x_perc, dataset, csv_file)

      frac_random = frac_top_x(top_x_perc, dataset, csv_file_random)

      print("The percentage of molecules recovered from the top", top_x_perc, "% is", frac, "% for", metrics, "and", frac_random, "% for random")

      EF = frac/frac_random

      print("The Enrichement Factor EF was calculated to be", EF)

      return frac, EF

#=======================================================================================================================================================================================================================================
'''
Tanimoto Similarity functions
'''

class NCircles:
    def __init__(self, threshold=0.75):
        self.sim_mat_func = similarity_matrix_tanimoto
        self.t = threshold

    def get_circles(self, args):
        vecs, sim_mat_func, t = args
        circs = []

        for vec in vecs:
            if len(circs) > 0:
                dists = 1.0 - sim_mat_func([vec], circs)
                if dists.min() <= t:
                    continue
            circs.append(vec)

        return circs

    def measure1(self, vecs, n_chunk=64):
        for i in range(3):
            chunk_size = max(1, n_chunk // (2 ** i))  # Ensure chunk_size is at least 1
            vecs_list = list(mit.chunked(vecs, chunk_size))
            args = zip(vecs_list, [self.sim_mat_func] * len(vecs_list), [self.t] * len(vecs_list))
            circs_list = list(map(self.get_circles, args))
            vecs = [c for ls in circs_list for c in ls]
            random.shuffle(vecs)

        vecs = self.get_circles((vecs, self.sim_mat_func, self.t))
        return len(vecs), vecs  # Ensure measure returns a tuple

    def measure(self, vecs, n_chunk=64):
        for i in range(3):
            vecs_list = list(mit.divide(n_chunk // (2 ** i), vecs))
            args = zip(vecs_list, [self.sim_mat_func] * len(vecs_list), [self.t] * len(vecs_list))
            circs_list = list(map(self.get_circles, args))
            vecs = [c for ls in circs_list for c in ls]
            random.shuffle(vecs)
        vecs = self.get_circles((vecs, self.sim_mat_func, self.t))
        return len(vecs), vecs


def get_ncircle(df):
    if 'FPS' not in df:
        df['FPS'] = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in df['MOL']]
    return NCircles().measure(df['FPS'])


def get_ncircle1(df):
    if 'FPS' not in df:
        df['FPS'] = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in df['MOL']]
    n_circles, vecs = NCircles().measure(df['FPS'])  # Ensure it returns a tuple
    return n_circles, vecs

def similarity_matrix_tanimoto(fps1, fps2):
    similarities = [BulkTanimotoSimilarity(fp, fps2) for fp in fps1]
    return np.array(similarities)

def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)

def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    else:
        return None

def bitvect_to_list(bitvect):
    return [int(bitvect.GetBit(i)) for i in range(bitvect.GetNumBits())]

ef plot_circles1(df, n_cluster):

    """
    This script provides functions for visualizing and determining the optimal number of clusters for molecular data.
    The steps include:

    1. **plot_circles1(df, n_cluster)**:
    - Unpack the number of circles and vectors from the dataset.
    - Convert molecular fingerprints to a numpy array.
    - Perform K-Means clustering on the data.
    - Use PCA to reduce the data to 2 dimensions.
    - Plot the reduced data with cluster labels for visual inspection.

    2. **elbow_method(data, max_k)**:
    - Compute the Sum of Squared Errors (SSE) for a range of cluster numbers.
    - Plot the SSE against the number of clusters to use the Elbow Method for determining the optimal number of clusters.
    """
    n_circles, vecs = get_ncircle1(df)  # Properly unpack the returned tuple
    print(f'Number of circles: {n_circles}')

    # Convert fingerprints to numpy array
    vecs_array = np.array([bitvect_to_list(fp) for fp in vecs])

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    cluster_labels = kmeans.fit_predict(vecs_array)

    # Perform PCA to reduce dimensions to 2
    pca = PCA(n_components=2)
    reduced_vecs = pca.fit_transform(vecs_array)

    # Plot the reduced vectors with cluster labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_vecs[:, 0], reduced_vecs[:, 1], c=cluster_labels, cmap='viridis', marker='o')
    plt.title('Chemical Space 2D Plot of Circles with Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

def elbow_method(data, max_k):
    iters = range(1, max_k+1, 1)
    sse = []

    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)  # WCSS

    plt.figure(figsize=(10, 8))
    plt.plot(iters, sse, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared distances (WCSS)')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid(True)
    plt.show()
