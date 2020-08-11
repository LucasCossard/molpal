# can add support for those who don't have configargparse
from configargparse import ArgumentTypeError, ArgumentParser, Namespace
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Union

# os.sched_getaffinity(0) returns the set of CPUs this process can use,
# but it is defined only for some UNIX platforms, so try to use it and, failing
# that, assume this process can use all system CPUs
try:
    MAX_CPU = len(os.sched_getaffinity(0))
except AttributeError:
    MAX_CPU = mp.cpu_count()

#################################
#       GENERAL ARGUMENTS       #
#################################
def add_general_args(parser: ArgumentParser) -> None:
    parser.add_argument('--config', is_config_file=True,
                        help='the filepath of the configuration file')
    parser.add_argument('--name',
                        help='the general name to be used for outputs')
    parser.add_argument('--seed', type=int,
                        help='the random seed to use for initialization.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='the level of output this program should print')
    parser.add_argument('-nj', '--njobs', 
                        default=MAX_CPU, type=int, metavar='N_JOBS',
                        help='the total number of cores available')

    parser.add_argument('--retrain-from-scratch', 
                        action='store_true', default=False,
                        help='whether the model should be retrained from scratch at each iteration as opposed to retraining online.')

    parser.add_argument('--write-intermediate', 
                        action='store_true', default=False,
                        help='whether to write a summary file with all of the explored inputs and their associated scores after each round of exploration')
    parser.add_argument('--write-final', action='store_true', default=False,
                        help='whether to write a summary file with all of the explored inputs and their associated scores')
    parser.add_argument('-m', '--top-m', type=restricted_float_or_int, 
                        default=1., dest='m',
                        help='the number of top inputs expressed either as a number or as a fraction of the pool to write, if necessary')

    parser.add_argument('--save-preds', action='store_true', default=False,
                        help='whether to write the full prediction data to a file each time the predictions are updated')
    parser.add_argument('--save-state', action='store_true', default=False,
                        help='whether to save the state of the explorer before each batch')

    parser.add_argument('--previous-scores',
                        help='the path to a file containing the scores from a previous run of molpal to load in as preliminary dataset.')
    parser.add_argument('--scores-csvs', nargs='+',
                        help='A list of filepaths containing the output from a previous exploration to read in and mimic its intermediate state. Will load these files in the order in which they are passed.')

    parser.add_argument('--root', default='.',
                        help='the root directory under which to organize all program outputs')
    parser.add_argument('--tmp', default=os.environ.get('TMP', '.'),
                        help='the path of your system\'s tmp or scratch directory')

#####################################
#       ENCODER ARGUMENTS           #
#####################################
def add_encoder_args(parser: ArgumentParser) -> None:
    parser.add_argument('--encoder', default='morgan',
                        choices={'morgan', 'rdkit', 'pair', 'maccs', 'map4'},
                        help='the type of encoder to use')
    parser.add_argument('--radius', type=int, default=2,
                        help='the radius to use for circular fingerprints')
    parser.add_argument('--length', type=int, default=2048,
                        help='the length of the fingerprint')

##############################
#       POOL ARGUMENTS       #
##############################
def add_pool_args(parser: ArgumentParser) -> None:
    parser.add_argument('--pool', default='eager',
                        help='the type of MoleculePool to use')

    parser.add_argument('--library', required=True, metavar='LIBRARY_FILEPATH',
                        help='the file containing members of the MoleculePool')
    parser.add_argument('--no-title-line', action='store_true', default=False,
                        help='whether there is no title line in the library file')
    parser.add_argument('--delimiter', default=',',
                        help='the column separator in the library file')
    parser.add_argument('--smiles-col',
                        help='the column containing the SMILES string in the library file')
    parser.add_argument('--fps', metavar='FPS_FILEPATH.<h5/hdf5>',
                        help='an hdf5 file containing the precalculated feature representations of the molecules')
    parser.add_argument('--cluster', action='store_true', default=False,
                        help='whether to cluster the MoleculePool')
    parser.add_argument('--cache', action='store_true', default=False,
                        help='whether to store the full MoleculePool in memory')
    parser.add_argument('--validated', action='store_true', default=False,
                        help='whether the pool has been manually validated and invalid SMILES strings have been removed.')

#####################################
#       ACQUISITION ARGUMENTS       #
#####################################
def add_acquisition_args(parser: ArgumentParser) -> None:
    parser.add_argument('--metric', '--alpha', default='random',
                        choices={'random', 'greedy', 'threshold',
                                 'ucb', 'ei', 'pi', 'thompson'},
                        help='the acquisition metric to use')

    parser.add_argument('--init-size', 
                        type=restricted_float_or_int, default=0.01,
                        help='the number of ligands or fraction of total library to initially dock')
    parser.add_argument('--batch-size',
                        type=restricted_float_or_int, default=0.01,
                        help='the number of ligands or fraction of total library for each batch of exploration')
    parser.add_argument('--epsilon', type=float, default=0.,
                        help='the fraction of each batch that should be acquired randomly')

    parser.add_argument('--temp-i', type=float,
                        help='the initial temperature for tempeture scaling when calculating the decay factor for cluster scaling')
    parser.add_argument('--temp-f', type=float, default=1.,
                        help='the final temperature used in the greedy metric')

    parser.add_argument('--xi', type=float, default=0.01,
                        help='the xi value to use in EI and PI metrics')
    parser.add_argument('--beta', type=int, default=2,
                        help='the beta value to use in the UCB metric')
    parser.add_argument('--threshold', type=float,
                        help='the threshold value as a positive number to use in the threshold metric')

###################################
#       OBJECTIVE ARGUMENTS       #
###################################
def add_objective_args(parser: ArgumentParser) -> None:
    parser.add_argument('-o', '--objective', required=True,
                        choices={'lookup', 'docking'},
                        help='the objective function to use')
    parser.add_argument('--minimize', action='store_true', default=False,
                        help='whether to minimize the objective function')

    # DockingObjective args
    parser.add_argument('-d', '--docker', default='vina',
                        required='docking' in sys.argv,
                        choices={'vina', 'psovina', 'smina', 'qvina'},
                        help='the name of the docking program to use')
    parser.add_argument('-r', '--receptor',
                        required='docking' in sys.argv,
                        help='the filename of the receptor')
    parser.add_argument('-c', '--center', type=float, nargs=3,
                        metavar=('CENTER_X', 'CENTER_Y', 'CENTER_Z'),
                        required='docking' in sys.argv,
                        help='the x-, y-, and z-coordinates of the center of the docking box')
    parser.add_argument('-s', '--size', type=int, nargs=3,
                        metavar=('SIZE_X', 'SIZE_Y', 'SIZE_Z'),
                        required='docking' in sys.argv,
                        help='the x-, y-, and z-dimensions of the docking box')
    parser.add_argument('-nc', '--ncpu', default=4, type=int, metavar='N_CPU',
                        help='the number of cores to run the docking program with')
    parser.add_argument('--opt', action='store_true', default=False,
                        help='(NOT IMPLEMENTED) whether the docking should self-optimize the external:internal parallelism ratio')
    parser.add_argument('--boltzmann', action='store_true', default=False,
                        help='whether to calculate the boltzmann average of the docking scores')

    # LookupObjective args
    parser.add_argument('--lookup-path',
                        required='lookup' in sys.argv,
                        help='filepath pointing to a file containing lookup scoring data')
    parser.add_argument('--no-lookup-title-line', action='store_true',
                        default=False,
                        help='whether there is a title line in the data lookup file')
    parser.add_argument('--lookup-sep', default=',',
                        help='the column separator in the data lookup file')
    parser.add_argument('--lookup-smiles-col', default=0, type=int,
                        help='the column containing the SMILES strings in the data lookup file')
    parser.add_argument('--lookup-data-col', default=1, type=int,
                        help='the column containing the score data in the data lookup file')

def modify_objective_args(args: Namespace) -> None:
    if args.objective == 'docking':
        modify_DockingObjective_args(args)

    elif args.objective == 'lookup':
        modify_LookupObjective_args(args)

def modify_DockingObjective_args(args: Namespace) -> None:
    rec = Path(args.receptor).stem
    lib = Path(args.library).stem

    args.name = args.name or f'{rec}_{lib}'

    # input files are generic, so are named solely by their ligand supply file
    if args.input is None:
        args.input = f'input/{lib}'
    if args.output is None:
        args.output = f'output/{args.name}'

    args.ncpu = min(MAX_CPU, args.ncpu)
    args.njobs = min(MAX_CPU, args.njobs)

def modify_LookupObjective_args(args: Namespace) -> None:
    args.lookup_title_line = not args.no_lookup_title_line
    delattr(args, 'no_lookup_title_line')

    if args.name is None:
        args.name = Path(args.library).stem

###############################
#       MODEL ARGUMENTS       #
###############################
def add_model_args(parser: ArgumentParser) -> None:
    parser.add_argument('--model', choices={'rf', 'gp', 'nn', 'mpn'},
                        default='rf',
                        help='the model type to use')
    parser.add_argument('--test-batch-size', type=int,
                        help='the size of batch of predictions during model inference')
    
    # GP args
    parser.add_argument('--gp-kernel', choices={'dotproduct'},
                        default='dotproduct',
                        help='Kernel to use for Gaussian Process model')

    ## NN args
    # parser.add_argument('--ensemble-size', type=int, default=5,
    #                     help='Number of independent models to fit')

    # MPNN args
    parser.add_argument('--device', choices={'cpu', 'cuda'}, default='cpu',
                        help='the device on which to run MPNN training/inference. Not specifying will choose based on what\'s available.')
    parser.add_argument('--init-lr', type=float, default=1e-4,
                        help='the initial learning rate for the MPNN model')
    parser.add_argument('--max-lr', type=float, default=1e-3,
                        help='the maximum learning rate for the MPNN model')
    parser.add_argument('--final-lr', type=float, default=1e-4,
                        help='the final learning rate for the MPNN model')

    # NN/MPNN args
    parser.add_argument('--conf-method', default='ensemble',
                        choices={'ensemble', 'twooutput', 
                                 'mve', 'dropout', 'none'},
                        help='Confidence estimation method for NN/MPNN models')

##################################
#       STOPPING ARGUMENTS       #
##################################
def add_stopping_args(parser: ArgumentParser) -> None:
    parser.add_argument('-k', '--top-k', dest='k',
                        type=restricted_float_or_int, default=0.0005,
                        help='the top k ligands from which to calculate an average score expressed either as an integer or as a fraction of the pool')
    parser.add_argument('-w', '--window-size', type=int, default=3,
                        help='the window size to use for calculation of the moving average of the top-k scores')
    parser.add_argument('--delta', type=restricted_float, default=0.01,
                        help='the minimum acceptable difference between the moving average of the top-k scores and the current average the top-k score in order to continue exploration')
    parser.add_argument('--max-epochs', type=int, default=50,
                        help='the maximum number of epochs to explore for')
    parser.add_argument('--max-explore', 
                        type=restricted_float_or_int, default=1.0,
                        help='the maximum number of inputs to explore')

def cleanup_args(args: Namespace) -> None:
    """Remove unnecessary arguments"""
    args.title_line = not args.no_title_line

    args_to_remove = {'no_title_line'}

    docking_args = {'docker', 'receptor', 'center', 'size', 'ncpu',
                    'boltzmann', 'opt'}
    lookup_args = {'lookup_path', 'no_lookup_title_line', 'lookup_smiles_col',
                    'lookup_data_col', 'lookup_sep'}
    
    gp_args = {'gp_kernel'}
    nn_args = set() # {'ensemble_size'}
    mpnn_args = {'device', 'init_lr', 'max_lr', 'final_lr'}

    if args.objective == 'docking':
        args_to_remove |= lookup_args
    elif args.objective == 'lookup':
        args_to_remove |= docking_args

    if args.metric != 'ei' or args.metric != 'pi':
        args_to_remove.add('xi')
    if args.metric != 'ucb':
        args_to_remove.add('beta')
    if args.metric != 'threshold':
        args_to_remove.add('threshold')

    if not args.cluster:
        args_to_remove |= {'temp_i', 'temp_f'}

    if args.model != 'gp':
        args_to_remove |= gp_args
    if args.model != 'nn':
        args_to_remove |= nn_args
    if args.model != 'mpn':
        args_to_remove |= mpnn_args
    if args.model != 'nn' and args.model != 'mpn':
        args_to_remove.add('conf_method')

    for arg in args_to_remove:
        delattr(args, arg)

def gen_args(args=None) -> Namespace:
    parser = ArgumentParser()

    add_general_args(parser)
    add_encoder_args(parser)
    add_pool_args(parser)
    add_acquisition_args(parser)
    add_objective_args(parser)
    add_model_args(parser)
    add_stopping_args(parser)

    args = parser.parse_args(args)

    modify_objective_args(args)
    
    cleanup_args(args)

    return args

##############################
#       TYPE FUNCTIONS       #
##############################
def restricted_float_or_int(arg: str) -> Union[float, int]:
    try:
        value = int(arg)
        if value < 0:
            raise ArgumentTypeError(f'{value} is less than 0')
    except ValueError:
        value = float(arg)
        if value < 0 or value > 1:
            raise ArgumentTypeError(f'{value} must be in [0,1]')
    
    return value

def restricted_float(arg: str) -> float:
    value = float(arg)
    if value < 0 or value > 1:
        raise ArgumentTypeError(f'{value} must be in [0,1]')
    
    return value
