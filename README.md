![Project Logo](assets/Molpal_Logo_README_Best.png)

<h1 align="center">
Understanding and Leveraging Active Learning Through MolPAL
</h1>

**Note :This repository is a fork from https://github.com/coleygroup/molpal**

**Any file added or modified within the last 2 years must be considered as part of our `molpal version`.**

## 🔍 Overview 🔎

Molpal (Molecular Pool-based Active Learning) is a software created by the coleygroup for the accelerated discovery of compounds in high-throughput virtual screening environments.
Using  a modified version of Molpal, the objective of this project is to find a good way to understand and leverage active learning to minimize the number of total molecules acquired by the algorithm, while recovering a good percentage of the top X%  of the candidates depending on the dataset.

## 🔨 Installation 🔨

1️⃣ Clone the modified version of Molpal from this repository:

```
git clone https://github.com/Nohalyan/molpal
```

2️⃣ Install the necessary dependencies.

2️⃣.1️⃣ You can use the environment.yml for Conda:
```
cd /path_to_molpal
```
Then create the environment: 
```
conda env create -f environment.yml
```
And activate the environment::
```
conda activate molpal
```

2️⃣.2️⃣ Or use the 'requirements.txt' using pip after the  installation of the repository.

```
pip install -r requirements.txt
```

3️⃣ Install the repository:
 
```
pip install -e 
```

4️⃣ Import all the functions using the following command:

```
from molpal.acquirer import Acquirer
from molpal.featurizer import Featurizer
from molpal.models import Model
from molpal.objectives import Objective
from molpal.pools import pool
from molpal.exceptions import IncompatibilityError, InvalidExplorationError
from molpal.explorer import Explorer
```

## 📝 Notebooks 📝
In the notebooks file, we provide 3 notebooks we used to carry out our various experiments. 

Notebook 1️⃣ to observe the effects of:
- UCB & Score correlation
- Initialization size
- UCB & β study
- Acquisition function on active learning
- Diversity
- UCB & Score correlation on an Artficial 10k dataset

Notebook 2️⃣ to:
- Generate an Artficial 10k dataset usign HTS

Notebook 3️⃣ to plot:
- UMAP for Enamine50k

## 📁 Functions 📁
We added the file `AI4Chem.py` in `/molpal/molapl/`, listing all the functions we have defined for this project. 

Unfortunately, some of these functions use molpal with `!molpal run`, which makes this file unusable, so it is present as a showcase of our functions.

These functions are of course operational in notebooks. 

## 📗 License 📕 
This project and Molpal software used are licensed under the MIT License.

## 📜 References 📜

**This repository is a fork from https://github.com/coleygroup/molpal**

This project is based on Molpal software, developed by the Coley Group: https://github.com/coleygroup/molpal

## 📖 Authors 📖
- Gaedecke Kelian: https://github.com/KelianGaedecke
- Segura David: https://github.com/Davidmingsegura
- Cossard Lucas: https://github.com/Nohalyan

**This project was carried out as part of EPFL's AI for Chemistry course.**
