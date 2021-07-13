# QM-augmented_GNN

This repository contains the main code associated with the project "Quantum chemistry augmented neuralnetworks for reactivity prediction: Performance, generalizability and interpretability", cf xxx. Note that each of the graph neural networks presented here are adaptations of the original models developed by Yanfei Guan and co-workers. For more information, see the repository [reactivity_predictions_substitution](https://github.com/yanfeiguan/reactivity_predictions_substitution).

## Requirements

1. python 3.7
2. tensorflow 2.0.0
3. rdkit
3. [qmdesc](https://github.com/yanfeiguan/chemprop-atom-bond) (python package for predicting QM descriptors on the fly)

### Conda environment
To set up a conda environment:
```
conda env create --name <env-name> --file environment.yml
```

## Data

Curated data sets have been included in each of the main directories in a format that is compatible with the respective models (cf the `datasets` directories). In the case of `regression_e2_sn2`, data points are formatted as follows:

```
,reaction_id,smiles,reaction_core,activation_energy
0,0,[NH2:1][C@H:2]([C@H:3]([Cl:4])[N+:5](=[O:6])[O-:7])[N+:8](=[O:9])[O-:10].[H-:11],"[[3, 2, 1, 10]]",1.886101230187571
1,1,[CH3:1][CH2:2][C@:3]([NH2:4])([F:5])[N+:6](=[O:7])[O-:8].[H-:9],"[[4, 2, 1, 8]]",16.259824710850626
```
where smiles corresponds to the reactant smiles and reaction_core indicates the index of the sites/heavy atoms undergoing a change in their bonding situation throughout the reaction (indexing starts from 0). Note that the numbering of the reactant smiles has to be ordered (i.e., the atom at index 0 carries number 1, the atom at index 1 carries number 2 etc.).

In the case of `classification_e2_sn2`, data points are formatted in a similar manner, but now the two reaction cores for the competing reaction pathways are included in a single data point, with the pathway to which the lowest-energy transition state is associated listed first:

```
,reaction_id,smiles,products_run
0,0,[N:1]#[C:2][C@@H:3]([NH2:4])[CH2:5][Br:6].[H-:7],"[[5, 4, 6], [5, 4, 2, 6]]"
1,1,[CH3:1][C@@H:2]([NH2:3])[CH2:4][Cl:5].[F-:6],"[[4, 3, 5], [4, 3, 1, 5]]"
```

In the case of `classification_aromatic_substitution`, data points are formatted as:

```
,reaction_id,rxn_smiles,products_run
,reaction_id,rxn_smiles,products_run,PatentNumber
0,86,[CH3:1][O:2][C:3](=[O:4])[c:5]1[cH:6][cH:7][c:8]2[c:9]([cH:10]1)[O:11][CH2:12][CH2:13][O:14]2.[N+:15]([O-:16])([OH:17])=[O:18]>C(O)(=O)C>[CH3:1][O:2][C:3](=[O:4])[c:5]1[c:6]([N+:15]([O-:17])=[O:18])[cH:7][c:8]2[c:9]([cH:10]1)[O:11][CH2:12][CH2:13][O:14]2,[CH3:1][O:2][C:3](=[O:4])[c:5]1[c:6]([N+:15]([O-:17])=[O:18])[cH:7][c:8]2[c:9]([cH:10]1)[O:11][CH2:12][CH2:13][O:14]2.[CH3:1][O:2][C:3](=[O:4])[c:5]1[cH:6][cH:7][c:8]2[c:9]([c:10]1[N+:15]([O-:17])=[O:18])[O:11][CH2:12][CH2:13][O:14]2.[CH3:1][O:2][C:3](=[O:4])[c:5]1[cH:6][c:7]([N+:15]([O-:17])=[O:18])[c:8]2[c:9]([cH:10]1)[O:11][CH2:12][CH2:13][O:14]2,US03931179
1,126,[F:1][c:2]1[cH:3][cH:4][c:5]([C:6]([CH2:7][CH2:8][C:9]([OH:10])=[O:11])=[O:12])[cH:13][cH:14]1.[N+:15]([O-:16])([OH:17])=[O:18]>>[F:1][c:2]1[cH:3][cH:4][c:5]([C:6]([CH2:7][CH2:8][C:9]([OH:10])=[O:11])=[O:12])[cH:13][c:14]1[N+:15]([O-:17])=[O:18],[F:1][c:2]1[cH:3][cH:4][c:5]([C:6]([CH2:7][CH2:8][C:9]([OH:10])=[O:11])=[O:12])[cH:13][c:14]1[N+:15]([O-:17])=[O:18].[F:1][c:2]1[cH:3][cH:4][c:5]([C:6]([CH2:7][CH2:8][C:9]([OH:10])=[O:11])=[O:12])[c:13]([N+:15]([O-:17])=[O:18])[cH:14]1,US03931177
```

in which, rxn_smiles are the full reaction SMILES and products_run are the potential products (major.minor1.minor2.....).

## Training
This repository cotains three main directories, each providing two graph neural network models (GNN and ml-QM-GNN), tailored to the considered data set and task, as described in the paper.

### GNN
Conventional graph neural networks that relies only on the machine learned reaction representation of a given reaction. 
To train the model, run:
```
python reactivitiy.py -m GNN --data_path <path to the .csv file> --model_dir <directory to save the trained model> 
```

For example, to train the model on the E2/SN2 data set for barrier height prediction (cf the "regression_e2_sn2" directory):
```angular2
python reactivitiy.py -m GNN --data_path datasets/e2_sn2_regression.csv --model_dir trained_model/GNN_e2_sn2
```

A checkpoint file, `best_model.hdf5`, will be saved in the `trained_model/GNN_e2_sn2` directory.

### ml-QM-GNN

These are the fusion models, which combine machine learned reaction representation and on-the-fly
calculated QM descriptors. To use this architecture, the [Chemprop-atom-bond](https://github.com/yanfeiguan/chemprop-atom-bond) 
must be installed. To train the model, run:

```
python reactivitiy.py --data_path <path to the .csv file> --model_dir <directory to save the trained model> 
``` 

The `reactivity.py` use `ml-QM-GNN` mode by default. The workflow first predict QM atomic/bond descriptors for all reactants found in the reactions.
The predicted descriptors are then scaled through a min-max scaler. A dictionary containing scikit-learn scaler object will be saved 
as `scalers.pickle` in the `model_dir` for later predicting tasks. A checkpoint file, `best_model.hdf5` will also be saved in the `model_dir`

For example:
```angular2
python reactivitiy.py -m ml_QM_GNN --data_path datasets/e2_sn2_regression.csv --model_dir trained_model/GNN_e2_sn2
```

## Predicting
To use the trained model, run:

```
python reactivitiy -m <mode> --data_path <path to the predicting .csv file> --model_dir <directory containing the trained model> -p 
```

where `data_path` is path to the predicting `.csv` file, whose format is the same as the one discussed. `model_dir` is the directory holding the trained model. 
The model must be named as `best_model.hdb5` and stores parameters only. The `model_dir` must also include a `scalers.pickle` under `ml_QM_GNN` mode as discussed in the
[training](#Training) session.
