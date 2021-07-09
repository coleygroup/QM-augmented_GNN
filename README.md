# QM-augmented_GNN

This repository contains the main code associated with the project "Quantum chemistry augmented neuralnetworks for reactivity prediction: Performance, generalizability and interpretability", cf xxx. Note that each of the graph neural networks presented here are adaptations of the original models developed by Yanfei Guan and co-workers. For more information, see the original repository [reactivity_predictions_substitution](https://github.com/yanfeiguan/reactivity_predictions_substitution).

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

...

## Training
This repository cotains three main directories, each providing two graph neural network models tailored to the considered data set and task, as described in the paper.

### GNN
A conventional graph neural network that relies only on the machine learned reaction representation of a given reaction. 
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

This is the fusion model, which combines machine learned reaction representation and on-the-fly
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
