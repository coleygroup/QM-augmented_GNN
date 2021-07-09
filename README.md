# QM-augmented_GNN

This repository contains the main code associated with the project "Quantum chemistry augmented neuralnetworks for reactivity prediction: Performance, generalizability and interpretability", cf xxx. Note that the graph neural networks presented here are adaptations of the original models developed by Yanfei Guan and co-workers. For more information, see the original repository [reactivity_predictions_substitution](https://github.com/yanfeiguan/reactivity_predictions_substitution).

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

## Predicting
To use the trained model, run:

```
python reactivitiy -m <mode> --data_path <path to the predicting .csv file> --model_dir <directory containing the trained model> -p 
```

where `data_path` is path to the predicting `.csv` file, whose format is the same as the one discussed. `model_dir` is the directory holding the trained model. 
The model must be named as `best_model.hdb5` and stores parameters only. The `model_dir` must also include a `scalers.pickle` under `ml_QM_GNN` mode as discussed in the
[training](#Training) session.
