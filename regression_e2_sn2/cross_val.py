import os
import pickle

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import pandas as pd
from scipy.special import softmax

from tqdm import tqdm
from utils import lr_multiply_ratio, parse_args

args, dataloader, regressor = parse_args(cross_val=True)
reactivity_data = pd.read_csv(args.data_path, index_col=0)

if args.model == 'ml_QM_GNN':
    from ml_QM_GNN.graph_utils.mol_graph import initialize_qm_descriptors
    from predict_desc.predict_desc import predict_desc, reaction_to_reactants
    from predict_desc.post_process import min_max_normalize
    qmdf = predict_desc(args, normalize=False)
    qmdf.to_csv(args.model_dir + "/descriptors.csv")
    print(args.select_descriptors)

df = pd.read_csv(args.data_path, index_col=0)
df = df.sample(frac=1, random_state=0)

# split df into k_fold groups
k_fold_arange = np.linspace(0, len(df), args.k_fold+1).astype(int)

score = []
mae_list = []
for i in range(args.k_fold):
    test = df[k_fold_arange[i]:k_fold_arange[i+1]]
    valid = df[~df.reaction_id.isin(test.reaction_id)].sample(frac=1/(args.k_fold-1), random_state=1)
    train = df[~(df.reaction_id.isin(test.reaction_id) | df.reaction_id.isin(valid.reaction_id))]

    if args.sample:
        try:
            train = train.sample(n=args.sample, random_state=1)
        except Exception:
            pass

    print(len(test), len(valid), len(train))

    train_rxn_id = train['reaction_id'].values
    train_smiles = train.smiles.str.split('>', expand=True)[0].values
    train_core = train.reaction_core.values
    train_activation = train.activation_energy.values

    valid_rxn_id = valid['reaction_id'].values
    valid_smiles = valid.smiles.str.split('>', expand=True)[0].values
    valid_core = valid.reaction_core.values
    valid_activation = valid.activation_energy.values

    if args.model == 'ml_QM_GNN':
        train_reactants = reaction_to_reactants(train['smiles'].tolist())
        qmdf_temp, _ = min_max_normalize(qmdf.copy(), train_smiles=train_reactants)
        initialize_qm_descriptors(df=qmdf_temp)

    train_gen = dataloader(train_smiles, train_core, train_rxn_id, train_activation, args.selec_batch_size,
                           args.select_descriptors)
    train_steps = np.ceil(len(train_smiles) / args.selec_batch_size).astype(int)

    valid_gen = dataloader(valid_smiles, valid_core, valid_rxn_id, valid_activation, args.selec_batch_size,
                           args.select_descriptors)
    valid_steps = np.ceil(len(valid_smiles) / args.selec_batch_size).astype(int)

    model = regressor(args.feature, args.depth, args.select_descriptors)
    opt = tf.keras.optimizers.Adam(lr=args.ini_lr, clipnorm=5)
    model.compile(
        optimizer=opt,
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.RootMeanSquaredError(
            name='root_mean_squared_error', dtype=None), tf.keras.metrics.MeanAbsoluteError(
            name='mean_absolute_error', dtype=None), ]
    )

    save_name = os.path.join(args.model_dir, 'best_model_{}.hdf5'.format(i))
    checkpoint = ModelCheckpoint(save_name, monitor='val_loss', save_best_only=True, save_weights_only=True)
    reduce_lr = LearningRateScheduler(lr_multiply_ratio(args.ini_lr, args.lr_ratio), verbose=1)

    callbacks = [checkpoint, reduce_lr]

    print('training the {}th iteration'.format(i))
    hist = model.fit_generator(
        train_gen, steps_per_epoch=train_steps, epochs=args.selec_epochs,
        validation_data=valid_gen, validation_steps=valid_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=args.workers,
    )

    with open(os.path.join(args.model_dir, 'history_{}.pickle'.format(i)), 'wb') as hist_pickle:
        pickle.dump(hist.history, hist_pickle)

    model.load_weights(save_name)

    test_rxn_id = test['reaction_id'].values
    test_smiles = test.smiles.str.split('>', expand=True)[0].values
    test_core = test.reaction_core.values
    test_activation = test.activation_energy.values

    test_gen = dataloader(test_smiles, test_core, test_rxn_id, test_activation, args.selec_batch_size,
                          args.select_descriptors, shuffle=False)
    test_steps = np.ceil(len(test_smiles) / args.selec_batch_size).astype(int)

    predicted = []
    mse = 0
    mae = 0
    for x, y in tqdm(test_gen, total=int(len(test_smiles) / args.selec_batch_size)):
        out = model.predict_on_batch(x)
        out = np.reshape(out, [-1])
        for y_predicted, y_true in zip(out, y):
            predicted.append(y_predicted)
            mae += abs(y_predicted - y_true)/int(len(test_smiles))
            mse += (y_predicted - y_true)**2/int(len(test_smiles))

    rmse = np.sqrt(mse)
    test_predicted = pd.DataFrame({'rxn_id': test_rxn_id, 'predicted': predicted})
    test_predicted.to_csv(os.path.join(args.model_dir, 'test_predicted_{}.csv'.format(i)))

    score.append(rmse)
    mae_list.append(mae)
    print('success rate for iter {}: {}, {}'.format(i, rmse, mae))

print('RMSE for {}-fold cross-validation: {}'.format(args.k_fold, np.mean(np.array(score))))
print('MAE for {}-fold cross-validation: {}'.format(args.k_fold, np.mean(np.array(mae_list))))



