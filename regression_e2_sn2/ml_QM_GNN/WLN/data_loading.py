# import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from random import shuffle
from ..graph_utils.mol_graph import get_bond_edits, smiles2graph_pr, pack1D, pack2D, pack2D_withidx, get_mask
from ..graph_utils.ioutils_direct import binary_features_batch


class Graph_DataLoader(Sequence):
    def __init__(self, smiles, reaction_core, rxn_id, activation, batch_size, selected_descriptors, shuffle=True, predict=False):
        self.smiles = smiles
        self.reaction_core = reaction_core
        self.rxn_id = rxn_id
        self.activation = activation
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.atom_classes = {}
        self.predict = predict
        self.selected_descriptors = selected_descriptors

        if self.predict:
            self.shuffle = False

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.smiles) / self.batch_size))

    def __getitem__(self, index):
        smiles_tmp = self.smiles[index * self.batch_size:(index + 1) * self.batch_size]
        reaction_core_tmp = self.reaction_core[index * self.batch_size:(index + 1) * self.batch_size]
        rxn_id_tmp = self.rxn_id[index * self.batch_size:(index + 1) * self.batch_size]
        activation_tmp = self.activation[index * self.batch_size:(index + 1) * self.batch_size]

        if not self.predict:
            x, y = self.__data_generation(smiles_tmp, reaction_core_tmp, rxn_id_tmp, activation_tmp)
            return x, y
        else:
            x = self.__data_generation(smiles_tmp, reaction_core_tmp, rxn_id_tmp, activation_tmp)
            return x

    def on_epoch_end(self):
        if self.shuffle == True:
            zipped = list(zip(self.smiles, self.reaction_core, self.rxn_id, self.activation))
            shuffle(zipped)
            self.smiles, self.reaction_core, self.rxn_id, self.activation = zip(*zipped)

    def __data_generation(self, smiles_tmp, reaction_core_tmp, rxn_id_tmp, activation_tmp):
        prs_extend = []
        activation_extend = []
        rxn_id_extend = []

        for r, p, rxn_id, activation in zip(smiles_tmp, reaction_core_tmp, rxn_id_tmp, activation_tmp):
            rxn_id_extend.extend([rxn_id])
            prs_extend.extend([smiles2graph_pr(r, p, self.selected_descriptors)])
            activation_extend.extend([activation])

        rs_extends, smiles_extend = zip(*prs_extend)

        fatom_list, fatom_qm_list, fbond_list, gatom_list, gbond_list, nb_list, core_mask = \
            zip(*rs_extends)
        res_graph_inputs = (pack2D(fatom_list), pack2D(fbond_list), pack2D_withidx(gatom_list),
                            pack2D_withidx(gbond_list), pack1D(nb_list), get_mask(fatom_list),
                            binary_features_batch(smiles_extend), pack1D(core_mask), pack2D(fatom_qm_list))
        if self.predict:
            return res_graph_inputs
        else:
            return res_graph_inputs, np.array(activation_extend).astype('float')
