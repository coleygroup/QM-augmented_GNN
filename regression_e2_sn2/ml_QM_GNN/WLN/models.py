import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from .layers import WLN_Layer, Global_Attention
import sys
import numpy as np
import time

np.set_printoptions(threshold=np.inf)


class WLNRegressor(tf.keras.Model):

    def __init__(self, hidden_size, depth, selected_descriptors, max_nb=10):
        super(WLNRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.reactants_WLN = WLN_Layer(hidden_size, depth, max_nb)
        self.selected_descriptors = selected_descriptors

        if "only_bonds" in self.selected_descriptors:
            self.reaction_score0 = layers.Dense(hidden_size, activation=K.relu,
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                use_bias=False)
            self.attention = Global_Attention(hidden_size)
        else:
            self.reaction_score0 = layers.Dense(hidden_size + len(self.selected_descriptors) * 50, activation=K.relu,
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                use_bias=False)
            self.attention = Global_Attention(hidden_size + len(self.selected_descriptors) * 50)

        self.reaction_score = layers.Dense(1, kernel_initializer=tf.random_normal_initializer(stddev=0.1))

        self.node_reshape = layers.Reshape((-1, 1))
        self.core_reshape = layers.Reshape((-1, 1))

    def call(self, inputs):
        res_inputs = inputs[:8]

        res_atom_mask = res_inputs[-3]

        res_core_mask = res_inputs[-1]

        fatom_qm = inputs[-1]

        res_atom_hidden = self.reactants_WLN(res_inputs)
        if "only_bonds" not in self.selected_descriptors:
            res_atom_hidden = K.concatenate([res_atom_hidden, fatom_qm], axis=-1)
        res_atom_mask = self.node_reshape(res_atom_mask)
        res_core_mask = self.core_reshape(res_core_mask)
        res_att_context, _ = self.attention(res_atom_hidden, res_inputs[-2])
        res_atom_hidden = res_atom_hidden + res_att_context
        res_atom_hidden = self.reaction_score0(res_atom_hidden)
        res_mol_hidden = K.sum(res_atom_hidden * res_atom_mask * res_core_mask, axis=-2)
        reaction_score = self.reaction_score(res_mol_hidden)

        return reaction_score
