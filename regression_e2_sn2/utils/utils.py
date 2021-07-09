import tensorflow as tf
from tensorflow.keras import backend as K


def lr_multiply_ratio(initial_lr, lr_ratio):
    def lr_multiplier(idx):
        return initial_lr*lr_ratio**idx
    return lr_multiplier
