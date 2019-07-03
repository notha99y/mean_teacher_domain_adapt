import keras.backend as K


def weighted_sum_loss(sq_diff_layer, ratio=0.5):
    '''
    Custom loss that takes in the squared difference layer and adds a weighted average with a categorical crossentropy loss
    '''
    def loss(y_true, y_pred):
        return ratio * K.categorical_crossentropy(y_true, y_pred) + (1 - ratio)*sq_diff_layer

    return loss
