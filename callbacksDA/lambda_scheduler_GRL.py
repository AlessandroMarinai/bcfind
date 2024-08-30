import numpy as np
import tensorflow as tf

class LambdaSchedulerGRL(tf.keras.callbacks.Callback):

    def __init__(self, max_epochs, k=10):
        super(LambdaSchedulerGRL, self).__init__()
        self.k = k
        self.max_epochs = max_epochs

    def sigmoid_decay_schedule(self, epoch):
        return (2 / (1 + np.exp(-self.k * epoch / self.max_epochs))) - 1

    def on_epoch_end(self, epoch, logs=None):
        new_lambda = self.sigmoid_decay_schedule(epoch)
        self.model.get_layer('sequential').get_layer('gradient_reversal').set_lambda(-1.0)#new_lambda)
        print("Lambda value at epoch {}: {}".format(epoch, new_lambda))


