import tensorflow as tf

"""
class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, lambda_val):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val

    def call(self, x):
        return GradientReversal(x, self.lambda_val)

@staticmethod
@tf.custom_gradient
def GradientReversal(x, lambda_val):
    y = tf.identity(x)
    def grad(dy):
        return -lambda_val * dy
    return y, grad


class GradientReversalLayer(tf.keras.layers.Layer):
    '''Flip the sign of gradient during training.'''

    @tf.custom_gradient
    def grad_reverse(self, x):
        y = tf.identity(x)
        def custom_grad(dy):
            return -self.hp_lambda * dy
        return y, custom_grad

    # --------------------------------------
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.hp_lambda = tf.Variable(hp_lambda, dtype='float32',     name='hp_lambda')


    # --------------------------------------
    def call(self, x, mask=None):
        return self.grad_reverse(x)

    # --------------------------------------
    def set_hp_lambda(self,hp_lambda):
        #self.hp_lambda = hp_lambda
        tf.set_value(self.hp_lambda, hp_lambda)

    # --------------------------------------
    def increment_hp_lambda_by(self,increment):
        new_value = float(tf.get_value(self.hp_lambda)) +  increment
        tf.set_value(self.hp_lambda, new_value)

# --------------------------------------
    def get_hp_lambda(self):
        return float(tf.get_value(self.hp_lambda))
"""
import tensorflow as tf

class GradientReversalLayer(tf.keras.layers.Layer):
    """Flip the sign of gradient during training.
    based on https://github.com/michetonu/gradient_reversal_keras_tf
    ported to tf 2.x
    """

    def __init__(self, lambda_da=0.01, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lam = lambda_da

    @staticmethod
    @tf.custom_gradient
    def reverse_gradient(x, lam):
        # @tf.custom_gradient suggested by Hoa's comment at
        # https://stackoverflow.com/questions/60234725/how-to-use-gradient-override-map-with-tf-gradienttape-in-tf2-0
        return tf.identity(x), lambda dy: (-dy * lam, None)

    def call(self, x):
        return self.reverse_gradient(x, self.lam)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super(GradientReversalLayer, self).get_config() | {'λ': self.lam}
    
    def set_lambda(self, new_lambda):
        self.lam = new_lambda

    def get_lambda(self):
        return self.lam