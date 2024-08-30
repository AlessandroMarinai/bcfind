import tensorflow as tf

class MixedLoss(tf.keras.losses.Loss):
    def __init__(self, seg_loss_fn, domain_loss_weight=1.0):
        super(MixedLoss, self).__init__()
        self.seg_loss_fn = seg_loss_fn
        self.domain_loss_weight = domain_loss_weight
        self.domain_loss = tf.keras.losses.BinaryCrossentropy()

    
    # must have the signature (y_true, y_pred) with dimensions (batch_size, d0, .. dN) so put in d0 the source and target
    # check https://stackoverflow.com/questions/65985168/custom-loss-function-with-multiple-targets
    # their idea is to pass it as a dict

    def call(self, y_true, y_pred, y_true_domain, y_pred_domain):
        seg_loss = self.seg_loss_fn(y_true, y_pred)  
        domain_loss = self.domain_loss(y_true_domain, y_pred_domain)  
        mixed_loss = seg_loss + self.domain_loss_weight * domain_loss
        return mixed_loss
