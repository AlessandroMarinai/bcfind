import tensorflow as tf

from layers import EncoderBlock, DecoderBlock, GradientReversalLayer

import wandb
class UNetSkipHead_A(tf.keras.Model):
    """Class for 3D UNet model.


    Refers to:
        - O. Ronneberger et al. 'UNet: Convolutional Networks for Biomedical Image Segmenation <https://arxiv.org/pdf/1505.04597.pdf>'
        
    It also implements Domain Adaptation through a reversal gradient block put on the top of the unet.
    """

    def __init__(
        self,
        n_blocks,
        n_filters,
        k_size,
        k_stride,
        lambda_da = 0.01,
        dropout=None,
        regularizer=None,
        **kwargs
    ):
        """Constructor method.

        Parameters
        ----------
        n_blocks : int
            depth of the UNet encoder
        n_filters : int
            number of filters for the first layer. Consecutive layers increase esponentially their number of filters.
        k_size : int or tuple of ints
            size of the kernel for convolutional layers
        k_stride : int or tuple of ints
            stride for the convolutional layers. The last two encoding and the first two decoding layers will however have no stride.
        dropout : bool, optional
            whether or not to add dropout layer after each convolutional block, by default None.
        regularizer : string or tf.keras.regularizers, optional
            a regularization method for keras layers, by default None.
        """
        super(UNetSkipHead_A, self).__init__(**kwargs)
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.k_size = k_size
        self.k_stride = k_stride
        self.dropout = dropout
        self.regularizer = regularizer
        self.lambda_da = lambda_da
        #self.input_spec_first_mlp = tf.keras.layers.InputSpec(shape=(20* 30* 30* 256)) 
        self.csv_path = "/home/amarinai/DeepLearningThesis/losses.csv"
        #Binary Cross Entropy Loss
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.bce_target = tf.keras.losses.BinaryCrossentropy()

        # Input channel expansion
        self.conv_block_1 = EncoderBlock(
            n_filters=self.n_filters,
            k_size=self.k_size,
            k_stride=(1, 1, 1),
            regularizer=self.regularizer,
            normalization="batch",
            activation="relu",
        )


        # Encoder
        self.encoder_blocks = []
        for i in range(self.n_blocks):
            if i >= self.n_blocks - 2:  # last two blocks have no stride
                encoder_block = EncoderBlock(
                    n_filters=self.n_filters * (2 ** (i + 1)),
                    k_size=self.k_size,
                    k_stride=(1, 1, 1),
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )
            else:
                encoder_block = EncoderBlock(
                    n_filters=self.n_filters * (2 ** (i + 1)),
                    k_size=self.k_size,
                    k_stride=self.k_stride,
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )

            self.encoder_blocks.append(encoder_block)

        # Decoder
        self.decoder_blocks = []
        for i in range(self.n_blocks):
            if i < 2:  # first two blocks have no stride
                decoder_block = DecoderBlock(
                    n_filters=self.n_filters * (2 ** (self.n_blocks - i - 1)),
                    k_size=self.k_size,
                    k_stride=(1, 1, 1),
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )
            else:
                decoder_block = DecoderBlock(
                    n_filters=self.n_filters * (2 ** (self.n_blocks - i - 1)),
                    k_size=self.k_size,
                    k_stride=self.k_stride,
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )

            self.decoder_blocks.append(decoder_block)

        # Maybe dropout layers
        if dropout:
            self.dropouts = []
            for i in range(self.n_blocks * 2 - 1):
                if i == 0:
                    drp = tf.keras.layers.SpatialDropout3D(dropout / 2)
                    self.dropouts.append(drp)
                else:
                    drp = tf.keras.layers.SpatialDropout3D(dropout)
                    self.dropouts.append(drp)

        # Last predictor layer
        self.predictor = DecoderBlock(
            n_filters=1,
            k_size=self.k_size,
            k_stride=(1, 1, 1),
            regularizer=None,
            normalization="batch",
            activation="linear",
        )

        #Domain Classifier
        
        self.domain_classifier = tf.keras.Sequential([
            GradientReversalLayer(self.lambda_da, name = "gradient_reversal"),
            DecoderBlock(
                n_filters=1,
                k_size=self.k_size,
                k_stride=(1, 1, 1),
                regularizer=None,
                normalization="batch",
                activation="linear",
            ),
            tf.keras.layers.AveragePooling3D(
                pool_size=(3, 3, 3),
                padding='same',
            ),
            tf.keras.layers.AveragePooling3D(
                pool_size=(3, 3, 3),
                padding='same',
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax') 
            ], name = "domain_classifier")
            #Maybe need to add Global Average Pooling to reduce memory consumption
        """
                    
            tf.keras.layers.AveragePooling3D(
                pool_size=(3, 3, 3),
                padding='same',
            ),
            tf.keras.layers.AveragePooling3D(
                pool_size=(3, 3, 3),
                padding='same',
            ),
            
        """ 

        

        
    def call(self, inputs, training=True):
        h0 = self.conv_block_1(inputs)

        encodings = []
        for i_e, encoder_block in enumerate(self.encoder_blocks):
            if i_e == 0:
                h = encoder_block(h0, training=training)
            else:
                h = encoder_block(h, training=training)
            if self.dropout:
                h = self.dropouts[i_e](h, training=training)

            encodings.append(h)

        for i_d, decoder_block in enumerate(self.decoder_blocks):
            if i_d == 0:
                h = decoder_block(encodings[-1], encodings[-2], training=training)  #skip connections are the second argument
            elif i_d < self.n_blocks - 1:
                h = decoder_block(h, encodings[-i_d - 2], training=training)
            elif i_d == self.n_blocks - 1:
                h = decoder_block(h, h0, training=training)
            if self.dropout:
                h = self.dropouts[i_e + i_d](h, training=training)

        pred = self.predictor(h, training=training)
        if training == True:
            domain_pred = self.domain_classifier(h, training=training)  #it is taking the dropouted version here
            return pred, domain_pred
        else:
            return pred, [1,1]

    def get_config(
        self,
    ):
        config = super(UNetSkipHead_A, self).get_config()
        config.update(
            {
                "n_blocks": self.n_blocks,
                "n_filters": self.n_filters,
                "k_size": self.k_size,
                "k_stride": self.k_stride,
                "dropout": self.dropout,
                "regularizer": self.regularizer,
            }
        )
        return config

"""
    #adds to the costum loss defined in compile() the binomial cross entropy loss on the predicted domains
    def compute_loss(self,  y_true=None, y_pred=None, y_domain_source=None, y_domain_target=None):
        lambda_ce = 0.9
        #loss = lambda_ce*self.compiled_loss(y_true, y_pred)
        loss = lambda_ce*self.compiled_loss(y_true, y_true)
        #wandb.log({'comp_loss': loss.numpy()})
        y_true_domain = [1, 0]  #source domain maybe the opposite
        y_true_domain = tf.expand_dims(y_true_domain, axis=0)
        #loss = loss + (1-lambda_ce) * self.bce(y_true=y_true_domain, y_pred=y_domain_source)/2
        #loss = loss + (1-lambda_ce) * self.bce(y_true=y_true_domain, y_pred=y_true_domain)/2
        y_true_domain_1 = [0, 1]  #source domain maybe the opposite
        y_true_domain_1 = tf.expand_dims(y_true_domain_1, axis=0)
        #loss = loss + (1-lambda_ce) * self.bce(y_true=y_true_domain_1, y_pred=y_domain_target)/2
        
        with self.train_writer.as_default(step=self._train_counter):
            tf.summary.scalar('loss', loss)
            wandb.log({'loss': loss})
            y_true_domain = [1, 0]  #source domain maybe the opposite
            y_true_domain = tf.expand_dims(y_true_domain, axis=0)
            tf.summary.scalar("bce",self.bce(y_true=y_true_domain, y_pred=y_domain_source))
            wandb.log({'bce': self.bce(y_true=y_true_domain, y_pred=y_domain_source)})
            y_true_domain_1 = [0, 1]  #source domain maybe the opposite
            y_true_domain_1 = tf.expand_dims(y_true_domain_1, axis=0)
            tf.summary.scalar("bce1",self.bce(y_true=y_true_domain_1, y_pred=y_domain_target))
            wandb.log({'bce1': self.bce(y_true=y_true_domain_1, y_pred=y_domain_target)})
        
        return loss


        

    def train_step(self, data):
        #need to define the logic of the loss you can sum up more than one loss
        # base my implementation on moe_unet.py and https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        source, x_target = data
        x, y = source
        with tf.GradientTape(persistent=True) as tape:
            y_pred, y_domain_source = self(x, training=True)
            comp_loss = self.compiled_loss(y_pred, y)
            y_true_domain = [1.0, 0.0]  #source domain maybe the opposite
            y_true_domain = tf.expand_dims(y_true_domain, axis=0)
            bce_loss = self.bce(y_domain_source, y_true_domain)


        with tf.GradientTape(persistent=True) as tape_target:
            _ , y_domain_target = self(x_target, training=True)
            y_true_domain_target = [0.0, 1.0]  #source domain maybe the opposite
            y_true_domain_target = tf.expand_dims(y_true_domain_target, axis=0)
            bce_loss_target = self.bce_target(y_domain_target, y_true_domain_target)

        total_loss = comp_loss + bce_loss + bce_loss_target
        grads = tape.gradient([comp_loss, bce_loss], self.trainable_weights)
        grads = grads + tape_target.gradient(bce_loss_target, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        #self.compiled_metrics.update_state(y, y_pred)
        
        for metric in self.compiled_metrics._metrics:
            if metric.name == "bce_source" or metric.name == "acc_domain_source":
                y_true_domain = [1.0, 0.0] 
                y_true_domain = tf.expand_dims(y_true_domain, axis=0)
                metric.update_state(y_true_domain, y_domain_source)
            elif metric.name == "bce_target" or metric.name == "acc_domain_target":
                y_true_domain = [0.0, 1.0] 
                y_true_domain = tf.expand_dims(y_true_domain, axis=0)
                metric.update_state(y_true_domain, y_domain_target)
            else:
                metric.update_state(y, y_pred)
        
        results = {m.name: m.result() for m in self.compiled_metrics._metrics}

        #metric_loss = self.compute_loss(y_true=y, y_pred=y_pred, y_domain_source=y_domain_source, y_domain_target=y_domain_target)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(total_loss)
                results["loss"] = metric.result()
        
        return results
    
    def test_step(self, data):
        source, x_target = data
        x, y = source
        y_pred = self(x, training=True)
        y_pred, y_domain_source = y_pred[0], y_pred[1]
        _ , y_domain_target = self(x_target, training=True)

        # Compute the loss value
        loss = self.compute_loss(y_true=y, y_pred=y_pred, y_domain_source=y_domain_source, y_domain_target=y_domain_target)

        #self.compiled_metrics.update_state(y, y_pred)
        
        for metric in self.compiled_metrics._metrics:
            if metric.name == "bce_source" or metric.name == "acc_domain_source":
                y_true_domain = [1, 0] 
                y_true_domain = tf.expand_dims(y_true_domain, axis=0)
                metric.update_state( y_true_domain, y_domain_source)
            elif metric.name == "bce_target" or metric.name == "acc_domain_target":
                y_true_domain = [0, 1] 
                y_true_domain = tf.expand_dims(y_true_domain, axis=0)
                metric.update_state(y_true_domain, y_domain_target)
            else:
                metric.update_state(y, y_pred)
        
        results = {m.name: m.result() for m in self.compiled_metrics._metrics}

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss) #TODO maybe you can pass the loss by using a metric
                results["loss"] = metric.result()
        return results
"""
        


if __name__ == "__main__":
    unet = UNetSkipHead(4, 32, 3, 2)
    unet.build((None, None, None, None, 1))
    unet.summary()

    x = tf.random.normal((4, 48, 48, 48, 1))
    pred = unet(x, training=False)
    print(pred.shape)

    unet.save("prova.tf")
    del unet

    unet = tf.keras.models.load_model("prova.tf")
    unet.build((None, None, None, None, 1))
    unet.summary()

    x = tf.random.normal((4, 48, 100, 100, 1))
    pred = unet(x, training=False)
    print(pred.shape)
