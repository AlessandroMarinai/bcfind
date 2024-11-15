import tensorflow as tf

from layers import EncoderBlock, DecoderBlock


class UNetNoSkip(tf.keras.Model):
    """Class for 3D UNet model.


    Refers to:
        - O. Ronneberger et al. 'UNet: Convolutional Networks for Biomedical Image Segmenation <https://arxiv.org/pdf/1505.04597.pdf>'
    """

    def __init__(
        self,
        n_blocks,
        n_filters,
        k_size,
        k_stride,
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
        super(UNetNoSkip, self).__init__(**kwargs)
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.k_size = k_size
        self.k_stride = k_stride
        self.dropout = dropout
        self.regularizer = regularizer

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

    def call(self, inputs, training=None):
        print(inputs)
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
                h = decoder_block(encodings[-1], None, training=training)
            elif i_d < self.n_blocks - 1:
                h = decoder_block(h, None, training=training)
            elif i_d == self.n_blocks - 1:
                h = decoder_block(h, None, training=training)

            if self.dropout:
                h = self.dropouts[i_e + i_d](h, training=training)

        pred = self.predictor(h, training=training)
        return pred

    def get_config(
        self,
    ):
        config = super(UNetNoSkip, self).get_config()
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


if __name__ == "__main__":
    unet = UNetNoSkip(4, 32, 3, 2)
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
