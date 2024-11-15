import os
import lmdb
import pickle
import logging
import numpy as np
import tensorflow as tf
import concurrent.futures as cf
import random

from utils.data import get_input_tf
from .artificial_targets import get_target_tf
from .augmentation import random_crop_tf, augment
from config_manager import TrainConfiguration, TrainDAConfiguration

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TrainingDataset_FDA(tf.keras.utils.Sequence):
    """Training data flow for 3D gray images with cell locations as targets.

    Args:
        tiff_list (list):

        marker_list (list):

        batch_size (list, tuple):

        output_shape (list, tuple):

        dim_resolution (float, list, tuple):

        augmentations (optional, list, dict):

        augmentations_prob (optional, float, list):

    Returns:
        tensorflow.datasets.Data: a tensorflow dataset.

    Yields:
        tensorflow.Tensor: batch of inputs and targets.
    """

    @staticmethod
    @tf.function
    def parse_imgs(tiff_path, marker_path, dim_resolution, preprocess_kwargs):
        logger.info(f"loading {tiff_path}")
        input_image = get_input_tf(tiff_path, **preprocess_kwargs)

        logger.info(f"creating blobs from {marker_path}")
        blobs = get_target_tf(marker_path, tf.shape(input_image), dim_resolution) #TODO set default radius

        xy = tf.concat(
            [tf.expand_dims(input_image, 0), tf.expand_dims(blobs, 0)], axis=0
        )
        return tf.ensure_shape(xy, (2, None, None, None))
    
    
    @staticmethod
    @tf.function
    def parse_imgs_target(tiff_path, preprocess_kwargs):
        logger.info(f"loading {tiff_path}")


        input_image = get_input_tf(tiff_path, **preprocess_kwargs)
        input_image = tf.expand_dims(input_image, 0)
        return tf.ensure_shape(input_image, (1, None, None, None))
    

    @staticmethod
    @tf.function
    def apply_fda_transformation_tf(src_img_y, trg_dataset, L=0.2, lmbda=1):
        """
        L = random.uniform(0.1, 0.3)
        lmbda = random.uniform(0.2, 0.8)
        """
        print(L, lmbda)
        trg_img = next(iter(trg_dataset))[0]

        src_img = src_img_y[0, ...]

        print(trg_img.shape)
        print(src_img.shape)

        fft_src = tf.signal.fft3d(tf.cast(src_img, tf.complex64))
        fft_trg = tf.signal.fft3d(tf.cast(trg_img, tf.complex64))

        # Get amplitude and phase of source and target
        amp_src, pha_src = tf.abs(fft_src), tf.math.angle(fft_src)
        amp_trg, _ = tf.abs(fft_trg), tf.math.angle(fft_trg)

        amp_src = tf.signal.fftshift(amp_src, axes=(0, 1, 2))
        amp_trg = tf.signal.fftshift(amp_trg, axes=(0, 1, 2))

        d, h, w = amp_src.shape
        if d is None:
            return None #src_img_y
        b = (np.floor(np.amin((h,w,d))*L)).astype(int)
        print(b,d,h,w)
        c_h = np.floor(h/2.0).astype(int)
        c_w = np.floor(w/2.0).astype(int)
        c_z = np.floor(d/2.0).astype(int)
        h1, h2 = c_h - b, c_h + b + 1
        w1, w2 = c_w - b, c_w + b + 1
        d1, d2 = c_z - b, c_z + b + 1

        amp_src_slice = amp_src[d1:d2, h1:h2, w1:w2]
        amp_trg_slice = amp_trg[d1:d2, h1:h2, w1:w2]

        # Use tf.slice to extract the slices
        amp_src_slice = tf.slice(amp_src, begin=[d1, h1, w1], size=[d2-d1, h2-h1, w2-w1])
        amp_trg_slice = tf.slice(amp_trg, begin=[d1, h1, w1], size=[d2-d1, h2-h1, w2-w1])

        print(amp_src_slice.shape)

        # Then perform the FDA transformation on the slices
        new_amp_src_slice = lmbda * amp_trg_slice + (1 - lmbda) * amp_src_slice

        amp_src_updated = tf.concat([
            amp_src[:d1],  # Part before the slice along depth
            tf.concat([
                amp_src[d1:d2, :h1],  # Left part before the height slice
                tf.concat([
                    amp_src[d1:d2, h1:h2, :w1],  # Part before the width slice
                    new_amp_src_slice,  # The new slice to be inserted
                    amp_src[d1:d2, h1:h2, w2:]  # Part after the width slice
                ], axis=2),
                amp_src[d1:d2, h2:]  # Part after the height slice
            ], axis=1),
            amp_src[d2:]  # Part after the slice along depth
        ], axis=0)

        amp_src_ = tf.signal.ifftshift(amp_src_updated, axes=(0, 1, 2))
        print(amp_src_.shape)
    
        # mutated fft of source
        fft_src_ = tf.complex(amp_src_, tf.zeros_like(amp_src_)) * tf.exp(tf.complex(tf.zeros_like(pha_src), pha_src))
        print(fft_src_.shape)
        # Perform inverse FFT to get the transformed source image
        src_in_trg = tf.signal.ifft3d(fft_src_)
        src_in_trg = tf.expand_dims(src_in_trg, axis=0)
        src_in_trg = tf.expand_dims(src_in_trg, axis=-1)

        gt = tf.expand_dims(src_img_y[1,...], axis=0)
        gt = tf.expand_dims(gt, axis=-1)

    
        print(src_in_trg.shape)
        ret = tf.concat([tf.math.real(src_in_trg), gt], axis=0)

        print("finalll")
        print(ret.shape)
        
        return ret
        
    
    #new is a __init__ but static
    def __new__(
        cls,
        tiff_list,
        marker_list,
        tiff_list_target,
        batch_size,
        #preprocess_kwargs_target,
        dim_resolution=1.0,
        output_shape=None,
        augmentations=None,
        augmentations_prob=0.5,
        use_lmdb_data=False,
        n_workers=10,
        **preprocess_kwargs,

    ):
        #tf.config.run_functions_eagerly(True)

        if not use_lmdb_data:
            data = tf.data.Dataset.from_tensor_slices((tiff_list, marker_list))
            
            # load images and targets from paths
            data = data.map(
                lambda x, y: cls.parse_imgs(x, y, dim_resolution, preprocess_kwargs),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
            
            data = data.cache().shuffle(len(marker_list), reshuffle_each_iteration=True)
            

                        # random crop inputs and targets
            if output_shape is not None:
                data = data.map(
                    lambda xy: random_crop_tf(xy, output_shape),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
            
            
            #repeat for target data
            target_domain_data = tf.data.Dataset.from_tensor_slices((tiff_list_target))
             # load images and targets from paths
            
            target_domain_data = target_domain_data.map(
                lambda x: cls.parse_imgs_target(x, preprocess_kwargs),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )

            # cache data after time consuming map and make it shuffle every epoch
            target_domain_data = target_domain_data.cache().shuffle(len(tiff_list_target), reshuffle_each_iteration=True)
            
            # random crop inputs and targets
            if output_shape is not None:
                target_domain_data = target_domain_data.map(
                    lambda x: random_crop_tf(x, output_shape),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
            
            L = random.uniform(0.1, 0.3)
            lmbda = random.uniform(0.2, 1.0)

            data = data.map(
                lambda xy: cls.apply_fda_transformation_tf(xy, target_domain_data, L=L, lmbda=lmbda),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
            
            print("aaa")
            print(data)
            

            # do augmentations
            if augmentations is not None:
                data = data.map(
                    lambda xy: augment(xy, augmentations, augmentations_prob),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
            """
            data = data.map(
                lambda xy: tf.expand_dims(xy, axis=-1),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
            """

            # unstack xy
            data = data.map(
                lambda xy: tf.unstack(xy, num=2),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )            #target_image_list = list(target_domain_data.as_numpy_iterator())
            
            print(data)
            return data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            return super(TrainingDataset_FDA, cls).__new__(cls)

    def __init__(
        self,
        tiff_list,
        marker_list,
        batch_size,
        dim_resolution=1.0,
        output_shape=None,
        augmentations=None,
        augmentations_prob=0.5,
        use_lmdb_data=False,
        n_workers=10,
        **preprocess_kwargs,
    ):
        self.tiff_list = tiff_list
        self.marker_list = marker_list
        self.batch_size = batch_size
        self.dim_resolution = dim_resolution
        self.output_shape = output_shape
        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.use_lmdb_data = use_lmdb_data
        self.n_workers = n_workers
        self.preprocess_kwargs = preprocess_kwargs
        self.n = len(self.tiff_list)

        if isinstance(dim_resolution, (float, int)):
            dim_resolution = [dim_resolution] * 3

        if use_lmdb_data:
            self.lmdb_path = os.path.join(
                *self.tiff_list[0].split("/")[:-3], "Train_lmdb"
            )

            if not os.path.isdir(self.lmdb_path):
                self.create_lmdb()
            else:
                print(
                    f"Found lmdb data at {self.lmdb_path}. Data will be taken from there"
                )

            # NOTE: hardcoded data shape for lmdb size!
            nbytes = (
                np.prod((160, 480, 480)) * 4
            )  # 4 bytes for float32: 1 byte for uint8
            self.map_size = 2 * self.n * nbytes * 10
            self.lmdb_env = lmdb.open(self.lmdb_path, map_size=self.map_size, max_dbs=2)
            self.inputs = self.lmdb_env.open_db("Inputs".encode())
            self.targets = self.lmdb_env.open_db("Targets".encode())

    def create_lmdb(
        self,
    ):
        print("Creating lmdb data")
        self.lmdb_env = lmdb.open(self.lmdb_path, map_size=self.map_size, max_dbs=2)
        self.inputs = self.lmdb_env.open_db("Inputs".encode())
        self.targets = self.lmdb_env.open_db("Targets".encode())

        with self.lmdb_env.begin(write=True) as txn:
            i = 0
            for tiff_file, marker_file in zip(self.tiff_list, self.marker_list):
                print(f"Writing {i+1}/{len(self.tiff_list)} input-target pair to lmdb")
                i += 1

                x = get_input_tf(tiff_file, **self.preprocess_kwargs)
                y = get_target_tf(marker_file, tf.shape(x), self.dim_resolution)

                fname = tiff_file.split("/")[-1]
                txn.put(key=fname.encode(), value=pickle.dumps(x), db=self.inputs)
                txn.put(key=fname.encode(), value=pickle.dumps(y), db=self.targets)

        print("Closing lmdb")
        self.lmdb_env.close()

    def on_epoch_end(
        self,
    ):
        print("Epoch ended. Shuffling files.")
        tif_mark = list(zip(self.tiff_list, self.marker_list))
        np.random.shuffle(tif_mark)
        self.tiff_list, self.marker_list = zip(*tif_mark)

    def __len__(
        self,
    ):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, idx):   # THIS IS THE PART WHERE YOU SHOULD RETURN ALL TOGHETER THE DATA MAYBE JUST RETURN 3 THINGS
        # THE X1, THE MARKERS, THE DOMAIN, THE X2, THE DOMAIN
        # MAYBE IT IS JUST X1, MARKERS, X2  
        files = self.tiff_list[idx * self.batch_size : (idx + 1) * self.batch_size]
        fnames = [f.split("/")[-1] for f in files]

        with self.lmdb_env.begin() as txn:
            x_batch, y_batch = [], []
            for f in fnames:
                x = txn.get(key=f.encode(), db=self.inputs)
                x = pickle.loads(x)

                y = txn.get(key=f.encode(), db=self.targets)
                y = pickle.loads(y)

                xy = tf.concat([x[tf.newaxis, ...], y[tf.newaxis, ...]], axis=0)

                if self.output_shape:
                    xy = random_crop_tf(xy, self.output_shape)
                if self.augmentations:
                    xy = augment(xy, self.augmentations, self.augmentations_prob)

                x, y = tf.unstack(xy)

                x_batch.append(x[tf.newaxis, ..., tf.newaxis])
                y_batch.append(y[tf.newaxis, ..., tf.newaxis])

        return tf.concat(x_batch, axis=0), tf.concat(y_batch, axis=0)

    def getitem(self, i):
        for item in self[i]:
            yield item

    def __iter__(self):
        with cf.ThreadPoolExecutor(self.n_workers) as pool:
            futures = [pool.submit(self.getitem, i) for i in range(len(self))]
            for future in cf.as_completed(futures):
                yield future.result()

def get_inputs_target_paths(input_dir, target_dir):
    fnames = os.listdir(input_dir)
    input_files = sorted([f"{input_dir}/{fname}" for fname in fnames])
    target_files = sorted([f"{target_dir}/{fname}.marker" for fname in fnames])
    return input_files, target_files

def get_inputs_target_paths_no_gt(input_dir):
    fnames = os.listdir(input_dir)
    input_files = sorted([f"{input_dir}/{fname}" for fname in fnames])
    return input_files


if __name__ == "__main__":
    config_file = ""
    conf = TrainDAConfiguration(config_file)

    train_tiff_files, train_marker_files = get_inputs_target_paths(
            conf.source.train_tif_dir, conf.source.train_gt_dir
        )

    train_tiff_files_target = get_inputs_target_paths_no_gt(conf.target.train_tif_dir)

    test_tiff_files, test_marker_files = get_inputs_target_paths(
            conf.source.test_tif_dir, conf.source.test_gt_dir
        )
