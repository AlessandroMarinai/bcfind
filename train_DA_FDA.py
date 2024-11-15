import os
import lmdb
import json
import pickle
import shutil
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
import yaml
import csv

from numba import cuda
import time

from config_manager import TrainConfiguration, TrainDAConfiguration
from data import TrainingDataset_DA_FDA, TrainingDataset, TrainingDataset_DA_gamma
from losses import FramedCrossentropy3D
from metrics import Precision, Recall, F1
from models import UNetNoSkip_DA, UNet_DA_Pixelwise, UNetSkipHead, UNetSkipHead_2, UNetSkipHead_3, UNetSkipHead_4, ResUNet_DA, ResUNet
from localizers import BlobDoG
from utils.models import predict
from utils.data import get_input_tf, get_gt_as_numpy
from utils.base import evaluate_df
#from callbacksDA import LambdaSchedulerGRL
import warnings
from data.augmentation import gamma_tf
from scheduler import CustomCosineDecayRestarts



class Trainer:
    def __init__(self):
        self.seed = 2408
        self.reduce_loss = True

    def make_unet_data_DA(               
        self,
        train_inputs,
        train_targets,
        train_inputs_target,
        dim_resolution,
        input_shape,
        augmentations,
        augmentations_prob,
        batch_size,
        preprocess_kwargs,
        preprocess_kwargs_target,
        val_inputs=None,
        val_targets=None,
        val_inputs_target=None,
        use_lmdb=False,
        
    ):
        self.unet_train_data = TrainingDataset_DA_FDA(
            tiff_list=train_inputs,
            marker_list=train_targets,
            tiff_list_target=train_inputs_target,
            batch_size=batch_size,
            dim_resolution=dim_resolution,
            output_shape=input_shape,
            augmentations=augmentations,
            augmentations_prob=augmentations_prob,
            use_lmdb_data=use_lmdb,
            preprocess_kwargs = preprocess_kwargs,
            preprocess_kwargs_target = preprocess_kwargs_target,
        )

        if val_inputs and val_targets:
            self.unet_val_data = TrainingDataset_DA_FDA(
                tiff_list=val_inputs,
                marker_list=val_targets,
                tiff_list_target=val_inputs_target,
                batch_size=batch_size,
                dim_resolution=dim_resolution,
                output_shape=input_shape,
                augmentations=augmentations,
                augmentations_prob=augmentations_prob,
                use_lmdb_data=use_lmdb,
                preprocess_kwargs = preprocess_kwargs,
                preprocess_kwargs_target = preprocess_kwargs_target,
            )
        else:
            self.unet_val_data = None

    def build_unet(
        self,
        n_blocks,
        n_filters,
        k_size,
        k_stride,
        dropout=None,
        regularizer=None,
        mult_skip=False,
        model_type="unet_noskip_DA",
        lambda_da=0.01,
        squeeze_factor=None,
        moe_n_experts=None,
        moe_top_k_experts=None,
        moe_noise=None,
        moe_balance_loss=None,
    ):
        if model_type == "unet_noskip_DA":
            self.unet = UNetNoSkip_DA(
                n_blocks=n_blocks,
                n_filters=n_filters,
                k_size=k_size,
                k_stride=k_stride,
                dropout=dropout,
                regularizer=regularizer,
                lambda_da=lambda_da,
            )
            self.pixel_wise = False

        elif model_type == "unet_DA_pixelwise":
            self.unet = UNet_DA_Pixelwise(
                n_blocks=n_blocks,
                n_filters=n_filters,
                k_size=k_size,
                k_stride=k_stride,
                dropout=dropout,
                regularizer=regularizer,
                lambda_da=lambda_da,
            )
            self.pixel_wise = True

        elif model_type == "unet_skip_head_DA":
            self.unet = UNetSkipHead(
                n_blocks=n_blocks,
                n_filters=n_filters,
                k_size=k_size,
                k_stride=k_stride,
                dropout=dropout,
                regularizer=regularizer,
                lambda_da=lambda_da,
            )
            self.pixel_wise = False

        elif model_type == "unet_skip_head_DA_2":
            self.unet = UNetSkipHead_2(
                n_blocks=n_blocks,
                n_filters=n_filters,
                k_size=k_size,
                k_stride=k_stride,
                dropout=dropout,
                regularizer=regularizer,
                lambda_da=lambda_da,
            )
            self.pixel_wise = False

        elif model_type == "unet_skip_head_DA_3":
            self.unet = UNetSkipHead_3(
                n_blocks=n_blocks,
                n_filters=n_filters,
                k_size=k_size,
                k_stride=k_stride,
                dropout=dropout,
                regularizer=regularizer,
                lambda_da=lambda_da,
            )
            self.pixel_wise = False

        elif model_type == "unet_skip_head_DA_4":
            self.unet = UNetSkipHead_4(
                n_blocks=n_blocks,
                n_filters=n_filters,
                k_size=k_size,
                k_stride=k_stride,
                dropout=dropout,
                regularizer=regularizer,
                lambda_da=lambda_da,
            )
            self.pixel_wise = False

        elif model_type == "res_unet_DA":
            self.unet = ResUNet_DA(
                n_blocks=n_blocks,
                n_filters=n_filters,
                k_size=k_size,
                k_stride=k_stride,
                dropout=dropout,
                regularizer=regularizer,
                lambda_da=lambda_da,
            )
            self.pixel_wise = False
        elif model_type == "res_unet":
            self.unet = ResUNet(
                n_blocks=n_blocks,
                n_filters=n_filters,
                k_size=k_size,
                k_stride=k_stride,
                dropout=dropout,
                regularizer=regularizer,
            )
            self.pixel_wise = False

        else:
            raise ValueError(
                f'UNet model must be "unet_noskip_DA" \
                Received {model_type}.'
            )

        #self.unet.build((None, 48, 48, 48, 1)) #define the dimensions
        #self.unet.build((None, 48, 96, 96, 1))
        self.unet.build((None, 80, 120, 120, 1))


    def compile_unet(
        self,
        exclude_border,
        input_shape,
        learning_rate,
        decay_steps=100,
        t_mul=2,
        m_mul=0.8,
        alpha=1e-4,
    ):
        self.comp_loss = FramedCrossentropy3D(
            exclude_border, input_shape, from_logits=True, reduce=self.reduce_loss
        )
        self.bce = tf.keras.losses.BinaryCrossentropy()
        prec = Precision(0.2, input_shape, exclude_border, from_logits=True)
        rec = Recall(0.2, input_shape, exclude_border, from_logits=True)
        f1 = F1(0.2, input_shape, exclude_border, from_logits=True)

        self.lr_schedule = CustomCosineDecayRestarts(
            learning_rate,
            decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
        )
    
        bce_source = tf.keras.metrics.BinaryCrossentropy(name = "bce_source")
        bce_target = tf.keras.metrics.BinaryCrossentropy(name = "bce_target")
        acc_domain_source = tf.keras.metrics.BinaryAccuracy(name="acc_domain_source") #default threshold 0.5
        acc_domain_target = tf.keras.metrics.BinaryAccuracy(name="acc_domain_target")

        self.metrics = [prec, rec, f1, bce_source, bce_target, acc_domain_source, acc_domain_target]
        
        self.optimizer = tf.keras.optimizers.SGD(
            self.lr_schedule, momentum=0.9, nesterov=True, weight_decay=7e-4
        )
        """
        self.optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
        )
        """
        #self.unet.compile(loss=loss, optimizer=optimizer, metrics=[prec, rec, f1, bce_source, bce_target, acc_domain_source, acc_domain_target])    

            

    def fit_unet(
        self, epochs, checkpoint_dir=None, tensorboard_dir=None, test_as_val=False, 
    ):  
        #warnings.filterwarnings("ignore")
        print("Validation?")
        print(test_as_val)
        all_trainable_variables = (
            self.unet.encoder_blocks.trainable_weights +
            self.unet.decoder_blocks.trainable_weights +
            self.unet.predictor.trainable_weights +
            self.unet.domain_classifier.trainable_weights
        )

        self.optimizer.build(all_trainable_variables)

        for epoch in range(epochs):
            print("\nStart of epoch %d/%d" % (epoch+1, epochs))
            losses = []
            comp_losses = []
            bce_losses = []
            for step, data in enumerate(self.unet_train_data):
                #loss, comp_loss, bce_loss = self.train_step_unet(data)
                source, x_target = data
                x, y = source
                batch_size = x.shape[0]
                inputs = tf.concat([x, x_target], axis=0)
                classifier_weight = 1.0 #float(0.1*(epoch/epochs))
                
                with tf.GradientTape(persistent=True) as tape:
                    y_pred, y_domain = self.unet(inputs, training=True)
                    y_pred_source = y_pred[:batch_size, ...]
                    comp_loss = self.comp_loss(y_true=y, y_pred=y_pred_source)
                    gt_domain = get_gt_domain(batch_size, self.pixel_wise)
                    bce_loss = self.bce(y_pred=y_domain, y_true=gt_domain)*classifier_weight

                comp_grads = tape.gradient(comp_loss ,self.unet.encoder_blocks.trainable_weights)
                comp_grads_dec = tape.gradient(comp_loss ,self.unet.decoder_blocks.trainable_weights)
                comp_grads_pred = tape.gradient(comp_loss ,self.unet.predictor.trainable_weights)

                self.optimizer.apply_gradients(zip(comp_grads, self.unet.encoder_blocks.trainable_weights))
                self.optimizer.apply_gradients(zip(comp_grads_dec, self.unet.decoder_blocks.trainable_weights))
                self.optimizer.apply_gradients(zip(comp_grads_pred, self.unet.predictor.trainable_weights))

                comp_grads = tape.gradient(bce_loss ,self.unet.encoder_blocks.trainable_weights)
                comp_grads_dec = tape.gradient(bce_loss ,self.unet.decoder_blocks.trainable_weights)
                comp_grads_pred = tape.gradient(bce_loss ,self.unet.domain_classifier.trainable_weights)

                self.optimizer.apply_gradients(zip(comp_grads, self.unet.encoder_blocks.trainable_weights))
                self.optimizer.apply_gradients(zip(comp_grads_dec, self.unet.decoder_blocks.trainable_weights))
                self.optimizer.apply_gradients(zip(comp_grads_pred, self.unet.domain_classifier.trainable_weights))

                """   
                grads = tape.gradient(loss, self.unet.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.unet.trainable_weights))
                """
                
                """
                all_trainable_variables = (
                    self.unet.encoder_blocks.trainable_weights +
                    self.unet.decoder_blocks.trainable_weights +
                    self.unet.predictor.trainable_weights +
                    self.unet.domain_classifier.trainable_weights
                )

                self.optimizer.build(all_trainable_variables)
                with tf.GradientTape(persistent=True) as tape:
                    y_pred, y_domain = self.unet(inputs, training=True)
                    y_pred_source = y_pred[:batch_size, ...]
                    comp_loss = self.comp_loss(y_true=y, y_pred=y_pred_source)
                comp_grads = tape.gradient(comp_loss ,self.unet.encoder_blocks.trainable_weights)
                comp_grads_dec = tape.gradient(comp_loss ,self.unet.decoder_blocks.trainable_weights)
                comp_grads_pred = tape.gradient(comp_loss ,self.unet.predictor.trainable_weights)

                self.optimizer.apply_gradients(zip(comp_grads, self.unet.encoder_blocks.trainable_weights))
                self.optimizer.apply_gradients(zip(comp_grads_dec, self.unet.decoder_blocks.trainable_weights))
                self.optimizer.apply_gradients(zip(comp_grads_pred, self.unet.predictor.trainable_weights))



                with tf.GradientTape(persistent=True) as tape:
                    y_pred, y_domain = self.unet(inputs, training=True)
                    gt_domain = get_gt_domain(batch_size, self.pixel_wise)
                    bce_loss = self.bce(y_pred=y_domain, y_true=gt_domain)*classifier_weight
                comp_grads = tape.gradient(bce_loss ,self.unet.encoder_blocks.trainable_weights)
                comp_grads_dec = tape.gradient(bce_loss ,self.unet.decoder_blocks.trainable_weights)
                comp_grads_pred = tape.gradient(bce_loss ,self.unet.domain_classifier.trainable_weights)

                self.optimizer.apply_gradients(zip(comp_grads, self.unet.encoder_blocks.trainable_weights))
                self.optimizer.apply_gradients(zip(comp_grads_dec, self.unet.decoder_blocks.trainable_weights))
                self.optimizer.apply_gradients(zip(comp_grads_pred, self.unet.domain_classifier.trainable_weights))
                
                loss = comp_loss + bce_loss
                """
                wandb.log({"lr": self.lr_schedule.current_learning})
                wandb.log({"classifier_weight": classifier_weight})
                loss = comp_loss + bce_loss

                
                y_domain_source = y_domain[:batch_size, ...]
                y_domain_target = y_domain[batch_size:, ...]
                for metric in self.metrics:
                    if metric.name == "bce_source" or metric.name == "acc_domain_source":
                        y_true_domain = gt_domain[:batch_size, ...]
                        metric.update_state(y_true=y_true_domain, y_pred=y_domain_source)
                    elif metric.name == "bce_target" or metric.name == "acc_domain_target":
                        y_true_domain_target = gt_domain[batch_size:, ...]
                        metric.update_state(y_true=y_true_domain_target, y_pred=y_domain_target)
                    else:
                        metric.update_state(y_true=y, y_pred=y_pred_source)
                        losses.append(float(loss))
                        comp_losses.append(float(comp_loss))
                        bce_losses.append(float(bce_loss))
                
                for metric in self.metrics:
                        metric.update_state(y_true=y, y_pred=y_pred_source)
                        losses.append(float(loss))
                        comp_losses.append(float(comp_loss))
                  


            results = {}
            total_loss = comp_loss + bce_loss
            for metric in self.metrics:   
                results[metric.name] = metric.result().numpy()
            results["loss"] = np.mean(losses)
            results["compiled_loss"] = np.mean(comp_losses)
            results["classifier_loss"] = np.mean(bce_losses)
            new_lambda = sigmoid_decay_schedule(epoch=epoch, max_epochs=epochs)
            self.unet.get_layer('domain_classifier').get_layer('gradient_reversal').set_lambda(new_lambda)
            results["lambda"] = self.unet.get_layer('domain_classifier').get_layer('gradient_reversal').get_lambda()
            results["lr"] = self.lr_schedule.current_learning.numpy()
            results["classifier_weight"] = classifier_weight
            print(results)
            wandb.log(results)

            
            #validation results
            if epoch % 50 == 0 and test_as_val:
                print("\nValidation at epoch %d/%d" % (epoch+1, epochs))
                losses = []
                comp_losses = []
                bce_losses = []
                for step, data in enumerate(self.unet_val_data):
                    source, x_target = data
                    x, y = source
                    batch_size = x.shape[0]
                    inputs = tf.concat([x, x_target], axis=0)
                    y_pred, y_domain = self.unet(inputs, training=True)
                    y_pred_source = y_pred[:batch_size, ...]
                    comp_loss = self.comp_loss(y_true=y, y_pred=y_pred_source)
                    gt_domain = get_gt_domain(batch_size, self.pixel_wise)
                    bce_loss = self.bce(y_pred=y_domain, y_true=gt_domain)*0.01*epoch/epochs
                    loss = bce_loss + comp_loss
                    y_domain_source = y_domain[:batch_size, ...]
                    y_domain_target = y_domain[batch_size:, ...]
                    for metric in self.metrics:
                        if metric.name == "bce_source" or metric.name == "acc_domain_source":
                            y_true_domain = gt_domain[:batch_size, ...]
                            metric.update_state(y_true=y_true_domain, y_pred=y_domain_source)
                        elif metric.name == "bce_target" or metric.name == "acc_domain_target":
                            y_true_domain_target = gt_domain[batch_size:, ...]
                            metric.update_state(y_true=y_true_domain_target, y_pred=y_domain_target)
                        else:
                            metric.update_state(y_true=y, y_pred=y_pred_source)
                    losses.append(float(loss))
                    comp_losses.append(float(comp_loss))
                    bce_losses.append(float(bce_loss))
                    

                results = {}
                #total_loss = comp_loss + bce_loss
                for metric in self.metrics:   
                    results[f"val_{metric.name}"] = metric.result().numpy()
                results["val_loss"] = np.mean(losses)
                results["val_compiled_loss"] = np.mean(comp_losses)
                results["val_classifier_loss"] = np.mean(bce_losses)
                print(results)
                wandb.log(results)
                
            #save model
            if epoch % 500 == 0:
                self.unet.save_weights(f"{checkpoint_dir}/saved_model{epoch}/model.tf")
        

        self.unet.save_weights(f"{checkpoint_dir}/final_model/model.tf")
        """
        callbacks = []

        #create directory
        if checkpoint_dir:
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)

            monitor = "loss"
            if test_as_val:
                monitor = "val_loss"

            MC_callback = tf.keras.callbacks.ModelCheckpoint(
                f"{checkpoint_dir}/model.tf",
                initial_value_threshold=2,
                save_best_only=True,
                save_weights_only = True,
                save_format="tf",
                save_freq="epoch",
                monitor=monitor,
                mode="min",
                verbose=1,
            )
            callbacks.append(MC_callback)

        if tensorboard_dir:
            if os.path.isdir(tensorboard_dir):
                shutil.rmtree(tensorboard_dir, ignore_errors=True)
            os.makedirs(tensorboard_dir)

            TB_callback = tf.keras.callbacks.TensorBoard(
                tensorboard_dir,
                # update_freq="epoch",
                profile_batch=0,
                write_graph=True,
                write_images=True,
            )
            callbacks.append(TB_callback)

        lambda_scheduler_callback = LambdaSchedulerGRL(epochs)
        callbacks.append(lambda_scheduler_callback)

        if len(callbacks) == 0:
            callbacks = None
        
        self.unet.fit(
            self.unet_train_data, 
            epochs=epochs,
            callbacks=callbacks,
            validation_data=self.unet_val_data,
            verbose=1,
        )
        """
       
    def make_dog_data_48(
        self, tiff_files, marker_files, data_shape, lmdb_dir, gamma = None, **preprocessing_kwargs
    ):
        n = len(tiff_files)
        nbytes = np.prod(data_shape) * 1  # 4 bytes for float32: 1 byte for uint8
        db = lmdb.open(lmdb_dir, map_size=n * nbytes * 10)

        # UNet predictions
        print(f"Saving U-Net predictions in {lmdb_dir}")
        with db.begin(write=True) as fx:
            for i, tiff_file in enumerate(tiff_files):
                print(f"\nUnet prediction on file {i+1}/{len(tiff_files)}")

                a = get_input_tf(tiff_file, **preprocessing_kwargs)
                for i in range(4):
                    if i == 0:
                        x = a[:48,:48,:48]
                    if i == 1:
                        x = a[:48,48:96,:48]
                    if i == 2:
                        x = a[:48,:48,48:96]
                    if i == 3:
                        x = a[:48,48:96,48:96]                        
                    if gamma is not None:
                        x= gamma_tf(x, gamma)
                        print(f"gamma con {gamma}")
                    
                    pred = predict(x, self.unet).numpy()
                    pred = tf.sigmoid(tf.squeeze(pred)).numpy()
                    pred = (pred * 255).astype("uint8")

                    fname = tiff_file.split("/")[-1]
                    print(i)
                    fx.put(key=(f"{fname}_{i}".encode()), value=pickle.dumps(pred))

        db.close()
        dog_inputs = lmdb.open(lmdb_dir, readonly=True)
        
        # True cell coordinates
        dog_targets = []
        for marker_file in sorted(marker_files):
            print(f"Loading file {marker_file}")
            y = get_gt_as_numpy(marker_file)
            for i in range(4):
                if i == 0:
                    coords = y[(y[:,0] <48) & 
                            (y[:,1] <48) & 
                            (y[:,2] <48)]
                if i == 1:
                    coords = y[(y[:,0] <48) & 
                            (y[:,1] >=48 ) & 
                            (y[:,1] <96 ) &
                            (y[:,2] <48)]
                    coords[:,1] -= 48
                if i == 2:
                    coords = y[(y[:,0] <48) & 
                            (y[:,1] <48 ) & 
                            (y[:,2] >=48 ) &
                            (y[:,2] <96)]
                    coords[:,2] -= 48

                if i == 3:
                    coords = y[(y[:,0] <48) & 
                            (y[:,1] >=48 ) & 
                            (y[:,1] <96 ) & 
                            (y[:,2] >=48 ) &
                            (y[:,2] <96)]
                    coords[:,2] -= 48
                    coords[:,1] -= 48

                
                dog_targets.append(coords)
            print(len(dog_targets))
                

            #dog_targets.append(y)
        
        return dog_inputs, dog_targets

    def make_dog_data(
        self, tiff_files, marker_files, data_shape, lmdb_dir, gamma = None, **preprocessing_kwargs
    ):
        n = len(tiff_files)
        nbytes = np.prod(data_shape) * 1  # 4 bytes for float32: 1 byte for uint8
        db = lmdb.open(lmdb_dir, map_size=n * nbytes * 10)

        # UNet predictions
        print(f"Saving U-Net predictions in {lmdb_dir}")
        with db.begin(write=True) as fx:
            for i, tiff_file in enumerate(tiff_files):
                print(f"\nUnet prediction on file {i+1}/{len(tiff_files)}")

                x = get_input_tf(tiff_file, **preprocessing_kwargs)
                if gamma is not None:
                     x= gamma_tf(x, gamma)
                     print(f"gamma con {gamma}")
                
                pred = predict(x, self.unet).numpy()
                pred = tf.sigmoid(tf.squeeze(pred)).numpy()
                pred = (pred * 255).astype("uint8")

                fname = tiff_file.split("/")[-1]
                fx.put(key=fname.encode(), value=pickle.dumps(pred))

        db.close()
        dog_inputs = lmdb.open(lmdb_dir, readonly=True)
        
        # True cell coordinates
        dog_targets = []
        for marker_file in sorted(marker_files):
            print(f"Loading file {marker_file}")
            y = get_gt_as_numpy(marker_file)
            dog_targets.append(y)

        return dog_inputs, dog_targets

    def fit_dog(
        self,
        inputs_lmdb,
        targets_list,
        iterations,
        dim_resolution,
        exclude_border,
        max_match_dist,
        checkpoint_dir=None,
        n_cpu=1,
    ):
        cuda.close() #TODO check this

        if checkpoint_dir:
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)

        self.dog = BlobDoG(3, dim_resolution, exclude_border)
        with inputs_lmdb.begin() as fx:
            X = fx.cursor()
            self.dog.fit(
                X=X,
                Y=targets_list,
                max_match_dist=max_match_dist,
                n_iter=iterations,
                checkpoint_dir=checkpoint_dir,
                n_cpu=n_cpu,
            )

        inputs_lmdb.close()
        return self.dog

    def run(
        self, config_file, only_dog=False, test_as_val=None, use_lmdb=False, gpu=-1, checkpoint=False,
    ):
        
        gpus = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[gpu], "GPU")
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        tf.config.experimental.set_memory_growth(gpus[gpu], True)  #non occupa tutta la memoria subito
        
        conf = TrainDAConfiguration(config_file)

        

        with open(config_file, 'r') as f:
            data = yaml.load(f, Loader=yaml.BaseLoader)
        wandb.login()
        wandb.init(project='my-tf-integration', name=conf.exp.name, config=data)


            

        train_tiff_files, train_marker_files = get_inputs_target_paths(
            conf.source.train_tif_dir, conf.source.train_gt_dir
        )

        test_tiff_files, test_marker_files = get_inputs_target_paths(
            conf.source.test_tif_dir, conf.source.test_gt_dir
        )

        train_tiff_files_target = get_inputs_target_paths_no_gt(conf.target.train_tif_dir)

        test_tiff_files_target = get_inputs_target_paths_no_gt(conf.target.test_tif_dir)



        if test_as_val:
            print("ATTN!! Using part of the test-set as validation")
            nt = len(test_tiff_files)
            np.random.seed(self.seed)
            #val_idx = np.random.randint(0, nt, size=nt // 3)
            val_idx = np.random.choice(nt, size=nt // 3, replace=False)
            val_tiff_files = [test_tiff_files[i] for i in val_idx]
            val_marker_files = [test_marker_files[i] for i in val_idx]
            for vtf, vmf in zip(val_tiff_files, val_marker_files):
                test_tiff_files.remove(vtf)
                test_marker_files.remove(vmf)

            nt = len(test_tiff_files_target)
            val_idx = np.random.choice(nt, size=nt // 3, replace=False)
            val_tiff_files_target = [test_tiff_files_target[i] for i in val_idx]
            for vtf in val_tiff_files_target:
                test_tiff_files_target.remove(vtf)

        if not only_dog:
            ######################################
            ############ CREATE DIRS #############
            ######################################
            # Create experiment directory and copy the config file there
            if not os.path.isdir(conf.exp.basepath):
                os.makedirs(conf.exp.basepath, exist_ok=True)
                config_name = config_file.split("/")[-1]
                shutil.copyfile(config_file, f"{conf.exp.basepath}/{config_name}")

            ####################################
            ############ UNET DATA #############
            ####################################
            print("\nLOADING UNET DATA")
            if test_as_val:
                self.make_unet_data_DA(
                    train_inputs=train_tiff_files,
                    train_targets=train_marker_files,
                    train_inputs_target = train_tiff_files_target,
                    dim_resolution=conf.source.dim_resolution,
                    input_shape=conf.unet.input_shape,
                    augmentations=conf.data_aug.op_args,
                    augmentations_prob=conf.data_aug.op_probs,
                    batch_size=conf.unet.batch_size,
                    val_inputs=val_tiff_files,
                    val_targets=val_marker_files,
                    val_inputs_target=val_tiff_files_target,
                    use_lmdb=use_lmdb,
                    preprocess_kwargs=conf.preproc,
                    preprocess_kwargs_target=conf.preproct
                )
            else:
                self.make_unet_data_DA(
                    train_inputs=train_tiff_files,
                    train_targets=train_marker_files,
                    train_inputs_target = train_tiff_files_target,
                    dim_resolution=conf.source.dim_resolution,
                    input_shape=conf.unet.input_shape,
                    augmentations=conf.data_aug.op_args,
                    augmentations_prob=conf.data_aug.op_probs,
                    batch_size= conf.unet.batch_size,
                    use_lmdb=use_lmdb,
                    preprocess_kwargs=conf.preproc,
                    preprocess_kwargs_target=conf.preproct
                )

            ####################################
            ############## UNET ################
            ####################################
            print("\nBUILDING UNET")
            self.build_unet(
                n_blocks=conf.unet.n_blocks,
                n_filters=conf.unet.n_filters,
                k_size=conf.unet.k_size,
                k_stride=conf.unet.k_stride,
                dropout=conf.unet.dropout,
                regularizer=conf.unet.regularizer,
                model_type=conf.unet.model,
                lambda_da=conf.unet.lambda_da,
                squeeze_factor=conf.unet.squeeze_factor,
                moe_n_experts=conf.unet.moe_n_experts,
                moe_top_k_experts=conf.unet.moe_top_k_experts,
                moe_noise=conf.unet.moe_noise,
                moe_balance_loss=conf.unet.moe_balance_loss,
            )

            if checkpoint:
                print("\nLOADING TRAINED UNET")
                #checkpoint_path = f"{conf.unet.checkpoint_dir}/saved_model/model.tf"
                checkpoint_path = "/home/amarinai/DeepLearningThesis/Results/Unet_DA_lowloss/UNet_checkpoints/saved_model2000/model.tf"
                self.unet.load_weights(checkpoint_path).expect_partial()
                print(f"\nCHECKPOINT LOADED from {checkpoint_path}")


            print("\nCOMPILING UNET")
            self.compile_unet(
                exclude_border=conf.unet.exclude_border,
                input_shape=conf.unet.input_shape,
                learning_rate=conf.unet.learning_rate,
            )

            #self.unet.compile(optimizer=self.optimizer)

            #retake run
            """
            checkpoint_lr = f"{conf.unet.checkpoint_dir}/status/optimizer.json"
            self.lr_schedule = CustomCosineDecayRestarts.load(checkpoint_lr)
            self.optimizer = tf.keras.optimizers.SGD(
                self.lr_schedule, momentum=0.9, nesterov=True, weight_decay=7e-4
            )
            checkpoint_unet = f"{conf.unet.checkpoint_dir}/final_model/model.tf"
            print(checkpoint_unet)
            #/home/amarinai/DeepLearningThesis/Results/Unet_DA_3/UNet_checkpoints/final_model
            status = self.unet.load_weights(checkpoint_unet)
            print(status)
            
            self.unet.load_weights(f"{conf.unet.checkpoint_dir}/final/model.tf")
            """
            self.unet.summary()
            print("\nTRAINING UNET")
            
            
            self.fit_unet(
                epochs=conf.unet.epochs,
                checkpoint_dir=conf.unet.checkpoint_dir,
                tensorboard_dir=conf.unet.tensorboard_dir,
                test_as_val=test_as_val,
            )
            """
            checkpoint_lr = f"{conf.unet.checkpoint_dir}/status/optimizer.json"
            print(checkpoint_lr)
            self.lr_schedule.save(checkpoint_lr, conf.unet.epochs, len(self.unet_train_data))
            self.unet.save_weights(f"{conf.unet.checkpoint_dir}/final/model.tf")
            """
        

        else:
            print("\nBUILDING UNET")
            self.build_unet(
                        n_blocks=conf.unet.n_blocks,
                        n_filters=conf.unet.n_filters,
                        k_size=conf.unet.k_size,
                        k_stride=conf.unet.k_stride,
                        dropout=conf.unet.dropout,
                        regularizer=conf.unet.regularizer,
                        model_type=conf.unet.model,
                        lambda_da=conf.unet.lambda_da,
                        squeeze_factor=conf.unet.squeeze_factor,
                        moe_n_experts=conf.unet.moe_n_experts,
                        moe_top_k_experts=conf.unet.moe_top_k_experts,
                        moe_noise=conf.unet.moe_noise,
                        moe_balance_loss=conf.unet.moe_balance_loss,
                    )

            checkpoint = f"{conf.unet.checkpoint_dir}/final_model/model.tf"
            #checkpoint = f"{conf.unet.checkpoint_dir}/saved_model/model.tf"
            self.unet.load_weights(checkpoint).expect_partial()


        ####################################
        ############ DOG DATA ##############
        ####################################
        print("\nLOADING DoG DATA")
        if test_as_val:
            tiff_files = val_tiff_files
            marker_files = val_marker_files
            db_name = f"{conf.source.name}_val_pred_lmdb"
        else:
            tiff_files = train_tiff_files
            marker_files = train_marker_files
            db_name = f"{conf.source.name}_train_pred_lmdb"


        self.dog_train_inputs, self.dog_train_targets = self.make_dog_data(
            tiff_files=tiff_files,
            marker_files=marker_files,
            data_shape=conf.source.shape,
            lmdb_dir=f"{conf.exp.basepath}/{db_name}",
        )
        

        ####################################
        ############### DOG ################
        ####################################
        print("\nTRAINING DoG: IMPORTANT YOU CHANGED THIS PART AND NOW IT IS NOT FITTING DOG, DECOMMENT TO MAKE IT FIT")
        self.fit_dog(
            inputs_lmdb=self.dog_train_inputs,
            targets_list=self.dog_train_targets,
            iterations=conf.dog.iterations,
            dim_resolution=conf.source.dim_resolution,
            exclude_border=conf.dog.exclude_border,
            max_match_dist=conf.dog.max_match_dist,
            checkpoint_dir=conf.dog.checkpoint_dir,
            n_cpu=conf.dog.n_cpu,
        )
        
        print(f"Best parameters found for DoG: {self.dog.get_parameters()}")
        

    def test(self, config_file, test_as_val, gpu=-1, test_on_target=False):
        gpus = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[gpu], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu], True)

        conf = TrainDAConfiguration(config_file)

        if not test_on_target:
            data_conf = conf.source
        else:
            data_conf = conf.target

        test_tiff_files, test_marker_files = get_inputs_target_paths(
            data_conf.test_tif_dir, data_conf.test_gt_dir
        )
        if test_as_val:
            print("ATTN!! Using part of the test-set as validation")
            nt = len(test_tiff_files)
            np.random.seed(self.seed)
            #val_idx = np.random.randint(0, nt, size=nt // 3)
            val_idx = np.random.choice(nt, size=nt // 3, replace=False)
            val_tiff_files = [test_tiff_files[i] for i in val_idx]
            val_marker_files = [test_marker_files[i] for i in val_idx]
            for vtf, vmf in zip(val_tiff_files, val_marker_files):
                test_tiff_files.remove(vtf)
                test_marker_files.remove(vmf)

        ####################################
        ############ LOAD UNET #############
        ####################################
        """
        self.unet = tf.keras.models.load_model(
            f"{conf.unet.checkpoint_dir}/model.tf",
            compile=False,
        )
        """
        self.build_unet(
                        n_blocks=conf.unet.n_blocks,
                        n_filters=conf.unet.n_filters,
                        k_size=conf.unet.k_size,
                        k_stride=conf.unet.k_stride,
                        dropout=conf.unet.dropout,
                        regularizer=conf.unet.regularizer,
                        model_type=conf.unet.model,
                        lambda_da=conf.unet.lambda_da,
                        squeeze_factor=conf.unet.squeeze_factor,
                        moe_n_experts=conf.unet.moe_n_experts,
                        moe_top_k_experts=conf.unet.moe_top_k_experts,
                        moe_noise=conf.unet.moe_noise,
                        moe_balance_loss=conf.unet.moe_balance_loss,
                    )
        
        checkpoint = f"{conf.unet.checkpoint_dir}/final_model/model.tf"
        self.unet.load_weights(checkpoint)

        ###########################################
        ############ UNET PREDICTIONS #############
        ###########################################
        print(f"\nPREPARING TEST DATA")
        self.dog_test_inputs, self.dog_test_targets = self.make_dog_data(
            tiff_files=test_tiff_files,
            marker_files=test_marker_files,
            data_shape=data_conf.shape,
            lmdb_dir=f"{conf.exp.basepath}/{data_conf.name}_test_pred_lmdb",
            **conf.preproc,
        )
        cuda.close()



        ##########################################
        ############ DOG PREDICTIONS #############
        ##########################################
        print(f"\nDoG predictions and evaluation on test-set")
        self.dog = BlobDoG(3, data_conf.dim_resolution, conf.dog.exclude_border)
        dog_par = json.load(
            open(f"{conf.dog.checkpoint_dir}/BlobDoG_parameters.json", "r")
        )
        self.dog.set_parameters(dog_par)
        print(f"Best parameters found for DoG: {dog_par}")

        with self.dog_test_inputs.begin() as fx:
            db_iterator = fx.cursor()
            res = []
            for i, (fname, x) in enumerate(db_iterator):
                if i>=24:
                    break
                pred = self.dog.predict_and_evaluate(
                    x, self.dog_test_targets[i], conf.dog.max_match_dist, "counts"
                )
                
                a = pickle.loads(x)
                print(np.mean(a))
                print(np.std(a))
                print(pred)
                pred["f1"] = pred["TP"] / (pred["TP"] + 0.5 * (pred["FP"] + pred["FN"]))
                pred["file"] = fname.decode()
                res.append(pred)
        self.dog_test_inputs.close()

        res = pd.concat(res)
        res.to_csv(f"{conf.exp.basepath}/{data_conf.name}_test_eval.csv", index=False)
        perf = evaluate_df(res)
        print(f"\nTest-set evaluated with {perf}")
        print("")

    #test the target domain with also a gamma correction added
    def test_gamma_target(self, config_file, test_as_val, gpu=-1, test_on_target=False):
        gpus = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[gpu], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu], True)

        conf = TrainDAConfiguration(config_file)

        if test_on_target:   
            gamma = 2.0
            print("using targets and augmented with gamma:") 
            print(gamma)
            data_conf = conf.target
        else:
            data_conf = conf.source
            gamma = None
        

        test_tiff_files, test_marker_files = get_inputs_target_paths(
            data_conf.test_tif_dir, data_conf.test_gt_dir
        )


        if test_as_val:
            print("ATTN!! Using part of the test-set as validation")
            nt = len(test_tiff_files)
            np.random.seed(self.seed)
            #val_idx = np.random.randint(0, nt, size=nt // 3)
            val_idx = np.random.choice(nt, size=nt // 3, replace=False)
            val_tiff_files = [test_tiff_files[i] for i in val_idx]
            val_marker_files = [test_marker_files[i] for i in val_idx]
            for vtf, vmf in zip(val_tiff_files, val_marker_files):
                test_tiff_files.remove(vtf)
                test_marker_files.remove(vmf)

        ####################################
        ############ LOAD UNET #############
        ####################################
        """
        self.unet = tf.keras.models.load_model(
            f"{conf.unet.checkpoint_dir}/model.tf",
            compile=False,
        )
        """
        self.build_unet(
                        n_blocks=conf.unet.n_blocks,
                        n_filters=conf.unet.n_filters,
                        k_size=conf.unet.k_size,
                        k_stride=conf.unet.k_stride,
                        dropout=conf.unet.dropout,
                        regularizer=conf.unet.regularizer,
                        model_type=conf.unet.model,
                        lambda_da=conf.unet.lambda_da,
                        squeeze_factor=conf.unet.squeeze_factor,
                        moe_n_experts=conf.unet.moe_n_experts,
                        moe_top_k_experts=conf.unet.moe_top_k_experts,
                        moe_noise=conf.unet.moe_noise,
                        moe_balance_loss=conf.unet.moe_balance_loss,
                    )
        
        checkpoint = f"{conf.unet.checkpoint_dir}/final_model/model.tf"
        self.unet.load_weights(checkpoint).expect_partial()

        ###########################################
        ############ UNET PREDICTIONS #############
        ###########################################
        print("TODO CHANGE ALSO MAKE DOG DATA TO HAVE PREPROC ARGS for TARGET")

        print(f"\nPREPARING TEST DATA")
        self.dog_test_inputs, self.dog_test_targets = self.make_dog_data(
            tiff_files=test_tiff_files,
            marker_files=test_marker_files,
            data_shape=data_conf.shape,
            lmdb_dir=f"{conf.exp.basepath}/{data_conf.name}_test_pred_lmdb",
            gamma=gamma,
            **conf.preproc,
        )

        cuda.close()

        ##########################################
        ############ DOG PREDICTIONS #############
        ##########################################
        print(f"\nDoG predictions and evaluation on test-set")
        self.dog = BlobDoG(3, data_conf.dim_resolution, conf.dog.exclude_border)
        dog_par = json.load(
            open(f"{conf.dog.checkpoint_dir}/BlobDoG_parameters.json", "r")
        )
        self.dog.set_parameters(dog_par)
        print(f"Best parameters found for DoG: {dog_par}")

        with self.dog_test_inputs.begin() as fx:
            db_iterator = fx.cursor()
            res = []
            for i, (fname, x) in enumerate(db_iterator):
                pred = self.dog.predict_and_evaluate(
                    x, self.dog_test_targets[i], conf.dog.max_match_dist, "counts"
                )

                pred["f1"] = pred["TP"] / (pred["TP"] + 0.5 * (pred["FP"] + pred["FN"]))
                pred["file"] = fname.decode()
                res.append(pred)
        self.dog_test_inputs.close()

        res = pd.concat(res)
        res.to_csv(f"{conf.exp.basepath}/{data_conf.name}_test_eval.csv", index=False)
        perf = evaluate_df(res)
        print(f"\nTest-set evaluated with {perf}")
        print("")

    #write on csv scatter plot of f1 source and target for random DoG hyperparams
    #go on watch results to see it
    def test_domains(self, config_file, test_as_val, gpu=-1, test_on_target=False):
        gpus = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[gpu], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu], True)

        conf = TrainDAConfiguration(config_file)

        data_conf = conf.target

        test_tiff_files, test_marker_files = get_inputs_target_paths(
            data_conf.test_tif_dir, data_conf.test_gt_dir
        )

        self.build_unet(
                        n_blocks=conf.unet.n_blocks,
                        n_filters=conf.unet.n_filters,
                        k_size=conf.unet.k_size,
                        k_stride=conf.unet.k_stride,
                        dropout=conf.unet.dropout,
                        regularizer=conf.unet.regularizer,
                        model_type=conf.unet.model,
                        lambda_da=conf.unet.lambda_da,
                        squeeze_factor=conf.unet.squeeze_factor,
                        moe_n_experts=conf.unet.moe_n_experts,
                        moe_top_k_experts=conf.unet.moe_top_k_experts,
                        moe_noise=conf.unet.moe_noise,
                        moe_balance_loss=conf.unet.moe_balance_loss,
                    )
        
        checkpoint = f"{conf.unet.checkpoint_dir}/final_model/model.tf"
        self.unet.load_weights(checkpoint).expect_partial()



        ###########################################
        ############ UNET PREDICTIONS #############
        ###########################################
        print(f"\nPREPARING TEST DATA")
        self.dog_test_inputs, self.dog_test_targets = self.make_dog_data(
            tiff_files=test_tiff_files,
            marker_files=test_marker_files,
            data_shape=data_conf.shape,
            lmdb_dir=f"{conf.exp.basepath}/{data_conf.name}_test_pred_lmdb",
            **conf.preproc,
        )

        #target domain
        data_conf = conf.target

        test_tiff_files, test_marker_files = get_inputs_target_paths(
            data_conf.test_tif_dir, data_conf.test_gt_dir
        )

        ###########################################
        ############ UNET PREDICTIONS #############
        ###########################################
        print(f"\nPREPARING TEST DATA")
        self.dog_test_inputs_target, self.dog_test_targets_target = self.make_dog_data(
            tiff_files=test_tiff_files,
            marker_files=test_marker_files,
            data_shape=data_conf.shape,
            lmdb_dir=f"{conf.exp.basepath}/{data_conf.name}_test_pred_lmdb",
            **conf.preproc,
        )

        cuda.close()




        ##########################################
        ############ DOG PREDICTIONS #############
        ##########################################

        num_points = 300


        with open("/home/amarinai/DeepLearningThesis/Results/Unet_FDA_06/results.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['min_rad', 'max_rad', 'sigma_ratio', 'overlap',"threshold",
                            'f1_source', 'prec_source', 'rec_source', 'acc_source',
                            'f1_target', 'prec_target', 'rec_target', 'acc_target'])
                
            
            for i in range(num_points):
                print()
                print(i)
                print(f"\nDoG predictions and evaluation on test-set")
                self.dog = BlobDoG(3, data_conf.dim_resolution, conf.dog.exclude_border)
                """dog_par = json.load(
                    open(f"{conf.dog.checkpoint_dir}/BlobDoG_parameters.json", "r")
                )"""

                min = np.random.uniform(4.0, 15.0)
                diff = np.random.uniform(1.0, 10.0)
                dog_par = {"min_rad": min, 
                        "max_rad": min+diff,
                        "sigma_ratio": np.random.uniform(1.1, 2.0),
                        "overlap": np.random.uniform(0.05, 0.5),
                        "threshold": np.random.uniform(0.05, 0.5),
                }
                
                self.dog.set_parameters(dog_par)
                print(f"Best parameters found for DoG: {dog_par}")
                

                with self.dog_test_inputs.begin() as fx:
                    db_iterator = fx.cursor()
                    res = []
                    for i, (fname, x) in enumerate(db_iterator):
                        pred = self.dog.predict_and_evaluate(
                            x, self.dog_test_targets[i], conf.dog.max_match_dist, "counts"
                        )

                        pred["f1"] = pred["TP"] / (pred["TP"] + 0.5 * (pred["FP"] + pred["FN"]))
                        pred["file"] = fname.decode()
                        res.append(pred)
                

                res = pd.concat(res)
                res.to_csv(f"{conf.exp.basepath}/{data_conf.name}_test_eval.csv", index=False)
                perf = evaluate_df(res)
                print(f"\nTest-set evaluated with {perf}")
                print("")

                with self.dog_test_inputs_target.begin() as fx:
                    db_iterator = fx.cursor()
                    res = []
                    for i, (fname, x) in enumerate(db_iterator):
                        pred = self.dog.predict_and_evaluate(
                            x, self.dog_test_targets_target[i], conf.dog.max_match_dist, "counts"
                        )

                        pred["f1"] = pred["TP"] / (pred["TP"] + 0.5 * (pred["FP"] + pred["FN"]))
                        pred["file"] = fname.decode()
                        res.append(pred)
                

                res = pd.concat(res)
                res.to_csv(f"{conf.exp.basepath}/{data_conf.name}_test_eval.csv", index=False)
                perf2 = evaluate_df(res) # return {"prec": prec, "rec": rec, "f1": f1, "acc": acc}
                print(f"\nTest-set target evaluated with {perf2}")
                print("")

                writer.writerow([dog_par["min_rad"], dog_par["max_rad"], dog_par["sigma_ratio"], 
                                 dog_par["overlap"],dog_par["threshold"], 
                                 perf["f1"], perf["prec"], perf["rec"], perf["acc"],
                                 perf2["f1"], perf2["prec"], perf2["rec"], perf2["acc"]])

     
            self.dog_test_inputs.close()
            self.dog_test_inputs_target.close()

    #get oracle f1 on target aumented with gamma 
    def get_oracle_f1_gamma(self, config_file, test_as_val, gpu=-1, test_on_target=False):
        
        gpus = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[gpu], "GPU")
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        tf.config.experimental.set_memory_growth(gpus[gpu], True)  #non occupa tutta la memoria subito
        
        conf = TrainDAConfiguration(config_file)

        wandb.init(project='my-tf-integration', name=conf.exp.name)
        gamma = None

        self.build_unet(
                        n_blocks=conf.unet.n_blocks,
                        n_filters=conf.unet.n_filters,
                        k_size=conf.unet.k_size,
                        k_stride=conf.unet.k_stride,
                        dropout=conf.unet.dropout,
                        regularizer=conf.unet.regularizer,
                        model_type=conf.unet.model,
                        lambda_da=conf.unet.lambda_da,
                        squeeze_factor=conf.unet.squeeze_factor,
                        moe_n_experts=conf.unet.moe_n_experts,
                        moe_top_k_experts=conf.unet.moe_top_k_experts,
                        moe_noise=conf.unet.moe_noise,
                        moe_balance_loss=conf.unet.moe_balance_loss,
                    )
        
        checkpoint = f"{conf.unet.checkpoint_dir}/final_model/model.tf"
        self.unet.load_weights(checkpoint)



        ###########################################
        ############ UNET PREDICTIONS #############
        ###########################################
        print(f"\nPREPARING TEST DATA")

        #target domain
        data_conf = conf.target

        test_tiff_files, test_marker_files = get_inputs_target_paths(
            data_conf.test_tif_dir, data_conf.test_gt_dir
        )

        ###########################################
        ############ UNET PREDICTIONS #############
        ###########################################
        print(f"\nPREPARING TEST DATA")
        self.dog_test_inputs_target, self.dog_test_targets_target = self.make_dog_data(
            tiff_files=test_tiff_files,
            marker_files=test_marker_files,
            data_shape=data_conf.shape,
            lmdb_dir=f"{conf.exp.basepath}/{data_conf.name}_test_pred_lmdb",
            gamma=gamma,
            **conf.preproc,
        )

        cuda.close()



        ##########################################
        ############ DOG PREDICTIONS #############
        ##########################################

        num_points = 150
                
        maxf1 = 0.0  
        for i in range(num_points):
            print()
            print(i)
            print(f"\nDoG predictions and evaluation on test-set")
            self.dog = BlobDoG(3, data_conf.dim_resolution, conf.dog.exclude_border)
            """dog_par = json.load(
                open(f"{conf.dog.checkpoint_dir}/BlobDoG_parameters.json", "r")
            )"""

            min = np.random.uniform(1.0, 10.0)
            #min = np.random.uniform(4.0, 15.0)
            diff = np.random.uniform(1.0, 10.0)
            dog_par = {"min_rad": min, 
                    "max_rad": min+diff,
                    "sigma_ratio": np.random.uniform(1.1, 2.0),
                    "overlap": np.random.uniform(0.05, 0.5),
                    "threshold": np.random.uniform(0.05, 0.5),
            }
            
            self.dog.set_parameters(dog_par)
            print(f"Best parameters found for DoG: {dog_par}")

            with self.dog_test_inputs_target.begin() as fx:
                db_iterator = fx.cursor()
                res = []
                for i, (fname, x) in enumerate(db_iterator):
                    pred = self.dog.predict_and_evaluate(
                        x, self.dog_test_targets_target[i], conf.dog.max_match_dist, "counts"
                    )

                    pred["f1"] = pred["TP"] / (pred["TP"] + 0.5 * (pred["FP"] + pred["FN"]))
                    pred["file"] = fname.decode()
                    res.append(pred)
            

            res = pd.concat(res)
            res.to_csv(f"{conf.exp.basepath}/{data_conf.name}_test_eval.csv", index=False)
            perf2 = evaluate_df(res) # return {"prec": prec, "rec": rec, "f1": f1, "acc": acc}
            print(f"\nTest-set target evaluated with {perf2}")
            print("")
            
            f1_oracle = perf2["f1"]
            print(f1_oracle)
            maxf1 = max(f1_oracle, maxf1)
            print(f"new max: {maxf1}")

        self.dog_test_inputs_target.close()
        wandb.log({"f1_oracle": float(maxf1)})

    def get_statistics_DA(self, config_file):
        conf = TrainDAConfiguration(config_file)
            

        train_tiff_files, train_marker_files = get_inputs_target_paths(
            conf.source.train_tif_dir, conf.source.train_gt_dir
        )

        train_tiff_files_target = get_inputs_target_paths_no_gt(conf.target.train_tif_dir)

        self.make_unet_data_DA(
                    train_inputs=train_tiff_files,
                    train_targets=train_marker_files,
                    train_inputs_target = train_tiff_files_target,
                    dim_resolution=conf.source.dim_resolution,
                    input_shape=conf.unet.input_shape,
                    augmentations=conf.data_aug.op_args,
                    augmentations_prob=conf.data_aug.op_probs,
                    batch_size= conf.unet.batch_size,
                    preprocessing_kwargs=conf.preproc,
                    preprocessing_kwargs_target=conf.preproct                
                    )

        for step, data in enumerate(self.unet_train_data):
            source, x_target = data
            x, _ = source
            print(np.max(x), np.max(x_target))

        

          


def get_inputs_target_paths(input_dir, target_dir):
    fnames = os.listdir(input_dir)
    input_files = sorted([f"{input_dir}/{fname}" for fname in fnames])
    target_files = sorted([f"{target_dir}/{fname}.marker" for fname in fnames])
    return input_files, target_files

def get_inputs_target_paths_no_gt(input_dir):
    fnames = os.listdir(input_dir)
    input_files = sorted([f"{input_dir}/{fname}" for fname in fnames])
    return input_files

def get_gt_domain(repeats, pixel_wise):
    if not pixel_wise:
        y_true_domain = [1, 0.0]  #source domain maybe the opposite
        y_true_domain = tf.expand_dims(y_true_domain, axis=0)
        y_true_domain = tf.repeat(y_true_domain, repeats=repeats, axis=0)
        y_true_domains = [0.0, 1]  #target domain maybe the opposite
        y_true_domains = tf.expand_dims(y_true_domains, axis=0)
        y_true_domains = tf.repeat(y_true_domains, repeats=repeats, axis=0)
        y_true_domain = tf.concat([y_true_domain, y_true_domains], axis=0)
    else:
        y_true_domain = tf.zeros((repeats*2, 80, 120, 120, 1), dtype=tf.float32)
        y_true_domain = tf.tensor_scatter_nd_update(y_true_domain, tf.constant([[i] for i in range(repeats)]), tf.ones((repeats, 80, 120, 120, 1), dtype=tf.float32))
    return y_true_domain

def sigmoid_decay_schedule(epoch, max_epochs):
    return (2 / (1 + np.exp(-10 * epoch / max_epochs))) - 1

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="train_DA.py",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, max_help_position=52, width=90
        ),
    )
    parser.add_argument("config", type=str, help="YAML Configuration file")
    parser.add_argument("--gpu", type=int, default=-1, help="Index of GPU to use")
    parser.add_argument(
        "--lmdb",
        default=False,
        action="store_true",  #means that just the argument change from false to true
        help="In case of huge dataset store it as lmdb to save RAM usage",
    )
    parser.add_argument(
        "--only-dog",
        default=False,
        action="store_true",
        help="Skip UNet training and train only the DoG",
    )
    parser.add_argument(
        "--val-from-test",
        default=False,
        action="store_true",
        help="part of the test-set as validation. UNet weights will be saved only when validation loss improves",
    )
    parser.add_argument(
        "--only-test",
        default=False,
        action="store_true",
        help="Run only evaluation on the test-set",
    )
    parser.add_argument(        
        "--test-on-target",
        default=False,
        action="store_true",
        help="Test on the target domain which needs to have GT files",)
    parser.add_argument(        
        "--checkpoint-model",
        default=False,
        action="store_true",
        help="Loads the model in the checkpoint directory",)
    parser.add_argument("--slice", type=int, default=-1, help="Defines if the model is trained on broca's slices and which one is left out")
    return parser.parse_args()


def main():
    args = parse_args()

    trainer = Trainer()
    #trainer.get_statistics_DA(args.config)
    
    if not args.only_test:
        trainer.run(args.config, args.only_dog, args.val_from_test, args.lmdb, args.gpu, args.checkpoint_model)
        #trainer.get_oracle_f1_gamma(args.config, args.val_from_test, args.gpu, args.test_on_target)
    else:
        #trainer.test(args.config, args.val_from_test, args.gpu, args.test_on_target)
        trainer.test(args.config, args.val_from_test, args.gpu, args.test_on_target, args.slice)
        #trainer.test_gamma_target(args.config, args.val_from_test, args.gpu, args.test_on_target)
        #trainer.get_oracle_f1_gamma(args.config, args.val_from_test, args.gpu, args.test_on_target)


if __name__ == "__main__":
    main()
