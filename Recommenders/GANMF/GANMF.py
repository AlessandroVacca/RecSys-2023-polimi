#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
"""

import os
import sys
import time
from tensorflow.python.framework.ops import reset_default_graph
import tqdm
import pickle
import shutil
import numpy as np
import tensorflow as tf
from datetime import datetime
from Recommenders.BaseRecommender import BaseRecommender
from Utils_ import EarlyStoppingScheduler, save_weights


class GANMF(BaseRecommender):
    RECOMMENDER_NAME = 'GANMF'

    def __init__(self, URM_train, mode='user', verbose=False, seed=1234, is_experiment=False):
        super(GANMF, self).__init__(URM_train, verbose)
        tf.compat.v1.disable_eager_execution()
        if mode not in ['user', 'item']:
            raise ValueError('Accepted training modes are `user` and `item`. Given was {}.', mode)

        self.mode = mode
        if self.mode == 'item':
            self.URM_train = URM_train.T.tocsr()
        else:
            self.URM_train = URM_train
        self.num_users, self.num_items = self.URM_train.shape
        self.config = None
        self.seed = seed
        self.verbose = verbose
        self.logsdir = os.path.join('plots', self.RECOMMENDER_NAME, datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.is_experiment = is_experiment

        if not os.path.exists(self.logsdir) and not self.is_experiment:
            os.makedirs(self.logsdir, exist_ok=False)

        if not self.is_experiment:
            # Save this file inside logsdir/code
            codesdir = os.path.join(self.logsdir, 'code')
            os.makedirs(codesdir, exist_ok=False)
            shutil.copy(os.path.abspath(sys.modules[self.__module__].__file__), codesdir)

    def build(self, num_factors=10, emb_dim=32):
        self.num_factors = num_factors
        self.emb_dim = emb_dim

        glorot_uniform = tf.compat.v1.glorot_uniform_initializer()

        ########################
        # AUTOENCODER FUNCTION #
        ########################
        def autoencoder(input_data):
            with tf.compat.v1.variable_scope('autoencoder', reuse=tf.compat.v1.AUTO_REUSE):
                encoding = tf.compat.v1.layers.dense(input_data, units=emb_dim, kernel_initializer=glorot_uniform,
                                           name='encoding')
                decoding = tf.compat.v1.layers.dense(encoding, units=self.num_items, kernel_initializer=glorot_uniform,
                                           name='decoding')
            loss = tf.compat.v1.losses.mean_squared_error(input_data, decoding)
            # loss = -tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=input_data, logits=decoding))
            return encoding, loss

        ######################
        # GENERATOR FUNCTION #
        ######################
        def generator(condition):
            with tf.compat.v1.variable_scope('generator', reuse=tf.compat.v1.AUTO_REUSE):
                user_embeddings = tf.compat.v1.get_variable(shape=[self.num_users, num_factors], trainable=True,
                                                  initializer=glorot_uniform, name='user_embeddings')
                item_embeddings = tf.compat.v1.get_variable(shape=[self.num_items, num_factors], trainable=True,
                                                  initializer=glorot_uniform, name='item_embeddings')

            user_lookup = tf.compat.v1.nn.embedding_lookup(user_embeddings, condition)
            fake_data = tf.compat.v1.matmul(tf.compat.v1.squeeze(user_lookup, axis=1), item_embeddings, transpose_b=True)
            return fake_data, user_embeddings, item_embeddings

        self.autoencoder, self.generator = autoencoder, generator

    def fit(self, num_factors=10, emb_dim=32, epochs=300, batch_size=32, d_lr=1e-4, g_lr=1e-4, d_steps=1, g_steps=1,
            d_reg=0, g_reg=0, m=1, recon_coefficient=1e-2, allow_worse=5, freq=1, after=0, metrics=['MAP'],
            sample_every=None, validation_evaluator=None, validation_set=None):

        # Construct the model config
        self.config = dict(locals())
        del self.config['self']

        # First clear the session to save GPU memory
        tf.compat.v1.reset_default_graph()
        # Set fixed seed for the tf.compat.v1 graph
        tf.compat.v1.set_random_seed(self.seed)

        self.build(num_factors, emb_dim)

        # Create optimizers
        opt_gen = tf.compat.v1.train.AdamOptimizer(learning_rate=g_lr)
        opt_disc = tf.compat.v1.train.AdamOptimizer(learning_rate=d_lr)

        # placeholders
        real_profile = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=[None, self.num_items])
        user_id = tf.compat.v1.placeholder(dtype=tf.compat.v1.int32, shape=[None, 1])

        # generator ops
        fake_profile, _, _ = self.generator(user_id)

        # autoencoder ops
        real_encoding, real_recon_loss = self.autoencoder(real_profile)
        fake_encoding, fake_recon_loss = self.autoencoder(fake_profile)

        # model parameters
        self.params = {}
        self.params['D'] = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='autoencoder')
        self.params['G'] = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        self.best_params = {}
        for p in self.params:
            self.best_params[p] = []
            for idx, var in enumerate(self.params[p]):
                self.best_params[p].append(tf.compat.v1.get_variable(p + '_best_params_' + str(idx), shape=var.get_shape(),
                                                           trainable=False))

        # losses
        dloss = real_recon_loss + tf.compat.v1.maximum(0.0, m * real_recon_loss - fake_recon_loss) + \
                d_reg * tf.compat.v1.add_n([tf.compat.v1.nn.l2_loss(var) for var in self.params['D']])
        gloss = (1 - recon_coefficient) * fake_recon_loss + \
                recon_coefficient * tf.compat.v1.losses.mean_squared_error(real_encoding, fake_encoding) + \
                g_reg * tf.compat.v1.add_n([tf.compat.v1.nn.l2_loss(var) for var in self.params['G']])

        # update ops
        dtrain = opt_disc.minimize(dloss, var_list=self.params['D'])
        gtrain = opt_gen.minimize(gloss, var_list=self.params['G'])

        ##################
        # START TRAINING #
        ##################

        # DO NOT allocate all GPU memory to this process
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.compat.v1.initialize_all_variables())

        self._stop_training = False
        if validation_evaluator is not None:
            early_stop = EarlyStoppingScheduler(self, evaluator=validation_evaluator, allow_worse=allow_worse,
                                                freq=freq, metrics=metrics, after=after)

        all_users = np.array(range(self.num_users))
        step = batch_size

        train_g_loss = []
        train_d_loss = []

        if self.verbose:
            print('Starting training...')

        t_start = time.time()
        e_start = time.time()

        epoch = 1

        pbar = tqdm.tqdm(total=epochs, initial=1)

        while not self._stop_training and epoch < epochs + 1:
            batch_d_loss = []
            batch_g_loss = []
            np.random.shuffle(all_users)
            for _ in range(d_steps):
                start_idx = 0
                while start_idx < len(all_users):
                    end_idx = start_idx + step
                    if end_idx > len(all_users):
                        end_idx = len(all_users)

                    uids = all_users[start_idx: end_idx]
                    real_histories = self.URM_train[uids].toarray()

                    _, _dloss = self.sess.run([dtrain, dloss],
                                              {real_profile: real_histories, user_id: uids.reshape(-1, 1)})
                    batch_d_loss.append(_dloss)
                    start_idx = end_idx

            for _ in range(g_steps):
                start_idx = 0
                while start_idx < len(all_users):
                    end_idx = start_idx + step
                    if end_idx > len(all_users):
                        end_idx = len(all_users)

                    uids = all_users[start_idx: end_idx]
                    real_histories = self.URM_train[uids].toarray()
                    _, _gloss = self.sess.run([gtrain, gloss],
                                              {real_profile: real_histories, user_id: uids.reshape(-1, 1)})
                    batch_g_loss.append(_gloss)
                    start_idx = end_idx

            mean_epoch_g_loss = np.mean(batch_g_loss)
            mean_epoch_d_loss = np.mean(batch_d_loss)

            train_g_loss.append(mean_epoch_g_loss)
            train_d_loss.append(mean_epoch_d_loss)

            if validation_set is not None and sample_every is not None and epoch % sample_every == 0:
                t_end = time.time()
                total = t_end - e_start
                print('Epoch : {:d}. Total: {:.2f} secs, {:.2f} secs/epoch.'.format(epoch, total, total / sample_every))
                if self.mode == 'item':
                    self.URM_train = self.URM_train.T.tocsr()
                _, results_run_string = validation_evaluator.evaluateRecommender(self)
                if self.mode == 'item':
                    self.URM_train = self.URM_train.T.tocsr()
                print(results_run_string)
                e_start = time.time()

            if validation_evaluator is not None:
                if self.mode == 'item':
                    self.URM_train = self.URM_train.T.tocsr()
                early_stop(epoch)
                if self.mode == 'item':
                    self.URM_train = self.URM_train.T.tocsr()

                if self._stop_training:
                    print('Training stopped, epoch:', epoch)

            epoch += 1
            pbar.update()
        pbar.close()

        t_end = time.time()
        if self.verbose:
            print('Training took {:.2f} seconds'.format(t_end - t_start))

        if self.mode == 'item':
            self.URM_train = self.URM_train.T.tocsr()

        return epoch - 1 if self._stop_training else epoch

    def stop_fit(self):
        self._stop_training = True

    def save_current_model(self):
        for model in self.params:
            save_weights(self.sess, self.params[model], self.best_params[model])

    def load_model(self):
        for model in self.best_params:
            save_weights(self.sess, self.best_params[model], self.params[model])

    def load_weights(self, best_params, weights):
        self.build(best_params['num_factors'], best_params['emb_dim'])

        # placeholders
        real_profile = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=[None, self.num_items])
        user_id = tf.compat.v1.placeholder(dtype=tf.compat.v1.int32, shape=[None, 1])

        # generator ops
        fake_profile = self.generator(user_id)

        # autoencoder ops
        real_encoding, real_recon_loss = self.autoencoder(real_profile)
        fake_encoding, fake_recon_loss = self.autoencoder(fake_profile)

        # model parameters
        self.params = {}
        self.params['D'] = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='autoencoder')
        self.params['G'] = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)

        best_params = {}
        for model in weights:
            best_params[model] = [tf.compat.v1.convert_to_tensor(w) for w in weights[model]]
            save_weights(self.sess, best_params[model], self.params[model])

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        user_id = tf.compat.v1.placeholder(dtype=tf.compat.v1.int32, shape=[None, 1])
        fake_profile, _, _ = self.generator(user_id)
        if self.mode == 'item':
            predictions = self.sess.run(fake_profile, {user_id: np.array(range(self.num_users)).reshape(-1, 1)})
            return predictions.transpose()[user_id_array]
        else:
            return self.sess.run(fake_profile, {user_id: user_id_array.reshape(-1, 1)})

    def user_factors(self):
        user_id = tf.compat.v1.placeholder(dtype=tf.compat.v1.int32, shape=[None, 1])
        _, user_embeddings, _ = self.generator(user_id)
        return self.sess.run(user_embeddings, {user_id: np.array(range(self.num_users)).reshape(-1, 1)})

    def item_factors(self):
        user_id = tf.compat.v1.placeholder(dtype=tf.compat.v1.int32, shape=[None, 1])
        _, _, item_embeddings = self.generator(user_id)
        return self.sess.run(item_embeddings, {user_id: np.array(range(self.num_items)).reshape(-1, 1)})

    def autoencoder_codes(self):
        real_profile = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=[None, self.num_items])
        real_encoding, _ = self.autoencoder(real_profile)
        return self.sess.run(real_encoding, {real_profile: self.URM_train.toarray()})

    def saveModel(self, folder_path, file_name=None):
        build_params = {'num_factors': self.num_factors, 'emb_dim': self.emb_dim}
        with open(os.path.join(folder_path, 'build_params.pkl'), 'wb') as f:
            pickle.dump(build_params, f, pickle.HIGHEST_PROTOCOL)
        all_params = [var for k in self.params.keys() for var in self.params[k]]
        tf.compat.v1.train.Saver(all_params, max_to_keep=1).save(self.sess, os.path.join(folder_path,
                                                                               self.RECOMMENDER_NAME + '_' + self.mode if file_name is None else file_name),
                                                       write_meta_graph=False, write_state=False)

    def loadModel(self, folder_path, file_name=None):
        filepath = os.path.join(folder_path, 'build_params.pkl')
        if self.verbose:
            print(self.RECOMMENDER_NAME + ': Loading model from file ' + filepath)

        with open(filepath, 'rb') as f:
            build_params = pickle.load(f)

        # First clear the session to save GPU memory
        tf.compat.v1.reset_default_graph()
        # Set fixed seed for the tf.compat.v1 graph
        tf.compat.v1.set_random_seed(self.seed)

        self.build(**build_params)

        real_profile = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=[None, self.num_items])
        user_id = tf.compat.v1.placeholder(dtype=tf.compat.v1.int32, shape=[None, 1])

        self.autoencoder(real_profile)
        self.generator(user_id)

        saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session()
        saver.restore(self.sess, os.path.join(folder_path,
                                              self.RECOMMENDER_NAME + '_' + self.mode if file_name is None else file_name))

        if self.verbose:
            print(self.RECOMMENDER_NAME + ': Loading complete')