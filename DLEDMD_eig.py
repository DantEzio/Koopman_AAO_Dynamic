# -*- coding: utf-8 -*-

"""

Created on Sat Oct  8 21:19:07 2022



@author: chongloc

"""

import tensorflow as tf

from tensorflow.keras import layers, models, optimizers

import numpy as np



class DLEDMD_net():

    def __init__(self,params):

        super(DLEDMD_net, self).__init__()

        tf.compat.v1.disable_eager_execution()

        tf.compat.v1.config.set_soft_device_placement(True)

        self.params=params

          

    def build_net(self,inlr):

        self.inlr=inlr

        # Input

        self.s = layers.Input(shape=self.params['s_input_shape'], name='s_input')

        self.c = layers.Input(shape=self.params['c_input_shape'], name='c_input')

        self.i = layers.Input(shape=self.params['i_input_shape'], name='i_input')

        self.o = layers.Input(shape=self.params['o_input_shape'], name='o_output')



        # S net

        s_prev0 = self.s

        for i in np.arange(self.params['snet_layer']):

            s_prev0 = layers.Dense(self.params['s_net'][i]['num'], activation='tanh', name='snet_' + str(i))(s_prev0)

        s_prev0 = layers.Dense(100, activation='tanh', name='soutput')(s_prev0)



        # C net

        c_prev0 = self.c

        for i in np.arange(self.params['cnet_layer']):

            c_prev0 = layers.Dense(self.params['c_net'][i]['num'], activation='tanh', name='cnet_' + str(i))(c_prev0)

        c_prev0 = layers.Dense(100, activation='tanh', name='coutput')(c_prev0)



        # I net

        i_prev=self.i

        for i in np.arange(self.params['inet_layer']):

            F_prev=layers.Dense(self.params['i_net'][i]['num'], activation='tanh', name='inet_'+str(i))(i_prev)

        i_prev=layers.Dense(100, activation='tanh',name='ioutput')(i_prev)



        # Combine

        self.csci = layers.concatenate([s_prev0, c_prev0, i_prev], name='Combine')



        #encoding layers

        normal_initializer = tf.random_normal_initializer()

        prev_layeru = self.csci

        for i in np.arange(self.params['Encoding_layer']):

            prev_layeru = layers.Dense(self.params['Encoding_param'][i]['num'], activation='tanh', name='Encoding_'+str(i))(prev_layeru)

        

        '''

        #Koopman net

        prev_layeru = layers.Dense(self.params['UEncoding_param'][self.params['UEncoding_layer'] - 1]['num'],

                                   activation='linear',use_bias=False, name='UK_')(prev_layeru)

        '''

        self.faiXu = prev_layeru

        #self.Gu = tf.matmul(tf.transpose(self.faiXu), self.faiXu) * (1 / self.params['training_data_num'])

        #self.Au = tf.matmul(tf.transpose(self.faiXu), self.o) * (1 / self.params['training_data_num'])

        #self.Ku = tf.matmul(tf.compat.v1.matrix_inverse(self.Gu + 0.00001 * np.eye(self.Gu.shape[0])), self.Au)

        
        self.model = models.Model(inputs=[self.s, self.c, self.i, self.o],outputs=[self.faiXu])

        adam = optimizers.Adam(lr=inlr)

        self.model.compile(optimizer=adam, loss=['mse'])



        #self.model.summary()

        