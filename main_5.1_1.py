# -*- coding: utf-8 -*-

"""

Created on Sat Oct  8 21:19:58 2022



@author: chongloc

"""

import numpy as np

import data_process as dp

import DLEDMD

import MLP



from tensorflow.keras import models

from tensorflow.keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt

import time



import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"



#2019, week10 to week15 for training

#2019, week16 to week20 for testing

#selected = [10]

#DATA_U,DATA_V,DATA_R,DATA_F,U_loc_log,V_loc_log,R_loc_log=get_data('2019',selected)



'''

Training data

'''



Utrain=np.load('./data/new version data/2019/U.npy',allow_pickle=True)

U_loc_log=np.load('./data/new version data/2019/U_loc.npy',allow_pickle=True).item()

Vtrain=np.load('./data/new version data/2019/V.npy',allow_pickle=True)

V_loc_log=np.load('./data/new version data/2019/V_loc.npy',allow_pickle=True).item()

Ftrain=np.load('./data/new version data/2019/F.npy',allow_pickle=True)



print(Utrain.shape,Vtrain.shape,Ftrain.shape)

t_DATA_F,fmax,fmin=dp.normlize_F(Ftrain)

t_DATA_U,umax,umin=dp.normlize(Utrain)

t_DATA_V,vmax,vmin=dp.normlize(Vtrain)

np.save('./5.1 Results/Umax.npy',umax)

np.save('./5.1 Results/Umin.npy',umin)

np.save('./5.1 Results/Vmax.npy',vmax)

np.save('./5.1 Results/Vmin.npy',vmin)

n=t_DATA_U.shape[0]





for mmd in ['MLP_net']:



    r=[]

    selected=np.array([i for i in range(20,2200,100)])

    lag=10



    Utrain = t_DATA_U[0:n - lag, :]

    Vtrain = t_DATA_V[0:n - lag, :]

    Ftrain = t_DATA_F[1:n - lag + 1, :]

    UYtrain = t_DATA_U[1:n - lag + 1, :]

    VYtrain = t_DATA_V[1:n - lag + 1, :]

    Utrain_out1 = t_DATA_U[2:n - lag + 2, :]

    Vtrain_out1 = t_DATA_V[2:n - lag + 2, :]

    Utrain_out2 = t_DATA_U[3:n - lag + 3, :]

    Vtrain_out2 = t_DATA_V[3:n - lag + 3, :]

    Utrain_out3 = t_DATA_U[4:n - lag + 4, :]

    Vtrain_out3 = t_DATA_V[4:n - lag + 4, :]

    '''

    Utrain = t_DATA_U[selected, :]

    Vtrain = t_DATA_V[selected, :]

    Ftrain = t_DATA_F[selected + 1, :]

    UYtrain = t_DATA_U[selected + 1,:]

    VYtrain = t_DATA_V[selected + 1, :]

    Utrain_out1 = t_DATA_U[selected + 2,:]

    Vtrain_out1 = t_DATA_V[selected + 2,:]

    Utrain_out2 = t_DATA_U[selected + 3, :]

    Vtrain_out2 = t_DATA_V[selected + 3, :]

    Utrain_out3 = t_DATA_U[selected + 4, :]

    Vtrain_out3 = t_DATA_V[selected + 4, :]

    '''

    if mmd=='DLEDMD_net':

        params = {

                'u_input_shape': Utrain.shape[1], 'v_input_shape': Vtrain.shape[1], 'f_input_shape': Ftrain.shape[1],

                'unet_layer': 3,

                'u_net': [{'num': 50}, {'num': 50}, {'num': 50}],

                'vnet_layer': 3,

                'v_net': [{'num': 50}, {'num': 50}, {'num': 50}],

                'fnet_layer': 3,

                'f_net': [{'num': 50}, {'num': 50},  {'num': 50}],

                'UEncoding_layer': 4,

                'UEncoding_param': [{'num': 50} for _ in range(4)],

                'VEncoding_layer': 4,

                'VEncoding_param': [{'num': 50} for _ in range(4)],

                'UOutput_layer': 6,

                'UOutput_param': [{'num': 50} for _ in range(5)] + [{'num': Utrain_out1.shape[1]}],

                'VOutput_layer': 6,

                'VOutput_param': [{'num': 50} for _ in range(5)] + [{'num': Vtrain_out1.shape[1]}],

                'training_data_num': Utrain.shape[0]

        }

    else:

        params = {

                'u_input_shape': Utrain.shape[1], 'v_input_shape': Vtrain.shape[1], 'f_input_shape': Ftrain.shape[1],

                'unet_layer': 3,

                'u_net': [{'num': 50}, {'num': 50}, {'num': 50}],

                'vnet_layer': 3,

                'v_net': [{'num': 50}, {'num': 50}, {'num': 50}],

                'fnet_layer': 3,

                'f_net': [{'num': 50}, {'num': 50},  {'num': 50}],

                'UOutput_layer': 6,

                'UOutput_param': [{'num': 50} for _ in range(5)] + [{'num': Utrain_out1.shape[1]}],

                'VOutput_layer': 6,

                'VOutput_param': [{'num': 50} for _ in range(5)] + [{'num': Vtrain_out1.shape[1]}],

                'training_data_num': Utrain.shape[0]

        }

    print(params)

    inlr=1e-4

    if mmd=='DLEDMD_net':

        model=DLEDMD.DLEDMD_net(params)

    else:

        model=MLP.MLP_net(params)

    model.build_net(inlr)

    print('hahaha')

    model.model = models.load_model('./5.1 Results/' + mmd + '/' + mmd + '/model_v2.h5')

    '''

    Initial training

    '''

    step=1000

    Train = True

    BAT = False

    r=[]

    if Train:

        #model.model=models.load_model('./5.1 Results/'+mmd+'/'+mmd+'/model_v5_1.h5')

        print('train model')

        if BAT:

            print('batch')

            batch_size=100

            for p in range(step):

                print('step:', p)

                for it in range(int(n/batch_size)-1):

                    x,y=[Utrain[it * batch_size:(it + 1) * batch_size],

                         Vtrain[it * batch_size:(it + 1) * batch_size],

                         Ftrain[it * batch_size:(it + 1) * batch_size],

                         UYtrain[it * batch_size:(it + 1) * batch_size],

                         VYtrain[it * batch_size:(it + 1) * batch_size]],\

                        [Utrain_out[it*batch_size:(it+1)*batch_size],

                         Vtrain_out[it*batch_size:(it+1)*batch_size]]

                    y_batch = model.model.train_on_batch(x=x, y=y)

                r.append(y_batch[0])

                model.model.save('./5.1 Results/' + mmd + '/' + mmd  + '/model_v2.h5')

                np.save('./5.1 Results/' + mmd + '/' + mmd + '/model_train_v2.npy', r)

        else:

            print('full')

            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, verbose=1)

            hist=model.train(Utrain,Vtrain,Ftrain,UYtrain,VYtrain,

                             [Utrain_out1,Utrain_out2,Utrain_out3],[Vtrain_out1,Vtrain_out2,Vtrain_out3],

                             step,reduce_lr)

            model.model.save('./5.1 Results/' + mmd + '/' + mmd + '/model_v2.h5')

            r+=hist.history['loss']

            np.save('./5.1 Results/' + mmd + '/' + mmd + '/model_train4.npy', r)

        plt.figure()

        plt.plot(r)

        plt.savefig('./5.1 Results/' + mmd + '/' + mmd + '/train_error_v4.tif')



    else:

        print('load model')

        model.model=models.load_model('./5.1 Results/'+mmd+'/'+mmd+'/model_v2.h5')



    '''

    Training set test

    '''

    MSE,times=[],[]

    umax = np.load('./5.1 Results/Umax.npy')

    umin = np.load('./5.1 Results/Umin.npy')

    vmax = np.load('./5.1 Results/Vmax.npy')

    vmin = np.load('./5.1 Results/Vmin.npy')

    for t in selected:

        print('test:',t)

        t1=time.time()



        Utest=t_DATA_U[t].reshape((1,-1))

        Vtest=t_DATA_V[t].reshape((1,-1))

        Ftest=t_DATA_F[t+1].reshape((1,-1))

        Utest_y = t_DATA_U[t + 1].reshape((1, -1))

        Vtest_y = t_DATA_V[t + 1].reshape((1, -1))

        Utest_out1 = t_DATA_U[t + 2].reshape((1, -1))

        Vtest_out1 = t_DATA_V[t + 2].reshape((1, -1))

        Utest_out2 = t_DATA_U[t + 3].reshape((1, -1))

        Vtest_out2 = t_DATA_V[t + 3].reshape((1, -1))

        Utest_out3 = t_DATA_U[t + 4].reshape((1, -1))

        Vtest_out3 = t_DATA_V[t + 4].reshape((1, -1))





        UYtt1, UYtt2, UYtt3, VYtt1, VYtt2, VYtt3 = model.model.predict([Utest, Vtest, Ftest, Utest_y, Vtest_y])

        Uytt1 = dp.renormalize(UYtt1, umax, umin)

        Vytt1 = dp.renormalize(VYtt1, vmax, vmin)

        UYstand1 = dp.renormalize(Utest_out1, umax, umin)

        VYstand1 = dp.renormalize(Vtest_out1, vmax, vmin)



        Uytt2 = dp.renormalize(UYtt2, umax, umin)

        Vytt2 = dp.renormalize(VYtt2, vmax, vmin)

        UYstand2 = dp.renormalize(Utest_out2, umax, umin)

        VYstand2 = dp.renormalize(Vtest_out2, vmax, vmin)



        Uytt3 = dp.renormalize(UYtt3, umax, umin)

        Vytt3 = dp.renormalize(VYtt3, vmax, vmin)

        UYstand3 = dp.renormalize(Utest_out3, umax, umin)

        VYstand3 = dp.renormalize(Vtest_out3, vmax, vmin)



        np.save('./5.1 Results/' + mmd + '/results U/sim results/steps'+str(t)+'_1.npy',Uytt1[0])

        np.save('./5.1 Results/' + mmd + '/results V/sim results/steps' + str(t) + '_1.npy', Vytt1[0])

        np.save('./5.1 Results/' + mmd + '/results U/sim results/Normalized_steps' + str(t) + '_1.npy', UYtt1[0])

        np.save('./5.1 Results/' + mmd + '/results V/sim results/Normalized_steps' + str(t) + '_1.npy', VYtt1[0])



        np.save('./5.1 Results/' + mmd + '/results U/sim results/steps' + str(t) + '_2.npy', Uytt2[0])

        np.save('./5.1 Results/' + mmd + '/results V/sim results/steps' + str(t) + '_2.npy', Vytt2[0])

        np.save('./5.1 Results/' + mmd + '/results U/sim results/Normalized_steps' + str(t) + '_2.npy', UYtt2[0])

        np.save('./5.1 Results/' + mmd + '/results V/sim results/Normalized_steps' + str(t) + '_2.npy', VYtt2[0])



        np.save('./5.1 Results/' + mmd + '/results U/sim results/steps' + str(t) + '_3.npy', Uytt3[0])

        np.save('./5.1 Results/' + mmd + '/results V/sim results/steps' + str(t) + '_3.npy', Vytt3[0])

        np.save('./5.1 Results/' + mmd + '/results U/sim results/Normalized_steps' + str(t) + '_3.npy', UYtt3[0])

        np.save('./5.1 Results/' + mmd + '/results V/sim results/Normalized_steps' + str(t) + '_3.npy', VYtt3[0])

        t2=time.time()



        #MSE.append(np.sqrt(np.mean(np.square(Ytt-test_out))))

        #times.append(t2-t1)



        #draw and save figures and MSE and computing time

        for si in ['1','2','3']:

            if si == '1':

                UYstand,Uytt,VYstand,Vytt,Utest_out,UYtt,Vtest_out,VYtt = UYstand1,Uytt1,VYstand1,Vytt1,Utest_out1,UYtt1,Vtest_out1,VYtt1

            elif si == '2':

                UYstand, Uytt, VYstand, Vytt, Utest_out, UYtt, Vtest_out, VYtt = UYstand2, Uytt2, VYstand2, Vytt2, Utest_out2, UYtt2, Vtest_out2, VYtt2

            else:

                UYstand, Uytt, VYstand, Vytt, Utest_out, UYtt, Vtest_out, VYtt = UYstand3, Uytt3, VYstand3, Vytt3, Utest_out3, UYtt3, Vtest_out3, VYtt3

            plt.figure()

            plt.plot(UYstand[0],'b:',label='True')

            plt.plot(Uytt[0],'r',label='Prediction')

            plt.legend()

            plt.savefig('./5.1 Results/' + mmd + '/results U/Allpoints_U_2019_steps' + str(t) + '_' +si+'.tif')

            plt.figure()

            plt.plot(VYstand[0], 'b:', label='True')

            plt.plot(Vytt[0], 'r', label='Prediction')

            plt.legend()

            plt.savefig('./5.1 Results/' + mmd + '/results V/Allpoints_V_2019_steps' + str(t) + '_'+si+'.tif')

            plt.figure()

            plt.plot(Utest_out[0], 'b:', label='True')

            plt.plot(UYtt[0], 'r', label='Prediction')

            plt.legend()

            plt.savefig('./5.1 Results/' + mmd + '/results U/Norm_Allpoints_U_2019_steps' + str(t) + '_'+si+'.tif')

            plt.figure()

            plt.plot(Vtest_out[0], 'b:', label='True')

            plt.plot(VYtt[0], 'r', label='Prediction')

            plt.legend()

            plt.savefig('./5.1 Results/' + mmd + '/results V/Norm_Allpoints_V_2019_steps' + str(t) + '_'+si+'.tif')



            plt.clf()

            plt.close()