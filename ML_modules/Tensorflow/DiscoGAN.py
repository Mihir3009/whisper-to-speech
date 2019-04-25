import os, sys
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt 
import random
import scipy
from scipy import io as sio
from scipy.io import savemat
import h5py
import pickle

###########################################  directory ############################################
mainfolder = "/media/speechlab/Nirmesh_New/Neil_Nirmesh_NAM2WHSP_Journal/Mihir_discogan/WHSP2SPCH/utterances/200k/MCEP"

directory_spectrum = mainfolder+'/test_spectrum_discoGAN'

directory_model = mainfolder+'/model_pathdiscoGAN'
loadtestingpath ="/media/speechlab/Nirmesh_New/Neil_Nirmesh_NAM2WHSP_Journal/WHSP2SPCH_MCEP/batches/Testing_complementary_feats"

# decide which model to load
model_path = directory_model + "/model" + str(100) + ".ckpt"

############################################### parameters ###############################################
# no.of noisy features
number = 108
start = 0
end = 275

data = sio.loadmat(loadtestingpath+'/Test_Batch_0.mat')
whisper_contxt_full = data['Feat']  #11-context  
whisper_contxt = whisper_contxt_full[:,start:end] #no context 1000 X 25 [define context]                       

n_hidden1_gen = 512 
n_hidden2_gen = 512 
n_hidden3_gen = 512 

n_hidden1_dis = 512 
n_hidden2_dis = 512 
n_hidden3_dis = 512 

n_input = whisper_contxt.shape[1]
n_output = 25

learning_rate = 0.001

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)*0.01

###########################################################################################################
x = tf.placeholder(tf.float32, [None,n_input], name="whisperbatch") # ip to G n/w
y_ = tf.placeholder(tf.float32, [None,n_output], name="centralnormalframe") # actual clean central spectrum, MSE loss at G n/w
MEAN = tf.placeholder(tf.float32, [1])
STD = tf.placeholder(tf.float32, [1])

weights = {
    'Gw1_w': tf.Variable(tf.random_normal([n_input, n_hidden1_gen])*0.0001, name="Gw1_w"),
    'Gw2_w': tf.Variable(tf.random_normal([n_hidden1_gen, n_hidden2_gen])*0.0001, name="Gw2_w"),
    'Gw3_w': tf.Variable(tf.random_normal([n_hidden2_gen, n_hidden3_gen])*0.0001, name="Gw3_w"),
    'Gwout_w': tf.Variable(tf.random_normal([n_hidden3_gen, n_output])*0.0001, name="Gwout_w"),
    
    'Gw1_n': tf.Variable(tf.random_normal([n_output, n_hidden1_gen])*0.0001, name="Gw1_n"),
    'Gw2_n': tf.Variable(tf.random_normal([n_hidden1_gen, n_hidden2_gen])*0.0001, name="Gw2_n"),
    'Gw3_n': tf.Variable(tf.random_normal([n_hidden2_gen, n_hidden3_gen])*0.0001, name="Gw3_n"),
    'Gwout_n': tf.Variable(tf.random_normal([n_hidden3_gen, n_input])*0.0001, name="Gwout_n"),

    'Dw1_w': tf.Variable(xavier_init([n_output, n_hidden1_dis])*0.001, name="Dw1_w"),
    'Dw2_w': tf.Variable(xavier_init([n_hidden1_dis, n_hidden2_dis])*0.001, name="Dw2_w"),
    'Dw3_w': tf.Variable(xavier_init([n_hidden2_dis, n_hidden3_dis])*0.001, name="Dw3_w"),
    'Dwout_w': tf.Variable(xavier_init([n_hidden3_dis, n_output])*0.001, name="Dwout_w"),
    
    'Dw1_n': tf.Variable(xavier_init([n_input, n_hidden1_dis])*0.001, name="Dw1_n"),
    'Dw2_n': tf.Variable(xavier_init([n_hidden1_dis, n_hidden2_dis])*0.001, name="Dw2_n"),
    'Dw3_n': tf.Variable(xavier_init([n_hidden2_dis, n_hidden3_dis])*0.001, name="Dw3_n"),
    'Dwout_n': tf.Variable(xavier_init([n_hidden3_dis, n_input])*0.001, name="Dwout_n")
}

biases = {
    'Gb1_w': tf.Variable(tf.random_normal(shape=[n_hidden1_gen])*0.01, name="Gb1_w"),
    'Gb2_w': tf.Variable(tf.random_normal(shape=[n_hidden2_gen])*0.01, name="Gb2_w"),
    'Gb3_w': tf.Variable(tf.random_normal(shape=[n_hidden3_gen])*0.01, name="Gb3_w"),
    'Gbout_w': tf.Variable(tf.random_normal(shape=[n_output])*0.01, name="Gbout_w"),
    
    'Gb1_n': tf.Variable(tf.random_normal(shape=[n_hidden1_gen])*0.01, name="Gb1_n"),
    'Gb2_n': tf.Variable(tf.random_normal(shape=[n_hidden2_gen])*0.01, name="Gb2_n"),
    'Gb3_n': tf.Variable(tf.random_normal(shape=[n_hidden3_gen])*0.01, name="Gb3_n"),
    'Gbout_n': tf.Variable(tf.random_normal(shape=[n_input])*0.01, name="Gbout_n"),

    'Db1_w': tf.Variable(tf.zeros(shape=[n_hidden1_dis]), name="Db1_w"),
    'Db2_w': tf.Variable(tf.zeros(shape=[n_hidden2_dis]), name="Db2_w"),
    'Db3_w': tf.Variable(tf.zeros(shape=[n_hidden3_dis]), name="Db3_w"),
    'Dbout_w': tf.Variable(tf.zeros(shape=[n_output]), name="Dbout_w"),

    'Db1_n': tf.Variable(tf.zeros(shape=[n_hidden1_dis]), name="Db1_n"),
    'Db2_n': tf.Variable(tf.zeros(shape=[n_hidden2_dis]), name="Db2_n"),
    'Db3_n': tf.Variable(tf.zeros(shape=[n_hidden3_dis]), name="Db3_n"),
    'Dbout_n': tf.Variable(tf.zeros(shape=[n_input]), name="Dbout_n")
}

theta_G = [weights['Gw1_w'], weights['Gw2_w'], weights['Gw3_w'], weights['Gwout_w'], weights['Gw1_n'], weights['Gw2_n'], weights['Gw3_n'],                weights['Gwout_n'], biases['Gb1_w'], biases['Gb2_w'], biases['Gb3_w'], biases['Gbout_w'], biases['Gb1_n'], biases['Gb2_n'],                    biases['Gb3_n'], biases['Gbout_n']]
theta_D = [weights['Dw1_w'], weights['Dw2_w'], weights['Dw3_w'], weights['Dwout_w'], weights['Dw1_n'], weights['Dw2_n'], weights['Dw3_n'],                weights['Dwout_n'], biases['Db1_w'], biases['Db2_w'], biases['Db3_w'], biases['Dbout_w'], biases['Db1_n'], biases['Db2_n'],                    biases['Db3_n'], biases['Dbout_n']]


def generator_WN(x, weights, biases):    
    layer_1 = tf.nn.relu(tf.add(tf.matmul((x), weights['Gw1_w']), biases['Gb1_w'])) # 1000x512
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['Gw2_w']), biases['Gb2_w'])) # 1000x512
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['Gw3_w']), biases['Gb3_w'])) # 1000x512
    esti_spec = tf.add(tf.matmul(layer_3, weights['Gwout_w']), biases['Gbout_w'], name='mask') # 1000x1
    return esti_spec

def generator_NW(y, weights, biases):    
    layer_1 = tf.nn.relu(tf.add(tf.matmul((y), weights['Gw1_n']), biases['Gb1_n'])) # 1000x512
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['Gw2_n']), biases['Gb2_n'])) # 1000x512
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['Gw3_n']), biases['Gb3_n'])) # 1000x512
    esti_spec = tf.add(tf.matmul(layer_3, weights['Gwout_n']), biases['Gbout_n'], name='mask') # 1000x25
    return esti_spec

def discriminator_W(X):
    layer_1 = tf.nn.tanh(tf.matmul(X, weights['Dw1_n']) + biases['Db1_n'])
    layer_2 = tf.nn.tanh(tf.matmul(layer_1, weights['Dw2_n']) + biases['Db2_n'])
    layer_3 = tf.nn.tanh(tf.matmul(layer_2, weights['Dw3_n']) + biases['Db3_n'])
    logit = tf.matmul(layer_3, weights['Dwout_n']) + biases['Dbout_n']
    prob = tf.nn.sigmoid(logit, name="D_prob")
    return prob, logit

def discriminator_N(Y):
    layer_1 = tf.nn.tanh(tf.matmul(Y, weights['Dw1_w']) + biases['Db1_w'])
    layer_2 = tf.nn.tanh(tf.matmul(layer_1, weights['Dw2_w']) + biases['Db2_w'])
    layer_3 = tf.nn.tanh(tf.matmul(layer_2, weights['Dw3_w']) + biases['Db3_w'])
    logit = tf.matmul(layer_3, weights['Dwout_w']) + biases['Dbout_w']
    prob = tf.nn.sigmoid(logit, name="D_prob")
    return prob, logit

# construct the model
esti_spec_WN = generator_WN(x, weights, biases) # noisy context ip to G
D_real_N, D_logit_real_N = discriminator_N((y_-MEAN)/STD)         # actual clean central ip to D
D_fake_N, D_logit_fake_N = discriminator_N((esti_spec_WN-MEAN)/STD) # estimated clean central ip to D

esti_spec_NW = generator_NW(y_, weights, biases) # noisy context ip to G
D_real_W, D_logit_real_W = discriminator_W((x-MEAN)/STD)         # actual clean central ip to D
D_fake_W, D_logit_fake_W = discriminator_W((esti_spec_NW-MEAN)/STD) # estimated clean central ip to D

recon_W = generator_NW(esti_spec_WN,weights,biases)
recon_N = generator_WN(esti_spec_NW,weights,biases)

# calculate the loss of discriminator
D_loss_real_N = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_real_N), logits=D_logit_real_N)
D_loss_fake_N = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_logit_fake_N), logits=D_logit_fake_N)

D_loss_N = tf.reduce_mean(D_loss_real_N) + tf.reduce_mean(D_loss_fake_N)

D_loss_real_W = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_real_W), logits=D_logit_real_W)
D_loss_fake_W = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_logit_fake_W), logits=D_logit_fake_W)

D_loss_W = tf.reduce_mean(D_loss_real_W) + tf.reduce_mean(D_loss_fake_W)

D_loss = D_loss_W + D_loss_N

# calculate the loss of generator
G_RMSE_N = 0.5*(tf.reduce_mean(tf.square(tf.subtract(y_, esti_spec_WN))))
G_RMSE_W = 0.5*(tf.reduce_mean(tf.square(tf.subtract(x, esti_spec_NW))))

G_gan_WN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_fake_N), logits=D_logit_fake_N))

G_gan_NW = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_fake_W), logits=D_logit_fake_W))

G_loss_WN = G_gan_WN + G_RMSE_N
G_loss_NW = G_gan_NW + G_RMSE_W

G_loss = G_loss_WN + G_loss_NW

# solvers for optimization 
D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=theta_G)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=None)

#########################################  TESTING ########################################
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, model_path)
    for i in range(0, number):

       # get the data
       data = sio.loadmat(loadtestingpath+'/Test_Batch_'+ str(i)+ '.mat')
       whisper_contxt_full = data['Feat']
       whisper_contxt = whisper_contxt_full[:,start:end]                 
       # obtain the predicted mask
       pred_spectrum = sess.run(esti_spec_WN, feed_dict={x: whisper_contxt})  

       file = directory_spectrum+ '/File_'+ str(i)+ '.mat'
       scipy.io.savemat(file,  mdict={'PRED_SPEC': pred_spectrum})
       print("file"+str(i))
