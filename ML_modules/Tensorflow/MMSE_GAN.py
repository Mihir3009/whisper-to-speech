import os, sys
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt 
import random
import scipy
import h5py
from scipy import io as sio
from scipy.io import savemat
from scipy.io import loadmat

###########################################  make directory ############################################
mainfolder = "/media/speechlab/Nirmesh_New/NAM2SPCH_Uniform/WHSP2SPCH/GAN/symmetric/4context/MCEP"   # main folder where you have all codes

directory_spectrum = mainfolder+'/test_spectrum_new'   # choose a folder where you want to save your test files
if not os.path.exists(directory_spectrum):
    os.makedirs(directory_spectrum)

directory_model = mainfolder+'/model_pathGAN_new'   # choose a folder where you want to save your model files
if not os.path.exists(directory_model):
    os.makedirs(directory_model)

loadtrainingpath = "/media/speechlab/Nirmesh_New/NAM2SPCH_Uniform/WHSP2SPCH/Training_Testing_data/MCEP/Training_complementary_feats"
loadvalidationpath =  "/media/speechlab/Nirmesh_New/NAM2SPCH_Uniform/WHSP2SPCH/Training_Testing_data/MCEP/Validation_complementary_feats"

start= 25
end= 250

data = sio.loadmat(loadtrainingpath + "/Batch_0.mat")
ip_full = data['Feat'] #11 context  1000 X 275
ip = ip_full[:,start:end] #no context 1000 X 25 [define context]
op = data['Clean_cent']

############################################### parameters ###############################################
n_input = ip.shape[1] 
n_output = op.shape[1]

print ("input: " + str(n_input)) 
print ("output: " + str(n_output))

n_hidden1_gen = 512 
n_hidden2_gen = 512 
n_hidden3_gen = 512 

n_hidden1_dis = 512 
n_hidden2_dis = 512 
n_hidden3_dis = 512 
learning_rate = 0.001

training_epochs = 100   # define how many epochs you want to train
training_batches = 1242 # how many training batches you have
validation_batches = 200 # how many validation batches you have
VALIDATION_RMSE = []  # to store validation error for normal speech in every epoch


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)*0.01

###########################################################################################################
x = tf.placeholder(tf.float32, [None,n_input], name="whisperbatch")  # ip to generator [whisper signal]
y_ = tf.placeholder(tf.float32, [None,n_output], name="centralnormalframe") # actual normal signal central spectrum, MSE loss at G n/w
MEAN = tf.placeholder(tf.float32, [1])
STD = tf.placeholder(tf.float32, [1])

weights = {
    'Gw1': tf.Variable(tf.random_normal([n_input, n_hidden1_gen])*0.0001, name="Gw1"),
    'Gw2': tf.Variable(tf.random_normal([n_hidden1_gen, n_hidden2_gen])*0.0001, name="Gw2"),
    'Gw3': tf.Variable(tf.random_normal([n_hidden2_gen, n_hidden3_gen])*0.0001, name="Gw3"),
    'Gwout': tf.Variable(tf.random_normal([n_hidden3_gen, n_output])*0.0001, name="Gwout"),

    'Dw1': tf.Variable(xavier_init([n_output, n_hidden1_dis])*0.001, name="Dw1"),
    'Dw2': tf.Variable(xavier_init([n_hidden1_dis, n_hidden2_dis])*0.001, name="Dw2"),
    'Dw3': tf.Variable(xavier_init([n_hidden2_dis, n_hidden3_dis])*0.001, name="Dw3"),
    'Dwout': tf.Variable(xavier_init([n_hidden3_dis, n_output])*0.001, name="Dwout")
}

biases = {
    'Gb1': tf.Variable(tf.random_normal(shape=[n_hidden1_gen])*0.01, name="Gb1"),
    'Gb2': tf.Variable(tf.random_normal(shape=[n_hidden2_gen])*0.01, name="Gb2"),
    'Gb3': tf.Variable(tf.random_normal(shape=[n_hidden3_gen])*0.01, name="Gb3"),
    'Gbout': tf.Variable(tf.random_normal(shape=[n_output])*0.01, name="Gbout"),

    'Db1': tf.Variable(tf.zeros(shape=[n_hidden1_dis]), name="Db1"),
    'Db2': tf.Variable(tf.zeros(shape=[n_hidden2_dis]), name="Db2"),
    'Db3': tf.Variable(tf.zeros(shape=[n_hidden3_dis]), name="Db3"),
    'Dbout': tf.Variable(tf.zeros(shape=[n_output]), name="Dbout")
}

theta_G = [weights['Gw1'], weights['Gw2'], weights['Gw3'], weights['Gwout'], 
            biases['Gb1'], biases['Gb2'], biases['Gb3'], biases['Gbout']]
theta_D = [weights['Dw1'], weights['Dw2'], weights['Dw3'], weights['Dwout'], 
            biases['Db1'], biases['Db2'], biases['Db3'], biases['Dbout']]

def generator(x, weights, biases):    
    layer_1 = tf.nn.relu(tf.add(tf.matmul((x), weights['Gw1']), biases['Gb1'])) # 500x512
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['Gw2']), biases['Gb2'])) # 500x512
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['Gw3']), biases['Gb3'])) # 500x512
    esti_spec = tf.add(tf.matmul(layer_3, weights['Gwout']), biases['Gbout'], name='mask') # 500x64
    return esti_spec

def discriminator(X):
    layer_1 = tf.nn.tanh(tf.matmul(X, weights['Dw1']) + biases['Db1'])
    layer_2 = tf.nn.tanh(tf.matmul(layer_1, weights['Dw2']) + biases['Db2'])
    layer_3 = tf.nn.tanh(tf.matmul(layer_2, weights['Dw3']) + biases['Db3'])
    logit = tf.matmul(layer_3, weights['Dwout']) + biases['Dbout']
    prob = tf.nn.sigmoid(logit, name="D_prob")
    return prob, logit

# construct the model
esti_spec = generator(x, weights, biases) # noisy context ip to G
D_real, D_logit_real = discriminator((y_-MEAN)/STD)         # actual clean central ip to D
D_fake, D_logit_fake = discriminator((esti_spec-MEAN)/STD) # estimated clean central ip to D

# calculate the loss
D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_real), logits=D_logit_real)
D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_logit_fake), logits=D_logit_fake)

D_loss = tf.reduce_mean(D_loss_real) + tf.reduce_mean(D_loss_fake)

G_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_fake), logits=D_logit_fake))
G_loss = G_gan + G_RMSE

D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=theta_G)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

################################################### TRAINING ################################################################
k = 0
model_path = directory_model + "/model" + str(k) + ".ckpt"

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, model_path)

    for epoch in range(0,training_epochs):
        saver.restore(sess, model_path)

        Rand_files = np.random.permutation(training_batches)
        batch_index = 0
        for batch in Rand_files:

            data = sio.loadmat(loadtrainingpath + "/Batch_" + str(batch)+".mat")          
            
            batch_whisper_full = data['Feat']                      
            batch_whisper = batch_whisper_full[:,start:end] #no context
            batch_normal = data['Clean_cent'] 
            
            mean = np.array([np.mean(batch_normal[:,:])])
            std = np.array([np.std(batch_normal[:,:])])
            
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={x: batch_whisper, y_: batch_normal, MEAN: mean, STD: std})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={x: batch_whisper, y_: batch_normal, MEAN: mean, STD: std})
            G_RMSE_curr = sess.run(G_RMSE, feed_dict={x: batch_whisper, y_: batch_normal, MEAN: mean, STD: std})

            print("Epoch: "+str(epoch)+" Batch_index: "+str(batch_index)+" D_Cost= "+str(D_loss_curr)+ " G_cost= "+str(G_loss_curr)+ " G_rmse= "+str(G_RMSE_curr))
            batch_index = batch_index+1
            
        ################################################### validation ################################################################   
        RMSE = []
        for v_speech in range(0,validation_batches): 
            data = sio.loadmat(loadvalidationpath + "/Test_Batch_" + str(v_speech)+".mat") 

            batch_whisper_full = data['Feat']                      
            batch_whisper = batch_whisper_full[:,start:end] #no context
            batch_normal = data['Clean_cent'] 
            
            mean = np.array([np.mean(batch_normal[:,:])])
            std = np.array([np.std(batch_normal[:,:])])
           
            G_RMSE_curr = sess.run(G_RMSE, feed_dict={x: batch_whisper, y_: batch_normal, MEAN: mean, STD: std})
            RMSE.append(G_RMSE_curr)

        print("After epoch "+str(epoch)+" Validation error is "+str(np.average(RMSE)))
        VALIDATION_RMSE.append(np.average(RMSE))
   
        k = k+1
        model_path = directory_model + "/model" + str(k) + ".ckpt"
        save_path = saver.save(sess, model_path)


scipy.io.savemat(mainfolder+"/"+str('Validation_errorganrmse_new.mat'),  mdict={'foo': VALIDATION_RMSE})
plt.figure(1)
plt.plot(VALIDATION_RMSE)
plt.savefig(mainfolder+'/validationerrorganrmse_new.png')
