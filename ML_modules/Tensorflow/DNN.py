import os
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt 
import random
import scipy
from scipy import io as sio
from scipy.io import savemat

###########################################  make directory ############################################
mainfolder = "E:\\Neil_interspeech/Nam_murmur"   # main folder where you have all codes

directory_spectrum = mainfolder+'test_spectrum'   # choose a folder where you want to save your test files
if not os.path.exists(directory_spectrum):
    os.makedirs(directory_spectrum)

directory_model = mainfolder+'model_pathDNN'   # choose a folder where you want to save your model files
if not os.path.exists(directory_model):
    os.makedirs(directory_model)

loadtrainingpath = ""   # load [path] your training data
loadvalidationpath = ""   # load [path] your validation data

data = sio.loadmat(loadtrainingpath + "/Batch_0.mat")

ip_full = data['Feat'] #11-context
print(ip_full.shape)        
ip = ip_full[:,125:150] #no context 1000 X 25 [define context]
print(ip.shape) 
op = data['Clean_cent']

############################################## parameters #############################################
n_input = ip.shape[1] # input shape [here 25]
n_hidden1 = 512 
n_hidden2 = 512 
n_hidden3 = 512 
n_output = op.shape[1] # output shape [here 25]
learning_rate = 0.001

training_epochs = 100   # define how many epochs you want to train
training_batches = 1242  # how many training batches you have
validation_batches = 200 # how many validation batches you have
VALIDATION_RMSE = [] # to store validation error for normal speech in every epoch

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)*0.01

#######################################################################################
# Placeholders
x = tf.placeholder(tf.float32, [None,n_input], name="whisperbatch") # b_s x ip
y_ = tf.placeholder(tf.float32, [None,n_output], name="centralnormalframe") # b_s x op

# Create model
def FFN(x, weights, biases):    
    layer_1 = tf.add(tf.matmul((x), weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    
    y = tf.add(tf.matmul(layer_3, weights['out']), biases['out']) # linear activation
    return y

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])*.0001),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])*.0001),
    'h3': tf.Variable(tf.random_normal([n_hidden2, n_hidden3])*.0001),
    'out': tf.Variable(tf.random_normal([n_hidden3, n_output])*.0001)
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])*0.01),
    'b2': tf.Variable(tf.random_normal([n_hidden2])*0.01),
    'b3': tf.Variable(tf.random_normal([n_hidden3])*0.01),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Construct model
y = FFN(x, weights, biases)

# compute cross entropy as our loss function
cost = 0.5*(tf.reduce_mean(tf.square(tf.subtract(y_, y))))

# use GD as optimizer to train network
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize all variables
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=None)

################################################### TRAINING ################################################################
k = 0
model_path = directory_model + "/" + "model" + str(k) + ".ckpt"

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, model_path)

    for epoch in range(0,training_epochs):
        saver.restore(sess, model_path)

        Rand_files = np.random.permutation(training_batches)
        batch_index = 0
        for batch in Rand_files:

            data = sio.loadmat(loadtrainingpath + "/Batch_" + str(batch)+".mat")          
            
            batch_whisper_full = data['Feat'] # log-features, input total 5 context, making 64*5 dim input
            batch_whisper = batch_whisper_full[:,125:150]
            batch_normal = data['Clean_cent'] # log-true labels

            costs,_ = sess.run([cost,optimizer], feed_dict={x: batch_whisper, y_: batch_normal})

            print("Epoch: "+str(epoch)+" Batch_index: "+str(batch_index)+" Cost= "+str(costs))
            batch_index = batch_index+1

        ################################################### validation ################################################################   
        RMSE = []
        for v_speech in range(0,validation_batches): 
            data = sio.loadmat(loadvalidationpath + "/Test_Batch_" + str(v_speech)+".mat") 

            batch_whisper_full = data['Feat'] # log-features, input total 5 context, making 64*5 dim input
            batch_whisper = batch_whisper_full[:,125:150]
            batch_normal = data['Clean_cent'] # log-true labels

            costs,_ = sess.run([cost,optimizer], feed_dict={x: batch_whisper, y_: batch_normal})
            RMSE.append(costs)

        print("After epoch "+str(epoch)+" Validation error is "+str(np.average(RMSE)))
        VALIDATION_RMSE.append(np.average(RMSE))
   
        k = k+1
        model_path = directory_model + "/model" + str(k) + ".ckpt"
        save_path = saver.save(sess, model_path)

######################## save validation results ##############################################################
scipy.io.savemat(mainfolder+"/"+str('Validation_errorDNN.mat'),  mdict={'foo': VALIDATION_RMSE})
plt.figure(1)
plt.plot(VALIDATION_RMSE)
plt.savefig(mainfolder+"/validationerrorDNN.png")
plt.show()
