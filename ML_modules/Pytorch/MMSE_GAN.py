import numpy as np
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets, models
from torch import Tensor

import visdom
import math
import matplotlib.pyplot as plt 
import scipy
import h5py
from scipy import io as sio
from scipy.io import savemat
from scipy.io import loadmat

viz = visdom.Visdom()

#parameters
batch_size=1

print("\n\n\n\n\nCuda available:",torch.cuda.is_available(),"\n\n\n\n\n")

# Generator consists of DNN
class generator(nn.Module):
    
    # Weight Initialization [we initialize weights here]
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)

        nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.bias)
        nn.init.xavier_uniform_(self.out.bias)

    def __init__(self, G_in, G_out, w1, w2, w3):
        super(generator, self).__init__()
        
        self.fc1= nn.Linear(G_in, w1)
        self.fc2= nn.Linear(w1, w2)
        self.fc3= nn.Linear(w2, w3)
        self.out= nn.Linear(w3, G_out)

        # self.weight_init()
    
    # Deep neural network [you are passing data layer-to-layer]    
    def forward(self, x):
        
        # x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        # x = x.view(1, 1, 1000, 25)
        return x
        

# Discriminator also consists of DNN
class discriminator(nn.Module):

    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)
    
    def __init__(self, D_in, D_out, w1, w2, w3):
        super(discriminator, self).__init__()
        
        self.fc1= nn.Linear(D_in, w1)
        self.fc2= nn.Linear(w1, w2)
        self.fc3= nn.Linear(w2, w3)
        self.out= nn.Linear(w3, D_out)

        # self.weight_init()
        
    def forward(self, y):
        
        # y = y.view(y.size(0), -1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = F.sigmoid(self.out(y))
        return y


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

# Class for load the data into system
class speech_data(Dataset):
    
    def __init__(self, folder_path):
        self.path = folder_path
        self.files = listdir(folder_path)
        self.length = len(self.files)
        
    def __getitem__(self, index):
        d = loadmat(join(self.path, self.files[int(index)]))
        return np.array(d['Clean_cent']), np.array(d['Feat'])
    
    def __len__(self):
        return self.length

# Class for loading the testing Data
class test_speech_data(Dataset):
    
    def __init__(self, folder_path):
        self.path = folder_path
        self.files = listdir(folder_path)
        self.length = len(self.files)
        
    def __getitem__(self, index):
        # result = np.zeros((1000,275))
        d = loadmat(join(self.path, self.files[int(index)]))
        print(index)
        # result[:d.shape[0],:d.shape[1]] = d 
        return np.array(d['Feat'])	
    
    def __len__(self):
        return self.length

# Path where you want to store your results        
mainfolder = "/media/speechlab/Nirmesh_VC/mihir_savan/whisper2speech/DNN-GAN/result_DNNGAN/MCEP/modelpath_DNNGAN"

# Training Data path
traindata = speech_data(folder_path="/media/speechlab/Nirmesh_VC/mihir_savan/whisper2speech/wTIMIT_data/WHSP2SPCH_MCEP/batches/Training_complementary_feats")
train_dataloader = DataLoader(dataset=traindata, batch_size=1, shuffle=True, num_workers=2)

# Path for validation data
valdata = speech_data(folder_path="/media/speechlab/Nirmesh_VC/mihir_savan/whisper2speech/wTIMIT_data/WHSP2SPCH_MCEP/batches/Validation_complementary_feats")
val_dataloader = DataLoader(dataset=valdata, batch_size=1, shuffle=True, num_workers=2)

# Loss Functions
adversarial_loss = nn.BCELoss()
mmse_loss = nn.MSELoss()

# Initialization
Gnet = generator(40, 40, 512, 512, 512).cuda()
Dnet = discriminator(40, 1, 512, 512, 512).cuda()

# Optimizers
optimizer_G = torch.optim.Adam(Gnet.parameters(), lr=0.0001)
optimizer_D = torch.optim.Adam(Dnet.parameters(), lr=0.0001)


# Training Function
def training(data_loader, n_epochs):
    Gnet.train()
    Dnet.train()
    
    for en, (b, a) in enumerate(data_loader):
        a = Variable(a.squeeze(0)).cuda()
        b = Variable(b.squeeze(0)).cuda()

        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).cuda()

        optimizer_G.zero_grad()
        Gout = Gnet(a)
        G_loss = adversarial_loss(Dnet(Gout), valid) + mmse_loss(Gout, b)*10
        
        G_loss.backward()
        optimizer_G.step()
        
        #G_running_loss = 0
        #G_running_loss += G_loss.item()

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(Dnet(b), valid)
        fake_loss = adversarial_loss(Dnet(Gout.detach()), fake)
        D_loss = (real_loss + fake_loss) / 2

        D_loss.backward()
        optimizer_D.step()
        # D_loss = 0

        #D_running_loss = 0
        #D_running_loss += D_loss.item()
        
        print ("[Epoch: %d] [Iter: %d/%d] [D loss: %f] [G loss: %f]" % (n_epochs, en, len(data_loader), D_loss, G_loss.cpu().data.numpy()))
    


def validating(data_loader):
    Gnet.eval()
    Dnet.eval()
    Grunning_loss = 0
    Drunning_loss = 0

    valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False).cuda()
    fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False).cuda()
    
    for en, (b, a) in enumerate(data_loader):
        a = Variable(a.squeeze(0)).cuda()
        b = Variable(b.squeeze(0)).cuda()
        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).cuda()

        # optimizer_G.zero_grad()
        Gout = Gnet(a)
        G_loss = adversarial_loss(Dnet(Gout), valid) + mmse_loss(Gout, b)

        Grunning_loss += G_loss.item()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(Dnet(b), valid)
        fake_loss = adversarial_loss(Dnet(Gout.detach()), fake)
        D_loss = (real_loss + fake_loss) / 2

        Drunning_loss += D_loss.item()
        
    return Drunning_loss/(en+1),Grunning_loss/(en+1)
    
    
 
isTrain = True


if isTrain:
    epoch = 100
    dl_arr = []
    gl_arr = []
    for ep in range(epoch):

        training(train_dataloader, ep+1)
        if (ep+1)%5==0:
            torch.save(Gnet, join(mainfolder,"gen_g_{}_d_{}_Ep_{}.pth".format(1,1,ep+1)))
            torch.save(Dnet, join(mainfolder,"dis_g_{}_d_{}_Ep_{}.pth".format(1,1,ep+1)))
        dl,gl = validating(val_dataloader)
        print("D_loss: " + str(dl) + " G_loss: " + str(gl))
        dl_arr.append(dl)
        gl_arr.append(gl)
        if ep == 0:
            gplot = viz.line(Y=np.array([gl]), X=np.array([ep]), opts=dict(title='Generator'))
            dplot = viz.line(Y=np.array([dl]), X=np.array([ep]), opts=dict(title='Discriminator'))
        else:
            viz.line(Y=np.array([gl]), X=np.array([ep]), win=gplot, update='append')
            viz.line(Y=np.array([dl]), X=np.array([ep]), win=dplot, update='append')

            
    savemat(mainfolder+"/"+str('discriminator_loss.mat'),  mdict={'foo': dl_arr})
    savemat(mainfolder+"/"+str('generator_loss.mat'),  mdict={'foo': dl_arr})

    plt.figure(1)
    plt.plot(dl_arr)
    plt.savefig(mainfolder+'/discriminator_loss.png')
    plt.figure(2)
    plt.plot(gl_arr)
    plt.savefig(mainfolder+'/generator_loss.png')

else:
    print("Testing")
    save_folder = "/media/speechlab/Nirmesh_VC/mihir_savan/whisper2speech/pytorch_codes/models/dnn_gan_test/"
    test_folder_path="/media/speechlab/Nirmesh_VC/mihir_savan/whisper2speech/wTIMIT_data/WHSP2SPCH_MCEP/batches/Testing_complementary_feats"
    n = len(listdir(test_folder_path))
    Gnet = torch.load(join(mainfolder,"gen_g_10_d_1_Ep_100.pth"))
    for i in range(n):
        d = loadmat(join(test_folder_path, "Test_Batch_{}.mat".format(str(i))))
        a = torch.from_numpy(d['Feat'])
        a = Variable(a.squeeze(0).type('torch.FloatTensor')).cuda()
        Gout = Gnet(a)
        # np.save(join(save_folder,'Test_Batch_{}.npy'.format(str(i))), Gout.cpu().data.numpy())
        savemat(join(save_folder,'Test_Batch_{}.mat'.format(str(i))),  mdict={'foo': Gout.cpu().data.numpy()})
            
