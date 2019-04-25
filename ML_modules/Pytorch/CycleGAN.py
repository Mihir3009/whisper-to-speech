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
import itertools

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
    
    # Deep neural network [you are passing data layer-to-layer]
    def forward(self, x):
        
        # x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        
        return x


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
        return  np.array(d['Feat']), np.array(d['Clean_cent'])
    
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
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

#I/O variables
in_g = 40
out_g = 40
in_d = 40
out_d = 1
learning_rate = 0.0001


# Initialization
Gnet_ws = generator(in_g, out_g, 512, 512, 512).cuda()
Gnet_sw = generator(in_g, out_g, 512, 512, 512).cuda()
Dnet_w = discriminator(in_d, out_d, 512, 512, 512).cuda()
Dnet_s = discriminator(in_d, out_d, 512, 512, 512).cuda()


# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(Gnet_ws.parameters(), Gnet_sw.parameters()),lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_w = torch.optim.Adam(Dnet_w.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_s = torch.optim.Adam(Dnet_s.parameters(), lr=learning_rate, betas=(0.5, 0.999))


# Training Function
def training(data_loader, n_epochs):
    Gnet_ws.train()
    Gnet_sw.train()
    Dnet_w.train()
    Dnet_s.train()
    
    for en, (a, b) in enumerate(data_loader):
        a = Variable(a.squeeze(0)).cuda()
        b = Variable(b.squeeze(0)).cuda()

        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).cuda()
        
        
        
        
        ###### Generators W2S and S2W ######
        optimizer_G.zero_grad()
        
        # Identity loss
        # G_W2S(S) should equal S if real S is fed
        same_s = Gnet_ws(b)
        loss_identity_s = criterion_identity(same_s, b)*5.0
        # G_S2W(W) should equal W if real W is fed
        same_w = Gnet_sw(a)
        loss_identity_w = criterion_identity(same_w, a)*5.0

        # GAN loss
        Gout_ws = Gnet_ws(a)
        loss_GAN_W2S = criterion_GAN(Dnet_s(Gout_ws), valid)
        
        Gout_sw = Gnet_sw(b)
        loss_GAN_S2W = criterion_GAN(Dnet_w(Gout_sw), valid)
        
        # Cycle loss
        recovered_W = Gnet_sw(Gout_ws)
        loss_cycle_WSW = criterion_cycle(recovered_W, a)*10.0
        
        recovered_B = Gnet_ws(Gout_sw)
        loss_cycle_SWS = criterion_cycle(recovered_S, b)*10.0
        
        # Total loss
        loss_G =  loss_identity_w + loss_identity_s + loss_GAN_W2S + loss_GAN_S2W + loss_cycle_WSW + loss_cycle_SWS
        loss_G.backward()
        
        optimizer_G.step()

        
        
#        Gout = Gnet(a)
#        G_loss = adversarial_loss(Dnet(Gout), valid) + mmse_loss(Gout, b)*10
#
#        G_loss.backward()
#        optimizer_G.step()
        
        
        ###### Discriminator W ######
        optimizer_D_w.zero_grad()

        # Real loss
        loss_D_real = criterion_GAN(Dnet_w(a), valid)
        
        # Fake loss
        loss_D_fake = criterion_GAN(Dnet_w(Gout_ws.detach()), fake)
        
        # Total loss
        loss_D_W = (loss_D_real + loss_D_fake)*0.5
        loss_D_w.backward()
        
        optimizer_D_w.step()
        
        ###################################
        
        ###### Discriminator B ######
        optimizer_D_s.zero_grad()
        
        # Real loss
        loss_D_real = criterion_GAN(Dnet_s(b), valid)
        
        # Fake loss
        loss_D_fake = criterion_GAN(Dnet_s(Gout_ws.detach()), fake)
        
        # Total loss
        loss_D_s = (loss_D_real + loss_D_fake)*0.5
        loss_D_s.backward()
        
        optimizer_D_s.step()
        ###################################
        
        
        
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
    Gnet_ws.eval()
    Gnet_sw.eval()
    Dnet_s.eval()
    Dnet_w.eval()
    Grunning_loss = 0
    Drunning_loss = 0

    
    for en, (b, a) in enumerate(data_loader):
        a = Variable(a.squeeze(0)).cuda()
        b = Variable(b.squeeze(0)).cuda()
        
        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).cuda()

        ###### Generators W2S and S2W ######
        
        # Identity loss
        # G_W2S(S) should equal S if real S is fed
        same_s = Gnet_ws(b)
        loss_identity_s = criterion_identity(same_s, b)*5.0
        # G_S2W(W) should equal W if real W is fed
        same_w = Gnet_sw(a)
        loss_identity_w = criterion_identity(same_w, a)*5.0
        
        # GAN loss
        Gout_ws = Gnet_ws(a)
        loss_GAN_W2S = criterion_GAN(Dnet_s(Gout_ws), valid)
        
        Gout_sw = Gnet_sw(b)
        loss_GAN_S2W = criterion_GAN(Dnet_w(Gout_sw), valid)
        
        # Cycle loss
        recovered_W = Gnet_sw(Gout_ws)
        loss_cycle_WSW = criterion_cycle(recovered_W, a)*10.0
        
        recovered_B = Gnet_ws(Gout_sw)
        loss_cycle_SWS = criterion_cycle(recovered_S, b)*10.0
        
        # Total loss
        loss_G =  loss_identity_w + loss_identity_s + loss_GAN_W2S + loss_GAN_S2W + loss_cycle_WSW + loss_cycle_SWS

        
        
        ###### Discriminator W ######
        optimizer_D_w.zero_grad()
        
        # Real loss
        loss_D_real = criterion_GAN(Dnet_w(a), valid)
        
        # Fake loss
        loss_D_fake = criterion_GAN(Dnet_w(Gout_ws.detach()), fake)
        
        # Total loss
        loss_D_w = (loss_D_real + loss_D_fake)*0.5

        
        ###### Discriminator B ######
        optimizer_D_s.zero_grad()
        
        # Real loss
        loss_D_real = criterion_GAN(Dnet_s(b), valid)
        
        # Fake loss
        loss_D_fake = criterion_GAN(Dnet_s(Gout_ws.detach()), fake)
        
        # Total loss
        loss_D_s = (loss_D_real + loss_D_fake)*0.5
 
        ###################################
        

        Grunning_loss += loss_G.item()

        Drunning_loss += loss_D.item()
        
    return Drunning_loss/(en+1),Grunning_loss/(en+1)
    
    
 
isTrain = True


if isTrain:
    epoch = 100
    dl_arr = []
    gl_arr = []
    for ep in range(epoch):

        training(train_dataloader, ep+1)
        if (ep+1)%5==0:
            torch.save(Gnet, join(mainfolder,"gen_wsEp_{}.pth".format(ep+1)))
        #torch.save(Dnet, join(mainfolder,"dis_g_{}_d_{}_Ep_{}.pth".format(1,1,ep+1)))
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
    savemat(mainfolder+"/"+str('generator_loss.mat'),  mdict={'foo': gl_arr})

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
    Gnet = torch.load(join(mainfolder,"gen_ws_Ep_100.pth"))
    for i in range(n):
        d = loadmat(join(test_folder_path, "Test_Batch_{}.mat".format(str(i))))
        a = torch.from_numpy(d['Feat'])
        a = Variable(a.squeeze(0).type('torch.FloatTensor')).cuda()
        Gout = Gnet(a)
        # np.save(join(save_folder,'Test_Batch_{}.npy'.format(str(i))), Gout.cpu().data.numpy())
        savemat(join(save_folder,'Test_Batch_{}.mat'.format(str(i))),  mdict={'foo': Gout.cpu().data.numpy()})
            
