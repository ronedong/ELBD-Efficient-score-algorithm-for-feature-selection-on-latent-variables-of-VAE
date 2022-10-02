import torch
from torch import nn,optim
from torch.autograd import Variable
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os 
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import copy

batch_size=128
data_path='D:/python_work/data/'
data_tf=transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize([0.5],[0.5])])
train_dataset=datasets.MNIST(data_path,train=True,transform=data_tf,download=True)
validation_dataset=datasets.MNIST(data_path,train=False,transform=data_tf,download=True)
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=batch_size,drop_last=True)
train_dataset=train_loader
validation_dataset=DataLoader(validation_dataset,batch_size=batch_size,drop_last=True)
device=torch.device('cuda',0)

class VAE(nn.Module):
    def __init__(self,dim_z,**kwargs):
        super().__init__(**kwargs)
        self.dim_z=dim_z
        self.encode_layer1=nn.Linear(784,400)
        self.encode_layer2_1=nn.Linear(400,dim_z)
        self.encode_layer2_2=nn.Linear(400,dim_z)
        self.decode_layer1=nn.Linear(dim_z,400)
        self.decode_layer2=nn.Linear(400,784)

    def encode(self,x):
        z=F.relu(self.encode_layer1(x))
        mean=self.encode_layer2_1(z)
        logvar=self.encode_layer2_2(z)
        # mean=mean.to(device)
        # logvar=logvar.to(device)
        return mean,logvar
    
    def decode(self,z):
        h=F.relu(self.decode_layer1(z))
        x=F.tanh(self.decode_layer2(h))
        # x=x.to(device)
        return x
    
    def reparametrize(self,mean,logvar):
        # mean=mean.to(device)
        std=torch.exp(0.5*logvar)
        eps=torch.randn(self.dim_z).to(device)
        return eps*std+mean
    
    def forward(self,x,del_features,K):
        mean,logvar=self.encode(x)
        z=self.reparametrize(mean,logvar)
        if del_features==[]:
            fake_img=self.decode(z)
            return fake_img,mean,logvar
        fake_img=0
        for _ in range(K):
            z_copy=torch.clone(z)
            sample=self.reparametrize(mean,logvar)
            for f in del_features:
                z_copy[:,f]=sample[:,f]
            fake_img+=self.decode(z_copy)/K
        return fake_img,mean,logvar

def to_img(x):
    x=(x+1)*0.5
    x=x.clamp(0,1)
    x=x.view(x.size(0),1,28,28)
    return x

def loss_function(fake_x,x,mean,logvar):
    mse_function=nn.MSELoss(reduction='sum')
    mse=mse_function(fake_x,x)
    KL=torch.sum(0.5*(mean**2+torch.exp(logvar)-logvar-1))
    return mse+KL

def train_mode(n_epochs,train_loader,optimizer,model):
    for epoch in range(1,n_epochs+1):
        for img,_ in train_loader:
            img=img.to(device=device)
            img=img.view(img.shape[0],-1)
            # img=Variable(img)
            # mean,logvar=model.encode(img)
            fake_img,mean,logvar=model(img,[],1)
            loss=loss_function(fake_img,img,mean,logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch:',epoch)
        print('Loss:',loss)
    return model

def val_set_loss(validation_dataset,model,del_features,K,device):
    val_loss=0
    pi=torch.ones(model.dim_z).to(device)
    for f in del_features:
        pi[f]=0
    for img,_ in validation_dataset:
        img=img.to(device)
        img=img.view(img.shape[0],-1)
        img=Variable(img)
        fake_img,mean,logvar=model(img,del_features,K)
        logvar*=pi
        mean*=pi
        loss=loss_function(fake_img,img,mean,logvar)
        val_loss+=loss/len(img)
    val_loss=val_loss/len(validation_dataset)
    print('ELBO on validation dataset:',val_loss)
    return val_loss

def test_set_loss(test_dataset,model,del_features,K,device):
    test_loss=0
    for img,_ in test_dataset:
        img=img.to(device)
        img=img.view(img.shape[0],-1)
        fake_img,_,_=model(img,del_features,K)
        mse=nn.MSELoss(reduction='sum')
        test_loss+=mse(fake_img,img)/len(img)
    test_loss=test_loss/len(test_dataset)
    print('L2 on validation dataset:',test_loss)
    return test_loss


def testing(del_features,validation_dataset,model,K,device):
    print('origin:')
    val_set_loss(validation_dataset,model,[],K,device)
    test_set_loss(validation_dataset,model,[],K,device)
    print('after_FS:')
    ELBO=val_set_loss(validation_dataset,model,del_features,K,device)
    L2=test_set_loss(validation_dataset,model,del_features,K,device)
    return ELBO, L2

if __name__=='__main__':
    device=torch.device('cuda',0)
    learning_rate=0.001
    
    dim_z=50
    K=10
    model=VAE(dim_z)
    model.to(device)


    optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    train_mode(n_epochs=100,train_loader=train_loader,optimizer=optimizer,model=model)
    testing([],validation_dataset,model,1,device)