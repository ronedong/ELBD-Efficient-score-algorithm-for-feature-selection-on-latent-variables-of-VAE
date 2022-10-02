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

class VAE(nn.Module):
    def __init__(self,dim_z,**kwargs):
        super().__init__(**kwargs)
        self.dim_z=dim_z
        self.device=torch.device('cuda',0)
        self.encode_layer1=nn.Linear(784,400).to(self.device)
        self.encode_layer2_1=nn.Linear(400,dim_z).to(self.device)
        self.encode_layer2_2=nn.Linear(400,dim_z).to(self.device)
        self.encode_layer2_3=nn.Linear(400,dim_z*dim_z).to(self.device)
        
        self.decode_layer1=nn.Linear(dim_z,400).to(self.device)
        self.decode_layer2=nn.Linear(400,784).to(self.device)
        
    def encode(self,x):
        h=F.selu((self.encode_layer1(x)))
        mean=self.encode_layer2_1(h)
        log_sigma=self.encode_layer2_2(h)
        L1=self.encode_layer2_3(h)
        L1=L1.view(-1,self.dim_z,self.dim_z)
        L1=torch.triu(L1,diagonal=1)
        return mean,log_sigma,L1
    
    def L(self,log_sigma,L1):
        diag=torch.diag_embed(torch.exp(log_sigma))
        L=L1+diag
        return L
    
    def decode(self,z):
        h=F.relu(self.decode_layer1(z))
        x=F.tanh(self.decode_layer2(h))
        return x
    
    def reparametrize(self,mean,L):
        eps=torch.randn(mean.shape[0],1,self.dim_z).to(self.device)
        mean=mean.view(mean.shape[0],1,self.dim_z)
        z=torch.bmm(eps,L)+mean
        eps=eps.squeeze(1)
        return z,eps
    
    def forward(self,x,del_features,K):
        mean,log_sigma,L1=self.encode(x)
        L=self.L(log_sigma,L1)
        z,_=self.reparametrize(mean,L)
        fake_img=0
        if del_features==[]:
            fake_img=self.decode(z)
            return fake_img
        for _ in range(K):
            z_copy=torch.clone(z)
            sample,_=self.reparametrize(mean,L)
            for f in del_features:
                z_copy[:,:,f]=sample[:,:,f]
            fake_img+=self.decode(z_copy)/K
        return fake_img
    

def to_img(x):
    x=(x+1)*0.5
    x=x.clamp(0,1)
    x=x.view(x.size(0),1,28,28)
    return x


def loss_function(fake_x,x,log_sigma,z,eps,device):
    c=torch.log(torch.Tensor([2*torch.pi])).to(device)
    L_qz=-0.5*torch.sum(eps**2+c+log_sigma)
    L_pz=-0.5*torch.sum(z**2+c)
    mse_function=nn.MSELoss(reduction='sum')
    L_px=-0.5*mse_function(fake_x,x)
    ELBO=L_px+L_pz-L_qz
    return -ELBO


def train_mode(n_epochs,train_loader,optimizer,model):
    for epoch in range(1,n_epochs+1):
        for img,_ in train_loader:
            img=img.to(device=device)
            img=img.view(img.shape[0],-1)
            img=Variable(img)
            mean,log_sigma,L1=model.encode(img)
            L=model.L(log_sigma,L1)
            z,eps=model.reparametrize(mean,L)
            fake_img=model.decode(z)
            fake_img=fake_img.view(img.shape[0],-1)
            loss=loss_function(fake_img,img,log_sigma,z,eps,device)
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
    pi_T=pi.view(-1,1)
    for img,_ in validation_dataset:
        img=img.to(device)
        img=img.view(img.shape[0],-1)
        img=Variable(img)
        mean,log_sigma,L1=model.encode(img)
        log_sigma_pi=log_sigma*pi
        L=model.L(log_sigma,L1)
        z,eps=model.reparametrize(mean,L)
        if del_features==[]:
            fake_img=model.decode(z)
        else:
            fake_img=0
            for _ in range(K):
                z_copy=torch.clone(z)
                sample,_=model.reparametrize(mean,L)
                for f in del_features:
                    z_copy[:,:,f]=sample[:,:,f]
                fake_img+=model.decode(z_copy)/K
        fake_img=fake_img.view(img.shape[0],-1)
        log_sigma_pi=log_sigma*pi
        mean_pi=mean*pi
        L_pi=model.L(log_sigma_pi,pi*L1*pi_T)
        mean_pi=mean_pi.view(mean_pi.shape[0],1,model.dim_z)
        eps_pi=eps.unsqueeze(1)
        w=torch.bmm(eps_pi,L_pi)+mean_pi
        loss=loss_function(fake_img,img,log_sigma_pi,z,eps,device)
        val_loss+=loss/len(img)
    val_loss=val_loss/len(validation_dataset)
    print('ELBO on validation dataset:',val_loss)
    return val_loss

def test_set_loss(test_dataset,model,del_features,K,device):
    test_loss=0
    for img,_ in test_dataset:
        img=img.to(device)
        img=img.view(img.shape[0],-1)
        fake_img=model(img,del_features,K)
        fake_img=fake_img.view(img.shape)
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
    return ELBO,L2

if __name__=='__main__':
    device=(torch.device('cuda',0))
    learning_rate=0.0003
    dim_z=50
    
    model=VAE(dim_z)
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    model=train_mode(n_epochs=100,train_loader=train_loader,optimizer=optimizer,model=model)
    testing([],validation_dataset,model,1,device)