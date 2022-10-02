import torch
from torch.autograd import Variable
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch import nn,optim
import numpy as np
import sys

from VAE_MF import train_dataset,validation_dataset,VAE,testing
device=torch.device('cuda',0)

def KL_divergences(model,data):
    KL=0
    for x in data:
        mean,logvar=model.encode(x)
        kl=0.5*torch.sum((mean**2+torch.exp(logvar)-logvar-1),0)
        KL+=kl/len(x)
    return KL/len(data)

def ES(model,data):
    Scores=[]
    for i in range(model.dim_z):
        Score=0
        pi=torch.ones(model.dim_z).to(device)
        pi[i]=0
        pi=Variable(pi,requires_grad=True)
        for x in data:
            mean,logvar=model.encode(x)
            z=model.reparametrize(mean,logvar)
            fake1=model.decode(z*pi)
            fake2=model.decode(z)
            mse_function=nn.MSELoss(reduction='sum')
            s1=mse_function(fake1,x)
            s2=mse_function(fake2,x)
            Score+=(s2-s1)/len(x)
        Score=Score/len(data)
        Scores.append(Score)
    return Scores

def generate_subdata(train_dataset,model,sub_size):
    data=[]
    for img,_ in train_dataset:
        img=img.view(img.shape[0],-1)
        img=img.to(device)
        data.append(img)
    data=data[:sub_size]
    return data

def k_index(score,k):
    score_index=[]
    for i,s in enumerate(score):
        score_index.append((i,s))
    score_index=sorted(score_index,key=lambda x:x[1],reverse=False)
    index=[score_index[i][0] for i in range(len(score_index)-k,len(score_index))]
    return index

def Eliminated_Score(model,data):
    KL=KL_divergences(model, data)
    MSE=ES(model,data)
    MSE=torch.Tensor(MSE).to(device)
    return abs(MSE)+KL

def Feature_Selection(model,data,k):
    LS=Eliminated_Score(model,data)
    index=k_index(LS,k)
    print(LS)
    return index
