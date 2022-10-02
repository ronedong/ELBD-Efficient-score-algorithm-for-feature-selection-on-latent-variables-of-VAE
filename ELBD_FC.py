import torch
from torch.autograd import Variable
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch import nn,optim
import numpy as np
import sys
from VAE_FC import train_dataset,validation_dataset,VAE,testing

device=torch.device('cuda',0)

def batch_trace(batch_size,L):
    trace=torch.zeros(batch_size,L.shape[1]).to(device)
    for i in range(batch_size):
        trace[i]=torch.trace(L[i])
    return trace

def exchange(Sigma,i,j):
    a=torch.clone(Sigma[:,i,:])
    Sigma[:,i,:]=Sigma[:,j,:]
    Sigma[:,j,:]=a
    b=torch.clone(Sigma[:,:,i])
    Sigma[:,:,i]=Sigma[:,:,j]
    Sigma[:,:,j]=b
    return Sigma

def conditional_dist(del_features,z,mean,Sigma):
    n=Sigma.shape[1]
    m=len(del_features)
    mean0=torch.zeros(mean.shape).to(device)
    z0=torch.zeros(z.shape).to(device)
    i,j=0,n-1
    for k in range(n):
        if k in del_features:
            Sigma=exchange(Sigma,i,k)
            mean0[:,i]=mean[:,k]
            z0[:,:,i]=z[:,:,k]
            i+=1
        else:
            Sigma=exchange(Sigma,j,k)
            mean0[:,j]=mean[:,k]
            z0[:,:,j]=z[:,:,k]
            j-=1
    Sigma0=torch.inverse(Sigma)
    Sigma00=Sigma0[:,:m,:m]
    Sigma_uw=torch.inverse(Sigma00)
    Sigma10=Sigma0[:,m:,:m]
    mean_u=mean0[:,:m]
    mean_w=mean0[:,m:]
    mean_u=torch.unsqueeze(mean_u,1)
    mean_w=torch.unsqueeze(mean_w,1)
    mean_uw=mean_u-torch.bmm(torch.bmm(z0[:,:,m:]-mean_w,Sigma10),Sigma_uw)
    mean_uw=mean_uw.view(mean_uw.shape[0],m)
    return Sigma_uw,mean_uw

def KL_divergences(model,data):
    KL=[]
    for i in range(model.dim_z):
        kl=0
        for x in data:
            mean,log_sigma,L1=model.encode(x)
            L=model.L(log_sigma,L1)
            z,_=model.reparametrize(mean,L)
            L_T=torch.transpose(L,1,2)
            Sigma=torch.bmm(L_T,L).to(device)
            Sigma_uw,mean_uw=conditional_dist([i],z,mean,Sigma)
            var_wz=batch_trace(len(x),Sigma_uw)
            kl+=torch.sum(0.5*(mean_uw**2+var_wz-1))-torch.sum(0.5*torch.log(torch.det(Sigma_uw)))
            kl=kl/len(x)
        kl=kl/len(data)
        KL.append(kl)
    KL=torch.Tensor(KL).to(device)
    return KL

def mutual_information(model,data):
    MI=[]
    for i in range(model.dim_z):
        mi=0
        for x in data:
            mean,log_sigma,L1=model.encode(x)
            L=model.L(log_sigma,L1)
            z,_=model.reparametrize(mean,L)
            L_T=torch.transpose(L,1,2)
            Sigma=torch.bmm(L_T,L).to(device)
            z=z.squeeze(1)
            mean[:,0],mean[:,i]=mean[:,i],mean[:,0]
            z[:,0],z[:,i]=z[:,i],z[:,0]
            Sigma=exchange(Sigma,i,0)
            Sigma_ww=Sigma[:,1:,1:]
            Sigma_uu=Sigma[:,:1,:1]
            z=z.unsqueeze(1)
            mean=mean.unsqueeze(1)
            z_T=z.transpose(1,2)
            mean_T=mean.transpose(1,2)
            Q_z=-0.5*torch.sum(torch.bmm(torch.bmm(z-mean,torch.inverse(Sigma)),z_T-mean_T))
            Q_w=-0.5*torch.sum(torch.bmm(torch.bmm(z[:,:,1:]-mean[:,:,1:],torch.inverse(Sigma_ww)),
                                         z_T[:,1:,:]-mean_T[:,1:,:]))
            Q_u=-0.5*torch.sum(torch.bmm(torch.bmm(z[:,:,:1]-mean[:,:,:1],torch.inverse(Sigma_uu)),
                                         z_T[:,:1,:]-mean_T[:,:1,:]))
            mi+=(Q_z-Q_w-Q_u)/len(x)
        mi=mi/len(data)
        MI.append(mi)
    MI=torch.Tensor(MI).to(device)
    return MI
            
def ES(model,data):
    Scores=[]
    for i in range(model.dim_z):
        Score=0
        pi=torch.ones(model.dim_z).to(device)
        pi[i]=0
        pi=Variable(pi,requires_grad=True)
        pi_T=pi.view(-1,1)
        for x in data:
            mean,log_sigma,L1=model.encode(x)
            L=model.L(log_sigma,L1)
            z,_=model.reparametrize(mean,L)
            fake1=model.decode(z*pi)
            fake2=model.decode(z)
            mse_function=nn.MSELoss(reduction='sum')
            s1=mse_function(fake1,x)
            s2=mse_function(fake2,x)
            Score+=(s2-s1)/len(x)
        Score=Score/len(data)
        Scores.append(Score)
    return Scores

def generate_subdata(train_dataset,sub_size):
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
    MI=mutual_information(model,data)
    return abs(MSE)+KL-MI

def Feature_Selection(model,data,k):
    LS=Eliminated_Score(model,data)
    index=k_index(LS,k)
    print(LS)
    return index



