# -*- coding: utf-8 -*-


# KMEAN-BIMODAL

import torch
import sys
import torch.nn as nn
from torch.nn import functional as F
import random
import import_ipynb
import math
import matplotlib.pyplot as plt
import soundfile as sf
import IPython
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
import numpy as np
import glob
import os
os.environ["OMP_NUM_THREADS"] = "5"
!export MKL_NUM_THREADS=5
!export NUMEXPR_NUM_THREADS=5
!export OMP_NUM_THREADS=5
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans

#PARAMETERS
lr=1e-05
weight_decay=1e-06
max_epoch=6000
stepsize=30
gamma=0.1
beta=1
num_episode=20
batch=164
sr=16000
len_audio=3
BATCH=1
device='cuda:1'
random.seed(10)

#LOAD AUDIO_FEATURES
MEL_DICT=np.load('audio_dict.npy',allow_pickle=True)[()]

audio_dir="/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/audio/" #OG audio folder
audio_split_dir="/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/Audio_split/" # Splitted audio into chunks
frame_feat_dir="/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/frame_features/" # Frame features 
name_list=[]
dir_wav=audio_dir+"*.wav"
audio_lIst = glob.glob(dir_wav)
for audio in audio_lIst:
    name=os.path.basename(os.path.normpath(audio))[:-4]
    name_list.append(name)
copy_name_list=name_list
random.shuffle(copy_name_list)

train_data = copy_name_list[:int(50*0.8)]
test_data = copy_name_list[int(50*0.8):]
len(train_data)

#DICTIONARY CREATION
NEW_MEL_DICT={}
for name in name_list:
    print("okay-",name)
    mel_pername,time_regionpername,audio_pername=MEL_DICT[name]
    
    
    all_image_feat=[]
    for time_int in time_regionpername:
        image_feat=[]
        lent=0
        for j in range(time_int[0],time_int[1],sr):
            lent=lent+1
            pos=j//sr
            feat=np.load(frame_feat_dir+name+'/out'+str(pos+1)+'.npy')#adding 1 cause location starts with 1
            image_feat.append(feat)
        image_feat=np.array(image_feat).mean(0)
        all_image_feat.append(image_feat)
    all_image_feat=np.array(all_image_feat).squeeze()
    print(all_image_feat.shape,mel_pername.shape)
    NEW_MEL_DICT[name]=(mel_pername,time_regionpername,audio_pername,all_image_feat)

pca = PCA(n_components=10)
principalComponents = pca.fit_transform(NEW_MEL_DICT['EE-bNr36nyA'][0])
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component 1', 'principal component 2'])
# principalDf.head(5)
print(principalComponents.shape)
plt.scatter(principalComponents[:,0],principalComponents[:,1])
plt.show()

def gen_non_zero_list(actions): #(batch, seq_len, 1)
    batch=actions.shape[0]
    actions=actions.squeeze() #batch X seq_len
    if batch==1:
        actions=actions.unsqueeze(0) #1 X seq_len
    complete_list=[]
    for i in range(batch):
        pick_idxs = actions[i].squeeze().nonzero().squeeze() #list of indices
        complete_list.append(pick_idxs)
    return complete_list #list of list



def correlation_coeff_loss(x,device): #batch X Lent of seq
    batch=x.shape[0]
    v_x=x-x.mean(1).unsqueeze(1) #batch X Lent of seq
    v_x_norm=v_x/v_x.norm(p=2, dim=1, keepdim=True)
    correlation_mat=torch.matmul(v_x,v_x.t())#batch X batch
    inv_eyez=1-torch.eye(correlation_mat.shape[0]).to(device)
    corr=torch.sum(correlation_mat*inv_eyez)/2
    mat=(torch.matmul(v_x_norm,v_x_norm.t())*inv_eyez)
    
    corr_coeff=torch.abs((torch.matmul(v_x_norm,v_x_norm.t())*inv_eyez)).sum()/(batch*batch-batch)
    return(corr,corr_coeff,mat)

import torch.nn.functional as F
def variational_diverence_loss(x,device):#batch X Lent of seq
#     print(x.shape)
    batch=x.shape[0]
    total=0
    for i in range(batch):
        total=total+torch.log(0.00001+torch.abs(F.softmax(x,1)-F.softmax(x[i].unsqueeze(0),1)).sum())
    return(total)

def square_diverence_loss(x,device):#batch X Lent of seq
#     print(x.shape)
    batch=x.shape[0]
    total=0
    for i in range(batch):
        total=total+torch.log(0.00001+(((F.softmax(x,1))-F.softmax(x[i].unsqueeze(0),1))**2).sum())
    return(total)


def l2_dist(x,y):
    l2=(x-y)**2
    return l2.mean()

class JSD(nn.Module):
    
    def __init__(self):
        super(JSD, self).__init__()
    
    def forward(self, net_1_logits, net_2_logits):
        net_1_probs =  F.softmax(net_1_logits, dim=1)
        net_2_probs=  F.softmax(net_2_logits, dim=1)

        total_m = 0.5 * (net_1_probs + net_1_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m) 
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m) 
     
        return (0.5 * loss)
def JSD_loss(x,device): #batch X Lent of seq
    JS=JSD()
    batch=x.shape[0]
    total=0
    for i in range(batch):
        for j in range(batch):
            total=total+JS(x[i].unsqueeze(0),x[j].unsqueeze(0))
    return(total)

def compute_reward(seq1,seq2, actions , ignore_far_sim=False, temp_dist_thre=20, use_gpu=True,include_audio=1):
    """
    Compute diversity reward and representativeness reward
    Args:
        seq1:audio seq2:img sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1) new-(batch, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    _seq1 = seq1.detach()
    _seq2 = seq2.detach()
    _actions = actions.detach()
    batch=_actions.shape[0]
    
    reward_list=[]
    pick_idxs_list = gen_non_zero_list(_actions) # list of list
    _seq1 = _seq1.squeeze()
    _seq2= _seq2.squeeze()
    n1 = _seq1.size(0)
    n2 = _seq2.size(0)
    normed_seq1 = _seq1 / _seq1.norm(p=2, dim=1, keepdim=True)
    normed_seq2 = _seq2 / _seq2.norm(p=2, dim=1, keepdim=True)
    dissim_mat1 = 1. - torch.matmul(normed_seq1, normed_seq1.t()) # dissimilarity matrix [Eq.4]
    dissim_mat2 = 1. - torch.matmul(normed_seq2, normed_seq2.t())    
    
    dist_mat1 = torch.pow(_seq1, 2).sum(dim=1, keepdim=True).expand(n1, n1)
    dist_mat2 = torch.pow(_seq2, 2).sum(dim=1, keepdim=True).expand(n2, n2)
    dist_mat1 = dist_mat1 + dist_mat1.t()
    dist_mat2 = dist_mat2 + dist_mat2.t()
    dist_mat1.addmm_(1, -2, _seq1, _seq1.t())
    dist_mat2.addmm_(1, -2, _seq2, _seq2.t())    
    
    
    for itr in range(batch):
        pick_idxs=pick_idxs_list[itr]
#         print(pick_idxs.shape)
        num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    
        
        if num_picks == 0:
            # give zero reward is no frames are selected
            reward = torch.tensor(0.)
            if use_gpu: reward = reward.to(device)
            reward_list.append(reward.cpu())
            continue



            # compute diversity reward
        if num_picks == 1:
            reward_div = torch.tensor(0.)
            if use_gpu: reward_div = reward_div.to(device)
            reward_list.append(reward_div.cpu())
            continue
        else:

            dissim_submat1 = dissim_mat1[pick_idxs,:][:,pick_idxs]
            dissim_submat2 = dissim_mat2[pick_idxs,:][:,pick_idxs]
            if ignore_far_sim:
                # ignore temporally distant similarity
                pick_mat = pick_idxs.expand(num_picks, num_picks)
                temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
                dissim_submat1[temp_dist_mat > temp_dist_thre] = 1.
                dissim_submat2[temp_dist_mat > temp_dist_thre] = 1
            reward_div1 = dissim_submat1.sum() / (num_picks * (num_picks - 1.)) # diversity reward [Eq.3]
            reward_div2 = dissim_submat2.sum() / (num_picks * (num_picks - 1.))
            reward_div = include_audio*reward_div1+reward_div2*6
        # compute representativeness reward

    
        dist_mat1_i = dist_mat1[:,pick_idxs]
        dist_mat2_i = dist_mat2[:,pick_idxs]
        dist_mat1_i = dist_mat1_i.min(1, keepdim=True)[0]
        dist_mat2_i = dist_mat2_i.min(1, keepdim=True)[0]
    #reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
        reward_rep1 = torch.exp(-dist_mat1_i.mean())
        reward_rep2 = torch.exp(-dist_mat2_i.mean())
        reward_rep=include_audio*reward_rep1+reward_rep2*2
    # combine the two rewards
        reward = (10*reward_div + 0.05*reward_rep) * 0.5*0.5
#     print(reward_div,reward_rep)
        reward_list.append(reward.cpu())
#     print(reward_list)

    return np.array(reward_list) # BATCH 

def tempsigmoid(x):
    nd=3.0 
    temp=nd/torch.log(torch.tensor(9.0)) 
    return torch.sigmoid(x/(temp))

X = np.array([[10, 12], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
(kmeans.labels_)

kmeans.predict([[20, 4], [12, 3]])

az=[]

def traj_mask(traject_mat,traject_mat2,reward_vec,reward_vec2,n_components=2,n_clusters=5,del_cluster=3):# tm- #episod X seq  ,reward_vect-episode
    n_episodes=traject_mat.shape[0]
    traject_mat_conc=torch.cat([traject_mat,traject_mat2],0)
    pca = PCA(n_components)
    principalComponents_conc=pca.fit_transform(traject_mat_conc.cpu().numpy())#2xepisod X n_components
    principalComponents1 =principalComponents_conc[0:n_episodes,:] 
    principalComponents2 = principalComponents_conc[n_episodes:2*n_episodes,:] 
#     print(principalComponents_conc.shape,principalComponents1.shape)
    kmeans = KMeans(n_clusters, random_state=0).fit(principalComponents1)
    traj1_labels=kmeans.labels_
    traj2_pred_labels=kmeans.predict(principalComponents2)
    reward_per_cluster=np.zeros(n_clusters)
    count_per_cluster=np.zeros(n_clusters)+0.00001
    for i in range(traject_mat.shape[0]):
        reward_per_cluster[traj1_labels[i]] +=reward_vec[i]
        count_per_cluster[traj1_labels[i]] +=1
    
    avg_per_cluster=reward_per_cluster/count_per_cluster
    deleted_cluster_list=[]
    for i in range(del_cluster):
        u=np.argmax(avg_per_cluster)
        deleted_cluster_list.append(u)
        avg_per_cluster[u]=-99999
    mask=np.zeros((traject_mat.shape[0]))+1
    for j in range(traject_mat.shape[0]):
        if traj2_pred_labels[j]in deleted_cluster_list:
            mask[j]=0
    return mask

class SelfAtten(nn.Module):
    def __init__(self,inp_dim,q_dim,k_dim,v_dim):
        super(SelfAtten, self).__init__()
        self.q=nn.Linear(inp_dim,q_dim)
        self.k=nn.Linear(inp_dim,k_dim)
        self.v=nn.Linear(inp_dim,v_dim)
        self.drop1=nn.Dropout(0.0)
        self.drop2=nn.Dropout(0.0)
        self.drop3=nn.Dropout(0.0)
    def forward(self,inp):#seq X B X inp_dim
        inp=inp.permute(1,0,2) #B X seq X inp_dim
        inp=self.drop1(inp)
        Q=self.q(inp)
        K=self.k(inp)
        V=self.v(inp)
        Q_K=torch.bmm(Q,K.permute(0,2,1))/10 #B X seq X seq
        soft_K=Q_K #B X seq X seq
        final=torch.bmm(soft_K,V) #B X seq X V_DIM
        final_reshape=final.permute(1,0,2) # seq X B X V_dim
#         print(inp.shape,final_reshape.shape)
        return(final_reshape)    




class DSN(nn.Module):
    """Deep AUDIO Summarization Network"""
    def __init__(self, in_dim=128*3, hid_dim=256*3, num_layers=1, cell='gru'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True)
        self.fc = nn.Linear(200, 1)
        self.fc2= nn.Linear(20, 20)
        self.atten=SelfAtten(hid_dim*2,200,200,200)

    def forward(self, x): #y- BX200

        z=x
        
        h, _ = self.rnn(z) #h-seq X B X num_layer*hid_dim
        h=self.atten(F.relu(h))# seq X B X 200
        
        return h
class DSN2(nn.Module):
    """Deep IMAGE Summarization Network"""
    def __init__(self, in_dim=1000, hid_dim=200, num_layers=1, cell='lstm'):
        super(DSN2, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True )
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True)
        self.fc = nn.Linear(200, 1)
#         self.fc2= nn.Linear(200, 200)
        self.atten=SelfAtten(hid_dim*2,100,100,200)
        

    def forward(self, x):
#         x=(x-x.mean())/x.std()

#         y=y.unsqueeze(0).repeat(x.shape[0],1,1)
#         y=self.fc2(y)
        z=x
        
        h, _ = self.rnn(z)
        h=self.atten(F.relu(h))# seq X B X 200
    
        
        return h
class Combo(nn.Module):
    def __init__(self,inp_dim,q_dim,k_dim,v_dim=200):
        super(Combo, self).__init__()        
        self.q_audio=nn.Linear(inp_dim,q_dim)
        self.k_audio=nn.Linear(inp_dim,k_dim)
        self.v_audio=nn.Linear(inp_dim,v_dim)    
        self.q_video=nn.Linear(inp_dim,q_dim)
        self.k_video=nn.Linear(inp_dim,k_dim)
        self.v_video=nn.Linear(inp_dim,v_dim)    
        self.fc = nn.Linear(v_dim*2, 1)
        
        
    def forward(self,video_inp,audio_inp):#seq X B X 200 seq X B X 200
        video_inp=video_inp.permute(1,0,2) #B X seq X inp_dim
        audio_inp=audio_inp.permute(1,0,2) #B X seq X inp_dim
        Q_audio=self.q_audio(audio_inp)
        K_audio=self.k_audio(audio_inp)
        V_audio=self.v_audio(audio_inp)      
        Q_video=self.q_video(video_inp)
        K_video=self.k_video(video_inp)
        V_video=self.v_video(video_inp) 
        Q_audio_K_video=torch.bmm(Q_audio,K_video.permute(0,2,1))/10
        Q_video_K_audio=torch.bmm(Q_video,K_audio.permute(0,2,1))/10
        
        soft_AV=Q_audio_K_video #B X seq X seq
        soft_VA=Q_video_K_audio
        final_AV=torch.bmm(soft_AV,V_audio) #B X seq X V_DIM
        final_VA=torch.bmm(soft_VA,V_video) #B X seq X V_DIM
        conc=torch.cat([final_AV,final_VA],2)
        final=torch.sigmoid(self.fc(conc))
        final_reshape=final.permute(1,0,2) # seq X B X 1
        return(final_reshape) #seq X B X 1

DeepProbAudio=DSN().to(device)
DeepProbImage=DSN2().to(device)

DeepProbAudio2=DSN().to(device)
DeepProbImage2=DSN2().to(device)
Combo_model1=Combo(200,200,300).to(device)
Combo_model2=Combo(200,200,300).to(device)
optimizer = torch.optim.Adam(list(Combo_model2.parameters())+list(Combo_model1.parameters())+list(DeepProbAudio.parameters())+list(DeepProbImage.parameters())+list(DeepProbAudio2.parameters())+list(DeepProbImage2.parameters()), lr=lr, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)
baselines1 = {key: 0. for key in train_data} # baseline rewards for videos
baselines2 = {key: 0. for key in train_data} # baseline rewards for videos
reward_writers1 = {key: [] for key in train_data} # record reward changes for each video
reward_writers2 = {key: [] for key in train_data} # record reward changes for each video

DeepProbAudio.load_state_dict(torch.load('aud1.pt',map_location='cpu'))
DeepProbAudio2.load_state_dict(torch.load('aud2.pt',map_location='cpu'))
DeepProbImage.load_state_dict(torch.load('vid1.pt',map_location='cpu'))
DeepProbImage2.load_state_dict(torch.load('vid2.pt',map_location='cpu'))
Combo_model1.load_state_dict(torch.load('combo1.pt',map_location='cpu'))
Combo_model2.load_state_dict(torch.load('combo2.pt',map_location='cpu'))

for epoch in range(100):
    idxs = np.arange(len(train_data))
    np.random.shuffle(idxs) 
    if epoch<5:
        req_audio=0
    else:
        req_audio=1
    print("cool")
    for idx in idxs:
        key = train_data[idx]
        full_seq_aud = torch.tensor(np.array(NEW_MEL_DICT[key][0]))#seq X dim(128*3)
        full_seq_img = torch.tensor(np.array(NEW_MEL_DICT[key][3]))#seq X dim(1000)
        full_seq_aud= (full_seq_aud-full_seq_aud.mean(1,keepdim=True))/full_seq_aud.std(1,keepdim=True)
        full_seq_img= (full_seq_img-full_seq_img.mean(1,keepdim=True))/full_seq_img.std(1,keepdim=True)
#         noise_img=NOISE_IMG.to(device)
#         noise_audio=NOISE_AUDIO.to(device)
        
#         print(full_seq.shape)



        probs_aud=DeepProbAudio(full_seq_aud.unsqueeze(1).repeat(1,BATCH,1).float().to(device))# seqlen X Batch X 200
        probs_img=DeepProbImage(full_seq_img.unsqueeze(1).repeat(1,BATCH,1).float().to(device))# seqlen X Batch X 200

        probs_aud2=DeepProbAudio2(full_seq_aud.unsqueeze(1).repeat(1,BATCH,1).float().to(device))# seqlen X Batch X 200
        probs_img2=DeepProbImage2(full_seq_img.unsqueeze(1).repeat(1,BATCH,1).float().to(device))# seqlen X Batch X 200
             
        
        probs=Combo_model1(probs_aud,probs_img)# seqlen X Batch X 1
        
        probs2=Combo_model2(probs_aud2,probs_img2)# seqlen X Batch X 1

        print("prob1-",probs.mean().item(),probs.max().item(),probs.min().item(),(probs.squeeze()>0.5).sum().item(),probs.size(0))

        print("prob2-",probs2.mean().item(),probs2.max().item(),probs2.min().item(),(probs2.squeeze()>0.5).sum().item(),probs2.size(0))

        cost = beta * (0.8*(probs - 0.5)**2).mean()#         cost =-0.1*total_var_loss+ beta * (0*(probs_aud - 0.5)**2 + 0.1*(probs_img - 0.5)**2).mean()# minimize summary length penalty term [Eq.11]
        cost += beta * (0.8*(probs2 - 0.5)**2).mean()
        #CONTRASTIVE
        cost +=0.0001*(l2_dist(probs_aud,probs_img)+l2_dist(probs_aud2,probs_img2)-l2_dist(probs_aud,probs_aud2)-l2_dist(probs_img,probs_img2))
        m = Bernoulli(probs)
        m2= Bernoulli(probs2)

        epis_rewards1 = []    
        epis_rewards2 = []
        trajectory_list1=[]
        trajectory_list2=[]
        reward_traj_list1=[]
        reward_traj_list2=[]
        log_prob_list1=[]
        log_prob_list2=[]
        reward_list_list1=[]
        reward_list_list2=[]
        for _ in range(num_episode):
            actions = m.sample() # seqlen X Batch X 1
            actions2 = m2.sample() # seqlen X Batch X 1
            trajectory_list1.append(actions[:,:,0])
            trajectory_list2.append(actions2[:,:,0])
            
            log_probs = m.log_prob(actions)
            log_probs2 = m2.log_prob(actions2)
            log_prob_list1.append(log_probs)
            log_prob_list2.append(log_probs2)
            
#             print(actions.shape)
            reward_list = compute_reward(full_seq_aud.unsqueeze(0).to(device),full_seq_img.unsqueeze(0).to(device), actions.permute(1,0,2),include_audio=req_audio)
            reward_list=torch.tensor(reward_list).unsqueeze(1).to(device)
            reward_list2 = compute_reward(full_seq_aud.unsqueeze(0).to(device),full_seq_img.unsqueeze(0).to(device), actions2.permute(1,0,2),include_audio=req_audio)
            reward_list2=torch.tensor(reward_list2).unsqueeze(1).to(device)
            reward_list_list1.append(reward_list)
            reward_list_list2.append(reward_list2)
   
        reward_vector1=torch.cat(reward_list_list1,0)[:,0] #episode
        reward_vector2=torch.cat(reward_list_list2,0)[:,0] #episode
        trajectory_mat1=torch.cat(trajectory_list1,1).permute(1,0) # episode X seq_len 
        trajectory_mat2=torch.cat(trajectory_list2,1).permute(1,0) # episode X seq_len 
#         print(trajectory_mat1.shape)
        
        mask2_frm_1=traj_mask(trajectory_mat1,trajectory_mat2,reward_vector1,reward_vector2)
        for h in range(num_episode):
            log_probs=log_prob_list1[h]
            log_probs2=log_prob_list2[h]
            reward_list=reward_list_list1[h]
            reward_list2=reward_list_list2[h]
            expected_reward = torch.matmul(torch.transpose(log_probs.mean(0),0,1),  (reward_list - baselines1[key]))[0][0]
            expected_reward2 = torch.matmul(torch.transpose(log_probs2.mean(0),0,1),  (reward_list2 - baselines2[key]))[0][0]
            cost -= (5*expected_reward/BATCH)+(5*expected_reward2/BATCH)*mask2_frm_1[h] # minimize negative expected reward
            epis_rewards1.append((reward_list.mean().item()))
            epis_rewards2.append((reward_list2.mean().item()))
            
            
#         print(mask2_frm_1)
#         print(reward_mat1.shape,reward_mat2.shape)
        optimizer.zero_grad()
        cost.backward()
#         torch.nn.utils.clip_grad_norm_(DeepProbImage.parameters(), 5.0)
        optimizer.step()
    
        baselines1[key] = 0.9 * baselines1[key] + 0.1 * np.mean(epis_rewards1) # update baseline reward via moving average
        baselines2[key] = 0.9 * baselines2[key] + 0.1 * np.mean(epis_rewards2)
        reward_writers1[key].append(np.mean(epis_rewards1))
        reward_writers2[key].append(np.mean(epis_rewards2))

    epoch_reward1 = np.mean([(reward_writers1[key][epoch]) for key in train_data]) 
    epoch_reward2 = np.mean([(reward_writers2[key][epoch]) for key in train_data]) 
    print("epoch_1 {}/{}\t reward {}\t".format(epoch+1, max_epoch, epoch_reward1))
    print("epoch_2 {}/{}\t reward {}\t".format(epoch+1, max_epoch, epoch_reward2))
#     torch.save(DeepProb.state_dict(),"first.pt")
#     if epoch % 11==0:
#         print('TEST REWARD-',test_reward(test_data,BATCH))

pca = PCA(10)

test_data

for i in range(2):
    traj_mask(az[3],az[1],None,None)

def generate_trailer_timelist(key,device,norm=True):
    full_seq_aud = torch.tensor(np.array(NEW_MEL_DICT[key][0]))#seq X dim(128*3)
    full_seq_img = torch.tensor(np.array(NEW_MEL_DICT[key][3]))#seq X dim(1000)
    if norm:
        full_seq_aud= (full_seq_aud-full_seq_aud.mean(1,keepdim=True))/full_seq_aud.std(1,keepdim=True)
        full_seq_img= (full_seq_img-full_seq_img.mean(1,keepdim=True))/full_seq_img.std(1,keepdim=True)    
    full_time_interval_list=torch.tensor(np.array(NEW_MEL_DICT[key][1]))
    probs_aud=DeepProbAudio(full_seq_aud.unsqueeze(1).repeat(1,BATCH,1).float().to(device))# seqlen X 1 X 1
    probs_img=DeepProbImage(full_seq_img.unsqueeze(1).repeat(1,BATCH,1).float().to(device))# seqlen X 1 X 1
    probs_aud2=DeepProbAudio2(full_seq_aud.unsqueeze(1).repeat(1,BATCH,1).float().to(device))# seqlen X 1 X 1
    probs_img2=DeepProbImage2(full_seq_img.unsqueeze(1).repeat(1,BATCH,1).float().to(device))# seqlen X 1 X 1
    probs=Combo_model1(probs_aud,probs_img)# seqlen X Batch X 1
    
    probs=np.array(probs.squeeze().detach().cpu())
    
#     arr.argsort()[-3:][::-1]
    probs2=Combo_model2(probs_aud2,probs_img2)
    probs2=np.array(probs2.squeeze().detach().cpu())
    mask1=np.zeros(probs.shape)
    mask2=np.zeros(probs2.shape)
    index1=probs.argsort()[-7:][::-1]
    index2=probs2.argsort()[-7:][::-1]
    mask1[index1]=1
    mask2[index2]=1
    print(full_time_interval_list.shape)
    req_list1=full_time_interval_list[torch.tensor(np.array(index1))]
#     pick_idxs2 = (probs2).nonzero().squeeze()
    req_list2=full_time_interval_list[torch.tensor(np.array(index2))]
#     print(req_list1.shape)
    return(probs,probs2,req_list1,req_list2,full_time_interval_list)
    
    
probs_set=[]    

probsz,probs2z,list1,list2,full_list=generate_trailer_timelist('PJrm840pAUI',device) #Feed the name of the video you want trailer
uu=[]
for i in range((probsz.shape[0])):
    uu.append(i)

print(probsz.shape)
plt.plot(uu,((probs2z)))

plt.show()

from moviepy.editor import *
# z=[]
def gen_trailer_frmList(og_loc,clip_list,sr,name):
    print(name)
    
    sub_clip_list=[]
    clip_list=sorted(clip_list, key=lambda tup: (tup[0],tup[1]) )
    
    for i in clip_list:
        i=i.numpy()
        print(i)
        my_clip=VideoFileClip(og_loc)
        
        sub_clip_list.append(my_clip.subclip(i[0]/sr,i[1]/sr))
    
    trailer = concatenate_videoclips(sub_clip_list, method='compose') 
#     z.append(trailer)
    trailer.write_videofile(name+'.mp4')
    
    return(trailer)
    

trailer_name_list=["3eYKfiOEJNs","4wU_LUjG5Ic","E11zDS9XGzg","GsAD1KT1xo8","PJrm840pAUI"]

for name in trailer_name_list:
    probs,probs2,list1,list2,_=generate_trailer_timelist(name,device)
    ll="/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/video/"+name+".mp4"
    gen_trailer_frmList(ll,list1,sr,name+"_CONTtrailer_1.mp4")
    gen_trailer_frmList(ll,list2,sr,name+"_CONTtrailer_2.mp4")


