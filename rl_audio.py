# -*- coding: utf-8 -*-



import torch
import sys
import torch.nn as nn
from torch.nn import functional as F
import random
import import_ipynb
import Preprocess
from Preprocess import *
import math
import matplotlib.pyplot as plt
import soundfile as sf
import IPython
import FeatureExtraction
from FeatureExtraction import *
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli



#PARAMETERS
lr=1e-05
weight_decay=1e-05
max_epoch=6000
stepsize=30
gamma=0.1
beta=1
num_episode=30
batch=16

#FEATURE EXTRACTING MODEL
urls= {'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',}
model_feat=VGGish(urls,)

audio_dir="/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/audio/"
audio_split_dir="/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/Audio_split/"
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
MEL_DICT={}
for name in name_list:
    print("okay-",name)
    mel_pername=[]
    audio_pername,time_regionpername=Shortaudio_timestamp(audio_split_dir+name+'/')
    for au in audio_pername:
#         print(au.shape)
        feat=model_feat.forward(au).reshape(-1).detach().numpy()
        mel_pername.append(feat)
        
#         mel_pername.append(gen_melspec(au))
    MEL_DICT[name]=(mel_pername,time_regionpername,audio_pername)




NEW_MEL_DICT={}
for key in MEL_DICT.keys():
    M=MEL_DICT[key][0]
    feat=[]
    for i in range(len(M)):
        Z=M[i]#.detach().numpy()
        feat.append(Z)
    feat=np.array(feat)
    NEW_MEL_DICT[key]=(feat,MEL_DICT[key][1],MEL_DICT[key][2])

np.save('audio_dict.npy',NEW_MEL_DICT)

#CHECKED

# t=np.load('audio_dict.npy',allow_pickle=True)[()].keys()
# t

def compute_reward(seq, actions, ignore_far_sim=False, temp_dist_thre=20, use_gpu=True):
    """
    Compute diversity reward and representativeness reward
    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    
    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
        return(reward_div)
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t()) # dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.)) # diversity reward [Eq.3]

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    dist_mat = dist_mat[:,pick_idxs]
#     print(dist_mat.shape)
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    #reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
    reward_rep = torch.exp(-dist_mat.mean())

    # combine the two rewards
    reward = (reward_div + reward_rep) * 0.5
#     print(reward_div,reward_rep)

    return reward

def tempsigmoid(x):
    nd=3.0 
    temp=nd/torch.log(torch.tensor(9.0)) 
    return torch.sigmoid(x/(temp))

class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=128*3, hid_dim=256*3, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        p = F.sigmoid(self.fc(h))
        return p
    
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)
    
audio_encoder = nn.Sequential(
        Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
        Conv2d(32, 32, kernel_size=3, stride=2, padding=1, residual=False),
        Conv2d(32, 32, kernel_size=3, stride=2, padding=1, residual=False),

        Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1),
        Conv2d(64, 64, kernel_size=3, stride=2, padding=1, residual=False),
        Conv2d(64, 64, kernel_size=3, stride=2, padding=1, residual=False),

        Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#         Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=False),

       )

device='cuda:0'
DeepProb=DSN().to(device)
# audio_encoder=audio_encoder.to(device)
optimizer = torch.optim.Adam(list(DeepProb.parameters()), lr=lr, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)
baselines = {key: 0. for key in train_data} # baseline rewards for videos
reward_writers = {key: [] for key in train_data} # record reward changes for each video
DeepProb.load_state_dict(torch.load('first.pt'))

# prob=DeepProb(audio_feat.reshape(audio_feat.size(0),-1).unsqueeze(1))
# prob.shape

for epoch in range(1):
    idxs = np.arange(len(train_data))
    np.random.shuffle(idxs) 
    print("cool")
    for idx in idxs:
        key = train_data[idx]
        full_seq = torch.tensor(np.array(NEW_MEL_DICT[key][0]))#seq X dim
        
#         print(full_seq.shape)



        probs=DeepProb(full_seq.unsqueeze(1).float().to(device))# seqlen X 1 X 1
        print(probs.mean().item(),probs.max().item(),probs.min().item(),(probs.squeeze()>0.5).sum().item(),probs.size(0))
        cost = beta * (probs.mean() - 0.5)**2 # minimize summary length penalty term [Eq.11]
        m = Bernoulli(probs)
        epis_rewards = []            
        for _ in range(num_episode):
            actions = m.sample()
            log_probs = m.log_prob(actions)
#             print(actions.shape)
            reward = compute_reward(full_seq.unsqueeze(0).to(device), actions.permute(1,0,2))
            expected_reward = log_probs.mean() * (reward - baselines[key])
            cost -= expected_reward # minimize negative expected reward
            epis_rewards.append(reward.item())      

#         optimizer.zero_grad()
#         cost.backward()
# #         torch.nn.utils.clip_grad_norm_(DeepProb.parameters(), 5.0)
#         optimizer.step()
    
        baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
        reward_writers[key].append(np.mean(epis_rewards))

    epoch_reward = np.mean([reward_writers[key][epoch] for key in train_data])      
    print("epoch {}/{}\t reward {}\t".format(epoch+1, max_epoch, epoch_reward))
#     torch.save(DeepProb.state_dict(),"first.pt")

len(reward_writers[key])
xs = [x for x in range(6000)]
# print((reward_writers[key][0]).shape,len(xs))
plt.scatter(xs,reward_writers['EE-bNr36nyA'])

reward_writers.keys()

plt.x

(probs.squeeze()>0.9).sum().item(),len(probs)

def gen_audio_from_impsplits(key,prob):
    prob=prob.squeeze()
    indx=(prob>=0.9).nonzero()
    time_coup=NEW_MEL_DICT[key][1]
    
    audio_signal=NEW_MEL_DICT[key][2]
    ep=[]
    new_signal=[]
    for i in indx:
        time_interval=time_coup[i][1]-time_coup[i][0]
        
        ep.append((time_coup[i][0],time_coup[i][1]))
#         print((time_coup[i][0]/16000)/60)
        new_signal=new_signal+list(audio_signal[i][0:time_interval])
    return(new_signal,ep)

def test_reward(test_data):
    reward_writers_test = {key: [] for key in test_data}
    for i,key in enumerate(test_data):
        full_seq = torch.tensor(np.array(NEW_MEL_DICT[key][0]))#seq X dim
        
        probs=DeepProb(full_seq.unsqueeze(1).float().to(device))# seqlen X 1 X 1
        print(probs.mean().item(),probs.max().item(),probs.min().item(),(probs.squeeze()>0.5).sum().item(),probs.size(0))
        m = Bernoulli(probs)
        epis_rewards = []            
        for _ in range(num_episode):
            actions = m.sample()
            print(actions.mean())
#             print(actions.shape)
            reward = compute_reward(full_seq.unsqueeze(0).to(device), actions.permute(1,0,2))
            epis_rewards.append(reward.item())          
        reward_writers_test[key].append(np.mean(epis_rewards))

    Mean_reward = np.mean([reward_writers_test[key][epoch] for key in test_data])  
    return(Mean_reward)
test_reward(test_data)

z,ez=gen_audio_from_impsplits(key,probs)
print(len(z),len(ez))
key

sf.write('new.wav', z, 16000)
IPython.display.Audio("new.wav")

u,_=librosa.load('/home2/kadumcs16/Aditya_Patra/VideoSummarisation/dataset/ydata-tvsum50-v1_1/audio/'+key+'.wav',sr=16000)


librosa.display.waveplot(u,sr=16000)

u_dup=np.zeros(len(u))
for l,m in ez:
    u_dup[l:m]=1

print(u_dup.sum()/(len(u)))

librosa.display.waveplot(u_dup,sr=16000)



train_data=['0tmA_C6XwfM',
 'kLxoNp-UchI',
 '4wU_LUjG5Ic',
 'PJrm840pAUI',
 'GsAD1KT1xo8',
 '37rzWOQsNIw',
 'akI8YFjEmUw',
 'XkqCExn6_Us',
 'byxOvuiIJV0',
 'Bhxk-O1Y7Ho',
 'xxdtq8mxegs',
 'AwmHb44_ouw',
 'EE-bNr36nyA',
 'JKpqYvAdIsw',
 'WG0MBPpPC6I',
 'qqR6AEXwxoQ',
 'jcoYJXDG9sw',
 'iVt07TCkFM0',
 'NyBmCxDoHJU',
 '3eYKfiOEJNs',
 '_xMr-HKMfVA',
 'EYqVtI9YWJA',
 'LRw_obCPUt0',
 'xmEERLqJ2kU',
 'E11zDS9XGzg',
 '-esJrBWj2d8',
 'gzDbaEs1Rlg',
 'XzYM3PfTM4w',
 'J0nA4VgnoCo',
 'JgHubY5Vw3Y',
 'sTEELN-vY30',
 'WxtbjNsCQ8A',
 'VuWGsYPqAX8',
 'Hl-__g2gn_A',
 'cjibtmSLxQ4',
 'RBCABdttQmI',
 'uGu_10sucQo',
 'b626MiF1ew4',
 'Yi4Ij2NM7U4',
 'HT5vyqe0Xaw']
Test_data=['z_6gVvQb2d0',
 'fWutDQy1nnY',
 'xwqBXPGE9pQ',
 'eQu1rNs0an0',
 'Se3oxnaPsz0',
 '91IHQYk1IQM',
 'vdmoEJ5YbrQ',
 'i3wAGJaaktw',
 '98MoyGZKHXc',
 'oDXZc0tZe04']



