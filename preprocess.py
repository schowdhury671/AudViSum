# -*- coding: utf-8 -*-


import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
import glob
import os
import math



def load_signal(loc,sr=None):
    y, sr = librosa.load(loc,sr)
#     print(sr)
    return(y)

def gen_melspec(y,n_fft=512, hop_length=256, n_mels=100,sr=16000):
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return(log_mel_spectrogram)

def pad_signal_borders(y,max_samples): # max is even ,y time signal
    gap=max_samples-len(y)
    if gap<0:
        print("error")
    if gap%2==0:
        l=r=gap//2
    else:
        l=gap//2
        r=l+1
    padded=np.concatenate([np.zeros(l),np.array(y),np.zeros(r)])
    return(padded)
def find_max(complete_list,sr=None):
    max_num=0
    for x in complete_list:
        y=load_signal(x,sr)
        max_num=max(max_num,len(y))
    if max_num%2!=0:
        return(max_num+1)
    return(max_num)

def loadList(filename):
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()

def Shortaudio_timestamp(audio_split_folder,timestep_limit=3*16000,sr=16000):
    dir_wav=audio_split_folder+"*.wav"
    audio_split_file=audio_split_folder+'listz.npy'
    time_stamp_list=loadList(audio_split_file)
    audio_lst = glob.glob(dir_wav)
    new_audio_feature_list=[]
    new_time_stamp_list=[]
#     print(len(audio_lst),len(time_stamp_list))
    for i in range(len(audio_lst)):
        time_interval=time_stamp_list[i][1]-time_stamp_list[i][0]
        signal=load_signal(audio_split_folder+str(i)+'.wav')
        no_of_times=math.ceil(len(signal)/timestep_limit)
        empty_time=np.zeros(no_of_times*timestep_limit)
        empty_time[0:len(signal)]=signal#side padding indirectly
        for j in range(no_of_times):
            new_audio_feature_list.append(empty_time[j*timestep_limit:(j+1)*timestep_limit])
            if j!=no_of_times-1:
                new_time_stamp_list.append((time_stamp_list[i][0]+j*timestep_limit,time_stamp_list[i][0]+(j+1)*timestep_limit))
            else:
                new_time_stamp_list.append((time_stamp_list[i][0]+j*timestep_limit,time_stamp_list[i][1]))
            
    return(new_audio_feature_list,new_time_stamp_list)

a,b=Shortaudio_timestamp('/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/Audio_split/esJrBWj2d8/')

#CHECKED

