# -*- coding: utf-8 -*-


import librosa
import soundfile as sf
import librosa.display
import numpy as np
import IPython
import os
import glob
import shutil

main_dir="/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/video/"
parent_dir="/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/"
audio_dir="/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/audio/"
audio_split_main="/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/Audio_split/"
dir_mp4=main_dir+"*.mp4"
dir_wav=audio_dir+"*.wav"
audio_lst = glob.glob(dir_wav)
len(audio_lst)

# MP4 TO WAV
import os
import glob
# files                                                                         
lst = glob.glob(dir_mp4)
# print(lst)
 
for file in lst:
    actual_file = file[:-4]

    os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format(file, actual_file))

def saveList(myList,filename):
    np.save(filename,myList)
    print("Saved successfully!")
    
    
def loadList(filename):
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()

def nonsilent_list(y): #y - audio signal 
    sr=16000
    rmse = librosa.feature.rms(y=y, frame_length=256, hop_length=64)[0]
    signal_power=np.abs(np.mean(librosa.power_to_db(rmse**2, ref=np.max)))
    lv = librosa.effects.split(y,top_db=signal_power-4)
    conc_list=[]
    print("LENGHT OF Original LIST-",len(lv))
    u=0
    last=[0,0]
    for i in lv:
#         print(i)
        chump=[i[0],i[1]]
        if last==[0,0]:
            if chump[1]-chump[0]>=3*sr:
                conc_list.append(chump)
            else:
                last=chump
        else:
            if chump[0]-last[1]>1.5*sr:# if gap between chump is big
                conc_list.append(last)
                last=chump
            elif chump[1]-last[0]<=3*sr:
    #             conc_list.append([last[0],chump[1]])
                last=[last[0],chump[1]]
            elif chump[1]-last[0]>3*sr:
                conc_list.append(last)
                last=chump
            else:
                print(error)
    if last != [0,0]:
        conc_list.append(last)
        last=[0,0]
    print("LENGHT OF BETTER LIST-",len(conc_list))    
    return(conc_list)



#AUDIO SPLITTING AND SAVING
for audio in audio_lst:
    name=os.path.basename(os.path.normpath(audio))[:-4]
    Split_loc=audio_split_main+name+'/'
    print(Split_loc)
    if os.path.exists(Split_loc):
        shutil.rmtree(Split_loc)
    os.makedirs(Split_loc)
    #reading audio
    y, sr = librosa.load(parent_dir+'audio/'+name+'.wav',sr=16000)
    final_list=nonsilent_list(y)
    for u,tup in enumerate(final_list): 
        sf.write(Split_loc+str(u)+'.wav',y[tup[0]:tup[1]], sr)
    saveList(final_list,Split_loc+"listz.npy")

