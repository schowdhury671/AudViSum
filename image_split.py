# -*- coding: utf-8 -*-



import os
from path import Path
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import pathlib
import shutil
import glob
import numpy as np

main_dir="/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/video/"
parent_dir='/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/'
audio_dir="/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/audio/"
vid_split_main="/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/frame_split/"
feat_main="/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/frame_features/"
model_ft = models.resnet50(pretrained=True)

dir_mp4=main_dir+"*.mp4"
name_list=[]
dir_wav=audio_dir+"*.wav"
audio_lIst = glob.glob(dir_wav)
for audio in audio_lIst:
    name=os.path.basename(os.path.normpath(audio))[:-4]
    name_list.append(name)



#video_to_frames
import os
import glob
# files                                                                         
lst = glob.glob(dir_mp4)

# # print(lst)
 
for file in name_list:
    file_loc=main_dir+file+'.mp4'
    new_loc=vid_split_main+file+'/'
    print(new_loc)
    if os.path.exists(new_loc):
        shutil.rmtree(new_loc)
    os.makedirs(new_loc)

    os.system('ffmpeg -i {} -f image2 -vf fps=fps=1 {}out%d.png'.format(file_loc,new_loc))
#     break

tfm=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
tfm(Image.open('/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/frame_split/esJrBWj2d8/out1.png')).shape



#extract features
for file in name_list:
    old_loc=vid_split_main+file+'/'
    lst = glob.glob(old_loc+"*.png")
    new_loc=feat_main+file+'/'
    print(new_loc)
    if os.path.exists(new_loc):
        shutil.rmtree(new_loc)
    os.makedirs(new_loc)
    for img in lst:
        img_name = pathlib.PurePath(img).name[:-4]
        img_tensor=tfm(Image.open(img)).unsqueeze(0)# 1 X 3 X H X W
        feat=model_ft(img_tensor).detach().numpy()
        np.save(new_loc+img_name+'.npy',feat)


#CHECKED NO ISSUE