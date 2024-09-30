import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from PIL import Image
import numpy as np
import cv2
import os
import time
import argparse
import struct
from model import VideoModel
import numpy as np
import glob
import os
from torch.utils.data import Dataset
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
import random
import matplotlib.pyplot as plt
from PIL import Image
import math

jpeg = TurboJPEG()
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
args = argparse.Namespace(
    gpus='0',
    lr=2e-5,
    batch_size=100,
    num_workers=12,
    max_epoch=60,
    shaking_prob=0.2,
    max_magnitude=0.07,
    test=False,
    n_dimention=500,
    temperture=0.07,
    save_prefix='checkpoints/',
    dataset='',
    weights='best.pt'
)
def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]

    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:',missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    for name, param in model.named_parameters():
        param.requires_grad = True
        # only fine tune the projection head if you dont have enough gpu memory
        if name in pretrained_dict.keys() and "prejection_head" not in name:
            param.requires_grad = False
    return model

def reduce_image_list(image_list):
    size = len(image_list)

    # If size is less than 50, return the original list
    if size < 50:
        return image_list

    # If size is less than 100, reduce to half
    elif size < 100:
        reduced_size = size // 2

    # If size is less than 200, reduce to a third
    elif size < 200:
        reduced_size = size // 3

    # If size is 200 or more, reduce to a fourth
    else:
        reduced_size = size // 4

    # Reduce uniformly across the list
    step = math.ceil(size / reduced_size)
    reduced_list = image_list[::step]

    return reduced_list[:reduced_size]


def extract_opencv(filename):
    video = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()  # BGR format
        if ret:
            frame = jpeg.encode(frame)
            video.append(frame)
        else:
            break
    cap.release()
    return video

def load_video(video_path, frames_limit=None, crop_size=(88, 88)):
    inputs = extract_opencv(video_path)
    inputs = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in inputs]
    inputs = np.stack(inputs, 0) / 255.0
    batch_img = inputs[:,:,:,0] # 25, h, w
    # batch_img = CenterCrop(batch_img, (88, 88))
    batch_img = reduce_image_list(batch_img)
    batch_img = torch.FloatTensor(batch_img[np.newaxis,:,np.newaxis,...])
    result = {}
    result['video'] = batch_img
    result['label'] =  os.path.basename(video_path)
    result['duration'] = [[1.0 for i in range(batch_img.shape[1])]]
    print(result['label'])
    return result

def embaddings(input):
    videos = input.get('video')
    border = torch.FloatTensor(input.get('duration'))
    with torch.no_grad():
        embedding = F.normalize(video_model(videos,border),p=2, dim=-1)
    return embedding.detach().cpu().numpy()

def predict(video_path):
  clf_loaded = joblib.load('random_forest_classifier.pkl')
  new_embedding = embaddings(load_video(video_path))
  predicted_folder = clf_loaded.predict([np.asarray(new_embedding).flatten()])
  return predicted_folder[0]

video_model = VideoModel(args)
weight = torch.load(args.weights, map_location=torch.device('cpu'))
video_model = load_missing(video_model, weight.get('video_model'))
