from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import json 
import os
import numpy as np
import random
from tqdm import tqdm
import re

def extract_vehicle_signals(input_string):
    # Patterns for matching speed, curvature, acceleration, and course data
    speed_pattern = r"Speed: \[([0-9., -]+)\]"
    curvature_pattern = r"Curvature: \[([0-9., -]+)\]"
    acceleration_pattern = r"Acceleration: \[([0-9., -]+)\]"
    course_pattern = r"Course: \[([0-9., -]+)\]"

    # Function to extract and convert data to a list of floats
    def extract_data(pattern):
        match = re.search(pattern, input_string)
        return [float(x) for x in match.group(1).split(',')] if match else []

    # Extract data
    speed = extract_data(speed_pattern)
    curvature = extract_data(curvature_pattern)
    acceleration = extract_data(acceleration_pattern)
    course = extract_data(course_pattern)

    return speed, curvature, acceleration, course



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer_norm = nn.LayerNorm(1052)  # Normalize the entire concatenated input

        self.layers = nn.Sequential(
            nn.Linear(1052, 2048),
            nn.GELU(),
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Linear(4096, 8192)  
        )

    def forward(self, x):
        x = self.layer_norm(x)
        return self.layers(x)




def get_emb():
    train_conv_path = "./video_process/final_conv_base/conversation_bddx_train.json"
    test_conv_path = "./video_process/final_conv_base/conversation_bddx_eval.json"
        
    with open(train_conv_path,"r") as ftr:
        train_conv = json.load(ftr)
    with open(test_conv_path,"r") as fte:
        test_conv = json.load(fte)
    conv = train_conv + test_conv
    # Load embeddings
    vp_emb_match = {}
    npz_file = np.load("./video_process/BDDX_Processed/new_emb/embeddings.npz")
    for key in npz_file:
        vp_emb_match[key] = npz_file[key]
    npz_file.close()
    
    # Info
    with open("./retrieval/bddx_vpath_info_match.json",'r') as im:
        vp_info = json.load(im)
    vp_signal_match = {vp['video']: vp_info[vp['video']] for vp in conv}
    vp_pure_signal = {vp:extract_vehicle_signals(val) for vp, val in vp_signal_match.items()}
    
    # Concat
    vp_f_emb = {}
    for key in vp_emb_match.keys():
        sig = np.concatenate([np.array(lst) for lst in vp_pure_signal[key]]).reshape(1,28)
        # print(sig.shape, vp_emb_match[key].shape)
        val = np.concatenate((vp_emb_match[key], sig), axis=1).reshape(1052).astype(np.float32)
        # print(val.shape)
        vp_f_emb.update({key:val})
    
    return vp_f_emb

vp_f_emb = get_emb()
model = MLP()
checkpoint = torch.load('./retrieval/projector/best_model.pth')
model.load_state_dict(checkpoint)
print("Loaded")
model.eval()
psim = {}
with torch.no_grad():
    for vp, emb in tqdm(vp_f_emb.items()):
        predictions = model(torch.tensor(emb))
        p = predictions.unsqueeze(0).numpy()
        # print(p.shape)
        psim.update({vp:p})
        # break
    
np.savez(f'./retrieval/embeddings_project.npz', **psim)