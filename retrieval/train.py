import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import json 
import os
import numpy as np
import random
import argparse
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
        # rep_sig = sig.repeat(1024, 1)
        # # print(sig.shape, vp_emb_match[key].shape)
        # val = np.concatenate((vp_emb_match[key], rep_sig), axis=1).reshape(1052,2048).astype(np.float32)
        # # print(val.shape)
        vp_f_emb.update({key:sig})
    
    return vp_f_emb

class EmbeddingDataset(Dataset):
    def __init__(self):
        match_path = f"./retrieval/BDDX_RAG_pos_3.json"
        with open(match_path,"r") as fi:
            matches = json.load(fi)
        
        nmatch = f"./retrieval/BDDX_RAG_neg_3.json"      
        with open(nmatch,"r") as fi:
            neg_matches = json.load(fi)
    
            
        self.sig_embeddings = get_emb()
        self.pos_matches = matches  
        self.neg_matches = neg_matches  
        
        self.vp_id = {key: index for index, key in enumerate(matches.keys())}
        self.id_vp = {val: key for key,val in self.vp_id.items()}
        
        # print(self.id_vp[15466])

    def __len__(self):
        return len(self.sig_embeddings)
    
    def get_emb_from_vp(self,vp):
        
        ep = vp.split("/")[-1].replace('.mp4','.npy')
        ep = os.path.join("./video_process/BDDX_Processed/new_emb2",ep)
        emb = np.load(ep)
        sig = self.sig_embeddings[vp]
        rep_sig = sig.repeat(1024, 1).reshape(1,28,1024)
        # print(emb.shape, rep_sig.shape)
        val = np.concatenate((emb, rep_sig), axis=1).reshape(1024,2076).astype(np.float32)
        
        return val
        
        

    def __getitem__(self, idx):
        vp = self.id_vp[idx] #  Current query vp
        
        pos_lis = self.pos_matches[vp]
        neg_lis = self.neg_matches[vp]
        pos_vp = random.choice(pos_lis)
        neg_vp = random.choice(neg_lis)

        anchor, pos, neg = [self.get_emb_from_vp(v) for v in [vp, pos_vp, neg_vp]]

        return (anchor, pos, neg)


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

def main(args):
    # Create dataset and data loader
    dataset = EmbeddingDataset()
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize your model and loss function
    model = MLP()
    loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    # Initialize your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn.to(device)


    # Training loop
    num_epochs = 2000
    save_interval = 200
    best_loss = float("inf")  # Initialize with a high value
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        num_batches = 0
        for anchor, positive, negative in data_loader:
            optimizer.zero_grad()  # Clear gradients
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)
            
            

            loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)

            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # Save the model checkpoint if it has the lowest loss so far
        if avg_loss < best_loss and epoch > 1000:
            best_loss = avg_loss
            torch.save(model.state_dict(), './retrieval/projector/best_model.pth')
        
        # Save the model checkpoint (remove this part if you don't need to save)
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f'./retrieval/projector/model_epoch{epoch+1}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script without DeepSpeed.")
    args = parser.parse_args()
    main(args)
