import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import math
from tqdm import tqdm

def random_subset(sample_list):
    """
    Randomly selects a varying number of elements (1 to 5) from the provided list.

    Parameters:
    sample_list (list): The list from which to select elements.

    Returns:
    list: A list containing a random subset of the original list.
    """
    # Randomly choose how many elements to select (between 1 and 5)
    # num_elements = random.randint(1, min(5, len(sample_list)))
    num_elements = 2

    # Randomly select the elements
    return random.sample(sample_list, num_elements)

if __name__ == "__main__":
    dataset = 'BDDX'
    split = 'train'
    
    if_hybird = True
    retr_strategy = "hybird"

    if dataset == "BDDX":
        if split == 'train':
            conv_path = f"./video_process/final_conv_base/conversation_bddx_{split}.json"
        elif split == 'test':
            conv_path = f"./video_process/final_conv_base/conversation_bddx_eval.json"
        with open(conv_path, "r") as fj:
            conv = json.load(fj)

        # Load embeddings
        id_emb_match = {}
        if split == 'train':
            npz_file = np.load("./video_process/BDDX_Processed/embeddings_train.npz")
        elif split == 'test':
            npz_file = np.load("./video_process/BDDX_Test/embeddings_test.npz")
            
        for key in npz_file:
            id_emb_match[key] = npz_file[key]
        npz_file.close()

    id_vpath_match = {ele["id"]: ele["video"] for ele in conv if ele["id"] in id_emb_match}
    vpath_id_match = {val:key for key,val in id_vpath_match.items()}
    id_conversation_match = {ele["id"]: ' '.join([ele["conversations"][1]['value'],ele["conversations"][3]['value']]) for ele in conv if ele["id"] in id_emb_match}

    # match json

    match_path = f"./retrieval/match_{retr_strategy}_similarity_{split}.json"
    with open(match_path,"r") as fmm:
        mm = json.load(fmm)

    new_mm = {}
    # Fixed Random sample
    exclude = []
    for key,val in mm.items():
        new_val = [id_vpath_match[ele] for ele in id_vpath_match if ele not in exclude]
        new_val = random_subset(new_val)
        new_mm.update({id_vpath_match[key]:new_val})

    id_match = {}
    for key,val in tqdm(new_mm.items()):
        new_val = [vpath_id_match[ele] for ele in val]
        id_match.update({vpath_id_match[key]:new_val})



    if dataset == 'BDDX':
        with open(f"./retrieval/BDDX_RAG_{retr_strategy}_vpmatch_{split}.json", "w") as fv:
            json.dump(new_mm, fv, indent=4)
        with open(f"./retrieval/BDDX_RAG_{retr_strategy}_vidmatch_{split}.json", "w") as fvid:
            json.dump(id_match, fvid, indent=4)



        