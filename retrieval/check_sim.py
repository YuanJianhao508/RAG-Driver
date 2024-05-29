import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re

# match_path = f"./retrieval/BDDX_RAG_hybird_vpmatch.json"
# match_path = f"./retrieval/BDDX_RAG_visual_vpmatch.json"
# match_path = f"./retrieval/BDDX_RAG_visual_vpmatch.json"
match_path = f"./retrieval/BDDX_RAG_tuned_vpmatch_t13.json"
# match_path = f"./retrieval/BDDX_RAG_tuned_vpmatch_t9.json"
id_info_path = "./retrieval/bddx_vpath_info_match.json"

def get_text(text):
    # Regex pattern to find the action and justification
    action_pattern = r"What is the action of ego car\?\nAssistant: (.+?)\n"
    justification_pattern = r"Why does the ego car doing this\?\nAssistant: (.+?)\n"

    # Extracting action and justification
    action_match = re.search(action_pattern, text)
    justification_match = re.search(justification_pattern, text)

    action = action_match.group(1) if action_match else "No action found"
    justification = justification_match.group(1) if justification_match else "No justification found"

    return [action, justification]

with open(match_path,"r") as fm:
    match_dict = json.load(fm)
with open(id_info_path,"r") as fi:
    id_info_match = json.load(fi)
    
id_info_match = {vp:" ".join(get_text(cont)) for vp, cont in id_info_match.items()}
    
vectorizer = TfidfVectorizer()
sim_f = []
for vp, rag in tqdm(match_dict.items()):
    cq_info = id_info_match[vp]
    rag_infos = [id_info_match[rvp] for rvp in rag]
    # print(rag_infos)
    # break
    info_lis = [cq_info] + rag_infos
    tfidf_matrix = vectorizer.fit_transform(info_lis)
    sim = cosine_similarity(tfidf_matrix)
    mean_sim = sum(sim[0][1:])
    sim_f.append(mean_sim)

print(sum(sim_f))
print(sum(sim_f)/len(sim_f))