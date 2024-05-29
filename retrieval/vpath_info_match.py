import json 

def format_conversation_as_qa(conversations):
    """
    Formats a list of conversation entries into a QA format.
    
    Parameters:
    conversations (list): A list of conversation dictionaries with 'from' and 'value' keys.

    Returns:
    str: A formatted string representing the conversation in QA format.
    """
    formatted_conversation = ""
    # print(conversations)
    for i in range(0, len(conversations), 2):
        # Check if the pair of conversation exists and matches the expected pattern
        if i+1 < len(conversations) and conversations[i]["from"] == "human" and conversations[i+1]["from"] == "gpt":
            question = conversations[i]["value"]
            answer = conversations[i+1]["value"]
            if "Predict" in question:
                continue
            formatted_conversation += f"Human: {question}\nAssistant: {answer}\n\n"

    return formatted_conversation

if __name__ == "__main__":
    dataset = 'BDDX'

    if dataset == 'BDDX':
        train_conv_path = "./video_process/final_conv_base/conversation_bddx_train.json"
        test_conv_path = "./video_process/final_conv_base/conversation_bddx_eval.json"
    elif dataset == 'SAX':
        train_conv_path = "./video_process/final_sax_conv_base/conversation_sax_train.json"
        test_conv_path = "./video_process/final_sax_conv_base/conversation_sax_eval.json"
        
    with open(train_conv_path,"r") as ftr:
        train_conv = json.load(ftr)
    with open(test_conv_path,"r") as fte:
        test_conv = json.load(fte)
        
    all_conv = train_conv + test_conv

    vpath_info_match = {}
    for item in all_conv:
        vpath,conv = item['video'][0], item['conversations']
        rconv = format_conversation_as_qa(conv)
        vpath_info_match.update({vpath:rconv})
        # break
        
    if dataset == 'BDDX':
        with open("./retrieval/bddx_vpath_info_match.json","w") as fic:
            json.dump(vpath_info_match,fic,indent=4)        
    elif dataset == 'SAX':
        with open("./retrieval/sax_vpath_info_match.json","w") as fic:
            json.dump(vpath_info_match,fic,indent=4)
