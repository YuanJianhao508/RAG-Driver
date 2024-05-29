import argparse
import torch

from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import os
import json
from tqdm import tqdm

# video_root = "/zhaobai46g/Project/Video-LLaVA/video_process/"
# retr_strategy = 'visual'
# train_match_file, test_match_file = [f"./retrieval/BDDX_RAG_{retr_strategy}_vpmatch_{split}.json" for split in ['train','test']]
# with open(train_match_file, "r") as fm:
#      train_match = json.load(fm)
# with open(test_match_file, "r") as ft:
#      test_match = json.load(ft)
# PATH_RETRIEVAL_MATCH = {**train_match, **test_match}

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def split_list_into_subsets(real_dst_train, cur_worker, num_chunks=4):
    # num_chunks = 8
    chunk_size = len(real_dst_train) // num_chunks
    chunk_index = cur_worker
    if chunk_index == num_chunks - 1:
        subset_indices = range(chunk_index * chunk_size, len(real_dst_train))
    else:
        subset_indices = range(chunk_index * chunk_size, (chunk_index + 1) * chunk_size)
    subset = [real_dst_train[i] for i in subset_indices]
    return subset


def main(args):
    # Questions:

    json_file = args.input
    os.makedirs(args.output, exist_ok=True)
    out_json_paths = [f"{args.output}/BDDX_Test_pred_{cap}_{args.cur_worker}.json" for cap in ['action','justification','control_signal']]
    
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                     args.load_8bit, args.load_4bit, device=args.device)
    # print(model, tokenizer, processor)
    # image_processor = processor['image']
    video_processor = processor['video']


    conv_mode = "driving"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()


    # gt
    with open(json_file, 'r') as file:
        data = json.load(file)
        
    # Split data
    sub_data = split_list_into_subsets(data, cur_worker=args.cur_worker, num_chunks=args.total_worker)
    data = sub_data

    # Pred
    out_jsons = [[],[],[]]
    
    for item in tqdm(data):
        q1, q2, q3 = item["conversations"][0]["value"], item["conversations"][2]["value"], item["conversations"][4]["value"]
        conv.messages.clear()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
            
        vps, vid = item["video"], item['id']
        
        video_paths = [os.path.join("./video_process",vp) for vp in vps]

        
        video_tensor = [video_processor(video_path, return_tensors='pt')['pixel_values'] for video_path in video_paths]
        if type(video_tensor) is list:
            tensor = [[video.to(model.device, dtype=torch.float16) for video in video_tensor]]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)
        key = ['video']

        
        inst_answers = []
        for qid, question in enumerate([q1,q2,q3]):
            # print(question)
            inp = question
            
            if vps is not None:
                # First Message
                # inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
                inp = inp
                conv.append_message(conv.roles[0], inp)
                video = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)

            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            
            print(len(tensor), key)
            # import pdb;pdb.set_trace()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[tensor, key],
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
                
            
            inst_pred = {
                "image_id":vid,
                "caption":outputs.replace("</s>","")
            }

            out_jsons[qid].append(inst_pred)
        # import pdb; pdb.set_trace()
        # break
    
    # Save separate json for action and justification
    for i in range(3):
        with open(out_json_paths[i],"w") as of:
            json.dump(out_jsons[i], of, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--total_worker", type=int, default=4)
    parser.add_argument("--cur_worker", type=int, default=0)
    args = parser.parse_args()
    main(args)
