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


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Questions:
    # q1 = "What is the action of ego car?"
    # q2 = "Why does the ego car doing this?"
    json_file = "./video_process/conv_icl3/conversation_bddx_icl_eval.json"
    out_json_paths = [f"./video_process/BDDX_train_caption.json"]
    
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                     args.load_8bit, args.load_4bit, device=args.device)

    video_processor = processor['video']


    conv_mode = "v1"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()


    # gt
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Pred
    out_jsons = [[],[]]
    
    for item in tqdm(data):
        
        conv.messages.clear()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
            
        video, vid = item["video"], item['id']
        video = os.path.join("./video_process",video)

        video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
        if type(video_tensor) is list:
            tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)
        key = ['video']


        
        inp = "Please Describe the video"
        
        if video is not None:
            # First Message
            inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            video = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()
        
        print(input_ids.shape)
        
        # print(input_ids.shape)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

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
        # break
    
    # Save separate json for action and justification
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
    args = parser.parse_args()
    main(args)
