"""
IMPORTANT!!

This module code mainly refers to the demo released by InternLM XCompositer around October 2023.
At present, the demo has been updated. If it cannot run normally or there are problems, you can visit the project according to the following link to find the currently available demo.
https://github.com/InternLM/InternLM-XComposer

The main code of the CoT module can refer to the content after line 137.
You can use the latest MLLM generation tool to simultaneously splice our CoT module.
"""

import os

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = '0,1,2,3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import logging
import sys
import json
from PIL import ImageFile

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import argparse
import torch

try:
		import torch_npu
		from torch_npu.contrib import transfer_to_npu
except:
		pass

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils import auto_configure_device_map

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--num_gpus", default=1, type=int)
args = parser.parse_args()

# init model and tokenizer
model = AutoModel.from_pretrained('D:/model_code/C4MMD-main-origin/internlm-xcomposer-7b', trust_remote_code=True).cuda().eval()
if args.num_gpus > 1:
    from accelerate import dispatch_model
    device_map = auto_configure_device_map(args.num_gpus)
    model = dispatch_model(model, device_map=device_map)

tokenizer = AutoTokenizer.from_pretrained('D:/model_code/C4MMD-main-origin/internlm-xcomposer-7b', trust_remote_code=True)
model.tokenizer = tokenizer

def save_json(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=2))

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# =======================CoT module=======================

process_file_list = ['train_data', 'val_data', 'test_data']
origin_data_path = 'D:/model_code/C4MMD-main-origin/data'
'''
The origin data path, the one you saved from https://github.com/liaolianfoka/MET-Meme-A-Multi-modal-Meme-Dataset-Rich-in-Metaphors
'''
new_data_path = 'D:/model_code/C4MMD-main-origin/data'
'''
The data path you processed with MLLM.
'''
image_file_path = 'D:/model_code/C4MMD-main-origin/data/image'
'''
Fill in the image save path here, which should have two folders containing Chinese and English images, respectively.
For example:

data/image -> the image file path, which contains two folders as follow.
    |_English
    |_Chinese
'''

for file in process_file_list:
    # Check first if there are any checkpoints present
    if os.path.exists(f'{new_data_path}/new_{file}.json'):
        datas = load_json(f'{new_data_path}/new_{file}.json')
    else:
        datas = load_json(f'{origin_data_path}/{file}.json')
        
    count = 0
    for line in tqdm(datas):
        if 'internlm_mix_info' in line:
            continue
        img_name = line['images_name']
        img = f'{image_file_path}/{img_name}'
        Question1 = 'Please temporarily ignore the text in the image and describe the content in the image. Try to be concise while ensuring the correctness of your answers.'
        response1 = model.generate(Question1, img)
        line['internlm_img_info'] = response1

        Question2 = f'The text in the picture is as follows: "{line["text"]}". Please analyze the meaning of the text. Note that there may be homophonic memes and puns, distinguish and explain them but do not over interpret while ensuring the correctness of the answer and be concise.'
        response2 = model.generate(Question2)
        line['internlm_text_info'] = response2

        Question3 = f'Image description: {response1}; Text: "{line["text"]}"; Text description: {response2}. Please combine the image, text, and their description information and try to understand the deep meaning of the combination of the image and text. No need to describe images and text, only answer implicit meanings. Ensure the accuracy of the answer and try to be concise as much as possible.'
        response3 = model.generate(Question3, img)
        line['internlm_mix_info'] = response3

        count += 1
        if count == 100:
            # This is a checkpoint to prevent unexpected stops during code execution and loss of previous results.
            print('save_a_part')
            count = 0
            save_json(f'{new_data_path}/new_{file}.json', datas)

    save_json(f'{new_data_path}/new_{file}.json', datas)

print('finish!')