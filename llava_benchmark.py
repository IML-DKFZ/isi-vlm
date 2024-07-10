import torch
import os
import random
import sys

import numpy as np
from tqdm import tqdm
import skimage
import pandas as pd

from datasets import load_dataset, concatenate_datasets
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from argparse import ArgumentParser

from llava.data_utils.data_utils import (
    load_yaml,
    construct_prompt,
    save_json,
    process_single_sample,
    CAT_SHORT2LONG,
)
from llava.data_utils.model_utils import call_llava_engine_df, llava_image_processor
from llava.data_utils.eval_utils import parse_multi_choice_response, parse_open_response
from llava.data_utils.set_seed import set_seed

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.data_utils.model_utils import tokenizer_image_token, deal_with_prompt
import gc

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
set_seed(42)

processor = None
call_model_engine = call_llava_engine_df
vis_process_func = llava_image_processor

files = pd.read_csv("./dash_app/assets/files.csv")


def timed(model):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = model.generate(
        input_ids,
        images=input_image.unsqueeze(0).half().cuda(),
        do_sample=True,
        temperature=1,
        top_p=None,
        num_beams=5,
        max_new_tokens=128,
        use_cache=False,
    )
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


gc.collect()
torch.cuda.empty_cache()
# load model
model_name = get_model_name_from_path("liuhaotian/llava-v1.5-13b")
tokenizer, model, vis_processors, _ = load_pretrained_model(
    "liuhaotian/llava-v1.5-13b", None, model_name, load_4bit=True, device_map= "auto",
)

input_prompt = files["input_text"].iloc[0] + " " + files["input_question"].iloc[0]
input_image = skimage.io.imread("./dash_app/assets/" + files["input_image"].iloc[0])

input_image = vis_process_func(input_image, vis_processors).to(device)

conv = conv_templates["vicuna_v1"].copy()
conv.append_message(conv.roles[0], input_prompt)
conv.append_message(conv.roles[1], None)
input_prompt = conv.get_prompt()
input_prompt = deal_with_prompt(input_prompt, model.config.mm_use_im_start_end)
input_ids = (
    tokenizer_image_token(
        input_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )
    .unsqueeze(0)
    .cuda()
)

#model.compile()

with torch.no_grad():
    output_ids, time = timed(model)

response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

print("Time: " + str(time) + " / Answer: " + response)