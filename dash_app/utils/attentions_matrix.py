# many are copied from https://github.com/mattneary/attention/blob/master/attention/attention.py
# here it nullifies the attention over the first token (<bos>)
# which in practice we find to be a good idea
from io import BytesIO
from PIL import Image
import requests
import torch
import numpy as np
import cv2
from typing import Tuple

def remove_default_prompt(attention_scores: Tuple[Tuple[torch.Tensor]], default_prompt_length: int) -> Tuple[Tuple[torch.Tensor]]:
    modified_attention_scores = []
    for output_attn in attention_scores:
        modified_output_attn = []
        for layer in output_attn:
            modified_layer = layer[:, :, default_prompt_length:, default_prompt_length:]
            #print("modified_layer size: ", modified_layer.size())
            modified_output_attn.append(modified_layer)
        modified_attention_scores.append(tuple(modified_output_attn))
    
    return tuple(modified_attention_scores)

def get_out_attn_matrix(attention_scores):
    # constructing the llm attention matrix for output tokens
    llm_out_attn_matrix = heterogenous_stack(list(map(aggregate_llm_attention, attention_scores)))
    return llm_out_attn_matrix

def get_llm_attn_matrix(attention_scores):
    # constructing the llm attention matrix
    aggregated_prompt_attention = []
    for i, layer in enumerate(attention_scores[0]):
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        cur = attns_per_head[:-1].cpu().clone()
        # following the practice in `aggregate_llm_attention`
        # we are zeroing out the attention to the first <bos> token
        # for the first row `cur[0]` (corresponding to the next token after <bos>), however,
        # we don't do this because <bos> is the only token that it can attend to
        cur[1:, 0] = 0.
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        aggregated_prompt_attention.append(cur)
    aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)

    # llm_attn_matrix will be of torch.Size([N, N])
    # where N is the total number of input (both image and text ones) + output tokens
    llm_attn_matrix = heterogenous_stack(
        [torch.tensor([1])]
        + list(aggregated_prompt_attention) 
        + list(map(aggregate_llm_attention, attention_scores))
    )
    #llm_attn_matrix = heterogenous_stack(list(map(aggregate_llm_attention, attention_scores)))
    return llm_attn_matrix


def aggregate_llm_attention(attn):
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            attns_per_head[-1][1:].cpu(),
            # attns_per_head[-1].cpu(),
            # add zero for the final generated token, which never
            # gets any attention
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)


def aggregate_vit_attention(attn, select_layer=-2, all_prev_layers=True):
    '''Assuming LLaVA-style `select_layer` which is -2 by default'''
    if all_prev_layers:
        avged = []
        for i, layer in enumerate(attn):
            if i > len(attn) + select_layer:
                break
            layer_attns = layer.squeeze(0)
            attns_per_head = layer_attns.mean(dim=0)
            vec = attns_per_head[1:, 1:].cpu() # the first token is <CLS>
            avged.append(vec / vec.sum(-1, keepdim=True))
        return torch.stack(avged).mean(dim=0)
    else:
        layer = attn[select_layer]
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = attns_per_head[1:, 1:].cpu()
        return vec / vec.sum(-1, keepdim=True)


def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])


def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')
    return image


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HSV)
    hm = np.float32(heatmap) / 255
    cam = hm + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap


class MultiModalAttention:
    def __init__(self, model, tokenizer, device):
        # Determine device (CPU or GPU)
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, attention_scores, prompt, description, question):
        llm_out_attn_matrix = get_out_attn_matrix(attention_scores)
        scores = self.multimodal_attention_aggregation(llm_out_attn_matrix, prompt, description, question)
        return scores
    
    def get_image_tokens(self):
        # Assuming the vision model is a part of the whole model
        vision_config = self.model.config.vision_config

        if self.model.config.architectures[0] == 'LlavaNextForConditionalGeneration':
            num_patches = self.model.image_end_pos - self.model.image_start_pos
            num_patches = num_patches.cpu().numpy()
        else:
            image_size = vision_config.image_size
            patch_size = vision_config.patch_size
            num_patches = (image_size // patch_size) ** 2
        return num_patches
    
    def get_positions(self, prompt, description, question):
        # the given prompt does not have the default vicuna instruction
        image_start = len(self.tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
        image_end = image_start + self.get_image_tokens()
        #print("Number of image tokens: ", self.get_image_tokens())
        description_start = image_end + 1  # 1 is for the <image> token
        description_end = description_start + len(self.tokenizer(description, return_tensors='pt')["input_ids"][0])
        question_start = description_end
        question_end = question_start + len(self.tokenizer(question, return_tensors='pt')["input_ids"][0])
        positions = {
            "image_start": image_start,
            "image_end": image_end,
            "description_start": description_start,
            "description_end": description_end,
            "question_start": question_start,
            "question_end": question_end
        }
        return positions
    
    def multimodal_attention_aggregation(self, llm_out_attn_matrix, prompt, description, question):
        P = self.get_positions(prompt, description, question)
        #print("Positions items: ", P)
        #print("llm_out_attn_matrix size: ", llm_out_attn_matrix.size())
        # Normalize on the input tokens only (not the output tokens)
        attn_to_image = llm_out_attn_matrix[:-1, P["image_start"]:P["image_end"]] #:-1 is to exclude the <eos> or </s> token
        attn_to_description = llm_out_attn_matrix[:-1, P["description_start"]:P["description_end"]] #:-1 is to exclude the <eos> or </s> token
        attn_to_question = llm_out_attn_matrix[:-1, P["question_start"]:P["question_end"]] #:-1 is to exclude the <eos> or </s> token
        #print("attn_to_question size: ", attn_to_question.size())
        #print("attn_to_question: ", attn_to_question[0, :10])

        scores = {
            "attn_I": torch.sum(attn_to_image).item(),
            "attn_T": torch.sum(attn_to_description).item(),
            "attn_Q": torch.sum(attn_to_question).item(),
        }
        return scores