import torch
import numpy as np
import gc
import torch
import torch
import plotly.graph_objs as go

from transformers import (
    AutoModelForPreTraining,
    LlavaNextProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    AutoProcessor
)
from dash_app.utils.set_seed import fix_random_seed
from dash_app.utils.image_export import plotly_fig2PIL
from dash_app.utils.attentions_matrix import MultiModalAttention
from dash_app.utils.modeling_llava_next import LlavaNextForConditionalGeneration_adapted

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_attention(input_text, question, figure, llava_version, load_4bit):
    gc.collect()
    torch.cuda.empty_cache()
    if llava_version == "llava 7b":
        model_id = "llava-hf/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(model_id)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config if load_4bit else None,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    elif llava_version == "llava-vicuna 7b":
        model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
        processor = LlavaNextProcessor.from_pretrained(model_id)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
        )
        model = LlavaNextForConditionalGeneration_adapted.from_pretrained(
            model_id,
            quantization_config=quantization_config if load_4bit else None,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    elif llava_version == "llava-next 7b":
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        processor = LlavaNextProcessor.from_pretrained(model_id)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
        )
        model = LlavaNextForConditionalGeneration_adapted.from_pretrained(
            model_id,
            quantization_config=quantization_config if load_4bit else None,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    input_prompt = input_text + " " + question
    fig = go.Figure(figure)
    image = plotly_fig2PIL(fig)

    if "next" in llava_version:
        prompt = f"[INST] <image>\n{input_prompt} [/INST]"
    else:
        prompt = f"USER: <image>\n{input_prompt} \nASSISTANT:"

    if "vicuna" in llava_version:
        default_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. "
        prompt = default_prompt + prompt

    inputs = processor(prompt, image, return_tensors="pt").to(device, torch.float32)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    attention_scores = outputs["attentions"]

    attention_model = MultiModalAttention(model, processor.tokenizer, device)
    answer_attn_scores = attention_model(
        attention_scores, prompt, input_text, question
    )
    return answer_attn_scores