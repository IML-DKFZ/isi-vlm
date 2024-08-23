import gc
import torch
import plotly.graph_objs as go

from transformers import (
    AutoProcessor,
    LlavaNextProcessor,
    BitsAndBytesConfig,
    AutoProcessor,
    LlavaForConditionalGeneration,
)

from dash_app.utils.modeling_llava_next import LlavaNextForConditionalGeneration_adapted
from dash_app.utils.image_export import plotly_fig2PIL

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def llava_inference(input_text, input_question, figure, llava_version, load_4bit):
    gc.collect()
    torch.cuda.empty_cache()
    if llava_version == "llava":
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
    elif llava_version == "llava-vicuna":
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
    elif llava_version == "llava-next":
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

    input_prompt = input_text + " " + input_question
    fig = go.Figure(figure)
    image = plotly_fig2PIL(fig)

    if "next" in llava_version:
        prompt = f"[INST] <image>\n{input_prompt} [/INST]"
    else:
        prompt = f"USER: <image>\n{input_prompt} \nASSISTANT:"

    if "vicuna" in llava_version:
        default_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. "
        prompt = default_prompt + prompt
        default_prompt_length = len(
            processor.tokenizer(default_prompt, return_tensors="pt")["input_ids"][0]
        )

    inputs = processor(prompt, image, return_tensors="pt").to(device, torch.float32)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            output_attentions=False,
            return_dict_in_generate=True,
        )

    answer = (
        processor.decode(outputs.sequences[0], skip_special_tokens=True)
        .split("ASSISTANT: ")[-1]
        .split("[/INST]")[-1]
        .strip()
    )

    return answer
