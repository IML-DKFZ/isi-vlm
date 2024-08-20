"""Compute uncertainty measures after generating answers. From: https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/compute_uncertainty_measures.py"""

import numpy as np
import gc
import torch
import torch
import plotly.graph_objs as go

from dash_app.utils.set_seed import fix_random_seed
from dash_app.utils.image_export import plotly_fig2PIL

from transformers import (
    AutoProcessor,
    LlavaNextProcessor,
    BitsAndBytesConfig,
    AutoProcessor,
    LlavaForConditionalGeneration,
)

from dash_app.utils.modeling_llava_next import LlavaNextForConditionalGeneration_adapted
from dash_app.utils.semantic_entropy import get_semantic_ids
from dash_app.utils.semantic_entropy import logsumexp_by_id
from dash_app.utils.semantic_entropy import predictive_entropy
from dash_app.utils.semantic_entropy import predictive_entropy_rao
from dash_app.utils.semantic_entropy import EntailmentDeberta

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def generate_uncertainty_score(input_text, question, figure, T, iter, llava_version, load_4bit):
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
        default_prompt_length = len(
            processor.tokenizer(default_prompt, return_tensors="pt")["input_ids"][0]
        )

    input = processor(prompt, image, return_tensors="pt").to(device, torch.float32)
    
    full_responses = []
    for i in range(iter):
        fix_random_seed(i)
        with torch.no_grad():
            outputs = model.generate(
                **input,
                do_sample=True,
                temperature=T,
                max_new_tokens=256,
                use_cache=True,
                output_scores=True,
                output_hidden_states=False,
                return_dict_in_generate=True,
            )

        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        log_likelihoods = [score.item() for score in transition_scores[0]]

        response = (
            processor.decode(outputs.sequences[0], skip_special_tokens=True)
            .split("ASSISTANT: ")[-1]
            .strip()
        )

        full_responses.append([response, log_likelihoods])

        gc.collect()
        torch.cuda.empty_cache()


    entailment_model = EntailmentDeberta()

    # Compute validation embeddings and entropies.
    responses_woq = [fr[0] for fr in full_responses]

    log_liks = [fr[1] for fr in full_responses]
    responses = [f"{question} {r}" for r in responses_woq]
    semantic_ids = get_semantic_ids(responses, model=entailment_model)
    num_clusters = len(np.unique(semantic_ids))
    log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]
    # Compute semantic entropy.
    log_likelihood_per_semantic_id = logsumexp_by_id(
        semantic_ids, log_liks_agg, agg="sum_normalized"
    )
    regular_entropy = predictive_entropy(log_liks_agg)
    semantic_entropy = predictive_entropy_rao(log_likelihood_per_semantic_id)

    del entailment_model

    return (
        np.round(semantic_entropy, 4),
        np.round(regular_entropy, 4),
        [semantic_ids, responses_woq],
        num_clusters,
    )
