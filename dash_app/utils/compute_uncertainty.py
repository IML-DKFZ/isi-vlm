"""Compute uncertainty measures after generating answers. From: https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/compute_uncertainty_measures.py"""

from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import gc
import torch
import plotly.graph_objs as go
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.data_utils.model_utils import tokenizer_image_token, deal_with_prompt
from llava.data_utils.model_utils import call_llava_engine_df, llava_image_processor
from llava.data_utils.set_seed import set_seed
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from dash_app.utils.image_export import plotly_fig2PIL

from dash_app.utils.semantic_entropy import get_semantic_ids
from dash_app.utils.semantic_entropy import logsumexp_by_id
from dash_app.utils.semantic_entropy import predictive_entropy
from dash_app.utils.semantic_entropy import predictive_entropy_rao
from dash_app.utils.semantic_entropy import EntailmentDeberta

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

processor = None
call_model_engine = call_llava_engine_df
vis_process_func = llava_image_processor


def generate_uncertainty_score(input_text, input_question, figure, T, iter):

    # Load model an generate iter amount of answers with temp
    model_name = get_model_name_from_path("liuhaotian/llava-v1.5-13b")

    input_prompt = input_text + " " + input_question
    fig = go.Figure(figure)
    input_image = plotly_fig2PIL(fig)

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    input_prompt = conv.get_prompt()

    full_responses = []
    for i in tqdm(range(iter)):
        set_seed(i)
        gc.collect()
        torch.cuda.empty_cache()

        tokenizer, model, vis_processors, _ = load_pretrained_model(
            "liuhaotian/llava-v1.5-13b",
            None,
            model_name,
            load_4bit=True,
            attn_implementation="eager",
        )

        if i == 0:
            input_image = vis_process_func(input_image, vis_processors).to(device)

            input_prompt = deal_with_prompt(
                input_prompt, model.config.mm_use_im_start_end
            )
            input_ids = (
                tokenizer_image_token(
                    input_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )
            input_ids = (
                tokenizer_image_token(
                    input_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                images=input_image.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=T,
                max_new_tokens=256,
                use_cache=False,
                output_scores=True,
                output_hidden_states=False,
                return_dict_in_generate=True,
            )

        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        log_likelihoods = [score.item() for score in transition_scores[0]]

        response = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[
            0
        ]

        full_responses.append([response, log_likelihoods])

        del model
        gc.collect()
        torch.cuda.empty_cache()

    entailment_model = EntailmentDeberta()

    # Compute validation embeddings and entropies.
    responses = [fr[0] for fr in full_responses]

    log_liks = [r[1] for r in full_responses]
    responses = [f"{input_question} {r}" for r in responses]
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
        full_responses,
        num_clusters,
    )
