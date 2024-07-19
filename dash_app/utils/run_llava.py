import gc
import torch
import plotly.graph_objs as go

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.data_utils.model_utils import tokenizer_image_token, deal_with_prompt
from llava.data_utils.model_utils import call_llava_engine_df, llava_image_processor
from llava.data_utils.set_seed import set_seed
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from dash_app.utils.image_export import plotly_fig2PIL

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

processor = None
call_model_engine = call_llava_engine_df
vis_process_func = llava_image_processor


def llava_inference(input_text, input_question, figure):
    gc.collect()
    torch.cuda.empty_cache()
    # load model
    model_name = get_model_name_from_path("liuhaotian/llava-v1.5-13b")
    tokenizer, model, vis_processors, _ = load_pretrained_model(
        "liuhaotian/llava-v1.5-13b", None, model_name, load_4bit=True
    )

    input_prompt = input_text + " " + input_question
    fig = go.Figure(figure)
    input_image = plotly_fig2PIL(fig)

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
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=input_image.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=1,
            num_beams=3,
            max_new_tokens=128,
            use_cache=False,
        )

    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return response
