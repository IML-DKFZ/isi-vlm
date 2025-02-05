# Import packages
import dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, ctx
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import dash_daq as daq
import skimage
from skimage.util import random_noise
import numpy as np
import os
import sys
import base64
import io
from PIL import Image

sys.path.append(os.getcwd())

from dash_app.utils.run_llava import llava_inference
from dash_app.utils.compute_uncertainty import generate_uncertainty_score
from dash_app.utils.compute_attention import generate_attention
from dash_app.utils.set_seed import fix_random_seed

from dash_app.utils.image_export import plotly_fig2PIL, pil_to_b64

fix_random_seed(42)

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

files = pd.read_csv("./dash_app/assets/vlm_dataset.csv")

options = [
    {"label": i, "value": j} for i, j in zip(files["image"], range(len(files["image"])))
]

NAVBAR = dbc.NavbarSimple(
    children=[
        dbc.Row(
            [
                dbc.Col(
                    dbc.NavItem(dbc.NavLink("LLaVA", id="llava_selected")),
                    width="auto",
                    align="right",
                    style={"margin-right": "0px"},
                ),
                dbc.Col(
                    dbc.NavItem(dbc.NavLink("7b", id="params_selected")),
                    width="auto",
                    style={"margin-left": "-30px"},
                ),
                dbc.Col(
                    dbc.DropdownMenu(
                        children=[
                            dbc.DropdownMenuItem("LLaVA", id="llava"),
                            dbc.DropdownMenuItem("LLaVA-Vicuna", id="llava-vicuna"),
                            dbc.DropdownMenuItem("LLaVA-Next", id="llava-next"),
                        ],
                        nav=True,
                        in_navbar=True,
                        label="VLM Version",
                        id="llava_dropdown",
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.DropdownMenu(
                        children=[
                            dbc.DropdownMenuItem("7b", id="7b"),
                            dbc.DropdownMenuItem("13b", id="13b"),
                            dbc.DropdownMenuItem("32b", id="32b"),
                        ],
                        nav=True,
                        in_navbar=True,
                        label="#B Parameter",
                        id="parameter_dropdown",
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Checkbox(
                        id="4bit_checkbox",
                        label="4bit",
                        label_style={"color": "#92BDFE"},
                        input_style={"color": "red"},
                        value=True,
                        style={
                            "margin-bottom": "-8px",
                            "margin-left": "0px",
                        },
                    ),
                    width="auto",
                ),
            ],
            align="center",
        )
    ],
    brand=dbc.Row(
        [
            dbc.Col(
                html.Img(
                    src="/assets/eth_logo_kurz_neg.png",
                    height="26px",
                    # style={"margin-right": "5px"},
                ),
                width="auto",
            ),
            dbc.Col(html.H4("|", style={"color": "white"}), width="auto"),
            dbc.Col(
                html.Img(
                    src="/assets/ibm_logo.png",
                    height="24px",
                    # style={"margin-right": "35px", "margin-left": "10px"},
                ),
                width="auto",
            ),
            dbc.Col(
                dbc.NavbarBrand(
                    "LLaVA Interactive Semantic Perturbations",
                    style={"font-size": 24},
                    className="ms-2",
                ),
                width="auto",
            ),
        ]
    ),
    brand_href="#",
    color="primary",
    dark=True,
    fluid=True,
    className="w-100",
)


LEFT_CONTAINER = [
    dbc.CardHeader(html.H4("Intervened Input")),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.P("Select observation from dataset:"),
                                    dcc.Dropdown(
                                        # label="Select Observation",
                                        value=0,
                                        options=options,
                                        clearable=False,
                                        id="dropdown_obs",
                                        style={"margin-bottom": "15px"},
                                    ),
                                ]
                            )
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.P("Or upload image:"),
                            dcc.Upload(
                                id="upload-data",
                                children=html.Div(
                                    ["Drag and Drop or ", html.B("Select Image")]
                                ),
                                style={
                                    "width": "100%",
                                    "height": "60px",
                                    "lineHeight": "60px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "margin": "10px",
                                },
                                # Allow multiple files to be uploaded
                                multiple=False,
                            ),
                        ],
                        width=6,
                    ),
                    html.Hr(className="my-1"),
                ]
            ),
            dbc.Row(
                html.H4("Input Image"),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="fig_perturb",
                                figure={},
                                config={
                                    "modeBarButtonsToAdd": [
                                        "drawrect",
                                        "eraseshape",
                                        "drawline",
                                    ],
                                    "modeBarButtonsToRemove": [
                                        "zoom",
                                        "pan",
                                        "zoomIn",
                                        "zoomOut",
                                        "autoScale",
                                        "resetScale",
                                    ],
                                    "displayModeBar": True,
                                    "displaylogo": False,
                                    "editable": True,
                                    "edits": {
                                        "shapePosition": True,
                                        "annotationPosition": True,
                                    },
                                },
                                style={
                                    "margin-bottom": "10px",
                                },
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            "Natural Preset",
                                            color="primary",
                                            className="me-1",
                                            id="natural_image_button",
                                            style={
                                                "margin-bottom": "20px",
                                                "margin-left": "0px",
                                                "align": "center",
                                            },
                                        ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Annotated Preset",
                                            color="primary",
                                            className="me-1",
                                            id="annotated_image_button",
                                            style={
                                                "margin-bottom": "20px",
                                                "margin-left": "0px",
                                                "align": "center",
                                            },
                                        ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Random Natural Image",
                                            color="secondary",
                                            className="me-1",
                                            id="random_image_button",
                                            style={
                                                "margin-bottom": "20px",
                                                "margin-left": "0px",
                                                "align": "center",
                                            },
                                        ),
                                        width="auto",
                                    ),
                                ]
                            ),
                        ],
                        width=9,
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                dbc.Col(
                                    [
                                        html.H5("Perturbations"),
                                        html.Hr(className="my-2"),
                                        html.B("Color Picker"),
                                        dbc.Input(
                                            type="color",
                                            id="color_picker",
                                            value="#000000",
                                            style={
                                                "width": 75,
                                                "height": 50,
                                                "margin-bottom": "10px",
                                            },
                                        ),
                                        html.B("Text Annotation"),
                                        dbc.Input(
                                            id="text_input_fig",
                                            placeholder="Text",
                                            type="text",
                                            style={
                                                "margin-bottom": "10px",
                                            },
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    daq.NumericInput(
                                                        id="text_size",
                                                        value=16,
                                                        min=0,
                                                        max=100,
                                                        style={
                                                            "margin-bottom": "10px",
                                                        },
                                                    ),
                                                    width=4,
                                                ),
                                                dbc.Col(
                                                    html.P(
                                                        "Text Size",
                                                        style={"textAlign": "left"},
                                                    ),
                                                    width=8,
                                                ),
                                            ],
                                            align="center",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Checklist(
                                                        options=[
                                                            {
                                                                "label": "",
                                                                "value": True,
                                                            },
                                                        ],
                                                        value=[],
                                                        id="switche_arrow",
                                                        inline=True,
                                                        switch=True,
                                                        style={
                                                            "margin-bottom": "10px",
                                                            "margin-left": "0px",
                                                        },
                                                    ),
                                                    width=4,
                                                ),
                                                dbc.Col(
                                                    html.P(
                                                        "Add Arrow",
                                                        style={"textAlign": "left"},
                                                    ),
                                                    width=8,
                                                ),
                                            ],
                                            align="center",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    daq.NumericInput(
                                                        id="arrow_size",
                                                        value=2,
                                                        min=0,
                                                        max=10,
                                                        style={
                                                            "margin-bottom": "10px",
                                                        },
                                                    ),
                                                    width=4,
                                                ),
                                                dbc.Col(
                                                    html.P(
                                                        "Arrow Size",
                                                        style={"textAlign": "left"},
                                                    ),
                                                    width=8,
                                                ),
                                            ],
                                            align="center",
                                        ),
                                        dbc.Row(
                                            dbc.Button(
                                                "New Text",
                                                color="primary",
                                                className="me-1",
                                                id="new_button",
                                                style={
                                                    "margin-bottom": "10px",
                                                    "margin-left": "0px",
                                                },
                                            ),
                                            className="d-grid gap-2 col-10 mx-auto",
                                        ),
                                    ],
                                    align="center",
                                )
                            ),
                            html.B("Gaussian Noise"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        daq.PrecisionInput(
                                            id="noise_level",
                                            value=0.2,
                                            min=0,
                                            max=10,
                                            precision=2,
                                            style={
                                                "margin-bottom": "10px",
                                            },
                                        ),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        html.P(
                                            "Sigma",
                                            style={"textAlign": "left"},
                                        ),
                                        width=6,
                                    ),
                                ],
                                align="center",
                            ),
                            dbc.Row(
                                dbc.Button(
                                    "Impute",
                                    color="primary",
                                    className="me-1",
                                    id="noise_button",
                                    style={
                                        "margin-bottom": "10px",
                                        "margin-left": "0px",
                                    },
                                ),
                                className="d-grid gap-2 col-10 mx-auto",
                            ),
                        ],
                        align="start",
                        className="h-100 p-3 bg-light text-dark border rounded-3",
                        width=3,
                    ),
                ]
            ),
            dbc.Row(html.H4("Input Context")),
            dbc.Row(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Prompt: "),
                        dbc.Textarea(placeholder={}, id="input_text"),
                    ],
                    className="mb-3",
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Complementary Preset",
                            color="primary",
                            className="me-1",
                            id="complementary_text_button",
                            style={
                                "margin-bottom": "10px",
                                "margin-left": "0px",
                                "align": "left",
                            },
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Contradictory Preset",
                            color="primary",
                            className="me-1",
                            id="contradictory_text_button",
                            style={
                                "margin-bottom": "10px",
                                "margin-left": "0px",
                                "align": "left",
                            },
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Random Complementary Text",
                            color="secondary",
                            className="me-1",
                            id="random_text_button",
                            style={
                                "margin-bottom": "30px",
                                "margin-left": "0px",
                                "align": "left",
                            },
                        ),
                        width="auto",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.H4("Input Question"), width="auto"),
                    dbc.Col(html.P("Ground truth based on image:"), width="auto"),
                    dbc.Col(
                        dbc.Badge(
                            {},
                            color={},
                            id="gt_badge",
                            className="border me-1",
                        ),
                        width="auto",
                        style={"margin-left": "-20px"},
                    ),
                ],
                align="bottom",
            ),
            dbc.Row(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Prompt: "),
                        dbc.Textarea(
                            placeholder="What is switzerlands capital?",
                            id="input_question",
                        ),
                    ],
                    className="mb-3",
                )
            ),
            dbc.Row(
                dbc.Button(
                    [
                        html.I(
                            className="bi bi-chat-right-text",
                            style={"margin-right": "5px"},
                        ),
                        "Generate Output",
                    ],
                    color="success",
                    className="me-1",
                    n_clicks=None,
                    id="button_gen",
                ),
                style={"verticalAlign": "right"},
            ),
        ]
    ),
]

# header


@callback(
    [
        Output(component_id="llava_selected", component_property="children"),
        Output(component_id="sem_entropy_tooltip", component_property="children"),
        Output(component_id="sem_cluster_tooltip", component_property="children"),
        Output(component_id="atte_question_tooltip", component_property="children"),
        Output(component_id="atte_image_tooltip", component_property="children"),
        Output(component_id="atte_context_tooltip", component_property="children"),
    ],
    [
        Input("llava", "n_clicks"),
        Input("llava-vicuna", "n_clicks"),
        Input("llava-next", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def update_llava_version(n1, n2, n3):
    id_lookup = {
        "llava": "LLaVA",
        "llava-vicuna": "LLaVA-Vicuna",
        "llava-next": "LLaVA-Next",
    }

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if id_lookup[button_id] == "LLaVA":
        return (
            id_lookup[button_id],
            ["Mean on VQA dataset:", html.Br(), html.B("1.21"), " (LLaVA)"],
            ["Mean on VQA dataset:", html.Br(), html.B("4.90"), " (LLaVA)"],
            ["Mean on VQA dataset:", html.Br(), html.B("0.15"), " (LLaVA)"],
            ["Mean on VQA dataset:", html.Br(), html.B("0.28"), " (LLaVA)"],
            ["Mean on VQA dataset:", html.Br(), html.B("0.09"), " (LLaVA)"],
        )
    if id_lookup[button_id] == "LLaVA-Vicuna":
        return (
            id_lookup[button_id],
            ["Mean on VQA dataset:", html.Br(), html.B("1.25"), " (LLaVA-Vicuna)"],
            ["Mean on VQA dataset:", html.Br(), html.B("4.67"), " (LLaVA-Vicuna)"],
            ["Mean on VQA dataset:", html.Br(), html.B("0.08"), " (LLaVA-Vicuna)"],
            ["Mean on VQA dataset:", html.Br(), html.B("0.11"), " (LLaVA-Vicuna)"],
            ["Mean on VQA dataset:", html.Br(), html.B("0.19"), " (LLaVA-Vicuna)"],
        )
    if id_lookup[button_id] == "LLaVA-Next":
        return (
            id_lookup[button_id],
            ["Mean on VQA dataset:", html.Br(), html.B("1.21"), " (LLaVA-Next)"],
            ["Mean on VQA dataset:", html.Br(), html.B("4.41"), " (LLaVA-Next)"],
            ["Mean on VQA dataset:", html.Br(), html.B("0.35"), " (LLaVA-Next)"],
            ["Mean on VQA dataset:", html.Br(), html.B("0.44"), " (LLaVA-Next)"],
            ["Mean on VQA dataset:", html.Br(), html.B("0.26"), " (LLaVA-Next)"],
        )


@callback(
    Output(component_id="params_selected", component_property="children"),
    [
        Input("7b", "n_clicks"),
        Input("13b", "n_clicks"),
        Input("32b", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def update_llava_version(n1, n2, n3):
    id_lookup = {
        "7b": "7b",
        "13b": "13b",
        "32b": "32b",
    }

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    return id_lookup[button_id]


# Callbacks for left
@callback(
    Output(component_id="input_question", component_property="value"),
    [
        Input(component_id="dropdown_obs", component_property="value"),
        Input(component_id="upload-data", component_property="contents"),
    ],
)
def update_input_text(value, contents):
    if ctx.triggered[0]["prop_id"].split(".")[0] == "upload-data":
        text = "Please enter your custom question here."
    else:
        text = files["question"].iloc[value]
    return text


@callback(
    Output(component_id="input_text", component_property="value"),
    [
        Input(component_id="dropdown_obs", component_property="value"),
        Input(component_id="upload-data", component_property="contents"),
    ],
)
def update_input_text(value, contents):
    if ctx.triggered[0]["prop_id"].split(".")[0] == "upload-data":
        text = "Please enter your custom context here."
    else:
        text = files["complementary"].iloc[value]
    return text


@callback(
    Output(component_id="input_text", component_property="value", allow_duplicate=True),
    Input(component_id="random_text_button", component_property="n_clicks"),
    prevent_initial_call=True,
)
def text_random(n_clicks):
    value = np.random.randint(len(files["complementary"]))
    text = files["complementary"].iloc[value]
    return text


@callback(
    Output(component_id="input_text", component_property="value", allow_duplicate=True),
    Input(component_id="contradictory_text_button", component_property="n_clicks"),
    State(component_id="dropdown_obs", component_property="value"),
    prevent_initial_call=True,
)
def text_cont(n_clicks, value):
    text = files["contradictory"].iloc[value]
    return text


@callback(
    Output(component_id="input_text", component_property="value", allow_duplicate=True),
    Input(component_id="complementary_text_button", component_property="n_clicks"),
    State(component_id="dropdown_obs", component_property="value"),
    prevent_initial_call=True,
)
def text_comp(n_clicks, value):
    text = files["complementary"].iloc[value]
    return text


@callback(
    [
        Output(component_id="gt_badge", component_property="children"),
        Output(component_id="gt_badge", component_property="color"),
    ],
    [
        Input(component_id="dropdown_obs", component_property="value"),
        Input(component_id="random_image_button", component_property="n_clicks"),
        Input(component_id="natural_image_button", component_property="n_clicks"),
        Input(component_id="annotated_image_button", component_property="n_clicks"),
        Input(component_id="upload-data", component_property="contents"),
    ],
)
def update_gt_badge(value, n1_clicks, n2_clicks, n3_clicks, contents):
    if (ctx.triggered[0]["prop_id"].split(".")[0] == "random_image_button") or (
        ctx.triggered[0]["prop_id"].split(".")[0] == "upload-data"
    ):
        return "?", "warning"
    else:
        gt = files["answer"].iloc[value]
        if gt == 1:
            return "Yes", "success"
        else:
            return "No", "danger"


@callback(
    Output(component_id="fig_perturb", component_property="figure"),
    [
        Input(component_id="dropdown_obs", component_property="value"),
        Input(component_id="upload-data", component_property="contents"),
    ],
    State("color_picker", "value"),
)
def fig_perturb(value, contents, color_value):

    if ctx.triggered[0]["prop_id"].split(".")[0] == "upload-data":
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        img = Image.open(io.BytesIO(decoded))
        fig = px.imshow(np.array(img, dtype=float), binary_string=True)

    else:
        fig = px.imshow(
            skimage.io.imread(
                "./dash_app/assets/natural_images/" + files["image"].iloc[value]
            )
        )

    fig.update_layout(
        newshape=dict(
            fillcolor=color_value, opacity=1.0, line=dict(color="black", width=0)
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        dragmode="drawrect",
        yaxis_visible=False,
        yaxis_showticklabels=False,
        xaxis_visible=False,
        xaxis_showticklabels=False,
    )

    return fig


@callback(
    Output(
        component_id="fig_perturb", component_property="figure", allow_duplicate=True
    ),
    Input(component_id="natural_image_button", component_property="n_clicks"),
    [
        State("color_picker", "value"),
        State(component_id="dropdown_obs", component_property="value"),
    ],
    prevent_initial_call=True,
)
def fig_natural(n_clicks, color_value, value):

    fig = px.imshow(
        skimage.io.imread(
            "./dash_app/assets/natural_images/" + files["image"].iloc[value]
        )
    )
    fig.update_layout(
        newshape=dict(
            fillcolor=color_value, opacity=1.0, line=dict(color="black", width=0)
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        dragmode="drawrect",
        yaxis_visible=False,
        yaxis_showticklabels=False,
        xaxis_visible=False,
        xaxis_showticklabels=False,
    )

    return fig


@callback(
    Output(
        component_id="fig_perturb", component_property="figure", allow_duplicate=True
    ),
    Input(component_id="annotated_image_button", component_property="n_clicks"),
    [
        State("color_picker", "value"),
        State(component_id="dropdown_obs", component_property="value"),
    ],
    prevent_initial_call=True,
)
def fig_annotated(n_clicks, color_value, value):

    fig = px.imshow(
        skimage.io.imread(
            "./dash_app/assets/annotated_images/" + files["image"].iloc[value]
        )
    )
    fig.update_layout(
        newshape=dict(
            fillcolor=color_value, opacity=1.0, line=dict(color="black", width=0)
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        dragmode="drawrect",
        yaxis_visible=False,
        yaxis_showticklabels=False,
        xaxis_visible=False,
        xaxis_showticklabels=False,
    )

    return fig


@callback(
    Output(
        component_id="fig_perturb", component_property="figure", allow_duplicate=True
    ),
    Input(component_id="random_image_button", component_property="n_clicks"),
    State("color_picker", "value"),
    prevent_initial_call=True,
)
def fig_random(n_clicks, color_value):
    value = np.random.randint(len(files["image"]))
    fig = px.imshow(
        skimage.io.imread(
            "./dash_app/assets/natural_images/" + files["image"].iloc[value]
        )
    )
    fig.update_layout(
        newshape=dict(
            fillcolor=color_value, opacity=1.0, line=dict(color="black", width=0)
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        dragmode="drawrect",
        yaxis_visible=False,
        yaxis_showticklabels=False,
        xaxis_visible=False,
        xaxis_showticklabels=False,
    )

    return fig


@callback(
    Output(
        component_id="fig_perturb", component_property="figure", allow_duplicate=True
    ),
    Input(component_id="noise_button", component_property="n_clicks"),
    [
        State("color_picker", "value"),
        State(component_id="dropdown_obs", component_property="value"),
        State(component_id="noise_level", component_property="value"),
    ],
    prevent_initial_call=True,
)
def fig_gaussian_noise(n_clicks, color_value, value, std):
    img = skimage.io.imread(
        "./dash_app/assets/natural_images/" + files["image"].iloc[value]
    )
    noised_image = np.round(
        random_noise(img, mode="gaussian", var=std**2) * 255
    ).astype(np.uint8)
    fig = px.imshow(noised_image)
    fig.update_layout(
        newshape=dict(
            fillcolor=color_value, opacity=1.0, line=dict(color="black", width=0)
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        dragmode="drawrect",
        yaxis_visible=False,
        yaxis_showticklabels=False,
        xaxis_visible=False,
        xaxis_showticklabels=False,
    )

    return fig


@callback(
    Output(
        component_id="fig_perturb", component_property="figure", allow_duplicate=True
    ),
    Input("color_picker", "value"),
    State(component_id="fig_perturb", component_property="figure"),
    prevent_initial_call=True,
)
def update_fig_style(color_value, figure):
    fig = go.Figure(figure)
    fig.update_layout(
        newshape=dict(
            fillcolor=color_value, opacity=1.0, line=dict(color="black", width=0)
        )
    )

    return fig


@callback(
    Output(
        component_id="fig_perturb", component_property="figure", allow_duplicate=True
    ),
    [
        State(component_id="fig_perturb", component_property="figure"),
        State(component_id="text_input_fig", component_property="value"),
        State(component_id="switche_arrow", component_property="value"),
        State(component_id="arrow_size", component_property="value"),
        State(component_id="text_size", component_property="value"),
    ],
    Input("new_button", "n_clicks"),
    prevent_initial_call=True,
)
def new_text_annotation(figure, text, arrow_value, arrow_size, text_size, n_clicks):

    fig = go.Figure(figure)

    if text:
        fig.add_annotation(
            x=fig["layout"]["xaxis"]["range"][1] / 2,
            y=fig["layout"]["yaxis"]["range"][0] / 2,
            text=text,
            font={"size": text_size},
            showarrow=True if len(arrow_value) > 0 else False,
            arrowcolor="black",
            arrowsize=arrow_size,
            arrowwidth=1,
            arrowhead=1,
        )

    return fig


@callback(
    Output(component_id="current_image", component_property="src"),
    [
        Input(component_id="button_gen", component_property="n_clicks"),
        Input(component_id="button_uncertainty", component_property="n_clicks"),
        Input(component_id="button_attention", component_property="n_clicks"),
    ],
    State(component_id="fig_perturb", component_property="figure"),
)
def update_fig_perturb(n1, n2, n3, figure):
    if all(v is None for v in (n1, n2, n3)):
        img = None
    else:
        fig = go.Figure(figure)
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
        )
        img = pil_to_b64(plotly_fig2PIL(fig))
    return img


# Right Container
RIGHT_CONTAINER = [
    dbc.CardHeader(html.H4("Current Input")),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(html.H5("Input Image")),
                            dbc.Row(html.Img(src=None, id="current_image")),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Row(html.H5("Input Context")),
                            dbc.Row(html.P(None, id="input_text_out")),
                        ],
                        width=8,
                    ),
                ],
                style={"margin-bottom": "15px"},
            ),
            dbc.Row([html.H5("Input Question")]),
            dbc.Row([html.P(None, id="input_question_out")]),
        ]
    ),
]

RIGHT_OUTPUT = [
    dcc.Loading(
        [
            dbc.CardHeader(
                [
                    html.H4(
                        "Generated Output", className="card-title", id="output_header"
                    ),
                ]
            ),
            dbc.CardBody(
                [html.P(None, className="card-text", id="output_text")],
            ),
        ],
        overlay_style={
            "visibility": "visible",
            "opacity": 0.5,
            "backgroundColor": "white",
        },
        custom_spinner=html.H4(["Generating... ", dbc.Spinner(color="warning")]),
    )
]

RIGHT_UNCERTAINTY = [
    dcc.Loading(
        [
            dbc.CardHeader(
                [
                    html.H4(
                        "Semantic Uncertainty",
                        className="card-title",
                        id="uncertainty_header",
                    ),
                ]
            ),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    "Compute Uncertainty",
                                    color="warning",
                                    className="me-1",
                                    id="button_uncertainty",
                                    style={
                                        "margin-bottom": "10px",
                                        "margin-left": "10px",
                                        "align": "left",
                                    },
                                ),
                                width=4,
                            ),
                            dbc.Col(
                                daq.NumericInput(
                                    label="Num. of Samples",
                                    id="num_uncertainty",
                                    value=10,
                                    min=1,
                                    max=50,
                                    labelPosition="top",
                                    style={
                                        "margin-bottom": "10px",
                                    },
                                ),
                                width=4,
                            ),
                            dbc.Col(
                                daq.PrecisionInput(
                                    label="Temperature",
                                    id="temp_uncertainty",
                                    value=0.9,
                                    min=0,
                                    max=1,
                                    precision=1,
                                    labelPosition="top",
                                    style={
                                        "margin-bottom": "10px",
                                    },
                                ),
                                width=4,
                            ),
                        ],
                        align="center",
                    ),
                    html.Hr(className="my-2"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.I(
                                        id="sem_entropy_icon",
                                        className="bi bi-arrows-move",
                                        style={
                                            "font-size": "50px",
                                            "height": "100%",
                                            "vertical-align": "middle",
                                        },
                                    ),
                                    dbc.Tooltip(
                                        id="sem_entropy_tooltip",
                                        children=[
                                            "Mean on VQA dataset:",
                                            html.Br(),
                                            html.B("1.21"),
                                            " (LLaVA)",
                                        ],
                                        target="sem_entropy_icon",
                                        placement="top",
                                    ),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                html.H1(
                                    id="uncertainty_score",
                                    children="-",
                                    style={
                                        "font-size": "40px",
                                        "height": "100%",
                                    },
                                ),
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    html.I(
                                        id="sem_cluster_icon",
                                        className="bi bi-grid me-2",
                                        style={"font-size": "50px"},
                                    ),
                                    dbc.Tooltip(
                                        id="sem_cluster_tooltip",
                                        children=[
                                            "Mean on VQA dataset:",
                                            html.Br(),
                                            html.B("4.90"),
                                            " (LLaVA)",
                                        ],
                                        target="sem_cluster_icon",
                                        placement="top",
                                    ),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                html.H1(
                                    id="uncertainty_cluster",
                                    children="-",
                                    style={
                                        "font-size": "40px",
                                        "height": "100%",
                                    },
                                ),
                                width=4,
                            ),
                        ],
                        align="center",
                    ),
                    dbc.Row(
                        dbc.Accordion(
                            dbc.AccordionItem(
                                None,
                                title="Answers per Cluster",
                                id="uncertainty_table",
                            ),
                            start_collapsed=True,
                        )
                    ),
                ],
            ),
        ],
        overlay_style={
            "visibility": "visible",
            "opacity": 0.7,
            "backgroundColor": "white",
        },
        custom_spinner=html.H4(["Calculating... ", dbc.Spinner(color="warning")]),
    )
]

RIGHT_ATTENTION = [
    dcc.Loading(
        [
            dbc.CardHeader(
                [
                    html.H4(
                        "Attention Attribution",
                        className="card-title",
                        id="attention_header",
                    ),
                ]
            ),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    "Compute Attention",
                                    color="warning",
                                    className="me-1",
                                    id="button_attention",
                                    style={
                                        "margin-bottom": "10px",
                                        "margin-left": "10px",
                                        "align": "left",
                                    },
                                ),
                                width=4,
                            ),
                            dbc.Col(
                                dbc.Switch(
                                    id="attention_normalize",
                                    label="Normalize",
                                    value=True,
                                ),
                                width=4,
                            ),
                        ],
                        align="center",
                    ),
                    html.Hr(className="my-2"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.I(
                                        id="atte_question_icon",
                                        className="bi bi-question-square",
                                        style={"font-size": "30px"},
                                    ),
                                    dbc.Tooltip(
                                        id="atte_question_tooltip",
                                        children=[
                                            "Mean on VQA dataset:",
                                            html.Br(),
                                            html.B("0.15"),
                                            " (LLaVA)",
                                        ],
                                        target="atte_question_icon",
                                        placement="top",
                                    ),
                                ],
                                width=1,
                            ),
                            dbc.Col(
                                html.H1(
                                    id="attention_question",
                                    children="-",
                                    style={
                                        "font-size": "30px",
                                        "height": "100%",
                                    },
                                ),
                                width=3,
                            ),
                            dbc.Col(
                                [
                                    html.I(
                                        id="atte_image_icon",
                                        className="bi bi-card-image me-2",
                                        style={
                                            "font-size": "30px",
                                            "height": "100%",
                                            "vertical-align": "middle",
                                        },
                                    ),
                                    dbc.Tooltip(
                                        id="atte_image_tooltip",
                                        children=[
                                            "Mean on VQA dataset:",
                                            html.Br(),
                                            html.B("0.28"),
                                            " (LLaVA)",
                                        ],
                                        target="atte_image_icon",
                                        placement="top",
                                    ),
                                ],
                                width=1,
                            ),
                            dbc.Col(
                                html.H1(
                                    id="attention_image",
                                    children="-",
                                    style={
                                        "font-size": "30px",
                                        "height": "100%",
                                    },
                                ),
                                width=3,
                            ),
                            dbc.Col(
                                [
                                    html.I(
                                        id="atte_context_icon",
                                        className="bi bi-card-text me-2",
                                        style={"font-size": "30px"},
                                    ),
                                    dbc.Tooltip(
                                        id="atte_context_tooltip",
                                        children=[
                                            "Mean on VQA dataset:",
                                            html.Br(),
                                            html.B("0.09"),
                                            " (LLaVA)",
                                        ],
                                        target="atte_context_icon",
                                        placement="top",
                                    ),
                                ],
                                width=1,
                            ),
                            dbc.Col(
                                html.H1(
                                    id="attention_text",
                                    children="-",
                                    style={
                                        "font-size": "30px",
                                        "height": "100%",
                                    },
                                ),
                                width=3,
                            ),
                        ],
                        align="center",
                    ),
                ],
            ),
        ],
        overlay_style={
            "visibility": "visible",
            "opacity": 0.7,
            "backgroundColor": "white",
        },
        custom_spinner=html.H4(["Calculating... ", dbc.Spinner(color="warning")]),
    )
]


# Callbacks for Right
@callback(
    Output(component_id="input_question_out", component_property="children"),
    [
        Input(component_id="button_gen", component_property="n_clicks"),
        Input(component_id="button_uncertainty", component_property="n_clicks"),
        Input(component_id="button_attention", component_property="n_clicks"),
    ],
    State(component_id="input_question", component_property="value"),
)
def update_input_question(n1, n2, n3, value):
    if any(v is not None for v in (n1, n2, n3)):
        return value


@callback(
    Output(component_id="input_text_out", component_property="children"),
    [
        Input(component_id="button_gen", component_property="n_clicks"),
        Input(component_id="button_uncertainty", component_property="n_clicks"),
        Input(component_id="button_attention", component_property="n_clicks"),
    ],
    State(component_id="input_text", component_property="value"),
)
def update_input_text(n1, n2, n3, value):
    if any(v is not None for v in (n1, n2, n3)):
        return value


@callback(
    Output(component_id="output_text", component_property="children"),
    Input(component_id="button_gen", component_property="n_clicks"),
    [
        State(component_id="input_text", component_property="value"),
        State(component_id="input_question", component_property="value"),
        State(component_id="fig_perturb", component_property="figure"),
        State(component_id="llava_selected", component_property="children"),
        State(component_id="4bit_checkbox", component_property="value"),
    ],
)
def llava_output(
    n_clicks, input_text, input_question, figure, llava_version, load_4bit
):
    if not (n_clicks == None):
        response = llava_inference(
            input_text, input_question, figure, llava_version.lower(), load_4bit
        )
        return response


@callback(
    [
        Output(component_id="uncertainty_score", component_property="children"),
        Output(component_id="uncertainty_cluster", component_property="children"),
        Output("uncertainty_table", "children"),
    ],
    Input(component_id="button_uncertainty", component_property="n_clicks"),
    [
        State(component_id="input_text", component_property="value"),
        State(component_id="input_question", component_property="value"),
        State(component_id="fig_perturb", component_property="figure"),
        State(component_id="num_uncertainty", component_property="value"),
        State(component_id="temp_uncertainty", component_property="value"),
        State(component_id="llava_selected", component_property="children"),
        State(component_id="4bit_checkbox", component_property="value"),
    ],
    prevent_initial_call=True,
)
def llava_uncertainty(
    n_clicks,
    input_text,
    input_question,
    figure,
    num_uncertainty,
    temp_uncertainty,
    llava_version,
    load_4bit,
):
    if not (n_clicks == None):
        semantic_entropy, regular_entropy, full_responses, num_clusters = (
            generate_uncertainty_score(
                input_text,
                input_question,
                figure,
                temp_uncertainty,
                num_uncertainty,
                llava_version.lower(),
                load_4bit,
            )
        )
        df_table = pd.DataFrame(
            {"Cluster": full_responses[0], "Answer": full_responses[1]}
        ).sort_values("Cluster")
        table = dbc.Table.from_dataframe(
            df_table,
            striped=True,
            bordered=True,
            hover=True,
            size="sm",
            style={"fontSize": "12px"},
        )
        return semantic_entropy, num_clusters, table


@callback(
    [
        Output(component_id="attention_question", component_property="children"),
        Output(component_id="attention_image", component_property="children"),
        Output(component_id="attention_text", component_property="children"),
    ],
    Input(component_id="button_attention", component_property="n_clicks"),
    [
        State(component_id="input_text", component_property="value"),
        State(component_id="input_question", component_property="value"),
        State(component_id="fig_perturb", component_property="figure"),
        State(component_id="llava_selected", component_property="children"),
        State(component_id="4bit_checkbox", component_property="value"),
        State(component_id="attention_normalize", component_property="value"),
    ],
    prevent_initial_call=True,
)
def llava_attention(
    n_clicks, input_text, input_question, figure, llava_version, load_4bit, norm
):
    if not (n_clicks == None):
        dict_atten = generate_attention(
            input_text, input_question, figure, llava_version.lower(), load_4bit
        )

        if norm:
            sum = dict_atten["attn_I"] + dict_atten["attn_T"] + dict_atten["attn_Q"]
            dict_atten["attn_I"] = dict_atten["attn_I"] / sum
            dict_atten["attn_T"] = dict_atten["attn_T"] / sum
            dict_atten["attn_Q"] = dict_atten["attn_Q"] / sum

        return (
            np.round(dict_atten["attn_Q"], 2),
            np.round(dict_atten["attn_I"], 2),
            np.round(dict_atten["attn_T"], 2),
        )


BODY = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(dbc.Card(LEFT_CONTAINER), width=6),
                dbc.Col(
                    [
                        dbc.Card(RIGHT_CONTAINER, style={"margin-bottom": "15px"}),
                        dbc.Card(
                            RIGHT_OUTPUT,
                            color="primary",
                            inverse=True,
                            style={"margin-bottom": "15px"},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dbc.Card(RIGHT_UNCERTAINTY), width=6),
                                dbc.Col(dbc.Card(RIGHT_ATTENTION), width=6),
                            ]
                        ),
                    ],
                    width=6,
                ),
            ],
            style={"marginTop": 30},
        ),
    ],
    className="mt-12",
    fluid=True,
)

# App layout
app.layout = html.Div(children=[NAVBAR, BODY])


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
