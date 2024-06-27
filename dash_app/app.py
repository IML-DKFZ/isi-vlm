# Import packages
import dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, ctx
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import dash_daq as daq
import skimage
import torch
import os
import sys

sys.path.append(os.getcwd())

from dash_app.utils.run_llava import llava_inference
from llava.data_utils.set_seed import set_seed

from dash_app.utils.image_export import plotly_fig2PIL, pil_to_b64

set_seed(42)


# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

files = pd.read_csv("./dash_app/assets/files.csv")

options = [
    {"label": i, "value": j} for i, j in zip(files["name"], range(len(files["name"])))
]

NAVBAR = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(
                                src="./assets/eth_logo_kurz_neg.png",
                                height="26px",
                                style={"margin-right": "10px"},
                            )
                        ),
                        dbc.Col(html.H1("|", style={"color": "white"})),
                        dbc.Col(
                            html.Img(
                                src="./assets/ibm_logo.png",
                                height="24px",
                                style={"margin-right": "35px", "margin-left": "10px"},
                            )
                        ),
                        dbc.Col(
                            dbc.NavbarBrand(
                                "LLaVa Interactive Perturbations",
                                style={"font-size": 25},
                                className="ms-2",
                            )
                        ),
                    ],
                    align="center",
                    className="g-0",
                ),
            ),
        ],
        fluid=True,
    ),
    color="primary",
    dark=True,
)

LEFT_CONTAINER = [
    dbc.CardHeader(html.H4("New Perturbed Input")),
    dbc.CardBody(
        [
            dbc.Row(
                html.Div(
                    [
                        html.P("Select observation:"),
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
            ),
            dbc.Row(
                html.H4("Input Image"),
            ),
            dbc.Row(
                [
                    dbc.Col(
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
                                "margin-bottom": "20px",
                            },
                        ),
                        width=9,
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                dbc.Col(
                                    [
                                        html.H5("Color Picker"),
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
                                        html.H5("Text Annotation"),
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
                                            ]
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
                                            ]
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
                                            ]
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
                                    style={"display": "inline-block", "float": "left"},
                                )
                            ),
                        ],
                        width=3,
                    ),
                ]
            ),
            dbc.Row(html.H4("Input Text")),
            dbc.Row(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Prompt: "),
                        dbc.Textarea(placeholder={}, id="input_text"),
                    ],
                    className="mb-3",
                )
            ),
            dbc.Row(html.H4("Input Question")),
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


# Callbacks for left
@callback(
    Output(component_id="input_question", component_property="value"),
    Input(component_id="dropdown_obs", component_property="value"),
)
def update_input_text(value):
    text = files["input_question"].iloc[value]
    return text


@callback(
    Output(component_id="input_text", component_property="value"),
    Input(component_id="dropdown_obs", component_property="value"),
)
def update_input_text(value):
    text = files["input_text"].iloc[value]
    return text


@callback(
    Output(component_id="fig_perturb", component_property="figure"),
    Input(component_id="dropdown_obs", component_property="value"),
    State("color_picker", "value"),
)
def fig_perturb(value, color_value):

    fig = px.imshow(
        skimage.io.imread("./dash_app/assets/" + files["input_image"].iloc[value])
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
    Input(component_id="button_gen", component_property="n_clicks"),
    State(component_id="fig_perturb", component_property="figure"),
)
def update_fig_perturb(n_clicks, figure):
    if not n_clicks:
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
                            dbc.Row(html.H5("Input Text")),
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


# Callbacks for Right
@callback(
    Output(component_id="input_question_out", component_property="children"),
    Input(component_id="button_gen", component_property="n_clicks"),
    State(component_id="input_question", component_property="value"),
)
def update_input_text(n_clicks, value):
    if not (n_clicks == None):
        return value


@callback(
    Output(component_id="input_text_out", component_property="children"),
    Input(component_id="button_gen", component_property="n_clicks"),
    State(component_id="input_text", component_property="value"),
)
def update_input_text(n_clicks, value):
    if not (n_clicks == None):
        return value


@callback(
    Output(component_id="output_text", component_property="children"),
    Input(component_id="button_gen", component_property="n_clicks"),
    [
        State(component_id="input_text", component_property="value"),
        State(component_id="input_question", component_property="value"),
        State(component_id="fig_perturb", component_property="figure"),
    ],
)
def llava_output(n_clicks, input_text, input_question, figure):
    if not (n_clicks == None):
        response = llava_inference(input_text, input_question, figure)
        return response


BODY = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(dbc.Card(LEFT_CONTAINER), width=6),
                dbc.Col(
                    [
                        # dbc.Row(
                        dbc.Card(RIGHT_CONTAINER, style={"margin-bottom": "15px"}),
                        # ),
                        dbc.Card(RIGHT_OUTPUT, color="primary", inverse=True),
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
