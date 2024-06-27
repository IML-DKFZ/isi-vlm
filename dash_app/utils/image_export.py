from PIL import Image
import numpy as np
import base64
from io import BytesIO


def plotly_fig2PIL(fig):
    fig_bytes = fig.to_image(format="png")
    buf = BytesIO(fig_bytes)
    img = Image.open(buf)
    return img


def pil_to_b64(im, ext="png"):
    buffer = BytesIO()
    im.save(buffer, format=ext)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/{ext};base64, " + encoded
