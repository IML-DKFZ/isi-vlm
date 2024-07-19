import re
import pandas as pd


def get_google_sheet():
    url = "https://docs.google.com/spreadsheets/d/1ZiSutVUkybp9L87o7AKE_KyPLIoLkHhoaEi_w90a-Ls/edit?gid=0#gid=0"
    pattern = r"https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)(/edit#gid=(\d+)|/edit.*)?"
    replacement = (
        lambda m: f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?"
        + (f"gid={m.group(3)}&" if m.group(3) else "")
        + "format=csv"
    )
    new_url = re.sub(pattern, replacement, url)
    df = pd.read_csv(new_url)
    return df
