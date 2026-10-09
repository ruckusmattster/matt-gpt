"""Gradio web UI for MattGPT. This is the file deployed to the Hugging Face Space.

Local:   python app.py                       (uses weights/weights.pth, or downloads it)
Space:   copy app.py, the mattgpt/ folder and requirements.txt next to weights.pth
"""

from __future__ import annotations

import os

import gradio as gr
import torch

from mattgpt import load_checkpoint

CKPT_CANDIDATES = ["weights.pth", "weights/weights.pth"]


def find_or_download_weights() -> str:
    for path in CKPT_CANDIDATES:
        if os.path.exists(path):
            return path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id="mattyhew/mattgpt", repo_type="space", filename="weights.pth")


device = "cuda" if torch.cuda.is_available() else "cpu"
model, tok, _ = load_checkpoint(find_or_download_weights(), device=device)


def generate(prompt: str, max_new_tokens: float, temperature: float, top_k: float, top_p: float) -> str:
    ids = tok.encode(prompt.strip()) or tok.encode("\n")  # empty / all-unknown prompt -> newline
    ctx = torch.tensor([ids], dtype=torch.long, device=device)
    max_new_tokens = max(1, min(int(max_new_tokens), 2000))  # gr.Number returns a float
    out = model.generate(ctx, max_new_tokens, temperature=temperature, top_k=int(top_k), top_p=top_p)
    return tok.decode(out[0].tolist())


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Starting context", placeholder="Enter your prompt here..."),
        gr.Number(label="Maximum output characters", value=400, precision=0),
        gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature"),
        gr.Slider(0, 100, value=50, step=1, label="Top-k (0 = off)"),
        gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p / nucleus (1 = off)"),
    ],
    outputs=gr.Textbox(label="Response"),
    title="mattgpt",
    article="i tell stories",
)

if __name__ == "__main__":
    demo.launch()
