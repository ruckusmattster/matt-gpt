"""Gradio web UI for MattGPT. This is the app the Hugging Face Space runs.

    python app.py

Uses weights/mattgpt.pth, downloading it from GitHub on first start if missing
(that's how the Space gets it). Set MATTGPT_WEIGHTS to use another checkpoint.
"""

from __future__ import annotations

import gradio as gr
import torch

from mattgpt import load_checkpoint
from mattgpt.weights import default_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tok, _ = load_checkpoint(default_weights(), device=device)

MAX_CHARS = 2000


def generate(prompt: str, max_new_tokens: float, temperature: float, top_k: float, top_p: float) -> str:
    ids = tok.encode(prompt) or tok.encode("\n")  # empty / all-unknown prompt -> newline
    ctx = torch.tensor([ids], dtype=torch.long, device=device)
    n = max(1, min(int(max_new_tokens or 1), MAX_CHARS))  # gr.Number can return float or None
    out = model.generate(ctx, n, temperature=temperature, top_k=int(top_k), top_p=top_p)
    return tok.decode(out[0].tolist())


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Starting context", placeholder="anyone know a good upscaler"),
        gr.Number(label="Characters to generate", value=400, precision=0, minimum=1, maximum=MAX_CHARS),
        gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature (lower = safer, higher = wilder)"),
        gr.Slider(0, 100, value=40, step=1, label="Top-k (0 = off)"),
        gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p / nucleus (1 = off)"),
    ],
    outputs=gr.Textbox(label="Response"),
    title="mattgpt",
    description="A 7.5M-parameter character-level GPT trained from scratch on ~74k of my Discord messages. "
                "It talks like me about 3D printers, flashlights and Stable Diffusion, and makes about as much sense.",
    article="Code, data and training details: https://github.com/ruckusmattster/matt-gpt",
    examples=[
        ["anyone know a good upscaler", 300, 0.8, 40, 1.0],
        ["what flashlight should i get", 300, 0.8, 40, 1.0],
        ["lol", 300, 0.8, 40, 1.0],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
