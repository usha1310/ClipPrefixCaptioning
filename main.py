# streamlit_clip_captioning.py
import os
import torch
import clip
import numpy as np
from torch import nn
import torch.nn.functional as nnf
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import skimage.io as io
import PIL.Image
import streamlit as st
from huggingface_hub import hf_hub_download
from typing import Tuple, Optional

# Load weights
@st.cache_resource
def download_weights():
    conceptual_weight = hf_hub_download(repo_id="akhaliq/CLIP-prefix-captioning-conceptual-weights", filename="conceptual_weights.pt")
    coco_weight = hf_hub_download(repo_id="akhaliq/CLIP-prefix-captioning-COCO-weights", filename="coco_weights.pt")
    return conceptual_weight, coco_weight

conceptual_weight, coco_weight = download_weights()

# Device setup
CPU = torch.device('cpu')
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MLP class
class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Main CLIP Caption model
class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super().__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = (
            nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
            if prefix_length > 10
            else MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))
        )

    def forward(self, tokens, prefix):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_proj = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_proj, embedding_text), dim=1)
        return self.gpt(inputs_embeds=embedding_cat)

# Caption generation
@torch.no_grad()
def generate_caption(model, tokenizer, embed, entry_length=67, top_p=0.8, temperature=1., stop_token: str = '.'):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    generated = embed
    tokens = None
    for _ in range(entry_length):
        outputs = model.gpt(inputs_embeds=generated)
        logits = outputs.logits[:, -1, :] / temperature
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        logits[:, sorted_indices[sorted_indices_to_remove]] = filter_value
        next_token = torch.argmax(logits, -1).unsqueeze(0)
        next_token_embed = model.gpt.transformer.wte(next_token)
        tokens = next_token if tokens is None else torch.cat((tokens, next_token), dim=1)
        generated = torch.cat((generated, next_token_embed), dim=1)
        if stop_token_index == next_token.item():
            break
    return tokenizer.decode(tokens.squeeze().cpu().numpy().tolist())

# Streamlit UI
st.set_page_config(page_title="📸 CLIP Prefix Captioning", layout="wide")
st.title("📸 CLIP Prefix Captioning")
st.markdown("""Upload an image and choose a pre-trained model (COCO / Conceptual) to generate a natural language caption using a transformer-based architecture.""")

# User input
uploaded_file = st.file_uploader("Upload Image (JPG, PNG)", type=["jpg", "jpeg", "png"])
model_choice = st.radio("Choose Pretrained Model", ["COCO", "Conceptual"], index=0)

if uploaded_file:
    device = get_device()
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    prefix_length = 10
    model = ClipCaptionModel(prefix_length)
    model_path = coco_weight if model_choice == "COCO" else conceptual_weight
    model.load_state_dict(torch.load(model_path, map_location=CPU), strict=False)
    model = model.to(device).eval()

    image = PIL.Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image_tensor).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        caption = generate_caption(model, tokenizer, embed=prefix_embed)

    st.markdown("### 📝 Generated Caption")
    st.success(caption)
else:
    st.info("Please upload an image to generate a caption.")