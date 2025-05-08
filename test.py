import os
from huggingface_hub import hf_hub_download
import clip
import numpy as np
import torch
from torch import nn
import torch.nn.functional as nnf
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import trange
import skimage.io as io
import PIL.Image
import gradio as gr
from typing import Tuple, List, Union, Optional

# Download weights
conceptual_weight = hf_hub_download(repo_id="akhaliq/CLIP-prefix-captioning-conceptual-weights", filename="conceptual_weights.pt")
coco_weight = hf_hub_download(repo_id="akhaliq/CLIP-prefix-captioning-COCO-weights", filename="coco_weights.pt")

# Device configuration
CPU = torch.device('cpu')
def get_device(device_id: int = 0) -> torch.device:
    return torch.device(f'cuda:{device_id}') if torch.cuda.is_available() else CPU

# Model classes
class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_proj = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_proj, embedding_text), dim=1)
        return self.gpt(inputs_embeds=embedding_cat)

# Caption generation function
def generate_caption(model, tokenizer, embed, entry_length=67, top_p=0.8, temperature=1., stop_token: str = '.'):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    generated = embed
    tokens = None
    with torch.no_grad():
        for _ in trange(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :] / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = model.gpt.transformer.wte(next_token)
            tokens = next_token if tokens is None else torch.cat((tokens, next_token), dim=1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            if stop_token_index == next_token.item():
                break
        output_list = tokens.squeeze().cpu().numpy().tolist()
        return tokenizer.decode(output_list)

# Inference handler
def inference(image_path, model_choice):
    prefix_length = 10
    device = get_device()
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model_path = coco_weight if model_choice == "COCO" else conceptual_weight
    model = ClipCaptionModel(prefix_length)
    model.load_state_dict(torch.load(model_path, map_location=CPU), strict=False)
    model = model.to(device).eval()

    image = io.imread(image_path)
    image = PIL.Image.fromarray(image)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

    caption = generate_caption(model, tokenizer, embed=prefix_embed)
    return caption

# Gradio Interface Setup
title = "📸 CLIP Prefix Captioning"
description="""
        📸 Image-to-Text Generation with Transformers

        CLIP Prefix Captioning is a powerful AI application that generates natural language descriptions from images by combining the strengths of OpenAI’s CLIP for image understanding and GPT-2 for language generation. This fusion enables the system to describe diverse image content in fluent, human-like captions.

        🚀 What This App Does:

        🖼️ Input: A photo or image (JPG, PNG).

        🧠 Process: Extract image embeddings using the CLIP ViT-B/32 model.

        ✍️ Generate: Feed visual context into a GPT-2 language model to produce rich, descriptive captions.

        🎯 Result: A coherent, grammatically correct caption that summarizes the image.

        🔧 Models Used
        Component	Model	Purpose
        CLIP (ViT-B/32)	clip	Extract visual features from images
        GPT-2	gpt2	Generate human-like captions from embeddings
        Prefix Mapping MLP	Custom lightweight MLP	Bridges CLIP and GPT2 by projecting embeddings
        🧪 Pretrained Weight Options
        Model	Dataset	Characteristics
        COCO Weights	MS-COCO	General object captions; more structured and localized
        Conceptual Weights	Conceptual Captions	Web-scale dataset; diverse, abstract, and expressive

        **Use COCO model for common objects (water, dogs, people, cars). Use Conceptual model for abstract scenes, aesthetic photos.**
"""

gr.Interface(
    fn=inference,
    inputs=[
        gr.Image(type="filepath", label="Upload Image (JPG, PNG)"),
        gr.Radio(choices=["COCO", "Conceptual"], label="Choose Pretrained Model", value="COCO")
    ],
    outputs=gr.Textbox(label="Generated Caption"),
    title=title,
    description=description,
    theme="default",
    allow_flagging="never"
).launch(debug=True, share=True)