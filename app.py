import streamlit as st
import torch
import clip
from PIL import Image
from transformers import GPT2Tokenizer
from your_model_file import ClipCaptionModel, generate_caption  # your model utils
import io

# Load models once
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = ClipCaptionModel(prefix_length=10).to(device)
    model.load_state_dict(torch.load("coco_weights.pt", map_location=device), strict=False)
    model.eval()
    return model, clip_model, preprocess, tokenizer, device

model, clip_model, preprocess, tokenizer, device = load_models()

# Streamlit UI
st.title("📸 CLIP Prefix Captioning with Streamlit")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with torch.no_grad():
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        prefix = clip_model.encode_image(image_tensor).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, 10, -1)
        caption = generate_caption(model, tokenizer, prefix_embed)

    st.markdown("### 📝 Generated Caption")
    st.success(caption)
