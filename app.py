import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Pneumonia Detection", page_icon="lungs", layout="centered")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["No Pneumonia", "Pneumonia"]
MODEL_PATH = "best_pneumonia_model.pth"

def build_resnet50(num_classes=2):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return model

@st.cache_resource
def load_model():
    model = build_resnet50()
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image, model):
    img = image.convert("L")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
    return CLASS_NAMES[idx], float(probs[idx].item()), probs.cpu().numpy()

st.title("Pneumonia Detection from Chest X-ray")
st.markdown("Upload a chest X-ray (PNG/JPG). Model predicts No Pneumonia or Pneumonia.")
model = load_model()
uploaded = st.file_uploader("Choose an X-ray image", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Predict", type="primary"):
        with st.spinner("Running inference..."):
            pred_class, confidence, all_probs = predict(image, model)
        st.subheader("Result")
        if pred_class == "Pneumonia":
            st.error("Predicted: " + pred_class)
        else:
            st.success("Predicted: " + pred_class)
        st.metric("Confidence", f"{confidence*100:.1f}%")
        st.write("Class probabilities")
        st.progress(float(all_probs[0]), text=f"No Pneumonia: {all_probs[0]*100:.1f}%")
        st.progress(float(all_probs[1]), text=f"Pneumonia: {all_probs[1]*100:.1f}%")
st.markdown("---")
st.caption("Model: ResNet50 (TL) | Capstone Pneumonia Detection | Great Learning PGP-AIML")
