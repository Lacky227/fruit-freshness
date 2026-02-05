import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

MODEL_PATH = "fruit_freshness_model.pth"
CLASS_NAMES = [
    'fresh_apple', 'fresh_banana', 'fresh_bitter_gourd', 'fresh_capsicum',
    'fresh_orange', 'fresh_tomato', 'stale_apple', 'stale_banana',
    'stale_bitter_gourd', 'stale_capsicum', 'stale_orange', 'stale_tomato'
]

st.set_page_config(
    page_title="Fruit Freshness AI",
    page_icon="🍎",
    layout="centered"
)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error(f"Файл моделі '{MODEL_PATH}' не знайдено! Покладіть його поруч з app.py")
        return None, device

model, device = load_model()

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0).to(device)

st.title("🍎 Перевірка свіжості фруктів")
st.write("Завантажте фото фрукта або овоча, і ШІ визначить його стан.")

uploaded_file = st.file_uploader("Оберіть зображення...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Завантажене фото', use_container_width=True)

    if st.button('Аналізувати 🔍'):
        with st.spinner('Нейромережа думає...'):
            img_tensor = process_image(image)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                top_p, top_class = probs.topk(1, dim=1)

            score = top_p.item() * 100
            label = CLASS_NAMES[top_class.item()]

            status = "SFRESH" if "fresh" in label else "STALE"
            fruit_name = (
                label.replace("fresh_", "")
                     .replace("stale_", "")
                     .replace("_", " ")
                     .title()
            )

            st.divider()

            if "fresh" in label:
                st.success(f"### Результат: СВІЖИЙ ({fruit_name}) ✅")
            else:
                st.error(f"### Результат: ЗІПСОВАНИЙ ({fruit_name}) ⚠️")

            st.info(f"Впевненість моделі: **{score:.2f}%**")

            st.write("---")
            st.write("📊 Деталі ймовірностей:")

            top3_prob, top3_idx = probs.topk(3, dim=1)

            for i in range(3):
                idx = top3_idx[0][i].item()
                prob = top3_prob[0][i].item()
                class_name = CLASS_NAMES[idx]
                st.progress(prob, text=f"{class_name}: {prob*100:.1f}%")

st.markdown("---")
st.caption("Розроблено на PyTorch & Streamlit")
