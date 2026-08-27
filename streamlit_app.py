import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
import joblib
import os
import gc
import traceback

torch.set_num_threads(1)  # evita que torch sature CPU/RAM en el plan gratuito

st.set_page_config(page_title="Blastocisto IA", page_icon="\U0001f9ec", layout="wide")


class MultiHeadEfficientNet(nn.Module):
    def __init__(self, num_exp=5, num_icm=4, num_te=4):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()
        num_features = 1280
        self.fc_exp = nn.Linear(num_features, num_exp)
        self.fc_icm = nn.Linear(num_features, num_icm)
        self.fc_te = nn.Linear(num_features, num_te)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_exp(features), self.fc_icm(features), self.fc_te(features)


class CombinedModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(input_dim, 1))

    def forward(self, x):
        return self.fc(x).squeeze(1)


@st.cache_resource(show_spinner=False)
def load_models():
    try:
        device = torch.device("cpu")

        archivos_necesarios = ["modelo_multi.safetensors", "modelo_combinado.safetensors", "scaler.pkl"]
        for f in archivos_necesarios:
            if not os.path.exists(f):
                raise FileNotFoundError(f"No se encuentra el archivo: {f}")

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        from safetensors.torch import load_file

        multi_model = MultiHeadEfficientNet().to(device)
        multi_model.load_state_dict(load_file("modelo_multi.safetensors"))
        multi_model.eval()

        backbone = multi_model.backbone
        backbone.eval()

        combined_model = CombinedModel(1282).to(device)
        combined_model.load_state_dict(load_file("modelo_combinado.safetensors"))
        combined_model.eval()

        scaler = joblib.load("scaler.pkl")
        gc.collect()

        return multi_model, backbone, combined_model, scaler, transform, device

    except Exception as e:
        traceback.print_exc()
        raise e


st.title("\U0001f9ec Blastocisto IA")
st.markdown(
    "Esta aplicacion predice los **scores Gardner** (EXP, ICM, TE) y la **probabilidad de nacido vivo (LB)**\n"
    "a partir de una imagen de blastocisto (dia 5) y datos clinicos (edad materna y latido fetal HA)."
)

with st.spinner("Cargando modelos, por favor espera..."):
    multi_model, backbone, combined_model, scaler, transform, device = load_models()
st.success("Modelos cargados correctamente")

col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.subheader("Imagen y datos clinicos")
    uploaded_file = st.file_uploader("Selecciona una imagen PNG o JPG", type=["png", "jpg", "jpeg"])
    edad = st.number_input("Edad materna", min_value=18, max_value=50, value=30, step=1)
    ha = st.selectbox("Latido fetal (HA)", options=[0, 1], format_func=lambda x: "Si (1)" if x == 1 else "No (0)")
    predecir_btn = st.button("Predecir", type="primary", use_container_width=True)

with col_der:
    st.subheader("Resultados")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        del file_bytes

        if image is None:
            st.error("No se pudo leer la imagen. Intenta con otro archivo.")
            st.stop()

        try:
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            st.error(f"Error en conversion de color: {e}")
            st.stop()

        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)

                try:
            st.image(image_rgb, caption="Imagen cargada", use_column_width=True)
        except Exception as e:
            st.error(f"Error al mostrar la imagen: {e}")
            st.stop()

        if predecir_btn:
            with st.spinner("Procesando imagen y calculando..."):
                try:
                    img_tensor = transform(image_rgb).unsqueeze(0).to(device)

                    with torch.inference_mode():
                        exp, icm, te = multi_model(img_tensor)
                        exp_class = exp.argmax(dim=1).item()
                        icm_class = icm.argmax(dim=1).item()
                        te_class = te.argmax(dim=1).item()
                        features = backbone(img_tensor).cpu().numpy().flatten()

                    clin_data = np.array([[edad, ha]], dtype=np.float32)
                    clin_scaled = scaler.transform(clin_data).flatten()

                    combined_input = np.concatenate([features, clin_scaled])
                    combined_tensor = torch.tensor(combined_input, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        logit = combined_model(combined_tensor)
                        prob_lb = torch.sigmoid(logit).item()

                    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                    col_res1.metric("EXP", exp_class)
                    col_res2.metric("ICM", icm_class)
                    col_res3.metric("TE", te_class)
                    col_res4.metric("Prob. LB", f"{prob_lb:.1%}")

                    if prob_lb > 0.5:
                        st.success(f"Probabilidad de nacido vivo: **{prob_lb:.1%}**")
                    else:
                        st.warning(f"Probabilidad de nacido vivo: **{prob_lb:.1%}**")

                    del img_tensor, features, combined_input, combined_tensor
                    gc.collect()

                except Exception as e:
                    st.error(f"Error durante la prediccion: {e}")
                    st.error(traceback.format_exc())
    else:
        st.info("Sube una imagen para comenzar.")

st.markdown("---")
st.markdown(
    "**Notas:**\n"
    "- **EXP**: 0-4, **ICM** y **TE**: 0-2 (segun sistema Gardner modificado).\n"
    "- **HA**: 0 = sin latido fetal, 1 = con latido fetal.\n"
    "- El modelo de imagen se basa en EfficientNet-B0 entrenado con mas de 2000 anotaciones.\n"
    "- La probabilidad de LB combina caracteristicas de imagen + edad + HA."
)
