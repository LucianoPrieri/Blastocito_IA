import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
import joblib
import os
import sys
import traceback

# ------------------------------------------------------------------
# 1. Configuración de la página (DEBE SER LO PRIMERO)
# ------------------------------------------------------------------
st.set_page_config(page_title="Blastocisto IA", page_icon="🧬", layout="wide")

# ------------------------------------------------------------------
# 2. Definición de los modelos (igual que en el entrenamiento)
# ------------------------------------------------------------------
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
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze(1)

# ------------------------------------------------------------------
# 3. Carga de modelos con caché (solo prints, sin st.write)
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    """
    Carga los modelos, el backbone y el escalador.
    Usa print para logs en consola (no interfiere con st.set_page_config).
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"⚙️ Dispositivo detectado: {device}")

        # Verificar que los archivos existen
        archivos_necesarios = [
            'modelo_multi.safetensors', 'modelo_combinado.safetensors', 'scaler.pkl'
        ]
        for f in archivos_necesarios:
            if not os.path.exists(f):
                raise FileNotFoundError(f"❌ No se encuentra el archivo: {f}")

        # Transformaciones
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Cargar modelo multi-cabeza (safetensors)
        from safetensors.torch import load_file
        multi_model = MultiHeadEfficientNet().to(device)
        multi_weights = load_file('modelo_multi.safetensors')
        multi_model.load_state_dict(multi_weights)
        multi_model.eval()
        print("✅ modelo_multi.safetensors cargado")

        backbone = multi_model.backbone
        backbone.eval()

        # Cargar modelo combinado
        combined_model = CombinedModel(1282).to(device)
        combined_weights = load_file('modelo_combinado.safetensors')
        combined_model.load_state_dict(combined_weights)
        combined_model.eval()
        print("✅ modelo_combinado.safetensors cargado")

        # Cargar escalador
        scaler = joblib.load('scaler.pkl')
        print("✅ scaler.pkl cargado")

        return multi_model, backbone, combined_model, scaler, transform, device

    except Exception as e:
        print(f"❌ Error al cargar modelos: {e}")
        traceback.print_exc()
        raise e  # Re-lanza para que Streamlit muestre el error

# ------------------------------------------------------------------
# 4. Interfaz de la aplicación
# ------------------------------------------------------------------
st.title("🧬 Blastocisto IA")
st.markdown("""
Esta aplicación predice los **scores Gardner** (EXP, ICM, TE) y la **probabilidad de nacido vivo (LB)**  
a partir de una imagen de blastocisto (día 5) y datos clínicos (edad materna y latido fetal HA).
""")

# Cargar modelos (con barra de progreso)
with st.spinner("Cargando modelos, por favor espera..."):
    multi_model, backbone, combined_model, scaler, transform, device = load_models()
st.success("✅ Modelos cargados correctamente")

# Crear columnas para organizar la entrada y la salida
col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.subheader("📤 Imagen y datos clínicos")
    uploaded_file = st.file_uploader("Selecciona una imagen PNG o JPG", type=["png", "jpg", "jpeg"])
    edad = st.number_input("Edad materna", min_value=18, max_value=50, value=30, step=1)
    ha = st.selectbox("Latido fetal (HA)", options=[0, 1], format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")
    predecir_btn = st.button("🔍 Predecir", type="primary", use_container_width=True)

with col_der:
    st.subheader("📊 Resultados")
    if uploaded_file is not None:
        # Leer la imagen
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)  # Lee con todos los canales
        if image is None:
            st.error("❌ No se pudo leer la imagen. Intenta con otro archivo.")
            st.stop()

        # Convertir a RGB de 3 canales (maneja diferentes formatos)
        if len(image.shape) == 2:  # escala de grises
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:  # asumimos BGR de 3 canales
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mostrar información de depuración (puedes eliminar esta línea después de verificar)
        st.write(f"Forma de la imagen: {image_rgb.shape}, tipo: {image_rgb.dtype}")

        st.image(image_rgb, caption="Imagen cargada", use_container_width=True)

        if predecir_btn:
            with st.spinner("Procesando imagen y calculando..."):
                try:
                    # Preprocesar imagen
                    img_tensor = transform(image_rgb).unsqueeze(0).to(device)

                    # --- Predicción de scores Gardner ---
                    with torch.no_grad():
                        exp, icm, te = multi_model(img_tensor)
                        exp_class = exp.argmax(dim=1).item()
                        icm_class = icm.argmax(dim=1).item()
                        te_class = te.argmax(dim=1).item()

                    # --- Extracción de características del backbone ---
                    with torch.no_grad():
                        features = backbone(img_tensor).cpu().numpy().flatten()

                    # --- Preparar datos clínicos escalados ---
                    clin_data = np.array([[edad, ha]], dtype=np.float32)
                    clin_scaled = scaler.transform(clin_data).flatten()

                    # --- Concatenar y predecir LB ---
                    combined_input = np.concatenate([features, clin_scaled])
                    combined_tensor = torch.tensor(combined_input, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logit = combined_model(combined_tensor)
                        prob_lb = torch.sigmoid(logit).item()

                    # --- Mostrar resultados en columnas ---
                    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                    col_res1.metric("EXP", exp_class)
                    col_res2.metric("ICM", icm_class)
                    col_res3.metric("TE", te_class)
                    col_res4.metric("Prob. LB", f"{prob_lb:.1%}")

                    # Interpretación
                    if prob_lb > 0.5:
                        st.success(f"✅ Probabilidad de nacido vivo: **{prob_lb:.1%}**")
                    else:
                        st.warning(f"⚠️ Probabilidad de nacido vivo: **{prob_lb:.1%}**")

                except Exception as e:
                    st.error(f"❌ Error durante la predicción: {e}")
                    st.error(traceback.format_exc())
    else:
        st.info("👈 Sube una imagen para comenzar.")
# ------------------------------------------------------------------
# 5. Pie de página
# ------------------------------------------------------------------
st.markdown("---")
st.markdown("""
**Notas:**
- **EXP**: 0‑4, **ICM** y **TE**: 0‑2 (según sistema Gardner modificado).
- **HA**: 0 = sin latido fetal, 1 = con latido fetal.
- El modelo de imagen se basa en EfficientNet‑B0 entrenado con más de 2000 anotaciones.
- La probabilidad de LB combina características de imagen + edad + HA.
""")
