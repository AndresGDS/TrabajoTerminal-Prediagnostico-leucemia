"""
app.py

Dashboard Streamlit del sistema HemVLM: sube una imagen de celula (sana o
leucemica) y genera su descripcion morfologica con el modelo ya entrenado.
Permite elegir entre las dos variantes de entrenamiento (2 etapas / 1 paso)
para comparar sus resultados.

Uso local (PowerShell):
    streamlit run app.py

Para Hugging Face Spaces: sube este archivo junto con requirements.txt y
la(s) carpeta(s) de modelo (hemblip_base_final_fixed, etc.) al repo del
Space, o sube el modelo aparte a un repo de modelo en el Hub y cambia
MODEL_OPTIONS para apuntar al repo_id en vez de una ruta local.
"""

import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


# --------------------------------------------------------------------------- #
# Configuracion -- ajusta estas rutas segun donde queden tus modelos finales
# --------------------------------------------------------------------------- #
MODEL_OPTIONS = {
    "2 etapas (sanas -> leucemicas)": "./checkpoints/hemblip_base/hemblip_base_final_fixed",
    "1 paso (mezclado, fiel al paper)": "./checkpoints/hemblip_base_single_stage/hemblip_base_single_stage_final",
}

st.set_page_config(page_title="HemVLM -- Prediagnostico de leucemia", layout="centered")


@st.cache_resource(show_spinner="Cargando modelo...")
def load_model(model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(model_dir)
    model = BlipForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()
    return model, processor, device


def generate_caption(model, processor, device, image, max_length=64, num_beams=4):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
        )
    return processor.decode(output_ids[0], skip_special_tokens=True)


def main():
    st.title("HemVLM -- Prediagnostico de leucemia")
    st.caption(
        "Prototipo academico. Genera descripciones morfologicas de celulas de "
        "sangre periferica a partir de una imagen de microscopio. No reemplaza "
        "el diagnostico de un hematologo."
    )

    with st.sidebar:
        st.header("Configuracion")
        model_label = st.selectbox("Variante del modelo", list(MODEL_OPTIONS.keys()))
        max_length = st.slider("Longitud maxima del caption", 16, 128, 64, step=8)
        num_beams = st.slider("Numero de beams (calidad vs. velocidad)", 1, 8, 4)

    model_dir = MODEL_OPTIONS[model_label]

    try:
        model, processor, device = load_model(model_dir)
    except Exception as e:
        st.error(
            f"No se pudo cargar el modelo desde '{model_dir}'. "
            f"Revisa que la carpeta exista y tenga los archivos de save_pretrained(). "
            f"Detalle: {e}"
        )
        return

    st.caption(f"Dispositivo activo: {device}")

    uploaded_file = st.file_uploader(
        "Sube una imagen de celula (microscopia, frotis de sangre periferica)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.image(image, caption="Imagen cargada", use_container_width=True)

        with col2:
            if st.button("Generar descripcion", type="primary"):
                with st.spinner("Generando descripcion..."):
                    caption = generate_caption(
                        model, processor, device, image,
                        max_length=max_length, num_beams=num_beams,
                    )
                st.success("Descripcion generada")
                st.write(caption)
                st.caption(f"Modelo usado: {model_label}")
    else:
        st.info("Sube una imagen para comenzar.")


if __name__ == "__main__":
    main()
