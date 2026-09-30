r"""
frontend_app.py

Frontend Streamlit del sistema HemVLM. Consume el backend FastAPI (api.py)
para generar descripciones morfologicas reales sobre imagenes de celulas.

Uso (PowerShell), en una SEGUNDA terminal (con api.py ya corriendo en la
primera):
    streamlit run frontend_app.py
"""

import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/analyze"
HEALTH_URL = "http://127.0.0.1:8000/health"

st.set_page_config(
    page_title="HemVLM -- Prediagnostico de leucemia",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Estilo editorial oscuro
# --------------------------------------------------------------------------- #
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1116;
        color: #E8E6E1;
    }
    .titulo-principal {
        font-size: 2.6rem;
        color: #E8E6E1;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0px;
        font-family: 'Georgia', serif;
    }
    .sub-titulo {
        font-size: 1.05rem;
        color: #B0473F;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    .aviso-medico {
        font-size: 0.85rem;
        color: #8A8D93;
        font-style: italic;
        border-left: 3px solid #B0473F;
        padding-left: 0.8rem;
        margin-bottom: 1.5rem;
    }
    .caja-resultado {
        background-color: #171B22;
        border: 1px solid #2A2F3A;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .etiqueta-diagnostico {
        font-size: 1.1rem;
        font-weight: 700;
        color: #D9776E;
    }
    section[data-testid="stSidebar"] {
        background-color: #12151B;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🩺 Panel clinico")
    st.info(
        "**Protocolo de captura:** imagen de frotis de sangre periferica, "
        "tincion Wright-Giemsa, aumento 100x recomendado."
    )
    st.divider()
    try:
        health = requests.get(HEALTH_URL, timeout=3).json()
        st.success(f"Backend activo -- dispositivo: {health.get('device', '?')}")
        st.caption(f"Modelo: {health.get('modelo', 'desconocido')}")
    except requests.exceptions.RequestException:
        st.error("No se pudo conectar con la API (api.py). ¿Esta corriendo?")
    st.divider()
    st.caption(
        "Trabajo Terminal 2027-A140 -- ESCOM, IPN. Sistema de apoyo, "
        "no reemplaza el diagnostico de un hematologo."
    )

# --------------------------------------------------------------------------- #
# Encabezado
# --------------------------------------------------------------------------- #
st.markdown('<p class="titulo-principal">HemVLM</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-titulo">Modelo de lenguaje y vision para el prediagnostico de leucemia</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="aviso-medico">Herramienta de apoyo academico. Todo hallazgo generado por este '
    'sistema debe ser validado por un hematologo antes de cualquier decision clinica.</p>',
    unsafe_allow_html=True,
)

col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.subheader("1. Imagen de la muestra")
    archivo_subido = st.file_uploader(
        "Sube la imagen de microscopio (frotis de sangre periferica)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if archivo_subido:
        imagen = Image.open(archivo_subido)
        st.image(imagen, caption="Muestra cargada", use_container_width=True)

with col_der:
    st.subheader("2. Analisis morfologico")

    if archivo_subido:
        st.caption("Presiona el boton para generar la descripcion con HemVLM.")

        if st.button("Generar descripcion morfologica", type="primary", use_container_width=True):
            with st.spinner("Analizando morfologia celular..."):
                try:
                    archivos = {
                        "file": (archivo_subido.name, archivo_subido.getvalue(), "image/jpeg")
                    }
                    respuesta = requests.post(API_URL, files=archivos, timeout=60)

                    if respuesta.status_code == 200:
                        datos = respuesta.json()
                        st.success("Analisis completado.")

                        st.markdown('<div class="caja-resultado">', unsafe_allow_html=True)
                        st.markdown(
                            f'<p class="etiqueta-diagnostico">{datos["diagnostico_detectado"]}</p>',
                            unsafe_allow_html=True,
                        )
                        st.write(datos["descripcion"])
                        st.caption(f"Confianza del texto generado: {datos.get('confianza', 'no determinado')}")
                        st.caption(f"Modelo: {datos.get('modelo', '')}")
                        st.markdown('</div>', unsafe_allow_html=True)

                        reporte = (
                            f"HemVLM -- Reporte de analisis morfologico\n"
                            f"{'=' * 50}\n\n"
                            f"Diagnostico detectado: {datos['diagnostico_detectado']}\n\n"
                            f"Descripcion generada:\n{datos['descripcion']}\n\n"
                            f"Confianza: {datos.get('confianza', 'no determinado')}\n\n"
                            f"Modelo: {datos.get('modelo', '')}\n\n"
                            f"NOTA: este reporte es generado automaticamente con fines "
                            f"academicos y no constituye un diagnostico medico valido.\n"
                        )
                        st.download_button(
                            "Descargar reporte (.txt)",
                            data=reporte,
                            file_name="reporte_hemvlm.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )
                    else:
                        detail = respuesta.json().get("detail", respuesta.text)
                        st.error(f"Error del servidor: {detail}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "No se pudo conectar con la API. Verifica que 'api_llm.py' este "
                        "corriendo (uvicorn api_llm:app --reload --port 8000)."
                    )
    else:
        st.write("Esperando una imagen...")
        st.warning("Sube una imagen en el panel izquierdo para comenzar.")
