r"""
frontend_app.py

Frontend Streamlit del sistema HemVLM. Consume el backend FastAPI
(api_llm.py) para generar descripciones morfologicas reales sobre imagenes de
celulas y muestra tambien una referencia morfologica educativa.

Paginas:
  - Inicio                  : flujo de analisis (subir imagen, elegir modelo,
                              analizar, ver resultado).
  - Referencia morfologica  : celula sana vs. celula leucemica (educativo).
  - Informacion del proyecto: que hace el sistema, modelos y limitaciones.

Uso (PowerShell), en una SEGUNDA terminal (con api_llm.py ya corriendo en la
primera: uvicorn api_llm:app --reload --port 8000):
    streamlit run frontend_app.py

Notas:
  - El tema claro se fija en .streamlit/config.toml (debe ejecutarse
    `streamlit run` desde la carpeta que contiene .streamlit).
  - Imagenes de referencia opcionales: si existen assets/referencia/sana.jpg y
    assets/referencia/leucemica.jpg (tambien .png/.jpeg/.webp) se muestran en
    lugar de las ilustraciones esquematicas.
  - Variable opcional HEMVLM_API_URL para apuntar a otro backend
    (por defecto http://127.0.0.1:8000).
"""

import base64
import functools
import html as htmllib
import inspect
import io
import math
import os
import random

import requests
import streamlit as st
from PIL import Image

API_BASE = os.environ.get("HEMVLM_API_URL", "http://127.0.0.1:8000")
API_URL = f"{API_BASE}/analyze"
HEALTH_URL = f"{API_BASE}/health"
MODELS_URL = f"{API_BASE}/models"

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "referencia")

PAGES = ["Inicio", "Referencia morfológica", "Información del proyecto"]

st.set_page_config(
    page_title="HemVLM · Prediagnóstico de leucemia",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="auto",
)


# --------------------------------------------------------------------------- #
# Utilidades de compatibilidad entre versiones de Streamlit
# --------------------------------------------------------------------------- #
def _stretch_kwargs(widget):
    """Ancho completo: `width="stretch"` en versiones nuevas,
    `use_container_width=True` en las anteriores."""
    if "width" in inspect.signature(widget).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


_FULL_BUTTON = _stretch_kwargs(st.button)
_FULL_DOWNLOAD = _stretch_kwargs(st.download_button)


def _keyed_container(key, border=False):
    """Contenedor con clase CSS `st-key-<key>` (para poder darle estilo)."""
    try:
        return st.container(border=border, key=key)
    except TypeError:
        return st.container()


def render(markup: str):
    """Pinta HTML. Se quitan sangrias y lineas vacias porque Markdown trata
    4 espacios de sangria como bloque de codigo y una linea vacia corta el
    bloque HTML."""
    cleaned = "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def esc(value) -> str:
    return htmllib.escape(str(value), quote=True)


# --------------------------------------------------------------------------- #
# Iconos e ilustraciones (SVG propios, incrustados como imagen)
# --------------------------------------------------------------------------- #
NAVY, BLUE, ROSE, ROSE_DARK = "#12294A", "#1F63B5", "#C93A71", "#A82B5A"


def _svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


_ICON_PATHS = {
    "upload": '<path d="M12 16V5M7.5 9.5 12 5l4.5 4.5M5 19h14"/>',
    "cell": '<circle cx="12" cy="12" r="8.5"/><circle cx="10.5" cy="11" r="3.2"/><circle cx="16" cy="15.5" r="1.2"/>',
    "drop": '<path d="M12 3.5c3.6 4.3 6 7.2 6 10a6 6 0 0 1-12 0c0-2.8 2.4-5.7 6-10z"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.8v.2"/>',
    "chip": '<rect x="6.5" y="6.5" width="11" height="11" rx="2"/><path d="M9.5 3.5v3M14.5 3.5v3M9.5 17.5v3M14.5 17.5v3M3.5 9.5h3M3.5 14.5h3M17.5 9.5h3M17.5 14.5h3"/>',
    "flask": '<path d="M9.5 3.5h5M10.5 3.5v5.2L5.4 17.3a2 2 0 0 0 1.7 3.2h9.8a2 2 0 0 0 1.7-3.2L13.5 8.7V3.5M8 14h8"/>',
    "doc": '<path d="M7 3.5h7l4 4V20a.5.5 0 0 1-.5.5h-10A.5.5 0 0 1 7 20V3.5zM14 3.5v4h4M9.5 12h5M9.5 15.5h5"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="m8.2 12.4 2.7 2.7 5-5.6"/>',
    "alert": '<path d="M12 4 3.5 19h17L12 4z"/><path d="M12 10v4.2M12 16.6v.2"/>',
}


@functools.lru_cache(maxsize=None)
def icon(name: str, color: str = BLUE, size: int = 24) -> str:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{size}" height="{size}" '
           f'fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" '
           f'stroke-linejoin="round">{_ICON_PATHS[name]}</svg>')
    return _svg_uri(svg)


LOGO_URI = _svg_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">'
    '<path d="M20 3c7 8.2 12 14 12 20a12 12 0 0 1-24 0C8 17 13 11.2 20 3z" fill="#1F63B5"/>'
    '<circle cx="20" cy="24" r="6.8" fill="#fff"/>'
    '<circle cx="18.2" cy="22.6" r="2.7" fill="#C93A71"/></svg>'
)

_RBC_POSITIONS = [(116, 92), (296, 100), (334, 206), (282, 310), (138, 330), (64, 250), (214, 70)]


def _rbc(cx, cy, r=26):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#rbc)" '
            f'stroke="#E4AFC6" stroke-width="1.5"/>')


def _neutrophil(rnd: random.Random) -> str:
    """Leucocito maduro (neutrofilo): nucleo segmentado de cromatina densa,
    citoplasma abundante con granulos finos, sin nucleolos."""
    cx, cy = 200, 196
    parts = [f'<circle cx="{cx}" cy="{cy}" r="43" fill="#F7E6E9" stroke="#E2BDC9" stroke-width="1.5"/>']
    for _ in range(70):
        ang = rnd.random() * 2 * math.pi
        rad = 40 * math.sqrt(rnd.random())
        parts.append(f'<circle cx="{cx + rad * math.cos(ang):.1f}" cy="{cy + rad * math.sin(ang):.1f}" '
                     f'r="1.2" fill="#D79FB5" opacity=".75"/>')
    parts.append('<path d="M190 182 L210 188 M212 190 L198 210" stroke="#4B3190" stroke-width="5" '
                 'stroke-linecap="round" fill="none"/>')
    lobes = [(186, 180, 16, 12, -20), (212, 188, 15, 12, 25), (196, 212, 16, 11, 10)]
    for lx, ly, rx, ry, rot in lobes:
        parts.append(f'<ellipse cx="{lx}" cy="{ly}" rx="{rx}" ry="{ry}" transform="rotate({rot} {lx} {ly})" '
                     f'fill="#5A3E9B" stroke="#43297F" stroke-width="1.5"/>')
        for _ in range(7):
            ang = rnd.random() * 2 * math.pi
            rad = 0.7 * min(rx, ry) * math.sqrt(rnd.random())
            color = "#3E2873" if rnd.random() < 0.6 else "#7E64BA"
            parts.append(f'<circle cx="{lx + rad * math.cos(ang):.1f}" cy="{ly + rad * math.sin(ang):.1f}" '
                         f'r="{1.2 + rnd.random():.1f}" fill="{color}" opacity=".8"/>')
    return "".join(parts)


def _blast(rnd: random.Random) -> str:
    """Blasto: celula grande, nucleo grande de cromatina fina con nucleolos,
    citoplasma escaso muy basofilo con vacuolas (relacion N/C alta)."""
    parts = ['<circle cx="200" cy="192" r="58" fill="#5E84D2" stroke="#4A6DB8" stroke-width="1.5"/>',
             '<circle cx="206" cy="198" r="47" fill="#9C7AD2" stroke="#7A57B8" stroke-width="1.5"/>']
    for _ in range(120):
        ang = rnd.random() * 2 * math.pi
        rad = 43 * math.sqrt(rnd.random())
        color = "#7C58BC" if rnd.random() < 0.5 else "#BBA0E6"
        parts.append(f'<circle cx="{206 + rad * math.cos(ang):.1f}" cy="{198 + rad * math.sin(ang):.1f}" '
                     f'r="{1.2 + 1.6 * rnd.random():.1f}" fill="{color}" opacity=".55"/>')
    parts.append('<circle cx="194" cy="186" r="7" fill="#F4EEFB" stroke="#7A57B8" stroke-width="1.2"/>')
    parts.append('<circle cx="220" cy="208" r="5" fill="#F4EEFB" stroke="#7A57B8" stroke-width="1.2"/>')
    parts.append('<circle cx="166" cy="152" r="4.2" fill="#DCE8FA" stroke="#4A6DB8" stroke-width="1"/>')
    parts.append('<circle cx="152" cy="172" r="3.2" fill="#DCE8FA" stroke="#4A6DB8" stroke-width="1"/>')
    return "".join(parts)


def _marker(n, x, y, tx, ty) -> str:
    return (f'<line x1="{x}" y1="{y}" x2="{tx}" y2="{ty}" stroke="{ROSE_DARK}" stroke-width="2"/>'
            f'<circle cx="{tx}" cy="{ty}" r="3" fill="{ROSE_DARK}"/>'
            f'<circle cx="{x}" cy="{y}" r="12" fill="#fff" stroke="{ROSE_DARK}" stroke-width="2.2"/>'
            f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-family="Figtree,Segoe UI,Arial,sans-serif" '
            f'font-size="15" font-weight="700" fill="{ROSE_DARK}">{n}</text>')


# Marcadores: 1 nucleo, 2 citoplasma, 3 eritrocito (escala), 4 nucleolos (solo blasto)
_MARKERS = {
    "sana": [(1, 110, 176, 173, 180), (2, 300, 246, 232, 208), (3, 270, 150, 288, 118)],
    "blasto": [(1, 104, 222, 166, 214), (2, 102, 148, 170, 157), (3, 270, 146, 288, 118), (4, 296, 232, 221, 208)],
}


@functools.lru_cache(maxsize=None)
def cell_scene_uri(kind: str, markers: bool = True) -> str:
    """Campo de microscopio esquematico: glóbulos rojos de fondo (escala) y
    una celula blanca. kind = 'sana' | 'blasto'."""
    rnd = random.Random(3 if kind == "sana" else 5)
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="40 40 320 320">',
           '<defs><radialGradient id="rbc" cx="50%" cy="50%" r="60%">'
           '<stop offset="0%" stop-color="#FCF1F5"/><stop offset="55%" stop-color="#F4D3E0"/>'
           '<stop offset="100%" stop-color="#E8B0C7"/></radialGradient>'
           '<clipPath id="lens"><circle cx="200" cy="200" r="147"/></clipPath></defs>',
           '<circle cx="200" cy="200" r="150" fill="#F8F2F6"/>',
           '<g clip-path="url(#lens)">']
    out += [_rbc(x, y) for x, y in _RBC_POSITIONS]
    out.append(_neutrophil(rnd) if kind == "sana" else _blast(rnd))
    out.append('</g>')
    out.append(f'<circle cx="200" cy="200" r="150" fill="none" stroke="{BLUE}" stroke-width="6"/>')
    if markers:
        out += [_marker(*m) for m in _MARKERS[kind]]
    out.append('</svg>')
    return _svg_uri("".join(out))


def _reference_photo(stem: str):
    """Foto de referencia opcional puesta por el equipo en assets/referencia/."""
    mimes = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
    for ext, mime in mimes.items():
        path = os.path.join(REFERENCE_DIR, f"{stem}.{ext}")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")
    return None


# --------------------------------------------------------------------------- #
# Estilo: sistema medico claro (blanco + azul, acento rosa)
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
:root{
  --navy:#12294A; --blue:#1F63B5; --blue-dark:#164C8E; --blue-tint:#EAF2FC; --blue-line:#D3E1F4;
  --rose:#C93A71; --rose-dark:#A82B5A; --rose-tint:#FDEDF3; --rose-line:#F5CADB;
  --slate:#51617A; --page:#F5F8FC;
  --shadow:0 1px 2px rgba(18,41,74,.05), 0 8px 24px rgba(31,99,181,.08);
  --body:'Figtree','Segoe UI',system-ui,-apple-system,sans-serif;
  --display:'Fraunces',Georgia,serif;
}
html, body, .stApp, [data-testid="stAppViewContainer"]{ background:var(--page) !important; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp dt, .stApp dd{ font-family:var(--body); }
.stApp, .stApp p, .stApp li, .stApp label{ color:var(--navy); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4{ font-family:var(--display); color:var(--navy); letter-spacing:-.01em; }
[data-testid="stHeader"]{ background:transparent; }
footer{ visibility:hidden; }
.block-container{ max-width:1160px; padding-top:4.2rem; padding-bottom:4rem; }
:focus-visible{ outline:3px solid #8DB4E6; outline-offset:2px; }

/* Marca y navegacion */
.marca{ display:flex; align-items:center; gap:.6rem; font-family:var(--display); font-weight:600;
        font-size:1.6rem; color:var(--navy); padding-top:.25rem; }
.marca img{ width:34px; height:34px; }
.st-key-nav [role="radiogroup"]{ gap:.4rem; flex-wrap:wrap; justify-content:flex-end; }
.st-key-nav [role="radiogroup"] label{ background:#fff; border:1px solid var(--blue-line);
        border-radius:999px; padding:.42rem 1rem; margin:0; cursor:pointer; transition:background .15s, border-color .15s; }
.st-key-nav [role="radiogroup"] label > div:first-child{ display:none; }
.st-key-nav [role="radiogroup"] label > span ~ div > div:first-child{ display:none; }
.st-key-nav [role="radiogroup"] label p{ font-weight:600; font-size:.92rem; color:var(--blue-dark) !important; margin:0; }
.st-key-nav [role="radiogroup"] label:hover{ border-color:var(--blue); }
.st-key-nav [role="radiogroup"] label:has(input:checked){ background:var(--blue); border-color:var(--blue); }
.st-key-nav [role="radiogroup"] label:has(input:checked) p{ color:#fff !important; }

/* Portada */
.hero{ display:flex; align-items:center; justify-content:space-between; gap:2rem; padding:1.4rem 0 .8rem; }
.hero h1{ font-size:2.6rem; line-height:1.1; margin:0 0 .75rem; padding:0; font-weight:600; max-width:19ch; }
.hero p{ font-size:1.08rem; line-height:1.55; color:var(--slate) !important; max-width:56ch; margin:0; }
.hero-art{ flex:0 0 auto; width:min(270px,32vw); }
.hero-art img{ width:100%; height:auto; display:block; }
.chip-prototipo{ display:inline-block; background:var(--rose-tint); color:var(--rose-dark); border:1px solid var(--rose-line);
        border-radius:999px; padding:.22rem .8rem; font-size:.85rem; font-weight:600; margin-bottom:.9rem; }
.aviso{ border-left:4px solid var(--rose); background:#fff; border-radius:0 12px 12px 0; padding:.75rem 1rem;
        color:var(--slate); font-size:.92rem; margin:.6rem 0 1.4rem; }

/* Tarjetas del flujo (contenedores con borde) */
[class*="st-key-card_"]{ background:#fff; border:1px solid var(--blue-line) !important; border-radius:18px;
        box-shadow:var(--shadow); padding:1.15rem 1.25rem; }
.paso{ display:flex; gap:.8rem; align-items:flex-start; margin-bottom:.9rem; }
.paso-n{ flex:0 0 auto; width:30px; height:30px; border-radius:50%; background:var(--blue-tint); color:var(--blue-dark);
        font-weight:700; display:flex; align-items:center; justify-content:center; border:1px solid var(--blue-line); }
.paso h3{ margin:0; padding:0; font-size:1.25rem; }
.paso p{ margin:.15rem 0 0; color:var(--slate) !important; font-size:.93rem; }
.vacio{ border:1.5px dashed var(--blue-line); border-radius:12px; padding:1.2rem 1rem; text-align:center;
        color:var(--slate) !important; margin-top:.7rem; font-size:.95rem; }
.vacio img{ width:30px; height:30px; display:block; margin:0 auto .3rem; }
.muestra img{ width:100%; height:auto; max-height:420px; object-fit:contain; display:block; background:#fff;
        border-radius:12px; border:1px solid var(--blue-line); margin-top:.8rem; }
.chips{ display:flex; gap:.5rem; flex-wrap:wrap; margin:.65rem 0 .2rem; }
.chip{ display:inline-flex; align-items:center; gap:.35rem; background:var(--blue-tint); color:var(--blue-dark);
        border:1px solid var(--blue-line); border-radius:999px; padding:.18rem .7rem; font-size:.84rem; font-weight:600; }
.chip.rosa{ background:var(--rose-tint); color:var(--rose-dark); border-color:var(--rose-line); }
.modelo-desc{ color:var(--slate) !important; font-size:.93rem; margin:.5rem 0 .2rem; }
.modelo-no{ color:var(--slate) !important; font-size:.88rem; margin:.2rem 0; }

/* Resultado */
.resultado{ border-radius:16px; padding:1.2rem 1.3rem; margin-top:1rem; margin-bottom:.9rem; border:1px solid var(--blue-line);
        border-left-width:7px; background:var(--blue-tint); }
.resultado.alerta{ border-color:var(--rose-line); border-left-color:var(--rose); background:var(--rose-tint); }
.resultado.calma{ border-left-color:var(--blue); }
.resultado-cab{ display:flex; gap:.8rem; align-items:center; }
.resultado-cab img{ width:42px; height:42px; flex:0 0 auto; }
.resultado h3{ margin:0; padding:0; font-size:1.7rem; line-height:1.15; }
.resultado.alerta h3{ color:var(--rose-dark); }
.resultado.calma h3{ color:var(--blue-dark); }
.r-etiqueta{ margin:0; font-size:.85rem; color:var(--slate) !important; font-weight:600; }
.r-detalle{ margin:.7rem 0 .9rem; font-size:.98rem; }
.r-texto{ margin:.2rem 0 .9rem; font-size:1.02rem; line-height:1.55; background:#fff; border:1px solid var(--blue-line);
        border-radius:10px; padding:.75rem .9rem; }
.r-aviso{ margin:.8rem 0 0; font-size:.84rem; color:var(--slate) !important; }

/* Referencia morfologica */
.seccion h2{ font-size:2rem; margin:.4rem 0 .4rem; padding:0; }
.seccion p{ color:var(--slate) !important; max-width:64ch; line-height:1.6; margin:0 0 1.2rem; }
.ref-card{ background:#fff; border:1px solid var(--blue-line); border-radius:18px; box-shadow:var(--shadow);
        padding:1.2rem 1.3rem 1.4rem; border-top:5px solid var(--blue); }
.ref-card.leucemica{ border-top-color:var(--rose); }
.lente{ width:100%; max-width:340px; display:block; margin:.2rem auto .6rem; height:auto; }
.lente.foto{ aspect-ratio:1; object-fit:cover; border-radius:50%; border:7px solid var(--blue); }
.ref-card.leucemica .lente.foto{ border-color:var(--rose); }
.ref-card h3{ margin:.2rem 0 .1rem; padding:0; font-size:1.4rem; }
.ref-sub{ color:var(--slate) !important; margin:0 0 .3rem; font-size:.93rem; }
.ref-nota{ color:var(--slate) !important; margin:0 0 .8rem; font-size:.82rem; font-style:italic; }
.leyenda{ display:flex; flex-wrap:wrap; gap:.35rem .9rem; margin:.2rem 0 1rem; padding:0; list-style:none;
        font-size:.85rem; color:var(--slate); }
.leyenda li{ display:flex; align-items:center; gap:.35rem; margin:0; }
.num{ display:inline-flex; align-items:center; justify-content:center; width:20px; height:20px; border-radius:50%;
        border:2px solid var(--rose-dark); color:var(--rose-dark); background:#fff; font-size:.74rem; font-weight:700; }
.rasgos{ margin:0; display:grid; gap:.65rem; }
.rasgos div{ border-top:1px solid var(--blue-line); padding-top:.55rem; }
.rasgos dt{ font-weight:700; font-size:.95rem; }
.rasgos dd{ margin:.1rem 0 0; color:var(--slate); font-size:.93rem; line-height:1.5; }
.nota{ background:#fff; border:1px solid var(--blue-line); border-radius:14px; padding:1rem 1.2rem; margin-top:1.2rem;
        color:var(--slate); font-size:.95rem; line-height:1.55; }
.nota strong{ color:var(--navy); }
.disclaimer{ background:var(--rose-tint); border:1px solid var(--rose-line); border-radius:14px; padding:1rem 1.2rem;
        margin-top:1rem; font-size:.95rem; line-height:1.55; }

/* Informacion del proyecto */
.info-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:1rem; margin:.4rem 0 1.4rem; }
.info-item{ background:#fff; border:1px solid var(--blue-line); border-radius:16px; padding:1.1rem 1.2rem; }
.info-item img{ width:30px; height:30px; margin-bottom:.4rem; }
.info-item h3{ margin:.1rem 0 .3rem; padding:0; font-size:1.15rem; }
.info-item p{ margin:0; color:var(--slate) !important; font-size:.95rem; line-height:1.55; }
.info-item ul{ margin:.2rem 0 0; padding-left:1.1rem; color:var(--slate); font-size:.95rem; line-height:1.55; }

/* Controles de Streamlit */
.stApp button[kind="primary"], .stApp [data-testid="stBaseButton-primary"]{
        background:var(--blue); border:1px solid var(--blue); color:#fff; border-radius:12px; font-weight:600;
        padding:.65rem 1.2rem; }
.stApp button[kind="primary"] p, .stApp [data-testid="stBaseButton-primary"] p{ color:#fff !important; }
.stApp button[kind="primary"]:hover, .stApp [data-testid="stBaseButton-primary"]:hover{ background:var(--blue-dark); border-color:var(--blue-dark); }
.stApp button[kind="primary"]:disabled, .stApp [data-testid="stBaseButton-primary"]:disabled{ background:#9DB9DE; border-color:#9DB9DE; }
.stApp button[kind="secondary"], .stApp [data-testid="stBaseButton-secondary"]{
        background:#fff; border:1px solid var(--blue-line); color:var(--blue-dark); border-radius:12px; font-weight:600; }
.stApp button[kind="secondary"]:hover, .stApp [data-testid="stBaseButton-secondary"]:hover{ border-color:var(--blue); color:var(--blue-dark); }
[data-testid="stFileUploaderDropzone"]{ background:var(--blue-tint); border:1.5px dashed #9DBBE3; border-radius:12px; }
.st-key-modelpick [role="radiogroup"]{ gap:.6rem; flex-wrap:wrap; }
.st-key-modelpick [role="radiogroup"] label{ border:1.5px solid var(--blue-line); border-radius:12px; background:#fff;
        padding:.55rem 1rem; margin:0; min-width:140px; cursor:pointer; }
.st-key-modelpick [role="radiogroup"] label:has(input:checked){ border-color:var(--blue); background:var(--blue-tint); }
section[data-testid="stSidebar"]{ background:#fff; border-right:1px solid var(--blue-line); }

@media (max-width:760px){
  .hero{ flex-direction:column-reverse; align-items:flex-start; gap:1rem; }
  .hero h1{ font-size:2rem; }
  .hero-art{ display:none; }
  .st-key-nav [role="radiogroup"]{ justify-content:flex-start; }
  .block-container{ padding-left:1rem; padding-right:1rem; }
}
@media (prefers-reduced-motion:reduce){ *{ transition:none !important; animation:none !important; } }
</style>
"""


# --------------------------------------------------------------------------- #
# Estado y backend
# --------------------------------------------------------------------------- #
def init_state():
    st.session_state.setdefault("page", PAGES[0])
    st.session_state.setdefault("sample", None)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("nonce", 0)
    st.session_state.setdefault("model_key", None)


@st.cache_data(ttl=4, show_spinner=False)
def fetch_backend_state():
    """Consulta /health y /models. Si el backend es una version anterior sin
    /models, 'models' queda en None y la interfaz usa el modelo unico."""
    state = {"online": False, "health": {}, "models": None}
    try:
        state["health"] = requests.get(HEALTH_URL, timeout=3).json()
        state["online"] = True
    except (requests.exceptions.RequestException, ValueError):
        return state
    try:
        resp = requests.get(MODELS_URL, timeout=3)
        if resp.status_code == 200:
            state["models"] = resp.json().get("modelos")
    except (requests.exceptions.RequestException, ValueError):
        pass
    return state


def run_analysis(sample: dict, model_key):
    """Devuelve (datos, error). Envia la imagen ORIGINAL al backend."""
    files = {"file": (sample["name"], sample["bytes"], sample["mime"])}
    data = {"modelo": model_key} if model_key else None
    try:
        resp = requests.post(API_URL, files=files, data=data, timeout=180)
    except requests.exceptions.ConnectionError:
        return None, ("No se pudo conectar con la API. Verifica que 'api_llm.py' esté corriendo "
                      "(uvicorn api_llm:app --reload --port 8000).")
    except requests.exceptions.Timeout:
        return None, ("El análisis tardó demasiado. Si es la primera vez que usas este modelo, "
                      "puede estar cargándose: espera unos segundos y vuelve a intentarlo.")
    if resp.status_code == 200:
        return resp.json(), None
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    return None, f"Error del servidor: {detail}"


def build_sample(name: str, raw: bytes, sig):
    """Guarda la muestra y una vista previa liviana (la original no se toca)."""
    try:
        img = Image.open(io.BytesIO(raw))
        width, height = img.size
        preview = img.convert("RGB")
    except Exception:
        return None
    preview.thumbnail((900, 900))
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=90)
    ext = os.path.splitext(name)[1].lower()
    return {
        "name": name, "bytes": raw, "sig": sig, "width": width, "height": height,
        "mime": "image/png" if ext == ".png" else "image/jpeg",
        "preview_uri": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
    }


# --------------------------------------------------------------------------- #
# Piezas de interfaz
# --------------------------------------------------------------------------- #
def render_header():
    col_brand, col_nav = st.columns([1, 2.4], gap="medium", vertical_alignment="center")
    with col_brand:
        render(f'<div class="marca"><img src="{LOGO_URI}" alt="">HemVLM</div>')
    with col_nav:
        with _keyed_container("nav"):
            st.radio("Navegación", PAGES, horizontal=True, key="page", label_visibility="collapsed")


def render_sidebar(state):
    with st.sidebar:
        st.markdown("### Panel clínico")
        st.info(
            "**Protocolo de captura:** imagen de frotis de sangre periférica, "
            "tinción Wright-Giemsa, aumento 100x recomendado."
        )
        st.divider()
        if state["online"]:
            health = state["health"]
            st.success(f"Backend activo · dispositivo: {health.get('device', '?')}")
            st.caption(f"Modelo: {health.get('modelo', 'desconocido')}")
        else:
            st.error("No se pudo conectar con la API (api_llm.py). ¿Está corriendo?")
        st.divider()
        st.caption(
            "Trabajo Terminal 2027-A140, ESCOM-IPN. Sistema de apoyo, "
            "no reemplaza el diagnóstico de un hematólogo."
        )


def step_header(number: int, title: str, subtitle: str):
    render(f"""
        <div class="paso"><span class="paso-n">{number}</span>
        <div><h3>{esc(title)}</h3><p>{esc(subtitle)}</p></div></div>
    """)


def render_hero():
    art = cell_scene_uri("blasto", False)
    render(f"""
        <div class="hero">
        <div class="hero-texto">
        <span class="chip-prototipo">Prototipo académico del Trabajo Terminal 2027-A140 (ESCOM, IPN)</span>
        <h1>Análisis morfológico de frotis de sangre periférica</h1>
        <p>Sube la imagen de una célula y el sistema describe lo que observa (núcleo, cromatina, citoplasma)
        e indica si encuentra características de un blasto, la célula inmadura asociada con la leucemia aguda.</p>
        </div>
        <div class="hero-art"><img src="{art}" alt="Campo de microscopio con glóbulos rojos y una célula blástica"></div>
        </div>
        <div class="aviso">Herramienta de apoyo académico. Todo hallazgo generado por este sistema debe ser
        validado por un hematólogo antes de cualquier decisión clínica.</div>
    """)


def model_selector(state):
    """Devuelve la clave del modelo a usar (None si el backend no soporta
    elegir). Solo ofrece modelos que el backend reporta como disponibles."""
    if not state["online"]:
        st.warning("No hay conexión con el backend. Inícialo con: uvicorn api_llm:app --reload --port 8000")
        return None

    models = state["models"]
    if not models:   # backend anterior sin /models: un solo modelo
        label = state["health"].get("modelo", "modelo del backend")
        render(f'<p class="modelo-desc"><strong>Modelo en uso:</strong> {esc(label)}</p>')
        return None

    available = [m for m in models if m.get("disponible")]
    if not available:
        st.error("El backend no tiene ningún modelo disponible.")
        return None

    keys = [m["clave"] for m in available]
    if st.session_state.model_key not in keys:
        default = next((m["clave"] for m in available if m.get("por_defecto")), keys[0])
        st.session_state.model_key = default
    by_key = {m["clave"]: m for m in available}

    if len(keys) > 1:
        def _sync():
            st.session_state.model_key = st.session_state.model_widget
        with _keyed_container("modelpick"):
            st.radio(
                "Modelo de análisis", keys, index=keys.index(st.session_state.model_key),
                format_func=lambda k: by_key[k]["etiqueta"], horizontal=True,
                key="model_widget", on_change=_sync, label_visibility="collapsed",
            )
    selected = by_key[st.session_state.model_key]
    note = "" if selected.get("cargado") else " La primera vez se carga en memoria y puede tardar unos segundos."
    render(f'<p class="modelo-desc"><strong>{esc(selected["etiqueta"])}.</strong> {esc(selected["descripcion"])}{esc(note)}</p>')
    for m in models:
        if not m.get("disponible"):
            render(f'<p class="modelo-no">{esc(m["etiqueta"])} no está disponible: {esc(m.get("motivo") or "motivo desconocido")}</p>')
    return st.session_state.model_key


def render_result(d: dict):
    blastic = d.get("diagnostico_detectado") == "leucemica"
    if blastic:
        tone, icon_name, color = "alerta", "alert", ROSE_DARK
        title = "Características blásticas detectadas"
        detail = ("El prototipo describió un tipo celular inmaduro (blasto), asociado con la leucemia aguda. "
                  "Requiere revisión por un especialista.")
    else:
        tone, icon_name, color = "calma", "check", "#164C8E"
        title = "Sin características blásticas"
        detail = "El prototipo no describió un tipo celular blástico en esta imagen."

    confianza = str(d.get("confianza", "no determinado")).strip().strip('"').strip("'")
    modelo = d.get("modelo_etiqueta") or d.get("modelo", "")
    chips = [f'<span class="chip">Confianza del texto: {esc(confianza)}</span>',
             f'<span class="chip">Modelo: {esc(modelo)}</span>']
    if d.get("dispositivo_modelo"):
        chips.append(f'<span class="chip">Procesado en: {esc(str(d["dispositivo_modelo"]).upper())}</span>')

    render(f"""
        <div class="resultado {tone}">
        <div class="resultado-cab"><img src="{icon(icon_name, color, 42)}" alt="">
        <div><p class="r-etiqueta">Predicción del prototipo</p><h3>{esc(title)}</h3></div></div>
        <p class="r-detalle">{esc(detail)}</p>
        <p class="r-etiqueta">Descripción morfológica</p>
        <p class="r-texto">{esc(d.get("descripcion", ""))}</p>
        <div class="chips">{"".join(chips)}</div>
        <p class="r-aviso">Resultado orientativo de un prototipo académico. No es un diagnóstico médico.</p>
        </div>
    """)

    reporte = (
        f"HemVLM -- Reporte de análisis morfológico\n"
        f"{'=' * 50}\n\n"
        f"Diagnóstico detectado: {d.get('diagnostico_detectado', '')}\n\n"
        f"Descripción generada:\n{d.get('descripcion', '')}\n\n"
        f"Confianza: {d.get('confianza', 'no determinado')}\n\n"
        f"Modelo: {d.get('modelo', '')}\n\n"
        f"NOTA: este reporte es generado automáticamente con fines "
        f"académicos y no constituye un diagnóstico médico válido.\n"
    )
    st.download_button(
        "Descargar reporte (.txt)", data=reporte, file_name="reporte_hemvlm.txt",
        mime="text/plain", **_FULL_DOWNLOAD,
    )


# --------------------------------------------------------------------------- #
# Paginas
# --------------------------------------------------------------------------- #
def page_home(state):
    render_hero()
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        with _keyed_container("card_muestra", border=True):
            step_header(1, "Muestra", "Sube una imagen de frotis de sangre periférica (JPG o PNG).")
            uploaded = st.file_uploader(
                "Imagen de la muestra", type=["jpg", "jpeg", "png"],
                key=f"uploader_{st.session_state.nonce}", label_visibility="collapsed",
            )
            if uploaded is not None:
                sig = (uploaded.name, uploaded.size)
                current = st.session_state.sample
                if current is None or current["sig"] != sig:
                    sample = build_sample(uploaded.name, uploaded.getvalue(), sig)
                    if sample is None:
                        st.error("No se pudo leer la imagen. Prueba con otro archivo JPG o PNG.")
                    else:
                        st.session_state.sample = sample
                        st.session_state.result = None

            sample = st.session_state.sample
            if sample:
                render(f"""
                    <div class="muestra"><img src="{sample['preview_uri']}" alt="Muestra cargada: {esc(sample['name'])}"></div>
                    <div class="chips"><span class="chip">{esc(sample['name'])}</span>
                    <span class="chip">{sample['width']} × {sample['height']} px</span></div>
                """)
                if st.button("Quitar imagen", key="remove_sample"):
                    st.session_state.sample = None
                    st.session_state.result = None
                    st.session_state.nonce += 1
                    st.rerun()
            else:
                render(f"""
                    <div class="vacio"><img src="{icon('upload', BLUE, 30)}" alt="">
                    Aún no hay imagen. Arrastra un archivo o usa el botón de arriba.</div>
                """)

    with col_right:
        with _keyed_container("card_analisis", border=True):
            step_header(2, "Modelo de análisis", "Elige qué modelo describirá la célula.")
            model_key = model_selector(state)
            st.write("")
            clicked = st.button(
                "Analizar muestra", type="primary", disabled=sample is None or not state["online"],
                key="run_analysis", **_FULL_BUTTON,
            )
            if clicked and sample is not None:
                with st.spinner("Analizando morfología celular..."):
                    datos, error = run_analysis(sample, model_key)
                if error:
                    st.session_state.result = None
                    st.error(error)
                else:
                    st.session_state.result = datos

            if st.session_state.result:
                render_result(st.session_state.result)
            elif sample is None:
                st.caption("Sube una imagen para habilitar el análisis.")


def _ref_card(kind: str, photo_stem: str, title: str, subtitle: str, legend, features, alt: str):
    photo = _reference_photo(photo_stem)
    if photo:
        image = f'<img class="lente foto" src="{photo}" alt="{esc(alt)}">'
        caption = '<p class="ref-nota">Imagen de referencia.</p>'
    else:
        image = f'<img class="lente" src="{cell_scene_uri(kind)}" alt="{esc(alt)}">'
        caption = '<p class="ref-nota">Ilustración esquemática, no es una fotografía.</p>'
    legend_html = "".join(f'<li><span class="num">{n}</span>{esc(text)}</li>' for n, text in legend)
    features_html = "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>" for k, v in features)
    css_kind = "leucemica" if kind == "blasto" else "sana"
    legend_block = "" if photo else f'<ul class="leyenda">{legend_html}</ul>'
    return f"""
        <div class="ref-card {css_kind}">
        {image}
        <h3>{esc(title)}</h3>
        <p class="ref-sub">{esc(subtitle)}</p>
        {caption}
        {legend_block}
        <dl class="rasgos">{features_html}</dl>
        </div>
    """


def page_reference():
    render("""
        <div class="seccion"><h2>Referencia morfológica</h2>
        <p>Una guía visual para entender qué observa el prototipo. Compara una célula sanguínea madura con un
        blasto, el tipo de célula inmadura que aparece en la leucemia aguda.</p></div>
    """)
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        render(_ref_card(
            "sana", "sana", "Célula sana", "Ejemplo: neutrófilo maduro.",
            [(1, "Núcleo"), (2, "Citoplasma"), (3, "Glóbulo rojo (escala)")],
            [
                ("Tamaño y forma", "Contorno redondeado y regular. Su tamaño depende del tipo: desde el de un glóbulo rojo hasta unas 2 veces más."),
                ("Núcleo", "Cromatina madura y agrupada. Según el tipo, es segmentado en lóbulos (neutrófilo), redondo y compacto (linfocito) o con muesca (monocito)."),
                ("Relación núcleo/citoplasma", "Equilibrada: se ve citoplasma suficiente alrededor del núcleo. El linfocito pequeño es la excepción: tiene poco citoplasma, pero su núcleo es muy denso."),
                ("Nucléolos", "No se distinguen en las células maduras."),
                ("Citoplasma", "Con gránulos finos (neutrófilo, eosinófilo, basófilo) o azul pálido y homogéneo (linfocito, monocito)."),
            ],
            "Ilustración de una célula sana: núcleo segmentado oscuro y citoplasma abundante con gránulos finos, rodeada de glóbulos rojos.",
        ))
    with col_b:
        render(_ref_card(
            "blasto", "leucemica", "Célula leucémica", "Ejemplo: blasto.",
            [(1, "Núcleo"), (2, "Citoplasma"), (3, "Glóbulo rojo (escala)"), (4, "Nucléolos")],
            [
                ("Tamaño y forma", "Tamaño variable, con frecuencia mayor que el de las células maduras. El contorno puede ser irregular."),
                ("Núcleo", "Grande, redondo u ovalado, a veces con hendiduras. Cromatina fina y abierta, menos densa que la de una célula madura."),
                ("Relación núcleo/citoplasma", "Alta: el núcleo ocupa la mayor parte de la célula."),
                ("Nucléolos", "Uno o varios, visibles y a veces prominentes."),
                ("Citoplasma", "Escaso, de un azul intenso (basófilo). Puede tener vacuolas."),
                ("Inmadurez", "Los blastos son células inmaduras que normalmente no circulan en la sangre periférica."),
            ],
            "Ilustración de un blasto: núcleo grande de cromatina fina con nucléolos y un borde delgado de citoplasma azul con vacuolas, rodeado de glóbulos rojos.",
        ))

    render("""
        <div class="nota"><strong>Los blastos no se ven todos iguales.</strong> Los mieloblastos suelen ser grandes y
        pueden mostrar bastones de Auer; los linfoblastos suelen ser más pequeños, con menos citoplasma y nucléolos
        poco visibles. Por eso el prototipo describe cada atributo por separado en lugar de comparar con un solo patrón.</div>
        <div class="disclaimer">Las características mostradas tienen fines educativos y sirven como referencia visual
        para comprender el análisis realizado por el prototipo. El sistema no sustituye la evaluación de un
        profesional de la salud.</div>
    """)


def page_project(state):
    render("""
        <div class="seccion"><h2>Información del proyecto</h2>
        <p>HemVLM es un prototipo académico que apoya la lectura de frotis de sangre periférica.</p></div>
    """)
    render(f"""
        <div class="info-grid">
        <div class="info-item"><img src="{icon('cell', BLUE)}" alt="">
        <h3>¿Qué hace este proyecto?</h3>
        <p>Usa inteligencia artificial para analizar la imagen de una célula de sangre, describir su morfología
        (forma del núcleo, cromatina, nucléolos, citoplasma) y señalar si la descripción corresponde a un blasto.</p></div>
        <div class="info-item"><img src="{icon('flask', BLUE)}" alt="">
        <h3>Para qué sirve</h3>
        <p>Es una herramienta de apoyo para estudiar y estandarizar el análisis morfológico, con el fin de reducir la
        variabilidad entre observadores. No emite diagnósticos ni sustituye la evaluación de un hematólogo.</p></div>
        <div class="info-item"><img src="{icon('drop', ROSE)}" alt="">
        <h3>Con qué se entrenó</h3>
        <p>Con imágenes públicas de células sanguíneas sanas (WBCAtt) y de leucemia aguda (LeukemiaAttri), cada una
        con su descripción morfológica.</p></div>
        </div>
    """)

    render("""
        <div class="seccion"><h2 style="font-size:1.5rem">Cómo funciona</h2></div>
        <div class="paso"><span class="paso-n">1</span><div><h3>Subes la imagen</h3>
        <p>Una célula de frotis de sangre periférica en formato JPG o PNG.</p></div></div>
        <div class="paso"><span class="paso-n">2</span><div><h3>El modelo la describe</h3>
        <p>Un modelo de visión y lenguaje genera la descripción y un modelo de lenguaje pequeño la traduce al español.</p></div></div>
        <div class="paso"><span class="paso-n">3</span><div><h3>Se revisa el tipo de célula</h3>
        <p>El sistema indica si la descripción menciona un blasto. La decisión final siempre es del especialista.</p></div></div>
    """)

    # Modelos reales segun el backend (no se simula ninguno)
    models = state["models"] if state["online"] else None
    if models:
        items = []
        for m in models:
            status = "disponible" if m.get("disponible") else "no disponible"
            items.append(f"<li><strong>{esc(m['etiqueta'])}</strong> ({status}): {esc(m['descripcion'])}</li>")
        render(f"""
            <div class="info-grid"><div class="info-item"><img src="{icon('chip', BLUE)}" alt="">
            <h3>Modelos de análisis</h3><ul>{"".join(items)}</ul></div>
            <div class="info-item"><img src="{icon('info', ROSE)}" alt="">
            <h3>Limitaciones conocidas</h3>
            <ul><li>Puede confundir linfocitos normales con linfoblastos.</li>
            <li>Fue entrenado con imágenes de fuentes públicas: puede comportarse distinto con otras tinciones,
            microscopios o resoluciones.</li>
            <li>Todo resultado debe ser validado por un hematólogo.</li></ul></div></div>
        """)
    else:
        render(f"""
            <div class="info-grid"><div class="info-item"><img src="{icon('info', ROSE)}" alt="">
            <h3>Limitaciones conocidas</h3>
            <ul><li>Puede confundir linfocitos normales con linfoblastos.</li>
            <li>Fue entrenado con imágenes de fuentes públicas: puede comportarse distinto con otras tinciones,
            microscopios o resoluciones.</li>
            <li>Todo resultado debe ser validado por un hematólogo.</li></ul></div></div>
        """)

    render("""
        <div class="disclaimer">Trabajo Terminal 2027-A140, ESCOM-IPN. Las imágenes de la referencia morfológica son
        educativas y el sistema no sustituye la evaluación de un profesional de la salud.</div>
    """)


# --------------------------------------------------------------------------- #
# Principal
# --------------------------------------------------------------------------- #
def main():
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    state = fetch_backend_state()
    render_sidebar(state)
    render_header()

    page = st.session_state.page
    if page == "Referencia morfológica":
        page_reference()
    elif page == "Información del proyecto":
        page_project(state)
    else:
        page_home(state)


main()
