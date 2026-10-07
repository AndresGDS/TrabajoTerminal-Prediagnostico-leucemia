r"""
prepare_biobert_tokenizer.py

Corre esto UNA SOLA VEZ. El repositorio dmis-lab/biobert-base-cased-v1.1 en
HuggingFace es viejo (2019-2020) y nunca publico un tokenizer.json moderno,
lo que hace que AutoTokenizer falle al intentar detectar automaticamente como
convertirlo (el error de "necesitas sentencepiece o tiktoken" es generico y
enganoso -- el problema real es la deteccion automatica, no una dependencia
faltante).

Este script construye el tokenizer DIRECTO con la clase correcta
(BertTokenizerFast, que no necesita ninguna conversion especial para
vocabularios WordPiece como el de BERT/BioBERT) y lo guarda localmente, ya
con su tokenizer.json generado. Despues de esto, evaluate_captions_exact.py
y compute_bertscore_only.py apuntan a esta carpeta local en vez de al
repositorio de HuggingFace directamente.

Uso (PowerShell):
    python prepare_biobert_tokenizer.py
"""

from transformers import BertTokenizerFast, BertConfig, BertModel

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
LOCAL_DIR = "./biobert_local"

print(f"Construyendo BertTokenizerFast directo desde {MODEL_NAME}...")
# model_max_length=512 explicito: BioBERT nunca definio este valor en su
# tokenizer_config.json (no existe ese archivo en el repo de 2019), asi que
# por defecto cae en el sentinel "sin limite" de HuggingFace (~1e30), un
# numero tan grande que desborda el tipo de entero del tokenizer rapido
# (Rust) al intentar truncar -- de ahi el OverflowError.
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME, model_max_length=512)
tokenizer.save_pretrained(LOCAL_DIR)

print(f"Descargando el modelo (pesos) de {MODEL_NAME} con clase explicita BertModel...")
# El config.json original de BioBERT (publicado en 2019) no tiene el campo
# 'model_type' que AutoModel/AutoConfig necesitan en versiones recientes de
# transformers. BertConfig/BertModel explicitos no necesitan ese campo.
config = BertConfig.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME, config=config)

print(f"Guardando modelo + tokenizer completos en {LOCAL_DIR}...")
model.save_pretrained(LOCAL_DIR)
config.save_pretrained(LOCAL_DIR)  # re-guarda un config.json COMPLETO, con model_type incluido

print("Listo. Verificando que ambos se puedan recargar con AutoTokenizer/AutoModel (como hara bert_score)...")
from transformers import AutoTokenizer, AutoModel
check_tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
check_model = AutoModel.from_pretrained(LOCAL_DIR)
print(f"OK -- tokenizer: {type(check_tok).__name__}, modelo: {type(check_model).__name__}")
print(f"\nUsa '{LOCAL_DIR}' como model_type en evaluate_captions_exact.py / compute_bertscore_only.py")
