"""Rutas configurables. Importar no lee datasets ni crea directorios.

Configurar HEMVLM_EDA_* antes de iniciar Jupyter. Rutas relativas al repositorio.
Los cinco fragmentos de LeukemiaAttri permanecen separados.
"""
import os
from pathlib import Path


def eda_paths():
    repo = Path(__file__).resolve().parents[2]

    def configured(name, default):
        value = Path(os.environ.get(name, str(default))).expanduser()
        return (value if value.is_absolute() else repo / value).resolve()

    dataset = configured("HEMVLM_EDA_DATASET_DIR", repo / "data_local" / "Dataset")
    output = configured("HEMVLM_EDA_OUTPUT_DIR", repo / "artifacts" / "eda")
    paths = {
        "repo": repo,
        "dataset": dataset,
        "wbcatt": configured("HEMVLM_EDA_WBC_DIR", dataset / "WBCAtt"),
        "pbc": configured("HEMVLM_EDA_PBC_DIR", dataset.parent / "bloodcells_dataset"),
        "leukemiaattri": configured("HEMVLM_EDA_LEUKEMIA_DIR", dataset / "LeukemiaAttri"),
        "captions": configured("HEMVLM_EDA_CAPTIONS_CSV", dataset / "captions" / "captions.csv"),
        "output": output,
    }
    for source in (dataset, paths["wbcatt"], paths["pbc"], paths["leukemiaattri"], paths["captions"].parent, repo / "notebooks", repo / "docs"):
        if output == source or source in output.parents:
            raise ValueError("La salida EDA debe estar fuera de fuentes, notebooks y docs")
    paths["revision_db"] = configured("HEMVLM_EDA_REVISION_DB", output / "resultados_leukemiaattri" / "tablas_eda.sqlite")
    if output not in paths["revision_db"].parents:
        raise ValueError("La SQLite de revisión debe ser una copia dentro de la salida EDA")
    return paths
