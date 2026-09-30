r"""
check_train_test_leakage.py

Verifica si alguna imagen (por hash MD5) aparece tanto en el split de
entrenamiento oficial como en el de prueba oficial -- fuga de datos real,
distinta de duplicados genericos dentro del mismo pool.

Uso (PowerShell):
    python check_train_test_leakage.py `
        --train_csv .\data\leukemic_captions_faithful_train.csv `
        --test_csv .\data\leukemic_captions_faithful_test.csv `
        --details_csv .\eda_full_report_faithful\leucemicas_image_details.csv
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--details_csv", type=str, required=True,
                         help="El *_image_details.csv (tiene el hash md5) generado por eda_full_dataset.py "
                              "sobre el archivo COMBINADO (train+test concatenados)")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    details_df = pd.read_csv(args.details_csv)

    path_to_md5 = dict(zip(details_df["path"], details_df["md5"]))

    train_hashes = set(train_df["image_path"].map(path_to_md5).dropna())
    test_hashes = set(test_df["image_path"].map(path_to_md5).dropna())

    overlap = train_hashes & test_hashes

    print(f"Imagenes en train: {len(train_df)} ({len(train_hashes)} hashes unicos)")
    print(f"Imagenes en test: {len(test_df)} ({len(test_hashes)} hashes unicos)")
    print(f"\nHashes que aparecen en AMBOS (fuga real de datos): {len(overlap)}")

    if overlap:
        print("\nADVERTENCIA: hay fuga de datos entre train y test.")
        affected_train = train_df[train_df["image_path"].map(path_to_md5).isin(overlap)]
        affected_test = test_df[test_df["image_path"].map(path_to_md5).isin(overlap)]
        print(f"Filas afectadas en train: {len(affected_train)}")
        print(f"Filas afectadas en test: {len(affected_test)}")
        print("\nEjemplos de rutas afectadas (train):")
        print(affected_train["image_path"].head(5).tolist())
    else:
        print("\nOK -- no hay fuga de datos entre train y test. Los splits son limpios.")


if __name__ == "__main__":
    main()
