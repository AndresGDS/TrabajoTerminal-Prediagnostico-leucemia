r"""
mix_third_domain.py

Mezcla una MUESTRA (no el dataset completo) de AML-Cytomorphology_LMU
dentro de tus CSV de entrenamiento actuales, agregando una tercera fuente
de adquisicion a AMBAS clases (sana/leucemica) -- esto es lo que rompe la
correlacion trivial "camara = clase" que causaba el sesgo de dominio
detectado con validate_domain_shortcut.py.

No se usan las 18,365 imagenes completas a proposito: dominarian tu
dataset actual y arrastrarian el estilo de captions hacia el formato
simple ("This is a X.") en vez del descriptivo con atributos que ya tienes.

Uso (PowerShell):
    python mix_third_domain.py `
        --healthy_csv .\data_ft_train\healthy_captions.csv `
        --leukemic_csv .\data\leukemic_captions_faithful_train.csv `
        --aml_cyto_csv .\data\aml_cytomorphology_captions.csv `
        --out_healthy_csv .\data\healthy_captions_augmented.csv `
        --out_leukemic_csv .\data\leukemic_captions_augmented.csv `
        --n_healthy_sample 3000 `
        --n_leukemia_sample 3000
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthy_csv", type=str, required=True)
    parser.add_argument("--leukemic_csv", type=str, required=True)
    parser.add_argument("--aml_cyto_csv", type=str, required=True)
    parser.add_argument("--out_healthy_csv", type=str, required=True)
    parser.add_argument("--out_leukemic_csv", type=str, required=True)
    parser.add_argument("--n_healthy_sample", type=int, default=3000,
                         help="Cuantas imagenes 'sanas' de AML-Cytomorphology mezclar")
    parser.add_argument("--n_leukemia_sample", type=int, default=3000,
                         help="Cuantas imagenes 'leucemicas' (blastos) de AML-Cytomorphology mezclar")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    healthy_df = pd.read_csv(args.healthy_csv)[["image_path", "caption"]]
    leukemic_df = pd.read_csv(args.leukemic_csv)[["image_path", "caption"]]
    aml_cyto_df = pd.read_csv(args.aml_cyto_csv)

    aml_healthy = aml_cyto_df[aml_cyto_df["label"] == "healthy"][["image_path", "caption"]]
    aml_leukemia = aml_cyto_df[aml_cyto_df["label"] == "leukemia"][["image_path", "caption"]]

    n_h = min(args.n_healthy_sample, len(aml_healthy))
    n_l = min(args.n_leukemia_sample, len(aml_leukemia))

    aml_healthy_sample = aml_healthy.sample(n=n_h, random_state=args.seed)
    aml_leukemia_sample = aml_leukemia.sample(n=n_l, random_state=args.seed)

    print(f"Sanas -- originales: {len(healthy_df)}, agregadas de AML-Cytomorphology: {n_h}")
    print(f"Leucemicas -- originales: {len(leukemic_df)}, agregadas de AML-Cytomorphology: {n_l}")

    healthy_out = pd.concat([healthy_df, aml_healthy_sample], ignore_index=True)
    leukemic_out = pd.concat([leukemic_df, aml_leukemia_sample], ignore_index=True)

    healthy_out.to_csv(args.out_healthy_csv, index=False)
    leukemic_out.to_csv(args.out_leukemic_csv, index=False)

    print(f"\nTotal sanas final: {len(healthy_out)} -> {args.out_healthy_csv}")
    print(f"Total leucemicas final: {len(leukemic_out)} -> {args.out_leukemic_csv}")


if __name__ == "__main__":
    main()
