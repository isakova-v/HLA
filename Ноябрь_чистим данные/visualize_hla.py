#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(counts_path, by_region_path, outdir):
    sns.set(style="whitegrid")

    # =======================
    # Load data
    # =======================
    df_counts = pd.read_csv(counts_path)
    df_reg = pd.read_csv(by_region_path)

    # -----------------------------------------
    # 1) Barplot: unique alleles per gene (global)
    # -----------------------------------------
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_counts.sort_values("n_unique_alleles", ascending=False),
        x="gene", y="n_unique_alleles", color="skyblue",
    )
    plt.title("Number of unique 2-field alleles per gene (whole sample)")
    plt.ylabel("Number of alleles")
    plt.xlabel("Gene")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    fig1 = f"{outdir}/unique_alleles_global.png"
    plt.savefig(fig1)
    print("[OK] Saved:", fig1)

    # -----------------------------------------
    # 2) Barplot: allele counts per region × gene
    # -----------------------------------------
    reg_gene_counts = df_reg.groupby(["region", "gene"])["allele_field2"].nunique().reset_index()
    reg_gene_counts = reg_gene_counts.rename(columns={"allele_field2": "n_alleles"})

    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=reg_gene_counts,
        x="gene", y="n_alleles", hue="region"
    )
    plt.title("Number of unique 2-field alleles per gene by region")
    plt.ylabel("Number of alleles")
    plt.xlabel("Gene")
    plt.xticks(rotation=45)
    plt.tight_layout()

    fig2 = f"{outdir}/unique_alleles_by_region.png"
    plt.savefig(fig2)
    print("[OK] Saved:", fig2)

    # -----------------------------------------
    # 3) Heatmap (region × gene)
    # -----------------------------------------
    pivot = reg_gene_counts.pivot(index="region", columns="gene", values="n_alleles").fillna(0)

    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".0f")
    plt.title("Number of 2-field alleles: region × gene")
    plt.xlabel("Gene")
    plt.ylabel("Region")
    plt.tight_layout()

    fig3 = f"{outdir}/allele_heatmap_region_gene.png"
    plt.savefig(fig3)
    print("[OK] Saved:", fig3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize HLA allele diversity by region and globally."
    )
    parser.add_argument("--counts", default="hla_allele_counts.csv")
    parser.add_argument("--by-region", default="hla_alleles_by_region.csv")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    main(args.counts, args.by_region, args.outdir)