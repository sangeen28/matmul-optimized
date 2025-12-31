#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

def plot_all_results():
    df = pd.read_csv("results/all_results.csv")
    df = df.sort_values(["N","variant","threads"]).reset_index(drop=True)

    variants_order = ["naive","jpi","blocked","packed_ic","packed_jc","openblas"]
    variants_present = [v for v in variants_order if v in df["variant"].unique()]

    # 1) GFLOP/s vs N
    plt.figure(figsize=(10,5))
    for v in variants_present:
        sub = df[df["variant"]==v].groupby("N")["gflops"].max().sort_index()
        plt.plot(sub.index, sub.values, marker="o", label=v)
    plt.xlabel("Matrix size N")
    plt.ylabel("GFLOP/s")
    plt.title("GFLOP/s vs N (ours vs OpenBLAS)")
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/01_gflops_vs_N.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Time vs N (log)
    plt.figure(figsize=(10,5))
    for v in variants_present:
        sub = df[df["variant"]==v].groupby("N")["time_ms"].min().sort_index()
        plt.plot(sub.index, sub.values, marker="o", label=v)
    plt.yscale("log")
    plt.xlabel("Matrix size N")
    plt.ylabel("Time (ms, log scale)")
    plt.title("Runtime vs N (log scale)")
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/02_time_vs_N_log.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Speedup vs naive (where available)
    if "naive" in df["variant"].unique():
        base = df[df["variant"]=="naive"].groupby("N")["time_ms"].min()
        plt.figure(figsize=(10,5))
        for v in ["jpi","blocked","packed_ic","packed_jc","openblas"]:
            if v in df["variant"].unique():
                sub = df[df["variant"]==v].groupby("N")["time_ms"].min()
                common = sorted(set(base.index).intersection(sub.index))
                if not common:
                    continue
                sp = (base.loc[common] / sub.loc[common]).values
                plt.plot(common, sp, marker="o", label=v)
        plt.xlabel("Matrix size N")
        plt.ylabel("Speedup vs naive (x)")
        plt.title("Speedup relative to naive")
        plt.grid(True)
        plt.legend()
        plt.savefig("figures/03_speedup_vs_naive.png", dpi=200, bbox_inches="tight")
        plt.close()

    # 4) Gap to OpenBLAS (GFLOP/s ratio)
    if "openblas" in df["variant"].unique():
        ob = df[df["variant"]=="openblas"].groupby("N")["gflops"].max()
        plt.figure(figsize=(10,5))
        for v in ["packed_ic","packed_jc","blocked","jpi","naive"]:
            if v in df["variant"].unique():
                sub = df[df["variant"]==v].groupby("N")["gflops"].max()
                common = sorted(set(ob.index).intersection(sub.index))
                if not common:
                    continue
                ratio = (sub.loc[common] / ob.loc[common]).values
                plt.plot(common, ratio, marker="o", label=v)
        plt.axhline(1.0, linestyle="--")
        plt.xlabel("Matrix size N")
        plt.ylabel("Ours / OpenBLAS (GFLOP/s)")
        plt.title("Performance gap to OpenBLAS (higher is better)")
        plt.grid(True)
        plt.legend()
        plt.savefig("figures/04_gap_to_openblas.png", dpi=200, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    plot_all_results()
    plot_thread_scaling()
    print("Saved figures to ./figures/")
