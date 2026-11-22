"""Eksperimen hiperparameter TA-07 (Langkah 5): pengaruh discount factor gamma.

Menjalankan Q-Learning dua kali (gamma default tinggi vs ekstrem rendah) dengan seed dan
hiperparameter lain yang sama, lalu membandingkan kurva pembelajaran dan performa kebijakan.

Pemakaian:  python experiment_hyperparameter.py            (seed 42)
            python experiment_hyperparameter.py --seed 7
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kurir_rl import convergence_episode, evaluate_policy, moving_average, train_q_learning

HYPER = dict(alpha=0.1, gamma=0.9, episodes=6000, max_steps=100,
             eps_start=1.0, eps_min=0.01, decay_rate=0.001)
GAMMAS = {"gamma = 0.9 (default)": 0.9, "gamma = 0.1 (ekstrem)": 0.1}
COLORS = {"gamma = 0.9 (default)": "#1b7f73", "gamma = 0.1 (ekstrem)": "#c0264b"}


def main(seed, window=100):
    os.makedirs("figures", exist_ok=True)
    results, rows = {}, []
    for name, g in GAMMAS.items():
        q, hist = train_q_learning(**{**HYPER, "gamma": g}, seed=seed)
        ev = evaluate_policy(q, n_episodes=1000)
        conv, plateau = convergence_episode(hist["rewards"])
        results[name] = hist
        rows.append({"Model": name, **ev, "Episode konvergensi": conv if conv is not None else "tidak konvergen",
                     "Reward latih (500 akhir)": float(np.mean(hist["rewards"][-500:]))})

    table = pd.DataFrame(rows).set_index("Model")
    print(table.round(3).to_string())
    table.round(4).to_csv("results_gamma.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for name, hist in results.items():
        axes[0].plot(moving_average(hist["rewards"], window), color=COLORS[name], linewidth=2, label=name)
        axes[1].plot(moving_average(hist["successes"], window), color=COLORS[name], linewidth=2, label=name)
    axes[0].set_title("Kurva pembelajaran: rata-rata reward (100 episode)")
    axes[1].set_title("Tingkat sukses mengantar paket (100 episode)")
    for ax in axes:
        ax.set_xlabel("Episode"); ax.legend(); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Reward"); axes[1].set_ylabel("Tingkat sukses")
    plt.tight_layout()
    plt.savefig("figures/experiment_gamma_script.png", dpi=150, bbox_inches="tight")
    print("\nTersimpan: results_gamma.csv dan figures/experiment_gamma_script.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args().seed)
