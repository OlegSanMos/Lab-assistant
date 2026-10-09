from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
rng = np.random.default_rng(42)


def ohm_law() -> None:
    u = np.arange(0.5, 7.01, 0.5) + rng.normal(0, 0.005, 14)
    i_ma = u / 101.5 * 1000 + rng.normal(0, 0.15, 14)
    df = pd.DataFrame({"U, В": u.round(2), "I, мА": i_ma.round(1)})
    df.to_csv(SAMPLES / "ohm_law.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")


def young_modulus() -> None:
    area = np.pi * 10**2 / 4
    force_kn = np.arange(1, 11, 1.0)
    dl = force_kn * 1000 * 100 / (2.04e5 * area) + rng.normal(0, 0.0004, 10)
    df = pd.DataFrame({"F, кН": force_kn, "Δl, мм": dl.round(4)})
    df.to_excel(SAMPLES / "young_modulus.xlsx", index=False)


def rc_discharge() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0, 4.01, 0.25)
    u = 9.8 * np.exp(-t / 1.06) + rng.normal(0, 0.03, t.size)
    df = pd.DataFrame({"t, с": t, "U, В": u.round(2)})
    df.to_csv(SAMPLES / "rc_discharge.csv", index=False)
    return t, u


def oscillogram() -> None:
    t = np.linspace(-0.5, 4.5, 800)
    u = np.where(t < 0, 9.8, 9.8 * np.exp(-np.clip(t, 0, None) / 1.06))
    u = u + rng.normal(0, 0.04, t.size)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120, facecolor="#101418")
    ax.set_facecolor("#0b1a12")
    ax.plot(t, u, color="#39ff88", linewidth=1.6)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-2, 12)
    ax.set_xticks(np.arange(-0.5, 4.51, 0.5))
    ax.set_yticks(np.arange(-2, 12.1, 2))
    ax.grid(color="#2f5d45", linestyle=":", linewidth=0.8)
    ax.tick_params(colors="#9fc5b0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2f5d45")
    ax.set_xlabel("t, с (0,5 с/дел)", color="#9fc5b0")
    ax.set_ylabel("U, В (2 В/дел)", color="#9fc5b0")
    ax.set_title(
        "CH1  DC  2.00 V/div    M 500 ms/div    Trig: CH1 fall",
        color="#e0f0e8",
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(SAMPLES / "rc_oscillogram.png", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    SAMPLES.mkdir(parents=True, exist_ok=True)
    ohm_law()
    young_modulus()
    rc_discharge()
    oscillogram()
    print("Демонстрационные данные сохранены в", SAMPLES)
