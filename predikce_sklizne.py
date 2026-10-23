#!/usr/bin/env python3
"""
Predikce sklizně zrnin v Česku podle kriminality (ČSÚ: ZEM03A + KRI10).

Použití:
    python predikce_sklizne.py [--zem ZEM03A.csv] [--kri KRI10.csv] [--out vystup]

Vyžaduje: pandas, numpy, scipy, matplotlib
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def nacti_data(zem_path, kri_path):
    z = pd.read_csv(zem_path)
    k = pd.read_csv(kri_path)
    z["plod"] = z["Druh zemědělské plodiny"].str.strip(" -")
    k["kat"] = k["Takticko statistická klasifikace kriminality Police ČR"]

    sklizen = (z[z["Ukazatel"] == "Sklizeň zemědělských plodin (t)"]
               .pivot_table(index="Roky", columns="plod", values="Hodnota"))
    vynos = (z[z["Ukazatel"] == "Hektarový výnos (t/ha)"]
             .pivot_table(index="Roky", columns="plod", values="Hodnota"))

    def krim(ukazatel):
        return (k[(k["Území"] == "Česko") & (k["Ukazatel"] == ukazatel)]
                .pivot(index="Roky", columns="kat", values="Hodnota"))

    na100k = krim("Počet registrovaných skutků na 100 000 obyvatel")
    pocet = krim("Počet registrovaných skutků")
    objasn = krim("Počet objasněných skutků")

    df = pd.DataFrame({
        "sklizen_zrniny_t": sklizen["Zrniny"],
        "vynos_zrniny_t_ha": vynos["Zrniny"],
        "krim_celkem_na_100k": na100k["Celková kriminalita"],
        "krim_majetkova_na_100k": na100k["Majetková kriminalita"],
        "krim_nasilna_na_100k": na100k["Násilná kriminalita"],
        "krim_hospodarska_na_100k": na100k["Hospodářská kriminalita"],
        "krim_celkem_pocet": pocet["Celková kriminalita"],
        "objasnenost_pct": 100 * objasn["Celková kriminalita"] / pocet["Celková kriminalita"],
    })
    return df.dropna(subset=["sklizen_zrniny_t", "krim_celkem_na_100k"])


def loo_predikce(X, y):
    """Leave-one-out predikce lineární regrese (OLS)."""
    pred = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        A = np.c_[np.ones(tr.sum()), X[tr]]
        b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pred[i] = np.r_[1, X[i]] @ b
    return pred


def metriky(pred, y):
    return (np.sqrt(np.mean((pred - y) ** 2)),
            100 * np.mean(np.abs((pred - y) / y)),
            1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zem", default="ZEM03A.csv")
    ap.add_argument("--kri", default="KRI10.csv")
    ap.add_argument("--out", default="vystup")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    m = nacti_data(args.zem, args.kri)
    m.round(2).to_csv(out / "spojeny_dataset.csv", index_label="rok")
    y = m["sklizen_zrniny_t"].values / 1e6      # mil. t
    roky = m.index.values.astype(float)
    xcol = "krim_celkem_na_100k"
    print(f"Spojeno {len(m)} let ({int(roky.min())}–{int(roky.max())})\n")

    # --- korelace (surové i po odstranění trendu) ---
    prediktory = [c for c in m.columns if c.startswith(("krim", "objasn"))]
    d = m.diff().dropna()
    print("Korelace se sklizní zrnin:")
    print(f"{'prediktor':28s} {'r (úrovně)':>12s} {'r (změny)':>12s}")
    for c in prediktory:
        r1, p1 = stats.pearsonr(m[c], y)
        r2, p2 = stats.pearsonr(d[c], d["sklizen_zrniny_t"])
        print(f"{c:28s} {r1:+6.2f} (p={p1:.2f}) {r2:+6.2f} (p={p2:.2f})")

    # --- křížová validace ---
    modely = {
        "Průměr (baseline)": np.array([y[np.arange(len(y)) != i].mean() for i in range(len(y))]),
        "Lineární trend v čase": loo_predikce(roky.reshape(-1, 1), y),
        "Kriminalita celkem/100k": loo_predikce(m[[xcol]].values, y),
        "Násilná+majetková+hosp.": loo_predikce(
            m[["krim_nasilna_na_100k", "krim_majetkova_na_100k", "krim_hospodarska_na_100k"]].values, y),
    }
    print("\nLeave-one-out validace:")
    for nazev, p in modely.items():
        rmse, mape, r2 = metriky(p, y)
        print(f"  {nazev:26s} RMSE={rmse:.3f} Mt  MAPE={mape:.1f}%  R2cv={r2:+.2f}")

    # --- finální model a predikce pro další rok ---
    x = m[xcol].values
    sl = stats.linregress(x, y)
    n = len(y)
    s = np.sqrt(np.sum((y - (sl.intercept + sl.slope * x)) ** 2) / (n - 2))
    t = stats.t.ppf(0.975, n - 2)
    print(f"\nModel: sklizeň = {sl.intercept:.3f} {sl.slope:+.6f} * kriminalita "
          f"(R²={sl.rvalue**2:.2f}, p={sl.pvalue:.2f})")

    dalsi_rok = int(roky.max()) + 1
    posl = x[-1]
    scenare = {f"jako {int(roky.max())} ({posl:.0f})": posl,
               "pokles o 7 %": posl * 0.93,
               "růst o 9 %": posl * 1.09}
    print(f"\nPredikce sklizně zrnin pro {dalsi_rok} (mil. t, 95% interval):")
    for nazev, c in scenare.items():
        pr = sl.intercept + sl.slope * c
        pm = t * s * np.sqrt(1 + 1 / n + (c - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
        print(f"  kriminalita {nazev:22s} -> {pr:.2f} ± {pm:.2f}")
    print(f"  (průměr sklizně {y.mean():.2f}, směrodatná odchylka {y.std(ddof=1):.2f})")

    # --- graf ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    a.plot(roky, y, "o-", color="#2e7d32")
    a.set_ylabel("Sklizeň zrnin (mil. t)", color="#2e7d32")
    a2 = a.twinx()
    a2.plot(roky, x, "s--", color="#c62828")
    a2.set_ylabel("Skutky / 100 tis. obyv.", color="#c62828")
    a.set_title("Sklizeň a kriminalita v Česku"); a.grid(alpha=.3)
    a = ax[1]
    a.scatter(x, y, color="#1565c0")
    for r, xx, yy in zip(roky, x, y):
        a.annotate(str(int(r)), (xx, yy), fontsize=7, xytext=(3, 3), textcoords="offset points")
    xs = np.linspace(x.min() * .95, x.max() * 1.05, 50)
    a.plot(xs, sl.intercept + sl.slope * xs, "r-", label=f"regrese R²={sl.rvalue**2:.2f}, p={sl.pvalue:.2f}")
    a.axhline(y.mean(), color="gray", ls=":", label="průměr")
    a.set_xlabel("Skutky / 100 tis. obyv."); a.set_ylabel("Sklizeň zrnin (mil. t)")
    a.legend(); a.grid(alpha=.3); a.set_title("Vztah kriminalita → sklizeň")
    plt.tight_layout()
    plt.savefig(out / "predikce.png", dpi=130)
    print(f"\nUloženo do složky: {out.resolve()}")


if __name__ == "__main__":
    main()
