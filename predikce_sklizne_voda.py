#!/usr/bin/env python3
"""
Predikce sklizně zrnin v Česku podle zásob vody v nádržích
(ČSÚ ZEM03A + ISVS Voda – Akumulace vod v nádržích 2015–2025).

Použití:
    python predikce_sklizne_voda.py [--zem ZEM03A.csv] [--voda soubor.xlsx] [--out vystup]

Bez argumentů (např. spuštění v Thonny) skript hledá vstupní soubory ve složce,
kde leží samotný skript: ZEM03A.csv a libovolný ISVSVoda*.xlsx.

Vyžaduje: pandas, numpy, scipy, matplotlib, openpyxl
"""
import argparse
from pathlib import Path
import sys
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

MESICE = [f"OBJ_VZD_{i:02d}" for i in range(1, 13)]   # objem vody v nádrži, mil. m3
# Pozn.: sloupec OBJ_VZD_13 je (v ~95 % případů) leden následujícího roku -> nepoužit.


def _norm(txt):
    """Normalizace textu: sjednotí zápis diakritiky, odstraní BOM/mezery, malá písmena."""
    return unicodedata.normalize("NFC", str(txt)).replace("\ufeff", "").strip().casefold()


def _najdi_sloupec(df, *klicova_slova):
    """Najde sloupec, jehož (normalizovaný) název obsahuje všechna klíčová slova."""
    for c in df.columns:
        n = _norm(c)
        if all(k in n for k in klicova_slova):
            return c
    sys.exit(f"V CSV nenalezen sloupec obsahující {klicova_slova}.\n"
             f"Skutečné sloupce v souboru: {list(df.columns)}")


def _precti_csv(cesta):
    """Zkusí běžná kódování a oddělovače (ČSÚ exporty bývají UTF-8, někdy cp1250 a ';')."""
    posledni = None
    for enc in ("utf-8-sig", "cp1250"):
        try:
            df = pd.read_csv(cesta, sep=None, engine="python", encoding=enc)
            if df.shape[1] > 3:
                return df
        except Exception as e:      # noqa: BLE001
            posledni = e
    sys.exit(f"CSV se nepodařilo přečíst: {posledni}")


def nacti_sklizen(cesta):
    z = _precti_csv(cesta)
    c_ukaz = _najdi_sloupec(z, "ukazatel")
    c_plod = _najdi_sloupec(z, "druh", "plodin")
    c_rok = _najdi_sloupec(z, "roky")
    c_hod = _najdi_sloupec(z, "hodnota")
    z = z.rename(columns={c_ukaz: "ukaz", c_plod: "plod", c_rok: "rok", c_hod: "hod"})
    z["plod"] = z["plod"].astype(str).map(lambda t: _norm(t).strip(" -"))
    z["ukaz_n"] = z["ukaz"].map(_norm)
    z["hod"] = pd.to_numeric(z["hod"].astype(str).str.replace(",", ".").str.replace("\xa0", ""),
                             errors="coerce")
    sk = z[z.ukaz_n.str.contains("sklize") & z.ukaz_n.str.contains(r"\(t\)")]
    vy = z[z.ukaz_n.str.contains("hektarov")]
    if sk.empty or vy.empty:
        sys.exit(f"V CSV chybí ukazatel sklizně nebo výnosu. Nalezené ukazatele: {sorted(z.ukaz.unique())}")
    sk = sk.pivot_table(index="rok", columns="plod", values="hod")
    vy = vy.pivot_table(index="rok", columns="plod", values="hod")
    if "zrniny" not in sk.columns:
        sys.exit(f"V CSV chybí plodina 'Zrniny'. Nalezené plodiny: {list(sk.columns)[:15]} ...")
    return sk["zrniny"].rename("sklizen_t"), vy["zrniny"].rename("vynos_t_ha")


def priznaky_voda(cesta, od=2016):
    """Roční ukazatele z vyrovnaného panelu nádrží (jen nádrže přítomné ve všech letech)."""
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        sys.exit("Chybí balíček 'openpyxl' (potřebný pro čtení .xlsx).\n"
                 "V Thonny: Tools -> Manage packages -> vyhledat 'openpyxl' -> Install.\n"
                 "Nebo v příkazové řádce: pip install openpyxl")
    df = pd.read_excel(cesta)
    df.columns = [str(c).strip().upper() for c in df.columns]
    chybi = [c for c in ["ROK", "MISTOID", "OBJ_OVL_P"] + MESICE if c not in df.columns]
    if chybi:
        sys.exit(f"V xlsx chybí sloupce {chybi}. Nalezené: {list(df.columns)[:20]} ...")
    d = df[df.ROK >= od]                      # rok 2015 má jen 11 nádrží -> vynechán
    n_let = d.ROK.nunique()
    pocet = d.groupby("MISTOID")["ROK"].nunique()
    d = d[d.MISTOID.isin(pocet[pocet == n_let].index)].copy()
    print(f"Panel: {d.MISTOID.nunique()} nádrží × {n_let} let ({od}–{d.ROK.max()})")

    vol = pd.DataFrame(d[MESICE].values, index=d.ROK.values).groupby(level=0).sum()
    vol.columns = range(1, 13)
    # naplněnost = objem / max. ovladatelný objem, průměr přes nádrže (bez dominance velkých)
    ref = d["OBJ_OVL_P"].replace(0, np.nan).values[:, None]
    fill = pd.DataFrame((d[MESICE].values / ref).clip(max=1.2), index=d.ROK.values)
    fill = fill.groupby(level=0).mean(); fill.columns = range(1, 13)

    return pd.DataFrame({
        "vol_rok_prumer": vol.mean(axis=1),
        "vol_vegetace_4_9": vol[[4, 5, 6, 7, 8, 9]].mean(axis=1),
        "vol_jaro_3_5": vol[[3, 4, 5]].mean(axis=1),
        "vol_leden": vol[1],
        "pokles_cerven_zari": vol[6] - vol[9],
        "naplnenost_vegetace": fill[[4, 5, 6, 7, 8, 9]].mean(axis=1),
        "naplnenost_zari": fill[9],
    })


def loo(X, y):
    p = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        b = np.linalg.lstsq(np.c_[np.ones(tr.sum()), X[tr]], y[tr], rcond=None)[0]
        p[i] = np.r_[1, X[i]] @ b
    return p


def metriky(p, y):
    return (np.sqrt(np.mean((p - y) ** 2)), 100 * np.mean(np.abs((p - y) / y)),
            1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))


def najdi_soubor(zadano, vzor):
    """Vrátí cestu k souboru: zadanou, nebo nalezenou podle vzoru ve složce skriptu / aktuální složce."""
    if zadano:
        p = Path(zadano)
        if p.exists():
            return p
        sys.exit(f"Soubor nenalezen: {p.resolve()}")
    for slozka in (Path(__file__).resolve().parent, Path.cwd()):
        nalezene = sorted(slozka.glob(vzor))
        if nalezene:
            return nalezene[0]
    sys.exit(f"Nenalezen soubor '{vzor}' ve složce skriptu ({Path(__file__).resolve().parent}). "
             f"Dejte ho vedle skriptu nebo zadejte cestu přes argument.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zem", default=None, help="CSV se sklizní (ZEM03A.csv)")
    ap.add_argument("--voda", default=None, help="XLSX s akumulací vod v nádržích")
    ap.add_argument("--out", default=None, help="výstupní složka (výchozí: 'vystup' vedle skriptu)")
    a = ap.parse_args()
    zem = najdi_soubor(a.zem, "ZEM03A*.csv")
    voda = najdi_soubor(a.voda, "ISVSVoda*.xlsx")
    out = Path(a.out) if a.out else Path(__file__).resolve().parent / "vystup"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Sklizeň: {zem}\nVoda:    {voda}\n")

    sk, vy = nacti_sklizen(zem)
    f = priznaky_voda(voda)
    m = f.join(sk, how="inner").join(vy, how="inner")
    m.round(3).to_csv(out / "spojeny_dataset.csv", index_label="rok")
    print(f"Spojeno {len(m)} let – to je velmi málo pozorování, výsledky jsou orientační.\n")

    dm = m.diff().dropna()
    n_testu = len(f.columns) * 2
    print(f"Korelace (celkem {n_testu} testů; při takovém počtu vyjde 1 'významný' výsledek i náhodou):")
    for t in ["sklizen_t", "vynos_t_ha"]:
        print(f"\n  cíl: {t}")
        for c in f.columns:
            r, p = stats.pearsonr(m[c], m[t])
            r2, p2 = stats.pearsonr(dm[c], dm[t])
            print(f"    {c:22s} r={r:+.2f} (p={p:.2f})   meziroční změny r={r2:+.2f} (p={p2:.2f})")

    print("\nLeave-one-out validace (nižší RMSE = lepší; R2cv<0 = horší než průměr):")
    for t in ["sklizen_t", "vynos_t_ha"]:
        y = m[t].values
        print(f"\n  cíl: {t}")
        modely = {"průměr (baseline)": np.array([np.delete(y, i).mean() for i in range(len(y))]),
                  "trend v čase": loo(m.index.values.reshape(-1, 1).astype(float), y)}
        for c in f.columns:
            modely[c] = loo(m[[c]].values, y)
        for n, p in modely.items():
            rmse, mape, r2 = metriky(p, y)
            print(f"    {n:22s} RMSE={rmse:10.3g}  MAPE={mape:4.1f}%  R2cv={r2:+.2f}")

    # --- model vybraný předem: naplnenost nádrží v září -> výnos ---
    xc, tc = "naplnenost_zari", "vynos_t_ha"
    x, y = m[xc].values, m[tc].values
    sl = stats.linregress(x, y); n = len(y)
    s = np.sqrt(np.sum((y - (sl.intercept + sl.slope * x)) ** 2) / (n - 2))
    t95 = stats.t.ppf(0.975, n - 2)
    print(f"\nModel: {tc} = {sl.intercept:.2f} + {sl.slope:.2f} * {xc}   (R²={sl.rvalue**2:.2f}, p={sl.pvalue:.3f})")
    print("POZOR: tento prediktor jsem vybral až po vyzkoušení 14 kombinací na 10 letech,")
    print("       p-hodnota je proto optimistická a vztah může být náhodný.")
    print("Predikce výnosu zrnin (t/ha) podle zářijové naplněnosti nádrží:")
    for nap in (0.60, 0.70, 0.75):
        pr = sl.intercept + sl.slope * nap
        pm = t95 * s * np.sqrt(1 + 1 / n + (nap - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
        print(f"  naplněnost {nap:.2f} -> {pr:.2f} ± {pm:.2f} t/ha")

    # --- graf ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    r_ = m.index.values
    ax[0].plot(r_, m[tc], "o-", color="#2e7d32"); ax[0].set_ylabel("Výnos zrnin (t/ha)", color="#2e7d32")
    b = ax[0].twinx(); b.plot(r_, m[xc], "s--", color="#1565c0"); b.set_ylabel("Naplněnost nádrží v září", color="#1565c0")
    ax[0].set_title("Výnos zrnin a naplněnost nádrží"); ax[0].grid(alpha=.3)
    ax[1].scatter(x, y, color="#1565c0")
    for r, xx, yy in zip(r_, x, y):
        ax[1].annotate(str(r), (xx, yy), fontsize=8, xytext=(3, 3), textcoords="offset points")
    xs = np.linspace(x.min() - .01, x.max() + .01, 50)
    ax[1].plot(xs, sl.intercept + sl.slope * xs, "r-", label=f"R²={sl.rvalue**2:.2f}, p={sl.pvalue:.2f}")
    ax[1].set_xlabel("Naplněnost nádrží v září"); ax[1].set_ylabel("Výnos zrnin (t/ha)")
    ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_title("Vztah voda → výnos (n = 10)")
    plt.tight_layout(); plt.savefig(out / "predikce_voda.png", dpi=130)
    print(f"\nUloženo do: {out.resolve()}")


if __name__ == "__main__":
    main()