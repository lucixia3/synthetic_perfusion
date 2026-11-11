import json
import argparse
from pathlib import Path

import numpy as np

from model import SEGMENTS, COSTATS
from dades import vas_a_prompt, carrega


def fraccio_per_regio(atles, tmax, cob, R):
    f = np.full(R, np.nan)
    for r in range(1, R + 1):
        m = (atles == r) & cob
        if m.sum() > 50:
            f[r - 1] = (tmax[m] > 6).mean()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mni_dir", type=Path, required=True)
    ap.add_argument("--vasos", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("taula_inicial.npy"))
    a = ap.parse_args()

    atles = carrega(a.mni_dir / "atles_mni.nii.gz").round().astype(np.int64)
    R = int(atles.max())
    vasos = json.loads(a.vasos.read_text(encoding="utf-8"))
    casos = [c for c in a.train.read_text().split() if c]
    suma = np.zeros((len(SEGMENTS), len(COSTATS), R))
    n = np.zeros_like(suma)

    for c in casos:
        p = vas_a_prompt(vasos.get(c, {}).get("vas"))
        if p is None:
            continue
        seg, costat = p
        tmax = carrega(a.mni_dir / f"{c}_tmax.nii.gz")
        cob = carrega(a.mni_dir / f"{c}_cob.nii.gz") > 0
        for t, cb, k in ((tmax, cob, costat), (tmax[::-1], cob[::-1], {0: 1, 1: 0, 2: 2}[costat])):
            f = fraccio_per_regio(atles, t, cb, R)
            ok = ~np.isnan(f)
            suma[seg, k, ok] += f[ok]
            n[seg, k, ok] += 1

    global_ = suma.sum((0, 1)) / np.maximum(n.sum((0, 1)), 1)
    prob = np.where(n > 0, suma / np.maximum(n, 1), global_[None, None])
    prob = np.clip(prob, 0.02, 0.98)
    np.save(a.out, np.log(prob / (1 - prob)).astype(np.float32))
    print(f"{len(casos)} casos de train, {R} regions -> {a.out}")
    for s, nom in enumerate(SEGMENTS):
        print(f"  {nom:18s}", "  ".join(f"{COSTATS[k]}: {int(n[s, k].max())}" for k in range(len(COSTATS))))


if __name__ == "__main__":
    main()
