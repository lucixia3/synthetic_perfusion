import csv
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import Generador, DiscriminadorPatch, SEGMENTS, COSTATS
from dades import TallsMNI, etiquetes_atles, ESCALA_TMAX

CAMPS = ["epoca", "lr", "l1", "adv", "atles", "suau", "lD", "val_mae_s", "val_atles", "temps_s"]


def perdua_l1(pred, y, cob):
    return (torch.abs(pred - y) * cob).sum() / (cob.sum() + 1e-6)


def perdua_atles(prior, y, cob):
    diana = (y * ESCALA_TMAX > 6).float()
    bce = F.binary_cross_entropy(prior.clamp(1e-4, 1 - 1e-4), diana, reduction="none")
    return (bce * cob).sum() / (cob.sum() + 1e-6)


def fixa_llavor(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def talls_mostra(ds, n=4):
    per_cas = {}
    for i, (c, z) in enumerate(ds.index):
        per_cas.setdefault(c, []).append((i, z))
    idx = []
    for c in list(per_cas)[:n]:
        _, tmax, cob = ds._volums(c)
        millor = max(per_cas[c], key=lambda t: ((tmax[t[1]] * ESCALA_TMAX > 6) & (cob[t[1]] > 0)).sum())
        idx.append(millor[0])
    return idx


@torch.no_grad()
def figura(G, ds, idx, dev, ruta, ep):
    G.eval()
    f, ax = plt.subplots(len(idx), 4, figsize=(11, 2.8 * len(idx)), squeeze=False)
    for fila, i in enumerate(idx):
        b = ds[i]
        x = b["ncct"][None].to(dev)
        pred, prior = G(x, b["z"][None].to(dev), b["seg"][None].to(dev), b["costat"][None].to(dev))
        cob = b["cob"][0].numpy()
        imgs = [(b["ncct"][1].numpy(), "gray", 0, 1, "NCCT"),
                (prior[0, 0].cpu().numpy(), "magma", 0, 1, "prior atles"),
                (pred[0, 0].cpu().numpy() * cob * ESCALA_TMAX, "jet", 0, 15, "Tmax generat"),
                (b["tmax"][0].numpy() * ESCALA_TMAX, "jet", 0, 15, "Tmax real")]
        for col, (im, cm, lo, hi, tit) in enumerate(imgs):
            a = ax[fila, col]
            a.imshow(np.rot90(im), cmap=cm, vmin=lo, vmax=hi)
            a.axis("off")
            if fila == 0:
                a.set_title(tit, fontsize=10)
        prompt = f"{SEGMENTS[int(b['seg'])]} {COSTATS[int(b['costat'])]}"
        ax[fila, 0].text(2, 12, f"{b['cas'][-4:]}  z={int(b['z'])}\n{prompt}", color="w", fontsize=8, va="top")
    f.suptitle(f"època {ep}  (convenció neurològica: esquerra del pacient a l'esquerra)", fontsize=11)
    plt.tight_layout()
    plt.savefig(ruta, dpi=80)
    plt.close(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mni_dir", type=Path, required=True)
    ap.add_argument("--vasos", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--taula", type=Path, default=None)
    ap.add_argument("--epoques", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_adv", type=float, default=0.01)
    ap.add_argument("--lambda_atles", type=float, default=1.0)
    ap.add_argument("--lambda_suau", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--cada_fig", type=int, default=5)
    ap.add_argument("--llavor", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    a = ap.parse_args()

    fixa_llavor(a.llavor)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "figures").mkdir(exist_ok=True)
    (a.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2))

    taula = None if a.taula is None else np.load(a.taula)
    llegeix = lambda p: [c for c in p.read_text().split() if c]
    dtr = TallsMNI(a.mni_dir, a.vasos, llegeix(a.train), augmenta=True)
    dva = TallsMNI(a.mni_dir, a.vasos, llegeix(a.val))
    g = torch.Generator().manual_seed(a.llavor)
    ltr = DataLoader(dtr, a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                     generator=g, persistent_workers=a.workers > 0, pin_memory=True)
    lva = DataLoader(dva, a.batch, shuffle=False, num_workers=a.workers,
                     persistent_workers=a.workers > 0, pin_memory=True)
    idx_fig = talls_mostra(dva)

    G = Generador(etiquetes_atles(a.mni_dir), taula_inicial=taula).to(dev)
    D = DiscriminadorPatch().to(dev)
    oG = torch.optim.Adam(G.parameters(), a.lr, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), a.lr, betas=(0.5, 0.999))
    sG = torch.optim.lr_scheduler.CosineAnnealingLR(oG, a.epoques, eta_min=a.lr * 0.05)
    sD = torch.optim.lr_scheduler.CosineAnnealingLR(oD, a.epoques, eta_min=a.lr * 0.05)

    n_g = sum(p.numel() for p in G.parameters())
    n_d = sum(p.numel() for p in D.parameters())
    print(f"train {len(dtr.casos)} casos / {len(dtr)} talls | val {len(dva.casos)} casos / {len(dva)} talls", flush=True)
    print(f"G {n_g / 1e6:.2f} M params | D {n_d / 1e6:.2f} M params | {dev}", flush=True)

    hist = a.out / "historial.csv"
    with open(hist, "w", newline="") as f:
        csv.writer(f).writerow(CAMPS)
    millor = float("inf")

    for ep in range(a.epoques):
        t0 = time.time()
        G.train()
        acc = {k: 0.0 for k in ("l1", "adv", "atles", "suau", "lD")}
        nb = 0
        for b in ltr:
            x, y, cob, z = (b[k].to(dev, non_blocking=True) for k in ("ncct", "tmax", "cob", "z"))
            seg, costat = b["seg"].to(dev), b["costat"].to(dev)
            pred, prior = G(x, z, seg, costat)

            oD.zero_grad(set_to_none=True)
            real = D(x, prior.detach(), y)
            fals = D(x, prior.detach(), pred.detach() * cob)
            lD = (F.mse_loss(real, torch.ones_like(real)) + F.mse_loss(fals, torch.zeros_like(fals))) / 2
            lD.backward()
            oD.step()

            oG.zero_grad(set_to_none=True)
            fals = D(x, prior, pred * cob)
            l1 = perdua_l1(pred, y, cob)
            adv = F.mse_loss(fals, torch.ones_like(fals))
            la = perdua_atles(prior, y, cob)
            ls = G.atles.suavitat()
            lG = l1 + a.lambda_adv * adv + a.lambda_atles * la + a.lambda_suau * ls
            lG.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), a.clip)
            oG.step()

            for k, v in (("l1", l1), ("adv", adv), ("atles", la), ("suau", ls), ("lD", lD)):
                acc[k] += v.item()
            nb += 1

        G.eval()
        tot, tot_a, n = 0.0, 0.0, 0
        with torch.no_grad():
            for b in lva:
                x, y, cob, z = (b[k].to(dev, non_blocking=True) for k in ("ncct", "tmax", "cob", "z"))
                pred, prior = G(x, z, b["seg"].to(dev), b["costat"].to(dev))
                tot += perdua_l1(pred, y, cob).item() * len(x)
                tot_a += perdua_atles(prior, y, cob).item() * len(x)
                n += len(x)
        mae_s = tot / max(n, 1) * ESCALA_TMAX
        lr = oG.param_groups[0]["lr"]
        sG.step()
        sD.step()

        fila = {"epoca": ep, "lr": lr, **{k: v / max(nb, 1) for k, v in acc.items()},
                "val_mae_s": mae_s, "val_atles": tot_a / max(n, 1), "temps_s": time.time() - t0}
        with open(hist, "a", newline="") as f:
            csv.writer(f).writerow([f"{fila[k]:.5g}" if isinstance(fila[k], float) else fila[k] for k in CAMPS])
        print(f"època {ep:3d} | L1 {fila['l1']:.4f}  adv {fila['adv']:.3f}  atles {fila['atles']:.4f}  "
              f"D {fila['lD']:.3f} | val MAE {mae_s:.2f} s  val atles {fila['val_atles']:.4f} | "
              f"{fila['temps_s']:.0f} s", flush=True)

        estat = {"G": G.state_dict(), "D": D.state_dict(), "oG": oG.state_dict(), "oD": oD.state_dict(),
                 "epoca": ep, "config": vars(a)}
        torch.save(estat, a.out / "ultim.pt")
        if mae_s < millor:
            millor = mae_s
            torch.save({"G": G.state_dict(), "epoca": ep, "val_mae_s": mae_s}, a.out / "millor.pt")
        if ep % a.cada_fig == 0 or ep == a.epoques - 1:
            figura(G, dva, idx_fig, dev, a.out / "figures" / f"epoca_{ep:03d}.png", ep)

    print(f"fet. millor val MAE {millor:.2f} s", flush=True)


if __name__ == "__main__":
    main()
