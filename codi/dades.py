import json
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from model import SEGMENTS, COSTATS

GRUP = {"ICA-C1-C5": "caròtida_cervical", "ICA-C6-C7": "caròtida_terminal", "M1": "M1", "M2": "M2", "M3": "M2",
        "A1A2": "ACA", "A3": "ACA", "P1P2": "ACP", "P3P4": "ACP", "BA": "basilar"}
ESCALA_TMAX = 30.0
MIDA = 256
MIN_CERVELL = 2000


def vas_a_prompt(vas):
    if not vas:
        return None
    if vas[:2] in ("R-", "L-"):
        costat, seg = ("dreta" if vas[0] == "R" else "esquerra"), vas[2:]
    else:
        costat, seg = "mig", vas
    if seg not in GRUP:
        return None
    return SEGMENTS.index(GRUP[seg]), COSTATS.index(costat)


def carrega(p):
    a = np.asanyarray(nib.as_closest_canonical(nib.load(str(p))).dataobj).astype(np.float32)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def a_mida(t, mode):
    return F.interpolate(t[None], size=(MIDA, MIDA), mode=mode,
                         **({} if mode == "nearest" else {"align_corners": False}))[0]


def etiquetes_atles(mni_dir):
    a = torch.from_numpy(carrega(Path(mni_dir) / "atles_mni.nii.gz").round()).permute(2, 0, 1)
    return a_mida(a, "nearest").long()


def talls_valids(mni_dir):
    a = carrega(Path(mni_dir) / "atles_mni.nii.gz")
    n = (a > 0).sum((0, 1))
    return [z for z in range(1, a.shape[2] - 1) if n[z] > MIN_CERVELL]


class TallsMNI(Dataset):
    def __init__(self, mni_dir, vasos_json, casos, augmenta=False):
        self.dir = Path(mni_dir)
        self.augmenta = augmenta
        vasos = json.loads(Path(vasos_json).read_text(encoding="utf-8"))
        self.prompt = {c: vas_a_prompt(vasos.get(c, {}).get("vas")) for c in casos}
        self.casos = [c for c in casos if self.prompt[c] is not None]
        zs = talls_valids(self.dir)
        self.cau = self.dir / "cau"
        self.cau.mkdir(exist_ok=True)
        self.index = []
        for c in self.casos:
            ruta = self.cau / f"{c}.npy"
            if not ruta.exists():
                ncct = np.clip(carrega(self.dir / f"{c}_ncct.nii.gz"), 0, 80) / 80.0
                tmax = np.clip(carrega(self.dir / f"{c}_tmax.nii.gz"), 0, ESCALA_TMAX) / ESCALA_TMAX
                cob = carrega(self.dir / f"{c}_cob.nii.gz") > 0
                np.save(ruta, np.stack([ncct, tmax, cob]).transpose(0, 3, 1, 2).astype(np.float16))
            cob = np.load(ruta, mmap_mode="r")[2].sum((1, 2))
            self.index += [(c, z) for z in zs if cob[z] > 0]

    def _volums(self, c):
        v = np.load(self.cau / f"{c}.npy", mmap_mode="r")
        return v[0], v[1], v[2]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        c, z = self.index[i]
        ncct, tmax, cob = self._volums(c)
        talls = torch.from_numpy(np.array(ncct[z - 1:z + 2], dtype=np.float32))
        mirall = torch.flip(talls[1:2], dims=[1])
        x = a_mida(torch.cat([talls, mirall], 0), "bilinear")
        y = a_mida(torch.from_numpy(np.array(tmax[z][None], dtype=np.float32)), "bilinear")
        cb = a_mida(torch.from_numpy(np.array(cob[z][None], dtype=np.float32)), "nearest")
        seg, costat = self.prompt[c]
        if self.augmenta and torch.rand(1) < 0.5:
            x, y, cb = (torch.flip(t, dims=[1]) for t in (x, y, cb))
            x[3] = torch.flip(x[1], dims=[0])
            costat = {0: 1, 1: 0, 2: 2}[costat]
        return {"ncct": x, "tmax": y, "cob": cb, "z": torch.tensor(z),
                "seg": torch.tensor(seg), "costat": torch.tensor(costat), "cas": c}
