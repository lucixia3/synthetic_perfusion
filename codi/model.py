import torch
import torch.nn as nn
import torch.nn.functional as F

SEGMENTS = ["caròtida_cervical", "caròtida_terminal", "M1", "M2", "ACA", "ACP", "basilar"]
COSTATS = ["dreta", "esquerra", "mig"]
MAPES = ["tmax"]


def prompt_onehot(seg, costat):
    return torch.cat([F.one_hot(seg, len(SEGMENTS)), F.one_hot(costat, len(COSTATS))], 1).float()


class EmbeddingPrompt(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(len(SEGMENTS) + len(COSTATS), dim), nn.ReLU(),
                                 nn.Linear(dim, dim), nn.ReLU())

    def forward(self, seg, costat):
        return self.mlp(prompt_onehot(seg, costat))


class ModulAtles(nn.Module):
    def __init__(self, etiquetes, taula_inicial=None, mida_correccio=(16, 32, 32)):
        super().__init__()
        self.register_buffer("etiquetes", etiquetes.to(torch.uint8))
        self.n_regions = int(etiquetes.max())
        t = torch.zeros(len(SEGMENTS), len(COSTATS), self.n_regions)
        if taula_inicial is not None:
            t = torch.as_tensor(taula_inicial, dtype=torch.float32)
        self.taula = nn.Parameter(t)
        self.correccio = nn.Parameter(torch.zeros(len(SEGMENTS), len(COSTATS), *mida_correccio))

    def _correccio(self, z, seg, costat, mida):
        c = self.correccio[seg, costat].unsqueeze(1)
        b = c.shape[0]
        zn = z.float() / (self.etiquetes.shape[0] - 1) * 2 - 1
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, mida[0], device=c.device),
                                torch.linspace(-1, 1, mida[1], device=c.device), indexing="ij")
        graella = torch.stack([gx.expand(b, -1, -1), gy.expand(b, -1, -1),
                               zn[:, None, None].expand(b, *mida)], -1).unsqueeze(1)
        return F.grid_sample(c, graella, align_corners=True)[:, 0, 0]

    def suavitat(self):
        c = self.correccio
        return sum((torch.diff(c, dim=d) ** 2).mean() for d in (2, 3, 4))

    def forward(self, z, seg, costat):
        et = self.etiquetes[z].long()
        t = self.taula[seg, costat]
        logit = torch.gather(t, 1, (et.clamp(min=1) - 1).flatten(1)).view_as(et)
        logit = logit + self._correccio(z, seg, costat, et.shape[-2:])
        return (torch.sigmoid(logit) * (et > 0)).unsqueeze(1)


class FiLM(nn.Module):
    def __init__(self, dim_emb, canals):
        super().__init__()
        self.lin = nn.Linear(dim_emb, 2 * canals)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, h, emb):
        g, b = self.lin(emb).chunk(2, 1)
        return h * (1 + g[:, :, None, None]) + b[:, :, None, None]


class Bloc(nn.Module):
    def __init__(self, c_in, c_out, dim_emb):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.n1 = nn.GroupNorm(8, c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.n2 = nn.GroupNorm(8, c_out)
        self.film = FiLM(dim_emb, c_out)

    def forward(self, x, emb):
        h = F.silu(self.n1(self.c1(x)))
        h = self.film(self.n2(self.c2(h)), emb)
        return F.silu(h)


class Encoder(nn.Module):
    def __init__(self, c_in, canals, dim_emb):
        super().__init__()
        self.blocs = nn.ModuleList()
        c = c_in
        for co in canals:
            self.blocs.append(Bloc(c, co, dim_emb))
            c = co

    def forward(self, x, emb):
        salts = []
        for i, b in enumerate(self.blocs):
            if i:
                x = F.avg_pool2d(x, 2)
            x = b(x, emb)
            salts.append(x)
        return salts


class Decoder(nn.Module):
    def __init__(self, canals, dim_emb):
        super().__init__()
        cs = list(reversed(canals))
        self.blocs = nn.ModuleList()
        for i in range(1, len(cs)):
            self.blocs.append(Bloc(cs[i - 1] + cs[i] + 1, cs[i], dim_emb))
        self.sortida = nn.Conv2d(cs[-1], 1, 1)

    def forward(self, salts, prior, emb):
        x = salts[-1]
        for b, s in zip(self.blocs, reversed(salts[:-1])):
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            p = F.interpolate(prior, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = b(torch.cat([x, s, p], 1), emb)
        return self.sortida(x)


class Generador(nn.Module):
    def __init__(self, etiquetes, c_in=4, canals=(32, 64, 128, 256), dim_emb=64, taula_inicial=None):
        super().__init__()
        self.emb = EmbeddingPrompt(dim_emb)
        self.atles = ModulAtles(etiquetes, taula_inicial)
        self.encoder = Encoder(c_in + 1, canals, dim_emb)
        self.decoder = Decoder(canals, dim_emb)

    def forward(self, ncct, z, seg, costat):
        emb = self.emb(seg, costat)
        prior = self.atles(z, seg, costat)
        salts = self.encoder(torch.cat([ncct, prior], 1), emb)
        return self.decoder(salts, prior, emb), prior


class DiscriminadorPatch(nn.Module):
    def __init__(self, c_in=4 + 1 + len(MAPES), base=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.xarxa = nn.Sequential(
            sn(nn.Conv2d(c_in, base, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(base, base * 2, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(base * 2, base * 4, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(base * 4, base * 4, 4, 1, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(base * 4, 1, 4, 1, 1)))

    def forward(self, ncct, prior, tmax):
        return self.xarxa(torch.cat([ncct, prior, tmax], 1))
