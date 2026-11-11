import csv
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PAS_Z = 2
SIGMA_MM = 3.0
EROSIO_MM = 4.0


def llegeix(p):
    im = sitk.ReadImage(str(p), sitk.sitkFloat32)
    a = np.nan_to_num(sitk.GetArrayFromImage(im), nan=0.0, posinf=0.0, neginf=0.0)
    out = sitk.GetImageFromArray(a)
    out.CopyInformation(im)
    return out


def cervell_ncct(ncct):
    r = [max(1, int(round(EROSIO_MM / s))) for s in ncct.GetSpacing()]
    m = sitk.BinaryThreshold(ncct, -5, 80, 1, 0)
    m = sitk.BinaryErode(m, r)
    m = sitk.Cast(sitk.RelabelComponent(sitk.ConnectedComponent(m), sortByObjectSize=True) == 1, sitk.sitkUInt8)
    m = sitk.BinaryDilate(m, r)
    return sitk.BinaryFillhole(m)


def suau(m):
    return sitk.SmoothingRecursiveGaussian(sitk.Cast(m, sitk.sitkFloat32), SIGMA_MM)


def registra(fix, mov):
    ini = sitk.CenteredTransformInitializer(fix, mov, sitk.Similarity3DTransform(),
                                            sitk.CenteredTransformInitializerFilter.MOMENTS)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2, seed=42)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-3, numberOfIterations=200,
                                               relaxationFactor=0.5)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SetInitialTransform(ini, inPlace=False)
    return R.Execute(fix, mov)


def graella(atles):
    sp = atles.GetSpacing()
    sz = atles.GetSize()
    ref = sitk.Image([sz[0], sz[1], sz[2] // PAS_Z], sitk.sitkFloat32)
    cx = np.nonzero(sitk.GetArrayFromImage(atles) > 0)[2].mean()
    mig = atles.TransformContinuousIndexToPhysicalPoint((cx, 0.0, 0.0))
    o = atles.TransformContinuousIndexToPhysicalPoint(((sz[0] - 1) / 2, 0.0, 0.0))
    ref.SetOrigin([a + b - c for a, b, c in zip(atles.GetOrigin(), mig, o)])
    ref.SetDirection(atles.GetDirection())
    ref.SetSpacing((sp[0], sp[1], sp[2] * PAS_Z))
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isles", type=Path, required=True)
    ap.add_argument("--atles", type=Path, required=True)
    ap.add_argument("--casos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    atles = sitk.ReadImage(str(a.atles), sitk.sitkUInt8)
    cervell_atles = atles > 0
    fix = suau(cervell_atles)
    ref = graella(atles)
    sitk.WriteImage(sitk.Resample(atles, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8),
                    str(a.out / "atles_mni.nii.gz"))
    cervell_ref = sitk.GetArrayFromImage(sitk.Resample(cervell_atles, ref, sitk.Transform(),
                                                       sitk.sitkNearestNeighbor, 0)) > 0

    files = []
    for c in [c for c in a.casos.read_text().split() if c]:
        ncct = llegeix(a.isles / "raw_data" / c / "ses-01" / f"{c}_ses-01_ncct.nii.gz")
        tmax = llegeix(a.isles / "derivatives" / c / "ses-01" / "perfusion-maps" / f"{c}_ses-01_space-ncct_tmax.nii.gz")
        tmax = sitk.Threshold(tmax, 0.0, 1e4, 0.0)
        cervell = cervell_ncct(ncct)
        T = registra(fix, suau(cervell))

        sitk.WriteImage(sitk.Resample(ncct, ref, T, sitk.sitkLinear, 0.0), str(a.out / f"{c}_ncct.nii.gz"))
        sitk.WriteImage(sitk.Resample(tmax, ref, T, sitk.sitkLinear, 0.0), str(a.out / f"{c}_tmax.nii.gz"))
        cob = sitk.Resample(sitk.Cast(tmax > 0, sitk.sitkUInt8), ref, T, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        sitk.WriteImage(cob, str(a.out / f"{c}_cob.nii.gz"))
        sitk.WriteTransform(T, str(a.out / f"{c}_mni_a_pacient.tfm"))

        m = sitk.GetArrayFromImage(sitk.Resample(cervell, ref, T, sitk.sitkNearestNeighbor, 0)) > 0
        dice = 2 * (m & cervell_ref).sum() / (m.sum() + cervell_ref.sum() + 1e-6)
        files.append({"cas": c, "dice_cervell": round(float(dice), 3)})
        print(f"{c}  dice cervell {dice:.3f}", flush=True)

    with open(a.out / "registre.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["cas", "dice_cervell"])
        w.writeheader()
        w.writerows(files)


if __name__ == "__main__":
    main()
