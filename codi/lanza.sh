set -e
ISLES=$1
ATLES=$2
OUT=${3:-sortida}
cd "$(dirname "$0")"
D=../dades
cat $D/train.txt $D/val.txt > $D/tots.txt
python -u registra_mni.py --isles "$ISLES" --atles "$ATLES" --casos $D/tots.txt --out "$OUT/mni"
python -u inicialitza_atles.py --mni_dir "$OUT/mni" --vasos $D/vas_ocluit_i_territori.json --train $D/train.txt --out "$OUT/taula_inicial.npy"
python -u entrena.py --mni_dir "$OUT/mni" --vasos $D/vas_ocluit_i_territori.json --train $D/train.txt --val $D/val.txt --taula "$OUT/taula_inicial.npy" --epoques 100 --out "$OUT/ck"
