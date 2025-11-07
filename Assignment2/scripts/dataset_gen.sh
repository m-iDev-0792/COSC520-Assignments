SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
cd ..
mkdir -p dataset
python src/dataset/dataset_gen.py --n 100000000 --out ./dataset --chunk-size 1000000 --num-workers 8 --overwrite

