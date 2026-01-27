# Dataset Preparation

## `prepare_datasets.py`

Processes raw text files into structured training/validation binaries.

### Input
- `data/raw/train/` — Organized into 5 subdirectories (`1/` to `5/`), representing increasing complexity
- `data/raw/val/` — Independent validation set (w3schools Python docs)

### Output
- `data/processed/train_cum_1.txt` to `train_cum_5.txt` — Cumulative training texts
- `data/processed/val.txt` — Validation text

### Usage
```bash
python scripts/prepare_datasets.py