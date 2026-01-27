"""
Prepare cumulative training and validation datasets from raw text files.

Input:
  data/raw/train/1/ ... /5/   → training stages (any nesting)
  data/raw/val/               → validation texts (any nesting)

Output:
  data/processed/train_cum_1.txt ... train_cum_5.txt
  data/processed/val.txt
"""

import os
import glob
from pathlib import Path

def collect_txt_files(root_dir: str) -> list[str]:
    """Recursively collect all .txt files in a directory."""
    return [str(p) for p in Path(root_dir).rglob("*.txt")]

def read_file_safe(path: str) -> str:
    """Read a text file with fallback encodings."""
    for encoding in ["utf-8", "cp1251", "latin1"]:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # If all fail, read as binary and decode with errors='replace'
    with open(path, "rb") as f:
        return f.read().decode(errors="replace")

def main():
    project_root = Path(__file__).parent.parent  # LLMagogyV2/
    raw_train_dir = project_root / "data" / "raw" / "train"
    raw_val_dir = project_root / "data" / "raw" / "val"
    processed_dir = project_root / "data" / "processed"

    # Create output directory
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Collect and build cumulative training datasets
    cumulative_content = ""
    for stage in range(1, 6):  # stages 1 to 5
        stage_dir = raw_train_dir / str(stage)
        if not stage_dir.exists():
            print(f"Warning: stage directory {stage_dir} does not exist. Skipping.")
            continue

        print(f"Processing stage {stage} from {stage_dir}")
        txt_files = collect_txt_files(str(stage_dir))
        print(f"  Found {len(txt_files)} .txt files")

        stage_content = ""
        for file_path in sorted(txt_files):  # sorted for reproducibility
            try:
                text = read_file_safe(file_path)
                stage_content += text + "\n\n"
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")

        cumulative_content += stage_content

        # Save cumulative dataset up to this stage
        output_path = processed_dir / f"train_cum_{stage:1d}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cumulative_content.strip())
        print(f"  Saved {output_path}")

    # Prepare validation dataset
    print("Processing validation dataset...")
    val_files = collect_txt_files(str(raw_val_dir))
    print(f"  Found {len(val_files)} .txt files in validation set")

    val_content = ""
    for file_path in sorted(val_files):
        try:
            text = read_file_safe(file_path)
            val_content += text + "\n\n"
        except Exception as e:
            print(f"  Error reading validation file {file_path}: {e}")

    val_output = processed_dir / "val.txt"
    with open(val_output, "w", encoding="utf-8") as f:
        f.write(val_content.strip())
    print(f"  Saved {val_output}")

    print("\n✅ Dataset preparation complete.")
    print(f"   Training datasets: {processed_dir}/train_cum_*.txt")
    print(f"   Validation dataset: {processed_dir}/val.txt")

if __name__ == "__main__":
    main()
