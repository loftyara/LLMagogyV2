# LLMagogyV2  
**Fast Multi-Stage Pre-Training of Large Language Models on Curated Datasets**

> *"Knowledge of a few principles frees one from knowledge of many facts."* — R. Descartes

This repository contains the implementation and experimental results of **LLMagogyV2**, a research project exploring **progressive model growth** as an alternative to traditional end-to-end pre-training of language models.

Unlike standard approaches that train a fixed-size model on massive chaotic corpora, LLMagogy trains a **small model first** and **gradually expands it** (depth and width) while training on a **structured, cumulative dataset**—inspired by human pedagogy.

The core idea was **not confirmed**: staged training on *non-repetitive, cumulative subsets* does **not** outperform full-dataset training.  
However, **progressive growth on the full dataset** consistently **reduces training time by 5–25%** without quality loss or even with best quality.

All code, datasets, and results are provided for full reproducibility.

---

## 📂 Repository Structure

- [`papers/`](papers/) — Research paper (`LLMagogyV2.pdf`) and raw experimental results (`LLMagogyV2.xlsx`)
- [`src/`](src/) — Training scripts for all stages (`llmagogy_step1.py` to `llmagogy_step10.py`)
- [`scripts/`](scripts/) — Dataset preparation utilities
- [`data/`](data/) — Input data (raw texts and processed binaries)

---

## 🚀 Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/loftyara/LLMagogyV2.git
   cd LLMagogyV2

2. **Set up environment**
   ```bash
   python -m venv venv
   # source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   pip install torch --index-url https://download.pytorch.org/whl/cu128

3. **Prepare datasets**
   ```bash
   python scripts/prepare_datasets.py

4. **Run an experiment (e.g., Step 5**
   ```bash
   python src/llmagogy_step5.py

💡 See [`papers/LLMagogyV2.pdf`](papers/LLMagogyV2.pdf) for full methodology, results, and conclusions.
