# Training Scripts

Each script implements one stage of the LLMagogyV2 experimental pipeline:

| Script | Purpose |
|--------|---------|
| `llmagogy_step1.py` | Baseline: End-to-end training on full dataset |
| `llmagogy_step2.py` | Progressive growth on full dataset (uniform expansion) |
| `llmagogy_step3.py` | Progressive growth on **cumulative subsets** (stage-wise data) |
| `llmagogy_step4.py` | Progressive growth with **more granular stages** |
| `llmagogy_step5.py` | Optimized stopping criterion (early stop on small loss delta) |
| `llmagogy_step6.py` | Step 5 + cumulative subsets |
| `llmagogy_step7.py` | Step 5 + more granular stages |
| `llmagogy_step8.py` | Step 5 + **warmup** of new weights |
| `llmagogy_step9.py` | Step 5 + **gradient freezing** (`weight_freeze = 0.8`) |
| `llmagogy_step10.py`| Step 5 + **warmup + gradient freezing** |

All scripts:
- Use the same GPT architecture
- Share common utilities (`expand_model`, `apply_freeze_and_warmup`)

> ⚙️ Modify hyperparameters directly in each script (scheduler, warmup, etc.).
