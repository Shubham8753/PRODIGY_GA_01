# GPT-2 Fine-tuning (small demo)

Steps to run locally:

1. Create a Python environment and install requirements:

```bash
python -m pip install -r requirements.txt
```

2. Put your training text into `dataset.txt` (one or more paragraphs).

3. Run training (adjust `per_device_train_batch_size` in `train.py` as needed):

```bash
python train.py
```

Notes:
- This script tokenizes input and sets `labels` equal to `input_ids` for causal LM training.
- For GPU training ensure `torch` is installed with CUDA support.
- For larger datasets or longer training, consider using `accelerate`.
