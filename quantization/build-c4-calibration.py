import json
import hashlib
from pathlib import Path
from datasets import load_dataset

OUT = Path("quantization/calibration/c4-en-00001-first1024.jsonl")

ds = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(1024))

texts = [row["text"] for row in ds]

with OUT.open("w", encoding="utf-8") as f:
    for i, text in enumerate(texts):
        f.write(json.dumps(
            {"id": i, "text": text},
            ensure_ascii=False
        ) + "\n")

sha = hashlib.sha256(OUT.read_bytes()).hexdigest()

print(f"samples : {len(texts)}")
print(f"output  : {OUT}")
print(f"bytes   : {OUT.stat().st_size}")
print(f"sha256  : {sha}")
