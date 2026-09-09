import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

BASE = "/models/Qwen3.8-27B-BF16"
SRC = Path("/work/quantization/calibration/c4-en-00001-first1024.jsonl")
OUT = Path("/work/quantization/calibration/c4-fixed-128x2048.json")

EXPECTED_SRC_SHA256 = (
    "0a5a4579688b24318d172343b62675c9980ec7830a3d10298a1abc8e7d35235c"
)

SEQ_LEN = 2048
NUM_SEQS = 128

src_sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
if src_sha != EXPECTED_SRC_SHA256:
    raise RuntimeError(f"Source calibration SHA mismatch: {src_sha}")

tokenizer = AutoTokenizer.from_pretrained(BASE)

all_ids = []

with SRC.open("r", encoding="utf-8") as f:
    for line in f:
        text = json.loads(line)["text"]

        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        all_ids.extend(ids)

        # Explicit separator between C4 documents.
        if tokenizer.eos_token_id is not None:
            all_ids.append(tokenizer.eos_token_id)

needed = NUM_SEQS * SEQ_LEN

if len(all_ids) < needed:
    raise RuntimeError(
        f"Not enough tokens: need {needed}, have {len(all_ids)}"
    )

all_ids = all_ids[:needed]

samples = []

for i in range(NUM_SEQS):
    start = i * SEQ_LEN
    end = start + SEQ_LEN
    ids = all_ids[start:end]

    samples.append({
        "id": i,
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
    })

payload = {
    "source_sha256": src_sha,
    "num_sequences": NUM_SEQS,
    "sequence_length": SEQ_LEN,
    "total_tokens": needed,
    "samples": samples,
}

OUT.write_text(
    json.dumps(payload, separators=(",", ":")),
    encoding="utf-8",
)

out_sha = hashlib.sha256(OUT.read_bytes()).hexdigest()

print(f"source sha256 : {src_sha}")
print(f"sequences     : {NUM_SEQS}")
print(f"sequence len  : {SEQ_LEN}")
print(f"total tokens  : {needed}")
print(f"output        : {OUT}")
print(f"output bytes  : {OUT.stat().st_size}")
print(f"output sha256 : {out_sha}")
