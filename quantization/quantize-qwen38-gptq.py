import argparse
import hashlib
import json
from pathlib import Path

from gptqmodel import GPTQModel, QuantizeConfig

BASE = "/models/Qwen3.8-27B-BF16"
CAL = Path("/work/quantization/calibration/c4-fixed-128x1024.json")

EXPECTED_CAL_SHA256 = (
    "ddfc570e23458c048951501231c2ff75fa175440b120045bbeb1790bea5d2599"
)

parser = argparse.ArgumentParser()
parser.add_argument("--group-size", type=int, required=True, choices=[32, 128])
parser.add_argument("--output", required=True)
args = parser.parse_args()

sha = hashlib.sha256(CAL.read_bytes()).hexdigest()
if sha != EXPECTED_CAL_SHA256:
    raise RuntimeError(
        f"Calibration SHA256 mismatch:\n"
        f"expected={EXPECTED_CAL_SHA256}\n"
        f"actual  ={sha}"
    )

payload = json.loads(CAL.read_text(encoding="utf-8"))

if payload["num_sequences"] != 128:
    raise RuntimeError(
        f"Expected 128 sequences, got {payload['num_sequences']}"
    )

if payload["sequence_length"] != 1024:
    raise RuntimeError(
        f"Expected sequence length 1024, got {payload['sequence_length']}"
    )

if payload["total_tokens"] != 131072:
    raise RuntimeError(
        f"Expected 131072 tokens, got {payload['total_tokens']}"
    )

calibration = []

for sample in payload["samples"]:
    ids = sample["input_ids"]
    mask = sample["attention_mask"]

    if len(ids) != 1024 or len(mask) != 1024:
        raise RuntimeError(
            f"Bad calibration sample {sample['id']}: "
            f"{len(ids)} ids / {len(mask)} mask"
        )

    calibration.append({
        "input_ids": ids,
        "attention_mask": mask,
    })

print(f"Base                : {BASE}")
print(f"Calibration         : {CAL}")
print(f"Calibration SHA256  : {sha}")
print(f"Sequences           : {len(calibration)}")
print(f"Sequence length     : 1024")
print(f"Total tokens        : {len(calibration) * 1024}")
print(f"Group size          : {args.group_size}")
print(f"Output              : {args.output}")

quant_config = QuantizeConfig(
    bits=4,
    group_size=args.group_size,
    sym=True,
    desc_act=False,
    lm_head=False,
    dynamic={
        "-:.*mtp.*": {}
    },
)

quant_config.hessian.chunk_size = 32
quant_config.dense_vram_strategy = "balanced"

model = GPTQModel.load(
    BASE,
    quantize_config=quant_config,
    device="xpu:0",
)

model.quantize(
    calibration,
    calibration_sort=None,
    batch_size=1,
)

model.save(args.output)

print("DONE")
