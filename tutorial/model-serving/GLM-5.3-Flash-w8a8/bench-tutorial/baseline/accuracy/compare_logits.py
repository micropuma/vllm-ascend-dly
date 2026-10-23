#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def load_capture(path: Path):
    with np.load(path, allow_pickle=False) as data:
        logits = np.asarray(data["logits"])
        prompt_token_ids = np.asarray(data["prompt_token_ids"])
        token_ids = np.asarray(data["token_ids"])
        native_dtype = str(data["native_dtype"]) if "native_dtype" in data else str(logits.dtype)
        native_bytes = np.asarray(data["native_bytes"]) if "native_bytes" in data else np.ascontiguousarray(logits).view(np.uint8).reshape(logits.shape[0], -1)
        valid_vocab_size = int(data["valid_vocab_size"]) if "valid_vocab_size" in data else logits.shape[-1]
    if logits.ndim != 2 or prompt_token_ids.ndim != 1 or token_ids.ndim != 1 or logits.shape[0] != token_ids.shape[0]:
        raise ValueError(
            f"invalid capture shapes in {path}: logits={logits.shape}, "
            f"prompt_token_ids={prompt_token_ids.shape}, token_ids={token_ids.shape}"
        )
    if not 0 < valid_vocab_size <= logits.shape[1]:
        raise ValueError(f"invalid valid_vocab_size in {path}: {valid_vocab_size}")
    sizes = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4}
    itemsize = sizes.get(native_dtype, logits.dtype.itemsize)
    if native_bytes.dtype != np.uint8 or native_bytes.shape != (logits.shape[0], logits.shape[1] * itemsize):
        raise ValueError(f"invalid native byte representation in {path}")
    return logits[:, :valid_vocab_size], prompt_token_ids, token_ids, native_dtype, native_bytes[:, :valid_vocab_size * itemsize]


def logsumexp(values):
    maximum = np.max(values, axis=-1, keepdims=True)
    return maximum + np.log(np.exp(values - maximum).sum(axis=-1, keepdims=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare aligned full-vocabulary teacher-forced logits captures.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    reference, ref_prompt_ids, ref_ids, ref_dtype, ref_bytes = load_capture(args.reference)
    candidate, candidate_prompt_ids, candidate_ids, candidate_dtype, candidate_bytes = load_capture(args.candidate)
    if (
        reference.shape != candidate.shape
        or not np.array_equal(ref_prompt_ids, candidate_prompt_ids)
        or not np.array_equal(ref_ids, candidate_ids)
    ):
        raise SystemExit(
            f"unaligned captures: shapes {reference.shape}/{candidate.shape}; "
            f"prompt token IDs equal={np.array_equal(ref_prompt_ids, candidate_prompt_ids)}; "
            f"teacher-forced token IDs equal={np.array_equal(ref_ids, candidate_ids)}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise SystemExit("NaN or Inf in logits")

    ref = reference.astype(np.float64)
    got = candidate.astype(np.float64)
    abs_diff = np.abs(got - ref)
    relative_floor = max(float(np.max(np.abs(ref))) * 1e-12, np.finfo(np.float64).tiny)
    relative_denominator = np.maximum(np.abs(ref), relative_floor)
    ref_norm = np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), np.finfo(np.float64).tiny)
    got_norm = np.maximum(np.linalg.norm(got, axis=1, keepdims=True), np.finfo(np.float64).tiny)
    ref_logp = ref - logsumexp(ref)
    got_logp = got - logsumexp(got)
    ref_p, got_p = np.exp(ref_logp), np.exp(got_logp)
    log_m = np.logaddexp(ref_logp, got_logp) - np.log(2.0)
    js = 0.5 * (
        np.sum(ref_p * (ref_logp - log_m), axis=1)
        + np.sum(got_p * (got_logp - log_m), axis=1)
    )
    close = np.isclose(got, ref, atol=args.atol, rtol=args.rtol)
    dtype_match = ref_dtype == candidate_dtype
    if dtype_match:
        bitwise_equal_elements = int(np.count_nonzero(np.all(ref_bytes == candidate_bytes, axis=1)))
    else:
        bitwise_equal_elements = 0
    result = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "shape": list(reference.shape),
        "reference_dtype": ref_dtype,
        "candidate_dtype": candidate_dtype,
        "bitwise_dtype_match": dtype_match,
        "prompt_token_ids_equal": True,
        "teacher_forced_token_ids_equal": True,
        "bitwise_equal_positions": bitwise_equal_elements,
        "positions": int(reference.shape[0]),
        "bitwise_equal_fraction": float(bitwise_equal_elements / reference.shape[0]),
        "allclose": bool(np.all(close)),
        "atol": args.atol,
        "rtol": args.rtol,
        "max_abs_error": float(abs_diff.max(initial=0)),
        "mean_abs_error": float(abs_diff.mean()),
        "max_relative_error": float((abs_diff / relative_denominator).max(initial=0)),
        "mean_relative_error": float((abs_diff / relative_denominator).mean()),
        "mean_cosine_similarity": float(np.mean(np.sum((ref / ref_norm) * (got / got_norm), axis=1))),
        "top1_agreement_fraction": float(np.mean(np.argmax(ref, axis=1) == np.argmax(got, axis=1))),
        "mean_js_divergence_nats": float(np.mean(js)),
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
