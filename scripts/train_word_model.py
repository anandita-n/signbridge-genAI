"""
SignBridge AI - Train the word-level sign model on extracted WLASL100 landmark sequences.

Run scripts/extract_sequences.py first. Uses the dataset's own train / val / test split and
also reports accuracy on test videos from signers that never appear in training (the honest
number for "a new person signs this word").

Usage (repo root):  python scripts/train_word_model.py [--epochs 120]
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from app.word_model import (FEATURE_DIM, PER_HAND, build_model, sample_frames, with_deltas)

ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = ROOT / "data" / "wlasl100" / "seq"
MODEL_OUT = ROOT / "models" / "word_model.pt"
METRICS_OUT = ROOT / "data" / "word_eval_metrics.json"


def load_all():
    clips = []
    for f in sorted(glob.glob(str(SEQ_DIR / "*.npz"))):
        z = np.load(f)
        clips.append({
            "x": z["features"], "label": int(z["label"]), "word": str(z["word"]),
            "split": str(z["split"]), "signer": int(z["signer"]),
        })
    return clips


def augment(x, rng):
    """x: (T, FEATURE_DIM). Noise, scale, and simulated missed detections."""
    x = x.copy()
    x *= rng.uniform(0.93, 1.07)
    x += rng.normal(0, 0.01, x.shape).astype(np.float32) * (np.abs(x) > 0)
    drop = rng.random(len(x)) < 0.08
    for slot in (0, 1):
        x[drop, slot * PER_HAND:(slot + 1) * PER_HAND] = 0.0
    return x


def make_tensor(clips, train, rng, mean=None, std=None):
    xs = []
    for c in clips:
        f = sample_frames(c["x"], train=train, rng=rng)
        if train:
            f = augment(f, rng)
        xs.append(with_deltas(f))
    x = np.stack(xs)  # (N, T, 2D)
    if mean is not None:
        x = (x - mean) / std
    return x


def evaluate(model, x, y, batch=256):
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            logits.append(model(torch.from_numpy(x[i:i + batch].transpose(0, 2, 1).copy())))
    logits = torch.cat(logits)
    top1 = (logits.argmax(1).numpy() == y).mean()
    k = min(5, logits.shape[1])
    top5 = np.mean([y[i] in logits[i].topk(k).indices.numpy() for i in range(len(y))])
    return float(top1), float(top5), logits.argmax(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--signer-split", action="store_true",
                    help="hold out whole signers for val/test (honest new-person accuracy); "
                         "does not overwrite the saved model")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)

    clips = load_all()
    words = {}
    for c in clips:
        words[c["label"]] = c["word"]
    vocab = [words[i] for i in sorted(words)]
    n_cls = max(words) + 1
    vocab = [words.get(i, f"class{i}") for i in range(n_cls)]
    if args.signer_split:
        # Whole signers go to val or test, so nobody in val/test appears in training.
        signers = sorted({c["signer"] for c in clips})
        np.random.default_rng(args.seed).shuffle(signers)
        counts = {s: sum(c["signer"] == s for c in clips) for s in signers}
        assign, filled = {}, {"test": 0, "val": 0}
        for s in signers:
            if filled["test"] < 0.15 * len(clips):
                assign[s] = "test"; filled["test"] += counts[s]
            elif filled["val"] < 0.12 * len(clips):
                assign[s] = "val"; filled["val"] += counts[s]
            else:
                assign[s] = "train"
        for c in clips:
            c["split"] = assign[c["signer"]]
    train = [c for c in clips if c["split"] == "train"]
    val = [c for c in clips if c["split"] == "val"]
    test = [c for c in clips if c["split"] == "test"]
    train_signers = {c["signer"] for c in train}
    if args.signer_split:
        print(f"signer-disjoint split: {len({c['signer'] for c in test})} test signers, "
              f"{len({c['signer'] for c in val})} val signers, {len(train_signers)} train signers")
    print(f"clips: {len(clips)} | train {len(train)} val {len(val)} test {len(test)} | classes {n_cls}")

    # channel statistics from un-augmented training frames
    base = make_tensor(train, False, rng)
    mean = base.reshape(-1, base.shape[-1]).mean(0)
    std = base.reshape(-1, base.shape[-1]).std(0) + 1e-3

    xv = make_tensor(val, False, rng, mean, std); yv = np.array([c["label"] for c in val])
    xt = make_tensor(test, False, rng, mean, std); yt = np.array([c["label"] for c in test])
    ytr = np.array([c["label"] for c in train])

    model = build_model(n_cls)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=args.epochs * ((len(train) + 63) // 64))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val, best_state, t0 = -1.0, None, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        xtr = make_tensor(train, True, rng, mean, std)
        perm = rng.permutation(len(train))
        total = 0.0
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            xb = torch.from_numpy(xtr[idx].transpose(0, 2, 1).copy())
            yb = torch.from_numpy(ytr[idx])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step(); sched.step()
            total += float(loss) * len(idx)
        if epoch % 5 == 0 or epoch == args.epochs:
            v1, v5, _ = evaluate(model, xv, yv)
            if v1 > best_val:
                best_val, best_state = v1, {k: t.clone() for k, t in model.state_dict().items()}
            print(f"epoch {epoch:3d}  loss {total / len(train):.3f}  val top1 {v1:.3f} top5 {v5:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    v1, v5, _ = evaluate(model, xv, yv)
    t1, t5, tp = evaluate(model, xt, yt)
    unseen = np.array([c["signer"] not in train_signers for c in test])
    unseen_acc = float((tp[unseen] == yt[unseen]).mean()) if unseen.any() else None
    print(f"\nBEST val  top1 {v1:.3f}  top5 {v5:.3f}")
    print(f"TEST      top1 {t1:.3f}  top5 {t5:.3f}  ({len(test)} clips)")
    print(f"TEST, signers never seen in training: "
          + (f"top1 {unseen_acc:.3f} on {int(unseen.sum())} clips" if unseen_acc is not None else "none in test split"))
    print(f"chance level: {1 / n_cls:.3f}")

    per_word = {}
    for i, w in enumerate(vocab):
        m = yt == i
        if m.any():
            per_word[w] = round(float((tp[m] == i).mean()), 3)
    ranked = sorted(per_word.items(), key=lambda kv: kv[1])
    print("hardest words:", ranked[:8]); print("easiest words:", ranked[-8:])

    if args.signer_split:
        print("(signer-split run: saved model and metrics left untouched)")
        return
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "words": vocab, "mean": mean, "std": std}, MODEL_OUT)
    METRICS_OUT.write_text(json.dumps({
        "classes": n_cls, "train_clips": len(train), "val_clips": len(val), "test_clips": len(test),
        "val_top1": v1, "val_top5": v5, "test_top1": t1, "test_top5": t5,
        "test_top1_unseen_signers": unseen_acc, "unseen_signer_clips": int(unseen.sum()),
        "chance": 1 / n_cls, "per_word_test_top1": per_word,
    }, indent=2))
    print(f"saved {MODEL_OUT} and {METRICS_OUT}")


if __name__ == "__main__":
    main()
