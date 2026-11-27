"""Train a GRU that reads each customer's statements in order.

    python scripts/train_gru.py                 # full training run
    python scripts/train_gru.py --sample 0.1    # quick run on 10% of training customers

This is the first of the project's improved models. Where the baselines see one
row of summary statistics per customer, the GRU reads the statements one month
at a time and carries a hidden state forward, so it can pick up patterns in the
order of events, such as payments slipping month after month, that a mean or a
max cannot represent.

Architecture, per month:
    11 category codes -> one small embedding each (4 numbers per category)
    [289 numeric inputs, 44 embedding values] -> linear layer -> LayerNorm -> ReLU
then over the months:
    2-layer GRU, hidden size 128, dropout 0.2 between layers
and from the GRU's state after the customer's last real statement:
    dropout -> 64 -> ReLU -> dropout -> 1 logit

Padding is handled with PyTorch's packed sequences: the GRU stops at each
customer's last real month, so padded months never touch the hidden state.

Training: binary cross-entropy, AdamW, batch 512. After each epoch the model is
scored on the validation customers with the competition metric, the learning
rate is halved when the score stops improving, and training stops after 3
epochs without improvement. The best epoch's weights are kept. As with
LightGBM's early stopping, this uses the validation set for model selection;
the test set is not loaded.

At the end the GRU is compared with the best baseline (LightGBM) on exactly
the same validation customers: overall, by history length, and blended.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from audit_data import ID, TARGET
from metric import amex_components
from paths import DATA
from train_baselines import PREDS, bootstrap

SEQ = DATA / "sequences"
MODELS = DATA / "models"
SEED = 5998

HIDDEN = 128
EMBED = 4
DROPOUT = 0.2
BATCH = 512
LR = 1e-3
MAX_EPOCHS = 20
PATIENCE = 3


class GRUModel(nn.Module):
    def __init__(self, n_inputs: int, cardinality: list[int]):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(k, EMBED) for k in cardinality)
        self.project = nn.Sequential(
            nn.Linear(n_inputs + EMBED * len(cardinality), HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
        )
        self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=2, batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 64), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(64, 1),
        )

    def forward(self, x, cat, lengths):
        emb = [e(cat[..., k]) for k, e in enumerate(self.embeddings)]
        h = self.project(torch.cat([x, *emb], dim=-1))
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.head(h_n[-1]).squeeze(-1)  # top layer's state after the last real month


def load(name: str):
    x = torch.from_numpy(np.load(SEQ / f"{name}_x.npy"))          # float16, cast per batch
    cat = torch.from_numpy(np.load(SEQ / f"{name}_cat.npy").astype(np.int64))
    lengths = torch.from_numpy(np.load(SEQ / f"{name}_mask.npy").sum(axis=1))
    y = np.load(SEQ / f"{name}_y.npy").astype(np.float32)
    ids = pd.read_parquet(SEQ / f"{name}_ids.parquet")[ID]
    return x, cat, lengths, y, ids


@torch.no_grad()
def predict(model, x, cat, lengths, batch=4096) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(x[i:i + batch].float(), cat[i:i + batch], lengths[i:i + batch]))
           for i in range(0, len(x), batch)]
    return torch.cat(out).numpy()


def compare(valid_ids, y, p_gru, lengths) -> None:
    lgb = pd.read_parquet(PREDS / "lightgbm_none_valid.parquet").set_index(ID).loc[valid_ids]
    p_lgb = lgb["pred"].to_numpy()
    assert (lgb[TARGET].to_numpy() == y).all(), "validation customers must line up"

    def rank(p):
        return pd.Series(p).rank(pct=True).to_numpy()

    p_blend = (rank(p_gru) + rank(p_lgb)) / 2
    rows = []
    for name, p in [("LightGBM (best baseline)", p_lgb), ("GRU", p_gru), ("Blend: mean rank", p_blend)]:
        parts = amex_components(y, p)
        lo, hi = np.percentile(bootstrap(y, p), [2.5, 97.5])
        rows.append({"Model": name, "Score": f"{parts['score']:.4f}", "95% CI": f"{lo:.4f} - {hi:.4f}",
                     "Gini": f"{parts['gini']:.4f}", "Top 4%": f"{parts['top_four']:.4f}"})
    print("\n## GRU against the best baseline\n")
    print(f"Same {len(y):,} validation customers for every row.\n")
    print(pd.DataFrame(rows).to_markdown(index=False))

    diff = bootstrap(y, p_gru, p_lgb)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f"\nGRU minus LightGBM: {diff.mean():+.4f}, 95% CI {lo:+.4f} to {hi:+.4f} "
          f"(paired bootstrap); GRU ahead in {(diff > 0).mean():.0%} of resamples.")
    diff = bootstrap(y, p_blend, p_lgb)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f"Blend minus LightGBM: {diff.mean():+.4f}, 95% CI {lo:+.4f} to {hi:+.4f}; "
          f"blend ahead in {(diff > 0).mean():.0%} of resamples.")

    # Where does reading the sequence help? Split by history length.
    L = lengths.numpy()
    rows = []
    for label, sel in [("All 13 months", L == 13), ("6 to 12 months", (L >= 6) & (L < 13)),
                       ("1 to 5 months", L < 6)]:
        g, b = amex_components(y[sel], p_gru[sel]), amex_components(y[sel], p_lgb[sel])
        rows.append({"History": label, "Customers": f"{sel.sum():,}",
                     "Default rate": f"{100 * y[sel].mean():.1f}%",
                     "LightGBM": f"{b['score']:.4f}", "GRU": f"{g['score']:.4f}",
                     "GRU - LightGBM": f"{g['score'] - b['score']:+.4f}"})
    print("\n## By history length\n")
    print(pd.DataFrame(rows).to_markdown(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", type=float, default=1.0, help="share of training customers to use")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="maximum epochs")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    meta = json.loads((SEQ / "meta.json").read_text())
    t0 = time.time()

    x, cat, lengths, y, _ = load("train")
    if args.sample < 1:
        keep = np.sort(rng.choice(len(y), int(len(y) * args.sample), replace=False))
        x, cat, lengths, y = x[keep], cat[keep], lengths[keep], y[keep]
    vx, vcat, vlen, vy, vids = load("valid")
    y_t = torch.from_numpy(y)
    print(f"Train {len(y):,} customers, validation {len(vy):,}; "
          f"{x.shape[1]} months x {x.shape[2]} inputs + {cat.shape[2]} categories "
          f"({time.time() - t0:.0f}s to load). {torch.get_num_threads()} CPU threads.\n")

    model = GRUModel(x.shape[2], meta["cardinality"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    loss_fn = nn.BCEWithLogitsLoss()
    print(f"GRU with {n_params:,} parameters.\n")
    print("| Epoch | Train loss | Valid score | Gini | Top 4% | LR | Minutes |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    MODELS.mkdir(parents=True, exist_ok=True)
    best, best_epoch, since_best = -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        model.train()
        order = torch.from_numpy(rng.permutation(len(y)))
        total = 0.0
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            loss = loss_fn(model(x[idx].float(), cat[idx], lengths[idx]), y_t[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)

        parts = amex_components(vy, predict(model, vx, vcat, vlen))
        lr = opt.param_groups[0]["lr"]
        sched.step(parts["score"])
        print(f"| {epoch} | {total / len(y):.4f} | {parts['score']:.4f} | {parts['gini']:.4f} | "
              f"{parts['top_four']:.4f} | {lr:.0e} | {(time.time() - t) / 60:.1f} |", flush=True)

        if parts["score"] > best:
            best, best_epoch, since_best = parts["score"], epoch, 0
            torch.save(model.state_dict(), MODELS / "gru.pt")
        else:
            since_best += 1
            if since_best >= PATIENCE:
                break

    print(f"\nBest epoch {best_epoch}: validation score {best:.4f}. "
          f"Training took {(time.time() - t0) / 60:.0f} minutes.")
    model.load_state_dict(torch.load(MODELS / "gru.pt"))
    p = predict(model, vx, vcat, vlen)

    if args.sample == 1:
        PREDS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ID: vids, TARGET: vy.astype(int), "pred": p}).to_parquet(PREDS / "gru_valid.parquet", index=False)
    compare(vids, vy.astype(int), p, vlen)


if __name__ == "__main__":
    main()
