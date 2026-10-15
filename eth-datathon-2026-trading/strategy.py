"""ETH Datathon 2026 - HRT trading challenge - team Tren-Test-Split.

For each trading session we see the first 50 price bars (OHLC) and the news
headlines published so far, and must choose a position held over the next 50
bars. The score is the Sharpe ratio of the resulting P&L across sessions.

Final strategy = blend of three "experts":
  1. volatility-normalised mean reversion (Parkinson volatility)
  2. FinBERT headline sentiment with exponential time decay
  3. residual MLP forecasting the unseen path from the seen OHLC sequence
The blend weights are tuned with Optuna to maximise the training Sharpe.

Run:  uv run strategy.py      ->  writes results/submission.csv
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import optuna

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Strategy D + sentiment
ALPHA_D      = 0.070
LAMBDA_DECAY = 0.175
LAST_BAR     = 49
CLIP_LO, CLIP_HI = -5, 10

# Shared FinBERT cache
CACHE_DIR = ROOT / "cache" / "finbert"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ResidualMLP hyperparameters
HIDDEN_DIM   = 128
N_BLOCKS     = 3
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 64
MAX_EPOCHS   = 300
PATIENCE     = 40
CV_FOLDS     = 5
SEED         = 42

N_TRIALS = 400

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
print(f"FinBERT cache: {CACHE_DIR}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 1. DATA LOADING
print("Loading datasets…")
df_train_bars   = pd.read_parquet(os.path.join(DATA_DIR, "bars_seen_train.parquet"))
df_train_unseen = pd.read_parquet(os.path.join(DATA_DIR, "bars_unseen_train.parquet"))
df_public_bars  = pd.read_parquet(os.path.join(DATA_DIR, "bars_seen_public_test.parquet"))
df_private_bars = pd.read_parquet(os.path.join(DATA_DIR, "bars_seen_private_test.parquet"))
df_test_bars    = pd.concat([df_public_bars, df_private_bars], ignore_index=True)

df_train_news  = pd.read_parquet(os.path.join(DATA_DIR, "headlines_seen_train.parquet"))
df_public_news = pd.read_parquet(os.path.join(DATA_DIR, "headlines_seen_public_test.parquet"))
df_private_news= pd.read_parquet(os.path.join(DATA_DIR, "headlines_seen_private_test.parquet"))
df_test_news   = pd.concat([df_public_news, df_private_news], ignore_index=True)


# 2. STRATEGY D BASE POSITIONS
def get_base_positions(df_bars, alpha=ALPHA_D):
    df = df_bars.sort_values(["session", "bar_ix"]).copy()
    df["log_hl_sq"] = np.log(df["high"] / df["low"]) ** 2
    g = df.groupby("session", sort=False)
    first_close = g["close"].first()
    last_close  = g["close"].last()
    parkinson_var = g["log_hl_sq"].mean() / (4.0 * np.log(2.0))
    sigma = np.sqrt(parkinson_var) + 1e-8
    z = (last_close.values / first_close.values - 1.0) / sigma.values
    return pd.Series(1.0 - alpha * z, index=first_close.index, name="base_position")


print("Computing Strategy D base positions…")
train_base = get_base_positions(df_train_bars)
test_base  = get_base_positions(df_test_bars)


# 3. FINBERT TIME-DECAY WEIGHTED SENTIMENT
# Lazy FinBERT: only loaded if at least one cache miss forces a compute.
_tokenizer = None
_fin_model = None


def _get_finbert():
    global _tokenizer, _fin_model
    if _tokenizer is None:
        print("Loading FinBERT…")
        _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _fin_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
        _fin_model.eval()
    return _tokenizer, _fin_model


def batch_sentiment(headlines, cache_name, batch_size=128):
    """FinBERT P(pos) − P(neg) per headline, cached on disk.

    Cache file: CACHE_DIR / f"sentiments_{cache_name}.npy"
    Shared across strategies — any script that calls with the same cache_name
    and the same headline list reuses the saved scores.
    """
    cache_path = CACHE_DIR / f"sentiments_{cache_name}.npy"
    if cache_path.exists():
        scores = np.load(cache_path)
        if len(scores) == len(headlines):
            print(f"  ← cache hit  {cache_path.name}  ({len(scores)} headlines)")
            return scores
        print(f"  ⚠ cache size mismatch ({len(scores)} vs {len(headlines)}); recomputing")

    tokenizer, model = _get_finbert()
    scores = []
    for i in tqdm(range(0, len(headlines), batch_size), desc=f"FinBERT[{cache_name}]", unit="batch"):
        batch = headlines[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        scores.extend((probs[:, 0] - probs[:, 1]).cpu().numpy())
    scores = np.asarray(scores, dtype=np.float32)
    np.save(cache_path, scores)
    print(f"  → cached     {cache_path.name}  ({len(scores)} headlines)")
    return scores


def compute_weighted_sentiment(df_news, cache_name):
    df = df_news.copy()
    df["sentiment"] = batch_sentiment(df["headline"].tolist(), cache_name=cache_name)
    df["w"] = np.exp(-LAMBDA_DECAY * (LAST_BAR - df["bar_ix"]))
    df["infl"] = df["sentiment"] * df["w"]
    return df.groupby("session")["infl"].sum().rename("weighted_sentiment")


print("Scoring train headlines…")
train_sent = compute_weighted_sentiment(df_train_news, cache_name="train")
print("Scoring test headlines…")
test_sent  = compute_weighted_sentiment(df_test_news, cache_name="test")

# Free FinBERT memory (if it was loaded) before training the DNN
if _fin_model is not None:
    del _fin_model
    _fin_model = None
torch.cuda.empty_cache()


# 4. DNN FEATURE EXTRACTION
def extract_sequences(df_bars):
    df = df_bars.sort_values(["session", "bar_ix"])
    return {
        int(s): g[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
        for s, g in df.groupby("session", sort=False)
    }


def prepare_train_dnn(seen, unseen):
    X, y, sess = [], [], []
    for s in sorted(seen):
        if s not in unseen:
            continue
        s_ohlc, u_ohlc = seen[s], unseen[s]
        if len(s_ohlc) != 50 or len(u_ohlc) != 50:
            continue
        anchor = s_ohlc[-1, 3]
        X.append(np.log(s_ohlc / anchor).flatten())
        y.append(np.log(u_ohlc[:, 3] / anchor))
        sess.append(s)
    return np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(sess)


def prepare_infer_dnn(seen):
    X, sess = [], []
    for s in sorted(seen):
        ohlc = seen[s]
        if len(ohlc) != 50:
            continue
        anchor = ohlc[-1, 3]
        X.append(np.log(ohlc / anchor).flatten())
        sess.append(s)
    return np.asarray(X, np.float32), np.asarray(sess)


print("Extracting DNN sequences…")
train_seen_seqs   = extract_sequences(df_train_bars)
train_unseen_seqs = extract_sequences(df_train_unseen)
test_seen_seqs    = extract_sequences(df_test_bars)

X_dnn_train, y_dnn_train, dnn_train_sess = prepare_train_dnn(train_seen_seqs, train_unseen_seqs)
X_dnn_test, dnn_test_sess = prepare_infer_dnn(test_seen_seqs)
print(f"  DNN train: X={X_dnn_train.shape}  y={y_dnn_train.shape}")
print(f"  DNN test:  X={X_dnn_test.shape}")


# 5. RESIDUAL-MLP MODEL
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, HIDDEN_DIM)
        self.blocks = nn.Sequential(*[ResidualBlock(HIDDEN_DIM, DROPOUT) for _ in range(N_BLOCKS)])
        self.head = nn.Linear(HIDDEN_DIM, out_dim)

    def forward(self, x):
        return self.head(self.blocks(self.proj(x)))


def train_dnn_one(X_tr, y_tr, X_vl, y_vl):
    model = ResidualMLP(X_tr.shape[1], y_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_vl_t = torch.from_numpy(X_vl).to(device)
    y_vl_t = torch.from_numpy(y_vl).to(device)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)

    best_vl, best_state, best_epoch, patience = float("inf"), None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl_loss = criterion(model(X_vl_t), y_vl_t).item()
        if vl_loss < best_vl:
            best_vl = vl_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, best_vl


def dnn_predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).to(device)).cpu().numpy()


# 6. DNN TRAINING: 5-fold OOF  +  final model
print(f"\n5-fold OOF ResidualMLP training (MAX_EPOCHS={MAX_EPOCHS})…")
set_seed(SEED)
oof_dnn = np.zeros_like(y_dnn_train)
best_epochs: list[int] = []
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_dnn_train), 1):
    model, be, bvl = train_dnn_one(
        X_dnn_train[tr_idx], y_dnn_train[tr_idx],
        X_dnn_train[vl_idx], y_dnn_train[vl_idx],
    )
    oof_dnn[vl_idx] = dnn_predict(model, X_dnn_train[vl_idx])
    best_epochs.append(be)
    print(f"  fold {fold}: best_epoch={be}  vl_mse={bvl:.5f}")

# Use the LAST-bar log-return prediction → convert to linear return
oof_dnn_ret       = np.exp(oof_dnn[:, -1]) - 1.0
oof_dnn_mean_tr   = float(oof_dnn_ret.mean())
oof_dnn_std_tr    = float(oof_dnn_ret.std() + 1e-12)
oof_dnn_centered  = oof_dnn_ret - oof_dnn_mean_tr

# Final DNN on all train data
final_epochs = max(int(np.round(np.mean(best_epochs))), 20)
print(f"\nFinal DNN on all {len(X_dnn_train)} samples for {final_epochs} epochs…")
set_seed(SEED)
final_model = ResidualMLP(X_dnn_train.shape[1], y_dnn_train.shape[1]).to(device)
opt = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()
X_all_t = torch.from_numpy(X_dnn_train).to(device)
y_all_t = torch.from_numpy(y_dnn_train).to(device)
loader = DataLoader(TensorDataset(X_all_t, y_all_t), batch_size=BATCH_SIZE, shuffle=True)
for _ in range(final_epochs):
    final_model.train()
    for xb, yb in loader:
        opt.zero_grad()
        criterion(final_model(xb), yb).backward()
        opt.step()

test_dnn_ret      = np.exp(dnn_predict(final_model, X_dnn_test)[:, -1]) - 1.0
test_dnn_centered = test_dnn_ret - oof_dnn_mean_tr    # center w/ train reference


# 7. ALIGN SIGNALS ON COMMON TRAIN SESSIONS
print("\nAligning train signals…")
dnn_train_set = set(dnn_train_sess.tolist())
common_idx = [s for s in sorted(dnn_train_set)
              if s in train_base.index and s in train_sent.index]
print(f"  common train sessions: {len(common_idx)}")

# Target returns from DNN labels (we already have them)
y_map = dict(zip(dnn_train_sess.tolist(), np.exp(y_dnn_train[:, -1]) - 1.0))
dnn_oof_map = dict(zip(dnn_train_sess.tolist(), oof_dnn_centered))

train_base_vals    = train_base.loc[common_idx].to_numpy(dtype=np.float64)
train_sent_vals    = train_sent.loc[common_idx].to_numpy(dtype=np.float64)
train_dnn_vals     = np.array([dnn_oof_map[s] for s in common_idx])
train_returns_vals = np.array([y_map[s] for s in common_idx])

# Rescale DNN centered signal so its std matches sentiment's → (β, γ) on same scale.
sent_std = float(train_sent_vals.std() + 1e-12)
dnn_std  = float(train_dnn_vals.std()  + 1e-12)
dnn_to_sent_scale = sent_std / dnn_std
train_dnn_scaled  = train_dnn_vals * dnn_to_sent_scale
print(f"  sent std = {sent_std:.4f}   dnn std = {dnn_std:.5f}   rescale ×{dnn_to_sent_scale:.2f}")


# 8. OPTUNA BLEND
print("\nRunning Optuna on (β, γ)…")


def objective(trial):
    beta  = trial.suggest_float("beta",  0.0, 1.0)   # sentiment weight
    gamma = trial.suggest_float("gamma", 0.0, 1.0)   # dnn weight
    positions = (1.0 - beta - gamma) * train_base_vals \
                + beta  * train_sent_vals            \
                + gamma * train_dnn_scaled
    positions = np.clip(positions, CLIP_LO, CLIP_HI)
    pnl = positions * train_returns_vals
    std = np.std(pnl)
    return (np.mean(pnl) / std) * 16.0 if std > 0 else 0.0


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
best_beta  = study.best_params["beta"]
best_gamma = study.best_params["gamma"]


def train_sharpe(positions):
    positions = np.clip(positions, CLIP_LO, CLIP_HI)
    pnl = positions * train_returns_vals
    s = np.std(pnl)
    return (np.mean(pnl) / s * 16.0) if s > 0 else 0.0


# Diagnostics: Sharpe at each corner for reference
al_s   = train_sharpe(np.ones_like(train_returns_vals))
base_s = train_sharpe(train_base_vals)
sent_s = train_sharpe(train_base_vals + 0.3 * train_sent_vals)
dnn_s  = train_sharpe(train_base_vals + 0.3 * train_dnn_scaled)
print(f"  always-long          : {al_s:.4f}")
print(f"  Strategy D alone     : {base_s:.4f}")
print(f"  D + 0.3·sent (ref)   : {sent_s:.4f}")
print(f"  D + 0.3·dnn  (ref)   : {dnn_s:.4f}")
print(f"  blend (β={best_beta:.3f}, γ={best_gamma:.3f}) : {study.best_value:.4f}")


# 9. ASSEMBLE TEST POSITIONS
print("\nAssembling test positions…")
test_sessions = sorted(df_test_bars["session"].unique())
test_dnn_map  = dict(zip(dnn_test_sess.tolist(), test_dnn_centered * dnn_to_sent_scale))

base_t = test_base.reindex(test_sessions).to_numpy(dtype=np.float64)
sent_t = test_sent.reindex(test_sessions).fillna(0.0).to_numpy(dtype=np.float64)
dnn_t  = np.array([test_dnn_map.get(int(s), 0.0) for s in test_sessions])

final_positions = (1.0 - best_beta - best_gamma) * base_t \
                  + best_beta  * sent_t                  \
                  + best_gamma * dnn_t
final_positions = np.clip(final_positions, CLIP_LO, CLIP_HI)

submission = pd.DataFrame({
    "session": test_sessions,
    "target_position": final_positions,
})
out_path = OUT_DIR / "submission.csv"
submission.to_csv(out_path, index=False)
print(f"\nSaved {out_path}  ({len(submission)} sessions)")
print(submission["target_position"].describe())
