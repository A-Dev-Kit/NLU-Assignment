"""Part 1 - Zipf's Law and Corpus Analysis (Q1-Q5).
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive; pyplot import must come after
import matplotlib.pyplot as plt

import nltk
import numpy as np
import pandas as pd

# The input data.
nltk.data.path.insert(0, str(Path(__file__).parent / "nltk_data"))


# ============================== Configuration ==============================

@dataclass(frozen=True)
class Config:
    corpus_name: str = "gutenberg"
    seed: int = 42
    lowercase: bool = True
    keep_only_alpha: bool = True
    ttr_checkpoints: Tuple[int, ...] = (10_000, 50_000, 100_000, 500_000, 1_000_000)
    mattr_window: int = 1_000
    mattr_stride: int = 500
    top_k: int = 20
    zipf_top_ranks: int = 10_000


CFG = Config()
REPO = Path(__file__).resolve().parent
ART = REPO / "artifacts" / "part1_zipf"
PLOTS = ART / "plots"
OUT = REPO / "outputs"


# ============================== Progress / IO ==============================

def _stage(title: str) -> None:
    print(f"--- {title} ---")


def _iteration(step: int, total: int | None, prefix: str, **metrics: Any) -> None:
    header = f"{prefix} {step}/{total}" if total is not None else f"{prefix} {step}"
    body = " ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in metrics.items()
    )
    print(f"{header} {body}".rstrip())


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================ Corpus utilities =============================

def load_tokens(corpus_name: str, lowercase: bool, keep_only_alpha: bool) -> List[str]:
    """Flatten the requested corpus to a single token stream with the given filters."""
    if corpus_name == "gutenberg":
        raw_sents = nltk.corpus.gutenberg.sents()
    elif corpus_name == "brown":
        raw_sents = nltk.corpus.brown.sents()
    else:
        raise ValueError(f"unknown corpus: {corpus_name!r}")
    tokens: List[str] = []
    for sent in raw_sents:
        for tok in sent:
            t = tok.lower() if lowercase else tok
            if keep_only_alpha and not any(c.isalpha() for c in t):
                continue
            tokens.append(t)
    return tokens


# =========================== Zipf, TTR, MATTR ==============================

def frequency_table(tokens: Sequence[str]) -> pd.DataFrame:
    """Rank-sorted (rank, word, frequency) DataFrame, rank starting at 1."""
    counts = Counter(tokens)
    ranked = counts.most_common()
    return pd.DataFrame({
        "rank": np.arange(1, len(ranked) + 1),
        "word": [w for w, _ in ranked],
        "frequency": [c for _, c in ranked],
    })


def zipf_fit(freq_df: pd.DataFrame, top_ranks: int | None) -> Tuple[float, float, float]:
    """Fit log(freq) = slope * log(rank) + intercept. Return (slope, intercept, R^2).

    If `top_ranks` is set, only the head is used: the massive hapax plateau at
    the tail would otherwise flatten the slope towards zero.
    """
    df = freq_df if top_ranks is None else freq_df.head(top_ranks)
    log_rank = np.log(df["rank"].to_numpy(dtype=float))
    log_freq = np.log(df["frequency"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(log_rank, log_freq, 1)
    pred = slope * log_rank + intercept
    ss_res = float(np.sum((log_freq - pred) ** 2))
    ss_tot = float(np.sum((log_freq - log_freq.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def ttr_at_sizes(tokens: Sequence[str], sizes: Iterable[int]) -> pd.DataFrame:
    """|V|/N at each prefix length in `sizes` (skipping sizes larger than the corpus)."""
    checkpoints = sorted(int(s) for s in sizes if int(s) <= len(tokens))
    _stage("ttr at fixed corpus sizes")
    rows: List[dict] = []
    for i, n in enumerate(checkpoints, start=1):
        types = len(set(tokens[:n]))
        ttr = types / n
        rows.append({"tokens": n, "types": types, "ttr": ttr})
        _iteration(i, len(checkpoints), "ttr", tokens=n, types=types, ttr=ttr)
    return pd.DataFrame(rows)


def mattr(tokens: Sequence[str], window: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Moving-average TTR over sliding windows. Returns (window_end_positions, values)."""
    if window <= 0:
        raise ValueError("window must be positive")
    if len(tokens) < window:
        return (np.array([len(tokens)]),
                np.array([len(set(tokens)) / max(1, len(tokens))]))
    positions: List[int] = []
    values: List[float] = []
    for start in range(0, len(tokens) - window + 1, stride):
        end = start + window
        positions.append(end)
        values.append(len(set(tokens[start:end])) / window)
    return np.asarray(positions), np.asarray(values)


def mattr_at_sizes(tokens: Sequence[str], sizes: Iterable[int], window: int,
                   stride: int) -> pd.DataFrame:
    """Mean MATTR across the prefix of length n, for each n in `sizes`."""
    checkpoints = sorted(int(s) for s in sizes if int(s) <= len(tokens))
    rows: List[dict] = []
    for i, n in enumerate(checkpoints, start=1):
        _, vals = mattr(tokens[:n], window=window, stride=stride)
        mean = float(vals.mean()) if vals.size else float("nan")
        rows.append({"tokens": n, "window": window, "mattr": mean})
        _iteration(i, len(checkpoints), "mattr", tokens=n, window=window, mattr=mean)
    return pd.DataFrame(rows)


# ============================= Section runners =============================

def _banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def q1_top_10_words(freq_df: pd.DataFrame) -> None:
    _banner("Q1 - Top-10 most frequent word types")
    top10 = freq_df.head(10).reset_index(drop=True)
    save_dataframe(top10, ART / "top_10_words.csv")
    print(top10.to_string(index=False))


def q2_zipf_fit(freq_df: pd.DataFrame) -> None:
    _banner("Q2 - Zipf's Law: log-log fit")
    _stage("zipf log-log fit")
    slope, intercept, r2 = zipf_fit(freq_df, top_ranks=CFG.zipf_top_ranks)
    print(f"zipf slope={slope:.4f} intercept={intercept:.4f} "
          f"r2={r2:.4f} top_ranks={CFG.zipf_top_ranks}")

    save_json({
        "slope": slope,
        "intercept": intercept,
        "r_squared": r2,
        "top_ranks_used_for_fit": CFG.zipf_top_ranks,
        "types_total": int(len(freq_df)),
        "tokens_total": int(freq_df["frequency"].sum()),
    }, ART / "zipf_fit.json")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(np.log(freq_df["rank"]), np.log(freq_df["frequency"]),
               s=3, alpha=0.4, label="observed")
    x = np.log(freq_df["rank"].to_numpy(dtype=float))
    ax.plot(x, slope * x + intercept, color="red", linewidth=1.5,
            label=f"fit: slope={slope:.3f}, intercept={intercept:.3f}")
    ax.set_xlabel("log(rank)")
    ax.set_ylabel("log(frequency)")
    ax.set_title(f"{CFG.corpus_name.capitalize()} corpus - Zipf's Law")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "zipf_log_log.png", dpi=140)
    plt.close(fig)

    print(f"slope     = {slope:.4f}")
    print(f"intercept = {intercept:.4f}")
    print(f"R^2       = {r2:.4f}")
    print(f"plot      -> {PLOTS / 'zipf_log_log.png'}")


def q3_top_bottom_types(freq_df: pd.DataFrame) -> None:
    _banner("Q3 - Top-20 and bottom-20 word types")
    _stage("top / bottom types")
    top20 = freq_df.head(CFG.top_k).reset_index(drop=True)
    bottom20 = (
        freq_df.tail(CFG.top_k)
               .sort_values(["frequency", "word"])
               .reset_index(drop=True)
    )
    save_dataframe(top20, ART / "top_20_words.csv")
    save_dataframe(bottom20, ART / "bottom_20_words.csv")

    hapax = int((freq_df["frequency"] == 1).sum())
    print(f"top20 max_freq={int(top20['frequency'].max()):,}")
    print(f"bottom20 min_freq={int(bottom20['frequency'].min())} "
          f"hapax_count={hapax:,}")

    side_by_side = pd.concat({"top": top20, "bottom": bottom20}, axis=1)
    print(side_by_side.to_string())


def q4_ttr(tokens: Sequence[str]) -> pd.DataFrame:
    _banner("Q4 - Type-Token Ratio at 10K / 50K / 100K / 500K / 1M tokens")
    ttr_df = ttr_at_sizes(tokens, CFG.ttr_checkpoints)
    save_dataframe(ttr_df, ART / "ttr_table.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ttr_df["tokens"], ttr_df["ttr"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("corpus size (tokens, log scale)")
    ax.set_ylabel("TTR = |V| / N")
    ax.set_title("Type-Token Ratio vs corpus size")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "ttr_vs_size.png", dpi=140)
    plt.close(fig)

    print(ttr_df.to_string(index=False))
    return ttr_df


def q5_mattr(tokens: Sequence[str], ttr_df: pd.DataFrame) -> None:
    _banner("Q5 - Length-controlled TTR: Moving-Average TTR (MATTR)")
    mattr_df = mattr_at_sizes(tokens, CFG.ttr_checkpoints,
                              window=CFG.mattr_window, stride=CFG.mattr_stride)
    save_dataframe(mattr_df, ART / "mattr_table.csv")

    positions_full, values_full = mattr(tokens, window=CFG.mattr_window,
                                        stride=CFG.mattr_stride)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(ttr_df["tokens"], ttr_df["ttr"], marker="o", label="TTR")
    axes[0].plot(mattr_df["tokens"], mattr_df["mattr"], marker="s",
                 label=f"MATTR (W={CFG.mattr_window})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("corpus size (tokens, log scale)")
    axes[0].set_ylabel("ratio")
    axes[0].set_title("TTR vs MATTR at checkpoints")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(positions_full, values_full, alpha=0.6)
    axes[1].set_xlabel("window end position")
    axes[1].set_ylabel(f"MATTR (W={CFG.mattr_window})")
    axes[1].set_title("MATTR trace across the corpus")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS / "mattr_vs_size.png", dpi=140)
    plt.close(fig)

    print(mattr_df.to_string(index=False))


# =================================== Main ==================================

_OUTPUT_MIRROR: Tuple[Tuple[str, str], ...] = (
    ("top_10_words.csv",   "part1_top10.csv"),
    ("top_20_words.csv",   "part1_top20.csv"),
    ("bottom_20_words.csv", "part1_bottom20.csv"),
    ("ttr_table.csv",      "part1_ttr.csv"),
    ("mattr_table.csv",    "part1_mattr.csv"),
    ("zipf_fit.json",      "part1_zipf_fit.json"),
)


def _mirror_outputs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in _OUTPUT_MIRROR:
        shutil.copyfile(ART / src_name, OUT / dst_name)


def main() -> None:
    for d in (ART, PLOTS, OUT):
        d.mkdir(parents=True, exist_ok=True)

    print(f"config={CFG}")

    _stage("load corpus")
    tokens = load_tokens(CFG.corpus_name, CFG.lowercase, CFG.keep_only_alpha)
    print(f"corpus={CFG.corpus_name} tokens={len(tokens):,}")

    _stage("frequency table")
    freq_df = frequency_table(tokens)
    print(f"types={len(freq_df):,} total_tokens={int(freq_df['frequency'].sum()):,}")

    q1_top_10_words(freq_df)
    q2_zipf_fit(freq_df)
    q3_top_bottom_types(freq_df)
    ttr_df = q4_ttr(tokens)
    q5_mattr(tokens, ttr_df)

    _mirror_outputs()
    print("part 1 done - artifacts written to artifacts/part1_zipf and outputs/")


if __name__ == "__main__":
    main()
