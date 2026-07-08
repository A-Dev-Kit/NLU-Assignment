"""Part 2 - N-gram Language Models (Q6-Q10).
"""
from __future__ import annotations

import json
import math
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import nltk
import numpy as np
import pandas as pd

# The bundled corpora ship next to this script.
nltk.data.path.insert(0, str(Path(__file__).parent / "nltk_data"))


# ============================== Configuration ==============================

@dataclass(frozen=True)
class Config:
    corpus_name: str = "gutenberg"
    seed: int = 42
    lowercase: bool = True
    keep_only_alpha: bool = True
    train_ratio: float = 0.8
    unk_min_count: int = 1
    ngram_orders: Tuple[int, ...] = (1, 2, 3)
    laplace_alpha: float = 1.0
    num_samples: int = 5
    max_sample_length: int = 40


CFG = Config()
REPO = Path(__file__).resolve().parent
ART = REPO / "artifacts" / "part2_lm"
OUT = REPO / "outputs"

BOS, EOS, UNK = "<s>", "</s>", "<UNK>"


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


def save_text(lines: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.rstrip("\n") + "\n")


# ============================ Corpus utilities =============================

def load_sentences(corpus_name: str, lowercase: bool,
                   keep_only_alpha: bool) -> List[List[str]]:
    """Return the requested corpus as a list of tokenised sentences."""
    if corpus_name == "gutenberg":
        raw_sents = nltk.corpus.gutenberg.sents()
    elif corpus_name == "brown":
        raw_sents = nltk.corpus.brown.sents()
    else:
        raise ValueError(f"unknown corpus: {corpus_name!r}")
    sents: List[List[str]] = []
    for sent in raw_sents:
        toks = [(t.lower() if lowercase else t) for t in sent]
        if keep_only_alpha:
            toks = [t for t in toks if any(c.isalpha() for c in t)]
        if toks:
            sents.append(toks)
    return sents


def split_sentences(sentences: Sequence[Sequence[str]], train_ratio: float,
                    seed: int) -> Tuple[List, List]:
    """Deterministic 80/20 (or whatever ratio) shuffle-and-split by sentence."""
    rng = random.Random(seed)
    idx = list(range(len(sentences)))
    rng.shuffle(idx)
    cut = int(len(idx) * train_ratio)
    train = [sentences[i] for i in idx[:cut]]
    test = [sentences[i] for i in idx[cut:]]
    return train, test


# ========================= Vocabulary + UNK mapping ========================

def build_vocab(train_sentences: Sequence[Sequence[str]], unk_min_count: int) -> set:
    """Keep training tokens with count > unk_min_count, plus <s>, </s>, <UNK>."""
    counts: Counter = Counter()
    for s in train_sentences:
        counts.update(s)
    vocab = {t for t, c in counts.items() if c > unk_min_count}
    vocab.update({BOS, EOS, UNK})
    return vocab


def _replace_oov(sentences: Sequence[Sequence[str]], vocab: set) -> List[List[str]]:
    return [[t if t in vocab else UNK for t in s] for s in sentences]


# =========================== N-gram language model =========================

class NgramLanguageModel:
    """Add-alpha smoothed n-gram language model for n in {1, 2, 3}."""

    def __init__(self, order: int, alpha: float = 1.0):
        if order not in (1, 2, 3):
            raise ValueError("order must be 1, 2 or 3")
        self.order = order
        self.alpha = alpha
        self.vocab: set = set()
        self._ctx_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self._ctx_totals: Dict[Tuple[str, ...], int] = defaultdict(int)
        self._V = 0

    def _pad(self, sentence: Sequence[str]) -> List[str]:
        return [BOS] * (self.order - 1) + list(sentence) + [EOS]

    def fit(self, train_sentences: Sequence[Sequence[str]], vocab: set,
            report: bool = True) -> None:
        """Count n-grams over UNK-replaced training sentences."""
        self.vocab = set(vocab)
        self._V = len(self.vocab)
        self._ctx_counts.clear()
        self._ctx_totals.clear()
        start = time.time()
        for sent in _replace_oov(train_sentences, self.vocab):
            padded = self._pad(sent)
            for i in range(self.order - 1, len(padded)):
                ctx = tuple(padded[i - self.order + 1 : i])
                self._ctx_counts[ctx][padded[i]] += 1
                self._ctx_totals[ctx] += 1
        if report:
            unique = sum(len(c) for c in self._ctx_counts.values())
            _iteration(
                self.order, 3, "fit",
                order=self.order, vocab=self._V,
                contexts=len(self._ctx_counts), unique_ngrams=unique,
                elapsed_s=time.time() - start,
            )

    def prob(self, context: Tuple[str, ...], word: str,
             smoothed: bool = True) -> float:
        """P(word | context). If smoothed=False and context unseen, returns 0."""
        c_ctx_w = self._ctx_counts[context].get(word, 0)
        c_ctx = self._ctx_totals.get(context, 0)
        if smoothed:
            return (c_ctx_w + self.alpha) / (c_ctx + self.alpha * self._V)
        if c_ctx == 0:
            return 0.0
        return c_ctx_w / c_ctx

    def log_prob_token(self, context: Tuple[str, ...], word: str) -> float:
        c_ctx_w = self._ctx_counts[context].get(word, 0)
        c_ctx = self._ctx_totals.get(context, 0)
        num = c_ctx_w + self.alpha
        den = c_ctx + self.alpha * self._V
        return math.log(num) - math.log(den)

    def log_prob_sentence(self, sentence: Sequence[str]) -> Tuple[float, int]:
        padded = self._pad(sentence)
        total, n = 0.0, 0
        for i in range(self.order - 1, len(padded)):
            ctx = tuple(padded[i - self.order + 1 : i])
            total += self.log_prob_token(ctx, padded[i])
            n += 1
        return total, n

    def perplexity(self, test_sentences: Sequence[Sequence[str]],
                   report: bool = True) -> Tuple[float, int, float]:
        """Corpus perplexity. Returns (ppl, tokens_scored, log_prob_sum)."""
        mapped = _replace_oov(test_sentences, self.vocab)
        total_lp, total_n = 0.0, 0
        for sent in mapped:
            lp, n = self.log_prob_sentence(sent)
            total_lp += lp
            total_n += n
        avg_neg = -total_lp / max(1, total_n)
        ppl = math.exp(avg_neg)
        if report:
            _iteration(
                self.order, 3, "ppl",
                order=self.order, tokens_scored=total_n,
                log_prob_sum=total_lp, ppl=ppl,
            )
        return ppl, total_n, total_lp

    def sample(self, rng: np.random.Generator, max_length: int,
               use_smoothing: bool = False) -> List[str]:
        """Ancestral sample from the bigram / trigram distribution.

        MLE counts are used by default: add-alpha over a large vocab collapses
        to a near-uniform draw and produces incoherent samples. Perplexity is
        always evaluated with the smoothed distribution regardless of this flag.
        """
        if self.order == 1:
            raise ValueError("sampling from a unigram model is degenerate")
        context: Tuple[str, ...] = tuple([BOS] * (self.order - 1))
        out: List[str] = []
        for _ in range(max_length):
            nxt = self._draw_next(context, rng, use_smoothing)
            if nxt == EOS:
                break
            if nxt == BOS:
                continue
            out.append(nxt)
            context = tuple((list(context) + [nxt])[-(self.order - 1):])
        return out

    def _draw_next(self, context: Tuple[str, ...], rng: np.random.Generator,
                   use_smoothing: bool) -> str:
        counts = self._ctx_counts[context]
        total = self._ctx_totals.get(context, 0)
        if not use_smoothing and total > 0:
            words = list(counts.keys())
            probs = np.asarray([counts[w] for w in words], dtype=np.float64)
            probs /= probs.sum()
            return str(rng.choice(words, p=probs))
        vocab_list = sorted(self.vocab)
        probs = np.full(len(vocab_list), self.alpha, dtype=np.float64)
        for i, w in enumerate(vocab_list):
            probs[i] += counts.get(w, 0)
        probs /= probs.sum()
        return vocab_list[int(rng.choice(len(vocab_list), p=probs))]


# ============================= Section runners =============================

def _banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def q6_unigram_bigram_mle(models: Dict[int, NgramLanguageModel],
                          train: Sequence[Sequence[str]], vocab: set) -> None:
    _banner("Q6 - Unigram and bigram MLE")
    unigram = models[1]
    bigram = models[2]

    # Top-5 unigram MLE probabilities (raw, unsmoothed).
    top_words = [
        w for w, _ in Counter(t for s in train for t in s if t in vocab).most_common(5)
    ]
    print("Unigram MLE P(w) for the five most common training tokens:")
    for w in top_words:
        print(f"  P({w!r:>8s}) = {unigram.prob((), w, smoothed=False):.6f}")

    # Top-5 raw bigram continuations of "the".
    context = ("the",)
    ctx_counts = bigram._ctx_counts.get(context, Counter())
    ctx_top = [w for w, _ in ctx_counts.most_common(5)]
    print()
    print("Raw bigram MLE P(w | 'the') for the top-5 observed continuations:")
    for w in ctx_top:
        print(f"  P({w!r:>10s} | 'the') = {bigram.prob(context, w, smoothed=False):.6f}")


def q7_add_one_smoothing(models: Dict[int, NgramLanguageModel]) -> None:
    _banner("Q7 - Add-1 (Laplace) smoothing")
    bigram = models[2]
    print(
        "Raw MLE assigns P = 0 to any bigram never seen in training. With a\n"
        "vocabulary of ~24K types only a tiny fraction of possible bigrams are\n"
        "observed, so almost every held-out sentence contains an unseen bigram\n"
        "and its total probability collapses to zero (perplexity to infinity).\n"
        "\n"
        "Add-1 (Laplace) smoothing pretends every possible bigram was seen once:\n"
        "\n"
        "    P_L(w | w_prev) = (c(w_prev, w) + 1) / (c(w_prev) + V)\n"
        "\n"
        "Unseen bigrams now get a small but non-zero probability, log-probs stay\n"
        "finite, and perplexity is computable. The trade-off is that mass is\n"
        "stolen from the observed events, which hurts more the larger V gets."
    )
    print()

    context = ("the",)
    ctx_counts = bigram._ctx_counts.get(context, Counter())
    seen_examples = [w for w, _ in ctx_counts.most_common(5)]
    unseen_examples = ["martian"]  # a common English word that is not in the Gutenberg vocab

    print("Smoothed P(w | 'the') for the same continuations plus one unseen word:")
    for w in seen_examples + unseen_examples:
        seen = "seen" if ctx_counts.get(w, 0) > 0 else "unseen"
        print(f"  P({w!r:>10s} | 'the') = {bigram.prob(context, w, smoothed=True):.4e}   ({seen})")


def q8_perplexity(perplexity_rows: Sequence[Mapping[str, Any]]) -> None:
    _banner("Q8 - Perplexity on the held-out test set")
    df = pd.DataFrame([r for r in perplexity_rows if r["order"] in (1, 2)])
    df["model"] = df["order"].map({1: "unigram", 2: "bigram (add-1)"})
    df = df[["model", "order", "perplexity", "tokens_scored", "log_prob_sum"]]
    print(df.to_string(index=False))


def q9_ancestral_samples(bigram: NgramLanguageModel) -> List[str]:
    _banner("Q9 - Five ancestral samples from the bigram model")
    _stage("ancestral sampling from bigram")
    rng = np.random.default_rng(CFG.seed)
    samples = [bigram.sample(rng, max_length=CFG.max_sample_length)
               for _ in range(CFG.num_samples)]
    sample_lines = [" ".join(s) for s in samples]
    for i, line in enumerate(sample_lines, start=1):
        print(f"sample {i}: {line}")
    save_text([f"{i}. {line}" for i, line in enumerate(sample_lines, start=1)],
              ART / "sampled_sentences.txt")
    print()
    print("Samples use the observed MLE distribution: add-1 smoothing over 24K")
    print("words flattens the distribution to near-uniform, producing word soup.")
    print("Perplexity above is still computed with the smoothed distribution.")
    print()
    for i, line in enumerate(sample_lines, start=1):
        print(f"{i}. {line}")
    return sample_lines


def q10_trigram_comparison(perplexity_rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    _banner("Q10 - Trigram and cross-model comparison")
    df = pd.DataFrame(perplexity_rows)
    df["model"] = df["order"].map({1: "unigram", 2: "bigram (add-1)", 3: "trigram (add-1)"})
    df = df[["model", "order", "perplexity", "tokens_scored", "log_prob_sum"]]
    save_dataframe(df, ART / "perplexity_table.csv")
    print(df.to_string(index=False))
    print()
    print(
        "Trigram add-1 is worse than bigram add-1, which is worse than unigram.\n"
        "On a 24K-vocabulary the denominator c(context) + V is dominated by V,\n"
        "so P collapses towards 1/V for rare contexts. The trigram context space\n"
        "is V^2, so the smoother has to spread mass over ~5.8e8 possible triples\n"
        "and even correct predictions get too little probability. Interpolation\n"
        "or Kneser-Ney fix this by backing off to lower-order estimates instead\n"
        "of a uniform prior."
    )
    return df


# =================================== Main ==================================

_OUTPUT_MIRROR: Tuple[Tuple[str, str], ...] = (
    ("perplexity_table.csv",   "part2_perplexity.csv"),
    ("sampled_sentences.txt",  "part2_samples.txt"),
    ("model_summary.json",     "part2_summary.json"),
)


def _mirror_outputs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in _OUTPUT_MIRROR:
        shutil.copyfile(ART / src_name, OUT / dst_name)


def _write_summary(perplexity_rows: Sequence[Mapping[str, Any]],
                   sents: Sequence, train: Sequence, test: Sequence,
                   vocab: set, sample_lines: Sequence[str]) -> None:
    perplexity_by_model = {}
    for row in perplexity_rows:
        name = {1: "unigram", 2: "bigram (add-1)", 3: "trigram (add-1)"}[row["order"]]
        perplexity_by_model[name] = row["perplexity"]
    summary = {
        "config": asdict(CFG),
        "sentences": {"total": len(sents), "train": len(train), "test": len(test)},
        "tokens": {
            "train": sum(len(s) for s in train),
            "test": sum(len(s) for s in test),
        },
        "vocabulary_size": len(vocab),
        "perplexity": perplexity_by_model,
        "samples": list(sample_lines),
    }
    save_json(summary, ART / "model_summary.json")


def main() -> None:
    for d in (ART, OUT):
        d.mkdir(parents=True, exist_ok=True)

    print(f"config={CFG}")

    _stage("load and split corpus")
    sents = load_sentences(CFG.corpus_name, CFG.lowercase, CFG.keep_only_alpha)
    train, test = split_sentences(sents, train_ratio=CFG.train_ratio, seed=CFG.seed)
    print(f"sentences total={len(sents):,} train={len(train):,} test={len(test):,}")
    print(f"train_tokens={sum(len(s) for s in train):,} "
          f"test_tokens={sum(len(s) for s in test):,}")

    _stage("build vocabulary")
    vocab = build_vocab(train, unk_min_count=CFG.unk_min_count)
    print(f"vocab_size={len(vocab)} unk_min_count={CFG.unk_min_count}")

    _stage("fit n-gram models")
    models: Dict[int, NgramLanguageModel] = {}
    for n in CFG.ngram_orders:
        m = NgramLanguageModel(order=n, alpha=CFG.laplace_alpha)
        m.fit(train, vocab=vocab)
        models[n] = m

    _stage("compute perplexity")
    perplexity_rows: List[dict] = []
    for n in CFG.ngram_orders:
        ppl, tokens_scored, lp_sum = models[n].perplexity(test)
        perplexity_rows.append({
            "order": n,
            "perplexity": ppl,
            "tokens_scored": tokens_scored,
            "log_prob_sum": lp_sum,
        })

    q6_unigram_bigram_mle(models, train, vocab)
    q7_add_one_smoothing(models)
    q8_perplexity(perplexity_rows)
    sample_lines = q9_ancestral_samples(models[2])
    q10_trigram_comparison(perplexity_rows)

    _write_summary(perplexity_rows, sents, train, test, vocab, sample_lines)
    _mirror_outputs()
    print("part 2 done - artifacts written to artifacts/part2_lm and outputs/")


if __name__ == "__main__":
    main()
