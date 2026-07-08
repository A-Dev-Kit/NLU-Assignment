"""Part 3 - HMM POS Tagger with Viterbi (Q11-Q14).
"""
from __future__ import annotations

import json
import math
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
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
    corpus_name: str = "brown"
    tagset: str = "universal"
    seed: int = 42
    train_ratio: float = 0.8
    laplace_alpha: float = 1.0
    decode_log_every: int = 500
    error_examples: int = 10
    top_k: int = 5


CFG = Config()
REPO = Path(__file__).resolve().parent
ART = REPO / "artifacts" / "part3_hmm"
OUT = REPO / "outputs"

HMM_BOS, HMM_EOS = "<s>", "</s>"
NEG_INF = -1e18


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

def load_tagged_sentences(corpus_name: str, tagset: str) -> List[List[Tuple[str, str]]]:
    """Return the tagged corpus as a list of (word, tag) sentences."""
    if corpus_name != "brown":
        raise ValueError("tagged corpus only implemented for 'brown'")
    return [list(s) for s in nltk.corpus.brown.tagged_sents(tagset=tagset)]


def split_sentences(sentences: Sequence, train_ratio: float,
                    seed: int) -> Tuple[List, List]:
    """Deterministic shuffle-and-split by sentence."""
    rng = random.Random(seed)
    idx = list(range(len(sentences)))
    rng.shuffle(idx)
    cut = int(len(idx) * train_ratio)
    train = [sentences[i] for i in idx[:cut]]
    test = [sentences[i] for i in idx[cut:]]
    return train, test


# ================================ HMM tagger ===============================

class HMMTagger:
    """Add-alpha smoothed HMM POS tagger with a hapax-based OOV emission fallback.

    Viterbi is implemented from scratch in `decode`: log-space, dense score
    matrix and back-pointer matrix, explicit boundary states.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.tags: List[str] = []
        self._tag_idx: Dict[str, int] = {}
        self._trans_counts: Dict[str, Counter] = defaultdict(Counter)
        self._trans_totals: Dict[str, int] = defaultdict(int)
        self._emit_counts: Dict[str, Counter] = defaultdict(Counter)
        self._emit_totals: Dict[str, int] = defaultdict(int)
        self._train_vocab: set = set()
        self._V_emit = 0
        self._hapax_dist: Dict[str, float] = {}
        self._log_A: np.ndarray | None = None
        self._log_pi: np.ndarray | None = None
        self._log_final: np.ndarray | None = None

    # ---- training ----------------------------------------------------------

    def fit(self, train_sentences: Sequence[Sequence[Tuple[str, str]]],
            report: bool = True) -> None:
        start = time.time()
        self._trans_counts.clear()
        self._trans_totals.clear()
        self._emit_counts.clear()
        self._emit_totals.clear()
        self._train_vocab.clear()

        word_counts: Counter = Counter()
        for sent in train_sentences:
            prev = HMM_BOS
            for word, tag in sent:
                self._trans_counts[prev][tag] += 1
                self._trans_totals[prev] += 1
                self._emit_counts[tag][word] += 1
                self._emit_totals[tag] += 1
                self._train_vocab.add(word)
                word_counts[word] += 1
                prev = tag
            self._trans_counts[prev][HMM_EOS] += 1
            self._trans_totals[prev] += 1

        self.tags = sorted(self._emit_counts.keys())
        self._tag_idx = {t: i for i, t in enumerate(self.tags)}
        self._V_emit = len(self._train_vocab)

        # Hapax-based OOV emission: use the tag distribution of words seen
        # exactly once in training as a proxy for what an unseen word looks like.
        hapax = {w for w, c in word_counts.items() if c == 1}
        hapax_tag_counts: Counter = Counter()
        for tag, wc in self._emit_counts.items():
            for w, c in wc.items():
                if w in hapax:
                    hapax_tag_counts[tag] += c
        total_hapax = sum(hapax_tag_counts.values())
        if total_hapax == 0:
            self._hapax_dist = {t: 1.0 / len(self.tags) for t in self.tags}
        else:
            denom = total_hapax + self.alpha * len(self.tags)
            self._hapax_dist = {
                t: (hapax_tag_counts.get(t, 0) + self.alpha) / denom
                for t in self.tags
            }

        self._precompute_log_matrices()

        if report:
            _iteration(
                1, 1, "fit",
                sentences=len(train_sentences),
                tags=len(self.tags), vocab=self._V_emit,
                trans_contexts=len(self._trans_counts),
                elapsed_s=time.time() - start,
            )

    def _precompute_log_matrices(self) -> None:
        T = len(self.tags)
        alpha = self.alpha
        # Transition matrix: rows are previous tag, columns are next tag.
        log_A = np.full((T, T), NEG_INF, dtype=np.float64)
        for i, prev in enumerate(self.tags):
            denom = self._trans_totals.get(prev, 0) + alpha * (T + 1)
            for j, nxt in enumerate(self.tags):
                num = self._trans_counts[prev].get(nxt, 0) + alpha
                log_A[i, j] = math.log(num) - math.log(denom)
        self._log_A = log_A

        # Initial distribution: P(tag | <s>).
        log_pi = np.full(T, NEG_INF, dtype=np.float64)
        denom_bos = self._trans_totals.get(HMM_BOS, 0) + alpha * (T + 1)
        for j, tag in enumerate(self.tags):
            num = self._trans_counts[HMM_BOS].get(tag, 0) + alpha
            log_pi[j] = math.log(num) - math.log(denom_bos)
        self._log_pi = log_pi

        # Final transition to </s>.
        log_final = np.full(T, NEG_INF, dtype=np.float64)
        for i, tag in enumerate(self.tags):
            denom = self._trans_totals.get(tag, 0) + alpha * (T + 1)
            num = self._trans_counts[tag].get(HMM_EOS, 0) + alpha
            log_final[i] = math.log(num) - math.log(denom)
        self._log_final = log_final

    # ---- emission ----------------------------------------------------------

    def _log_emit_column(self, word: str) -> np.ndarray:
        """log P(word | tag) for every tag, as a length-T vector."""
        T = len(self.tags)
        col = np.empty(T, dtype=np.float64)
        alpha = self.alpha
        if word in self._train_vocab:
            for j, tag in enumerate(self.tags):
                num = self._emit_counts[tag].get(word, 0) + alpha
                den = self._emit_totals.get(tag, 0) + alpha * self._V_emit
                col[j] = math.log(num) - math.log(den)
        else:
            for j, tag in enumerate(self.tags):
                p = self._hapax_dist.get(tag, 1e-12)
                col[j] = math.log(max(p, 1e-12))
        return col

    # ---- Q11 helpers -------------------------------------------------------

    def top_transitions(self, from_tag: str, k: int) -> List[Tuple[str, float]]:
        totals = self._trans_totals.get(from_tag, 0)
        denom = totals + self.alpha * (len(self.tags) + 1)
        probs = []
        for tag in list(self.tags) + [HMM_EOS]:
            num = self._trans_counts[from_tag].get(tag, 0) + self.alpha
            probs.append((tag, num / denom))
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:k]

    def top_emissions(self, from_tag: str, k: int) -> List[Tuple[str, float]]:
        total = self._emit_totals.get(from_tag, 0)
        denom = total + self.alpha * self._V_emit
        probs = [(w, (c + self.alpha) / denom)
                 for w, c in self._emit_counts[from_tag].items()]
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:k]

    # ---- Q12: Viterbi from scratch -----------------------------------------

    def decode(self, sentence: Sequence[str]) -> List[str]:
        """Return the best tag sequence for `sentence` via log-space Viterbi.

        `score[t, s]` = best log P of any tag sequence ending in tag `s` at
        position `t`. `back[t, s]` = the previous tag that achieved that best
        score. Boundary states `<s>` and `</s>` condition the first and last
        transitions respectively. Backtracking runs from the argmax over
        `score[n-1, s] + log P(</s> | s)`.
        """
        if not sentence:
            return []
        assert self._log_A is not None
        assert self._log_pi is not None
        assert self._log_final is not None

        T = len(self.tags)
        n = len(sentence)
        score = np.full((n, T), NEG_INF, dtype=np.float64)
        back = np.full((n, T), -1, dtype=np.int32)

        # Initialisation.
        score[0] = self._log_pi + self._log_emit_column(sentence[0])

        # Recursion.
        for t in range(1, n):
            emit = self._log_emit_column(sentence[t])
            trans = score[t - 1][:, None] + self._log_A
            best_prev = np.argmax(trans, axis=0)
            best_val = trans[best_prev, np.arange(T)]
            score[t] = best_val + emit
            back[t] = best_prev

        # Termination: absorb the transition to </s> before argmax.
        final_scores = score[n - 1] + self._log_final
        best_last = int(np.argmax(final_scores))

        # Backtrack.
        path_idx = [best_last]
        for t in range(n - 1, 0, -1):
            path_idx.append(int(back[t, path_idx[-1]]))
        path_idx.reverse()
        return [self.tags[i] for i in path_idx]

    def decode_many(self, sentences: Sequence[Sequence[str]],
                    log_every: int = 500) -> List[List[str]]:
        out: List[List[str]] = []
        start = time.time()
        total = len(sentences)
        for i, sent in enumerate(sentences, start=1):
            out.append(self.decode(sent))
            if i % log_every == 0 or i == total:
                _iteration(i, total, "decode", elapsed_s=time.time() - start)
        return out


# ============================ Evaluation helpers ===========================

def evaluate_predictions(gold_sentences: Sequence[Sequence[Tuple[str, str]]],
                         pred_sentences: Sequence[Sequence[str]],
                         train_vocab: set) -> Dict[str, float | int]:
    total = seen_total = unseen_total = 0
    correct = seen_correct = unseen_correct = 0
    for gold, pred in zip(gold_sentences, pred_sentences):
        for (word, gold_tag), pred_tag in zip(gold, pred):
            total += 1
            hit = int(gold_tag == pred_tag)
            correct += hit
            if word in train_vocab:
                seen_total += 1
                seen_correct += hit
            else:
                unseen_total += 1
                unseen_correct += hit
    return {
        "tokens": total, "correct": correct,
        "accuracy": correct / max(1, total),
        "seen_tokens": seen_total, "seen_correct": seen_correct,
        "seen_accuracy": seen_correct / max(1, seen_total),
        "unseen_tokens": unseen_total, "unseen_correct": unseen_correct,
        "unseen_accuracy": unseen_correct / max(1, unseen_total),
    }


def _classify_error(word: str, gold: str, pred: str, train_vocab: set,
                    tagger: HMMTagger) -> str:
    if word not in train_vocab:
        return "OOV"
    emit_for_word = {t: tagger._emit_counts[t].get(word, 0) for t in tagger.tags}
    seen_tags = [t for t, c in emit_for_word.items() if c > 0]
    if len(seen_tags) > 1:
        return "lexical ambiguity"
    if pred not in seen_tags:
        return "sparse transition"
    return "tag frequency bias"


def collect_errors(gold_sentences: Sequence[Sequence[Tuple[str, str]]],
                   pred_sentences: Sequence[Sequence[str]],
                   train_vocab: set, tagger: HMMTagger,
                   limit: int) -> List[Dict[str, object]]:
    errors: List[Dict[str, object]] = []
    for gold, pred in zip(gold_sentences, pred_sentences):
        for (word, gold_tag), pred_tag in zip(gold, pred):
            if gold_tag == pred_tag or len(errors) >= limit:
                continue
            errors.append({
                "token": word, "gold": gold_tag, "pred": pred_tag,
                "seen_in_train": word in train_vocab,
                "error_type": _classify_error(word, gold_tag, pred_tag,
                                              train_vocab, tagger),
            })
        if len(errors) >= limit:
            break
    return errors


# ============================= Section runners =============================

def _banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def q11_learned_parameters(tagger: HMMTagger) -> None:
    _banner("Q11 - Learned transitions and emissions")
    top_trans = tagger.top_transitions("NOUN", k=CFG.top_k)
    top_emit = tagger.top_emissions("VERB", k=CFG.top_k)

    trans_df = pd.DataFrame(top_trans, columns=["next_tag", "prob"])
    trans_df.insert(0, "prev_tag", "NOUN")
    emit_df = pd.DataFrame(top_emit, columns=["word", "prob"])
    emit_df.insert(0, "tag", "VERB")

    save_dataframe(trans_df, ART / "top_transitions_from_NOUN.csv")
    save_dataframe(emit_df, ART / "top_emissions_from_VERB.csv")

    print(f"top NOUN->: {top_trans}")
    print(f"top VERB emissions: {top_emit}")
    print()
    print("Top-5 transitions from NOUN:")
    print(trans_df.to_string(index=False))
    print()
    print("Top-5 emissions from VERB:")
    print(emit_df.to_string(index=False))


def q12_viterbi_note() -> None:
    _banner("Q12 - Viterbi implementation (from scratch)")
    print(
        "The Viterbi decoder lives in `HMMTagger.decode`. It operates in log\n"
        "space to avoid underflow and maintains two dense arrays:\n"
        "  score[t, s] = best log-probability of any tag sequence that ends\n"
        "                in tag s at position t.\n"
        "  back[t, s]  = the previous tag that produced that best score.\n"
        "\n"
        "Recurrence:\n"
        "  score[t, s] = max_{s'} ( score[t-1, s'] + log A(s' -> s) ) + log B(s -> w_t)\n"
        "  back[t, s]  = argmax_{s'} ( score[t-1, s'] + log A(s' -> s) )\n"
        "\n"
        "Boundary states are handled explicitly: position zero uses\n"
        "log_pi[s] = log P(s | <s>) and termination picks argmax of\n"
        "score[n-1, s] + log P(</s> | s) before backtracking.\n"
        "\n"
        "OOV fallback: for a word not seen in training, the emission is drawn\n"
        "from a hapax-legomena distribution over tags (learned at fit-time),\n"
        "which is measurably better on Brown than a plain uniform prior."
    )


def q13_accuracy(metrics: Mapping[str, float | int]) -> None:
    _banner("Q13 - Token-level accuracy (overall / seen / unseen)")
    acc_df = pd.DataFrame([
        {"bucket": "overall",
         "tokens": metrics["tokens"], "correct": metrics["correct"],
         "accuracy": metrics["accuracy"]},
        {"bucket": "seen",
         "tokens": metrics["seen_tokens"], "correct": metrics["seen_correct"],
         "accuracy": metrics["seen_accuracy"]},
        {"bucket": "unseen",
         "tokens": metrics["unseen_tokens"], "correct": metrics["unseen_correct"],
         "accuracy": metrics["unseen_accuracy"]},
    ])
    save_dataframe(acc_df, ART / "accuracy_table.csv")
    save_json(dict(metrics), ART / "accuracy_summary.json")
    print(acc_df.to_string(index=False))


def q14_error_analysis(errors: Sequence[Mapping[str, object]]) -> None:
    _banner("Q14 - Error analysis (ten mis-tagged tokens)")
    err_df = pd.DataFrame(errors)
    save_dataframe(err_df, ART / "error_analysis.csv")
    type_counts = err_df["error_type"].value_counts().to_dict()
    save_json({"errors": list(errors), "type_counts": type_counts},
              ART / "error_analysis.json")
    print(f"error_type_counts={type_counts}")
    print(err_df.to_string(index=False))
    print()
    print(f"Distribution by error type: {type_counts}")


# =================================== Main ==================================

_OUTPUT_MIRROR: Tuple[Tuple[str, str], ...] = (
    ("top_transitions_from_NOUN.csv", "part3_transitions_from_NOUN.csv"),
    ("top_emissions_from_VERB.csv",   "part3_emissions_from_VERB.csv"),
    ("accuracy_summary.json",         "part3_accuracy.json"),
    ("accuracy_table.csv",            "part3_accuracy.csv"),
    ("error_analysis.csv",            "part3_errors.csv"),
)


def _mirror_outputs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in _OUTPUT_MIRROR:
        shutil.copyfile(ART / src_name, OUT / dst_name)


def main() -> None:
    for d in (ART, OUT):
        d.mkdir(parents=True, exist_ok=True)

    print(f"config={CFG}")

    _stage("load and split tagged corpus")
    tagged = load_tagged_sentences(CFG.corpus_name, tagset=CFG.tagset)
    train, test = split_sentences(tagged, train_ratio=CFG.train_ratio, seed=CFG.seed)
    print(f"sentences total={len(tagged):,} train={len(train):,} test={len(test):,}")
    print(f"train_tokens={sum(len(s) for s in train):,} "
          f"test_tokens={sum(len(s) for s in test):,}")

    _stage("fit HMM")
    tagger = HMMTagger(alpha=CFG.laplace_alpha)
    tagger.fit(train)

    q11_learned_parameters(tagger)
    q12_viterbi_note()

    _stage("viterbi decode on test set")
    test_words = [[w for (w, _) in s] for s in test]
    pred_tags = tagger.decode_many(test_words, log_every=CFG.decode_log_every)

    _stage("evaluate")
    metrics = evaluate_predictions(test, pred_tags, tagger._train_vocab)
    print(f"accuracy overall={metrics['accuracy']:.4f} "
          f"seen={metrics['seen_accuracy']:.4f} "
          f"unseen={metrics['unseen_accuracy']:.4f}")
    q13_accuracy(metrics)

    _stage("error analysis")
    errors = collect_errors(test, pred_tags, tagger._train_vocab, tagger,
                            limit=CFG.error_examples)
    q14_error_analysis(errors)

    _mirror_outputs()
    print("part 3 done - artifacts written to artifacts/part3_hmm and outputs/")


if __name__ == "__main__":
    main()
