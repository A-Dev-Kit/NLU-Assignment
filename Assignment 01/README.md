# Assignment 01

Solutions to the three parts of NLU Assignment 1:

1. **Part 1** - Zipf's Law and corpus analysis on the NLTK Gutenberg corpus.
2. **Part 2** - Unigram, bigram and trigram language models with Laplace smoothing, perplexity, and ancestral sampling on the same corpus.
3. **Part 3** - A Hidden Markov Model POS tagger with a from-scratch Viterbi decoder on the NLTK Brown corpus (Universal tagset).

Each of the three notebooks is fully self-contained. They share no imports, no `src/` package, and no cross-notebook state, so any one of them can be opened and run first without touching the others.

## Layout

```
notebooks/
  Part_1_Zipf_and_Corpus_Analysis.ipynb   Part 1 (Q1-Q5)
  Part_2_Language_Models.ipynb            Part 2 (Q6-Q10)
  Part_3_HMM_Viterbi_POS_Tagger.ipynb     Part 3 (Q11-Q14)
artifacts/
  part1_zipf/                CSVs, JSON, PNGs for Part 1
  part2_lm/                  perplexity table, samples, model summary
  part3_hmm/                 transitions, emissions, accuracy, error analysis
logs/                        one *.log per part (from the last run)
requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
python -m ipykernel install --user --name nlu-asg1 --display-name "NLU Asg 1"
```

NLTK data (`gutenberg`, `brown`, `universal_tagset`, `punkt`) is fetched on first execution by the `ensure_corpora()` call at the top of every notebook.

## Running

One command per part, per the assignment spec:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/Part_1_Zipf_and_Corpus_Analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/Part_2_Language_Models.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/Part_3_HMM_Viterbi_POS_Tagger.ipynb
```

Progress for every part is streamed to `logs/part{1,2,3}_*.log` and to stdout.

## Reproducibility

- Fixed seed (`42`) for splits and sampling.
- No hidden randomness; every random-consuming function takes an explicit `seed` argument.
- Every knob (corpus name, split ratio, smoothing constants, TTR checkpoints, MATTR window, OOV threshold, file paths) lives in a `@dataclass` config at the top of each notebook.
