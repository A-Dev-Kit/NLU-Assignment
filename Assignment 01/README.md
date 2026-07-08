# Assignment 01

Three self-contained Python scripts for the NLU Assignment 1 tasks.

- `part1_zipf.py`            Zipf's Law and corpus analysis (Q1-Q5) on NLTK Gutenberg
- `part2_language_model.py`  Unigram / bigram / trigram add-1 language models (Q6-Q10) on NLTK Gutenberg
- `part3_hmm_tagger.py`      HMM POS tagger with a from-scratch Viterbi (Q11-Q14) on NLTK Brown

The corresponding NLTK corpora are bundled under `nltk_data/` at the folder root, so the scripts run offline without any `nltk.download()` step.

## Install

```
pip install -r requirements.txt
```

## Run

```
python part1_zipf.py
python part2_language_model.py
python part3_hmm_tagger.py
```

Each script prints its Q&A answers to stdout with `====` banners and writes tables / plots / JSON summaries under `./artifacts/partN_*/` (and a flat copy of the tables the report cites under `./outputs/`).

## Reproducibility

- Fixed seed (`42`) for splits and sampling.
- Same corpora as the packaged submission ZIPs, so all numbers are byte-identical.
