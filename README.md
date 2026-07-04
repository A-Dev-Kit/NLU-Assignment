# Sanskrit -> English Neural Machine Translation

Fine-tuned MT for Sanskrit (Devanagari) to English on a small parallel corpus
(10,000 train / 1,000 dev / 1,000 test). Several configurations were tried;
the winner - shipped in this repo - is a full fine-tune of the AI4Bharat
IndicTrans2 1.1B checkpoint. Its predictions on the standard test set are in
`outputs/submission.csv`.

## Winner

| metric                    | value                            |
|---------------------------|----------------------------------|
| model                     | IndicTrans2 1.1B (full fine-tune) |
| base checkpoint           | `ai4bharat/indictrans2-indic-en-1B` |
| test BLEU                 | 22.036                           |
| BERTScore F1 (rescaled)   | 0.578                            |
| BERTScore F1 (raw)        | 0.929                            |
| parameters (M)            | 1023                             |
| trainable (M)             | 1023 (full FT)                   |
| inference time (s / 1k)   | 265                              |
| train time (s)            | 1972 (about 33 min, 3 epochs)    |

The full ranking across every configuration tried is captured in
`artifacts/comparison/winner.json`. Loss and dev-BLEU curves are at
`artifacts/indictrans2_full/plots/training_curves.png`.

## Layout

```
notebooks/05_indictrans2_full.ipynb          the winning notebook
outputs/submission.csv                        winner's predictions on the standard test set
logs/indictrans2_full.log                     training + eval log
artifacts/indictrans2_full/metrics.json       final metrics
artifacts/indictrans2_full/examples.csv       eight sample translations
artifacts/indictrans2_full/plots/             training curves
artifacts/comparison/winner.json              cross-model ranking that picked this winner
```

The 2 GB `checkpoint.pt` is excluded from the repo (`.gitignore`). Reproduce
it locally with the steps below.

## Setup

```bash
python -m venv .venv
# Windows:   .\.venv\Scripts\pip install -r requirements.txt
# Linux/mac: ./.venv/bin/pip install -r requirements.txt
python -m ipykernel install --user --name nlu-asg2 --display-name "NLU Asg 2"
```

The notebook's first cell also pip-installs everything it needs, so a fresh
kernel works out of the box.

## Running the notebook

`notebooks/05_indictrans2_full.ipynb` supports three run modes through the
`NB_MODE` environment variable:

- `NB_MODE=train` (default): full pipeline. Trains for 3 epochs on a CUDA GPU
  with >= 8 GB VRAM (bf16 autocast + gradient checkpointing + `bitsandbytes`
  paged 8-bit AdamW). About 33 minutes on an RTX 5070 Laptop. Writes
  `artifacts/indictrans2_full/checkpoint.pt` and refreshes
  `outputs/submission.csv` at the end.
- `NB_MODE=eval_only`: skips training, loads
  `artifacts/indictrans2_full/checkpoint.pt`, evaluates on the standard test
  set. About 5 minutes.
- `NB_MODE=eval_custom` with `CUSTOM_CSV_PATH=<path>`: same as `eval_only` but
  the test set is replaced by the CSV at `CUSTOM_CSV_PATH`. That file must
  contain a `Sentence_sa` column with Sanskrit in Devanagari; predictions land
  in `outputs/predictions_indictrans2_full_custom.csv`.

## Pretrained models used

- `ai4bharat/indictrans2-indic-en-1B` (Apache-2.0 weights, MIT code)
- `IndicTransToolkit` (MIT), used only for source preprocessing / postprocessing
- `roberta-large` -- used ONLY inside `bert_score` for evaluation

No translation APIs and no other pretrained MT weights are used.

## Notes

- All numbers and outputs in this repo come from the actual training run
  recorded in `logs/indictrans2_full.log`. Nothing hand-edited.
- The ranking in `artifacts/comparison/winner.json` includes the other
  configurations that were tried but whose training code is not shipped in
  this repo.
