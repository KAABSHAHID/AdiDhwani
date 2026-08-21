<div align="center">

# AdiDhwani

**Preservation-Oriented ASR for Endangered Bhili and Mundari Languages**

Pooja Singh<sup>1\*</sup> · M Kaab Bin Shahid<sup>2\*</sup> · Sandeep Kumar<sup>1</sup>

<sup>1</sup>Indian Institute of Technology Delhi &nbsp;·&nbsp; <sup>2</sup>University of Stuttgart
<br><sub>\*Equal contribution</sub>

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-AdiDhwani-yellow.svg)](https://huggingface.co/misniitd)
[![License: Apache 2.0](https://img.shields.io/badge/Code%20License-Apache%202.0-blue.svg)](LICENSE)
[![Data License: CC BY-NC-SA 4.0](https://img.shields.io/badge/Data%20License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

</div>

---

## Overview

India's tribal (*Janjatiya*/*Adivasi*) languages remain absent from public ASR benchmarks. **AdiDhwani** is a documentation-oriented ASR benchmark for two under-documented Indian tribal languages:

| Language | ISO 639-3 | Family | Status | Speech |
|---|---|---|---|---|
| **Bhili** | `bhb` | Western Indo-Aryan | India's largest non-scheduled language (~10.4M speakers) | 55 h |
| **Mundari** | `unr` | North Munda (Austroasiatic) | **Vulnerable** (UNESCO) | 11 h |

The central claim of the paper is that **lower edit distance does not imply better documentation fidelity**. A system can reduce WER by substituting a community-attested Bhili form with a Hindi cognate, or by recognising a Mundari stem while silently deleting the suffix that carries its grammatical meaning. To make these failures measurable, we introduce four preservation diagnostics — **HNB**, **HNBA**, **MFR**, **SDR** — and report them alongside WER, CER, and the CER/WER ratio ρ.

This repository contains **all training, decoding, and analysis code** required to reproduce every table and figure in the paper.

### Headline results

| Language | Best system | WER | CER | ρ | HNB | HNBA | MFR | SDR |
|---|---|---|---|---|---|---|---|---|
| Bhili (`bhb`) | XLS-R 1B, real-only + KenLM | **22.32** | 8.83 | 0.40 | 18.70 | 17.09 | 0.62 | 3.40 |
| Bhili — *human* | fluent community transcribers | *11.24* | — | — | *8.38* | *7.89* | *0.11* | *1.20* |
| Mundari (`unr`) | XLS-R 1B, Syn→Org + KenLM | **37.52** | 17.91 | 0.48 | 21.24 | 17.41 | 1.92 | 4.47 |
| Mundari — *human* | fluent community transcribers | *14.31* | — | — | *10.17* | *9.18* | *0.32* | *2.17* |

> **Metric orthogonality.** Bhili's WER is 15.20 pp lower than Mundari's, yet their alignment-constrained Hindi-normalisation bias differs by only **0.32 pp** (17.09 % vs. 17.41 %). WER-based model selection does not minimise documentation-harm errors.

---

## Resources

| Resource | Link |
|---|---|
| Paper | https://arxiv.org/abs/XXXX.XXXXX &nbsp;*(placeholder — to be updated on publication)* |
| Dataset, splits, transcripts | https://huggingface.co/misniitd |
| Code (this repository) | https://github.com/misniitdelhi-ai/AdiDhwani |

The Hugging Face release provides: fixed **speaker-disjoint** train/test splits, audio with corresponding transcripts, the Hindi-cognate lexicon, and the community-validated suffix inventories used by the preservation metrics.

| Language | Split | Utterances | Duration | Mean length |
|---|---|---|---|---|
| Bhili | train | 19,000 | 55.0 h | 10.5 s |
| Bhili | test | 558 | 1.5 h | 10.2 s |
| Mundari | train | 7,500 | 11.0 h | 5.3 s |
| Mundari | test | 650 | 1.0 h | 5.1 s |

Auxiliary synthetic speech (Google Hindi TTS, `hi-IN-Wavenet-A`, rate 0.8×): 12 h for Bhili, 3 h for Mundari. **Synthetic audio is used only in training and never appears in development or test sets.**

---

## Repository structure

```
AdiDhwani/
├── Zero_Shot/
│   └── zero_shot.ipynb                     # 4 zero-shot multilingual baselines
├── Synthetic_Data_Generation/
│   └── Synthetic_Data.ipynb                # Google Hindi TTS synthesis + HF dataset build
├── Model_Training/
│   ├── XLSR_Models/                        # CTC — XLS-R 300M / 1B
│   │   ├── Original_and_Staged/train.py    # target-language fine-tuning (alpha sweep)
│   │   ├── Synthetic/train.py              # synthetic-only training
│   │   ├── Synthetic_to_Original/train.py  # Syn->Org: synthetic pre-train, real fine-tune
│   │   └── Transfer/
│   │       ├── 01_train.py                 # stage 1 — adapt to donor language
│   │       └── 02_train.py                 # stage 2 — adapt donor ckpt to target
│   └── Whisper_Models/                     # seq2seq — Tiny / Base / Small / Medium
│       ├── Original_and_Staged/finetune.py
│       ├── Synthetic/finetune.py
│       ├── Synthetic_to_Original/finetune.py
│       └── Transfer/{01_finetune.py, 02_finetune.py}
├── Language_Modeling/
│   └── XLSR_Models/
│       ├── Bhili_Mundari_LM_ASR.ipynb      # 5-gram KenLM + CTC shallow-fusion decoding
│       └── mundari_ASR_text.txt            # Mundari LM training text (train split only)
├── Analysis/
│   ├── ASR_Anlysis_figures.py              # all quantitative result figures
│   ├── Preservation_Analysis.py            # HNB / HNBA / MFR / SDR pipeline
│   ├── Linguistic_Analysis.py              # character-, word-, utterance-level diagnostics
│   ├── Transfer_language_analysis.ipynb    # layer-wise donor acoustic similarity
│   └── Marwari_Analysis.ipynb              # Rajasthani/Marwari donor probe (Appendix E.4)
├── LICENSE
└── README.md
```

> **Note on `Language_Modeling/`.** KenLM is applied to CTC systems only. Whisper's autoregressive decoder already provides an implicit language model, so no Whisper LM path exists by design (Section 5, *Donor Selection & Language-Model Decoding*).

---

## Code ↔ paper map

Every experimental claim in the paper is produced by one of the entry points below.

### Experimental pipeline

| Paper section | Regime | Code |
|---|---|---|
| §6 *Zero-Shot Transfer*, Fig. 1, Table 12 | Zero-shot transfer | `Zero_Shot/zero_shot.ipynb` |
| §6 *Target-Language Supervision*, Fig. 2, Fig. 4, Tables 13–14 | Target-language fine-tuning (α sweep) | `Model_Training/*/Original_and_Staged/` |
| §6 *Donor transfer*, Table 15, App. G.6 | Donor-language initialization | `Model_Training/*/Transfer/{01,02}_*` |
| §6 *Effect of Synthetic Supervision*, Fig. 9, App. G.5 | Synthetic supervision | `Model_Training/*/Synthetic/` · `Model_Training/*/Synthetic_to_Original/` |
| §6 *Language-Specific Pipelines*, Table 3, Fig. 13, App. G.7 | KenLM shallow fusion | `Language_Modeling/XLSR_Models/Bhili_Mundari_LM_ASR.ipynb` |
| App. F *Synthetic Speech Analysis* | TTS corpus construction | `Synthetic_Data_Generation/Synthetic_Data.ipynb` |

### Evaluation and analysis

| Paper artefact | Code | Notes |
|---|---|---|
| **Table 5** — preservation diagnostics vs. human baselines | `Analysis/Preservation_Analysis.py` | HNB, HNBA, MFR, SDR |
| **Tables 19, 20** — per-language metrics and deltas | `Analysis/Preservation_Analysis.py` | App. H |
| **Tables 21, 22** — global performance, severity bands | `Analysis/Linguistic_Analysis.py` | App. I.1, I.7 |
| **Figures 14–22** — error type, character/word confusion, position, length, severity | `Analysis/Linguistic_Analysis.py` | App. I |
| **Tables 8, 9, 10** · **Figures 5–8** — donor layer-zone similarity | `Analysis/Transfer_language_analysis.ipynb` | App. E.1–E.3 |
| **Table 11** — Marwari zero-shot representational probe | `Analysis/Marwari_Analysis.ipynb` | App. E.4 |
| **Figures 1, 2, 3, 4, 9, 10, 11, 12, 13** — quantitative result plots | `Analysis/ASR_Anlysis_figures.py` | see mapping below |

#### `ASR_Anlysis_figures.py` → paper figures

| Function | Paper figure | Content |
|---|---|---|
| `fig1_zeroshot_vs_finetuned()` | Figure 1 | Zero-shot WER vs. best fine-tuned reference |
| `fig2_scaling_laws()` | Figure 2 | WER vs. effective target hours + power-law fit |
| `fig3_alpha_heatmap()` | Figure 4 | α-sensitivity heatmap, optimal α per model |
| `fig4_pareto()` | Figure 12 | Parameter count vs. best WER, Pareto frontier |
| `fig5_synthetic()` / `fig11_synthetic()` | Figure 9 | Per-stage WER (XLS-R 1B / Whisper-Medium) |
| `fig6_transfer()` | Table 15 | Donor-language comparison |
| `fig7_lm_impact()` | Figure 13 | Pre/post-LM WER and CER by training path |
| `fig8_wer_cer_gap()` | Figure 10 | Post-LM WER vs. CER with ρ reference lines |
| `fig9_data_efficiency()` | Figure 11 | Marginal WER gain per effective training hour |
| `fig10_grand_summary()` | Figure 3 | Best WER per configuration, both languages |

---

## Installation

```bash
git clone https://github.com/misniitdelhi-ai/AdiDhwani.git
cd AdiDhwani

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

**Core dependencies**

```bash
pip install torch torchaudio                  # match your CUDA build
pip install transformers datasets accelerate
pip install jiwer editdistance
pip install numpy pandas scipy matplotlib seaborn tqdm
pip install jupyter
```

**KenLM decoding** (CTC systems only)

```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install pyctcdecode
```

Building the ARPA model additionally requires the KenLM binaries (`lmplz`, `build_binary`) — see [kpu/kenlm](https://github.com/kpu/kenlm). The notebook `Bhili_Mundari_LM_ASR.ipynb` includes the build steps.

**Synthetic speech generation** (only if regenerating the TTS corpus)

```bash
pip install google-cloud-texttospeech
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

**Devanagari rendering.** The analysis figures render Devanagari text. Install a Devanagari-capable font (e.g. *Noto Sans Devanagari*); `Linguistic_Analysis.py` auto-detects one via `_setup_devanagari_font()`.

---

## Data format

All training scripts consume a Hugging Face `datasets` directory saved with `save_to_disk()`, containing `train` and `test` splits with two columns:

| Column | Type | Description |
|---|---|---|
| `audio` | `Audio(sampling_rate=16000)` | 16 kHz, 16-bit mono waveform |
| `sentence` | `string` | Devanagari transcript |

Transcripts are NFC-normalised and reduced to Devanagari (`U+0900`–`U+097F`) plus a word delimiter inside each training script — the **same** deterministic normalisation is applied to references and hypotheses at scoring time.

The analysis scripts consume ASR output as a JSON list:

```json
[
  {"id": "bhb_test_0001", "reference": "...", "prediction": "..."},
  {"id": "bhb_test_0002", "reference": "...", "prediction": "..."}
]
```

`Preservation_Analysis.py` additionally requires the bilingual elicitation tier as a CSV with a `Hindi` column and a target column (`Bhili`, `Mundari`, or `Target`).

---

## Reproducing the results

Each stage below corresponds to one adaptation regime in Section 5 of the paper.

### 1 · Zero-shot baselines — Figure 1, Table 12

```bash
jupyter notebook Zero_Shot/zero_shot.ipynb
```

Evaluates the four zero-shot systems covering the major publicly available multilingual speech paradigms — encoder–decoder ASR, CTC-style recognition, and unified speech/text models:

| Model | Checkpoint | Bhili WER | Mundari WER |
|---|---|---|---|
| Whisper-large-v3 | `openai/whisper-large-v3` | 74.48 | 89.62 |
| SeamlessM4T-v2 | `facebook/seamless-m4t-v2-large` | 86.84 | 94.77 |
| MMS-1B | `facebook/mms-1b-all` | 101 | 106 |
| SpeechT5 | `microsoft/speecht5_asr` | 124 | 117 |

WER above 100 % indicates **model collapse** — hallucinated strings, repeated tokens, or near-empty output — not degraded-but-usable recognition. Large-scale multilingual pretraining does not guarantee graceful degradation on unseen languages.

### 2 · Target-language fine-tuning (α sweep) — Figure 2, Figure 4, Tables 13–14

Edit the `CONFIG` block at the top of the script, then run it:

```bash
# XLS-R (CTC)
python Model_Training/XLSR_Models/Original_and_Staged/train.py

# Whisper (seq2seq)
python Model_Training/Whisper_Models/Original_and_Staged/finetune.py
```

```python
# ==================== CONFIG ====================
model_checkpoint = "facebook/wav2vec2-xls-r-300m"   # or ...-1b
dataset_path     = "/path/to/adidhwani_bhili"       # save_to_disk() directory
batch_size       = 8
FREEZE_FEATURE_ENCODER = True
```

**Setting α.** The target-language fraction is controlled by the split selection near the top of the script:

```python
x    = len(train_data)
wndw = x // 5
train_data = train_data.select(range(4 * wndw))   # alpha = 0.8
```

Change the multiplier to sweep α — `1*wndw` → α=0.2, `2*wndw` → 0.4, `3*wndw` → 0.6, `4*wndw` → 0.8, and `range(x)` → α=1.0. Effective target hours are 11/22/33/44/55 h for Bhili and 2.2/4.4/6.6/8.8/11 h for Mundari.

**Key finding.** The optimal α is resource-dependent: Whisper-Medium is best at **α=1.0 (36.62 % WER)** for Bhili but at **α=0.8 (54.68 %)** for Mundari, where full target-only training overfits.

### 3 · Donor-language initialization — Table 15, Appendix G.6

Two stages, run in order:

```bash
# Stage 1 — adapt the base model to the donor language (Hindi, Marathi, Santali, ...)
python Model_Training/XLSR_Models/Transfer/01_train.py

# Stage 2 — adapt the donor checkpoint to the full target-language split
#   set model_checkpoint to the stage-1 output directory
python Model_Training/XLSR_Models/Transfer/02_train.py
```

Whisper's `02_finetune.py` exposes a separate `processor` path so the stage-1 tokenizer/processor is carried forward.

| Language | Donor | WER | Δ vs. no-donor |
|---|---|---|---|
| Bhili | **Hindi** | **45.33** | −1.20 pp |
| Bhili | Marathi | 57.91 | +11.38 pp |
| Bhili | Gujarati | 56.82 | +10.29 pp |
| Mundari | **Hindi** | **64.13** | -2.04 pp |
| Mundari | Santali | 77.81 | +11.64 pp |  
| Mundari | Bengali | 74.24 | +8.07 pp |      

Hindi is the only consistently helpful donor — **including for Mundari, where Santali is genealogically closer**. Acoustic representation similarity predicts transfer utility better than language-family membership.

Donor candidates are shortlisted *before* training using layer-wise similarity:

```bash
jupyter notebook Analysis/Transfer_language_analysis.ipynb   # Tables 8-10, Figures 5-8
jupyter notebook Analysis/Marwari_Analysis.ipynb             # Table 11, Appendix E.4
```

The phonetic zone (**L7–12**) is the most discriminative: donors above ≈0.6 are retained, those below 0.1 rejected (Gujarati 0.067, Bengali 0.063 — both rejected despite geographic proximity).

### 4 · Synthetic supervision — Figure 9, Appendix G.5

```bash
# Generate the auxiliary TTS corpus (optional — released with the dataset)
jupyter notebook Synthetic_Data_Generation/Synthetic_Data.ipynb

# Synthetic-only training
python Model_Training/XLSR_Models/Synthetic/train.py

# Syn->Org: synthetic pre-training, then real-data fine-tuning
python Model_Training/XLSR_Models/Synthetic_to_Original/train.py
```

The synthetic script takes **separate** `dataset_path` (synthetic train) and `test_dataset` (real test) paths, so synthetic audio never leaks into evaluation. For `Synthetic_to_Original`, point `model_checkpoint` at the synthetic-only output directory.

Synthetic-only training is far worse than real-data fine-tuning (Bhili 74.08 %, Mundari 79.50 %) — **TTS speech cannot replace real community speech.** Its value is conditional and only becomes visible after LM decoding (next step).

### 5 · KenLM shallow fusion — Table 3, Figure 13, Appendix G.7

```bash
jupyter notebook Language_Modeling/XLSR_Models/Bhili_Mundari_LM_ASR.ipynb
```

Trains a **5-gram KenLM** on target-language *training* transcripts only (development and test text are excluded) and combines CTC and LM scores by shallow fusion via `pyctcdecode` / `Wav2Vec2ProcessorWithLM`. `mundari_ASR_text.txt` is the Mundari LM training text.

| Language | Path | WER pre-LM | WER post-LM | Δ WER | CER post-LM | ρ |
|---|---|---|---|---|---|---|
| Bhili | Syn→Org + LM | 39.00 | 32.91 | −6.09 | 10.35 | 0.31 |
| Bhili | **Original + LM** | 44.06 | **22.32** | −21.74 | 8.83 | 0.40 |
| Mundari | **Syn→Org + LM** | 65.18 | **37.52** | −27.66 | 17.91 | 0.48 |
| Mundari | Original + LM | 62.01 | 42.74 | −19.27 | 20.17 | 0.47 |

**LM rescoring changes which training path is optimal.** With 55 h of real Bhili speech, synthetic pre-training is unnecessary and leaves acoustic artefacts the LM cannot repair. Under Mundari's 11-hour constraint, the 3-hour synthetic stage produces hypotheses that are easier to rerank — worse before the LM, best after it.

### 6 · Preservation diagnostics — Table 5, Tables 19–20

```bash
python Analysis/Preservation_Analysis.py \
    --parallel    hindi_bhili.csv \
    --predictions predictions_bhili.json \
    --language    bhili \
    --output_dir  ./output_bhili
```

| Argument | Default | Description |
|---|---|---|
| `--parallel` | *required* | Hindi–target parallel CSV (bilingual elicitation tier) |
| `--predictions` | *required* | JSON list of `{id, reference, prediction}` |
| `--language` | *required* | `bhili` or `mundari` — selects the suffix inventory |
| `--output_dir` | `./output` | Destination for all artefacts |
| `--min_cooc` | `2` | Minimum co-occurrence count for lexicon induction |
| `--min_pmi` | `0.5` | Minimum PMI for lexicon induction |
| `--use_cached` | off | Reuse existing `hindi_alts.json` / `agglutinated.json` |

Outputs `hindi_alts.json`, `agglutinated.json`, `metrics_report.json`, and a human-readable `metrics_report.txt`.

**What the four diagnostics measure**

| Metric | Question | Failure captured |
|---|---|---|
| **HNB** | Does an approved Hindi cognate appear *anywhere* in the hypothesis for a monitored token? | Broad dominant-language drift |
| **HNBA** | Does it appear at the *aligned* reference position (WER traceback)? | Direct Hindi substitution. By construction HNBA ≤ HNB; a small gap means substitutions, a large gap means incidental co-occurrence |
| **MFR** | Are stem and suffix both recognised but *not* preserved as one agglutinated word? | Morphological fragmentation (e.g. `teko` → `te` + `ko`) |
| **SDR** | Does the stem survive while *no* known suffix appears? | Suffix deletion — the more damaging error, since the transcript stays lexically plausible while losing grammatical information |

The monitored set is a **curated subset** of reference tokens flagged as at risk of Hindi substitution — induced by PMI from the bilingual tier and confirmed by community reviewers — not the full transcript vocabulary. The suffix inventories are community-validated (>30 suffixes for Bhili, >25 for Mundari); a token counts as agglutinated only if removing a known suffix leaves an attested stem of at least two characters.

### 7 · Linguistic error analysis — Figures 14–22, Tables 21–22

```bash
python Analysis/Linguistic_Analysis.py \
    --input      predictions_bhili.json \
    --output_dir outputs_bhili
```

Produces WER/CER summaries, error-type breakdown, top-30 character confusion pairs and full confusion matrices, top-25 word confusion pairs, Devanagari category breakdown, positional and length effects, substitution edit-distance distributions, severity bands, and a text summary report.

This analysis separates **normalisation artefacts** (ZWNJ omission, ASCII-colon vs. Unicode visarga) from **documentation-relevant errors** (nukta loss, vowel-length confusion, Hindi postposition over-generation, visarga-suffix deletion) — which determines whether future gains should come from Unicode normalisation, stronger acoustic modelling, or more data.

### 8 · Result figures — Figures 1–4, 9–13

```bash
python Analysis/ASR_Anlysis_figures.py
```

This script embeds the paper's reported result tables directly, so it regenerates every quantitative figure without re-running training. Before use:

1. Set `OUT` (near line 59) to a writable directory, e.g. `OUT = './figures'`.
2. Uncomment the desired figure calls in the `__main__` block — all but one are commented out by default.

---

## Training configuration

All hyperparameters are fixed uniformly across languages and configurations so no model receives a tuning advantage (Table 23).

| Hyperparameter | XLS-R (Wav2Vec2) | Whisper |
|---|---|---|
| Batch size (per device) | 8 | 8 |
| Gradient accumulation | 2 | 1 |
| Effective batch size | 16 | 8 |
| Learning rate | 1e-4 | 1e-5 |
| Warmup steps | 500 | 500 |
| Max gradient norm | 1.0 | — |
| Training epochs | 20 | 20 |
| Eval and save strategy | per epoch | per epoch |
| Best-checkpoint criterion | min WER | min WER |
| Checkpoints retained | best only | 3 |
| Gradient checkpointing | yes | no |
| Feature encoder frozen | yes | n/a |
| Decoding | greedy (CTC argmax) | beam search (beam 5) |
| Max generation length | — | 448 tokens |

**Whisper language token.** Neither Bhili nor Mundari has an official Whisper language token, so the Hindi token is used for all Whisper fine-tuning and decoding (`language="Hindi"`, `task="transcribe"`, `forced_decoder_ids=None`). Its effect on Hindi-normalisation bias is analysed in Appendix H.

**Compute.** All experiments ran on NVIDIA A100 (40 GB) GPUs with FP16 mixed precision via HuggingFace Transformers. Training time ranged from roughly 4 to 18 GPU-hours per run depending on model size and dataset scale.

---

## Extending AdiDhwani to other languages

The transferable contribution is methodological, not a specific WER value. To apply this framework to another under-documented language:

1. **Collect and adjudicate** speech with a maker–checker–superchecker workflow and fixed speaker-disjoint splits (Appendix B, C).
2. **Define the risk profile.** Dominant-language normalisation, morphology, tone, orthography, word segmentation, or named entities — whichever failure modes matter for that community.
3. **Replace the inventories.** Swap `BHILI_SUFFIXES` / `MUNDARI_SUFFIXES` and the `SUFFIX_TABLE` mapping in `Analysis/Preservation_Analysis.py`, and supply a parallel tier in the dominant contact language for lexicon induction.
4. **Select donors empirically** with `Analysis/Transfer_language_analysis.ipynb` — phonetic-zone similarity predicts transfer better than genealogy, and a donor must be *instantiable* (enough transcribed speech to adapt XLS-R, or an existing checkpoint).
5. **Report preservation diagnostics alongside WER and CER**, never instead of them.

---

## Limitations

AdiDhwani is an initial benchmark. It does not capture the full sociolinguistic diversity of either language; most recordings are read speech with limited spontaneous interaction; both languages are released primarily in Devanagari despite broader multi-script usage, particularly for Mundari; and the donor analysis is restricted to varieties with instantiable ASR conditions, which excludes the Rajasthani varieties most closely related to Bhili (Appendix E.4). See the paper's *Limitations* section for the full discussion.

---

## Citation

```bibtex
@inproceedings{singh2026adidhwani,
  title  = {AdiDhwani: Preservation-Oriented ASR for Endangered Bhili and Mundari Languages},
  author = {Singh, Pooja and Shahid, M Kaab Bin and Kumar, Sandeep},
  year   = {2026},
  note   = {Equal contribution: Pooja Singh and M Kaab Bin Shahid},
  url    = {https://github.com/misniitdelhi-ai/AdiDhwani}
}
```

*The BibTeX entry will be updated with full venue and page information on publication.*

---

## Licensing

| Artefact | License |
|---|---|
| Code in this repository | [Apache License 2.0](LICENSE) |
| Speech recordings and transcripts | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — non-commercial academic and research use |

---

## Ethics and data governance

All participants were informed about the project's goals, the recording process, and the intended research and documentation uses of the collected speech prior to participation. Consent procedures were conducted in locally understandable languages, and informed verbal and written consent was obtained from all speakers. Participants were compensated in accordance with prevailing local daily wages.

The benchmark was developed in collaboration with community linguists, educators, and local language practitioners. Individual speaker identities are not published unless public attribution is consistent with the corresponding consent and community-governance conditions. AdiDhwani is framed as a documentation-oriented research resource intended to support language preservation, education, and community-centered speech technology.

---

## Acknowledgments

This work was carried out under the **AdiVaani** initiative of the **Ministry of Tribal Affairs, Government of India**, aimed at advancing language technologies for tribal and endangered Indian languages. We gratefully acknowledge the Ministry's support to **IIT Delhi**, and **IIT Delhi HPC** for the computational and GPU resources used in our experiments.

We thank the **Tribal Research Institutes**, community partners, language experts, transcribers, checkers, supercheckers, field coordinators, and — above all — the **Bhili and Mundari speakers and their communities**, whose voices, time, and linguistic knowledge form the foundation of this work. A complete contributor roster accompanies the dataset release.
