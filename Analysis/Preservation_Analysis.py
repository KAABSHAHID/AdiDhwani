"""
ASR Linguistic Metrics Pipeline
================================
Computes HNB / HNBA, MFR, and SDR for a low-resource Indic ASR system.

Inputs
------
1. parallel_csv   : CSV with columns [Hindi, Bhili]  (or [Hindi, Mundari])
2. predictions_json: JSON list of {"id", "reference", "prediction"} dicts

Outputs
-------
- hindi_alts.json        : Bhili/Mundari token → set of Hindi alternatives
- agglutinated.json      : complex token → (stem, suffixes)
- metrics_report.json    : HNB, HNBA, MFR, SDR with per-utterance detail
- metrics_report.txt     : human-readable summary

Usage
-----
    python metrics_pipeline.py \
        --parallel   hindi_bhili.csv \
        --predictions predictions.json \
        --language   bhili \
        --output_dir ./output
"""

import argparse
import json
import re
import csv
import math
from pathlib import Path
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  TEXT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

# Devanagari punctuation + ASCII punctuation to strip before tokenising
PUNCT = re.compile(r'[।॥?!,،.،;:\-\u200d\u200c\u200b\ufeff"\'()]+')

def tokenize(text: str) -> list[str]:
    text = PUNCT.sub(' ', text)
    return [t for t in text.split() if t.strip()]


# Common Hindi function words — exclude from hindi_alts to reduce noise
HINDI_STOP = {
    'का','के','की','को','में','से','पर','है','हैं','था','थे','थी',
    'और','या','तो','भी','ही','न','नहीं','यह','वह','वे','हम','मैं',
    'आप','इस','उस','एक','कि','जो','जब','तब','अब','यहाँ','वहाँ',
    'कर','हो','ने','ना','नी','ना','से','ते','रहे','रही','जा','आ',
    'हैं','हूँ','हूं','था','थे','गए','गई','गया','लिए','लिया','दिया',
    'कोई','कुछ','सब','सभी','बहुत','अपने','अपना','अपनी','उनके','उनका',
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  WORD ALIGNMENT  (co-occurrence + pointwise mutual information)
# ═══════════════════════════════════════════════════════════════════════════════

def build_alignment(pairs: list[tuple[list,list]],
                    min_cooc: int = 2,
                    min_pmi: float = 0.5) -> dict[str, set[str]]:
    """
    Align Hindi tokens to target-language tokens using PMI over the corpus.

    Returns
    -------
    { hindi_token : set(target_tokens) }
    """
    cooc          = defaultdict(lambda: defaultdict(int))  # cooc[h][t]
    hindi_freq    = defaultdict(int)
    target_freq   = defaultdict(int)
    total_pairs   = 0

    for h_toks, t_toks in pairs:
        h_set = set(h_toks) - HINDI_STOP
        t_set = set(t_toks)
        for h in h_set:
            hindi_freq[h] += 1
        for t in t_set:
            target_freq[t] += 1
        for h in h_set:
            for t in t_set:
                cooc[h][t] += 1
                total_pairs += 1

    N = len(pairs)
    alignment: dict[str, set] = {}

    for h, t_counts in cooc.items():
        best: set = set()
        for t, cnt in t_counts.items():
            if cnt < min_cooc:
                continue
            # PMI = log[ p(h,t) / (p(h)*p(t)) ]
            p_ht = cnt / total_pairs
            p_h  = hindi_freq[h] / N
            p_t  = target_freq[t] / N
            pmi  = math.log(p_ht / (p_h * p_t + 1e-12) + 1e-12)
            if pmi >= min_pmi:
                best.add(t)
        if best:
            alignment[h] = best

    return alignment


def build_hindi_alts(alignment: dict[str, set]) -> dict[str, set]:
    """
    Reverse alignment: target token → set of Hindi tokens that map to it.
    These are the Hindi alternatives an ASR may hallucinate.
    """
    hindi_alts: dict[str, set] = defaultdict(set)
    for h_tok, t_set in alignment.items():
        for t in t_set:
            hindi_alts[t].add(h_tok)
    return dict(hindi_alts)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  AGGLUTINATION DETECTION  (suffix-stripping heuristic)
# ═══════════════════════════════════════════════════════════════════════════════

# Extend / edit these per language after community review
BHILI_SUFFIXES = [
    'वाळीयों','वाळाओं','वाळियां','रीयाओं',          # 4-char
    'वाळी','वाळा','वाळो','रीया','ईयां','आईने',        # longer first
    'ईने','ओयीं','युयीं','मुयीं','ईयों',
    'से','ने','ना','नी','वु','या','ये','ओं','ां',
    'ीं','ुं','री','रा','ई','ओ','वा','ला','ली','लो',
    'ती','ते','ता','तां','ीया','रीया',
]

MUNDARI_SUFFIXES = [
    'ओड़ोः','होड़ोः',                                  # common agglutinated forms
    'तइकेना','केआते','कोते',
    'रअः','रेअः','नेअः','लेल',
    'ोः','अः','इः','उः','एः','आः',                    # visarga suffixes
    'ते','रे','को','गे','ने','ले','से',
    'ना','नी','ता','ती','आ','इ','उ',
]

SUFFIX_TABLE = {
    'bhili':   sorted(BHILI_SUFFIXES,   key=len, reverse=True),
    'mundari': sorted(MUNDARI_SUFFIXES, key=len, reverse=True),
}


def detect_suffix(token: str, suffix_list: list[str]):
    """Return (stem, suffix) or None."""
    for suf in suffix_list:
        if token.endswith(suf) and len(token) > len(suf) + 1:
            return token[: -len(suf)], suf
    return None


def build_agglutinated(corpus_tokens: set[str],
                       language: str) -> dict[str, tuple[str, set]]:
    """
    For every token in the corpus, attempt suffix stripping.
    Returns { token: (stem, {suffix}) }
    """
    suffix_list = SUFFIX_TABLE.get(language, BHILI_SUFFIXES)
    result = {}
    for tok in corpus_tokens:
        hit = detect_suffix(tok, suffix_list)
        if hit:
            stem, suf = hit
            result[tok] = (stem, {suf})
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TOKEN ALIGNMENT  (edit-distance traceback)
# ═══════════════════════════════════════════════════════════════════════════════

def align_tokens(ref: list[str], hyp: list[str]) -> list[tuple]:
    """
    Returns list of (ref_tok_or_None, hyp_tok_or_None).
    None on the ref side = insertion; None on hyp side = deletion.
    """
    R, H = len(ref), len(hyp)
    # DP
    dp = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1): dp[i][0] = i
    for j in range(H + 1): dp[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    # Traceback
    alignment = []
    i, j = R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            alignment.append((ref[i-1], hyp[j-1], 'match'))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append((ref[i-1], hyp[j-1], 'sub'))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignment.append((ref[i-1], None, 'del'))
            i -= 1
        else:
            alignment.append((None, hyp[j-1], 'ins'))
            j -= 1
    return list(reversed(alignment))


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hnb(data: list[dict],
                hindi_alts: dict[str, set]) -> dict:
    """
    HNB  : Hindi alternative appears anywhere in hypothesis.
    HNBA : Hindi alternative appears at the aligned position only.
    """
    hnb_num = hnba_num = denom = 0
    cases = []

    for item in data:
        ref_toks = tokenize(item['reference'])
        hyp_toks = tokenize(item['prediction'])
        hyp_set  = set(hyp_toks)
        alignment = align_tokens(ref_toks, hyp_toks)

        for r_tok, h_tok, op in alignment:
            if r_tok is None or r_tok not in hindi_alts:
                continue
            denom += 1
            cognates = hindi_alts[r_tok]

            hit_unaligned = bool(cognates & hyp_set)
            hit_aligned   = h_tok in cognates if h_tok else False

            if hit_unaligned:
                hnb_num += 1
            if hit_aligned:
                hnba_num += 1

            if hit_unaligned or hit_aligned:
                cases.append({
                    'id':           item['id'],
                    'ref_token':    r_tok,
                    'hyp_token':    h_tok,
                    'cognates':     list(cognates),
                    'HNB_hit':      hit_unaligned,
                    'HNBA_hit':     hit_aligned,
                })

    return {
        'HNB':         round(hnb_num  / denom, 4) if denom else 0.0,
        'HNBA':        round(hnba_num / denom, 4) if denom else 0.0,
        'numerator_HNB':  hnb_num,
        'numerator_HNBA': hnba_num,
        'denominator':    denom,
        'cases':          cases,
    }


def compute_mfr_sdr(data: list[dict],
                    agglutinated: dict[str, tuple]) -> dict:
    """
    MFR: agglutinated reference token split into 2+ hypothesis tokens.
    SDR: stem recognised but suffix(es) absent in hypothesis.
    """
    mfr_num = sdr_num = denom = 0
    mfr_cases = []
    sdr_cases  = []

    for item in data:
        ref_toks  = tokenize(item['reference'])
        hyp_toks  = tokenize(item['prediction'])
        hyp_set   = set(hyp_toks)
        alignment = align_tokens(ref_toks, hyp_toks)

        for r_tok, h_tok, op in alignment:
            if r_tok is None or r_tok not in agglutinated:
                continue
            denom += 1
            stem, suffixes = agglutinated[r_tok]

            stem_in_hyp    = stem in hyp_set
            # any suffix appears as its own token
            loose_suffix   = any(s in hyp_set for s in suffixes)
            # suffix appears attached to stem in hypothesis token
            suffix_in_htok = h_tok and any(h_tok.endswith(s) for s in suffixes)

            # ── MFR ──────────────────────────────────────────────────────────
            # Stem is a standalone hyp token AND a suffix is also standalone
            if stem_in_hyp and loose_suffix and h_tok != r_tok:
                mfr_num += 1
                mfr_cases.append({
                    'id':               item['id'],
                    'ref_token':        r_tok,
                    'hyp_aligned':      h_tok,
                    'stem_found':       stem,
                    'suffixes_found':   list(suffixes & hyp_set),
                })

            # ── SDR ──────────────────────────────────────────────────────────
            # Stem recognised (standalone OR as prefix of aligned hyp token)
            # but no suffix present anywhere
            elif (stem_in_hyp or (h_tok and h_tok.startswith(stem))) \
                    and not loose_suffix and not suffix_in_htok:
                sdr_num += 1
                sdr_cases.append({
                    'id':               item['id'],
                    'ref_token':        r_tok,
                    'hyp_aligned':      h_tok,
                    'stem':             stem,
                    'missing_suffixes': list(suffixes),
                })

    return {
        'MFR':         round(mfr_num / denom, 4) if denom else 0.0,
        'SDR':         round(sdr_num / denom, 4) if denom else 0.0,
        'numerator_MFR':  mfr_num,
        'numerator_SDR':  sdr_num,
        'denominator':    denom,
        'MFR_cases':      mfr_cases,
        'SDR_cases':      sdr_cases,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  SERIALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def save_json(obj, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: Path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def sets_to_lists(d: dict) -> dict:
    return {k: list(v) if isinstance(v, set) else v for k, v in d.items()}

def lists_to_sets(d: dict) -> dict:
    return {k: set(v) if isinstance(v, list) else v for k, v in d.items()}

def agglut_to_serializable(d: dict) -> dict:
    return {k: [stem, list(suffs)] for k, (stem, suffs) in d.items()}

def agglut_from_serializable(d: dict) -> dict:
    return {k: (v[0], set(v[1])) for k, v in d.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(parallel_csv: str,
                 predictions_json: str,
                 language: str,
                 output_dir: str,
                 # Alignment quality knobs
                 min_cooc: int = 2,
                 min_pmi: float = 0.5,
                 # Set True to skip dict rebuild if JSON files already exist
                 use_cached_dicts: bool = False):

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ha_path  = out / 'hindi_alts.json'
    agg_path = out / 'agglutinated.json'

    # ── Step 1: load parallel corpus ─────────────────────────────────────────
    print(f"[1/5] Loading parallel corpus from {parallel_csv} ...")
    pairs: list[tuple[list, list]] = []
    all_target_tokens: set[str] = set()

    with open(parallel_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Accept 'Bhili', 'Mundari', or generic 'Target' as second column
        fieldnames = reader.fieldnames or []
        target_col = next((c for c in fieldnames if c != 'Hindi'), None)
        if target_col is None:
            raise ValueError(f"Cannot find target column in {fieldnames}")
        print(f"   Detected columns: Hindi | {target_col}")

        for row in reader:
            h_toks = tokenize(row['Hindi'])
            t_toks = tokenize(row[target_col])
            pairs.append((h_toks, t_toks))
            all_target_tokens.update(t_toks)

    print(f"   {len(pairs):,} sentence pairs | "
          f"{len(all_target_tokens):,} unique target tokens")

    # ── Step 2: build / load dictionaries ────────────────────────────────────
    if use_cached_dicts and ha_path.exists() and agg_path.exists():
        print("[2/5] Loading cached dictionaries ...")
        hindi_alts   = lists_to_sets(load_json(ha_path))
        agglutinated = agglut_from_serializable(load_json(agg_path))
    else:
        print("[2/5] Building alignment & dictionaries ...")
        alignment    = build_alignment(pairs, min_cooc=min_cooc, min_pmi=min_pmi)
        hindi_alts   = build_hindi_alts(alignment)
        agglutinated = build_agglutinated(all_target_tokens, language)

        save_json(sets_to_lists(hindi_alts),         ha_path)
        save_json(agglut_to_serializable(agglutinated), agg_path)
        print(f"   hindi_alts : {len(hindi_alts):,} entries  → {ha_path}")
        print(f"   agglutinated: {len(agglutinated):,} entries → {agg_path}")

    # ── Step 3: load ASR predictions ─────────────────────────────────────────
    print(f"[3/5] Loading predictions from {predictions_json} ...")
    data: list[dict] = load_json(predictions_json)
    print(f"   {len(data):,} utterances")

    # ── Step 4: compute metrics ───────────────────────────────────────────────
    print("[4/5] Computing HNB / HNBA ...")
    hnb_result = compute_hnb(data, hindi_alts)

    print("[4/5] Computing MFR / SDR ...")
    morph_result = compute_mfr_sdr(data, agglutinated)

    # ── Step 5: save reports ──────────────────────────────────────────────────
    print("[5/5] Saving reports ...")

    full_report = {
        'language':   language,
        'utterances': len(data),
        'HNB':        hnb_result['HNB'],
        'HNBA':       hnb_result['HNBA'],
        'MFR':        morph_result['MFR'],
        'SDR':        morph_result['SDR'],
        'details': {
            'HNB':  hnb_result,
            'morph': morph_result,
        }
    }
    save_json(full_report, out / 'metrics_report.json')

    # Human-readable summary
    summary = f"""
=================================================================
  ASR LINGUISTIC METRICS — {language.upper()}
=================================================================
  Utterances analysed       : {len(data):,}
  Monitored tokens (HNB)    : {hnb_result['denominator']:,}
  Agglutinated tokens (MFR/SDR): {morph_result['denominator']:,}

  ── Dominant-Language Bias ──────────────────────────────────
  HNB  (unaligned)           : {hnb_result['HNB']:.2%}
  HNBA (alignment-constrained): {hnb_result['HNBA']:.2%}
    → {hnb_result['numerator_HNB']} / {hnb_result['denominator']} monitored tokens Hindi-normalised

  ── Morphological Integrity ─────────────────────────────────
  MFR  (fragmentation rate)  : {morph_result['MFR']:.2%}
    → {morph_result['numerator_MFR']} agglutinated tokens split into pieces
  SDR  (suffix deletion rate): {morph_result['SDR']:.2%}
    → {morph_result['numerator_SDR']} agglutinated tokens with suffix lost

=================================================================
  Sample HNB cases (first 5):
"""
    for c in hnb_result['cases'][:5]:
        summary += (f"    [{c['id']}] ref={c['ref_token']} "
                    f"hyp={c['hyp_token']} cognates={c['cognates']}\n")

    summary += "\n  Sample MFR cases (first 5):\n"
    for c in morph_result['MFR_cases'][:5]:
        summary += (f"    [{c['id']}] ref={c['ref_token']} "
                    f"stem={c['stem_found']} split_suffixes={c['suffixes_found']}\n")

    summary += "\n  Sample SDR cases (first 5):\n"
    for c in morph_result['SDR_cases'][:5]:
        summary += (f"    [{c['id']}] ref={c['ref_token']} "
                    f"hyp={c['hyp_aligned']} missing={c['missing_suffixes']}\n")

    summary += "=================================================================\n"

    txt_path = out / 'metrics_report.txt'
    txt_path.write_text(summary, encoding='utf-8')
    print(summary)
    print(f"Reports saved to {out}/")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute HNB, MFR, SDR for ASR output.')
    parser.add_argument('--parallel',     required=True,
                        help='Path to Hindi–Target parallel CSV')
    parser.add_argument('--predictions',  required=True,
                        help='Path to predictions JSON [{id, reference, prediction}]')
    parser.add_argument('--language',     required=True,
                        choices=['bhili', 'mundari'],
                        help='Target language (selects suffix table)')
    parser.add_argument('--output_dir',   default='./output',
                        help='Directory to write all output files')
    parser.add_argument('--min_cooc',     type=int,   default=2,
                        help='Minimum co-occurrence count for alignment')
    parser.add_argument('--min_pmi',      type=float, default=0.5,
                        help='Minimum PMI for alignment')
    parser.add_argument('--use_cached',   action='store_true',
                        help='Skip dict rebuild if JSON files already exist')
    args = parser.parse_args()

    run_pipeline(
        parallel_csv      = args.parallel,
        predictions_json  = args.predictions,
        language          = args.language,
        output_dir        = args.output_dir,
        min_cooc          = args.min_cooc,
        min_pmi           = args.min_pmi,
        use_cached_dicts  = args.use_cached,
    )