"""
ASR Linguistic Analysis Pipeline — White/Academic Style
========================================================
Comprehensive linguistic error analysis for ASR model output.
Input: JSON file with fields: id, reference, prediction

Analyses produced:
  1.  WER / CER              — overall metrics
  2.  Error type breakdown   — substitution / deletion / insertion
  3.  Character-level confusion (top-N pairs)
  4.  Word-level confusion pairs
  5.  Error position analysis
  6.  Utterance length vs WER scatter
  7.  Phonetic similarity of substituted pairs
  8.  Deletion & insertion analysis
  9.  Script / Devanagari category analysis
  10. Error severity distribution
  11. CER vs WER scatter

Requirements:
    pip install jiwer editdistance matplotlib seaborn pandas numpy tqdm
"""

import json, re, os, sys, collections, itertools, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as _fm
import seaborn as sns
from tqdm import tqdm

try:
    import editdistance
    HAS_EDITDISTANCE = True
except ImportError:
    HAS_EDITDISTANCE = False

try:
    from jiwer import wer, cer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

INPUT_JSON       = "prediction.json"
OUTPUT_DIR       = "outputs_bhili"
TOP_N_CONFUSION  = 30
TOP_N_WORDS      = 25
DEVANAGARI_ONLY  = True

# ─────────────────────────────────────────────────────────────────────────────
# ACADEMIC WHITE PALETTE  (replaces the old dark PALETTE entirely)
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg":       "white",
    "surface":  "#F8F9FA",     # very light gray panels
    "border":   "#DEE2E6",     # subtle borders
    "accent1":  "#1A6DAF",     # blue  — primary bars
    "accent2":  "#C0392B",     # red   — errors / substitutions
    "accent3":  "#1E8449",     # green — correct / good
    "accent4":  "#7D3C98",     # purple — insertions
    "accent5":  "#D35400",     # orange — deletions
    "accent6":  "#2471A3",     # steel blue — secondary
    "text":     "#1A1A1A",     # near-black text
    "subtext":  "#555555",     # medium gray annotations
    "grid":     "#E0E0E0",     # light grid lines
    "spine":    "#AAAAAA",     # axis spine color
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE  — white background, print-ready
# ─────────────────────────────────────────────────────────────────────────────

def set_white_style():
    plt.rcParams.update({
        # Figure & axes background
        "figure.facecolor":      "white",
        "axes.facecolor":        "white",
        "savefig.facecolor":     "white",
        # Spines & ticks
        "axes.edgecolor":        PALETTE["spine"],
        "axes.linewidth":        0.8,
        "xtick.color":           PALETTE["text"],
        "ytick.color":           PALETTE["text"],
        "xtick.labelcolor":      PALETTE["text"],
        "ytick.labelcolor":      PALETTE["text"],
        "xtick.major.width":     0.8,
        "ytick.major.width":     0.8,
        # Labels & titles
        "axes.labelcolor":       PALETTE["text"],
        "axes.titlecolor":       PALETTE["text"],
        "text.color":            PALETTE["text"],
        "axes.titlesize":        12,
        "axes.titleweight":      "bold",
        "axes.labelsize":        10,
        "xtick.labelsize":       9,
        "ytick.labelsize":       9,
        # Grid
        "axes.grid":             True,
        "grid.color":            PALETTE["grid"],
        "grid.linestyle":        "--",
        "grid.linewidth":        0.6,
        "grid.alpha":            1.0,
        # Legend
        "legend.facecolor":      "white",
        "legend.edgecolor":      PALETTE["border"],
        "legend.fontsize":       9,
        # Font
        "font.family":           "DejaVu Sans",
        "figure.dpi":            150,
        "savefig.dpi":           200,
        "savefig.bbox":          "tight",
        # Remove top/right spines globally
        "axes.spines.top":       False,
        "axes.spines.right":     False,
    })

# ─────────────────────────────────────────────────────────────────────────────
# DEVANAGARI FONT SETUP  (unchanged logic, still works on white background)
# ─────────────────────────────────────────────────────────────────────────────

def _setup_devanagari_font():
    import glob, tempfile, urllib.request
    search_paths = []
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
    search_paths += [os.path.join(script_dir, "*.ttf")]

    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        search_paths += [os.path.join(windir, "Fonts", "*.ttf")]
    elif sys.platform == "darwin":
        search_paths += ["/Library/Fonts/*.ttf", "/System/Library/Fonts/*.ttf"]
    else:
        search_paths += [
            "/usr/share/fonts/truetype/noto/*.ttf",
            "/usr/share/fonts/**/*.ttf",
            os.path.expanduser("~/.fonts/*.ttf"),
        ]

    name_patterns = [
        "NotoSansDevanagari-Regular", "NotoSansDevanagari",
        "NotoSerifDevanagari-Regular", "NotoSerifDevanagari",
        "Mangal", "Aparajita", "Kokila", "Utsaah", "FreeSans",
    ]

    for pattern in search_paths:
        try:
            for fpath in glob.glob(pattern, recursive=True):
                fname = os.path.splitext(os.path.basename(fpath))[0]
                for np_ in name_patterns:
                    if np_.lower() in fname.lower():
                        return _fm.FontProperties(fname=fpath, size=10)
        except Exception:
            pass

    for np_ in name_patterns:
        matches = [f for f in _fm.fontManager.ttflist
                   if np_.lower() in f.name.lower()]
        if matches:
            chosen = matches[0]
            return _fm.FontProperties(fname=chosen.fname, size=10)

    # Try downloading Noto Sans Devanagari
    try:
        url  = ("https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
                "NotoSansDevanagari/NotoSansDevanagari-Regular.ttf")
        dest = os.path.join(tempfile.gettempdir(), "NotoSansDevanagari-Regular.ttf")
        if not os.path.exists(dest):
            print("  Downloading NotoSansDevanagari-Regular.ttf ...")
            urllib.request.urlretrieve(url, dest)
        if os.path.exists(dest):
            _fm.fontManager.addfont(dest)
            return _fm.FontProperties(fname=dest, size=10)
    except Exception as e:
        print(f"  [warn] Font download failed: {e}")

    print("  [warn] No Devanagari font found — tick labels may not render correctly")
    return None

DEVA_FONT = _setup_devanagari_font()

def apply_deva_font(ax, axis="both"):
    if DEVA_FONT is None:
        return
    if axis in ("x", "both"):
        for lbl in ax.get_xticklabels():
            lbl.set_fontproperties(DEVA_FONT)
    if axis in ("y", "both"):
        for lbl in ax.get_yticklabels():
            lbl.set_fontproperties(DEVA_FONT)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = f"{OUTPUT_DIR}/{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved > {path}")
    return path

def tokenize(text):
    return [w.strip("।.,!?\"'\u201c\u201d\u2018\u2019")
            for w in text.split()
            if w.strip("।.,!?\"'\u201c\u201d\u2018\u2019")]

def is_devanagari(ch):
    return '\u0900' <= ch <= '\u097F' or '\u0966' <= ch <= '\u096F'

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    refs  = [d["reference"]  for d in data]
    preds = [d["prediction"] for d in data]
    ids   = [d.get("id", i) for i, d in enumerate(data)]
    print(f"  Loaded {len(data):,} utterances")
    return ids, refs, preds

# ─────────────────────────────────────────────────────────────────────────────
# ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def levenshtein_ops(ref_tokens, hyp_tokens):
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i-1] == hyp_tokens[j-1]:
            ops.append(("equal",     ref_tokens[i-1], hyp_tokens[j-1])); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(("substitute",ref_tokens[i-1], hyp_tokens[j-1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(("delete",    ref_tokens[i-1], None));             i -= 1
        else:
            ops.append(("insert",    None,             hyp_tokens[j-1])); j -= 1
    ops.reverse()
    return dp[n][m], ops

def compute_alignments(refs, preds):
    all_ops = []
    for ref, pred in tqdm(zip(refs, preds), total=len(refs), desc="  Word-aligning"):
        _, ops = levenshtein_ops(tokenize(ref), tokenize(pred))
        all_ops.append(ops)
    return all_ops

def compute_char_alignments(refs, preds):
    all_ops = []
    for ref, pred in tqdm(zip(refs, preds), total=len(refs), desc="  Char-aligning"):
        _, ops = levenshtein_ops(list(ref.replace(" ", "")),
                                 list(pred.replace(" ", "")))
        all_ops.append(ops)
    return all_ops

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(refs, preds, all_ops):
    rows = []
    for ref, pred, ops in zip(refs, preds, all_ops):
        r_tok  = tokenize(ref)
        n_ref  = max(len(r_tok), 1)
        S = sum(1 for op in ops if op[0] == "substitute")
        D = sum(1 for op in ops if op[0] == "delete")
        I = sum(1 for op in ops if op[0] == "insert")
        C = sum(1 for op in ops if op[0] == "equal")

        r_chars = list(ref.replace(" ", ""))
        p_chars = list(pred.replace(" ", ""))
        n_chars = max(len(r_chars), 1)
        if HAS_EDITDISTANCE:
            char_dist = editdistance.eval(r_chars, p_chars)
        else:
            char_dist, _ = levenshtein_ops(r_chars, p_chars)
            char_dist    = sum(1 for op in _ if op[0] != "equal")

        rows.append({
            "ref_len": len(r_tok), "hyp_len": len(tokenize(pred)),
            "S": S, "D": D, "I": I, "C": C,
            "WER": (S + D + I) / n_ref,
            "CER": char_dist / n_chars,
            "n_ref_chars": len(r_chars),
        })
    return pd.DataFrame(rows)

def collect_word_confusions(all_ops):
    counter = collections.Counter()
    for ops in all_ops:
        for op, r, h in ops:
            if op == "substitute":
                counter[(r, h)] += 1
    return counter

def collect_char_confusions(char_ops_list):
    counter = collections.Counter()
    for ops in char_ops_list:
        for op, r, h in ops:
            if op == "substitute" and r and h:
                if DEVANAGARI_ONLY:
                    if is_devanagari(r) and is_devanagari(h):
                        counter[(r, h)] += 1
                else:
                    counter[(r, h)] += 1
    return counter

def collect_deletion_insertions(all_ops):
    del_c = collections.Counter()
    ins_c = collections.Counter()
    for ops in all_ops:
        for op, r, h in ops:
            if op == "delete"  and r: del_c[r] += 1
            if op == "insert"  and h: ins_c[h] += 1
    return del_c, ins_c

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES  — all white-background, print-ready
# ─────────────────────────────────────────────────────────────────────────────

# ── G1: Overall metrics ───────────────────────────────────────────────────────
def plot_overall_metrics(metrics_df):
    set_white_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("ASR Linguistic Analysis — Overall Metrics",
                 fontsize=13, fontweight="bold", y=1.03)

    total_S   = metrics_df["S"].sum()
    total_D   = metrics_df["D"].sum()
    total_I   = metrics_df["I"].sum()
    total_ref = metrics_df["ref_len"].sum()
    total_err = total_S + total_D + total_I
    global_wer = total_err / total_ref * 100
    global_cer = (metrics_df["CER"] * metrics_df["n_ref_chars"]).sum() \
                 / metrics_df["n_ref_chars"].sum() * 100

    # Panel 1 — WER / CER bars
    ax = axes[0]
    labels = ["WER", "CER"]
    vals   = [global_wer, global_cer]
    colors = [PALETTE["accent2"], PALETTE["accent5"]]
    bars   = ax.bar(labels, vals, color=colors, width=0.45,
                    edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{v:.2f}%", ha="center", va="bottom",
                color=PALETTE["text"], fontweight="bold", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_title("Global WER & CER")
    ax.set_ylabel("Error Rate (%)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    # Panel 2 — S/D/I pie
    ax = axes[1]
    sizes  = [total_S, total_D, total_I]
    clbls  = [f"Substitutions\n({total_S})",
               f"Deletions\n({total_D})",
               f"Insertions\n({total_I})"]
    ccolors = [PALETTE["accent2"], PALETTE["accent5"], PALETTE["accent4"]]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=clbls, colors=ccolors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"color": PALETTE["text"], "fontsize": 8.5},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(8.5)
        at.set_color("white")
    ax.set_title("Error Type Distribution")

    # Panel 3 — per-utterance WER histogram
    ax = axes[2]
    wer_vals = metrics_df["WER"].clip(0, 2) * 100
    ax.hist(wer_vals, bins=30, color=PALETTE["accent1"],
            edgecolor="white", linewidth=0.4, alpha=0.88)
    ax.axvline(wer_vals.mean(),   color=PALETTE["accent2"], lw=1.8,
               ls="--", label=f"Mean {wer_vals.mean():.1f}%")
    ax.axvline(wer_vals.median(), color=PALETTE["accent3"], lw=1.8,
               ls="--", label=f"Median {wer_vals.median():.1f}%")
    ax.set_xlabel("Per-utterance WER (%)")
    ax.set_ylabel("Count")
    ax.set_title("Utterance WER Distribution")
    ax.legend()
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    fig.tight_layout()
    return save(fig, "G1_overall_metrics")


# ── G2: Top-N character substitutions (horizontal bar) ───────────────────────
def plot_char_confusion(char_confusion):
    set_white_style()
    top = char_confusion.most_common(TOP_N_CONFUSION)
    if not top:
        print("  [skip G2] no character confusions")
        return None

    pairs  = [f"{r}  >  {h}" for (r, h), _ in top]
    counts = [c for _, c in top]

    fig, ax = plt.subplots(figsize=(11, max(6, len(pairs) * 0.32)))
    norm   = plt.Normalize(min(counts), max(counts))
    colors = plt.cm.Reds(norm(counts))

    bars = ax.barh(pairs[::-1], counts[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.4)
    for bar, cnt in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=8, color=PALETTE["subtext"])

    ax.set_xlabel("Frequency")
    ax.set_title(f"Top-{TOP_N_CONFUSION} Character Substitutions  (reference > hypothesis)",
                 pad=10)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    apply_deva_font(ax, "y")

    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Count", color=PALETTE["text"])
    cbar.ax.tick_params(labelcolor=PALETTE["text"])

    fig.tight_layout()
    return save(fig, "G2_char_confusion")


# ── G3: Character confusion matrix (heatmap) ─────────────────────────────────
def plot_char_confusion_matrix(char_confusion):
    set_white_style()
    char_counts = collections.Counter()
    for (r, h), cnt in char_confusion.items():
        char_counts[r] += cnt
        char_counts[h] += cnt
    top_chars = [c for c, _ in char_counts.most_common(18)]

    matrix = pd.DataFrame(0, index=top_chars, columns=top_chars)
    for (r, h), cnt in char_confusion.items():
        if r in top_chars and h in top_chars:
            matrix.loc[r, h] = cnt

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix, ax=ax,
        cmap="OrRd",
        linewidths=0.4, linecolor="#EEEEEE",
        annot=True, fmt="d",
        annot_kws={"size": 8, "color": PALETTE["text"]},
        cbar_kws={"label": "Substitution count", "shrink": 0.8},
    )
    ax.set_title("Character Confusion Matrix\n"
                 "(row = reference character, column = hypothesis character)",
                 pad=14)
    ax.set_xlabel("Predicted character")
    ax.set_ylabel("Reference character")
    ax.tick_params(labelsize=10, labelcolor=PALETTE["text"])
    apply_deva_font(ax, "both")

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(PALETTE["text"])
    cbar.ax.tick_params(colors=PALETTE["text"])

    fig.tight_layout()
    return save(fig, "G3_char_confusion_matrix")


# ── G4: Top word substitution pairs ──────────────────────────────────────────
def plot_word_confusions(word_confusion):
    set_white_style()
    top = word_confusion.most_common(TOP_N_WORDS)
    if not top:
        print("  [skip G4] no word confusions")
        return None

    pairs  = [f"{r}  >  {h}" for (r, h), _ in top]
    counts = [c for _, c in top]

    fig_h = max(7, len(pairs) * 0.38)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    norm   = plt.Normalize(min(counts), max(counts))
    colors = plt.cm.Blues(norm(counts))

    ax.barh(pairs[::-1], counts[::-1],
            color=colors[::-1], edgecolor="white", linewidth=0.4)
    for i, cnt in enumerate(counts[::-1]):
        ax.text(cnt + 0.05, i, str(cnt),
                va="center", fontsize=8, color=PALETTE["subtext"])

    ax.set_xlabel("Count")
    ax.set_title(f"Top-{TOP_N_WORDS} Word Substitution Pairs  (reference > hypothesis)",
                 pad=10)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    if DEVA_FONT:
        ax.set_yticklabels(pairs[::-1], fontproperties=DEVA_FONT)

    fig.tight_layout()
    return save(fig, "G4_word_confusions")


# ── G5: WER vs utterance length ───────────────────────────────────────────────
def plot_wer_vs_length(metrics_df):
    set_white_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("WER vs Utterance Length", fontsize=13, fontweight="bold")

    for ax, xcol, xlabel in zip(
        axes,
        ["ref_len",     "n_ref_chars"],
        ["Utterance length (words)", "Utterance length (characters)"],
    ):
        x = metrics_df[xcol]
        y = metrics_df["WER"].clip(0, 3) * 100

        sc = ax.scatter(x, y, alpha=0.45, c=y, cmap="RdYlGn_r",
                        edgecolors="none", s=22, zorder=3)
        z    = np.polyfit(x, y, 1)
        xlin = np.linspace(x.min(), x.max(), 200)
        ax.plot(xlin, np.polyval(z, xlin),
                color=PALETTE["accent1"], lw=1.8, ls="--",
                label=f"Trend (slope={z[0]:.2f})", zorder=4)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("WER (%)")
        ax.set_title(f"WER vs {xlabel}")
        ax.legend()
        ax.grid(True)
        ax.set_axisbelow(True)

        cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("WER (%)", color=PALETTE["text"])
        cbar.ax.tick_params(labelcolor=PALETTE["text"])

    fig.tight_layout()
    return save(fig, "G5_wer_vs_length")


# ── G6: Error position analysis ───────────────────────────────────────────────
def plot_error_positions(all_ops):
    set_white_style()

    zones      = ["Beginning\n(1st third)", "Middle\n(2nd third)", "End\n(3rd third)"]
    pos_errors = {z: 0 for z in zones}
    pos_total  = {z: 0 for z in zones}

    for ops in all_ops:
        n = max(len(ops), 1)
        for i, (op_type, r, h) in enumerate(ops):
            rel = i / n
            z   = zones[0] if rel < 0.33 else (zones[1] if rel < 0.67 else zones[2])
            pos_total[z]  += 1
            if op_type != "equal":
                pos_errors[z] += 1

    rates  = [pos_errors[z] / max(pos_total[z], 1) * 100 for z in zones]
    counts = [pos_errors[z] for z in zones]
    colors = [PALETTE["accent1"], PALETTE["accent5"], PALETTE["accent2"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Positional Error Analysis", fontsize=13, fontweight="bold")

    for ax, vals, ylabel, title in zip(
        axes,
        [rates,  counts],
        ["Error rate (%)", "Error count"],
        ["Error Rate by Word Position", "Absolute Error Count by Position"],
    ):
        bars = ax.bar(zones, vals, color=colors,
                      edgecolor="white", linewidth=0.8, width=0.5)
        for bar, v in zip(bars, vals):
            label = f"{v:.1f}%" if isinstance(v, float) else str(v)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.015,
                    label, ha="center", va="bottom",
                    color=PALETTE["text"], fontweight="bold", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(vals) * 1.18)

    fig.tight_layout()
    return save(fig, "G6_error_positions")


# ── G7: Deletions & insertions ───────────────────────────────────────────────
def plot_del_ins(del_counter, ins_counter):
    set_white_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Deletion & Insertion Analysis", fontsize=13, fontweight="bold")

    for ax, counter, title, cmap_name in zip(
        axes,
        [del_counter,             ins_counter],
        ["Most Frequently Deleted Words\n(in reference, missed by ASR)",
         "Most Frequently Inserted Words\n(in hypothesis, not in reference)"],
        ["Reds",                  "Purples"],
    ):
        top = counter.most_common(20)
        if not top:
            ax.text(0.5, 0.5, "None", ha="center", va="center",
                    transform=ax.transAxes, color=PALETTE["subtext"])
            ax.set_title(title)
            continue

        words  = [w for w, _ in top]
        counts = [c for _, c in top]
        norm   = plt.Normalize(min(counts), max(counts))
        colors = getattr(plt.cm, cmap_name)(norm(counts))

        ax.barh(words[::-1], counts[::-1],
                color=colors[::-1], edgecolor="white", linewidth=0.4)
        for i, cnt in enumerate(counts[::-1]):
            ax.text(cnt + 0.05, i, str(cnt),
                    va="center", fontsize=8, color=PALETTE["subtext"])
        ax.set_xlabel("Count")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        if DEVA_FONT:
            ax.set_yticklabels(words[::-1], fontproperties=DEVA_FONT)

    fig.tight_layout()
    return save(fig, "G7_deletions_insertions")


# ── G8: Phonetic / grapheme proximity of substitutions ───────────────────────
def plot_substitution_similarity(word_confusion):
    set_white_style()
    if not HAS_EDITDISTANCE:
        print("  [skip G8] editdistance not installed")
        return None

    distances, norm_distances = [], []
    for (ref_w, hyp_w), cnt in word_confusion.items():
        d       = editdistance.eval(list(ref_w), list(hyp_w))
        max_len = max(len(ref_w), len(hyp_w), 1)
        distances.extend([d] * cnt)
        norm_distances.extend([d / max_len] * cnt)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phonetic Proximity of Substitution Errors\n"
                 "(low edit distance = acoustically/graphemically similar characters)",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.hist(distances, bins=range(0, max(distances) + 2),
            color=PALETTE["accent1"], edgecolor="white", linewidth=0.4, alpha=0.88)
    ax.axvline(np.mean(distances), color=PALETTE["accent2"], lw=1.8,
               ls="--", label=f"Mean = {np.mean(distances):.2f}")
    ax.set_xlabel("Character edit distance between substituted pair")
    ax.set_ylabel("Count")
    ax.set_title("Edit Distance Distribution")
    ax.legend()
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.hist(norm_distances, bins=20,
            color=PALETTE["accent3"], edgecolor="white", linewidth=0.4, alpha=0.88)
    ax.axvline(np.mean(norm_distances), color=PALETTE["accent2"], lw=1.8,
               ls="--", label=f"Mean = {np.mean(norm_distances):.2f}")
    # shade "near-identical" zone
    ax.axvspan(0, 0.3, alpha=0.10, color=PALETTE["accent3"])
    ax.text(0.15, ax.get_ylim()[1] * 0.88, "Near-\nidentical",
            ha="center", fontsize=8, color=PALETTE["accent3"])
    ax.set_xlabel("Normalised edit distance  (0 = identical, 1 = fully different)")
    ax.set_ylabel("Count")
    ax.set_title("Normalised Edit Distance")
    ax.legend()
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    fig.tight_layout()
    return save(fig, "G8_substitution_similarity")


# ── G9: Devanagari character category breakdown ───────────────────────────────
def plot_script_analysis(char_confusion):
    set_white_style()

    MATRAS = set("ािीुूेैोौृ")
    NASALS = set("ंँणन")
    DIGITS = set("०१२३४५६७८९")
    NUKTA  = set("़")
    HALANT = set("्")

    def char_category(ch):
        if ch in MATRAS:      return "Vowel Matra\n(ा, ि, ी…)"
        if ch in NASALS:      return "Nasalisation\n(ं, ँ, ण)"
        if ch in DIGITS:      return "Digits"
        if ch in NUKTA:       return "Nukta (़)"
        if ch in HALANT:      return "Halant (्)"
        if is_devanagari(ch): return "Consonant"
        return "Other"

    cat_counts = collections.Counter()
    for (r, h), cnt in char_confusion.items():
        cat_counts[char_category(r)] += cnt

    order  = sorted(cat_counts.items(), key=lambda x: -x[1])
    cats   = [x[0] for x in order]
    counts = [x[1] for x in order]

    palette_list = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
                    PALETTE["accent4"], PALETTE["accent5"], PALETTE["subtext"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cats, counts,
                  color=[palette_list[i % len(palette_list)] for i in range(len(cats))],
                  edgecolor="white", linewidth=0.8, width=0.55)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                str(v), ha="center", va="bottom",
                color=PALETTE["text"], fontweight="bold", fontsize=10)
    ax.set_ylabel("Error count")
    ax.set_title("Substitution Errors by Devanagari Character Category", pad=12)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(counts) * 1.18)

    fig.tight_layout()
    return save(fig, "G9_script_category_errors")


# ── G10: Per-utterance error severity ────────────────────────────────────────
def plot_error_severity(metrics_df):
    set_white_style()

    BANDS = [
        ("Perfect (0%)",      0.00, 0.001, "#1E8449"),
        ("Excellent (<10%)",  0.001, 0.10, "#52BE80"),
        ("Good (10-30%)",     0.10,  0.30, "#2471A3"),
        ("Fair (30-50%)",     0.30,  0.50, "#D35400"),
        ("Poor (50-100%)",    0.50,  1.00, "#C0392B"),
        ("Very Poor (>100%)", 1.00,  9.99, "#922B21"),
    ]

    def severity(w):
        for label, lo, hi, _ in BANDS:
            if lo <= w < hi:
                return label
        return "Very Poor (>100%)"

    metrics_df = metrics_df.copy()
    metrics_df["severity"] = metrics_df["WER"].apply(severity)
    band_labels = [b[0] for b in BANDS]
    band_colors = {b[0]: b[3] for b in BANDS}

    counts = metrics_df["severity"].value_counts()
    cats   = [b for b in band_labels if b in counts.index]
    vals   = [counts[c] for c in cats]
    pcts   = [v / len(metrics_df) * 100 for v in vals]
    clrs   = [band_colors[c] for c in cats]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Error Severity Analysis", fontsize=13, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(cats, vals, color=clrs,
                  edgecolor="white", linewidth=0.8, width=0.55)
    for bar, v, p in zip(bars, vals, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{v}\n({p:.1f}%)", ha="center", va="bottom",
                color=PALETTE["text"], fontsize=8, fontweight="bold")
    ax.set_ylabel("Number of utterances")
    ax.set_title("Utterance Error Severity Distribution")
    ax.set_xticklabels(cats, rotation=18, ha="right", fontsize=8.5)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        vals, labels=cats, colors=clrs,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"color": PALETTE["text"], "fontsize": 8.5},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_color("white")
        at.set_fontsize(8)
    ax.set_title("Severity Breakdown")

    fig.tight_layout()
    return save(fig, "G10_error_severity")


# ── G11: CER vs WER scatter ───────────────────────────────────────────────────
def plot_cer_vs_wer(metrics_df):
    set_white_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    x = metrics_df["CER"].clip(0, 2) * 100
    y = metrics_df["WER"].clip(0, 2) * 100

    sc = ax.scatter(x, y, alpha=0.45, c=y, cmap="RdYlGn_r",
                    edgecolors="none", s=28, zorder=3)
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=PALETTE["spine"], lw=1.2,
            ls="--", label="WER = CER", zorder=2)

    ax.set_xlabel("CER per utterance (%)")
    ax.set_ylabel("WER per utterance (%)")
    ax.set_title("CER vs WER per Utterance\n"
                 "(above diagonal: more word errors than character errors)")
    ax.legend()
    ax.grid(True)
    ax.set_axisbelow(True)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("WER (%)", color=PALETTE["text"])
    cbar.ax.tick_params(labelcolor=PALETTE["text"])

    fig.tight_layout()
    return save(fig, "G11_cer_vs_wer")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_summary_report(metrics_df, word_confusion, char_confusion,
                         del_counter, ins_counter):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_S   = metrics_df["S"].sum()
    total_D   = metrics_df["D"].sum()
    total_I   = metrics_df["I"].sum()
    total_ref = metrics_df["ref_len"].sum()
    total_err = total_S + total_D + total_I
    global_wer = total_err / total_ref * 100
    global_cer = (metrics_df["CER"] * metrics_df["n_ref_chars"]).sum() \
                 / metrics_df["n_ref_chars"].sum() * 100

    lines = [
        "=" * 65,
        "  ASR LINGUISTIC ANALYSIS — SUMMARY REPORT",
        "=" * 65,
        "",
        f"  Utterances analysed        : {len(metrics_df):,}",
        f"  Total reference words      : {total_ref:,}",
        "",
        "  ── Global Metrics ──────────────────────────────────",
        f"  Word Error Rate  (WER)     : {global_wer:.2f}%",
        f"  Char Error Rate  (CER)     : {global_cer:.2f}%",
        f"  Mean utterance WER         : {metrics_df['WER'].mean()*100:.2f}%",
        f"  Median utterance WER       : {metrics_df['WER'].median()*100:.2f}%",
        "",
        "  ── Error Breakdown ─────────────────────────────────",
        f"  Substitutions              : {total_S:>6}  ({total_S/total_err*100:.1f}%)",
        f"  Deletions                  : {total_D:>6}  ({total_D/total_err*100:.1f}%)",
        f"  Insertions                 : {total_I:>6}  ({total_I/total_err*100:.1f}%)",
        f"  Total errors               : {total_err:,}",
        "",
        "  ── Top-10 Word Substitutions ───────────────────────",
    ]
    for (r, h), cnt in word_confusion.most_common(10):
        lines.append(f"    {r:<20} > {h:<20}  x{cnt}")

    lines += ["", "  ── Top-10 Character Substitutions ─────────────────"]
    for (r, h), cnt in char_confusion.most_common(10):
        lines.append(f"    {r}  >  {h}   x{cnt}")

    lines += ["", "  ── Top-10 Deleted Words ────────────────────────────"]
    for w, cnt in del_counter.most_common(10):
        lines.append(f"    {w:<30}  x{cnt}")

    lines += ["", "  ── Top-10 Inserted Words ───────────────────────────"]
    for w, cnt in ins_counter.most_common(10):
        lines.append(f"    {w:<30}  x{cnt}")

    lines += ["", "=" * 65]
    report = "\n".join(lines)
    print("\n" + report)

    rpath = f"{OUTPUT_DIR}/summary_report.txt"
    Path(rpath).write_text(report, encoding="utf-8")
    print(f"\n  Report saved > {rpath}")

    metrics_df.to_csv(f"{OUTPUT_DIR}/per_utterance_metrics.csv", index=False)
    pd.DataFrame(
        [(r, h, c) for (r, h), c in word_confusion.most_common()],
        columns=["reference_word", "hypothesis_word", "count"]
    ).to_csv(f"{OUTPUT_DIR}/word_confusions.csv", index=False)
    pd.DataFrame(
        [(r, h, c) for (r, h), c in char_confusion.most_common()],
        columns=["reference_char", "hypothesis_char", "count"]
    ).to_csv(f"{OUTPUT_DIR}/char_confusions.csv", index=False)
    print(f"  CSVs saved > {OUTPUT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="ASR Linguistic Analysis Pipeline (white/academic style)")
    parser.add_argument("--input",      default=INPUT_JSON,
                        help="Path to ASR output JSON")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help="Directory for all outputs")
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    print("  ASR Linguistic Analysis Pipeline")
    print(f"{'='*55}\n")

    print("[1/6] Loading data ...")
    ids, refs, preds = load_data(args.input)

    print("[2/6] Computing alignments ...")
    all_ops  = compute_alignments(refs, preds)
    char_ops = compute_char_alignments(refs, preds)

    print("[3/6] Computing per-utterance metrics ...")
    metrics_df = compute_metrics(refs, preds, all_ops)

    print("[4/6] Collecting confusion pairs ...")
    word_confusion           = collect_word_confusions(all_ops)
    char_confusion           = collect_char_confusions(char_ops)
    del_counter, ins_counter = collect_deletion_insertions(all_ops)

    print("[5/6] Generating figures ...")
    plot_overall_metrics(metrics_df)
    plot_char_confusion(char_confusion)
    plot_char_confusion_matrix(char_confusion)
    plot_word_confusions(word_confusion)
    plot_wer_vs_length(metrics_df)
    plot_error_positions(all_ops)
    plot_del_ins(del_counter, ins_counter)
    plot_substitution_similarity(word_confusion)
    plot_script_analysis(char_confusion)
    plot_error_severity(metrics_df)
    plot_cer_vs_wer(metrics_df)

    print("[6/6] Writing summary report ...")
    write_summary_report(metrics_df, word_confusion, char_confusion,
                         del_counter, ins_counter)

    print(f"\n  All outputs saved to: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()