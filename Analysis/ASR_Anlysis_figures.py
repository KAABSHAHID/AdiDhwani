"""
ASR Benchmarking: Complete Data Analysis
Covers: Scaling Laws, Data Efficiency, Synthetic Data Quality,
        Alpha Sensitivity, Transfer Learning, Zero-Shot vs Finetune
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ─── Global Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

# ─── Color Palette ────────────────────────────────────────────────────────────
C = {
    'bhili':    '#2563EB',   # blue
    'mundari':  '#DC2626',   # red
    'hindi':    '#16A34A',   # green
    'wav2vec':  '#7C3AED',   # purple
    'whisper':  '#D97706',   # amber
    'syn':      '#0891B2',   # cyan
    'real':     '#BE185D',   # pink
    'lm':       '#059669',   # emerald
    'no_lm':    '#9CA3AF',   # gray
    'transfer': '#F97316',   # orange
    'neutral':  '#6B7280',
}

LANGS = {
    'bhili':   {'hours': 55, 'color': C['bhili'],   'marker': 'o'},
    'mundari': {'hours': 11, 'color': C['mundari'],  'marker': 's'},
}

OUT = '/figures'

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

# ── Zero-Shot Baselines ──────────────────────────────────────────────────────
zero_shot = pd.DataFrame({
    'model':   ['SeamlessM4T', 'MMS-1B', 'Whisper-large-v3', 'SpeechT5'],
    'bhili':   [86.84, 1.01, 74.48, 1.24],
    'mundari': [94.77, 1.06, 89.62, 1.17],
})

# ── Staged Finetuning: Bhili = 55 hours ──────────────────────────────────────
alphas = [0.2, 0.4, 0.6, 0.8, 1.0]

staged_bhili = {
    'Wav2vec2 xlsr-300m': [53.72, 47.99, 46.62, 48.22, 46.53],
    'Wav2vec2 xlsr-1B':   [49.73, 45.71, 44.06, 46.47, 47.68],
    'Whisper Tiny 39M':   [60.84, 56.87, 55.29, 51.90, 51.36],
    'Whisper Base 74M':   [56.41, 52.94, 50.94, 48.75, 46.12],
    'Whisper Small 244M': [51.22, 46.84, 43.86, 42.17, 39.55],
    'Whisper Medium 769M':[47.94, 44.76, 41.21, 38.37, 36.62],
}

# ── Staged Finetuning: Mundari = 11 hours ────────────────────────────────────
staged_mundari = {
    'Wav2vec2 xlsr-300m': [75.48, 66.17, 67.00, 73.03, 72.09],
    'Wav2vec2 xlsr-1B':   [62.01, 64.37, 64.49, 71.20, 68.24],
    'Whisper Tiny 39M':   [73.49, 68.76, 67.44, 65.06, 65.12],
    'Whisper Base 74M':   [69.25, 65.10, 62.36, 60.91, 61.63],
    'Whisper Small 244M': [68.25, 62.55, 59.71, 57.89, 57.09],
    'Whisper Medium 769M':[63.88, 58.09, 55.19, 54.68, 56.93],
}

# ── Best WER per model (alpha=optimal) ──────────────────────────────────────
model_params = {
    'Wav2vec2 xlsr-300m': 300,
    'Wav2vec2 xlsr-1B':   1000,
    'Whisper Tiny 39M':   39,
    'Whisper Base 74M':   74,
    'Whisper Small 244M': 244,
    'Whisper Medium 769M':769,
}

bhili_best   = {m: min(v) for m, v in staged_bhili.items()}
mundari_best = {m: min(v) for m, v in staged_mundari.items()}

# ── Transfer Learning Results ─────────────────────────────────────────────────
# Bhili: Wav2vec2-1B, Whisper-Medium
transfer_bhili = {
    'Wav2vec2 xlsr-1B': {
        'staged_best': 44.06,
        'Hindi':       46.77,
        'Gujarati':    None,
        'Marathi':     None,
        'Syn':         72.8,
        'Syn->Org':    39.0,
        'No LM':       32.91, 'LM': 22.32,
        'No LM CER':   10.35, 'LM CER': 8.83,
    },
    'Wav2vec2 xlsr-300m': {
        'staged_best': 46.53,
        'Hindi':       45.33,
        'Gujarati':    56.82,
        'Marathi':     57.91,
        'Syn':         None,
        'Syn->Org':    None,
        'No LM': None, 'LM': None,
    },
    'Whisper Medium 769M': {
        'staged_best': 36.62,
        'Hindi':       42.03,
        'Gujarati':    None,
        'Marathi':     None,
        'Syn':         74.08,
        'Syn->Org':    36.06,
        'No LM': None, 'LM': None,
    },
}

# Mundari: similar
transfer_mundari = {
    'Wav2vec2 xlsr-1B': {
        'staged_best': 62.01,
        'Hindi':       65.20,
        'Santali':     None,
        'Bengali':     None,
        'Syn':         82.87,
        'Syn->Org':    65.18,
        'No LM':       37.52, 'LM': 42.74,
        'No LM CER':   17.91, 'LM CER': 20.17,
    },
    'Wav2vec2 xlsr-300m': {
        'staged_best': 66.17,
        'Hindi':       64.13,
        'Santali':     77.81,
        'Bengali':     74.24,
        'Syn':         None,
        'Syn->Org':    None,
        'No LM': None, 'LM': None,
    },
    'Whisper Medium 769M': {
        'staged_best': 54.68,
        'Hindi':       57.44,
        'Santali':     None,
        'Bengali':     None,
        'Syn':         79.5,
        'Syn->Org':    53.21,
        'No LM': None, 'LM': None,
    },
}

# ── Hindi 100-hour training (WER) ─────────────────────────────────────────────
hindi_training = {
    'Wav2vec2 300m': 29.78,
    'Wav2vec2 1B':   37.85,   # inconsistent
    'Whisper Tiny':  34.43,
    'Whisper Base':  29.01,
    'Whisper Small': 22.78,
    'Whisper Medium':22.63,
}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Zero-Shot Baselines vs Finetuned Best
# ══════════════════════════════════════════════════════════════════════════════
def fig1_zeroshot_vs_finetuned():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Figure 1: Zero-Shot Baselines vs. Best Finetuned WER', fontsize=14, fontweight='bold', y=1.01)

    for ax, lang, best_dict, color, zs_col in zip(
        axes,
        ['Bhili (55 hrs)', 'Mundari (11 hrs)'],
        [bhili_best, mundari_best],
        [C['bhili'], C['mundari']],
        ['bhili', 'mundari'],
    ):
        models_zs = zero_shot['model'].tolist()
        wer_zs    = zero_shot[zs_col].tolist()

        # bars: zero-shot
        x = np.arange(len(models_zs))
        bars = ax.bar(x - 0.2, wer_zs, width=0.35, color='#CBD5E1', label='Zero-shot', zorder=3)

        # best finetuned (single value – best across all models)
        best_val = min(best_dict.values())
        best_model = min(best_dict, key=best_dict.get)
        ax.axhline(best_val, color=color, lw=2, ls='--', zorder=4,
                   label=f'Best finetuned ({best_model.split()[0]}-{best_model.split()[-1]}): {best_val:.1f}')

        # annotate zero-shot bars
        for b, val in zip(bars, wer_zs):
            label_text = f'{val:.1f}' if val > 2 else f'{val:.2f}†'
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                    label_text, ha='center', va='bottom', fontsize=9, color='#374151')

        ax.set_xticks(x - 0.2)
        ax.set_xticklabels(models_zs, rotation=15, ha='right')
        ax.set_ylabel('WER (%) — lower is better')
        ax.set_title(lang)
        ax.set_ylim(0, 105)
        ax.legend(framealpha=0.9)

        # footnote for near-zero values
        ax.text(0.01, 0.10, '† Near-zero WER = model collapse (empty/garbage output)',
                transform=ax.transAxes, fontsize=8, color='#6B7280', style='italic')

    plt.tight_layout()
    plt.savefig('fig1_zeroshot_vs_finetuned.png')
    plt.close()
    print('✓ Fig 1 saved')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Scaling Law Curves (hours proxy via alpha mix)
# ══════════════════════════════════════════════════════════════════════════════
def power_law(x, a, b):
    return a * np.power(x, -b)

def fig2_scaling_laws():
    """
    We approximate effective target-language hours from alpha:
       effective_hours = alpha * total_target_hours
    Bhili total = 55h, Mundari total = 11h
    Then fit a power-law curve: WER = a * hours^(-b)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Figure 2: Data Scaling Laws — WER vs Effective Target-Language Hours', fontsize=14, fontweight='bold', y=1.01)

    configs = [
        ('Bhili', staged_bhili, 55, C['bhili'], axes[0]),
        ('Mundari', staged_mundari, 11, C['mundari'], axes[1]),
    ]

    model_colors = {
        'Wav2vec2 xlsr-300m':  '#7C3AED',
        'Wav2vec2 xlsr-1B':    '#4F46E5',
        'Whisper Tiny 39M':    '#D97706',
        'Whisper Base 74M':    '#B45309',
        'Whisper Small 244M':  '#92400E',
        'Whisper Medium 769M': '#78350F',
    }
    model_markers = {
        'Wav2vec2 xlsr-300m': 'o', 'Wav2vec2 xlsr-1B': 's',
        'Whisper Tiny 39M': '^', 'Whisper Base 74M': 'D',
        'Whisper Small 244M': 'v', 'Whisper Medium 769M': '*',
    }

    for lang_name, staged_dict, total_hours, lang_color, ax in configs:
        eff_hours = [a * total_hours for a in alphas]

        for model, wers in staged_dict.items():
            ax.plot(eff_hours, wers, marker=model_markers[model], color=model_colors[model],
                    label=model.replace(' xlsr', '').replace('Whisper ', 'W-'), lw=1.5,
                    ms=6, alpha=0.85)

        # Fit power law on best model (Whisper Medium)
        best_model = 'Whisper Medium 769M'
        y_fit = np.array(staged_dict[best_model])
        x_fit = np.array(eff_hours)
        try:
            popt, _ = curve_fit(power_law, x_fit, y_fit, p0=[100, 0.3], maxfev=5000)
            x_smooth = np.linspace(x_fit.min(), x_fit.max(), 200)
            y_smooth = power_law(x_smooth, *popt)
            ax.plot(x_smooth, y_smooth, '--', color=lang_color, lw=2, alpha=0.6,
                    label=f'Power-law fit (Whisper-Med)\nWER∝h^(-{popt[1]:.2f})')
        except Exception:
            pass

        ax.set_xlabel('Effective target-language hours (α × total)')
        ax.set_ylabel('WER (%) — lower is better')
        ax.set_title(f'{lang_name} ({total_hours} hrs total)')
        ax.legend(fontsize=8, ncol=2, framealpha=0.85)
        ax.set_xlim(left=0)

        # Annotate saturation zone
        best_wer = min([min(v) for v in staged_dict.values()])
        ax.axhline(best_wer, color='#9CA3AF', lw=1, ls=':', alpha=0.7)
        ax.text(eff_hours[-1]*0.6, best_wer - 1.5, f'Best WER: {best_wer:.1f}%',
                fontsize=8, color='#6B7280', style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig2_scaling_laws.png')
    plt.close()
    print('✓ Fig 2 saved')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Alpha Sensitivity Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig3_alpha_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Figure 3: Alpha (Mix Ratio) Sensitivity — WER Heatmap', fontsize=14, fontweight='bold', y=1.01)

    models_order = list(staged_bhili.keys())

    for ax, staged_dict, lang_name, cmap in zip(
        axes,
        [staged_bhili, staged_mundari],
        ['Bhili (55 hrs)', 'Mundari (11 hrs)'],
        ['Blues_r', 'Reds_r'],
    ):
        data = np.array([staged_dict[m] for m in models_order])
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=data.min()-2, vmax=data.max()+2)

        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f'α={a}' for a in alphas])
        ax.set_yticks(range(len(models_order)))
        ax.set_yticklabels([m.replace(' xlsr', '').replace('Whisper ', 'W-') for m in models_order], fontsize=9)
        ax.set_xlabel('Alpha (target-language fraction in mix)')
        ax.set_title(lang_name)

        # Annotate cells
        for i in range(len(models_order)):
            for j in range(len(alphas)):
                val = data[i, j]
                # highlight minimum per row
                is_min = (val == data[i].min())
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=9, fontweight='bold' if is_min else 'normal',
                        color='white' if is_min else '#1F2937')
                if is_min:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                               fill=False, edgecolor='#FBBF24', lw=2))

        plt.colorbar(im, ax=ax, label='WER (%)', shrink=0.8)

    fig.text(0.5, -0.02, '★ Yellow border = optimal alpha per model', ha='center', fontsize=9, color='#6B7280', style='italic')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig3_alpha_heatmap.png')
    plt.close()
    print('✓ Fig 3 saved')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Model Size vs WER (Pareto Frontier)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_pareto():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Figure 4: Model Parameters vs Best WER — Pareto Efficiency', fontsize=14, fontweight='bold', y=1.01)

    for ax, best_dict, lang_name, lang_color in zip(
        axes,
        [bhili_best, mundari_best],
        ['Bhili (55 hrs)', 'Mundari (11 hrs)'],
        [C['bhili'], C['mundari']],
    ):
        params = [model_params[m] for m in best_dict]
        wers   = list(best_dict.values())
        labels = list(best_dict.keys())

        arch_colors = [C['wav2vec'] if 'Wav2vec' in m else C['whisper'] for m in labels]
        arch_markers= ['s' if 'Wav2vec' in m else 'o' for m in labels]

        for p, w, lbl, col, mrk in zip(params, wers, labels, arch_colors, arch_markers):
            ax.scatter(p, w, s=120, color=col, marker=mrk, zorder=5, edgecolors='white', lw=1.5)
            short = lbl.replace('Wav2vec2 xlsr-', 'W2V-').replace('Whisper ', 'W-')
            ax.annotate(short, (p, w), textcoords='offset points', xytext=(6, 4), fontsize=8.5)

        # Pareto frontier (lower-left dominant points)
        sorted_pts = sorted(zip(params, wers), key=lambda x: x[0])
        pareto_x, pareto_y = [], []
        min_wer = float('inf')
        for px, py in sorted_pts:
            if py < min_wer:
                pareto_x.append(px)
                pareto_y.append(py)
                min_wer = py
        if len(pareto_x) > 1:
            ax.plot(pareto_x, pareto_y, '--', color=lang_color, lw=1.5, alpha=0.5, label='Pareto frontier')

        # Log scale x
        ax.set_xscale('log')
        ax.set_xlabel('Model parameters (M) — log scale')
        ax.set_ylabel('Best WER (%) — lower is better')
        ax.set_title(lang_name)

        wav_patch   = mpatches.Patch(color=C['wav2vec'], label='Wav2vec2 (CTC)')
        whis_patch  = mpatches.Patch(color=C['whisper'], label='Whisper (Seq2Seq)')
        ax.legend(handles=[wav_patch, whis_patch], framealpha=0.9)

        # Efficiency annotation
        best_idx = np.argmin(wers)
        ax.annotate(f'Best: {wers[best_idx]:.1f}%\n{labels[best_idx].split()[-1]}',
                    xy=(params[best_idx], wers[best_idx]),
                    xytext=(params[best_idx]*0.4, wers[best_idx]+5),
                    arrowprops=dict(arrowstyle='->', color='#374151', lw=1),
                    fontsize=8.5, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#F9FAFB', ec='#D1D5DB'))

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig4_pareto.png')
    plt.close()
    print('✓ Fig 4 saved')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Synthetic Data Analysis: ROI & Stage Comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig5_synthetic():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Figure 5: Synthetic Data Analysis — ROI and Training Stages', fontsize=14, fontweight='bold', y=1.01)

    # ─ Left: WER across training stages (Bhili, Wav2vec2-1B) ─────────────────
    ax = axes[0]
    stages_bhili = {
        'Zero-shot\n(SeamlessM4T)':   86.84,
        'Staged FT\n(best α)':         44.06,
        'Synthetic\nonly':             72.80,
        'Syn → Org\nFT':               39.00,
        'Syn→Org\n+ LM':              22.32,
    }
    stages_mundari = {
        'Zero-shot\n(SeamlessM4T)':   94.77,
        'Staged FT\n(best α)':         62.01,
        'Synthetic\nonly':             82.87,
        'Syn → Org\nFT':               65.18,
        'Syn→Org\n+ LM':              42.74,
    }

    x = np.arange(len(stages_bhili))
    bvals = list(stages_bhili.values())
    mvals = list(stages_mundari.values())
    labs  = list(stages_bhili.keys())

    b1 = ax.bar(x - 0.2, bvals, 0.35, color=C['bhili'],   alpha=0.85, label='Bhili')
    b2 = ax.bar(x + 0.2, mvals, 0.35, color=C['mundari'], alpha=0.85, label='Mundari')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=8.5)

    # Draw improvement arrows for Bhili
    for i in range(1, len(bvals)):
        if bvals[i] < bvals[i-1]:
            drop = bvals[i-1] - bvals[i]
            ax.annotate('', xy=(i-0.2, bvals[i]+1), xytext=(i-1-0.2, bvals[i-1]-1),
                        arrowprops=dict(arrowstyle='->', color=C['bhili'], lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel('WER (%) — lower is better')
    ax.set_title('WER at Each Training Stage (Wav2vec2 xlsr-1B)')
    ax.legend()
    ax.set_ylim(0, 105)

    # ─ Right: Synthetic-to-Real exchange rate ─────────────────────────────────
    ax2 = axes[1]

    # Real hours: Bhili=55, Mundari=11
    # Synthetic hours used: Bhili=72.8h generated, Mundari=82.87h
    # WER improvement from syn→org over staged:
    # Bhili:  staged_best=44.06, syn->org=39.0  → improvement=5.06 WER points
    # Mundari: staged_best=62.01, syn->org=65.18 → WORSE (no benefit)
    data_roi = {
        'Bhili': {
            'real_hrs': 55, 'syn_hrs': 72.8,
            'staged_wer': 44.06, 'synorg_wer': 39.0,
            'improvement': 44.06 - 39.0,
        },
        'Mundari': {
            'real_hrs': 11, 'syn_hrs': 82.87,
            'staged_wer': 62.01, 'synorg_wer': 65.18,
            'improvement': 62.01 - 65.18,  # negative!
        },
    }

    categories = ['Staged FT (real)', 'Syn-only', 'Syn→Org FT']
    bhili_wers   = [44.06, 72.80, 39.00]
    mundari_wers = [62.01, 82.87, 65.18]

    x3 = np.arange(len(categories))
    ax2.plot(x3, bhili_wers,   'o-', color=C['bhili'],   lw=2, ms=8, label='Bhili')
    ax2.plot(x3, mundari_wers, 's-', color=C['mundari'],  lw=2, ms=8, label='Mundari')

    ax2.fill_between(x3, bhili_wers,   alpha=0.1, color=C['bhili'])
    ax2.fill_between(x3, mundari_wers, alpha=0.1, color=C['mundari'])

    # Annotations
    for xi, (bw, mw, lbl) in enumerate(zip(bhili_wers, mundari_wers, categories)):
        ax2.text(xi, bw - 2.5, f'{bw:.1f}', ha='center', fontsize=9, color=C['bhili'], fontweight='bold')
        ax2.text(xi, mw + 1.5, f'{mw:.1f}', ha='center', fontsize=9, color=C['mundari'], fontweight='bold')

    # Highlight: syn-only is WORSE than staged for both
    ax2.annotate('Syn-only\nworse than\nstaged FT!', xy=(1, 72.8), xytext=(1.3, 80),
                 arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1),
                 fontsize=8.5, color='#DC2626',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#FEF2F2', ec='#FCA5A5'))

    ax2.set_xticks(x3)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel('WER (%) — lower is better')
    ax2.set_title('Synthetic Data: Real vs Syn Comparison')
    ax2.legend()
    ax2.set_ylim(30, 92)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig5_synthetic.png')
    plt.close()
    print('✓ Fig 5 saved')

def fig11_synthetic():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Figure 5: Synthetic Data Analysis — ROI and Training Stages', fontsize=14, fontweight='bold', y=1.01)

    # ─ Left: WER across training stages (Bhili, whisper-medium) ─────────────────
    ax = axes[0]
    stages_bhili = {
        'Zero-shot\n(SeamlessM4T)':   86.84,
        'Staged FT\n(best α)':         36.62,
        'Synthetic\nonly':             74.08,
        'Syn → Org\nFT':               36.06,
    }
    stages_mundari = {
        'Zero-shot\n(SeamlessM4T)':   94.77,
        'Staged FT\n(best α)':         54.68,
        'Synthetic\nonly':             79.5,
        'Syn → Org\nFT':               53.21,
    }

    x = np.arange(len(stages_bhili))
    bvals = list(stages_bhili.values())
    mvals = list(stages_mundari.values())
    labs  = list(stages_bhili.keys())

    b1 = ax.bar(x - 0.2, bvals, 0.35, color=C['bhili'],   alpha=0.85, label='Bhili')
    b2 = ax.bar(x + 0.2, mvals, 0.35, color=C['mundari'], alpha=0.85, label='Mundari')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=8.5)

    # Draw improvement arrows for Bhili
    for i in range(1, len(bvals)):
        if bvals[i] < bvals[i-1]:
            drop = bvals[i-1] - bvals[i]
            ax.annotate('', xy=(i-0.2, bvals[i]+1), xytext=(i-1-0.2, bvals[i-1]-1),
                        arrowprops=dict(arrowstyle='->', color=C['bhili'], lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel('WER (%) — lower is better')
    ax.set_title('WER at Each Training Stage (whisper-medium)')
    ax.legend()
    ax.set_ylim(0, 105)

    # ─ Right: Synthetic-to-Real exchange rate ─────────────────────────────────
    ax2 = axes[1]

    # Real hours: Bhili=55, Mundari=11
    # Synthetic hours used: Bhili=72.8h generated, Mundari=82.87h
    # WER improvement from syn→org over staged:
    # Bhili:  staged_best=44.06, syn->org=39.0  → improvement=5.06 WER points
    # Mundari: staged_best=62.01, syn->org=65.18 → WORSE (no benefit)
    data_roi = {
        'Bhili': {
            'real_hrs': 55, 'syn_hrs': 72.8,
            'staged_wer': 44.06, 'synorg_wer': 39.0,
            'improvement': 44.06 - 39.0,
        },
        'Mundari': {
            'real_hrs': 11, 'syn_hrs': 82.87,
            'staged_wer': 62.01, 'synorg_wer': 65.18,
            'improvement': 62.01 - 65.18,  # negative!
        },
    }

    categories = ['Staged FT (real)', 'Syn-only', 'Syn→Org FT']
    bhili_wers   = [36.62, 74.08, 36.06]
    mundari_wers = [54.68, 79.5, 53.21]

    x3 = np.arange(len(categories))
    ax2.plot(x3, bhili_wers,   'o-', color=C['bhili'],   lw=2, ms=8, label='Bhili')
    ax2.plot(x3, mundari_wers, 's-', color=C['mundari'],  lw=2, ms=8, label='Mundari')

    ax2.fill_between(x3, bhili_wers,   alpha=0.1, color=C['bhili'])
    ax2.fill_between(x3, mundari_wers, alpha=0.1, color=C['mundari'])

    # Annotations
    for xi, (bw, mw, lbl) in enumerate(zip(bhili_wers, mundari_wers, categories)):
        ax2.text(xi, bw - 2.5, f'{bw:.1f}', ha='center', fontsize=9, color=C['bhili'], fontweight='bold')
        ax2.text(xi, mw + 1.5, f'{mw:.1f}', ha='center', fontsize=9, color=C['mundari'], fontweight='bold')

    # Highlight: syn-only is WORSE than staged for both
    ax2.annotate('Syn-only\nworse than\nstaged FT!', xy=(1, 72.8), xytext=(1.3, 80),
                 arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1),
                 fontsize=8.5, color='#DC2626',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#FEF2F2', ec='#FCA5A5'))

    ax2.set_xticks(x3)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel('WER (%) — lower is better')
    ax2.set_title('Synthetic Data: Real vs Syn Comparison')
    ax2.legend()
    ax2.set_ylim(30, 92)

    plt.tight_layout()
    plt.savefig('fig11_synthetic.png')
    plt.close()
    print('✓ Fig 11 saved')
# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Transfer Learning: Donor Language Comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig6_transfer():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Figure 6: Transfer Learning — Donor Language Impact', fontsize=14, fontweight='bold', y=1.01)

    # Bhili: Wav2vec2-300m has all 3 donors
    ax = axes[0]
    donors_bhili = {
        'Staged\n(no transfer)': 46.53,
        'Hindi\ntransfer':       45.33,
        'Gujarati\ntransfer':    56.82,
        'Marathi\ntransfer':     57.91,
    }
    bar_colors_b = ['#9CA3AF', '#16A34A', '#F97316', '#DC2626']
    bars = ax.bar(donors_bhili.keys(), donors_bhili.values(), color=bar_colors_b, alpha=0.85, width=0.55)
    for bar, val in zip(bars, donors_bhili.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Highlight best donor
    ax.add_patch(plt.Rectangle(
        (bars[1].get_x()-0.05, 0), bars[1].get_width()+0.1, donors_bhili['Hindi\ntransfer']+3,
        fill=False, edgecolor='#16A34A', lw=2.5, linestyle='--'
    ))
    ax.text(bars[1].get_x() + bars[1].get_width()/2, donors_bhili['Hindi\ntransfer']+5,
            '✓ Best', ha='center', fontsize=9, color='#16A34A', fontweight='bold')

    ax.set_ylabel('WER (%) — lower is better')
    ax.set_title('Bhili — Donor Language Comparison\n(Wav2vec2 xlsr-300m)')
    ax.set_ylim(0, 70)

    # Mundari: Wav2vec2-300m has 3 donors (Hindi, Santali, Bengali)
    ax2 = axes[1]
    donors_mundari = {
        'Staged\n(no transfer)': 66.17,
        'Hindi\ntransfer':       64.13,
        'Santali\ntransfer':     77.81,
        'Bengali\ntransfer':     74.24,
    }
    bar_colors_m = ['#9CA3AF', '#16A34A', '#F97316', '#DC2626']
    bars2 = ax2.bar(donors_mundari.keys(), donors_mundari.values(), color=bar_colors_m, alpha=0.85, width=0.55)
    for bar, val in zip(bars2, donors_mundari.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.add_patch(plt.Rectangle(
        (bars2[1].get_x()-0.05, 0), bars2[1].get_width()+0.1, donors_mundari['Hindi\ntransfer']+3,
        fill=False, edgecolor='#16A34A', lw=2.5, linestyle='--'
    ))
    ax2.text(bars2[1].get_x() + bars2[1].get_width()/2, donors_mundari['Hindi\ntransfer']+5,
             '✓ Best', ha='center', fontsize=9, color='#16A34A', fontweight='bold')

    ax2.set_ylabel('WER (%) — lower is better')
    ax2.set_title('Mundari — Donor Language Comparison\n(Wav2vec2 xlsr-300m)')
    ax2.set_ylim(0, 90)

    plt.tight_layout()
    plt.savefig('fig6_transfer.png')
    plt.close()
    print('✓ Fig 6 saved')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – LM Impact: WER vs CER decomposition
# ══════════════════════════════════════════════════════════════════════════════
def fig7_lm_impact():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Figure 7: Language Model Integration — WER and CER Impact', fontsize=14, fontweight='bold', y=1.01)

    # Bhili: No LM vs LM (Wav2vec2-1B, syn->org)
    ax = axes[0]
    conditions = ['No LM', 'With LM']
    bhili_wer = [32.91, 22.32]
    bhili_cer = [10.35, 8.83]

    x = np.arange(2)
    b1 = ax.bar(x - 0.2, bhili_wer, 0.35, color='#60A5FA', label='WER', alpha=0.85)
    b2 = ax.bar(x + 0.2, bhili_cer, 0.35, color='#34D399', label='CER', alpha=0.85)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Reduction annotations
    wer_drop = bhili_wer[0] - bhili_wer[1]
    cer_drop = bhili_cer[0] - bhili_cer[1]
    ax.annotate(f'−{wer_drop:.2f} pp\n({wer_drop/bhili_wer[0]*100:.1f}% ↓)',
                xy=(1-0.2, bhili_wer[1]), xytext=(0.4, 28),
                arrowprops=dict(arrowstyle='->', color='#1D4ED8', lw=1.2),
                fontsize=9, color='#1D4ED8', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='#EFF6FF', ec='#BFDBFE'))
    ax.annotate(f'−{cer_drop:.2f} pp\n({cer_drop/bhili_cer[0]*100:.1f}% ↓)',
                xy=(1+0.2, bhili_cer[1]), xytext=(1.4, 20),
                arrowprops=dict(arrowstyle='->', color='#065F46', lw=1.2),
                fontsize=9, color='#065F46', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='#ECFDF5', ec='#6EE7B7'))

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Error Rate (%) — lower is better')
    ax.set_title('Bhili — LM Impact (Wav2vec2-1B, Syn→Org)')
    ax.legend()
    ax.set_ylim(0, 40)

    # Mundari: LM actually HURTS
    ax2 = axes[1]
    mundari_wer = [37.52, 42.74]
    mundari_cer = [17.91, 20.17]

    b3 = ax2.bar(x - 0.2, mundari_wer, 0.35, color='#F87171', label='WER', alpha=0.85)
    b4 = ax2.bar(x + 0.2, mundari_cer, 0.35, color='#FBBF24', label='CER', alpha=0.85)

    for bar in list(b3) + list(b4):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    wer_inc = mundari_wer[1] - mundari_wer[0]
    ax2.annotate(f'+{wer_inc:.2f} pp\n(LM HURTS!)',
                 xy=(1-0.2, mundari_wer[1]), xytext=(0.4, 45),
                 arrowprops=dict(arrowstyle='->', color='#991B1B', lw=1.2),
                 fontsize=9, color='#991B1B', ha='center',
                 bbox=dict(boxstyle='round,pad=0.2', fc='#FEF2F2', ec='#FCA5A5'))

    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions)
    ax2.set_ylabel('Error Rate (%) — lower is better')
    ax2.set_title('Mundari — LM Impact (Wav2vec2-1B, Syn→Org)')
    ax2.legend()
    ax2.set_ylim(0, 52)
    ax2.text(0.5, 0.02, '⚠ LM degrades performance for Mundari — likely OOV/domain mismatch',
             transform=ax2.transAxes, ha='center', fontsize=8.5, color='#991B1B', style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig7_lm_impact.png')
    plt.close()
    print('✓ Fig 7 saved')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 – WER vs CER Gap Analysis (Morphological insight)
# ══════════════════════════════════════════════════════════════════════════════
def fig8_wer_cer_gap():
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle('Figure 8: WER vs CER Gap — Morphological Complexity Indicator', fontsize=14, fontweight='bold')

    # Data points: (WER, CER, label, lang)
    points = [
        (32.91, 10.35, 'Bhili\nNo LM',  'Bhili',   'o'),
        (22.32, 8.83,  'Bhili\n+LM',    'Bhili',   'o'),
        (37.52, 17.91, 'Mundari\nNo LM','Mundari',  's'),
        (42.74, 20.17, 'Mundari\n+LM',  'Mundari',  's'),
    ]

    for wer, cer, lbl, lang, mrk in points:
        col = C['bhili'] if lang == 'Bhili' else C['mundari']
        ax.scatter(wer, cer, s=140, color=col, marker=mrk, zorder=5, edgecolors='white', lw=1.5)
        ax.annotate(lbl, (wer, cer), textcoords='offset points', xytext=(8, 4), fontsize=9)

    # Identity line (WER=CER): would mean every word error is a single character
    lim_max = 50
    ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.2, lw=1.5, label='WER = CER (baseline)')

    # Ratio lines
    for ratio, lbl, ls in [(0.5, 'CER = 0.5×WER', ':'), (0.3, 'CER = 0.3×WER', '-.')]:
        ax.plot([0, lim_max], [0, lim_max*ratio], color='#9CA3AF', ls=ls, lw=1, label=lbl)

    # Bhili ratio annotation
    bhili_ratio = 10.35 / 32.91
    mundari_ratio = 17.91 / 37.52
    ax.text(35, 5, f'Bhili CER/WER ≈ {bhili_ratio:.2f}\n(word-level errors dominant)',
            fontsize=9, color=C['bhili'],
            bbox=dict(boxstyle='round,pad=0.4', fc='#EFF6FF', ec='#BFDBFE'))
    ax.text(35, 25, f'Mundari CER/WER ≈ {mundari_ratio:.2f}\n(more char-level errors)',
            fontsize=9, color=C['mundari'],
            bbox=dict(boxstyle='round,pad=0.4', fc='#FEF2F2', ec='#FCA5A5'))

    ax.set_xlabel('WER (%) — lower is better')
    ax.set_ylabel('CER (%) — lower is better')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 30)
    ax.legend(fontsize=9, loc='upper left')

    b_patch = mpatches.Patch(color=C['bhili'],   label='Bhili (55 hrs)')
    m_patch = mpatches.Patch(color=C['mundari'], label='Mundari (11 hrs)')
    ax.legend(handles=[b_patch, m_patch], loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('fig8_wer_cer_gap.png')
    plt.close()
    print('✓ Fig 8 saved')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 – Data Efficiency: WER Improvement per Hour
# ══════════════════════════════════════════════════════════════════════════════
def fig9_data_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Figure 9: Data Efficiency — WER Improvement per Training Hour', fontsize=14, fontweight='bold', y=1.01)

    for ax, staged_dict, total_hours, baseline_zs, lang_name, lang_color in zip(
        axes,
        [staged_bhili, staged_mundari],
        [55, 11],
        [86.84, 94.77],  # SeamlessM4T zero-shot
        ['Bhili (55 hrs)', 'Mundari (11 hrs)'],
        [C['bhili'], C['mundari']],
    ):
        model_names = list(staged_dict.keys())
        short_names = [m.replace('Wav2vec2 xlsr-', 'W2V-').replace('Whisper ', 'W-') for m in model_names]
        eff_hours   = [a * total_hours for a in alphas]

        # Compute marginal gain per hour at each alpha step
        # i.e., (WER[alpha_prev] - WER[alpha_curr]) / (hours_curr - hours_prev)
        # For first point: gain from zero-shot to alpha=0.2

        # Use Whisper Medium as example, also compute average
        all_marginal = {}
        for model, wers in staged_dict.items():
            wer_with_zs = [baseline_zs] + wers
            hrs_with_zs = [0] + eff_hours
            marginals   = []
            for i in range(1, len(wer_with_zs)):
                dh = hrs_with_zs[i] - hrs_with_zs[i-1]
                dw = wer_with_zs[i-1] - wer_with_zs[i]  # positive = improvement
                marginals.append(dw / dh if dh > 0 else 0)
            all_marginal[model] = marginals

        # Plot average marginal gain
        avg_marginal = np.mean([all_marginal[m] for m in model_names], axis=0)
        mid_hours    = [(eff_hours[i] + ([0]+eff_hours)[i])/2 for i in range(len(eff_hours))]

        bars = ax.bar(range(len(alphas)), avg_marginal, color=[
            lang_color if v > 0 else '#EF4444' for v in avg_marginal
        ], alpha=0.8, width=0.6)

        for bar, val in zip(bars, avg_marginal):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.05 if val >= 0 else -0.35),
                    f'{val:+.2f}', ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=9, fontweight='bold')

        ax.axhline(0, color='#374151', lw=1)
        ax.set_xticks(range(len(alphas)))
        alpha_labels = [f'0→α={alphas[0]}', f'α={alphas[0]}→{alphas[1]}',
                        f'α={alphas[1]}→{alphas[2]}', f'α={alphas[2]}→{alphas[3]}',
                        f'α={alphas[3]}→{alphas[4]}']
        ax.set_xticklabels(alpha_labels, rotation=22, ha='right', fontsize=8.5)
        ax.set_ylabel('Avg WER points improved per hour')
        ax.set_title(lang_name)

        # Highlight diminishing returns
        ax.text(0.97, 0.97, 'Positive = improvement\nNegative = degradation',
                transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#6B7280',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#E5E7EB', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig9_data_efficiency.png')
    plt.close()
    print('✓ Fig 9 saved')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 – Grand Summary: Best System per Configuration
# ══════════════════════════════════════════════════════════════════════════════
def fig10_grand_summary():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Figure 10: Grand Summary — Best WER per Training Configuration', fontsize=14, fontweight='bold', y=1.01)

    configs_bhili = {
        'Zero-shot\n(SeamlessM4T)': 86.84,
        'Zero-shot\n(Whisper-L)':   74.48,
        'Staged FT\nα=1.0 (W-Med)': 36.62,
        'Staged FT\nα=0.6 (W2V-1B)':44.06,
        'Transfer\nHindi (W2V-300m)':45.33,
        'Syn→Org\n(W2V-1B)':        39.00,
        'Syn→Org\n+LM (W2V-1B)':   22.32,
    }

    configs_mundari = {
        'Zero-shot\n(SeamlessM4T)': 94.77,
        'Zero-shot\n(Whisper-L)':   89.62,
        'Staged FT\nα=0.8 (W-Med)': 54.68,
        'Staged FT\nα=0.2 (W2V-1B)':62.01,
        'Transfer\nHindi (W2V-300m)':64.13,
        'Syn→Org\n(W-Med)':         53.21,
        'Syn→Org\n+LM (W2V-1B)':   42.74,
    }

    palette = ['#CBD5E1', '#94A3B8', '#3B82F6', '#6366F1', '#F97316', '#10B981', '#059669']

    for ax, configs, lang_name, lang_color in zip(
        axes,
        [configs_bhili, configs_mundari],
        ['Bhili (55 hrs)', 'Mundari (11 hrs)'],
        [C['bhili'], C['mundari']],
    ):
        labels = list(configs.keys())
        values = list(configs.values())
        colors = palette[:len(labels)]

        bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.9, height=0.65)

        for bar, val in zip(bars, values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', ha='left', fontsize=9.5, fontweight='bold')

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('WER (%) — lower is better')
        ax.set_title(lang_name)
        ax.set_xlim(0, 105)

        # Best system marker
        best_idx = np.argmin(values)
        ax.annotate('★ Best', xy=(values[best_idx], best_idx),
                    xytext=(values[best_idx] + 15, best_idx),
                    arrowprops=dict(arrowstyle='->', color='#065F46', lw=1.2),
                    fontsize=9, color='#065F46', va='center', fontweight='bold')

        # Relative improvement from zero-shot to best
        zs = values[0]
        best = min(values)
        improvement = (zs - best) / zs * 100
        ax.text(0.98, 0.04, f'Total improvement\nover zero-shot:\n{improvement:.1f}%',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', fc='#F0FDF4', ec='#86EFAC'),
                color='#065F46')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig10_grand_summary.png')
    plt.close()
    print('✓ Fig 10 saved')


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating all figures...\n')
    #fig1_zeroshot_vs_finetuned()
    
    #fig2_scaling_laws()
    #fig3_alpha_heatmap()
    #fig4_pareto()
    #fig5_synthetic()
    #fig11_synthetic()
    #fig6_transfer()
    #fig7_lm_impact()
    fig8_wer_cer_gap()
    #fig9_data_efficiency()
    #fig10_grand_summary()
    print(f'\n✅ All 10 figures saved to {OUT}/')
