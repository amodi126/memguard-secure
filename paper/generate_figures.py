"""Generate IEEE Access publication-quality figures for MemGuard paper."""

import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

OUT_DIR = str(pathlib.Path(__file__).parent)

# IEEE Access style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
})


def generate_architecture_diagram():
    """Figure 1: MemGuard architecture flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(7.16, 3.5))  # IEEE double-column width
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    # Colors - grayscale-safe
    box_agent = '#D9D9D9'
    box_voter = '#F2F2F2'
    box_allow = '#C8C8C8'
    box_block = '#E8E8E8'
    diamond_color = '#E0E0E0'
    arrow_color = '#333333'
    border_color = '#444444'

    box_props = dict(boxstyle='round,pad=0.3', facecolor=box_voter,
                     edgecolor=border_color, linewidth=1.2)
    agent_props = dict(boxstyle='round,pad=0.4', facecolor=box_agent,
                       edgecolor=border_color, linewidth=1.5)
    arrow_props = dict(arrowstyle='->', color=arrow_color, linewidth=1.3,
                       connectionstyle='arc3,rad=0')

    # --- Agent box (left) ---
    ax.text(1.0, 2.75, 'Letta Agent\n(GPT-4o-mini)', ha='center', va='center',
            fontsize=10, fontweight='bold', bbox=agent_props)

    # --- Arrow: Proposed Memory Write ---
    ax.annotate('', xy=(2.8, 2.75), xytext=(1.8, 2.75), arrowprops=arrow_props)
    ax.text(2.3, 3.05, 'Proposed\nMemory Write', ha='center', va='bottom',
            fontsize=7, style='italic', color='#333333')

    # --- Input label ---
    ax.text(3.05, 0.55, 'Each voter receives:\ncurrent state, proposed state,\nuser message',
            ha='center', va='center', fontsize=7, style='italic', color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAFAFA',
                      edgecolor='#AAAAAA', linewidth=0.8, linestyle='--'))

    # --- Interceptor bar ---
    ax.plot([2.85, 2.85], [1.2, 4.3], color=border_color, linewidth=1.5,
            linestyle='--', alpha=0.6)
    ax.text(2.85, 4.5, 'MemGuard\nHook', ha='center', va='bottom',
            fontsize=7, fontweight='bold', color='#555555')

    # --- Three voter boxes (center) ---
    voters = [
        ('Voter 1: Memory Consistency\n(Claude Haiku)', 4.2),
        ('Voter 2: Request Classification\n(GPT-4o-mini)', 2.75),
        ('Voter 3: Instruction Detection\n(Gemini Flash)', 1.3),
    ]

    voter_x = 4.2
    for label, y in voters:
        ax.text(voter_x, y, label, ha='center', va='center',
                fontsize=7.5, bbox=box_props)
        # Arrow from interceptor to voter
        ax.annotate('', xy=(3.15, y), xytext=(2.9, 2.75),
                    arrowprops=dict(arrowstyle='->', color=arrow_color,
                                   linewidth=0.9, connectionstyle='arc3,rad=0'))

    # --- Arrows from voters to diamond ---
    diamond_x = 6.5
    diamond_y = 2.75
    for _, y in voters:
        ax.annotate('', xy=(diamond_x - 0.55, diamond_y + (y - diamond_y) * 0.2),
                    xytext=(5.3, y),
                    arrowprops=dict(arrowstyle='->', color=arrow_color,
                                   linewidth=0.9, connectionstyle='arc3,rad=0'))

    # --- Diamond: Majority Vote ---
    diamond_size = 0.55
    diamond = mpatches.FancyBboxPatch(
        (diamond_x - diamond_size, diamond_y - diamond_size),
        diamond_size * 2, diamond_size * 2,
        boxstyle='round,pad=0.05', facecolor=diamond_color,
        edgecolor=border_color, linewidth=1.3,
        transform=ax.transData
    )
    # Draw diamond as rotated square
    diamond_verts = np.array([
        [diamond_x, diamond_y + 0.6],
        [diamond_x + 0.7, diamond_y],
        [diamond_x, diamond_y - 0.6],
        [diamond_x - 0.7, diamond_y],
        [diamond_x, diamond_y + 0.6],
    ])
    diamond_patch = mpatches.Polygon(diamond_verts, closed=True,
                                      facecolor=diamond_color,
                                      edgecolor=border_color, linewidth=1.3)
    ax.add_patch(diamond_patch)
    ax.text(diamond_x, diamond_y, 'Vote\n≥2/3', ha='center', va='center',
            fontsize=8, fontweight='bold')

    # --- ALLOW arrow (right-up) ---
    ax.annotate('', xy=(8.5, 3.8), xytext=(diamond_x + 0.7, diamond_y + 0.25),
                arrowprops=dict(arrowstyle='->', color='#2a7f2a', linewidth=1.5,
                               connectionstyle='arc3,rad=0.15'))
    ax.text(7.8, 3.55, 'ALLOW', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#2a7f2a')
    ax.text(8.8, 3.8, 'Commit Write\nto Memory', ha='center', va='center',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#D5E8D4',
                      edgecolor='#2a7f2a', linewidth=1.3))

    # --- BLOCK arrow (right-down) ---
    ax.annotate('', xy=(8.5, 1.7), xytext=(diamond_x + 0.7, diamond_y - 0.25),
                arrowprops=dict(arrowstyle='->', color='#aa3333', linewidth=1.5,
                               connectionstyle='arc3,rad=-0.15'))
    ax.text(7.8, 1.95, 'BLOCK', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#aa3333')
    ax.text(8.8, 1.7, 'Reject Write\n(Preserve Current)', ha='center', va='center',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#F8D7DA',
                      edgecolor='#aa3333', linewidth=1.3))

    plt.tight_layout(pad=0.3)
    fig.savefig(f'{OUT_DIR}/fig_architecture.pdf',
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved fig_architecture.pdf")


def generate_ablation_heatmap():
    """Figure 2: Per-voter detection/acceptance heatmap.

    Data matches Table tab:ablation_detail (enhanced configuration, full
    MemGuard voter prompts):
        - Social Eng. detection:  Claude 100, GPT-4o-mini 0,  Gemini 100, Ens 100
        - Authority Inj. detection: Claude 100, GPT-4o-mini 93.7, Gemini 100, Ens 100
        - Legitimate Update Acceptance (= 1 - FPR): all voters 100 (all FPR 0%)
    """
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.2))  # wider+taller to de-crowd labels

    # Data matrix  (rows: voters; cols: Social Eng., Authority Inj., Legit. Accept.)
    data = np.array([
        [100, 100,  100],   # Claude Haiku
        [0,   93.7, 100],   # GPT-4o-mini   (authority 93.7%, not old 86.3%)
        [100, 100,  100],   # Gemini Flash  (now 100% acceptance, not old 0%)
        [100, 100,  100],   # 3-Voter Ensemble
    ])

    # De-crowded labels: short bold name on line 1, lens in smaller parens line 2.
    row_labels = [
        'Claude Haiku\n(Mem. Consistency)',
        'GPT-4o-mini\n(Req. Classification)',
        'Gemini Flash\n(Instr. Detection)',
        '3-Voter Ensemble',
    ]
    col_labels = [
        'Social\nEngineering',
        'Authority\nInjection',
        'Legitimate\nAcceptance',
    ]

    # Plot heatmap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, ha='center')
    ax.set_yticklabels(row_labels, fontsize=7)

    # Move x labels to top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(top=True, bottom=False)
    ax.tick_params(axis='x', which='major', pad=6)

    # Add separator line before ensemble row
    ax.axhline(y=2.5, color='black', linewidth=1.5)

    # Annotate cells with percentage values
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data[i, j]
            # Choose text color for readability
            if val < 30:
                text_color = 'white'
                fontweight = 'bold'
            elif val > 70:
                text_color = 'black'
                fontweight = 'bold' if i == 3 else 'normal'
            else:
                text_color = 'black'
                fontweight = 'normal'

            text = f'{val:.0f}%' if val == int(val) else f'{val:.1f}%'
            txt = ax.text(j, i, text, ha='center', va='center',
                          fontsize=9, fontweight=fontweight, color=text_color)
            # Add outline for readability on the dark-red 0% cell
            if val < 30:
                txt.set_path_effects([
                    pe.withStroke(linewidth=2, foreground='#8B0000')
                ])

    # Grid lines
    ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', size=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Rate (%)', fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout(pad=0.5)
    # NOTE: the paper does \includegraphics{fig_ablation_heatmap.png}, so save PNG.
    # A PDF copy is also written in case you switch the \includegraphics extension.
    fig.savefig(f'{OUT_DIR}/fig_ablation_heatmap.png',
                bbox_inches='tight', pad_inches=0.05)
    fig.savefig(f'{OUT_DIR}/fig_ablation_heatmap.pdf',
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved fig_ablation_heatmap.png and .pdf")


if __name__ == '__main__':
    generate_architecture_diagram()
    generate_ablation_heatmap()
    print("Done — both figures saved.")