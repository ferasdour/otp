import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from textwrap import wrap
from datetime import datetime

# Try GPU (CuPy); fall back to NumPy
try:
    import cupy as cp
    GPU_ENABLED = True
except ImportError:
    cp = None
    GPU_ENABLED = False

# ============================================================
# CONFIGURATION
# ============================================================

PASSWORD_LENGTH = 10
CHARSET_SIZE = 62  # A-Z, a-z, 0-9
KEYSPACE_SIZE = CHARSET_SIZE ** PASSWORD_LENGTH

WINDOW_SECONDS = 5 * 60  # 5 minutes

# Logical total attempts (for probability model)
TOTAL_ATTEMPTS = 50_000_000

BRUTEFORCE_RATE = 50_000
COLLISION_RATE = 50_000

WEAK_ENTROPY_BITS = 40
OVERLAP_FACTOR = 0.1

PDF_REPORT = "otp_attack_analysis_hperf.pdf"
IMAGE_DIR = "exported_images"
IMAGE_DPI = 600

# Plotting downsample sizes
PLOT_POINTS_SCATTER = 200_000
PLOT_POINTS_LOG = 200_000
SURFACE_ATTEMPT_BINS = 120  # binned anyway

os.makedirs(IMAGE_DIR, exist_ok=True)

# ============================================================
# CORE MATH (CPU/GPU AGNOSTIC, CHUNKED)
# ============================================================

def cumulative_prob_chunked(p, total_attempts, chunk_size=1_000_000, use_gpu=False):
    """
    Compute cumulative probability P(k) = 1 - (1 - p)^k
    for k = 1..total_attempts in chunks, optionally on GPU.
    Returns a NumPy array of length total_attempts.
    """
    if use_gpu and GPU_ENABLED:
        xp = cp
    else:
        xp = np

    result = np.empty(total_attempts, dtype=np.float64)
    remaining = total_attempts
    offset = 0

    while remaining > 0:
        n = min(remaining, chunk_size)
        k_chunk = xp.arange(offset + 1, offset + n + 1, dtype=xp.float64)
        p_val = xp.float64(p)
        chunk = 1.0 - xp.power(1.0 - p_val, k_chunk)
        chunk_cpu = cp.asnumpy(chunk) if (use_gpu and GPU_ENABLED) else chunk
        result[offset:offset + n] = chunk_cpu
        offset += n
        remaining -= n

    return result

# ============================================================
# DERIVED PROBABILITIES
# ============================================================

p_bf = 1.0 / KEYSPACE_SIZE
weak_space_size = 2 ** WEAK_ENTROPY_BITS
p_coll = OVERLAP_FACTOR / weak_space_size

# ============================================================
# SIMULATION (LOGICAL MODEL)
# ============================================================

attempts = np.arange(1, TOTAL_ATTEMPTS + 1, dtype=np.int64)

time_bf = attempts / BRUTEFORCE_RATE
time_coll = attempts / COLLISION_RATE

window_bf = (time_bf // WINDOW_SECONDS).astype(int)
window_coll = (time_coll // WINDOW_SECONDS).astype(int)

cum_bf = cumulative_prob_chunked(p_bf, TOTAL_ATTEMPTS, use_gpu=True)
cum_coll = cumulative_prob_chunked(p_coll, TOTAL_ATTEMPTS, use_gpu=True)

# ============================================================
# DOWNSAMPLING HELPERS
# ============================================================

def downsample_indices(n, target):
    if n <= target:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, target, dtype=np.int64)

def downsample_for_plot(x, y, z, target_points):
    n = len(x)
    idx = downsample_indices(n, target_points)
    return x[idx], y[idx], z[idx]

def downsample_1d(x, y, target_points):
    n = len(x)
    idx = downsample_indices(n, target_points)
    return x[idx], y[idx]

# ============================================================
# HELPERS: TEXT PAGES
# ============================================================

def add_text_page(pdf, title, paragraphs, footer=None):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    fig.text(0.08, 0.93, title, fontsize=18, weight='bold')
    fig.text(0.08, 0.91, "_" * 80, fontsize=8, color="gray")

    y = 0.87
    for para in paragraphs:
        lines = wrap(para, 100)
        for line in lines:
            fig.text(0.08, y, line, fontsize=11)
            y -= 0.02
        y -= 0.01

    if footer:
        fig.text(0.5, 0.03, footer, fontsize=8, ha='center', color='gray')

    pdf.savefig(fig)
    plt.close(fig)

# ============================================================
# HELPERS: SURFACE GENERATION (ON DOWNSAMPLED DATA)
# ============================================================

def make_surface(attempts, windows, cum_prob, attempt_bins=SURFACE_ATTEMPT_BINS):
    unique_windows = np.unique(windows)
    window_bins = max(3, len(unique_windows))

    a_min, a_max = attempts.min(), attempts.max()
    w_min, w_max = windows.min(), windows.max()

    a_edges = np.linspace(a_min, a_max, attempt_bins + 1)
    w_edges = np.linspace(w_min, w_max, window_bins + 1)

    A_centers = 0.5 * (a_edges[:-1] + a_edges[1:])
    W_centers = 0.5 * (w_edges[:-1] + w_edges[1:])

    Z = np.zeros((window_bins, attempt_bins), dtype=float)

    for i in range(attempt_bins):
        a_mask = (attempts >= a_edges[i]) & (attempts < a_edges[i+1])
        if not np.any(a_mask):
            continue
        for j in range(window_bins):
            w_mask = (windows >= w_edges[j]) & (windows < w_edges[j+1])
            mask = a_mask & w_mask
            if np.any(mask):
                Z[j, i] = np.mean(cum_prob[mask])
            else:
                Z[j, i] = 0.0

    A_grid, W_grid = np.meshgrid(A_centers, W_centers)
    return A_grid, W_grid, Z

def plot_surface(A, W, Z, title, zlabel="Cumulative Success Probability"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        A, W, Z,
        cmap='viridis', edgecolor='none', alpha=0.9
    )

    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.set_xlabel("Attempt Index (binned)")
    ax.set_ylabel("5-Min Window Index (binned)")
    ax.set_zlabel(zlabel)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(zlabel)
    return fig

def save_fig(fig, filename_base):
    path = os.path.join(IMAGE_DIR, f"{filename_base}.png")
    fig.savefig(path, dpi=IMAGE_DPI, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# HELPERS: SCATTER & LOG-SCALE PLOTS (DOWNSAMPLED)
# ============================================================

def plot_scatter_3d(x, y, z, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.6)
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.set_xlabel("Attempt Index")
    ax.set_ylabel("5-Min Window Index")
    ax.set_zlabel("Cumulative Success Probability")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Cumulative Success Probability")
    return fig

def plot_logscale(attempts, cum_prob, title):
    eps = 1e-300
    y = np.log10(np.clip(cum_prob, eps, 1.0))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(attempts, y, linewidth=0.8)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel("Attempt Index")
    ax.set_ylabel("log10(Cumulative Success Probability)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    return fig

# ============================================================
# HELPERS: SIDE-BY-SIDE PLOTS (DOWNSAMPLED)
# ============================================================

def plot_side_by_side_scatter(x_bf, w_bf, c_bf,
                              x_coll, w_coll, c_coll, title):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    sc1 = ax1.scatter(x_bf, w_bf, c_bf, c=c_bf,
                      cmap='viridis', s=1, alpha=0.6)
    ax1.set_title("Brute Force", fontsize=12, weight='bold')
    ax1.set_xlabel("Attempt Index")
    ax1.set_ylabel("5-Min Window Index")
    ax1.set_zlabel("Cum. Prob.")

    sc2 = ax2.scatter(x_coll, w_coll, c_coll, c=c_coll,
                      cmap='viridis', s=1, alpha=0.6)
    ax2.set_title("Collision", fontsize=12, weight='bold')
    ax2.set_xlabel("Attempt Index")
    ax2.set_ylabel("5-Min Window Index")
    ax2.set_zlabel("Cum. Prob.")

    fig.suptitle(title, fontsize=14, weight='bold')
    fig.colorbar(sc1, ax=ax1, shrink=0.6, pad=0.1)
    fig.colorbar(sc2, ax=ax2, shrink=0.6, pad=0.1)
    return fig

def plot_side_by_side_surface(A_bf, W_bf, Z_bf,
                              A_coll, W_coll, Z_coll, title):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    surf1 = ax1.plot_surface(A_bf, W_bf, Z_bf,
                             cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title("Brute Force", fontsize=12, weight='bold')
    ax1.set_xlabel("Attempt Index (binned)")
    ax1.set_ylabel("5-Min Window Index (binned)")
    ax1.set_zlabel("Cum. Prob.")

    surf2 = ax2.plot_surface(A_coll, W_coll, Z_coll,
                             cmap='viridis', edgecolor='none', alpha=0.9)
    ax2.set_title("Collision", fontsize=12, weight='bold')
    ax2.set_xlabel("Attempt Index (binned)")
    ax2.set_ylabel("5-Min Window Index (binned)")
    ax2.set_zlabel("Cum. Prob.")

    fig.suptitle(title, fontsize=14, weight='bold')
    fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.1)
    fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.1)
    return fig

# ============================================================
# PREPARE DOWNSAMPLED VIEWS FOR PLOTTING
# ============================================================

# Scatter views
x_bf_s, w_bf_s, c_bf_s = downsample_for_plot(
    attempts, window_bf, cum_bf, PLOT_POINTS_SCATTER
)
x_coll_s, w_coll_s, c_coll_s = downsample_for_plot(
    attempts, window_coll, cum_coll, PLOT_POINTS_SCATTER
)

# Log views
x_bf_log, c_bf_log = downsample_1d(attempts, cum_bf, PLOT_POINTS_LOG)
x_coll_log, c_coll_log = downsample_1d(attempts, cum_coll, PLOT_POINTS_LOG)

# Surface views (we use full logical arrays but binned)
A_bf, W_bf, Z_bf = make_surface(attempts, window_bf, cum_bf)
A_coll, W_coll, Z_coll = make_surface(attempts, window_coll, cum_coll)

# ============================================================
# PDF GENERATION + PNG EXPORT
# ============================================================

with PdfPages(PDF_REPORT) as pdf:
    # Title page
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    fig.text(0.5, 0.7, "OTP Attack Analysis Report",
             fontsize=24, weight='bold', ha='center')
    fig.text(0.5, 0.64,
             "Brute Force vs Collision-Based Attacks on Time-Limited OTPs",
             fontsize=14, ha='center')
    fig.text(0.5, 0.58,
             f"Generated: {datetime.now().isoformat(timespec='seconds')}",
             fontsize=10, ha='center', color='gray')
    pdf.savefig(fig)
    plt.close(fig)

    # TOC
    toc_paragraphs = [
        "1. Mathematical Foundations",
        "2. Threat Model",
        "3. Attack Model",
        "4. Security Implications",
        "5. Brute Force Scatter Plot",
        "6. Collision Scatter Plot",
        "7. Brute Force Surface Plot",
        "8. Collision Surface Plot",
        "9. Log-Scale Plots",
        "10. Side-by-Side Comparisons",
        "11. GPU/Chunking Notes",
        "12. Conclusions & Future Exploration",
    ]
    add_text_page(pdf, "Table of Contents", toc_paragraphs)

    # Mathematical Foundations
    math_paragraphs = [
        f"Keyspace: 62^10 = {KEYSPACE_SIZE:.3e}",
        f"Brute-force probability p_bf = {p_bf:.3e}",
        f"Weak entropy bits = {WEAK_ENTROPY_BITS}",
        f"Weak space size = {weak_space_size:.3e}",
        f"Overlap factor = {OVERLAP_FACTOR}",
        f"Collision probability p_coll = {p_coll:.3e}",
        "Cumulative probability: P(k) = 1 - (1 - p)^k.",
    ]
    add_text_page(pdf, "Mathematical Foundations", math_paragraphs)

    # Threat Model
    threat_paragraphs = [
        "Attacker: unlimited attempts, automation, knowledge of OTP timing and format.",
        "Defender: relies solely on OTP, no rate limiting, no MFA, no anomaly detection.",
    ]
    add_text_page(pdf, "Threat Model", threat_paragraphs)

    # Attack Model
    attack_paragraphs = [
        "Brute force: uniform sampling over 62^10.",
        "Collision: attacker exploits weak entropy (E bits) and overlap with true OTP distribution.",
    ]
    add_text_page(pdf, "Attack Model", attack_paragraphs)

    # Security Implications
    sec_paragraphs = [
        "Weak or biased OTP generators drastically increase attack success.",
        "Single-factor OTP authentication is unsafe without rate limiting and MFA.",
    ]
    add_text_page(pdf, "Security Implications", sec_paragraphs)

    # Brute Force Scatter
    fig = plot_scatter_3d(x_bf_s, w_bf_s, c_bf_s,
                          "Brute Force Attack Scatter (Downsampled)")
    pdf.savefig(fig)
    save_fig(fig, "brute_force_scatter")

    # Collision Scatter
    fig = plot_scatter_3d(x_coll_s, w_coll_s, c_coll_s,
                          "Collision Attack Scatter (Downsampled)")
    pdf.savefig(fig)
    save_fig(fig, "collision_scatter")

    # Surfaces
    fig = plot_surface(A_bf, W_bf, Z_bf,
                       "Brute Force Success Probability Surface")
    pdf.savefig(fig)
    save_fig(fig, "brute_force_surface")

    fig = plot_surface(A_coll, W_coll, Z_coll,
                       "Collision Success Probability Surface")
    pdf.savefig(fig)
    save_fig(fig, "collision_surface")

    # Log-scale plots
    fig = plot_logscale(x_bf_log, c_bf_log,
                        "Brute Force log10(Cumulative Success Probability) (Downsampled)")
    pdf.savefig(fig)
    save_fig(fig, "brute_force_logscale")

    fig = plot_logscale(x_coll_log, c_coll_log,
                        "Collision log10(Cumulative Success Probability) (Downsampled)")
    pdf.savefig(fig)
    save_fig(fig, "collision_logscale")

    # Side-by-side scatter
    fig = plot_side_by_side_scatter(
        x_bf_s, w_bf_s, c_bf_s,
        x_coll_s, w_coll_s, c_coll_s,
        "Brute Force vs Collision (Scatter, Downsampled)"
    )
    pdf.savefig(fig)
    save_fig(fig, "side_by_side_scatter")

    # Side-by-side surface
    fig = plot_side_by_side_surface(
        A_bf, W_bf, Z_bf,
        A_coll, W_coll, Z_coll,
        "Brute Force vs Collision (Surface)"
    )
    pdf.savefig(fig)
    save_fig(fig, "side_by_side_surface")

    # GPU / Chunking notes
    gpu_paragraphs = [
        f"GPU acceleration enabled: {GPU_ENABLED}.",
        "Cumulative probabilities are computed in chunks to avoid excessive memory usage.",
        "If CuPy is installed and a compatible GPU is available, chunks are computed on GPU.",
        "Otherwise, the script transparently falls back to NumPy on CPU.",
        f"Logical total attempts modeled: {TOTAL_ATTEMPTS:,}.",
        f"Scatter/log plots are downsampled to at most {PLOT_POINTS_SCATTER:,} / {PLOT_POINTS_LOG:,} points.",
    ]
    add_text_page(pdf, "GPU and Chunking Notes", gpu_paragraphs)

    # Conclusions
    conclusions_paragraphs = [
        "Single-factor OTP authentication is insufficient for high-value accounts.",
        "Weak entropy or biased generators can make collision-style attacks far more effective "
        "than naive brute force.",
        "Mitigations: strong RNGs, rate limiting, MFA, device binding, anomaly detection.",
    ]
    add_text_page(pdf, "Conclusions & Future Exploration", conclusions_paragraphs)

print(f"[+] PDF report generated: {PDF_REPORT}")
print(f"[+] PNG images exported to: {IMAGE_DIR}/")
