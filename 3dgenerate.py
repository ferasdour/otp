import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from textwrap import wrap
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

PASSWORD_LENGTH = 10
CHARSET_SIZE = 62  # A-Z, a-z, 0-9
KEYSPACE_SIZE = CHARSET_SIZE ** PASSWORD_LENGTH

WINDOW_SECONDS = 5 * 60  # 5 minutes
TOTAL_ATTEMPTS = 1_000_000

BRUTEFORCE_RATE = 50_000
COLLISION_RATE = 50_000

WEAK_ENTROPY_BITS = 40
OVERLAP_FACTOR = 0.1

PDF_REPORT = "otp_attack_analysis.pdf"

# ============================================================
# DERIVED PROBABILITIES
# ============================================================

p_bf = 1.0 / KEYSPACE_SIZE
weak_space_size = 2 ** WEAK_ENTROPY_BITS
p_coll = OVERLAP_FACTOR / weak_space_size

# ============================================================
# SIMULATION
# ============================================================

attempts = np.arange(1, TOTAL_ATTEMPTS + 1)

time_bf = attempts / BRUTEFORCE_RATE
time_coll = attempts / COLLISION_RATE

window_bf = (time_bf // WINDOW_SECONDS).astype(int)
window_coll = (time_coll // WINDOW_SECONDS).astype(int)

def cumulative_prob(p, k):
    return 1.0 - np.power(1.0 - p, k)

cum_bf = cumulative_prob(p_bf, attempts)
cum_coll = cumulative_prob(p_coll, attempts)

# ============================================================
# HELPER: TEXT PAGES (REPORT STYLE)
# ============================================================

def add_text_page(pdf, title, paragraphs, footer=None):
    """
    Create a report-style text page:
    - title at top
    - paragraphs as wrapped text
    - optional footer
    """
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    # Title
    fig.text(0.08, 0.93, title, fontsize=18, weight='bold')
    fig.text(0.08, 0.91, "_" * 80, fontsize=8, color="gray")

    y = 0.87
    for para in paragraphs:
        lines = wrap(para, 100)
        for line in lines:
            fig.text(0.08, y, line, fontsize=11)
            y -= 0.02
        y -= 0.01  # paragraph spacing

    if footer:
        fig.text(0.5, 0.03, footer, fontsize=8, ha='center', color='gray')

    pdf.savefig(fig)
    plt.close(fig)

# ============================================================
# HELPER: PLOTS
# ============================================================

def plot_scatter_page(pdf, attempts, windows, cum_prob, title):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        attempts, windows, cum_prob,
        c=cum_prob, cmap='viridis', s=1, alpha=0.6
    )

    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.set_xlabel("Attempt Index")
    ax.set_ylabel("5-Min Window Index")
    ax.set_zlabel("Cumulative Success Probability")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Cumulative Success Probability")

    pdf.savefig(fig)
    plt.close(fig)

def make_surface(attempts, windows, cum_prob,
                 attempt_bins=100, window_bins=50):
    a_min, a_max = attempts.min(), attempts.max()
    w_min, w_max = windows.min(), windows.max()

    a_edges = np.linspace(a_min, a_max, attempt_bins + 1)
    w_edges = np.linspace(w_min, w_max, window_bins + 1)

    A_centers = 0.5 * (a_edges[:-1] + a_edges[1:])
    W_centers = 0.5 * (w_edges[:-1] + w_edges[1:])

    Z = np.zeros((window_bins, attempt_bins))

    for i in range(attempt_bins):
        a_mask = (attempts >= a_edges[i]) & (attempts < a_edges[i+1])
        for j in range(window_bins):
            w_mask = (windows >= w_edges[j]) & (windows < w_edges[j+1])
            mask = a_mask & w_mask
            Z[j, i] = np.mean(cum_prob[mask]) if np.any(mask) else np.nan

    A_grid, W_grid = np.meshgrid(A_centers, W_centers)
    return A_grid, W_grid, Z

def plot_surface_page(pdf, A, W, Z, title):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111, projection='3d')

    Z_masked = np.ma.masked_invalid(Z)

    surf = ax.plot_surface(
        A, W, Z_masked,
        cmap='viridis', edgecolor='none', alpha=0.9
    )

    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.set_xlabel("Attempt Index (binned)")
    ax.set_ylabel("5-Min Window Index (binned)")
    ax.set_zlabel("Cumulative Success Probability")

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Cumulative Success Probability")

    pdf.savefig(fig)
    plt.close(fig)

# ============================================================
# PDF GENERATION
# ============================================================

with PdfPages(PDF_REPORT) as pdf:
    # ---------- Page 1: Title ----------
    title = "OTP Attack Analysis Report"
    subtitle = "Brute Force vs Collision-Based Attacks on Time-Limited OTPs"
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    fig.text(0.5, 0.7, title, fontsize=24, weight='bold', ha='center')
    fig.text(0.5, 0.64, subtitle, fontsize=14, ha='center')
    fig.text(0.5, 0.58, f"Generated: {datetime.now().isoformat(timespec='seconds')}",
             fontsize=10, ha='center', color='gray')

    fig.text(0.5, 0.3,
             "This report models and visualizes the risk of brute-force and\n"
             "collision-style attacks against a 10-character OTP over a 62-character\n"
             "alphabet, where the OTP changes every 5 minutes and no rate limiting\n"
             "or secondary authentication factor is enforced.",
             fontsize=11, ha='center')

    pdf.savefig(fig)
    plt.close(fig)

    # ---------- Page 2: Table of Contents ----------
    toc_paragraphs = [
        "1. Mathematical Foundations",
        "2. Threat Model",
        "3. Attack Model",
        "4. Security Implications",
        "5. Brute Force Scatter Plot",
        "6. Collision Scatter Plot",
        "7. Brute Force Surface Plot",
        "8. Collision Surface Plot",
        "9. Conclusions & Future Exploration",
    ]
    add_text_page(pdf, "Table of Contents", toc_paragraphs)

    # ---------- Page 3: Mathematical Foundations ----------
    math_paragraphs = [
        f"Keyspace for a 10-character password over 62 characters:",
        f"N = 62^10 = {KEYSPACE_SIZE:.3e}",
        f"Brute-force per-guess probability:",
        f"p_bf = 1 / N = {p_bf:.3e}",
        f"Weak generator entropy: {WEAK_ENTROPY_BITS} bits",
        f"Weak space size: 2^E = {weak_space_size:.3e}",
        f"Overlap factor (fraction of weak space overlapping strong outputs): {OVERLAP_FACTOR}",
        f"Collision per-guess probability:",
        f"p_coll = overlap / weak_space = {p_coll:.3e}",
        "Cumulative success probability after k independent attempts with per-guess "
        "success probability p is given by:",
        "P(k) = 1 - (1 - p)^k.",
    ]
    add_text_page(pdf, "Mathematical Foundations", math_paragraphs)

    # ---------- Page 4: Threat Model ----------
    threat_paragraphs = [
        "This analysis assumes an attacker with the following capabilities:",
        "- Unlimited OTP login attempts (no rate limiting).",
        "- Ability to automate requests at high speed.",
        "- Knowledge of OTP format and timing.",
        "- Ability to exploit weak or biased OTP generation.",
        "",
        "The defender is assumed to:",
        "- Rely solely on OTP for authentication.",
        "- Not enforce device binding or IP throttling.",
        "- Not monitor for abnormal login velocity.",
        "- Not enforce MFA or challenge-response mechanisms.",
        "",
        "Under these assumptions, the system is vulnerable to both brute-force and "
        "collision-style attacks, with the latter being significantly more dangerous "
        "if the OTP generator exhibits entropy weaknesses.",
    ]
    add_text_page(pdf, "Threat Model", threat_paragraphs)

    # ---------- Page 5: Attack Model ----------
    attack_paragraphs = [
        "We consider a 10-character OTP drawn from a 62-character alphabet "
        "(A–Z, a–z, 0–9). The OTP changes every 5 minutes, similar to a TOTP.",
        "",
        "Brute-force attack:",
        "- The attacker samples uniformly from the full keyspace.",
        "- Per-guess success probability is p_bf = 1 / 62^10.",
        "- Over a 5-minute window, the attacker can make many guesses, but the "
        "probability remains extremely small if the OTP is truly random.",
        "",
        "Collision-style attack:",
        "- The attacker uses a weaker generator with effective entropy E bits.",
        "- The weak generator covers a space of size 2^E.",
        "- Only a fraction of that space overlaps the true OTP distribution; this "
        "is modeled by an overlap factor f.",
        "- Per-guess success probability is p_coll = f / 2^E.",
        "",
        "If the OTP generator is biased or has structural weaknesses, a collision-style "
        "attack can be significantly more effective than naive brute force.",
    ]
    add_text_page(pdf, "Attack Model", attack_paragraphs)

    # ---------- Page 6: Security Implications ----------
    sec_paragraphs = [
        "A system that allows unlimited OTP login attempts with no rate limiting is "
        "vulnerable to both brute-force and collision-style attacks. Even though the "
        "brute-force probability is extremely low, a weak or biased OTP generator can "
        "dramatically increase the attacker's success rate.",
        "",
        "Single-factor OTP login is fundamentally unsafe because:",
        "- OTPs are guessable values.",
        "- No rate limiting allows unbounded attempts.",
        "- No MFA means a single correct guess grants full access.",
        "- Weak entropy or biased RNGs can be exploited.",
        "",
        "Future exploration includes entropy leakage analysis, RNG bias detection, "
        "timing channels, and OTP clustering behavior.",
    ]
    add_text_page(pdf, "Security Implications", sec_paragraphs)

    # ---------- Page 7: Brute Force Scatter Plot ----------
    plot_scatter_page(
        pdf,
        attempts,
        window_bf,
        cum_bf,
        "Brute Force Attack: 10-Character OTP, 62-Character Alphabet"
    )

    # ---------- Page 8: Collision Scatter Plot ----------
    plot_scatter_page(
        pdf,
        attempts,
        window_coll,
        cum_coll,
        "Collision-Style Attack: Weak Entropy Generator vs Strong OTP"
    )

    # ---------- Page 9: Brute Force Surface Plot ----------
    A_bf, W_bf, Z_bf = make_surface(attempts, window_bf, cum_bf)
    plot_surface_page(
        pdf,
        A_bf,
        W_bf,
        Z_bf,
        "Brute Force Cumulative Success Probability Surface"
    )

    # ---------- Page 10: Collision Surface Plot ----------
    A_coll, W_coll, Z_coll = make_surface(attempts, window_coll, cum_coll)
    plot_surface_page(
        pdf,
        A_coll,
        W_coll,
        Z_coll,
        "Collision-Style Cumulative Success Probability Surface"
    )

    # ---------- Page 11: Conclusions & Future Exploration ----------
    conclusions_paragraphs = [
        "This analysis demonstrates that single-factor OTP authentication is "
        "insufficient for protecting user accounts, especially when no rate "
        "limiting is enforced. Even with a large keyspace, the presence of a "
        "weak or biased OTP generator can drastically increase the probability "
        "of successful attacks.",
        "",
        "Future exploration should include:",
        "- Entropy leakage measurement.",
        "- Statistical analysis of OTP distribution.",
        "- RNG bias detection and clustering.",
        "- Timing channel analysis.",
        "- Modeling adversarial sampling strategies.",
        "- Evaluating MFA and device-binding mitigations.",
        "",
        "The evidence strongly supports migrating away from single-factor OTP "
        "authentication toward multi-factor, rate-limited, and device-bound "
        "authentication mechanisms.",
    ]
    add_text_page(pdf, "Conclusions & Future Exploration", conclusions_paragraphs)

print(f"[+] PDF report generated: {PDF_REPORT}")
