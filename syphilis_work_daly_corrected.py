"""
syphilis_ed_cea.py  ·  Emergency Department Universal Syphilis Screening — CEA  (v4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v4 changes vs v3
────────────────
DALY CORRECTION PATCH
- Adds explicit component accounting for stillbirth YLL (IUFD ≥28w),
  CS post-neonatal excess-mortality YLL, miscarriage grief YLD,
  maternal hospitalisation YLD, and preterm infant YLD.
- Background life-table deaths in the infant Markov model are not charged as
  CS YLL; only user-specified excess mortality generates post-neonatal YLL.

METHODOLOGICAL
1.  VSL / DALY framework separation
      - Health-sector ICER:  DALY-based, payer costs only.
      - Societal ICER:       DALY-based, adds productivity losses (human capital)
                             and caregiver time costs. VSL removed from ICER math.
      - VSL net-benefit:     Presented as a separate, explicitly-labelled analysis
                             in a new "Societal (VSL)" expander. Not mixed into ICER.
2.  Maternal morbidity module  (MaternalMorbidity dataclass)
      - Cardiovascular syphilis: DW 0.070, P(event|untreated late latent) 0.05
      - Neurosyphilis:           DW 0.440, P(event|tertiary) 0.10
      - Pregnancy hospitalisation: mean $4 200 per episode, P 0.12
      - Toggled via sidebar checkbox; propagates through PSA and deterministic helpers.
3.  Productivity loss module  (ProductivityLoss dataclass)
      - BLS 2023 age/sex median weekly earnings → annualised, by 10-yr age band.
      - Applied to: maternal morbidity episodes (lost work-days), neonatal deaths
        (caregiver bereavement + discounted lost infant future earnings),
        CS mild sequelae (10% wage penalty), CS severe sequelae (40% wage penalty).
      - Human capital method; friction-cost variant available via checkbox.
4.  Gestational-stratum-specific prop_late
      - Stratum-specific early-loss fractions replace the single scalar default.
      - Weighted mean computed from GES_STRATA cohort weights; sidebar override retained.
5.  q_progress calibration
      - calibrate_q_progress() solves numerically for q that yields a user-specified
        target lifetime progression probability under the 2021 US Life Table.
      - Implied vs. stated rate displayed in Infant Markov tab with ✅/⚠️ flag.

PSA REPORTING
6.  Formal OWSA table  (parameter, base, low, high, ICER-low, ICER-high)
      downloadable as CSV alongside the tornado chart.
7.  PSA convergence diagnostic  — rolling ICER mean ± 5 % tolerance band.
8.  CE-plane quadrant analysis  — % of iterations in each quadrant, as a table.
9.  EVPI curve  — E[max(NMB,0)] − max(E[NMB],0) plotted vs WTP (full curve).
10. All CrI labels changed from "2.5%/97.5%" to "95% CrI (2.5th–97.5th pctile)".

STRUCTURAL
11. Single-file maintained for journal supplement portability.
    Internal organisation: §1 Utilities · §2 Parameters · §3 Productivity & Maternal ·
    §4 PSA Engine · §5 Deterministic Helpers · §6 Reporting · §7 Figures · §8 Streamlit UI.
    Next project will split into params / model / psa / reporting / figures / app modules.

Sign convention: ΔCost = program_costs − outcome_savings  (negative = saves money).
CEAC / EVPI: NMB = λ·ΔDALYs − ΔCost.
Societal NMB adds productivity-loss savings to ΔCost reduction and DALYs to numerator.
"""

import warnings, io
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import beta as bd, lognorm, gamma as gd
from scipy.optimize import brentq
# Sklearn is used for EVPPI regressions
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse
from matplotlib.ticker import StrMethodFormatter
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# §1 · UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def dollar_fmt(x, _):   return f"${x:,.0f}"
def millions_fmt(x, _): return f"${x*1e-6:,.1f}M"
def std2(lo, hi):        return (hi - lo) / 4.0
CPI = 585.10 / 494.629   # healthcare CPI 2019 → 2025

def pvf(t: float, r: float) -> float:
    if t <= 0:  return 0.0
    if r == 0:  return float(t)
    return (1.0 - (1.0 + r) ** (-t)) / r

def beta_ab(m, lo, hi) -> Tuple[float, float]:
    m   = float(np.clip(m, 1e-6, 1 - 1e-6))
    var = max(((hi - lo) / 4.0) ** 2, 1e-10)
    ab  = max(m * (1 - m) / var - 1.0, 1e-3)
    return max(m * ab, 1e-3), max((1 - m) * ab, 1e-3)

def gamma_ab(mu, sd) -> Tuple[float, float]:
    mu, sd = max(float(mu), 1e-9), max(float(sd), 1e-9)
    return (mu / sd) ** 2, sd ** 2 / mu

def lnorm_ms(m, lo, hi) -> Tuple[float, float]:
    lo = max(lo, 1e-12); hi = max(hi, lo * 1.01)
    return np.log(max(m, 1e-12)), (np.log(hi) - np.log(lo)) / 3.92

def summarize(a) -> dict:
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a):
        return {"mean": np.nan, "median": np.nan, "95% CrI lo": np.nan, "95% CrI hi": np.nan}
    return {
        "mean":       float(a.mean()),
        "median":     float(np.median(a)),
        "95% CrI lo": float(np.percentile(a, 2.5)),
        "95% CrI hi": float(np.percentile(a, 97.5)),
    }

def ci_ellipse(ax, x, y, ec="steelblue"):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 5: return
    mu  = np.array([x.mean(), y.mean()])
    cov = np.cov(x, y)
    ev, evec = np.linalg.eigh(cov)
    i = np.argsort(ev)[::-1]; ev, evec = ev[i], evec[:, i]
    ang = np.arctan2(evec[1, 0], evec[0, 0])
    w   = 2 * np.sqrt(max(ev[0], 0) * 5.991)
    h   = 2 * np.sqrt(max(ev[1], 0) * 5.991)
    ax.add_patch(Ellipse(mu, w, h, angle=np.rad2deg(ang),
                         edgecolor=ec, facecolor="none", lw=2, zorder=5))


# ══════════════════════════════════════════════════════════════════════════════
# §2 · PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# ── US Life Table 2021 (CDC NCHS) ────────────────────────────────────────────
LIFE_TABLE_QX: List[float] = [
    0.00586,  0.00042,  0.000272, 0.000225, 0.000184, 0.000157, 0.00014,  0.000128,
    0.000122, 0.000123, 0.000129, 0.000138, 0.000164, 0.00022,  0.00031,  0.000446,
    0.000637, 0.000868, 0.0011,   0.00127,  0.001373, 0.001488, 0.001605, 0.001714,
    0.001835, 0.001963, 0.002082, 0.002202, 0.00233,  0.002457, 0.002574, 0.002683,
    0.002787, 0.002881, 0.002974, 0.003074, 0.003175, 0.003295, 0.003444, 0.003608,
    0.00378,  0.003958, 0.004144, 0.004337, 0.00454,  0.004774, 0.005064, 0.005399,
    0.005796, 0.006214, 0.006671, 0.007167, 0.007736, 0.008351, 0.009035, 0.00977,
    0.010567, 0.011398, 0.012291, 0.013224, 0.014267, 0.015353, 0.016484, 0.017617,
    0.018759, 0.019914, 0.021104, 0.022423, 0.023847, 0.025357, 0.02705,  0.02897,
    0.031188, 0.033754, 0.036747, 0.040563, 0.044308, 0.048498, 0.053229, 0.058778,
    0.064617, 0.070947, 0.077834, 0.085686, 0.094809, 0.10509,  0.116592, 0.129306,
    0.142732, 0.157638, 0.174458, 0.193027, 0.21293,  0.232657, 0.251826, 0.270943,
    0.289756, 0.307998, 0.325393, 0.341662, 0.358746, 0.376683, 0.395517, 0.415293,
    0.436058, 0.45786,  0.480753, 0.504791, 0.530031, 0.556532,
]

def lt_qx(age: int) -> float:
    if age < 0: return 0.0
    if age < len(LIFE_TABLE_QX): return LIFE_TABLE_QX[age]
    return 0.6

# ── BLS 2023 median weekly earnings → annualised, by 10-yr age band ──────────
# Source: BLS Current Population Survey Table 5 (2023), all workers, both sexes.
# Used for human-capital productivity-loss calculations.

BLS_ANNUAL_EARNINGS: Dict[str, float] = {
    "16–24": 33_280,   # $640/wk × 52
    "25–34": 57_200,   # $1,100/wk
    "35–44": 65_520,   # $1,260/wk
    "45–54": 64_480,   # $1,240/wk
    "55–64": 60_320,   # $1,160/wk
    "65+":   49_400,   # $950/wk
}
# Maternal age distribution at ED presentation (assumed; adjust via sidebar)
MATERNAL_AGE_DIST: Dict[str, float] = {
    "16–24": 0.28,
    "25–34": 0.52,
    "35–44": 0.18,
    "45–54": 0.02,
    "55–64": 0.00,
    "65+":   0.00,
}
MATERNAL_WEIGHTED_EARNINGS: float = sum(
    MATERNAL_AGE_DIST[b] * BLS_ANNUAL_EARNINGS[b] for b in MATERNAL_AGE_DIST
)

# ── Gestational-age strata ────────────────────────────────────────────────────
GES_STRATA = {
    "<14w":   dict(w=0.20, p_uc=0.08, p_tx=0.95, prop_late=0.80),
    "14–27w": dict(w=0.35, p_uc=0.35, p_tx=0.88, prop_late=0.55),
    "28–36w": dict(w=0.30, p_uc=0.58, p_tx=0.72, prop_late=0.25),
    "≥37w":   dict(w=0.15, p_uc=0.78, p_tx=0.38, prop_late=0.05),
}

def ges_eff(strata: dict = None) -> Tuple[float, float, float]:
    """Return (eff_uc_screen, eff_tx_complete, eff_prop_late) weighted by gestational strata."""
    s = strata or GES_STRATA
    return (
        sum(v["w"] * v["p_uc"]       for v in s.values()),
        sum(v["w"] * v["p_tx"]       for v in s.values()),
        sum(v["w"] * v["prop_late"]  for v in s.values()),
    )

# ── Infant Markov parameters ──────────────────────────────────────────────────
INFANT_MK = {
    "p_severe_cs_comp":      dict(m=0.35, lo=0.20, hi=0.50),
    "p_mild_cs_comp":        dict(m=0.40, lo=0.25, hi=0.55),
    "p_mild_cs_uncomp":      dict(m=0.06, lo=0.02, hi=0.14),
    "dw_mild":               dict(m=0.110, lo=0.050, hi=0.210),
    "dw_severe":             dict(m=0.390, lo=0.260, hi=0.530),
    "cost_mild_ann":         dict(mu=8_500,  sd=2_500),
    "cost_sev_ann":          dict(mu=26_000, sd=7_500),
    "mu_bg":                 0.003,      # legacy; superseded by LIFE_TABLE_QX
    "q_progress":            0.002,      # calibrated via calibrate_q_progress()
    "q_progress_target":     0.20,       # target lifetime progression probability
    # Optional congenital-syphilis-attributable post-neonatal mortality.
    # Kept at 0 by default to avoid charging background life-table deaths as CS YLL;
    # non-zero values activate excess YLL accumulation inside the Markov loop.
    "mu_excess_mild":        dict(m=0.000, lo=0.000, hi=0.003),
    "mu_excess_severe":      dict(m=0.000, lo=0.000, hi=0.015),
    "cs_early_cure_rate":    0.95,
    "cs_late_manifest_rate": 0.20,
    "cs_neuro_disorder_rate":0.20,
}

# ── Baseline background outcome risks ─────────────────────────────────────────
BASE_BETA = {
    "preterm":        dict(a=1040, b=8960),
    "lbw":            dict(a=850,  b=9150),
    "stillbirth":     dict(a=55,   b=9945),
    "neonatal_death": dict(a=36,   b=9964),
    "miscarriage":    dict(a=1500, b=8500),
}

# ── Untreated syphilis absolute risks ────────────────────────────────────────
UNT_ABS = dict(
    preterm=0.232, lbw=0.234, stillbirth=0.264,
    miscarriage=0.149, neonatal_death=0.162, cs_any=0.360,
)

# ── Treatment relative risks ──────────────────────────────────────────────────
TX_RR = {
    "preterm":        dict(rr=0.48, lo=0.39, hi=0.58),
    "lbw":            dict(rr=0.50, lo=0.42, hi=0.59),
    "stillbirth":     dict(rr=0.21, lo=0.10, hi=0.35),
    "neonatal_death": dict(rr=0.20, lo=0.13, hi=0.32),
    "cs_any":         dict(rr=0.03, lo=0.02, hi=0.07),
}

# ── Disability weights ────────────────────────────────────────────────────────
DW_P = {
    "lbw":     dict(m=0.106, lo=0.035, hi=0.159, dur=0.25),
    # Acute infant morbidity. These are intentionally conservative defaults and
    # are exposed through the DALY component table so they can be audited.
    "preterm": dict(m=0.049, lo=0.020, hi=0.090, dur=0.25),
    # Maternal grief / acute maternal morbidity components.
    "mat_sb":  dict(m=0.740, lo=0.600, hi=0.800, dur=1.00),
    "mat_nnd": dict(m=0.658, lo=0.528, hi=0.768, dur=1.00),
    "miscarriage_grief": dict(m=0.110, lo=0.050, hi=0.200, dur=14.0 / 365.25),
    "mat_hosp": dict(m=0.133, lo=0.051, hi=0.264, dur=None),  # duration comes from MaternalMorbidity.dur_hosp_days
}

# ── Cost parameters ───────────────────────────────────────────────────────────
@dataclass
class Costs:
    poc:       float = 50.00;          poc_sd:       float = 10.00
    soc_work:  float = 500.00;         soc_work_sd:  float = 125.00
    rpr:       float = 9.82  * CPI;    rpr_sd:       float = std2(6.71,    26.85)  * CPI
    fta:       float = 31.07 * CPI;    fta_sd:       float = std2(20.14,   53.71)  * CPI
    pen:       float = 20.0;           pen_sd:       float = 4.0
    sf_wu:     float = 75.0;           sf_wu_sd:     float = 25.0
    staff:     float = 30.0;           staff_sd:     float = 10.0
    iufd:      float = 13_049 * CPI;   iufd_sd:      float = std2(10_742, 20_141) * CPI
    preterm:   float = 37_780 * CPI;   preterm_sd:   float = std2(26_855, 53_709) * CPI
    term_del:  float = 13_828 * CPI;   term_del_sd:  float = std2(6_714,  26_855) * CPI
    sb:        float = 141_792 * CPI;  sb_sd:        float = std2(120_846,201_410)* CPI
    nnd:       float = 189_784 * CPI;  nnd_sd:       float = std2(147_701,268_547)* CPI
    lbw_hs:    float = 64_086;         lbw_hs_sd:    float = std2(60_205,  67_891)
    nicu:      float = 50_000.0;       nicu_sd:      float = 10_000.0
    cs_wu:     float = 1_643.68 * CPI; cs_wu_sd:     float = std2(939.91,2_685.47)* CPI


@dataclass
class LongTermCare:
    """
    Direct long-term care costs for children with congenital syphilis sequelae.
    Distinct from ProductivityLoss (which captures caregiver *opportunity* costs).
    These are out-of-pocket / payer-facing direct expenditures.

    Special education
    -----------------
    IDEA Part B covers ages 3–21. Incremental cost above regular education
    estimated from SEEP (2003) and Chambers et al. (2010): ~$10K–$20K/yr
    additional, higher for severe developmental disability.
    Source: Chambers, J.G. et al. (2010). Special Education Expenditure Project.

    Direct caregiver costs
    ----------------------
    Paid home care, respite care, adaptive equipment, therapy co-pays.
    Distinct from informal caregiver time (modelled in ProductivityLoss).
    Genworth Cost of Care Survey (2023) and AHRQ estimates used as anchors.
    Most intensive in early childhood; modelled through age caregiver_end_age.
    """
    # Special education — incremental cost above regular schooling
    p_sped_severe:      float = 0.85    # P(special ed | severe CS sequelae)
    p_sped_severe_lo:   float = 0.70
    p_sped_severe_hi:   float = 0.95
    p_sped_mild:        float = 0.35    # P(special ed | mild CS sequelae)
    p_sped_mild_lo:     float = 0.15
    p_sped_mild_hi:     float = 0.55
    cost_sped_ann:      float = 14_000  # Incremental annual special ed cost ($2023)
    cost_sped_sd:       float = 4_000
    sped_start_age:     int   = 3       # IDEA Part B eligibility
    sped_end_age:       int   = 21      # IDEA coverage ceiling

    # Direct paid caregiver / support costs
    cost_cg_severe_ann: float = 32_000  # Paid home care + respite, severe sequelae
    cost_cg_severe_sd:  float = 9_000
    cost_cg_mild_ann:   float = 4_500   # Therapy co-pays + adaptive equipment, mild
    cost_cg_mild_sd:    float = 1_800
    caregiver_end_age:  int   = 18      # Most intensive direct care through childhood


# ── Scenario presets ──────────────────────────────────────────────────────────
PRESETS = {
    "Custom": {},
    "High-burden urban ED":    dict(p_act=0.015,  sc_e=0.92, p_adeq=0.80),
    "Moderate-burden (base)":  dict(p_act=0.0075, sc_e=0.90, p_adeq=0.85),
    "Low-prevalence rural ED": dict(p_act=0.001,  sc_e=0.85, p_adeq=0.75),
    "Best-case operations":    dict(p_act=0.010,  sc_e=0.95, p_adeq=0.95),
}


# ══════════════════════════════════════════════════════════════════════════════
# §3 · PRODUCTIVITY LOSS & MATERNAL MORBIDITY MODULES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MaternalMorbidity:
    """
    Maternal morbidity from untreated syphilis.
    DWs from GBD 2019; probabilities from CDC/WHO epidemiological reviews.
    All costs CPI-adjusted unless noted.
    """
    # Cardiovascular syphilis (late latent → tertiary)
    p_cardio:         float = 0.050   # P(cardiovascular event | untreated late latent)
    dw_cardio:        float = 0.070   # GBD 2019 DW for ischaemic heart disease (moderate)
    dur_cardio:       float = 5.0     # mean years with condition
    cost_cardio:      float = 18_500  # annual treatment cost (2025 USD)
    cost_cardio_sd:   float = 4_500
    # Neurosyphilis
    p_neuro:          float = 0.100   # P(neurosyphilis | untreated late latent; ~10% of late latent)
    dw_neuro:         float = 0.440   # GBD 2019 DW for neurological sequelae
    dur_neuro:        float = 8.0     # mean years
    cost_neuro:       float = 32_000  # annual (hospitalisation + outpatient)
    cost_neuro_sd:    float = 8_000
    # Pregnancy-related hospitalisation (chorioamnionitis, preterm labour)
    p_hosp:           float = 0.120   # P(hospitalisation episode | active infection)
    cost_hosp:        float = 4_200   # per episode
    cost_hosp_sd:     float = 1_100
    dur_hosp_days:    float = 3.2     # lost work-days per episode

    # PSA distribution shapes (Gamma for costs, Beta for probabilities)
    p_cardio_lo:      float = 0.020; p_cardio_hi:  float = 0.100
    p_neuro_lo:       float = 0.040; p_neuro_hi:   float = 0.200
    p_hosp_lo:        float = 0.070; p_hosp_hi:    float = 0.200


@dataclass
class ProductivityLoss:
    """
    Human-capital productivity losses averted by screening.
    BLS 2023 annual earnings; friction-cost variant available (friction_period_days).
    """
    # Maternal bereavement (neonatal death or stillbirth)
    bereavement_days:     float = 30.0    # mean lost work-days
    bereavement_days_sd:  float = 10.0
    # CS mild sequelae: lifetime wage penalty
    wage_penalty_mild:    float = 0.10    # 10% lifetime earnings reduction
    wage_penalty_mild_sd: float = 0.04
    # CS severe sequelae: lifetime wage penalty (caregiver + patient combined)
    wage_penalty_severe:  float = 0.40    # 40% reduction
    wage_penalty_severe_sd: float = 0.10
    # Friction cost period (days) — set to 0 to use full human-capital method
    friction_period_days: float = 0.0
    # Caregiver time cost for CS complicated infant (hours/week × wage)
    caregiver_hrs_wk:     float = 10.0   # additional weekly caregiver hours
    caregiver_hrs_wk_sd:  float = 4.0
    caregiver_wage_frac:  float = 0.50   # fraction of weighted maternal earnings

def _prod_loss_per_case(
    pl:          ProductivityLoss,   # carries friction_period_days, caregiver_wage_frac
    psa_pl:      dict,               # per-iteration draws (N,) arrays
    n_sb:        np.ndarray,
    n_nnd:       np.ndarray,
    n_cs_comp:   np.ndarray,
    n_cs_uncomp: np.ndarray,
    mk:          dict,
    r:           float,
    LE:          float,
    earnings:    float = MATERNAL_WEIGHTED_EARNINGS,
) -> np.ndarray:
    """
    Like _prod_loss_per_case but draws bereavement_days, wage_penalty_mild,
    wage_penalty_severe, and caregiver_hrs_wk from per-iteration PSA arrays
    rather than point estimates.
    """
    daily_wage = earnings / 260.0
    friction   = pl.friction_period_days

    # Bereavement — per-iteration draw (N,)
    bdays = psa_pl["bereavement_days"]
    if friction > 0:
        bdays = np.minimum(bdays, friction)
    bereavement = (n_sb + n_nnd) * bdays * daily_wage

    # Infant future earnings
    pv_infant = earnings * pvf(max(LE - 20, 0), r) * (1 + r) ** (-20)
    if friction > 0:
        pv_infant = min(pv_infant, friction * daily_wage)
    infant_earnings = n_nnd * pv_infant

    # Wage penalties — per-iteration draws
    pv_working = earnings * pvf(45, r) * (1 + r) ** (-20)
    if friction > 0:
        pv_working = min(pv_working, friction * daily_wage)

    cs_mild_loss = (n_cs_comp * mk["p_mild_cs_comp"]
                    + n_cs_uncomp * mk["p_mild_cs_uncomp"]) \
                   * psa_pl["wage_penalty_mild"] * pv_working   # ← per-iteration
    cs_sev_loss  = n_cs_comp * mk["p_severe_cs_comp"] \
                   * psa_pl["wage_penalty_severe"] * pv_working  # ← per-iteration

    # Caregiver — per-iteration draw
    caregiver_ann = (psa_pl["caregiver_hrs_wk"] * 52 * earnings / 2080
                     * pl.caregiver_wage_frac)                   # ← per-iteration
    caregiver_pv  = caregiver_ann * pvf(18, r)
    if friction > 0:
        caregiver_pv = np.minimum(caregiver_pv, friction * daily_wage)
    caregiver_total = n_cs_comp * caregiver_pv

    return bereavement + infant_earnings + cs_mild_loss + cs_sev_loss + caregiver_total


def _prod_loss_det(
    pl: ProductivityLoss,
    n_sb, n_nnd, n_cs_comp, n_cs_uncomp,
    r: float, LE: float,
    earnings: float = MATERNAL_WEIGHTED_EARNINGS,
) -> float:
    """Deterministic (scalar) productivity-loss savings."""
    daily = earnings / 260.0
    bdays = min(pl.bereavement_days, pl.friction_period_days) \
            if pl.friction_period_days > 0 else pl.bereavement_days
    bereavement  = (n_sb + n_nnd) * bdays * daily
    
    pv_infant    = earnings * pvf(max(LE - 20, 0), r) * (1 + r) ** (-20)
    if pl.friction_period_days > 0:
        pv_infant = min(pv_infant, pl.friction_period_days * daily)
    infant_earn  = n_nnd * pv_infant
    
    pv_working   = earnings * pvf(45, r) * (1 + r) ** (-20)

    if pl.friction_period_days > 0:
        pv_working = min(pv_working, pl.friction_period_days * daily)
        
    p_mc = INFANT_MK["p_mild_cs_comp"]["m"]
    p_mu = INFANT_MK["p_mild_cs_uncomp"]["m"]
    p_sv = INFANT_MK["p_severe_cs_comp"]["m"]
    cs_mild = (n_cs_comp * p_mc + n_cs_uncomp * p_mu) * pl.wage_penalty_mild * pv_working
    cs_sev  = n_cs_comp * p_sv * pl.wage_penalty_severe * pv_working
    
    cg_ann  = pl.caregiver_hrs_wk * 52 * earnings / 2080 * pl.caregiver_wage_frac
    cg_pv   = cg_ann * pvf(18, r)
    if pl.friction_period_days > 0:
        cg_pv = min(cg_pv, pl.friction_period_days * daily)
    cg_total = n_cs_comp * cg_pv
    
    return float(bereavement + infant_earn + cs_mild + cs_sev + cg_total)


def _mat_morb_dalys(
    mm: MaternalMorbidity,
    n_maternal_tx: np.ndarray,   # syphilis+ cases averted (N,)
    p_eff_delta: np.ndarray,
    r: float,
    psa_mm: dict,
    dw_hosp: np.ndarray = None,
    include_hosp_yld: bool = True,
    return_components: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maternal morbidity DALYs and costs averted (N,).
    p_eff_delta = effective screening coverage difference (scalar or N,).

    Hospitalisation episodes already generate a direct medical cost and a
    productivity-loss term. This function now also permits an acute YLD term
    for the hospitalised days so the societal DALY numerator is auditable.
    """
    # Fraction of syphilis+ who are late latent eligible for cardiovascular / neuro
    # (conservative: ~30% of active cases become late latent if untreated)
    p_late_latent = 0.30
    n_cardio = n_maternal_tx * p_late_latent * psa_mm["p_cardio"]
    n_neuro  = n_maternal_tx * p_late_latent * psa_mm["p_neuro"]
    n_hosp   = n_maternal_tx * psa_mm["p_hosp"]

    daly_cardio = n_cardio * mm.dw_cardio * pvf(mm.dur_cardio, r)
    daly_neuro  = n_neuro  * mm.dw_neuro  * pvf(mm.dur_neuro,  r)
    if include_hosp_yld:
        if dw_hosp is None:
            dw_hosp = np.full_like(n_maternal_tx, DW_P["mat_hosp"]["m"], dtype=float)
        hosp_dur = max(float(mm.dur_hosp_days), 0.0) / 365.25
        daly_hosp = n_hosp * dw_hosp * pvf(hosp_dur, r)
    else:
        daly_hosp = np.zeros_like(n_maternal_tx, dtype=float)

    cost_cardio = n_cardio * psa_mm["cost_cardio"] * pvf(mm.dur_cardio, r)
    cost_neuro  = n_neuro  * psa_mm["cost_neuro"]  * pvf(mm.dur_neuro,  r)
    cost_hosp   = n_hosp   * psa_mm["cost_hosp"]

    total_daly = daly_cardio + daly_neuro + daly_hosp
    total_cost = cost_cardio + cost_neuro + cost_hosp
    if return_components:
        comp = {
            "mat_cardio_dal": daly_cardio,
            "mat_neuro_dal":  daly_neuro,
            "mat_hosp_dal":   daly_hosp,
        }
        return total_daly, total_cost, comp
    return total_daly, total_cost


def _mat_morb_det(
    mm: MaternalMorbidity,
    n_maternal_tx: float,
    r: float,
    include_hosp_yld: bool = True,
) -> Tuple[float, float]:
    p_ll = 0.30
    n_cardio = n_maternal_tx * p_ll * mm.p_cardio
    n_neuro  = n_maternal_tx * p_ll * mm.p_neuro 
    n_hosp   = n_maternal_tx * mm.p_hosp
    dal_hosp = 0.0
    if include_hosp_yld:
        hosp_dur = max(float(mm.dur_hosp_days), 0.0) / 365.25
        dal_hosp = n_hosp * DW_P["mat_hosp"]["m"] * pvf(hosp_dur, r)
    dal  = (n_cardio * mm.dw_cardio * pvf(mm.dur_cardio, r) 
         + n_neuro  * mm.dw_neuro  * pvf(mm.dur_neuro,  r)
         + dal_hosp)
    cost = (n_cardio * mm.cost_cardio * pvf(mm.dur_cardio, r)
            + n_neuro * mm.cost_neuro * pvf(mm.dur_neuro, r)
            + n_hosp  * mm.cost_hosp)
    return float(dal), float(cost)
# ─────────────────────────────────────────────────────────────────────────────
# §BIA-A · DATACLASSES AND FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────



@dataclass
class BIAPopulation:
    """
    Population estimation module for ISPOR-compliant BIA.
    Provides a transparent funnel from covered lives to
    incremental patients effectively reached by the program.

    Parameters
    ----------
    covered_lives : float
        Total covered lives in the payer's book of business.
    frac_repro_female : float
        Fraction of covered lives who are female, aged 15–44.
        Default 0.135 (US actuarial average).
    pregnancy_rate : float
        Annual pregnancy rate among reproductive-age females.
        Default 0.085 (~8.5%, consistent with CDC NSFG).
    p_ed_visit : float
        P(at least one ED visit during pregnancy | pregnant).
        Literature range 0.35–0.55; default 0.45.
    p_unscreened : float
        P(not previously screened antenatally | ED presentation).
        Captures the incremental catchment above existing prenatal
        screening programmes. Higher in low-prenatal-care areas.
    payer_fraction : float
        Fraction of ED volume attributable to this payer.
        e.g. 0.40 for a Medicaid plan covering ~40% of ED visits.
    """
    covered_lives:     float = 100_000
    frac_repro_female: float = 0.135
    pregnancy_rate:    float = 0.085
    p_ed_visit:        float = 0.45
    p_unscreened:      float = 0.35
    payer_fraction:    float = 0.40


# ── Scenario presets ──────────────────────────────────────────────────────────
BIA_SCENARIOS: Dict[str, dict] = {
    "Conservative": dict(
        pop=BIAPopulation(p_ed_visit=0.35, p_unscreened=0.25, payer_fraction=0.35),
        t_half=2.5, t_ninety=4.5,
    ),
    "Base case": dict(
        pop=BIAPopulation(p_ed_visit=0.45, p_unscreened=0.35, payer_fraction=0.40),
        t_half=1.5, t_ninety=3.0,
    ),
    "Optimistic": dict(
        pop=BIAPopulation(p_ed_visit=0.55, p_unscreened=0.50, payer_fraction=0.45),
        t_half=1.0, t_ninety=2.0,
    ),
}

def common_population_funnel(pop: BIAPopulation) -> dict:
    """
    Shared population denominator for both CEA scaling and BIA.

    N_eligible should represent the population to whom incremental ED screening
    applies. In the current model, this is payer-covered pregnant ED patients
    who were not previously screened antenatally.
    """
    n_repro = pop.covered_lives * pop.frac_repro_female
    n_pregnant = n_repro * pop.pregnancy_rate
    n_ed_payer = n_pregnant * pop.p_ed_visit * pop.payer_fraction
    n_eligible = n_ed_payer * pop.p_unscreened

    return {
        "n_repro": n_repro,
        "n_pregnant": n_pregnant,
        "n_ed_payer": n_ed_payer,
        "n_eligible": n_eligible,
    }
def sigmoid_coverage_at_t(
    t:float,
    t_half:float,
    t_ninety:float,
    sc_uc:float,
    sc_e:float,) -> float:
    if t_ninety <= t_half or sc_e <= sc_uc:
        return float(sc_e)
    k = np.log(9.0) / (t_ninety - t_half)
    u = 1.0/(1.0 + np.exp(-k * (t - t_half)))
    eff_cov = sc_uc + (sc_e - sc_uc) * u
    return float(np.clip(eff_cov, sc_uc, sc_e))

def sigmoid_ramp(
    t_half:   float,
    t_ninety: float,
    n_years:  int,
    sc_uc: float,
    sc_e: float,
) -> Dict[int, float]:
    """
    Fit a sigmoid to two implementation milestones.

    Derivation
    ----------
    sigmoid(t) = 1 / (1 + exp(-k(t - t_half)))
    At t = t_half  : sigmoid = 0.50  (by definition)
    At t = t_ninety: sigmoid = 0.90
    => k = ln(9) / (t_ninety - t_half)

    The curve is normalised so coverage = target at t_ninety
    rather than asymptotically approaching it.

    Parameters
    ----------
    t_half   : year at which 50% of target coverage is achieved
    t_ninety : year at which 90% of target coverage is achieved
    n_years  : BIA projection horizon

    Returns
    -------
    Dict[year -> effective coverage]
    """
    return {
        yr: sigmoid_coverage_at_t(yr, t_half, t_ninety, sc_uc, sc_e) for yr in range(1, n_years + 1)
        }

def bia_population_funnel(
    pop:    BIAPopulation,
    year:   int,
    ramp:   Dict[int, float],
    sc_uc:  float,
    p_id:   float,
) -> dict:
    """
    Population flow for one BIA year.
    Returns all intermediate counts for display.

    Note: effective_coverage already incorporates the uptake ramp,
    so (effective_coverage - sc_uc) is the incremental coverage
    attributable to the programme in that year.
    """
    base = common_population_funnel(pop)
    
    n_repro = base["n_repro"]
    n_pregnant = base["n_pregnant"]
    n_ed = base["n_ed_payer"]
    n_unscreened = base["n_eligible"]
    
    eff_cov      = float(ramp.get(year, ramp[max(ramp)]))

    # Incremental patients reached above usual-care baseline
    n_intr = n_unscreened * eff_cov  * p_id
    n_uc   = n_unscreened * sc_uc   * p_id
    n_incr = float(np.maximum(n_intr - n_uc, 0.0))

    return {
        "year":          year,
        "n_repro":       n_repro,
        "n_pregnant":    n_pregnant,
        "n_ed":          n_ed,
        "n_unscreened":  n_unscreened,
        "n_eligible":    n_unscreened,
        "n_intr":        n_intr,
        "n_uc":          n_uc,
        "n_incremental": n_incr,
        "eff_coverage":  eff_cov,
    }
def bia_screening_cascade(
    n_screened:   float,
    p_act:        float,
    p_sf:         float,
    sens:         float,
    spec:         float,
    p_adeq:       float,
    tx_eff:       float,
    p_trepo_sf:   float,
    p_ux_sf:      float,
    treat_fp:     bool,
) -> dict:
    """
    Deterministic BIA cascade for a given number of screened patients.

    n_screened can be:
      - n_incremental for net budget impact, or
      - n_intr / n_uc if you want gross intervention and usual-care arms.
    """
    n_screened = float(max(n_screened, 0.0))

    p_sn = float(max(1.0 - p_act - p_sf, 0.0))

    # Active syphilis true positives
    n_tp_detected = n_screened * p_act * sens
    n_tp_treated  = n_tp_detected * p_adeq * tx_eff

    # Serofast / prior-treated positives
    n_sf_detected = n_screened * p_sf * p_trepo_sf

    # Interpret p_ux_sf as P(unnecessarily treated | serofast detected).
    # Since the UI label already says "treated", do not multiply again by p_adeq.
    n_sf_treated = n_sf_detected * p_ux_sf

    # Seronegative false positives
    n_fp_detected = n_screened * p_sn * (1.0 - spec)
    n_fp_treated  = n_fp_detected * p_adeq * tx_eff if treat_fp else 0.0

    n_confirmatory = n_tp_detected + n_sf_detected + n_fp_detected

    return {
        "n_screened":       n_screened,
        "n_tp_detected":    n_tp_detected,
        "n_tp_treated":     n_tp_treated,
        "n_sf_detected":    n_sf_detected,
        "n_sf_treated":     n_sf_treated,
        "n_fp_detected":    n_fp_detected,
        "n_fp_treated":     n_fp_treated,
        "n_confirmatory":   n_confirmatory,
        "n_treated_total":  n_tp_treated + n_sf_treated + n_fp_treated,
    }


def bia_annual_impact(
    funnel:     dict,
    co:         Costs,
    p_act:      float,
    sens:       float,
    spec:       float,
    p_adeq:     float,
    tx_eff:     float,
    prop_symp:  float,
    prop_late:  float,
    p_sf:       float,
    p_trepo_sf: float,
    p_ux_sf:    float,
    treat_fp:   bool,
) -> dict:
    """
    Within-horizon program costs and medical savings for one BIA year.

    This version explicitly separates:
      - true-positive active syphilis detection/treatment,
      - serofast detection and unnecessary treatment,
      - seronegative false positives,
      - clinical outcome savings from true-positive treatment only.
    """
    n_incr = funnel["n_incremental"]

    cas = bia_screening_cascade(
        n_screened = n_incr,
        p_act      = p_act,
        p_sf       = p_sf,
        sens       = sens,
        spec       = spec,
        p_adeq     = p_adeq,
        tx_eff     = tx_eff,
        p_trepo_sf = p_trepo_sf,
        p_ux_sf    = p_ux_sf,
        treat_fp   = treat_fp,
    )

    # ------------------------------------------------------------------
    # Program costs
    # ------------------------------------------------------------------

    # Screening tests for everyone incrementally screened
    cost_screening_tests = cas["n_screened"] * (co.poc + co.rpr)

    # Confirmatory treponemal test for detected true positives,
    # serofast positives, and seronegative false positives.
    cost_confirmatory = cas["n_confirmatory"] * co.fta

    # Staff/workflow cost for everyone incrementally screened
    cost_staff = cas["n_screened"] * co.staff

    # Treatment costs
    cost_tx_tp = cas["n_tp_treated"] * (co.pen + co.soc_work)
    cost_tx_sf = cas["n_sf_treated"] * (co.pen + co.soc_work)
    cost_tx_fp = cas["n_fp_treated"] * (co.pen + co.soc_work)

    cost_treatment = cost_tx_tp + cost_tx_sf + cost_tx_fp

    # Serofast workup cost, separate from serofast treatment
    cost_sf_workup = cas["n_sf_detected"] * co.sf_wu

    prog_total = (
        cost_screening_tests
        + cost_confirmatory
        + cost_staff
        + cost_treatment
        + cost_sf_workup
    )

    # ------------------------------------------------------------------
    # Outcomes averted
    # Only treated true-positive active syphilis cases generate benefit.
    # ------------------------------------------------------------------

    n_tx_tp = cas["n_tp_treated"]

    def averted(unt_p, rr):
        return n_tx_tp * unt_p * (1.0 - rr)

    n_cs_av  = averted(UNT_ABS["cs_any"],         TX_RR["cs_any"]["rr"])
    n_pt_av  = averted(UNT_ABS["preterm"],        TX_RR["preterm"]["rr"])
    n_sb_av  = averted(UNT_ABS["stillbirth"],     TX_RR["stillbirth"]["rr"])
    n_nnd_av = averted(UNT_ABS["neonatal_death"], TX_RR["neonatal_death"]["rr"])

    n_cs_comp   = n_cs_av * prop_symp
    n_cs_uncomp = n_cs_av * (1.0 - prop_symp)

    n_iufd    = n_sb_av * prop_late
    n_sb_term = float(max(n_sb_av - n_iufd, 0.0))

    # ------------------------------------------------------------------
    # Within-horizon medical savings
    # ------------------------------------------------------------------

    sav_cs = (
        n_cs_comp * (co.cs_wu + co.nicu)
        + n_cs_uncomp * co.cs_wu
    )

    sav_preterm = n_pt_av * co.preterm
    sav_sb      = n_iufd * co.iufd + n_sb_term * co.sb
    sav_nnd     = n_nnd_av * co.nnd

    sav_total = sav_cs + sav_preterm + sav_sb + sav_nnd

    net = prog_total - sav_total

    return {
        "year":                 funnel["year"],
        "eff_coverage":         funnel["eff_coverage"],

        # Population
        "n_incremental":        cas["n_screened"],

        # Active syphilis cascade
        "n_tp_detected":        cas["n_tp_detected"],
        "n_tp_treated":         cas["n_tp_treated"],

        # Serofast cascade
        "n_sf_detected":        cas["n_sf_detected"],
        "n_sf_treated":         cas["n_sf_treated"],

        # False-positive cascade
        "n_fp_detected":        cas["n_fp_detected"],
        "n_fp_treated":         cas["n_fp_treated"],

        # Total treatment count
        "n_treated_total":      cas["n_treated_total"],

        # Outcomes
        "n_cs_averted":         n_cs_av,
        "n_preterm_averted":    n_pt_av,
        "n_sb_averted":         n_sb_av,
        "n_nnd_averted":        n_nnd_av,

        # Cost breakdown
        "cost_screening_tests": cost_screening_tests,
        "cost_confirmatory":    cost_confirmatory,
        "cost_staff":           cost_staff,
        "cost_tx_tp":           cost_tx_tp,
        "cost_tx_sf":           cost_tx_sf,
        "cost_tx_fp":           cost_tx_fp,
        "cost_treatment":       cost_treatment,
        "cost_sf_workup":       cost_sf_workup,
        "program_cost":         prog_total,

        # Savings breakdown
        "sav_cs":               sav_cs,
        "sav_preterm":          sav_preterm,
        "sav_sb":               sav_sb,
        "sav_nnd":              sav_nnd,
        "medical_savings":      sav_total,

        # Net
        "net_impact":           net,
    }

def run_bia_scenario(
    pop:        BIAPopulation,
    t_half:     float,
    t_ninety:   float,
    n_years:    int,
    co:         Costs,
    sc_e:       float,
    sc_uc:      float,
    p_act:      float,
    p_id:       float,
    sens:       float,
    spec:       float,
    p_adeq:     float,
    tx_eff:     float,
    prop_symp:  float,
    prop_late:  float,
    p_sf:       float,
    p_trepo_sf: float,
    p_ux_sf:    float,
    treat_fp:   bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a full BIA scenario over n_years.

    Returns
    -------
    df_impact  : year-by-year costs, savings, net impact
    df_funnel  : year-by-year population funnel counts
    """
    ramp = sigmoid_ramp(t_half, t_ninety, n_years, sc_uc, sc_e)

    impact_rows = []
    funnel_rows = []
    cum = 0.0

    for yr in range(1, n_years + 1):
        funnel = bia_population_funnel(pop, yr, ramp, sc_uc, p_id)
        impact = bia_annual_impact(
            funnel, co,
            p_act      = p_act,
            sens       = sens,
            spec       = spec,
            p_adeq     = p_adeq,
            tx_eff     = tx_eff,
            prop_symp  = prop_symp,
            prop_late  = prop_late,
            p_sf       = p_sf,
            p_trepo_sf = p_trepo_sf,
            p_ux_sf    = p_ux_sf,
            treat_fp   = treat_fp,
        )
        cum += impact["net_impact"]
        impact["cumulative_net"] = cum
        impact_rows.append(impact)
        funnel_rows.append(funnel)

    return pd.DataFrame(impact_rows), pd.DataFrame(funnel_rows)



# ══════════════════════════════════════════════════════════════════════════════
# §4 · PSA ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _draw_all(N: int, rs: np.random.RandomState, co: Costs):
    br = {k: bd(v["a"], v["b"]).rvs(N, random_state=rs) for k, v in BASE_BETA.items()}
    ur = {k: bd(v * 1000, (1 - v) * 1000).rvs(N, random_state=rs) for k, v in UNT_ABS.items()}
    rr = {}
    for k, p in TX_RR.items():
        mu, sig = lnorm_ms(p["rr"], p["lo"], p["hi"])
        rr[k] = lognorm(s=sig, scale=np.exp(mu)).rvs(N, random_state=rs)
    dw = {}
    for k, p in DW_P.items():
        a, b = beta_ab(p["m"], p["lo"], p["hi"])
        dw[k] = bd(a, b).rvs(N, random_state=rs)
    d = asdict(co)
    cs = {}
    for k, v in d.items():
        if k.endswith("_sd"): continue
        sd = d.get(k + "_sd", 0.0)
        if sd > 0:
            sh, sc = gamma_ab(v, sd)
            cs[k] = gd(sh, scale=sc).rvs(N, random_state=rs)
        else:
            cs[k] = np.full(N, float(v))
    return br, ur, rr, dw, cs


def _draw_infant_mk(N: int, rs: np.random.RandomState) -> dict:
    mk = {}
    for key in ("p_severe_cs_comp", "p_mild_cs_comp", "p_mild_cs_uncomp",
                "dw_mild", "dw_severe"):
        p = INFANT_MK[key]
        a, b = beta_ab(p["m"], p["lo"], p["hi"])
        mk[key] = bd(a, b).rvs(N, random_state=rs)
    excess = np.maximum(mk["p_severe_cs_comp"] + mk["p_mild_cs_comp"] - 1.0, 0.0)
    mk["p_mild_cs_comp"]    -= excess * 0.5
    mk["p_severe_cs_comp"]  -= excess * 0.5
    for key in ("cost_mild_ann", "cost_sev_ann"):
        p = INFANT_MK[key]
        sh, sc = gamma_ab(p["mu"], p["sd"])
        mk[key] = gd(sh, scale=sc).rvs(N, random_state=rs)

    # Optional excess post-neonatal mortality attributable to CS sequelae.
    # If the mean is 0, keep the PSA draw exactly zero rather than allowing
    # beta_ab()'s numerical clipping to introduce artificial mortality.
    for key in ("mu_excess_mild", "mu_excess_severe"):
        p = INFANT_MK[key]
        if float(p.get("m", 0.0)) <= 0.0:
            mk[key] = np.zeros(N, dtype=float)
        else:
            a, b = beta_ab(p["m"], p["lo"], p["hi"])
            mk[key] = bd(a, b).rvs(N, random_state=rs)
    return mk


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 1 — New dataclass (add to §2, after Costs dataclass)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LongTermCare:
    """
    Direct long-term care costs for children with congenital syphilis sequelae.
    Distinct from ProductivityLoss (which captures caregiver *opportunity* costs).
    These are out-of-pocket / payer-facing direct expenditures.

    Special education
    -----------------
    IDEA Part B covers ages 3–21. Incremental cost above regular education
    estimated from SEEP (2003) and Chambers et al. (2010): ~$10K–$20K/yr
    additional, higher for severe developmental disability.
    Source: Chambers, J.G. et al. (2010). Special Education Expenditure Project.

    Direct caregiver costs
    ----------------------
    Paid home care, respite care, adaptive equipment, therapy co-pays.
    Distinct from informal caregiver time (modelled in ProductivityLoss).
    Genworth Cost of Care Survey (2023) and AHRQ estimates used as anchors.
    Most intensive in early childhood; modelled through age caregiver_end_age.
    """
    # Special education — incremental cost above regular schooling
    p_sped_severe:      float = 0.85    # P(special ed | severe CS sequelae)
    p_sped_severe_lo:   float = 0.70
    p_sped_severe_hi:   float = 0.95
    p_sped_mild:        float = 0.35    # P(special ed | mild CS sequelae)
    p_sped_mild_lo:     float = 0.15
    p_sped_mild_hi:     float = 0.55
    cost_sped_ann:      float = 14_000  # Incremental annual special ed cost ($2023)
    cost_sped_sd:       float = 4_000
    sped_start_age:     int   = 3       # IDEA Part B eligibility
    sped_end_age:       int   = 21      # IDEA coverage ceiling

    # Direct paid caregiver / support costs
    cost_cg_severe_ann: float = 32_000  # Paid home care + respite, severe sequelae
    cost_cg_severe_sd:  float = 9_000
    cost_cg_mild_ann:   float = 4_500   # Therapy co-pays + adaptive equipment, mild
    cost_cg_mild_sd:    float = 1_800
    caregiver_end_age:  int   = 18      # Most intensive direct care through childhood


def _draw_ltc(N: int, rs: np.random.RandomState, ltc: LongTermCare) -> dict:
    """PSA draws for long-term care parameters."""
    psa = {}
    # Special ed probabilities — Beta
    for key, lo, hi in [
        ("p_sped_severe", ltc.p_sped_severe_lo, ltc.p_sped_severe_hi),
        ("p_sped_mild",   ltc.p_sped_mild_lo,   ltc.p_sped_mild_hi),
    ]:
        a, b = beta_ab(getattr(ltc, key), lo, hi)
        psa[key] = bd(a, b).rvs(N, random_state=rs)
    # Costs — Gamma
    for key, mu, sd in [
        ("cost_sped_ann",      ltc.cost_sped_ann,      ltc.cost_sped_sd),
        ("cost_cg_severe_ann", ltc.cost_cg_severe_ann, ltc.cost_cg_severe_sd),
        ("cost_cg_mild_ann",   ltc.cost_cg_mild_ann,   ltc.cost_cg_mild_sd),
    ]:
        sh, sc = gamma_ab(mu, sd)
        psa[key] = gd(sh, scale=sc).rvs(N, random_state=rs)
    return psa

def _draw_mat_morb(N: int, rs: np.random.RandomState, mm: MaternalMorbidity) -> dict:
    """PSA draws for maternal morbidity parameters."""
    psa = {}
    for key, lo, hi in [
        ("p_cardio", mm.p_cardio_lo, mm.p_cardio_hi),
        ("p_neuro", mm.p_neuro_lo, mm.p_neuro_hi),
        ("p_hosp",   mm.p_hosp_lo,   mm.p_hosp_hi),
    ]:
        a, b = beta_ab(getattr(mm, key), lo, hi)
        psa[key] = bd(a, b).rvs(N, random_state=rs)
    for key, mu, sd in [
        ("cost_cardio", mm.cost_cardio, mm.cost_cardio_sd),
        ("cost_neuro",  mm.cost_neuro,  mm.cost_neuro_sd),
        ("cost_hosp",   mm.cost_hosp,   mm.cost_hosp_sd),
    ]:
        sh, sc = gamma_ab(mu, sd)
        psa[key] = gd(sh, scale=sc).rvs(N, random_state=rs)
    return psa

def _draw_serofast(
    N: int,
    rs: np.random.RandomState,
    p_sf: float,
    p_trepo_sf: float,
    p_ux_sf: float,
) -> dict:
    """
    PSA draws for serofast module parameters.
    Beta distributions fitted to point estimate ± assumed uncertainty range.
    Ranges reflect the sparse literature on serofast prevalence
    and clinical decision variability.
    """
    psa = {}

    # p_sf: serofast prevalence — wide uncertainty, ±50% of point estimate
    a, b = beta_ab(p_sf,
                   lo=max(p_sf * 0.50, 0.001),
                   hi=min(p_sf * 1.50, 0.15))
    psa["p_sf"] = bd(a, b).rvs(N, random_state=rs)

    # p_trepo_sf: P(treponemal+ | serofast) — high but uncertain
    a, b = beta_ab(p_trepo_sf, lo=0.70, hi=0.99)
    psa["p_trepo_sf"] = bd(a, b).rvs(N, random_state=rs)

    # p_ux_sf: P(unnecessary treatment | serofast detected) — high variability
    a, b = beta_ab(p_ux_sf,
                   lo=max(p_ux_sf - 0.15, 0.01),
                   hi=min(p_ux_sf + 0.15, 0.60))
    psa["p_ux_sf"] = bd(a, b).rvs(N, random_state=rs)

    return psa

def _draw_prod_loss(N: int, rs: np.random.RandomState, pl: ProductivityLoss) -> dict:
    """PSA draws for productivity-loss parameters."""
    psa = {}
    for key, mu, sd in [
        ("bereavement_days",    pl.bereavement_days,    pl.bereavement_days_sd),
        ("caregiver_hrs_wk",    pl.caregiver_hrs_wk,    pl.caregiver_hrs_wk_sd),
        ("wage_penalty_mild",   pl.wage_penalty_mild,   pl.wage_penalty_mild_sd),
        ("wage_penalty_severe", pl.wage_penalty_severe, pl.wage_penalty_severe_sd),
    ]:
        sh, sc = gamma_ab(mu, sd)
        psa[key] = gd(sh, scale=sc).rvs(N, random_state=rs)
    # Build a ProductivityLoss-like object from PSA draws for vectorised use
    return psa


def _arm(sc, p_act, p_id, sens, p_adeq, p_tx, cohort,
         br, ur, rr, prop_symp, prop_late):
    """Simulate one arm (vectorised over N PSA draws)."""
    p_eff = sc * p_id * sens * p_adeq * p_tx

    # Linear subgroup mixture — no conditional hierarchy needed because all
    # ur values are unconditional P(outcome | syphilis+ pregnancy).
    def mix(unt, rr_key):
        tx = np.minimum(unt * rr[rr_key], 1.0)
        return p_eff * tx + (1.0 - p_eff) * unt

    sb_syph   = mix(ur["stillbirth"],     "stillbirth")
    neo_syph  = mix(ur["neonatal_death"], "neonatal_death")
    cs_syph   = mix(ur["cs_any"],         "cs_any")
    pt_syph   = mix(ur["preterm"],        "preterm")
    lbw_syph  = mix(ur["lbw"],            "lbw")
    misc_syph = ur["miscarriage"]   # no RR available; use untreated rate

    sb_rate   = p_act * sb_syph  + (1.0 - p_act) * br["stillbirth"]
    neo_rate  = p_act * neo_syph + (1.0 - p_act) * br["neonatal_death"]
    cs_rate   = p_act * cs_syph  # syphilis-specific; no meaningful background rate
    pt_rate   = p_act * pt_syph  + (1.0 - p_act) * br["preterm"]
    lbw_rate  = p_act * lbw_syph + (1.0 - p_act) * br["lbw"]
    misc_rate = p_act * misc_syph + (1.0 - p_act) * br["miscarriage"]

    def cnt(x): return np.maximum(x, 0.0) * cohort
    return {
        "preterm":        cnt(pt_rate),
        "lbw":            cnt(lbw_rate),
        "stillbirth":     cnt(sb_rate),
        "miscarriage":    cnt(misc_rate),
        "neonatal_death": cnt(neo_rate),
        "cs_comp":        cnt(cs_rate * prop_symp),
        "cs_uncomp":      cnt(cs_rate * (1.0 - prop_symp)),
        "iufd_subset":    cnt(sb_rate * prop_late),
    }

def _infant_markov_lifetime(n_cs_comp, n_cs_uncomp, mk, r_disc, T,
                            ltc: LongTermCare = None, psa_ltc: dict = None,
                            include_cs_yll: bool = True):
    """
    Vectorised infant lifetime Markov — states: Healthy/Mild/Severe/Dead.
    Long-term care costs (special education + direct caregiver) are accumulated
    in separate age-gated cycles and added to medical costs.

    DALYs now have two auditable parts:
      * YLD while alive in mild/severe states.
      * YLL from optional CS-attributable excess post-neonatal mortality.

    Background life-table deaths are *not* charged as CS YLL. To activate
    post-neonatal YLL, set INFANT_MK["mu_excess_mild"] and/or
    INFANT_MK["mu_excess_severe"] above zero.
    """
    N    = len(n_cs_comp)
    q    = INFANT_MK["q_progress"]
    dw_m = mk["dw_mild"]; dw_s = mk["dw_severe"]
    c_m  = mk["cost_mild_ann"]; c_s = mk["cost_sev_ann"]
    mu_x_m = mk.get("mu_excess_mild", np.zeros(N, dtype=float))
    mu_x_s = mk.get("mu_excess_severe", np.zeros(N, dtype=float))
    if not include_cs_yll:
        mu_x_m = np.zeros(N, dtype=float)
        mu_x_s = np.zeros(N, dtype=float)

    if ltc is not None:
        p_sped_s   = psa_ltc["p_sped_severe"]   if psa_ltc else np.full(N, ltc.p_sped_severe)
        p_sped_m   = psa_ltc["p_sped_mild"]     if psa_ltc else np.full(N, ltc.p_sped_mild)
        c_sped     = psa_ltc["cost_sped_ann"]   if psa_ltc else np.full(N, ltc.cost_sped_ann)
        c_cg_sev   = psa_ltc["cost_cg_severe_ann"] if psa_ltc else np.full(N, ltc.cost_cg_severe_ann)
        c_cg_mild  = psa_ltc["cost_cg_mild_ann"]   if psa_ltc else np.full(N, ltc.cost_cg_mild_ann)
        sped_start = ltc.sped_start_age
        sped_end   = ltc.sped_end_age
        cg_end     = ltc.caregiver_end_age
    else:
        p_sped_s = p_sped_m = c_sped = c_cg_sev = c_cg_mild = np.zeros(N)
        sped_start = sped_end = cg_end = 0

    def _run(S0):
        S = S0.copy().astype(float)
        yld = np.zeros(N, dtype=float)
        yll = np.zeros(N, dtype=float)
        cst = np.zeros(N, dtype=float)
        for t in range(T):
            disc   = (1.0 + r_disc) ** (-t)
            mu_t   = lt_qx(t)

            # DALY YLD while alive with sequelae
            yld += (S[:, 1] * dw_m + S[:, 2] * dw_s) * disc

            # DALY YLL from CS-attributable excess mortality only.
            mx_m = np.minimum(np.asarray(mu_x_m, dtype=float), np.maximum(1.0 - mu_t - q, 0.0))
            mx_s = np.minimum(np.asarray(mu_x_s, dtype=float), np.maximum(1.0 - mu_t, 0.0))
            excess_deaths = S[:, 1] * mx_m + S[:, 2] * mx_s
            yll += excess_deaths * disc * pvf(max(T - t, 0), r_disc)

            # Medical costs (existing)
            med = S[:, 1] * c_m + S[:, 2] * c_s

            # Special education — age-gated
            sped_active = float(sped_start <= t < sped_end)
            sped = sped_active * c_sped * (
                S[:, 1] * p_sped_m + S[:, 2] * p_sped_s
            )

            # Direct caregiver costs — age-gated
            cg_active = float(t < cg_end)
            cg = cg_active * (S[:, 1] * c_cg_mild + S[:, 2] * c_cg_sev)

            cst += (med + sped + cg) * disc

            # Transition. Background mortality is retained for state occupancy;
            # excess mortality is the only portion charged as CS YLL above.
            S_new = np.zeros_like(S)
            S_new[:, 0] += S[:, 0] * (1.0 - mu_t)
            S_new[:, 3] += S[:, 0] * mu_t
            rm = np.maximum(1.0 - mu_t - q - mx_m, 0.0)
            S_new[:, 1] += S[:, 1] * rm
            S_new[:, 2] += S[:, 1] * q
            S_new[:, 3] += S[:, 1] * (mu_t + mx_m)
            rs = np.maximum(1.0 - mu_t - mx_s, 0.0)
            S_new[:, 2] += S[:, 2] * rs
            S_new[:, 3] += S[:, 2] * (mu_t + mx_s)
            S_new[:, 3] += S[:, 3]
            S = S_new
        return yld + yll, cst, yld, yll

    p_sev = mk["p_severe_cs_comp"]; p_mc = mk["p_mild_cs_comp"]
    p_h_c = np.maximum(1.0 - p_sev - p_mc, 0.0)
    dpc, cpc, yld_pc, yll_pc = _run(np.stack([p_h_c, p_mc, p_sev, np.zeros(N)], axis=1))

    p_mu  = mk["p_mild_cs_uncomp"]; p_h_u = np.maximum(1.0 - p_mu, 0.0)
    dpu, cpu, yld_pu, yll_pu = _run(np.stack([p_h_u, p_mu, np.zeros(N), np.zeros(N)], axis=1))

    n_comp = n_cs_comp.astype(float)
    n_uncomp = n_cs_uncomp.astype(float)
    dal_total = n_comp * dpc + n_uncomp * dpu
    cst_total = n_comp * cpc + n_uncomp * cpu
    yld_total = n_comp * yld_pc + n_uncomp * yld_pu
    yll_total = n_comp * yll_pc + n_uncomp * yll_pu
    return dal_total, cst_total, yld_total, yll_total

def _dalys_non_cs(
    d, dw, r, LE, inc_lbw, inc_mat,
    inc_sb_yll: bool = True,
    inc_misc_yld: bool = True,
    inc_preterm_yld: bool = True,
    return_components: bool = False,
):
    """
    Non-CS DALYs averted, decomposed for auditability.

    Components:
      - neonatal-death YLL
      - stillbirth YLL for IUFD >=28w (uses iufd_subset, not all >=20w SB)
      - LBW YLD, if enabled by the legacy toggle
      - preterm acute infant YLD, if enabled
      - maternal grief YLD for SB/NND, if enabled by the legacy toggle
      - miscarriage grief YLD, if enabled and if miscarriage events differ by arm
    """
    af  = lambda t: pvf(t, r)
    ref = d["neonatal_death"].astype(float)
    zero = np.zeros_like(ref, dtype=float)

    nnd_yll = ref * af(LE)
    # Use the late-gestation subset to avoid assigning full infant YLL to earlier
    # pregnancy losses that are included in the broader stillbirth count.
    sb_base = d.get("iufd_subset", d["stillbirth"]).astype(float)
    stillbirth_yll = sb_base * af(LE) if inc_sb_yll else zero.copy()
    lbw_yld = (d["lbw"].astype(float) * dw["lbw"] * af(DW_P["lbw"]["dur"])
               if inc_lbw else zero.copy())
    preterm_yld = (d["preterm"].astype(float) * dw["preterm"] * af(DW_P["preterm"]["dur"])
                   if inc_preterm_yld else zero.copy())
    miscarriage_yld = (d["miscarriage"].astype(float) * dw["miscarriage_grief"]
                       * af(DW_P["miscarriage_grief"]["dur"])
                       if inc_misc_yld else zero.copy())
    mat_sb_grief = zero.copy()
    mat_nnd_grief = zero.copy()
    if inc_mat:
        mat_sb_grief  = d["stillbirth"].astype(float) * dw["mat_sb"]  * af(DW_P["mat_sb"]["dur"])
        mat_nnd_grief = d["neonatal_death"].astype(float) * dw["mat_nnd"] * af(DW_P["mat_nnd"]["dur"])

    comp = {
        "nnd_yll": nnd_yll,
        "stillbirth_yll": stillbirth_yll,
        "lbw_yld": lbw_yld,
        "preterm_yld": preterm_yld,
        "miscarriage_yld": miscarriage_yld,
        "mat_sb_grief_yld": mat_sb_grief,
        "mat_nnd_grief_yld": mat_nnd_grief,
    }
    tot = sum(comp.values())
    if return_components:
        return tot.astype(float), {k: v.astype(float) for k, v in comp.items()}
    return tot.astype(float)


def _serofast_cost(n_screened, p_sf, p_trepo, p_ux, rpr, sf_wu, pen, soc_work):
    """
    Incremental serofast workup cost.
    n_screened: patients *incrementally* screened above usual-care baseline
    BIA uses full ED volume
    """
    n_sf = np.asarray(n_screened, dtype=float) * p_sf * p_trepo
    return n_sf * (sf_wu + p_ux * (pen + soc_work))


def _icost(d, cs, sf_cost, sc_b, sc_e, p_act, p_sf, p_id,
           sens, spec, p_adeq, tx_eff,          # ← tx_eff added
           treat_fp, cohort,
           mk_lt_cost_saving,
           mat_cost_saving=None,
           prod_loss_saving=None):
    """
    Health-sector incremental cost.
    tx_eff: treatment completion rate — should match the value used in _arm()
            so that costs and benefits are charged to the same patient set.
    """
    extra   = max(sc_e * p_id - sc_b * p_id, 0.0) * cohort
    p_sn    = max(1.0 - p_act - p_sf, 0.0)
    p_fp    = p_sn * (1.0 - spec)

    # n_tx: patients who complete treatment (generates both cost AND benefit)
    n_tx    = extra * p_act * sens * p_adeq * tx_eff   # ← tx_eff applied
    if treat_fp:
        n_tx = n_tx + extra * p_fp * p_adeq * tx_eff  # ← tx_eff applied to FPs too

    test    = (extra * cs["poc"] + extra * cs["rpr"]
               + extra * (p_act * sens + p_fp) * cs["fta"])
    prog    = (test + n_tx * (cs["pen"] + cs["soc_work"])
               + extra * cs["staff"] + sf_cost)

    n_iufd    = d["iufd_subset"].astype(float)
    n_sb_term = np.maximum(d["stillbirth"].astype(float) - n_iufd, 0.0)
    sav = (n_iufd      * cs["iufd"]
           + n_sb_term * cs["sb"]
           + d["neonatal_death"] * cs["nnd"]
           + d["lbw"]            * cs["lbw_hs"]
           + d["preterm"]        * cs["preterm"]
           + d["cs_comp"]   * (cs["cs_wu"] + cs["nicu"])
           + d["cs_uncomp"] * cs["cs_wu"])
    ic_hs  = (prog - sav - mk_lt_cost_saving).astype(float)
    ic_soc = ic_hs.copy()
    if mat_cost_saving  is not None: ic_soc -= mat_cost_saving
    if prod_loss_saving is not None: ic_soc -= prod_loss_saving
    return ic_hs, ic_soc

# ── q_progress calibration ────────────────────────────────────────────────────

def calibrate_q_progress(target_prob: float, T: int, tol: float = 1e-6) -> float:
    """
    Solve for annual mild→severe transition rate q such that cumulative
    lifetime progression probability equals target_prob, accounting for
    competing mortality from US Life Table 2021.

    Implemented via Brent's method on the implicit equation:
        P(ever progress | alive at birth) = target_prob

    Returns the calibrated q value.
    """
    def _lifetime_prog(q):
        """Expected fraction of a mild-sequelae cohort that progresses to severe."""
        S_mild = 1.0   # start with 100% mild
        ever_prog = 0.0
        for t in range(T):
            mu_t = lt_qx(t)
            prog_this_year = S_mild * q
            ever_prog += prog_this_year
            S_mild *= max(1 - mu_t - q, 0.0)
        return ever_prog

    try:
        q_cal = brentq(lambda q: _lifetime_prog(q) - target_prob,
                       1e-6, 0.10, xtol=tol, maxiter=200)
    except ValueError:
        q_cal = INFANT_MK["q_progress"]  # fallback to default
    return float(q_cal)


def implied_lifetime_prog(q: float, T: int) -> float:
    """Compute implied cumulative progression probability given q and life table."""
    S_mild = 1.0; ever_prog = 0.0
    for t in range(T):
        mu_t = lt_qx(t)
        ever_prog += S_mild * q
        S_mild    *= max(1 - mu_t - q, 0.0)
    return ever_prog


@st.cache_data(show_spinner=False)
def run_psa(
    N, seed, cohort,
    p_act, p_sf, p_id, sc_b, sc_e,
    sens, spec, p_adeq, p_tx_override,
    p_trepo_sf, p_ux_sf,
    prop_symp, prop_late,
    r, LE, inc_lbw, inc_mat,
    inc_sb_yll, inc_cs_yll, inc_misc_yld, inc_mat_hosp_yld, inc_preterm_yld,
    treat_fp,
    vsl,
    use_mat_morb, use_prod_loss, use_friction,
    mm_p_cardio, mm_p_neuro, mm_p_hosp,
    mm_cost_cardio, mm_cost_neuro, mm_cost_hosp,
    pl_bereavement_days, pl_wage_mild, pl_wage_severe,
    pl_caregiver_hrs, pl_caregiver_wage_frac,
    # ── NEW: Markov slider values ──────────────────────────────────────────
    mk_p_sev,      # INFANT_MK["p_severe_cs_comp"]["m"]
    mk_p_mc,       # INFANT_MK["p_mild_cs_comp"]["m"]
    mk_p_mu,       # INFANT_MK["p_mild_cs_uncomp"]["m"]
    mk_c_mild,     # INFANT_MK["cost_mild_ann"]["mu"]
    mk_c_sev,      # INFANT_MK["cost_sev_ann"]["mu"]
    mk_q_target,   # INFANT_MK["q_progress_target"]
    mk_mu_x_mild,  # INFANT_MK["mu_excess_mild"]["m"]
    mk_mu_x_sev,   # INFANT_MK["mu_excess_severe"]["m"]
    use_ltc, ltc_p_sped_sev, ltc_p_sped_mid, ltc_cost_sped,
    ltc_cost_cg_sv, ltc_cost_cg_ml, ltc_sped_start, ltc_sped_end, ltc_cg_end,
):
    # ── Push Markov slider values into global before any draws ───────────
    # This makes run_psa() self-contained: even if the sidebar mutation
    # order changes, the correct values are guaranteed to be in INFANT_MK
    # when _draw_infant_mk() and _infant_markov_lifetime() read them.
    INFANT_MK["p_severe_cs_comp"]["m"] = mk_p_sev
    INFANT_MK["p_mild_cs_comp"]["m"]   = mk_p_mc
    INFANT_MK["p_mild_cs_uncomp"]["m"] = mk_p_mu
    INFANT_MK["cost_mild_ann"]["mu"]   = mk_c_mild
    INFANT_MK["cost_sev_ann"]["mu"]    = mk_c_sev
    INFANT_MK["q_progress_target"]     = mk_q_target
    INFANT_MK["mu_excess_mild"]["m"]   = mk_mu_x_mild
    INFANT_MK["mu_excess_severe"]["m"] = mk_mu_x_sev
    # Recalibrate q_progress to match the (possibly updated) target
    T_calib = max(int(LE), 1)
    INFANT_MK["q_progress"] = calibrate_q_progress(mk_q_target, T_calib)

    # ── Rest of body is unchanged from original run_psa() ────────────────
    rs = np.random.RandomState(seed)
    br, ur, rr, dw, cs = _draw_all(N, rs, Costs())
    mk = _draw_infant_mk(N, rs)

    mm = MaternalMorbidity(
        p_cardio=mm_p_cardio, p_neuro=mm_p_neuro, p_hosp=mm_p_hosp,
        cost_cardio=mm_cost_cardio, cost_neuro=mm_cost_neuro, cost_hosp=mm_cost_hosp,
    )
    pl = ProductivityLoss(
        bereavement_days=pl_bereavement_days,
        wage_penalty_mild=pl_wage_mild, wage_penalty_severe=pl_wage_severe,
        caregiver_hrs_wk=pl_caregiver_hrs, caregiver_wage_frac=pl_caregiver_wage_frac,
        friction_period_days=90.0 if use_friction else 0.0,
    )

    ltc = LongTermCare(
        p_sped_severe      = ltc_p_sped_sev,
        p_sped_mild        = ltc_p_sped_mid,
        cost_sped_ann      = ltc_cost_sped,
        cost_cg_severe_ann = ltc_cost_cg_sv,
        cost_cg_mild_ann   = ltc_cost_cg_ml,
        sped_start_age     = int(ltc_sped_start),
        sped_end_age       = int(ltc_sped_end),
        caregiver_end_age  = int(ltc_cg_end),
    ) if use_ltc else None

    psa_ltc = _draw_ltc(N, rs, ltc) if use_ltc else None
    
    psa_mm = _draw_mat_morb(N, rs, mm)
    psa_pl = _draw_prod_loss(N, rs, pl)  
    psa_sf = _draw_serofast(N, rs, p_sf, p_trepo_sf, p_ux_sf)

    sc_uc, tx_eff, _ = ges_eff()
    if p_tx_override is not None: tx_eff = p_tx_override

    comp = _arm(sc_uc, p_act, p_id, sens, p_adeq, tx_eff, cohort,
                br, ur, rr, prop_symp, prop_late)
    intr = _arm(sc_e,  p_act, p_id, sens, p_adeq, tx_eff, cohort,
                br, ur, rr, prop_symp, prop_late)
    dlt  = {k: comp[k] - intr[k] for k in comp}

    comp_means = {k: float(v.mean()) for k, v in comp.items()}
    intr_means = {k: float(v.mean()) for k, v in intr.items()}

    T    = max(int(LE), 1)

    mk_dal, mk_cst, mk_yld, mk_yll = _infant_markov_lifetime(
        dlt["cs_comp"], dlt["cs_uncomp"], mk, r, T,
        ltc=ltc, psa_ltc=psa_ltc, include_cs_yll=bool(inc_cs_yll),
    )
    
    n_cases_averted = np.maximum(dlt["cs_comp"] + dlt["cs_uncomp"]
                                 + dlt["stillbirth"] + dlt["neonatal_death"], 0).astype(float)
    p_eff_delta = max(sc_e - sc_uc, 0.0) * p_id * sens * p_adeq * tx_eff
    n_maternal_tx = np.full(N, cohort * p_act * p_eff_delta, dtype=float)

    if use_mat_morb:
        mm_dal, mm_cost, mm_components = _mat_morb_dalys(
            mm, n_maternal_tx, p_eff_delta, r, psa_mm,
            dw_hosp=dw.get("mat_hosp"),
            include_hosp_yld=bool(inc_mat_hosp_yld),
            return_components=True,
        )
    else:
        mm_dal = mm_cost = np.zeros(N, dtype=float)
        mm_components = {
            "mat_cardio_dal": np.zeros(N, dtype=float),
            "mat_neuro_dal":  np.zeros(N, dtype=float),
            "mat_hosp_dal":   np.zeros(N, dtype=float),
        }

    if use_prod_loss:
        prod_sav = _prod_loss_per_case(   # ← Fix 5: uses PSA draws
            pl=pl, psa_pl=psa_pl,
            n_sb=dlt["stillbirth"].astype(float),
            n_nnd=dlt["neonatal_death"].astype(float),
            n_cs_comp=dlt["cs_comp"].astype(float),
            n_cs_uncomp=dlt["cs_uncomp"].astype(float),
            mk=mk, r=r, LE=LE,
        )
    else:
        prod_sav = np.zeros(N)

    n_incr_screened = np.full(
        N, max(sc_e * p_id - sc_uc * p_id, 0.0) * cohort, dtype=float,
    )

    sf = _serofast_cost(
        n_incr_screened,
        psa_sf["p_sf"],       # shape (N,)
        psa_sf["p_trepo_sf"], # shape (N,)
        psa_sf["p_ux_sf"],    # shape (N,)
        cs["rpr"],            # shape (N,) 
        cs["sf_wu"],          # shape (N,)
        cs["pen"],            # shape (N,)
        cs["soc_work"],       # shape (N,)
    )

    # Health-sector DALYs include infant outcomes, LBW/preterm YLD, maternal
    # grief YLD if selected, stillbirth YLL if selected, and congenital-syphilis
    # lifetime sequelae from the Markov model. Societal DALYs then add only
    # societal-perspective health effects not already counted in the health-sector
    # denominator. Productivity losses remain on the cost side.
    dal_non_cs_hs, non_cs_components = _dalys_non_cs(
        dlt, dw, r, LE, inc_lbw, inc_mat,
        inc_sb_yll=bool(inc_sb_yll),
        inc_misc_yld=bool(inc_misc_yld),
        inc_preterm_yld=bool(inc_preterm_yld),
        return_components=True,
    )
    dal_hs  = dal_non_cs_hs + mk_dal
    soc_daly_increment = mm_dal if use_mat_morb else np.zeros(N, dtype=float)
    dal_soc = dal_hs + soc_daly_increment

    # Guard against a silent regression where the societal denominator mirrors
    # the health-sector denominator despite the maternal morbidity module being on.
    if use_mat_morb and np.allclose(soc_daly_increment, 0.0):
        warnings.warn(
            "Societal DALYs are identical to health-sector DALYs because "
            "the maternal morbidity DALY increment is zero. Check maternal "
            "morbidity parameters and n_maternal_tx.",
            RuntimeWarning,
        )

    ic_hs, ic_soc = _icost(dlt, cs, sf, sc_uc, sc_e, p_act, p_sf, p_id,
                            sens, spec, p_adeq, tx_eff,    # ← Fix 2: tx_eff added
                            treat_fp, cohort, mk_cst,
                            mat_cost_saving  = mm_cost  if use_mat_morb  else None,
                            prod_loss_saving = prod_sav if use_prod_loss else None)

    vsl_nb = vsl * (dlt["stillbirth"] + dlt["neonatal_death"]).astype(float) - ic_hs

    eps = 1e-12
    df  = pd.DataFrame({
        "dal_hs":   dal_hs.astype(float),
        "dal_soc":  dal_soc.astype(float),
        "ic_hs":    ic_hs.astype(float),
        "ic_soc":   ic_soc.astype(float),
        "icer_hs":  (ic_hs  / np.maximum(dal_hs,  eps)).astype(float),
        "icer_soc": (ic_soc / np.maximum(dal_soc, eps)).astype(float),
        "vsl_nb":   vsl_nb.astype(float),
        "sf_cost":  sf.astype(float),
        "mk_dal":   mk_dal.astype(float),
        "mk_yld":   mk_yld.astype(float),
        "mk_yll":   mk_yll.astype(float),
        "mk_cst":   mk_cst.astype(float),
        "mm_dal":   mm_dal.astype(float),
        "soc_daly_increment": soc_daly_increment.astype(float),
        "mm_cost":  mm_cost.astype(float),
        "prod_sav": prod_sav.astype(float),
        **{f"dal_{k}": v.astype(float) for k, v in non_cs_components.items()},
        **{f"dal_{k}": v.astype(float) for k, v in mm_components.items()},
        **{f"rr_{k}":  v for k, v in rr.items()},
        **{f"ur_{k}":  v for k, v in ur.items()},
        **{f"br_{k}":  v for k, v in br.items()},
        **{f"dw_{k}":  v for k, v in dw.items()},
        **{f"co_{k}":  v for k, v in cs.items()},
        **{f"mkp_{k}": v for k, v in mk.items()},
        **{f"mmp_{k}": v for k, v in psa_mm.items()},
        **{f"d_{k}":   v.astype(float) for k, v in dlt.items()},
        **{f"sf_{k}": v for k, v in psa_sf.items()},
    })
    df["n_maternal_tx"] = n_maternal_tx.astype(float)

    smry = {
        "inc_cost_hs":        summarize(ic_hs),
        "inc_cost_soc":       summarize(ic_soc),
        "dalys_hs":           summarize(dal_hs),
        "dalys_soc":          summarize(dal_soc),
        "dalys_soc_increment": summarize(soc_daly_increment),
        "dalys_non_cs":       summarize(dal_non_cs_hs),
        "dalys_markov":       summarize(mk_dal),
        "dalys_markov_yld":   summarize(mk_yld),
        "dalys_markov_yll":   summarize(mk_yll),
        "dalys_mat_morb":     summarize(mm_dal),
        **{f"dalys_{k}": summarize(v) for k, v in non_cs_components.items()},
        **{f"dalys_{k}": summarize(v) for k, v in mm_components.items()},
        "icer_hs":            summarize(df.icer_hs[np.isfinite(df.icer_hs)]),
        "icer_soc":           summarize(df.icer_soc[np.isfinite(df.icer_soc)]),
        "vsl_nb":             summarize(vsl_nb),
        "p_cost_saving_hs":   float((ic_hs  < 0).mean()),
        "p_dominant_hs":      float(((ic_hs  < 0) & (dal_hs  > 0)).mean()),
        "p_cost_saving_soc":  float((ic_soc < 0).mean()),
        "p_dominant_soc":     float(((ic_soc < 0) & (dal_soc > 0)).mean()),
        "sf_cost":            summarize(sf),
        "mk_cst":             summarize(mk_cst),
        "prod_sav":           summarize(prod_sav),
        **{f"d_{k}": summarize(v) for k, v in dlt.items()},
        "comp_means": comp_means,
        "intr_means": intr_means,
    }
    smry["n_maternal_tx"] = summarize(n_maternal_tx)
    return df, smry, comp_means, intr_means


# ══════════════════════════════════════════════════════════════════════════════
# § · EVPPI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_evppi(
    df: pd.DataFrame,
    wtp: float,
    perspective: str,
    param_groups: Dict[str, List[str]],

    )-> Tuple[float, Dict[str, float]]:
    """
    Regression-based EVPPI estimator (Strong et al, 2014)
    For each param grp, fir a degree-2 Ridge polynomial to the NMB surface
    as a function of the grp's PSA draws, then compute:
    EVPPI_group = E[max(NMB_hat,0)] - max(e[NMB_hat],0)
    This will yield the expected value of resolving uncertainty in that group alone.
    All values are clipped to [0, EVPI] to enforce theoretical bound.
    05/01/26 - remember to add class notes around NMB surface in the technical material
    for reference

    Parameters
    ----------
    df              : PSA iteration df
    wtp             : WTP ($/DALY)
    perspective     : HS or soc
    param_groups    : Dict mapping group label

    Returns
    -------
    evpi_total      : float - EVPPI upper bound
    evppi_results   : Dict[label: float EVPPI value]
    """
    dal_col = f"dal_{perspective}"
    ic_col  = f"ic_{perspective}"
    
    nmb = (wtp * df[dal_col] - df[ic_col]).values.astype(float)
   
    # EVPI (upper bound)
    evpi_total = float(np.mean(np.maximum(nmb, 0)) - max(float(nmb.mean()), 0))

    evppi_results: Dict[str, float] = {}
    
    for label, cols in param_groups.items():
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            evppi_results[label] = 0.0
            continue
        
        X = df[valid_cols].values.astype(float)
        X = X[:, X.std(axis=0) > 1e-10]
        if X.shape[1] == 0:
            evppi_results[label] = 0.0
            continue
        
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=True)),
            ("ridge", Ridge(alpha=1.0)),])
        pipe.fit(X, nmb)
        nmb_hat = pipe.predict(X)
        # Ensure EVPPI is non-negative
        evppi_val = float(np.mean(np.maximum(nmb_hat,0)) - max(float(nmb_hat.mean()), 0))
        evppi_results[label] = float(np.clip(evppi_val, 0.0, evpi_total))

    return evpi_total, evppi_results
    

# ══════════════════════════════════════════════════════════════════════════════
# §5 · DETERMINISTIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _det_icost(
    p_act, p_sf, p_id, sc_b, sc_e, sens, spec,
    p_adeq, prop_symp, prop_late,
    p_trepo_sf, p_ux_sf,
    r, LE, inc_lbw, inc_mat, cohort,
    mm: MaternalMorbidity = None,
    pl: ProductivityLoss  = None,
    ltc: LongTermCare = None,
    inc_sb_yll: bool = True,
    inc_cs_yll: bool = True,
    inc_misc_yld: bool = True,
    inc_mat_hosp_yld: bool = True,
    inc_preterm_yld: bool = True,
) -> Tuple[float, float, float, float]:
    """Returns (ic_hs, dal_hs, ic_soc, dal_soc) — deterministic means."""
    sc_uc, tx_eff, _ = ges_eff()
    co = Costs(); eps = 1e-9

    def mean_arm(sc):
        p_eff = sc * p_id * sens * p_adeq * tx_eff
        sb_unt = UNT_ABS["stillbirth"]; sb_tx = sb_unt * TX_RR["stillbirth"]["rr"]
        sb_syph = p_eff * sb_tx + (1 - p_eff) * sb_unt
        lb_syph = 1 - sb_syph
        neo_unt_c = UNT_ABS["neonatal_death"] / max(1 - sb_unt, eps)
        neo_tx_c  = UNT_ABS["neonatal_death"] * TX_RR["neonatal_death"]["rr"] / max(1 - sb_tx, eps)
        neo_cond  = min(p_eff * neo_tx_c + (1 - p_eff) * neo_unt_c, 1.0)
        neo_syph  = lb_syph * neo_cond
        surv_syph = lb_syph * (1 - neo_cond)
        cs_cond   = min(p_eff * UNT_ABS["cs_any"] * TX_RR["cs_any"]["rr"]
                        + (1 - p_eff) * UNT_ABS["cs_any"], 1.0)
        cs_r      = p_act * surv_syph * cs_cond
        lbw_unt_c = UNT_ABS["lbw"] / max(1 - sb_unt, eps)
        lbw_tx_c  = UNT_ABS["lbw"] * TX_RR["lbw"]["rr"] / max(1 - sb_tx, eps)
        lbw_cond  = p_eff * lbw_tx_c + (1 - p_eff) * lbw_unt_c
        pt_unt_c  = min(UNT_ABS["preterm"] / max(1 - sb_unt, eps), 1.0)
        pt_tx_c   = min(UNT_ABS["preterm"] * TX_RR["preterm"]["rr"] / max(1 - sb_tx, eps), 1.0)
        pt_cond   = p_eff * pt_tx_c + (1 - p_eff) * pt_unt_c
        sb_bg  = BASE_BETA["stillbirth"]["a"]    / (BASE_BETA["stillbirth"]["a"]    + BASE_BETA["stillbirth"]["b"])
        neo_bg = BASE_BETA["neonatal_death"]["a"] / (BASE_BETA["neonatal_death"]["a"] + BASE_BETA["neonatal_death"]["b"])
        lbw_bg = BASE_BETA["lbw"]["a"]   / (BASE_BETA["lbw"]["a"]   + BASE_BETA["lbw"]["b"])
        pt_bg  = BASE_BETA["preterm"]["a"]/ (BASE_BETA["preterm"]["a"]+ BASE_BETA["preterm"]["b"])
        misc_bg = BASE_BETA["miscarriage"]["a"] / (BASE_BETA["miscarriage"]["a"] + BASE_BETA["miscarriage"]["b"])
        # No treatment RR for miscarriage is parameterised in this model, so
        # the comparator/intervention delta will usually be zero. The DALY term
        # is still included below so the omission is explicit rather than hidden.
        misc_r = p_act * UNT_ABS["miscarriage"] + (1 - p_act) * misc_bg
        sb_r   = p_act * sb_syph + (1 - p_act) * sb_bg
        return {
            "stillbirth":     sb_r * cohort,
            "neonatal_death": (p_act * neo_syph + (1 - p_act) * neo_bg) * cohort,
            "lbw":            (p_act * lb_syph * lbw_cond + (1 - p_act) * lbw_bg) * cohort,
            "preterm":        (p_act * lb_syph * pt_cond + (1 - p_act) * pt_bg) * cohort,
            "miscarriage":    misc_r * cohort,
            "cs_comp":        cs_r * prop_symp * cohort,
            "cs_uncomp":      cs_r * (1 - prop_symp) * cohort,
            "iufd_subset":    sb_r * prop_late * cohort,
        }

    c_arm = mean_arm(sc_uc); i_arm = mean_arm(sc_e)
    dlt   = {k: c_arm[k] - i_arm[k] for k in c_arm}

    af = lambda t: pvf(t, r)
    dal = dlt["neonatal_death"] * af(LE)
    if inc_sb_yll:
        dal += dlt["iufd_subset"] * af(LE)
    if inc_lbw:
        dal += dlt["lbw"] * DW_P["lbw"]["m"] * af(DW_P["lbw"]["dur"])
    if inc_preterm_yld:
        dal += dlt["preterm"] * DW_P["preterm"]["m"] * af(DW_P["preterm"]["dur"])
    if inc_misc_yld:
        dal += dlt.get("miscarriage", 0.0) * DW_P["miscarriage_grief"]["m"] * af(DW_P["miscarriage_grief"]["dur"])
    if inc_mat:
        dal += dlt["stillbirth"]     * DW_P["mat_sb"]["m"]  * af(DW_P["mat_sb"]["dur"])
        dal += dlt["neonatal_death"] * DW_P["mat_nnd"]["m"] * af(DW_P["mat_nnd"]["dur"])

    T_   = max(int(LE), 1)
    q    = INFANT_MK["q_progress"]
    p_sev = INFANT_MK["p_severe_cs_comp"]["m"]; p_mc = INFANT_MK["p_mild_cs_comp"]["m"]
    p_mu  = INFANT_MK["p_mild_cs_uncomp"]["m"]
    dw_m  = INFANT_MK["dw_mild"]["m"];          dw_s = INFANT_MK["dw_severe"]["m"]
    cm    = INFANT_MK["cost_mild_ann"]["mu"];    cs_  = INFANT_MK["cost_sev_ann"]["mu"]

    
    mu_x_m = float(INFANT_MK.get("mu_excess_mild", {}).get("m", 0.0)) if inc_cs_yll else 0.0
    mu_x_s = float(INFANT_MK.get("mu_excess_severe", {}).get("m", 0.0)) if inc_cs_yll else 0.0

    def _det_mk(p_h, p_m, p_s, ltc_obj=None):
        S = np.array([p_h, p_m, p_s, 0.0]); da = cc = 0.0
        for t in range(T_):
            disc = (1 + r) ** (-t); mu_t = lt_qx(t)
            da += (S[1] * dw_m + S[2] * dw_s) * disc

            mx_m = min(mu_x_m, max(1 - mu_t - q, 0.0))
            mx_s = min(mu_x_s, max(1 - mu_t, 0.0))
            excess_deaths = S[1] * mx_m + S[2] * mx_s
            da += excess_deaths * disc * pvf(max(T_ - t, 0), r)

            med = S[1] * cm + S[2] * cs_

            if ltc_obj is not None:
                sped_active = float(ltc_obj.sped_start_age <= t < ltc_obj.sped_end_age)
                sped = sped_active * ltc_obj.cost_sped_ann * (
                    S[1] * ltc_obj.p_sped_mild + S[2] * ltc_obj.p_sped_severe
                )
                cg_active = float(t < ltc_obj.caregiver_end_age)
                cg = cg_active * (S[1] * ltc_obj.cost_cg_mild_ann
                                  + S[2] * ltc_obj.cost_cg_severe_ann)
            else:
                sped = cg = 0.0

            cc += (med + sped + cg) * disc

            S_new = np.zeros(4)
            S_new[0] += S[0] * (1 - mu_t); S_new[3] += S[0] * mu_t
            rm = max(1 - mu_t - q - mx_m, 0.0)
            S_new[1] += S[1] * rm; S_new[2] += S[1] * q; S_new[3] += S[1] * (mu_t + mx_m)
            rs = max(1 - mu_t - mx_s, 0.0)
            S_new[2] += S[2] * rs; S_new[3] += S[2] * (mu_t + mx_s); S_new[3] += S[3]
            S = S_new
        return da, cc

    dpc, cpc = _det_mk(max(1 - p_sev - p_mc, 0.0), p_mc, p_sev, ltc_obj=ltc)
    dpu, cpu = _det_mk(max(1 - p_mu, 0.0), p_mu, 0.0, ltc_obj=ltc)
    
    mk_dal = dlt["cs_comp"] * dpc + dlt["cs_uncomp"] * dpu
    mk_cst = dlt["cs_comp"] * cpc + dlt["cs_uncomp"] * cpu
    dal   += mk_dal

    extra  = max(sc_e * p_id - sc_uc * p_id, 0.0) * cohort
    p_sn   = max(1 - p_act - p_sf, 0.0); p_fp = p_sn * (1 - spec)
    n_tx   = extra * p_act * sens * p_adeq * tx_eff
    test   = extra * co.poc + extra * co.rpr + extra * (p_act * sens + p_fp) * co.fta
    sf_cost = _serofast_cost(extra, p_sf, p_trepo_sf, p_ux_sf, co.rpr, co.sf_wu, co.pen, co.soc_work)
    prog   = test + n_tx * (co.pen + co.soc_work) + extra * co.staff + sf_cost
    
    n_iufd = dlt["iufd_subset"]; n_sb_t = max(dlt["stillbirth"] - n_iufd, 0.0)
    sav    = (n_iufd * co.iufd + n_sb_t * co.sb
              + dlt["neonatal_death"] * co.nnd + dlt["lbw"] * co.lbw_hs
              + dlt["preterm"] * co.preterm
              + dlt["cs_comp"] * (co.cs_wu + co.nicu) + dlt["cs_uncomp"] * co.cs_wu
              + mk_cst)
    ic_hs = float(prog - sav)
    dal_hs = float(dal)

    # Societal additions
    mm_dal_v = mm_cost_v = prod_sav_v = 0.0
    if mm is not None:
        n_maternal_tx = cohort * p_act * max(sc_e - sc_uc, 0.0) * p_id * sens * p_adeq * tx_eff
        n_cases = max(dlt["cs_comp"] + dlt["cs_uncomp"]
                      + dlt["stillbirth"] + dlt["neonatal_death"], 0.0)
        mm_dal_v, mm_cost_v = _mat_morb_det(mm, n_maternal_tx, r, include_hosp_yld=inc_mat_hosp_yld)
    if pl is not None:
        prod_sav_v = _prod_loss_det(
            pl, dlt["stillbirth"], dlt["neonatal_death"],
            dlt["cs_comp"], dlt["cs_uncomp"], r, LE)
    ic_soc  = ic_hs  - mm_cost_v - prod_sav_v
    dal_soc = dal_hs + mm_dal_v
    return ic_hs, dal_hs, ic_soc, dal_soc


def budget_impact_table(
    annual_vol, p_act, p_sf, p_id, sc_b, sc_e,
    sens, spec, p_adeq, prop_symp, prop_late,
    p_trepo_sf, p_ux_sf, r, LE, n_years=5,
) -> pd.DataFrame:
    co = Costs(); sc_uc, tx_eff, _ = ges_eff()
    extra   = max(sc_e * p_id - sc_uc * p_id, 0.0) * annual_vol
    p_sn    = max(1 - p_act - p_sf, 0.0); p_fp = p_sn * (1 - spec)
    n_tx_yr = extra * p_act * sens * p_adeq
    sf_yr = _serofast_cost(
        extra, p_sf, p_trepo_sf, p_ux_sf, co.rpr, co.sf_wu, co.pen, co.soc_work
    )
    prog_yr = (extra * co.poc + extra * co.rpr
               + extra * (p_act * sens + p_fp) * co.fta
               + n_tx_yr * (co.pen + co.soc_work)
               + extra * co.staff
               + sf_yr)
    p_eff_d = (sc_e - sc_uc) * p_id * sens * p_adeq * tx_eff
    cs_comp_yr   = annual_vol * p_act * p_eff_d * UNT_ABS["cs_any"] * prop_symp
    cs_uncomp_yr = annual_vol * p_act * p_eff_d * UNT_ABS["cs_any"] * (1 - prop_symp)
    sb_yr   = annual_vol * p_act * p_eff_d * UNT_ABS["stillbirth"]
    nnd_yr  = annual_vol * p_act * p_eff_d * UNT_ABS["neonatal_death"]
    lbw_yr  = annual_vol * p_act * p_eff_d * UNT_ABS["lbw"]
    preterm_yr = annual_vol * p_act * p_eff_d * UNT_ABS["preterm"]
    iufd_yr    = sb_yr * prop_late; sb_t_yr = max(sb_yr - iufd_yr, 0.0)
    sav_imm = (iufd_yr * co.iufd + sb_t_yr * co.sb + nnd_yr * co.nnd
               + lbw_yr * co.lbw_hs + preterm_yr * co.preterm
               + cs_comp_yr * (co.cs_wu + co.nicu) + cs_uncomp_yr * co.cs_wu)
    T_ = max(int(LE), 1); q = INFANT_MK["q_progress"]
    p_sev = INFANT_MK["p_severe_cs_comp"]["m"]; p_mc = INFANT_MK["p_mild_cs_comp"]["m"]
    p_mu  = INFANT_MK["p_mild_cs_uncomp"]["m"]
    cm    = INFANT_MK["cost_mild_ann"]["mu"];    cs_v = INFANT_MK["cost_sev_ann"]["mu"]
    def _det_mk_cost(p_h, p_m, p_s):
        S = np.array([p_h, p_m, p_s, 0.0]); acc = 0.0
        for t in range(T_):
            mu_t = lt_qx(t); acc += (S[1]*cm + S[2]*cs_v) * (1+r)**(-t)
            S_new = np.zeros(4)
            S_new[0] += S[0]*(1-mu_t); S_new[3] += S[0]*mu_t
            rm = max(1-mu_t-q, 0.0); S_new[1] += S[1]*rm; S_new[2] += S[1]*q
            S_new[3] += S[1]*mu_t; S_new[2] += S[2]*(1-mu_t); S_new[3] += S[2]*mu_t
            S_new[3] += S[3]; S = S_new
        return acc
    cpc = _det_mk_cost(max(1-p_sev-p_mc, 0.0), p_mc, p_sev)
    cpu = _det_mk_cost(max(1-p_mu, 0.0), p_mu, 0.0)
    sav_yr = sav_imm + cs_comp_yr * cpc + cs_uncomp_yr * cpu
    net_yr = prog_yr - sav_yr
    rows, cum = [], 0.0
    for yr in range(1, n_years + 1):
        cum += net_yr
        rows.append({"Year": yr, "Program cost ($)": prog_yr,
                     "Outcome savings ($)": sav_yr, "Net impact ($)": net_yr,
                     "Cumulative net ($)": cum,
                     "CS cases prevented": cs_comp_yr + cs_uncomp_yr,
                     "Stillbirths prevented": sb_yr})
    return pd.DataFrame(rows), prog_yr, sav_yr, net_yr, cs_comp_yr + cs_uncomp_yr


def nmb_surface(prev_grid, tx_grid, p_sf, p_id, sc_b, sc_e, sens, spec,
                prop_symp, prop_late, p_trepo_sf, p_ux_sf,
                r, LE, inc_lbw, inc_mat, cohort, wtp,
                mm=None, pl=None, societal=False,
                inc_sb_yll=True, inc_cs_yll=True, inc_misc_yld=True,
                inc_mat_hosp_yld=True, inc_preterm_yld=True) -> np.ndarray:
    G = np.zeros((len(tx_grid), len(prev_grid)))
    for i, p_adeq in enumerate(tx_grid):
        for j, p_act in enumerate(prev_grid):
            ic_hs, dal_hs, ic_soc, dal_soc = _det_icost(
               p_act, p_sf, p_id, sc_b, sc_e, sens, spec,
                                    p_adeq, prop_symp, prop_late,
                                    p_trepo_sf, p_ux_sf, r, LE, inc_lbw, inc_mat,
                                    cohort, mm, pl, None,
                                    inc_sb_yll=inc_sb_yll, inc_cs_yll=inc_cs_yll,
                                    inc_misc_yld=inc_misc_yld,
                                    inc_mat_hosp_yld=inc_mat_hosp_yld,
                                    inc_preterm_yld=inc_preterm_yld)
            ic  = ic_soc  if societal else ic_hs
            dal = dal_soc if societal else dal_hs
            G[i, j] = wtp * dal - ic
    return G


# ══════════════════════════════════════════════════════════════════════════════
# §6 · REPORTING & EVPI
# ══════════════════════════════════════════════════════════════════════════════

def ce_quadrant_table(dal: np.ndarray, ic: np.ndarray) -> pd.DataFrame:
    """
    Partition PSA iterations into the four CE-plane quadrants.
    Returns a DataFrame with quadrant label, %, and interpretation.
    """
    n = len(dal)
    q1 = ((dal > 0) & (ic < 0)).sum()   # Dominant (more DALY, less cost)
    q2 = ((dal > 0) & (ic > 0)).sum()   # Cost-effective (more DALY, more cost)
    q3 = ((dal < 0) & (ic > 0)).sum()   # Dominated (fewer DALY, more cost)
    q4 = ((dal < 0) & (ic < 0)).sum()   # Potentially cost-saving but fewer DALY
    return pd.DataFrame([
        {"Quadrant": "↗ Dominant",            "N": q1, "%": f"{100*q1/n:.1f}%",
         "Interpretation": "Saves money AND prevents DALYs"},
        {"Quadrant": "↗ Cost-effective",       "N": q2, "%": f"{100*q2/n:.1f}%",
         "Interpretation": "Prevents DALYs at additional cost"},
        {"Quadrant": "↙ Dominated",            "N": q3, "%": f"{100*q3/n:.1f}%",
         "Interpretation": "Costs more and prevents fewer DALYs"},
        {"Quadrant": "↙ Fewer DALYs, cheaper", "N": q4, "%": f"{100*q4/n:.1f}%",
         "Interpretation": "Cheaper but associated with fewer DALYs prevented"},
    ])


def evpi_curve(dal: np.ndarray, ic: np.ndarray, wtp_max: int = 200_000,
               step: int = 2_000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expected Value of Perfect Information curve.
    EVPI(λ) = E[max(λ·ΔDALY − ΔCost, 0)] − max(E[λ·ΔDALY − ΔCost], 0)
    Returns (lambda_grid, evpi_values).
    """
    lam  = np.arange(0, wtp_max + step, step)
    evpi = np.zeros(len(lam))
    for k, lam_k in enumerate(lam):
        nmb = lam_k * dal - ic
        evpi[k] = np.mean(np.maximum(nmb, 0)) - max(float(np.mean(nmb)), 0)
    return lam, evpi


def psa_convergence(dal: np.ndarray, ic: np.ndarray,
                    step: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling mean ICER as a function of PSA iteration count.
    Returns (n_iters, rolling_icer).
    """
    ns  = np.arange(step, len(dal) + 1, step)
    eps = 1e-12
    rolling = np.array([
        float(ic[:n].mean()) / np.maximum(float(dal[:n].mean()), eps)
        for n in ns
    ])
    return ns, rolling


def owsa_table(
    base_kw: dict,
    param_ranges: Dict[str, Tuple],   # {label: (lo_kw, hi_kw)}
) -> pd.DataFrame:
    """
    One-way sensitivity analysis table.
    param_ranges values are dicts of kwargs to pass to _det_icost for low/high runs.
    Returns a DataFrame with ICER at base, low, and high for each parameter.
    """
    ic_b, dal_b, _, _ = _det_icost(**base_kw)
    icer_b = ic_b / max(dal_b, 1e-9)
    rows = []
    for label, (lo_kw, hi_kw) in param_ranges.items():
        ic_lo, dal_lo, _, _ = _det_icost(**{**base_kw, **lo_kw})
        ic_hi, dal_hi, _, _ = _det_icost(**{**base_kw, **hi_kw})
        icer_lo = ic_lo / max(dal_lo, 1e-9)
        icer_hi = ic_hi / max(dal_hi, 1e-9)
        # Determine which direction is "low ICER" for tornado ordering
        low_icer  = min(icer_lo, icer_hi)
        high_icer = max(icer_lo, icer_hi)
        rows.append({
            "Parameter":      label,
            "Base ICER":      icer_b,
            "ICER (low)":     icer_lo,
            "ICER (high)":    icer_hi,
            "ICER min":       low_icer,
            "ICER max":       high_icer,
            "Range":          high_icer - low_icer,
            "Low param value": list(lo_kw.values())[0],
            "High param value": list(hi_kw.values())[0],
        })
    df = pd.DataFrame(rows).sort_values("Range", ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# §7 · FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def fig_ce_plane(dal, inc, title, wtp_lines=(50_000, 100_000, 150_000)):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(dal, inc / 1e6, s=.5, alpha=0.25, color="steelblue", rasterized=True)
    x_lim = np.array([min(dal.min() * 1.1, -50), dal.max() * 1.1])
    for wtp, col in zip(wtp_lines, ["#2a9d8f", "#e9c46a", "#e76f51"]):
        ax.plot(x_lim, wtp * x_lim / 1e6, ls="--", lw=1.2, color=col,
                label=f"${wtp/1000:.0f}K/DALY")
    ci_ellipse(ax, dal, inc / 1e6)
    ax.axhline(0, color="k", lw=0.6, zorder=3)
    ax.axvline(0, color="k", lw=0.6, zorder=3)
    # Shade quadrant labels
    xl, xr = ax.get_xlim() if ax.get_xlim()[0] != 0 else (x_lim[0], x_lim[1])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.1f}M"))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("ΔDALYs prevented (positive = intervention better)", fontsize=10)
    ax.set_ylabel("ΔCost  [positive = intervention costs more]", fontsize=9)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    ax.set_title(preset, fontsize=11, fontweight="bold")
    ax.legend(title="WTP threshold", fontsize=8, framealpha=0.7)
    ax.grid(alpha=0.15); ax.spines[["top","right"]].set_visible(False)
    return fig


def fig_ceac(dal_hs, ic_hs, dal_soc, ic_soc, wtp_max=200_000):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    lam = np.arange(0, wtp_max + 1_000, 1_000)
    for dal_, ic_, label, col in [
        (dal_hs,  ic_hs,  "Health sector",  "steelblue"),
        (dal_soc, ic_soc, "Societal (productivity + mat. morbidity)", "darkorange"),
    ]:
        probs = (lam[None, :] * dal_[:, None] - ic_[:, None] > 0).mean(axis=0)
        ax.plot(lam, probs, lw=2, label=label, color=col)
    for vline, col in [(50_000,"#2a9d8f"),(100_000,"#e9c46a"),(150_000,"#e76f51")]:
        ax.axvline(vline, ls=":", lw=1, color=col, alpha=0.8, label=f"${vline//1000}K")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.15)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(dollar_fmt))
    ax.set_xlabel("WTP threshold ($/DALY)"); ax.set_ylabel("P(cost-effective)")
    ax.set_title(preset, fontweight="bold")
    fig.suptitle("Cost-Effectiveness Acceptability Curve", fontweight="bold")
    ax.tick_params(axis = "x", which="major", labelsize=8)
    ax.tick_params(axis = "x", which="major", labelsize=8)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.spines[["top","right"]].set_visible(False)
    return fig


def fig_evpi(dal, ic, wtp_max=200_000):
    lam, evpi = evpi_curve(dal, ic, wtp_max)
    fig, ax   = plt.subplots(figsize=(7.5, 4))
    ax.plot(lam, evpi / 1e6, color="purple", lw=2)
    ax.fill_between(lam, evpi / 1e6, alpha=0.15, color="purple")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(dollar_fmt))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.2f}M"))
    ax.set_xlabel("WTP threshold ($/DALY)")
    ax.set_ylabel("EVPI per cohort ($M)")
    ax.set_title("Expected Value of Perfect Information", fontweight="bold")
    ax.grid(alpha=0.15); ax.spines[["top","right"]].set_visible(False)
    return fig


def fig_convergence(dal, ic, step=500):
    ns, rolling = psa_convergence(dal, ic, step)
    base = rolling[-1]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(ns, rolling, color="steelblue", lw=2, label="Rolling mean ICER")
    ax.axhline(base, color="k", ls="--", lw=1, label=f"Final mean: ${base:,.0f}")
    ax.fill_between(ns, base * 0.95, base * 1.05, alpha=0.12,
                    color="steelblue", label="±5% tolerance band")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(dollar_fmt))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("PSA iterations"); ax.set_ylabel("ICER ($/DALY) — health sector")
    ax.set_title("PSA Convergence Diagnostic", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.15)
    ax.spines[["top","right"]].set_visible(False)
    return fig


def fig_tornado(owsa_df: pd.DataFrame, base_icer: float, top_n: int = 12):
    df = owsa_df.head(top_n)
    n  = len(df)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * n + 1)))
    for i, row in df.iterrows():
        ax.barh(i, row["ICER (high)"] - base_icer, left=base_icer,
                height=0.55, color="#4a90d9", edgecolor="k", alpha=0.85)
        ax.barh(i, row["ICER (low)"]  - base_icer, left=base_icer,
                height=0.55, color="#e08050", edgecolor="k", alpha=0.85)
        # Annotate parameter values
        ax.text(row["ICER (high)"] + abs(base_icer)*0.005, i,
                f"{row['High param value']:.3g}", va="center", fontsize=7, color="#4a90d9")
        ax.text(row["ICER (low)"]  - abs(base_icer)*0.005, i,
                f"{row['Low param value']:.3g}", va="center", fontsize=7,
                ha="right", color="#e08050")
    ax.axvline(base_icer, color="k", lw=1.5, ls="--", zorder=4, label=f"Base: ${base_icer:,.0f}")
    ax.set_yticks(range(n)); ax.set_yticklabels(df["Parameter"].tolist(), fontsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(dollar_fmt))
    ax.set_xlabel("ICER ($/DALY) — health-sector perspective")
    ax.set_title("One-Way Sensitivity Analysis (Tornado)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.15, axis="x")
    ax.spines[["top","right"]].set_visible(False)
    return fig


def owsa_nmb_table(
    base_kw:      dict,
    param_ranges: Dict[str, Tuple],   # same structure as owsa_table
    wtp:          float,
    perspective:  str = "hs",         # "hs" | "soc"
) -> pd.DataFrame:
    """
    One-way sensitivity analysis on Net Monetary Benefit.

    NMB(λ) = λ · ΔDALYs − ΔCost

    Preferred over ICER-based OWSA because NMB is:
      - Linear in costs and effects → ranges are directly comparable across params
      - Well-behaved when the program is cost-saving (no ratio sign-flip)
      - Undefined/explosive ICER behaviour near ΔDALYs ≈ 0 is avoided

    Parameters
    ----------
    base_kw      : kwargs dict passed to _det_icost for the base-case run
    param_ranges : {label: (lo_kw, hi_kw)} — same format as owsa_table
    wtp          : willingness-to-pay threshold ($/DALY)
    perspective  : "hs" for health sector, "soc" for societal

    Returns
    -------
    DataFrame sorted descending by NMB range, columns:
        Parameter, Base NMB, NMB (low param), NMB (high param),
        NMB min, NMB max, Range, Low param value, High param value
    """
    def _nmb(kw_override):
        ic_hs, dal_hs, ic_soc, dal_soc = _det_icost(**{**base_kw, **kw_override})
        dal = dal_soc if perspective == "soc" else dal_hs
        ic  = ic_soc  if perspective == "soc" else ic_hs
        return wtp * dal - ic

    nmb_base = _nmb({})
    rows = []
    for label, (lo_kw, hi_kw) in param_ranges.items():
        nmb_lo = _nmb(lo_kw)
        nmb_hi = _nmb(hi_kw)
        rows.append({
            "Parameter":        label,
            "Base NMB":         nmb_base,
            "NMB (low param)":  nmb_lo,
            "NMB (high param)": nmb_hi,
            "NMB min":          min(nmb_lo, nmb_hi),
            "NMB max":          max(nmb_lo, nmb_hi),
            "Range":            abs(nmb_hi - nmb_lo),
            "Low param value":  list(lo_kw.values())[0],
            "High param value": list(hi_kw.values())[0],
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Range", ascending=False)
        .reset_index(drop=True)
    )


def fig_tornado_nmb(
    owsa_nmb_df: pd.DataFrame,
    base_nmb:    float,
    wtp:         float,
    perspective: str,
    top_n:       int = 12,
) -> plt.Figure:
    """
    Tornado chart for NMB-based OWSA.

    Bars extend from the base-case NMB toward the low/high parameter values.
    A vertical dashed line marks base-case NMB; a dotted line marks NMB = 0
    (break-even), making cost-effectiveness threshold visually explicit.

    Colour convention (consistent with fig_tornado):
        Blue  (#4a90d9) — direction that increases NMB
        Orange (#e08050) — direction that decreases NMB
    """
    df = owsa_nmb_df.head(top_n)
    n  = len(df)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * n + 1.2)))

    for idx, row in enumerate(df.itertuples()):
        nmb_lo = row._3   # "NMB (low param)"  — positional after index resets
        nmb_hi = row._4   # "NMB (high param)"

        # Determine which direction increases NMB for consistent colouring
        lo_delta = nmb_lo - base_nmb
        hi_delta = nmb_hi - base_nmb

        ax.barh(idx, lo_delta, left=base_nmb, height=0.55,
                color="#e08050" if lo_delta < hi_delta else "#4a90d9",
                edgecolor="k", linewidth=0.5, alpha=0.85)
        ax.barh(idx, hi_delta, left=base_nmb, height=0.55,
                color="#4a90d9" if hi_delta > lo_delta else "#e08050",
                edgecolor="k", linewidth=0.5, alpha=0.85)

        # Parameter-value annotations at bar tips
        x_right = max(nmb_lo, nmb_hi)
        x_left  = min(nmb_lo, nmb_hi)
        pad     = (owsa_nmb_df["NMB max"].max() - owsa_nmb_df["NMB min"].min()) * 0.005

        ax.text(x_right + pad, idx,
                f"{row._9:.3g}",          # "High param value"
                va="center", fontsize=7, color="#4a90d9")
        ax.text(x_left - pad, idx,
                f"{row._8:.3g}",          # "Low param value"
                va="center", ha="right", fontsize=7, color="#e08050")

    # Reference lines
    ax.axvline(base_nmb, color="k", lw=1.5, ls="--", zorder=4,
               label=f"Base NMB: ${base_nmb/1e6:,.2f}M")
    ax.axvline(0, color="dimgray", lw=0.9, ls=":", zorder=3,
               label="NMB = 0  (break-even)")

    # Shade negative NMB region as a visual cue
    x_min = ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else (
        owsa_nmb_df["NMB min"].min() * 1.15)
    if x_min < 0:
        ax.axvspan(x_min, 0, color="salmon", alpha=0.06, zorder=0,
                   label="NMB < 0 (not cost-effective at this WTP)")

    ax.set_yticks(range(n))
    ax.set_yticklabels(df["Parameter"].tolist(), fontsize=8)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"${x/1e6:,.1f}M"))
    ax.set_xlabel(
        f"Net Monetary Benefit ($M)  |  WTP = ${wtp/1000:.0f}K/DALY  |  "
        f"{'Health sector' if perspective == 'hs' else 'Societal'} perspective",
        fontsize=9)
    ax.set_title("One-Way Sensitivity Analysis — NMB Tornado", fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(alpha=0.15, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig

def fig_nmb_surface(prev_grid, tx_grid, nmb_G, wtp):
    fig, ax = plt.subplots(figsize=(8, 5))
    PP, TT  = np.meshgrid(prev_grid * 100, tx_grid * 100)
    vmax    = np.percentile(np.abs(nmb_G), 97); vmin = -vmax
    cf = ax.contourf(PP, TT, nmb_G / 1e6, levels=30,
                     cmap="RdYlGn", vmin=vmin/1e6, vmax=vmax/1e6)
    ax.contour(PP, TT, nmb_G, levels=[0], colors="k", linewidths=2)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("NMB ($M)", fontsize=9)
    ax.set_xlabel("Active syphilis prevalence (%)", fontsize=10)
    ax.set_ylabel("Same-day treatment rate (%)",   fontsize=10)
    ax.set_title(f"Net Monetary Benefit  |  WTP = ${wtp/1000:.0f}K/DALY\n"
                 "Black contour = break-even (NMB = 0)", fontsize=10, fontweight="bold")
    return fig


def fig_budget_bars(df_bi: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    yrs = df_bi["Year"].values
    ax.bar(yrs - 0.2, df_bi["Program cost ($)"] / 1e6, 0.4,
           label="Program cost",    color="#4a90d9", alpha=0.85)
    ax.bar(yrs + 0.2, df_bi["Outcome savings ($)"] / 1e6, 0.4,
           label="Outcome savings", color="#2a9d8f", alpha=0.85)
    ax.plot(yrs, df_bi["Cumulative net ($)"] / 1e6, "k--o",
            ms=5, lw=1.5, label="Cumulative net impact")
    ax.axhline(0, color="k", lw=0.6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.1f}M"))
    ax.set_xlabel("Year"); ax.set_title("Annual Budget Impact", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.15)
    ax.spines[["top","right"]].set_visible(False)
    return fig


def fig_markov_states(r_disc, LE, mk_means):
    T = max(int(LE), 1); q = INFANT_MK["q_progress"]
    p_sev = mk_means["p_severe_cs_comp"]; p_mc = mk_means["p_mild_cs_comp"]
    p_mu  = mk_means["p_mild_cs_uncomp"]
    mu_x_m = float(mk_means.get("mu_excess_mild", 0.0))
    mu_x_s = float(mk_means.get("mu_excess_severe", 0.0))
    STATES = ["Healthy", "Mild sequelae", "Severe sequelae", "Dead"]
    COLORS = ["#2a9d8f", "#e9c46a", "#e76f51", "#6c757d"]

    def _occ(S0):
        hist = [S0.copy()]; S = S0.copy()
        for t in range(T - 1):
            mu_t = lt_qx(t); S_new = np.zeros(4)
            mx_m = min(mu_x_m, max(1-mu_t-q, 0.0))
            mx_s = min(mu_x_s, max(1-mu_t, 0.0))
            S_new[0] += S[0]*(1-mu_t); S_new[3] += S[0]*mu_t
            rm = max(1-mu_t-q-mx_m, 0.0); S_new[1] += S[1]*rm
            S_new[2] += S[1]*q; S_new[3] += S[1]*(mu_t + mx_m)
            rs = max(1-mu_t-mx_s, 0.0)
            S_new[2] += S[2]*rs; S_new[3] += S[2]*(mu_t + mx_s); S_new[3] += S[3]
            S = S_new; hist.append(S.copy())
        return np.array(hist)

    S0_c = np.array([max(1-p_sev-p_mc, 0.0), p_mc, p_sev, 0.0])
    S0_u = np.array([max(1-p_mu, 0.0), p_mu, 0.0, 0.0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    years = np.arange(T)
    for ax, S0, title in [(axes[0], S0_c, "CS Complicated"),
                           (axes[1], S0_u, "CS Uncomplicated")]:
        occ = _occ(S0)
        ax.stackplot(years, occ[:,0], occ[:,1], occ[:,2], occ[:,3],
                     labels=STATES, colors=COLORS, alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Years after birth"); ax.grid(alpha=0.12)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("State occupancy probability")
    axes[1].legend(loc="center right", fontsize=8, framealpha=0.7)
    fig.suptitle("Infant Markov State Occupancy (mean parameters)", fontweight="bold")
    plt.tight_layout()
    return fig


def fig_markov_daly_dist(df_psa):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    total_cs  = (df_psa["d_cs_comp"] + df_psa["d_cs_uncomp"]).replace(0, np.nan)
    per_case  = (df_psa["mk_dal"] / total_cs).dropna()
    per_case  = per_case[np.isfinite(per_case) & (per_case > 0)]
    axes[0].hist(per_case, bins=60, color="#4a90d9", alpha=0.85, edgecolor="white")
    axes[0].axvline(per_case.mean(), color="k", lw=1.5, ls="--",
                    label=f"Mean = {per_case.mean():.2f} DALYs/case")
    axes[0].set_xlabel("Lifetime DALYs averted per CS case")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Per-Case Lifetime DALY (Markov: YLD + excess YLL)", fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.15)
    axes[0].spines[["top","right"]].set_visible(False)
    axes[1].hist(df_psa["mk_cst"] / 1e3, bins=60, color="#2a9d8f", alpha=0.85, edgecolor="white")
    axes[1].axvline(df_psa["mk_cst"].mean()/1e3, color="k", lw=1.5, ls="--",
                    label=f"Mean = ${df_psa['mk_cst'].mean()/1e3:,.0f}K")
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    axes[1].set_xlabel("Markov lifetime cost saving per cohort")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Lifetime Cost Saving (Markov)", fontweight="bold")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.15)
    axes[1].spines[["top","right"]].set_visible(False)
    plt.tight_layout(); return fig


def fig_prod_loss_breakdown(df_psa):
    """Stacked histogram of productivity-loss saving components — societal tab."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_psa["prod_sav"] / 1e6, bins=60, color="#8e44ad",
            alpha=0.80, edgecolor="white")
    ax.axvline(df_psa["prod_sav"].mean()/1e6, color="k", lw=1.5, ls="--",
               label=f"Mean = ${df_psa['prod_sav'].mean()/1e6:,.0f}M")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
    ax.set_xlabel("Productivity-loss savings averted per cohort ($M)")
    ax.set_ylabel("Frequency (PSA iterations)")
    ax.set_title("Productivity Loss Savings — Human Capital Method (PSA Distribution)",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.15)
    ax.tick_params(axis = "x", which="major", labelsize=7)
    ax.tick_params(axis = "y", which="major", labelsize=7)
    ax.spines[["top","right"]].set_visible(False)
    return fig

#   Break 03/09/2026
def fig_evppi_bar(
    evppi_results: Dict[str, float],
    evpi_total: float,
    wtp: float,
    perspective: str,
) -> plt.Figure:
    """
    Horizontal bar chart of EVPPI by parameter group.
    Bars are sorted descending. EVPI (upper bound) shown as a dashed line.
    Values shown in $K per cohort.
    """
    labels = list(evppi_results.keys())
    values = [evppi_results[k] / 1e3 for k in labels]   # → $K
    evpi_k = evpi_total / 1e3

    # Sort descending
    order  = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    n   = len(labels)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * n + 1.2)))

    colors = plt.cm.Blues(np.linspace(0.40, 0.80, n))[::-1]
    bars = ax.barh(range(n), values, color=colors, edgecolor="k",
                   alpha=0.88, height=0.6)

    # Value labels
    for bar_, val in zip(bars, values):
        ax.text(bar_.get_width() + evpi_k * 0.01, bar_.get_y() + bar_.get_height() / 2,
                f"${val:,.1f}K", va="center", fontsize=8)

    ax.axvline(evpi_k, color="crimson", lw=1.8, ls="--",
               label=f"EVPI (upper bound): ${evpi_k:,.1f}K")

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("EVPPI per cohort ($K)", fontsize=10)
    ax.set_title(
        f"Expected Value of Partially Perfect Information\n"
        f"Perspective: {'Health sector' if perspective == 'hs' else 'Societal'}  |  "
        f"WTP = ${wtp/1000:.0f}K/DALY",
        fontweight="bold", fontsize=11,
    )
    ax.legend(fontsize=9, framealpha=0.7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    ax.grid(alpha=0.15, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig



# ══════════════════════════════════════════════════════════════════════════════
# §8 · STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Syphilis ED Screening CEA v4",
    page_icon="🏥", layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Model Settings")
    preset = st.selectbox("Scenario preset", list(PRESETS.keys()), index=2)
    P = PRESETS[preset]
    def pv(key, default): return P.get(key, default)

    st.subheader("Population & Epidemiology")
    cohort = st.number_input("Cohort size", 1_000, 500_000, 100_000, 1_000)
    p_act = st.slider(
        "Active syphilis prevalence",
        min_value=0.001,
        max_value=0.030,          
        value=pv("p_act", 0.0075),
        step=0.0005,
        format="%.4f",
        help=(
            "Published ED screening studies:\n"
            "• High-burden urban: 1.0–2.0%\n"
            "• Moderate burden:   0.5–1.0%\n"
            "• Low-prevalence rural: ~0.1%\n"
            "Sources: Tao et al. 2023; Bristow et al. 2023; Tuite et al. 2023."
        ),
      )

    st.subheader("Serofast Module")

    # Derived default: serofast ≈ 20% of active syphilis cases
    # (Romanowski et al. 1991; Seña et al. 2011 — 15–25% of treated cases, Kiolbasa et. al 2025 15-20%
    #  remain serofast; 20% is the literature midpoint.)
    p_sf_default = round(0.20 * p_act, 4)

    p_sf_override = st.checkbox(
        f"Override serofast prevalence (default: 20% of p_act = {p_sf_default:.4f})",
        value=False,
        help=(
            "By default, serofast prevalence is set to 20% of active syphilis "
            "prevalence, consistent with the finding that ~15–25% of adequately "
            "treated syphilis cases remain serofast (Romanowski et al. 1991; "
            "Seña et al. 2011, Kiolbasa et al 2025). Uncheck to use this literature-derived default."
        ),
    )

    if p_sf_override:
        p_sf = st.slider(
            "Serofast / prior-treated prevalence (manual)",
            min_value=0.001,
            max_value=0.010,      
            value=p_sf_default,
            step=0.0001,
            format="%.4f",
        )
    else:
        p_sf = p_sf_default
        st.caption(
            f"Serofast prevalence locked to 20% × p_act = **{p_sf:.4f}** "
            f"({p_sf * 100:.2f}%). Adjust via override checkbox above."
        )

    st.subheader("ED Operational Parameters")
    p_id  = st.slider("P(pregnancy identified in ED workflow)", 0.50, 1.00, 0.85, 0.01)
    sc_e  = st.slider("Enhanced (ED) screening coverage", 0.50, 1.00, pv("sc_e", 0.90), 0.01)
    sens  = st.slider("Test sensitivity (treponemal screen)", 0.85, 1.00, 0.98, 0.01)
    spec  = st.slider("Test specificity", 0.85, 1.00, 0.98, 0.01)

    st.subheader("Treatment Cascade")
    p_adeq  = st.slider("P(adequate treatment | true positive detected)", 0.30, 1.00, pv("p_adeq", 0.85), 0.01)
    p_tx_ov = st.checkbox("Override strata-weighted tx completion", value=False)
    p_tx    = st.slider("Tx completion (override)", 0.30, 1.00, 0.77, 0.01) if p_tx_ov else None

    st.subheader("Serofast Module")
    p_trepo_sf = st.slider("P(treponemal+ | serofast)", 0.70, 1.00, 0.95, 0.01)
    p_ux_sf    = st.slider("P(unnecessarily treated | serofast detected)", 0.00, 0.60, 0.20, 0.01)
    treat_fp   = st.checkbox("Treat seronegative false-positives", value=False)

    st.subheader("Outcome Structure")
    prop_symp = st.slider("Proportion CS complicated (symptomatic)", 0.10, 0.70, 0.38, 0.01)
    _sc_uc_eff, _tx_eff, _prop_late_eff = ges_eff()
    prop_late_override = st.checkbox(
        f"Override stratum-weighted prop_late (default: {_prop_late_eff:.2f})", value=False)
    prop_late = st.slider("Proportion stillbirths that are IUFD ≥28w",
                          0.05, 0.90, _prop_late_eff, 0.01) if prop_late_override else _prop_late_eff

    st.subheader("Infant Markov Parameters")
    with st.expander("Sequelae & calibration", expanded=False):
        p_sev_ui = st.slider("P(severe seq | CS complicated)",  0.10, 0.60, INFANT_MK["p_severe_cs_comp"]["m"], 0.01)
        p_mc_ui  = st.slider("P(mild seq | CS complicated)",    0.10, 0.65, INFANT_MK["p_mild_cs_comp"]["m"],   0.01)
        p_mu_ui  = st.slider("P(mild seq | CS uncomplicated)",  0.01, 0.25, INFANT_MK["p_mild_cs_uncomp"]["m"], 0.01)
        c_mild_ui = st.number_input("Annual cost – mild sequelae ($)",   1_000,  50_000, int(INFANT_MK["cost_mild_ann"]["mu"]),  500)
        c_sev_ui  = st.number_input("Annual cost – severe sequelae ($)", 5_000, 100_000, int(INFANT_MK["cost_sev_ann"]["mu"]), 1_000)
        q_target  = st.slider("Target lifetime mild→severe progression (%)",
                              5, 40, int(INFANT_MK["q_progress_target"] * 100), 1) / 100.0
        mu_x_mild_ui = st.slider(
            "Annual excess mortality – mild CS sequelae",
            0.000, 0.020, float(INFANT_MK["mu_excess_mild"]["m"]), 0.001,
            format="%.3f",
            help="CS-attributable excess mortality above life-table mortality. Default 0 avoids charging background deaths as CS YLL."
        )
        mu_x_sev_ui = st.slider(
            "Annual excess mortality – severe CS sequelae",
            0.000, 0.050, float(INFANT_MK["mu_excess_severe"]["m"]), 0.001,
            format="%.3f",
            help="Non-zero values generate post-neonatal CS YLL in the Markov loop."
        )
    with st.expander("CS Natural History (calibration)", expanded=False):
        cs_cure_ui  = st.slider("P(cure | CS + early treatment)",     0.80, 1.00, INFANT_MK["cs_early_cure_rate"],     0.01)
        cs_late_ui  = st.slider("P(late complications | untreated CS)",0.05, 0.50, INFANT_MK["cs_late_manifest_rate"],  0.01)
        cs_neuro_ui = st.slider("P(neuro disorder | CS w/ complications)",0.05, 0.50, INFANT_MK["cs_neuro_disorder_rate"], 0.01)

    # Push overrides back
    INFANT_MK["p_severe_cs_comp"]["m"] = p_sev_ui
    INFANT_MK["p_mild_cs_comp"]["m"]   = p_mc_ui
    INFANT_MK["p_mild_cs_uncomp"]["m"] = p_mu_ui
    INFANT_MK["cost_mild_ann"]["mu"]   = float(c_mild_ui)
    INFANT_MK["cost_sev_ann"]["mu"]    = float(c_sev_ui)
    INFANT_MK["q_progress_target"]     = q_target
    INFANT_MK["mu_excess_mild"]["m"]   = float(mu_x_mild_ui)
    INFANT_MK["mu_excess_severe"]["m"] = float(mu_x_sev_ui)
    INFANT_MK["cs_early_cure_rate"]    = cs_cure_ui
    INFANT_MK["cs_late_manifest_rate"] = cs_late_ui
    INFANT_MK["cs_neuro_disorder_rate"]= cs_neuro_ui

    st.subheader("Long-Term Care Module")
    use_ltc = st.checkbox(
        "Include direct long-term care costs (special education + paid caregiving)",
        value=True,
        help="Adds age-gated special education (IDEA, ages 3–21) and direct paid "
             "caregiver costs to the infant Markov cycle. Distinct from the "
             "caregiver opportunity costs in the Productivity Loss module."
    )
    with st.expander("Long-term care parameters", expanded=False):
        ltc_p_sped_sev = st.slider("P(special ed | severe CS sequelae)",   0.50, 1.00, 0.85, 0.05)
        ltc_p_sped_mid = st.slider("P(special ed | mild CS sequelae)",     0.10, 0.70, 0.35, 0.05)
        ltc_cost_sped  = st.number_input("Annual incremental special ed cost ($)",
                                              5_000, 40_000, 14_000, 500)
        ltc_sped_start = st.number_input("Special ed start age (IDEA Part B)", 0, 5, 3, 1)
        ltc_sped_end   = st.number_input("Special ed end age (IDEA ceiling)",  18, 26, 21, 1)
        ltc_cost_cg_sv = st.number_input("Annual paid caregiver cost — severe ($)",
                                              10_000, 80_000, 32_000, 1_000)
        ltc_cost_cg_ml = st.number_input("Annual paid caregiver cost — mild ($)",
                                              500, 20_000, 4_500, 500)
        ltc_cg_end     = st.number_input("Caregiver cost end age", 5, 21, 18, 1)

    # Instantiate (after sidebar block, before run_psa):
    ltc_obj = LongTermCare(
        p_sped_severe      = ltc_p_sped_sev,
        p_sped_mild        = ltc_p_sped_mid,
        cost_sped_ann      = float(ltc_cost_sped),
        cost_cg_severe_ann = float(ltc_cost_cg_sv),
        cost_cg_mild_ann   = float(ltc_cost_cg_ml),
        sped_start_age     = int(ltc_sped_start),
        sped_end_age       = int(ltc_sped_end),
        caregiver_end_age  = int(ltc_cg_end),
    ) if use_ltc else None
        
    st.subheader("Maternal Morbidity Module")
    use_mat_morb = st.checkbox("Include maternal morbidity (cardiovascular, neuro, hospitalisation)", value=True)
    with st.expander("Maternal morbidity parameters", expanded=False):
        mm_p_cardio    = st.slider("P(cardiovascular syphilis | untreated late latent)", 0.01, 0.15, 0.05, 0.01)
        mm_p_neuro     = st.slider("P(neurosyphilis | tertiary)",                        0.02, 0.25, 0.10, 0.01)
        mm_p_hosp      = st.slider("P(pregnancy hospitalisation | active infection)",    0.03, 0.30, 0.12, 0.01)
        mm_cost_cardio = st.number_input("Annual cardiovascular treatment cost ($)", 5_000, 50_000, 18_500, 500)
        mm_cost_neuro  = st.number_input("Annual neurosyphilis treatment cost ($)", 10_000, 80_000, 32_000, 1_000)
        mm_cost_hosp   = st.number_input("Pregnancy hospitalisation cost ($)", 1_000, 15_000, 4_200, 200)

    st.subheader("Productivity Loss Module")
    use_prod_loss = st.checkbox("Include productivity losses (human capital)", value=True)
    use_friction  = st.checkbox("Use friction-cost variant (90-day cap)", value=False)
    with st.expander("Productivity loss parameters", expanded=False):
        pl_bereavement_days   = st.slider("Lost work-days per bereavement (SB/NND)", 5, 90, 30, 1)
        pl_wage_mild          = st.slider("Lifetime wage penalty – mild CS sequelae (%)", 2, 30, 10, 1) / 100
        pl_wage_severe        = st.slider("Lifetime wage penalty – severe CS sequelae (%)", 10, 60, 40, 1) / 100
        pl_caregiver_hrs      = st.slider("Weekly caregiver hours – CS complicated infant", 2, 30, 10, 1)
        pl_caregiver_wage_frac = st.slider("Caregiver wage fraction (of maternal weighted earnings)", 0.20, 1.00, 0.50, 0.05)

    st.subheader("DALYs & Discounting")
    r_disc  = st.number_input("Discount rate", 0.0, 0.08, 0.035, 0.005, format="%.3f")
    LE      = st.number_input("Life expectancy at birth (years)", 60.0, 90.0, 78.0, 1.0)
    inc_lbw = st.checkbox("Include LBW YLD", value=True)
    inc_mat = st.checkbox("Include maternal grief YLD", value=True)
    inc_sb_yll = st.checkbox(
        "Include stillbirth YLL (IUFD ≥28w subset)", value=True,
        help="Adds discounted lifetime YLL for d_iufd_subset = stillbirths classified as IUFD ≥28w."
    )
    inc_cs_yll = st.checkbox(
        "Include CS post-neonatal excess-mortality YLL", value=True,
        help="Requires non-zero annual excess mortality in the Infant Markov parameters; background life-table deaths are not charged as CS YLL."
    )
    inc_misc_yld = st.checkbox(
        "Include miscarriage grief YLD", value=True,
        help="The component is explicit, but remains zero unless miscarriage events differ between arms."
    )
    inc_mat_hosp_yld = st.checkbox(
        "Include maternal hospitalisation acute YLD", value=True,
        help="Converts pregnancy hospitalisation episodes into acute YLD using duration from MaternalMorbidity.dur_hosp_days."
    )
    inc_preterm_yld = st.checkbox("Include preterm birth acute infant YLD", value=True)

    st.subheader("VSL Reference Analysis")
    vsl = st.number_input("Value of Statistical Life ($)", 5_000_000, 25_000_000,
                          13_700_000, 500_000, format="%d")

    st.subheader("PSA Settings")
    N_iter  = st.number_input("MC iterations", 2_000, 100_000, 10_000, 1_000)
    seed    = st.number_input("Random seed", 0, 99_999, 2025, 1)
    wtp_max = st.number_input("Max WTP for CEAC / EVPI ($/DALY)", 50_000, 500_000, 200_000, 10_000)

   
    st.subheader("Budget Impact — Population & Ramp")

    with st.expander("Payer population", expanded=False):
        bia_covered   = st.number_input(
            "Covered lives", 10_000, 10_000_000, 500_000, 50_000)
        bia_frac_rf   = st.slider(
            "Fraction reproductive-age female (15–44)",
            0.05, 0.25, 0.135, 0.005, format="%.3f")
        bia_preg_rate = st.slider(
            "Annual pregnancy rate (reproductive-age females)",
            0.04, 0.15, 0.085, 0.005, format="%.3f")
        bia_p_ed      = st.slider(
            "P(ED visit during pregnancy)",
            0.20, 0.70, 0.45, 0.05)
        bia_p_unscr   = st.slider(
            "P(not previously screened | ED presentation)",
            0.10, 0.80, 0.35, 0.05,
            help="Captures catchment above existing prenatal screening. "
                 "Higher in low-prenatal-care populations.")
        bia_payer_frac = st.slider(
            "Payer fraction of ED volume",
            0.10, 1.00, 0.40, 0.05,
            help="Fraction of ED volume covered by this payer. "
                 "e.g. 0.40 for a Medicaid plan.")

    with st.expander("Implementation ramp (S-curve)", expanded=True):
        st.caption(
            "Define two milestones that characterise your implementation "
            "trajectory. The model fits a sigmoid between them."
        )
        bia_n_yrs = st.slider("Projection horizon (years)", 1, 10, 5)

        rc1, rc2 = st.columns(2)
        with rc1:
            bia_t_half = st.slider(
                "Year at 50% of coverage gain",
                min_value=0.5,
                max_value=float(bia_n_yrs) - 0.5,
                value=1.5, step=0.5,
                help="Inflection point — when adoption is growing fastest. "
                     "Corresponds to staff training complete, workflow embedded."
            )
        with rc2:
            bia_t_ninety = st.slider(
                "Year at 90% of coverage gain",
                min_value=bia_t_half + 0.5,
                max_value=float(bia_n_yrs),
                value=float(min(bia_t_half + 1.5, bia_n_yrs)),
                step=0.5,
                help="Operational maturity — programme fully embedded. "
                     "EHR integration complete, reflex testing routine."
            )

        if bia_t_ninety <= bia_t_half:
            st.error(
                "Year at 90% coverage must be later than year at 50% coverage.")


# ─── Calibrate q_progress ───────────────────────────────────────────────────
T_calib = max(int(LE), 1)
q_cal   = calibrate_q_progress(q_target, T_calib)
INFANT_MK["q_progress"] = q_cal
implied_prog = implied_lifetime_prog(q_cal, T_calib)


# ─── Run PSA ────────────────────────────────────────────────────────────────
with st.spinner("Running Monte Carlo PSA…"):
    df_psa, smry, comp_means, intr_means = run_psa(
        N=int(N_iter), seed=int(seed), cohort=int(cohort),
        p_act=float(p_act), p_sf=float(p_sf), p_id=float(p_id),
        sc_b=float(_sc_uc_eff), sc_e=float(sc_e),
        sens=float(sens), spec=float(spec),
        p_adeq=float(p_adeq), p_tx_override=float(p_tx) if p_tx is not None else None,
        p_trepo_sf=float(p_trepo_sf), p_ux_sf=float(p_ux_sf),
        prop_symp=float(prop_symp), prop_late=float(prop_late),
        r=float(r_disc), LE=float(LE),
        inc_lbw=bool(inc_lbw), inc_mat=bool(inc_mat),
        inc_sb_yll=bool(inc_sb_yll), inc_cs_yll=bool(inc_cs_yll),
        inc_misc_yld=bool(inc_misc_yld), inc_mat_hosp_yld=bool(inc_mat_hosp_yld),
        inc_preterm_yld=bool(inc_preterm_yld),
        treat_fp=bool(treat_fp), vsl=float(vsl),
        use_mat_morb=bool(use_mat_morb), use_prod_loss=bool(use_prod_loss),
        use_friction=bool(use_friction),
        mm_p_cardio=float(mm_p_cardio), mm_p_neuro=float(mm_p_neuro),
        mm_p_hosp=float(mm_p_hosp),
        mm_cost_cardio=float(mm_cost_cardio), mm_cost_neuro=float(mm_cost_neuro),
        mm_cost_hosp=float(mm_cost_hosp),
        pl_bereavement_days=float(pl_bereavement_days),
        pl_wage_mild=float(pl_wage_mild), pl_wage_severe=float(pl_wage_severe),
        pl_caregiver_hrs=float(pl_caregiver_hrs),
        pl_caregiver_wage_frac=float(pl_caregiver_wage_frac),
        mk_p_sev    = float(p_sev_ui),
        mk_p_mc     = float(p_mc_ui),
        mk_p_mu     = float(p_mu_ui),
        mk_c_mild   = float(c_mild_ui),
        mk_c_sev    = float(c_sev_ui),
        mk_q_target = float(q_target),
        mk_mu_x_mild = float(mu_x_mild_ui),
        mk_mu_x_sev  = float(mu_x_sev_ui),
        use_ltc         = bool(use_ltc),
        ltc_p_sped_sev  = float(ltc_p_sped_sev),
        ltc_p_sped_mid  = float(ltc_p_sped_mid),
        ltc_cost_sped   = float(ltc_cost_sped),
        ltc_cost_cg_sv  = float(ltc_cost_cg_sv),
        ltc_cost_cg_ml  = float(ltc_cost_cg_ml),
        ltc_sped_start  = int(ltc_sped_start),
        ltc_sped_end    = int(ltc_sped_end),
        ltc_cg_end      = int(ltc_cg_end),
    )

sc_uc_eff, tx_eff, prop_late_eff = ges_eff()


EVPPI_GROUPS = {
    "Epidemiological (RRs)": [
        c for c in df_psa.columns if c.startswith("rr_")
    ],
    "Epidemiological (untreated risks)": [
        c for c in df_psa.columns if c.startswith("ur_")
    ],
    "Background risks": [
        c for c in df_psa.columns if c.startswith("br_")
    ],
    "Cost inputs": [
        c for c in df_psa.columns if c.startswith("co_")
    ],
    "Disability weights": [
        c for c in df_psa.columns if c.startswith("dw_")
    ],
    "Infant Markov (sequelae probs + costs)": [
        c for c in df_psa.columns if c.startswith("mkp_")
    ],
    "Maternal morbidity": [
        c for c in df_psa.columns if c.startswith("mmp_")
    ],
    "Serofast":  [
        c for c in df_psa.columns if c.startswith("sf_")], 
}

_empty_groups = [k for k, v in EVPPI_GROUPS.items() if not v]
if _empty_groups:
    st.sidebar.warning(
        f"EVPPI: no draw columns found for : {', '.join(_empty_groups)}. "
        "These groups will show EVPPI = 0."
    )
# ─── Top KPI strip ──────────────────────────────────────────────────────────
st.title("🏥 ED Universal Syphilis Screening — CEA  (v4)")
st.caption(
    f"Strata-weighted baseline coverage: **{sc_uc_eff:.1%}** → Enhanced: **{sc_e:.0%}**  |  "
    f"P(pregnancy identified): **{p_id:.0%}**  |  Preset: **{preset}**  |  "
    f"Calibrated q_progress: **{q_cal:.4f}** (target {q_target:.0%} lifetime prog; "
    f"implied: **{implied_prog:.1%}**)  |  "
    f"v4: VSL/DALY separated · maternal morbidity · productivity loss (human capital) · "
    f"stratum-weighted prop_late · q_progress calibration · EVPI · convergence diagnostic · OWSA table"
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("ICER – Health Sector",  f"${smry['icer_hs']['mean']:,.0f}/DALY")
k2.metric("ICER – Societal",       f"${smry['icer_soc']['mean']:,.0f}/DALY")
k3.metric("ΔCost (Health Sector)", f"${smry['inc_cost_hs']['mean']/1e6:,.2f}M")
k4.metric("P(dominant – HS)",      f"{smry['p_dominant_hs']:.1%}")
k5.metric("DALYs averted (HS)",    f"{smry['dalys_hs']['mean']:,.0f}")
k6.metric("VSL net benefit (mean)",f"${smry['vsl_nb']['mean']/1e6:,.1f}M",
          help="Lives saved × VSL minus health-sector program cost. "
               "Presented as a separate willingness-to-pay metric, not mixed into ICER.")
st.divider()


# ─── Tabs ───────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Standard CEA",
    "🌍 Societal Perspective",
    "🏦 Budget Impact",
    "📍 Threshold Analysis",
    "🧬 Infant Markov",
    "🔬 Serofast Detail",
    "📋 Assumptions & Citations",
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 · Standard CEA (health-sector DALY perspective)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    
    st.subheader("Health-Sector Cost-Effectiveness Analysis")
    
    # ── Clinical Impact Summary ──────────────────────────────────────────
    st.subheader("Clinical Impact Summary")
    st.caption(
        "Deterministic values for screened and detected; "
        "probabilistic (mean, 95% CrI) for outcomes averted."
    )

    n_incr_screened = max(sc_e * p_id - sc_uc_eff * p_id, 0.0) * cohort
    n_detected      = n_incr_screened * p_act * sens
    n_treated       = n_detected * p_adeq * tx_eff

    ci1, ci2, ci3, ci4, ci5 = st.columns(5)
    ci1.metric(
        "Incremental women screened",
        f"{n_incr_screened:,.0f}",
        help="Above usual-care baseline; sc_e × p_id − sc_uc × p_id × cohort"
    )
    ci2.metric(
        "Incremental cases detected",
        f"{n_detected:,.1f}",
        help="Incremental screened × p_act × sensitivity"
    )
    ci3.metric(
        "Incremental cases treated",
        f"{n_treated:,.1f}",
        help="Cases detected × p_adeq × tx completion"
    )
    ci4.metric(
        "CS cases averted (mean)",
        f"{smry['d_cs_comp']['mean'] + smry['d_cs_uncomp']['mean']:,.1f}",
        help=(
            f"95% CrI: "
            f"{smry['d_cs_comp']['95% CrI lo'] + smry['d_cs_uncomp']['95% CrI lo']:,.1f}"
            f"–"
            f"{smry['d_cs_comp']['95% CrI hi'] + smry['d_cs_uncomp']['95% CrI hi']:,.1f}"
        )
    )
    ci5.metric(
        "Stillbirths + NNDs averted (mean)",
        f"{smry['d_stillbirth']['mean'] + smry['d_neonatal_death']['mean']:,.1f}",
        help=(
            f"95% CrI: "
            f"{smry['d_stillbirth']['95% CrI lo'] + smry['d_neonatal_death']['95% CrI lo']:,.1f}"
            f"–"
            f"{smry['d_stillbirth']['95% CrI hi'] + smry['d_neonatal_death']['95% CrI hi']:,.1f}"
        )
    )

    # Detailed outcomes table
    with st.expander("Outcomes averted — full breakdown with 95% CrI", expanded=False):
        impact_rows = []
        for label, key_comp, key_uncomp in [
            ("Congenital syphilis (total)",  "d_cs_comp", "d_cs_uncomp"),
        ]:
            lo = smry[key_comp]["95% CrI lo"] + smry[key_uncomp]["95% CrI lo"]
            hi = smry[key_comp]["95% CrI hi"] + smry[key_uncomp]["95% CrI hi"]
            mn = smry[key_comp]["mean"]       + smry[key_uncomp]["mean"]
            impact_rows.append({
                "Outcome": label,
                "Mean": f"{mn:,.1f}",
                "95% CrI": f"{lo:,.1f} – {hi:,.1f}",
            })
        for label, key in [
            ("CS — complicated",   "d_cs_comp"),
            ("CS — uncomplicated", "d_cs_uncomp"),
            ("Stillbirths",        "d_stillbirth"),
            ("Neonatal deaths",    "d_neonatal_death"),
            ("Preterm births",     "d_preterm"),
            ("Low birth weight",   "d_lbw"),
        ]:
            s = smry[key]
            impact_rows.append({
                "Outcome": label,
                "Mean": f"{s['mean']:,.1f}",
                "95% CrI": f"{s['95% CrI lo']:,.1f} – {s['95% CrI hi']:,.1f}",
            })
        st.dataframe(
            pd.DataFrame(impact_rows), hide_index=True, width="stretch"
        )

    st.divider()

    
    st.subheader("Health-Sector Cost-Effectiveness Analysis")
    st.info(
        "**Perspective: health sector (payer).** "
        "Costs = program costs minus direct medical savings. "
        "Effectiveness = DALYs averted (infant outcomes + CS lifetime Markov). "
        "VSL and productivity losses appear in the **Societal** tab."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig_ce_plane(df_psa["dal_hs"].values, df_psa["ic_hs"].values,
                               "CE Plane — Health Sector"), width="stretch")
        buf = io.BytesIO()
        fig = fig_ce_plane(df_psa["dal_hs"].values, df_psa["ic_hs"].values,
                               "CE Plane — Health Sector")
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(label="Download CE Plane", data=buf, file_name="ce_plane.png", mime="image/png")

    with c2:
        st.pyplot(fig_convergence(df_psa["dal_hs"].values, df_psa["ic_hs"].values),
                  width="stretch")

    st.pyplot(fig_ceac(df_psa["dal_hs"].values, df_psa["ic_hs"].values,
                       df_psa["dal_soc"].values, df_psa["ic_soc"].values,
                       wtp_max=int(wtp_max)), width="stretch")
    buf = io.BytesIO()
    fig = fig_ceac(df_psa["dal_hs"].values, df_psa["ic_hs"].values,
                       df_psa["dal_soc"].values, df_psa["ic_soc"].values,
                       wtp_max=int(wtp_max))
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    st.download_button(label="Download CEAC Curve", data=buf, file_name="ceac.png", mime="image/png")

    col_evpi, col_quad = st.columns([3, 2])
    with col_evpi:
        st.pyplot(
            fig_evpi(df_psa["dal_hs"].values, df_psa["ic_hs"].values, int(wtp_max)),
            width="stretch",
        )
    with col_quad:
        st.subheader("CE-Plane Quadrant Analysis")
        quad_df = ce_quadrant_table(df_psa["dal_hs"].values, df_psa["ic_hs"].values)
        st.dataframe(quad_df, width="stretch", hide_index=True)
        st.caption("Iterations: {:,}  |  Seed: {:,}".format(int(N_iter), int(seed)))

    st.subheader("Expected Value of Partially Perfect Information (EVPPI)")
    st.info(
        "EVPPI estimates the maximum value of resolving uncertainty in each "
        "parameter group alone, holding all other uncertainty fixed. "
        "It identifies *which* parameter groups most warrant further research. "
        "Method: degree-2 Ridge polynomial regression on NMB "
        "(Strong et al. 2014, *Med Decis Making*)."
    )

    evppi_persp = st.radio(
        "Perspective for EVPPI",
        ["Health sector", "Societal"],
        horizontal=True,
        key="evppi_persp",
    )
    evppi_persp_key = "hs" if evppi_persp == "Health sector" else "soc"
    evppi_wtp = st.selectbox(
        "WTP for EVPPI ($/DALY)",
        [50_000, 100_000, 150_000, 200_000],
        index=1,
        format_func=lambda x: f"${x/1000:.0f}K/DALY",
        key="evppi_wtp",
    )

    with st.spinner("Computing EVPPI (polynomial regression per parameter group)…"):
        evpi_total, evppi_results = compute_evppi(
            df          = df_psa,
            wtp         = float(evppi_wtp),
            perspective = evppi_persp_key,
            param_groups= EVPPI_GROUPS,
        )

    # KPI strip
    ep1, ep2, ep3 = st.columns(3)
    ep1.metric("EVPI (upper bound)",   f"${evpi_total/1e3:,.1f}K per cohort")
    top_group = max(evppi_results, key=evppi_results.get)
    ep2.metric("Highest-value group",  top_group)
    ep3.metric("EVPPI — top group",    f"${evppi_results[top_group]/1e3:,.1f}K per cohort")

    # Bar chart
    st.pyplot(
        fig_evppi_bar(evppi_results, evpi_total, float(evppi_wtp), evppi_persp_key),
        width="stretch",
    )
    buf = io.BytesIO()
    fig_evppi_bar(evppi_results, evpi_total,
                  float(evppi_wtp), evppi_persp_key).savefig(
        buf, format="png", dpi=300, bbox_inches="tight"
    )
    buf.seek(0)
    st.download_button("Download EVPPI chart", buf,
                       "evppi_bar.png", "image/png")

    # Summary table
    evppi_df = pd.DataFrame([
        {
            "Parameter group":          k,
            "EVPPI ($K per cohort)":    f"${v/1e3:,.1f}K",
            "% of EVPI":                f"{v / max(evpi_total, 1e-9):.1%}",
            "Research priority":        (
                "🔴 High"   if v / max(evpi_total, 1e-9) > 0.20 else
                "🟡 Medium" if v / max(evpi_total, 1e-9) > 0.05 else
                "🟢 Low"
            ),
        }
        for k, v in sorted(evppi_results.items(),
                            key=lambda x: x[1], reverse=True)
    ])
    st.dataframe(evppi_df, hide_index=True, width="stretch")
    st.caption(
        "Research priority thresholds: >20% of EVPI = High, 5–20% = Medium, <5% = Low. "
        "EVPPI values below $1K per cohort are practically negligible regardless of threshold."
    )

    # Download EVPPI table as CSV
    st.download_button(
        "⬇ Download EVPPI table (CSV)",
        evppi_df.to_csv(index=False).encode(),
        "evppi_results.csv",
        "text/csv",
    )

#   Break 03/16/26 - cont. PSA next time

    # PSA summary table
    st.subheader("PSA Summary — Health Sector")
    psa_rows = []
    for label, key in [
        ("Incremental cost ($)",           "inc_cost_hs"),
        ("DALYs averted",                  "dalys_hs"),
        ("ICER ($/DALY)",                  "icer_hs"),
        ("P(cost-saving)",                 None),
        ("P(dominant)",                    None),
    ]:
        if key:
            s = smry[key]
            psa_rows.append({
                "Metric": label,
                "Mean":       f"${s['mean']:,.1f}"   if "cost" in key or "ICER" in key else f"{s['mean']:,.1f}",
                "Median":     f"${s['median']:,.1f}" if "cost" in key or "ICER" in key else f"{s['median']:,.1f}",
                "95% CrI lo": f"${s['95% CrI lo']:,.1f}" if "cost" in key or "ICER" in key else f"{s['95% CrI lo']:,.1f}",
                "95% CrI hi": f"${s['95% CrI hi']:,.1f}" if "cost" in key or "ICER" in key else f"{s['95% CrI hi']:,.1f}",
            })
    psa_rows.append({"Metric": "P(cost-saving)", "Mean": f"{smry['p_cost_saving_hs']:.1%}",
                     "Median": "—", "95% CrI lo": "—", "95% CrI hi": "—"})
    psa_rows.append({"Metric": "P(dominant)",    "Mean": f"{smry['p_dominant_hs']:.1%}",
                     "Median": "—", "95% CrI lo": "—", "95% CrI hi": "—"})
    st.dataframe(pd.DataFrame(psa_rows), width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PATCH 2 of 2: Baseline outcomes display block
# ══════════════════════════════════════════════════════════════════════════════
    
    OUTCOME_LABELS = {
        "preterm":        "Preterm births",
        "lbw":            "Low birth weight",
        "stillbirth":     "Stillbirths (≥20w)",
        "iufd_subset":    "IUFD ≥28w (subset of SB)",
        "neonatal_death": "Neonatal deaths (<28d)",
        "cs_comp":        "CS – complicated",
        "cs_uncomp":      "CS – uncomplicated",
    }

    # Quick sanity-check rates (outcomes / cohort)
    def _rate(n, denom):
        return n / denom if denom > 0 else 0.0

    with st.expander("🔎 Baseline vs. Intervention Outcome Counts (Sanity Check)", expanded=False):
        st.caption(
            f"Comparator arm: strata-weighted usual-care coverage = **{sc_uc_eff:.1%}**  |  "
            f"Intervention arm: enhanced coverage = **{sc_e:.1%}**  |  "
            f"Cohort = **{cohort:,}**  |  Active syphilis prevalence = **{p_act:.1%}**"
        )

        comp_m = comp_means
        intr_m = intr_means

        rows_baseline = []
        for key, label in OUTCOME_LABELS.items():
            c_n   = comp_m[key]
            i_n   = intr_m[key]
            delta = c_n - i_n          # positive = prevented by intervention
            c_r   = _rate(c_n, cohort) * 1000   # per 1,000
            i_r   = _rate(i_n, cohort) * 1000
            pct_r = delta / c_n * 100 if c_n > 0 else 0.0
            rows_baseline.append({
                "Outcome":                   label,
                "Comparator (n)":            f"{c_n:,.1f}",
                "Rate per 1,000":            f"{c_r:.2f}",
                "Intervention (n)":          f"{i_n:,.1f}",
                "Rate per 1,000 ":           f"{i_r:.2f}",   # trailing space avoids dup col name
                "Prevented (Δn)":            f"{delta:,.1f}",
                "% reduction":               f"{pct_r:.1f}%",
            })

        st.dataframe(pd.DataFrame(rows_baseline), hide_index=True, width="stretch")

        # ── Implicit rate cross-checks ────────────────────────────────────────────
        st.subheader("Implicit rate cross-checks vs. published anchors")

        sb_rate_comp  = _rate(comp_m["stillbirth"],     cohort)
        nnd_rate_comp = _rate(comp_m["neonatal_death"],  cohort)
        cs_rate_comp  = _rate(comp_m["cs_comp"] + comp_m["cs_uncomp"], cohort)
        pt_rate_comp  = _rate(comp_m["preterm"],         cohort)

        # Expected: prevalence × untreated absolute risk + (1-prev) × background rate
        sc_uc, _, _ = ges_eff()
        p_eff_comp  = sc_uc * p_id * sens * p_adeq   # approx scalar for display

        expected_sb  = p_act * UNT_ABS["stillbirth"] + (1 - p_act) * (
            BASE_BETA["stillbirth"]["a"] / (BASE_BETA["stillbirth"]["a"] + BASE_BETA["stillbirth"]["b"])
        )
        expected_cs  = p_act * UNT_ABS["cs_any"]     # upper bound (no treatment at all)

        checks = [
             ("Stillbirth rate (comparator)",
             f"{sb_rate_comp*100:.3f}%",
             f"≈{expected_sb*100:.3f}% (naïve: prev×untreated + (1-prev)×background)",
             abs(sb_rate_comp - expected_sb) < 0.005),
             ("CS rate (comparator, all)",
             f"{cs_rate_comp*100:.3f}%",
             f"≤{expected_cs*100:.3f}% (untreated upper bound × prev = {p_act:.1%})",
             cs_rate_comp <= expected_cs + 0.001),
             ("Preterm rate (comparator)",
             f"{pt_rate_comp*100:.3f}%",
             "Typically 10–15% in mixed populations",
             0.08 <= pt_rate_comp <= 0.18),
             ("NND rate (comparator)",
             f"{nnd_rate_comp*1000:.2f} per 1,000",
             "Background ~3–4/1,000; syphilis+ ~160/1,000",
             nnd_rate_comp < 0.05),
        ]

        for desc, observed, reference, ok in checks:
            icon = "✅" if ok else "⚠️"
            st.markdown(f"{icon} **{desc}**: observed = `{observed}` | reference = {reference}")


    # DALY decomposition
    st.subheader("DALY Decomposition")
    d1, d2, d3, d4 = st.columns(4)
    tot_hs = smry["dalys_hs"]["mean"]
    d1.metric("Total (HS)",           f"{tot_hs:,.1f}")
    d2.metric("Markov CS lifetime",   f"{smry['dalys_markov']['mean']:,.1f}",
              delta=f"{smry['dalys_markov']['mean']/max(tot_hs,1):.0%} of total")
    d3.metric("Non-CS infant + grief", f"{smry['dalys_non_cs']['mean']:,.1f}")
    d4.metric("Maternal morbidity",   f"{smry['dalys_mat_morb']['mean']:,.1f}",
              delta="(societal only)" if use_mat_morb else "disabled")

    with st.expander("DALY component audit table", expanded=False):
        daly_component_rows = [
            ("Neonatal death YLL", "dalys_nnd_yll"),
            ("Stillbirth YLL — IUFD ≥28w", "dalys_stillbirth_yll"),
            ("CS Markov YLD", "dalys_markov_yld"),
            ("CS Markov excess-mortality YLL", "dalys_markov_yll"),
            ("LBW YLD", "dalys_lbw_yld"),
            ("Preterm birth acute YLD", "dalys_preterm_yld"),
            ("Miscarriage grief YLD", "dalys_miscarriage_yld"),
            ("Maternal grief YLD — stillbirth", "dalys_mat_sb_grief_yld"),
            ("Maternal grief YLD — NND", "dalys_mat_nnd_grief_yld"),
            ("Maternal cardiovascular syphilis YLD", "dalys_mat_cardio_dal"),
            ("Maternal neurosyphilis YLD", "dalys_mat_neuro_dal"),
            ("Maternal hospitalisation acute YLD", "dalys_mat_hosp_dal"),
        ]
        rows_daly = []
        for label, key in daly_component_rows:
            if key in smry:
                s_ = smry[key]
                rows_daly.append({
                    "Component": label,
                    "Mean": f"{s_['mean']:,.2f}",
                    "95% CrI": f"{s_['95% CrI lo']:,.2f} – {s_['95% CrI hi']:,.2f}",
                    "Included in": "Societal only" if label.startswith("Maternal ") and "grief" not in label else "Health-sector and societal",
                })
        st.dataframe(pd.DataFrame(rows_daly), hide_index=True, width="stretch")
        st.caption(
            "Stillbirth YLL uses d_iufd_subset (IUFD ≥28w), not all stillbirths ≥20w. "
            "CS Markov YLL remains zero unless excess-mortality sliders are set above zero."
        )

    # Outcomes table
    st.subheader("Outcomes Prevented — 95% CrI Summary")
    outcome_labels = {
        "d_preterm": "Preterm births", "d_lbw": "Low birth weight",
        "d_stillbirth": "Stillbirths (≥20w)", "d_iufd_subset": "IUFD ≥28w (subset of SB)",
        "d_neonatal_death": "Neonatal deaths (<28d)",
        "d_cs_comp": "CS – complicated", "d_cs_uncomp": "CS – uncomplicated",
    }
    rows = []
    for col, label in outcome_labels.items():
        if col in smry:
            s = smry[col]
            rows.append({"Outcome": label,
                         "Mean": f"{s['mean']:,.1f}", "Median": f"{s['median']:,.1f}",
                         "95% CrI lo": f"{s['95% CrI lo']:,.1f}",
                         "95% CrI hi": f"{s['95% CrI hi']:,.1f}"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # OWSA
    st.subheader("One-Way Sensitivity Analysis (OWSA)")
    base_kw = dict(
        p_act=p_act, p_sf=p_sf, p_id=p_id, sc_b=sc_uc_eff, sc_e=sc_e,
        sens=sens, spec=spec, p_adeq=p_adeq, prop_symp=prop_symp,
        prop_late=prop_late, p_trepo_sf=p_trepo_sf, p_ux_sf=p_ux_sf,
        r=r_disc, LE=LE, inc_lbw=inc_lbw, inc_mat=inc_mat, cohort=int(cohort),
        ltc=ltc_obj if use_ltc else None,
        inc_sb_yll=inc_sb_yll, inc_cs_yll=inc_cs_yll,
        inc_misc_yld=inc_misc_yld, inc_mat_hosp_yld=inc_mat_hosp_yld,
        inc_preterm_yld=inc_preterm_yld,
    )
    
    sf_ratio = p_sf / max(p_act, 1e-9)

    def linked_sf(prev):
        return float(np.clip(sf_ratio * prev, 0.0001, 0.010))

    if not p_sf_override:
        # Active prevalence OWSA varies active syphilis and the linked serofast pool.
        prevalence_low = {
            "p_act": 0.005,
            "p_sf": linked_sf(0.005),
        }
        prevalence_high = {
            "p_act": 0.030,
            "p_sf": linked_sf(0.030),
        }

        # Serofast row now varies the serofast-to-active ratio at fixed p_act.
        serofast_low = {
            "p_sf": 0.10 * p_act,
        }
        serofast_high = {
            "p_sf": 0.40 * p_act,
        }
        serofast_label = "Serofast ratio among active syphilis"

    else:
        # If the user manually overrides p_sf, treat active prevalence and
        # serofast prevalence as independent OWSA parameters.
        prevalence_low = {
            "p_act": 0.005,
        }
        prevalence_high = {
            "p_act": 0.030,
        }

        serofast_low = {
            "p_sf": max(p_sf * 0.25, 0.0001),
        }
        serofast_high = {
            "p_sf": min(p_sf * 3.0, 0.010),
        }
        serofast_label = "Serofast prevalence"
        
    param_ranges = {
        "Prevalence":          ({"p_act": 0.005}, {"p_act": 0.030}),
        "Treatment rate":      ({"p_adeq": 0.50}, {"p_adeq": 0.95}),
        "P(pregnancy ID)":     ({"p_id": 0.65},   {"p_id": 0.98}),
        "Serofast prevalence": (serofast_low, serofast_high),
        "Discount rate":       ({"r": 0.00},       {"r": 0.05}),
        "Test sensitivity":    ({"sens": 0.90},    {"sens": 1.00}),
        "Screen coverage":     ({"sc_e": 0.75},    {"sc_e": 0.98}),
        "Life expectancy":     ({"LE": 70.0},      {"LE": 85.0}),
        "P(ux tx | serofast)": ({"p_ux_sf": 0.05}, {"p_ux_sf": 0.50}),
        "Prop CS complicated": ({"prop_symp": 0.20}, {"prop_symp": 0.60}),
    }

    if not p_sf_override:
        st.caption(
            "OWSA note: because serofast prevalence is locked to active syphilis "
            "prevalence by default, the active-prevalence row varies both p_act "
            "and p_sf. The serofast row varies the serofast-to-active ratio at "
            "fixed active prevalence."
        )
        
    with st.spinner("Computing OWSA…"):
        owsa_df = owsa_table(base_kw, param_ranges)
        
    base_icer = owsa_df["Base ICER"].iloc[0]
    st.pyplot(fig_tornado(owsa_df, base_icer), width="stretch")
    fig = fig_tornado(owsa_df, base_icer)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(label="Tornado Plot", data=buf, file_name="tornado.png", mime="image/png")

    st.subheader("OWSA Detail Table")
    disp_owsa = owsa_df[["Parameter", "Low param value", "High param value",
                          "ICER (low)", "ICER (high)", "Range"]].copy()
    for c in ["ICER (low)", "ICER (high)", "Range"]:
        disp_owsa[c] = disp_owsa[c].map(lambda x: f"${x:,.0f}")
    st.dataframe(disp_owsa, width="stretch", hide_index=True)

    
    # ── NMB-based OWSA ───────────────────────────────────────────────────────
    st.subheader("One-Way Sensitivity Analysis — NMB (Preferred Method)")
    st.info(
        "NMB = λ · ΔDALYs − ΔCost.  "
        "Unlike the ICER tornado above, NMB is **linear** in costs and effects, "
        "**well-behaved when the program is cost-saving**, and avoids the ratio "
        "sign-flip that occurs when ΔCost crosses zero. "
        "Bars show how NMB changes as each parameter moves from its low to high value. "
        "The dotted line at NMB = 0 is the break-even threshold at the chosen WTP."
    )

    nmb_col1, nmb_col2 = st.columns([2, 1])
    with nmb_col1:
        nmb_wtp = st.selectbox(
            "WTP threshold for NMB tornado ($/DALY)",
            [50_000, 100_000, 150_000, 200_000],
            index=1,
            format_func=lambda x: f"${x/1000:.0f}K/DALY",
            key="nmb_owsa_wtp",
        )
    with nmb_col2:
        nmb_persp = st.radio(
            "Perspective",
            ["Health sector", "Societal"],
            horizontal=True,
            key="nmb_owsa_persp",
        )
    nmb_persp_key = "hs" if nmb_persp == "Health sector" else "soc"

    # Add societal-perspective kwargs when selected; mm/pl are already built
    # above for the threshold-analysis tab — reuse the same objects here.
    base_kw_nmb = dict(
        p_act=p_act, p_sf=p_sf, p_id=p_id, sc_b=sc_uc_eff, sc_e=sc_e,
        sens=sens, spec=spec, p_adeq=p_adeq, prop_symp=prop_symp,
        prop_late=prop_late, p_trepo_sf=p_trepo_sf, p_ux_sf=p_ux_sf,
        r=r_disc, LE=LE, inc_lbw=inc_lbw, inc_mat=inc_mat,
        cohort=int(cohort),
        ltc=ltc_obj if use_ltc else None,
        inc_sb_yll=inc_sb_yll, inc_cs_yll=inc_cs_yll,
        inc_misc_yld=inc_misc_yld, inc_mat_hosp_yld=inc_mat_hosp_yld,
        inc_preterm_yld=inc_preterm_yld,
        # Societal modules — only active when perspective == "soc"
        mm=(MaternalMorbidity(
                p_cardio=mm_p_cardio, p_neuro=mm_p_neuro, p_hosp=mm_p_hosp,
                cost_cardio=mm_cost_cardio, cost_neuro=mm_cost_neuro,
                cost_hosp=mm_cost_hosp)
            if (use_mat_morb and nmb_persp_key == "soc") else None),
        pl=(ProductivityLoss(
                bereavement_days=pl_bereavement_days,
                wage_penalty_mild=pl_wage_mild, wage_penalty_severe=pl_wage_severe,
                caregiver_hrs_wk=pl_caregiver_hrs,
                caregiver_wage_frac=pl_caregiver_wage_frac,
                friction_period_days=90.0 if use_friction else 0.0)
            if (use_prod_loss and nmb_persp_key == "soc") else None),
    )

    with st.spinner("Computing NMB OWSA…"):
        owsa_nmb_df = owsa_nmb_table(
            base_kw_nmb, param_ranges, float(nmb_wtp), nmb_persp_key
        )

    base_nmb_val = float(owsa_nmb_df["Base NMB"].iloc[0])

    # KPI strip
    nb1, nb2, nb3 = st.columns(3)
    nb1.metric("Base-case NMB",
               f"${base_nmb_val/1e6:,.2f}M",
               help=f"NMB = ${nmb_wtp/1000:.0f}K × ΔDALYs − ΔCost at mean parameters")
    nb2.metric("Highest-impact parameter",
               owsa_nmb_df["Parameter"].iloc[0])
    nb3.metric("NMB range — top parameter",
               f"${owsa_nmb_df['Range'].iloc[0]/1e6:,.2f}M",
               help="Width of the widest tornado bar; parameters above this dominate "
                    "one-way decision uncertainty at this WTP")

    st.pyplot(
        fig_tornado_nmb(owsa_nmb_df, base_nmb_val,
                        float(nmb_wtp), nmb_persp_key),
        width="stretch",
    )
    buf_nmb = io.BytesIO()
    fig_tornado_nmb(owsa_nmb_df, base_nmb_val,
                    float(nmb_wtp), nmb_persp_key).savefig(
        buf_nmb, format="png", dpi=300, bbox_inches="tight"
    )
    buf_nmb.seek(0)
    st.download_button("Download NMB tornado", buf_nmb,
                       "tornado_nmb.png", "image/png")

    # Detail table
    st.subheader("NMB OWSA Detail Table")
    disp_nmb = owsa_nmb_df[[
        "Parameter", "Low param value", "High param value",
        "NMB (low param)", "NMB (high param)", "Range"
    ]].copy()
    for c in ["NMB (low param)", "NMB (high param)", "Range"]:
        disp_nmb[c] = disp_nmb[c].map(lambda x: f"${x/1e6:,.2f}M")
    st.dataframe(disp_nmb, hide_index=True, width="stretch")

    # Comparison callout — flag parameters where ICER and NMB rankings diverge
    icer_order = list(owsa_df["Parameter"])
    nmb_order  = list(owsa_nmb_df["Parameter"])
    top3_icer  = set(icer_order[:3])
    top3_nmb   = set(nmb_order[:3])
    diverged   = top3_icer.symmetric_difference(top3_nmb)
    if diverged:
        st.warning(
            f"**Ranking divergence detected.** "
            f"The following parameters appear in the top-3 of one method "
            f"but not the other: **{', '.join(diverged)}**. "
            "This is a signal that ICER ratio instability is materially affecting "
            "the ICER tornado — the NMB ranking should be preferred for decision-making."
        )
    else:
        st.success(
            "Top-3 parameters are consistent across ICER and NMB tornado rankings "
            "at this WTP — ratio instability is not materially distorting the ICER results."
        )

    # CSV export for NMB OWSA
    st.download_button(
        "⬇ Download NMB OWSA (CSV)",
        owsa_nmb_df.to_csv(index=False).encode(),
        "owsa_nmb.csv",
        "text/csv",
    )
    # Export
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_psa.to_excel(writer, sheet_name="PSA_iterations", index=False)
        owsa_df.to_excel(writer, sheet_name="OWSA", index=False)
        pd.DataFrame(smry).T.to_excel(writer, sheet_name="Summary")
    st.download_button("⬇ Download PSA + OWSA (Excel)", buf.getvalue(),
                       "syphilis_ed_cea_v4.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 · Societal Perspective
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Societal Perspective")
    st.info(
        "**Perspective: societal.**  "
        "Adds to health-sector analysis: (1) maternal morbidity DALYs and averted treatment costs, "
        "(2) productivity losses averted via human-capital method (BLS 2023 earnings). "
        "The VSL net-benefit analysis is presented separately below — it uses a "
        "**willingness-to-pay** framework and is not directly comparable to the DALY-based ICER."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig_ce_plane(df_psa["dal_soc"].values, df_psa["ic_soc"].values,
                               "CE Plane — Societal (DALY-based)"), width="stretch")
        fig = fig_ce_plane(df_psa["dal_soc"].values, df_psa["ic_soc"].values, "CE Plane — Societal (DALY-based)")
        buf = io.BytesIO()    
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        st.download_button(label="Download Figure", data=buf, file_name="cea_soc.png", mime="image/png")

    with c2:
        quad_soc = ce_quadrant_table(df_psa["dal_soc"].values, df_psa["ic_soc"].values)
        st.subheader("Quadrant Analysis — Societal")
        st.dataframe(quad_soc, width="stretch", hide_index=True)

    # Societal PSA summary
    st.subheader("PSA Summary — Societal")

    ds1, ds2, ds3 = st.columns(3)
    ds1.metric("DALYs averted — health sector", f"{smry['dalys_hs']['mean']:,.1f}")
    ds2.metric("DALYs averted — societal", f"{smry['dalys_soc']['mean']:,.1f}")
    ds3.metric(
        "Societal DALY increment",
        f"{smry['dalys_soc_increment']['mean']:,.1f}",
        help=(
            "Societal DALYs minus health-sector DALYs. This currently equals "
            "maternal morbidity DALYs averted. Productivity losses affect "
            "societal costs, not the DALY denominator."
        ),
    )
    if use_mat_morb and abs(smry['dalys_soc_increment']['mean']) < 1e-9:
        st.warning(
            "Maternal morbidity is enabled, but the societal DALY increment is zero. "
            "Check maternal morbidity probabilities, treatment reach, and p_act."
        )
    elif not use_mat_morb:
        st.info(
            "Maternal morbidity is disabled, so DALYs are expected to be identical "
            "between health-sector and societal perspectives. Societal differences "
            "will come from costs only if productivity losses are enabled."
        )

    soc_rows = []
    for label, key in [
        ("Incremental cost — societal ($)", "inc_cost_soc"),
        ("DALYs averted — societal",        "dalys_soc"),
        ("Societal DALY increment",         "dalys_soc_increment"),
        ("ICER — societal ($/DALY)",        "icer_soc"),
    ]:
        s = smry[key]
        fmt = lambda v: f"${v:,.0f}" if "cost" in key or "ICER" in key else f"{v:,.1f}"
        soc_rows.append({"Metric": label,
                          "Mean": fmt(s["mean"]), "Median": fmt(s["median"]),
                          "95% CrI lo": fmt(s["95% CrI lo"]),
                          "95% CrI hi": fmt(s["95% CrI hi"])})
    soc_rows.append({"Metric": "P(cost-saving — societal)", "Mean": f"{smry['p_cost_saving_soc']:.1%}",
                     "Median": "—", "95% CrI lo": "—", "95% CrI hi": "—"})
    soc_rows.append({"Metric": "P(dominant — societal)",    "Mean": f"{smry['p_dominant_soc']:.1%}",
                     "Median": "—", "95% CrI lo": "—", "95% CrI hi": "—"})
    st.dataframe(pd.DataFrame(soc_rows), width="stretch", hide_index=True)

    # Productivity loss detail
    st.subheader("Productivity Loss — Human Capital Method")
    pl_s = smry["prod_sav"]
    pl1, pl2, pl3 = st.columns(3)
    pl1.metric("Mean productivity savings", f"${pl_s['mean']:,.1f}")
    pl2.metric("95% CrI lo",               f"${pl_s['95% CrI lo']:,.1f}")
    pl3.metric("95% CrI hi",               f"${pl_s['95% CrI hi']:,.1f}")
    st.caption(
        f"Weighted maternal earnings: ${MATERNAL_WEIGHTED_EARNINGS:,.0f}/year  |  "
        f"Friction cost: {'90-day cap applied' if use_friction else 'Full human-capital method'}  |  "
        f"BLS 2023 median weekly earnings by age band."
    )
    if use_prod_loss:
        st.pyplot(fig_prod_loss_breakdown(df_psa), width="stretch")
        fig = fig_prod_loss_breakdown(df_psa)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        st.download_button(label="Download Figure", data=buf, file_name="fig_prod_loss.png", mime="image/png")
    else:
        st.warning("Productivity loss module is disabled. Enable it in the sidebar.")

    # Maternal morbidity detail
    st.subheader("Maternal Morbidity — DALYs and Costs Averted")
    mm_s = smry["dalys_mat_morb"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Maternal morbidity DALYs averted (mean)", f"{mm_s['mean']:,.1f}")
    m2.metric("95% CrI",  f"{mm_s['95% CrI lo']:,.1f} – {mm_s['95% CrI hi']:,.1f}")
    m3.metric("Maternal morbidity cost saving (mean)",
              f"${df_psa['mm_cost'].mean():,.0f}")
    if not use_mat_morb:
        st.warning("Maternal morbidity module disabled. Enable in the sidebar.")

    # VSL analysis — clearly separated
    st.divider()
    with st.expander("📌 VSL Net-Benefit Analysis (separate framework — not an ICER)", expanded=True):
        st.markdown("""
        > **Methodological note:** The VSL analysis uses a willingness-to-pay framework
        > (regulatory benefit-cost analysis) and is **not** directly comparable to the
        > DALY-based ICERs above. VSL represents the aggregate population willingness to pay
        > for a statistical life saved, derived from wage-risk studies. It should be interpreted
        > as: :red[*"Does the monetary value society places on lives saved exceed program costs?"*]
        > It is not a cost-per-DALY metric.
        """)
        vsl_s = smry["vsl_nb"]
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Mean VSL net benefit",     f"${vsl_s['mean']/1e6:,.2f}M")
        v2.metric("95% CrI lo",               f"${vsl_s['95% CrI lo']/1e6:,.2f}M")
        v3.metric("95% CrI hi",               f"${vsl_s['95% CrI hi']/1e6:,.2f}M")
        v4.metric("P(net benefit > 0)",        f"{(df_psa['vsl_nb'] > 0).mean():.1%}")
        st.caption(
            f"VSL assumed = ${vsl:,.0f} (DHHS ASPE 2023 guidance).  "
            f"Net benefit = VSL × (stillbirths + neonatal deaths prevented) − health-sector program cost."
        )
        fig_vsln, ax_vsln = plt.subplots(figsize=(7, 3.5))
        ax_vsln.hist(df_psa["vsl_nb"] / 1e6, bins=60, color="#c0392b", alpha=0.8, edgecolor="white")
        ax_vsln.axvline(0, color="k", lw=1.5, label="Break-even")
        ax_vsln.axvline(vsl_s["mean"]/1e6, color="darkred", lw=1.5, ls="--",
                        label=f"Mean = ${vsl_s['mean']/1e6:,.1f}M")
        ax_vsln.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"${x:,.1f}M"))
        ax_vsln.set_xlabel("VSL net benefit ($M)"); ax_vsln.set_ylabel("Frequency")
        ax_vsln.set_title("VSL Net Benefit Distribution (PSA)", fontweight="bold")
        ax_vsln.legend(fontsize=8); ax_vsln.grid(alpha=0.15)
        ax_vsln.tick_params(axis = "x", which="major", labelsize=7)
        ax_vsln.tick_params(axis = "y", which="major", labelsize=7)
        ax_vsln.spines[["top","right"]].set_visible(False)
        st.pyplot(fig_vsln, width="stretch")
        buf = io.BytesIO()
        
        fig_vsln.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        st.download_button(label="Download Figure", data=buf, file_name="vsln.png", mime="image/png")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 · Budget Impact
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Hospital / Payer Budget Impact Analysis")
    st.info(
        "**ISPOR BIA framework.** "
        "Perspective: payer (health sector). "
        "All figures are nominal — **no discounting applied** within the BIA horizon. "
        "Lifetime Markov sequelae costs are excluded (captured in the CEA); "
        "only within-horizon, within-coverage medical costs and savings are counted. "
        "Coverage follows a sigmoid (S-curve) implementation ramp. "
        "Three scenarios (Conservative / Base / Optimistic) are shown side-by-side."
    )

    # ── Instantiate objects from sidebar inputs ──────────────────────────────
    co_bia = Costs()   # point estimates — no PSA draws

    bia_pop_base = BIAPopulation(
        covered_lives     = bia_covered,
        frac_repro_female = bia_frac_rf,
        pregnancy_rate    = bia_preg_rate,
        p_ed_visit        = bia_p_ed,
        p_unscreened      = bia_p_unscr,
        payer_fraction    = bia_payer_frac,
    )

    # User-defined ramp becomes the Base case; Conservative and Optimistic
    # are offset by ±1 year on both milestone parameters.
    SCENARIO_PARAMS = {
        "Conservative": dict(
            pop     = BIAPopulation(
                covered_lives     = bia_covered,
                frac_repro_female = bia_frac_rf,
                pregnancy_rate    = bia_preg_rate,
                p_ed_visit        = max(bia_p_ed    - 0.10, 0.10),
                p_unscreened      = max(bia_p_unscr - 0.10, 0.05),
                payer_fraction    = max(bia_payer_frac - 0.05, 0.05),
            ),
            t_half   = min(bia_t_half   + 1.0, float(bia_n_yrs) - 0.5),
            t_ninety = min(bia_t_ninety + 1.0, float(bia_n_yrs)),
        ),
        "Base case": dict(
            pop      = bia_pop_base,
            t_half   = bia_t_half,
            t_ninety = bia_t_ninety,
        ),
        "Optimistic": dict(
            pop     = BIAPopulation(
                covered_lives     = bia_covered,
                frac_repro_female = bia_frac_rf,
                pregnancy_rate    = bia_preg_rate,
                p_ed_visit        = min(bia_p_ed    + 0.10, 0.70),
                p_unscreened      = min(bia_p_unscr + 0.10, 0.80),
                payer_fraction    = min(bia_payer_frac + 0.05, 1.00),
            ),
            t_half   = max(bia_t_half   - 0.5, 0.5),
            t_ninety = max(bia_t_ninety - 0.5, bia_t_half + 0.5),
        ),
    }

    # ── DIAGNOSTIC: full cascade trace ───────────────────────────────────────
    st.subheader("🔍 Population Cascade Diagnostic")
    st.caption("Trace every multiplier from covered lives → incremental screened → treated. "
               "Check each row against your prior expectations before trusting cost outputs.")

    # Step 1 — demographic funnel
    n_repro    = bia_covered * bia_frac_rf
    n_pregnant = n_repro * bia_preg_rate
    n_ed_all   = n_pregnant * bia_p_ed
    n_ed_payer = n_ed_all * bia_payer_frac
    n_eligible = n_ed_payer * bia_p_unscr

    # Step 2 — coverage in year 1 and at maturity (year n)
    cov_yr1 = sigmoid_coverage_at_t(1, bia_t_half, bia_t_ninety, sc_uc_eff, sc_e)
    cov_mat  = sigmoid_coverage_at_t(bia_n_yrs, bia_t_half, bia_t_ninety, sc_uc_eff, sc_e)

    # Step 3 — incremental screened
    n_intr_yr1 = n_eligible * cov_yr1  * p_id
    n_uc_yr1   = n_eligible * sc_uc_eff * p_id
    n_incr_yr1 = max(n_intr_yr1 - n_uc_yr1, 0.0)

    n_intr_mat = n_eligible * cov_mat  * p_id
    n_incr_mat = max(n_intr_mat - n_uc_yr1, 0.0)

    # Step 4 — syphilis cascade on incremental screened (year 1)
    n_tp_det_yr1  = n_incr_yr1 * p_act * sens
    n_tp_tx_yr1   = n_tp_det_yr1 * p_adeq * tx_eff
    n_sf_det_yr1  = n_incr_yr1 * p_sf  * p_trepo_sf
    n_fp_det_yr1  = n_incr_yr1 * max(1 - p_act - p_sf, 0) * (1 - spec)

    # Step 5 — outcomes averted from treated TPs (year 1)
    n_cs_av_yr1  = n_tp_tx_yr1 * UNT_ABS["cs_any"]         * (1 - TX_RR["cs_any"]["rr"])
    n_sb_av_yr1  = n_tp_tx_yr1 * UNT_ABS["stillbirth"]     * (1 - TX_RR["stillbirth"]["rr"])
    n_nnd_av_yr1 = n_tp_tx_yr1 * UNT_ABS["neonatal_death"]  * (1 - TX_RR["neonatal_death"]["rr"])

    # Step 6 — quick cost sanity check (year 1, point estimates)
    co_d = Costs()
    cost_tests_yr1 = n_incr_yr1 * (co_d.poc + co_d.rpr)
    cost_conf_yr1  = (n_tp_det_yr1 + n_sf_det_yr1 + n_fp_det_yr1) * co_d.fta
    cost_tx_yr1    = n_tp_tx_yr1 * (co_d.pen + co_d.soc_work)
    cost_staff_yr1 = n_incr_yr1 * co_d.staff
    cost_total_yr1 = cost_tests_yr1 + cost_conf_yr1 + cost_tx_yr1 + cost_staff_yr1

    diag_rows = [
        # ── Demographic funnel ──────────────────────────────────────────────
        ("━━ DEMOGRAPHIC FUNNEL",                   "",          "",                    ""),
        ("Covered lives",                           f"{bia_covered:,.0f}",  "Input",   ""),
        ("× Repro-age female fraction",             f"{bia_frac_rf:.3f}",  "Input",    ""),
        ("= Reproductive-age females",              f"{n_repro:,.0f}",     "Derived",
             f"~{n_repro/bia_covered:.1%} of covered lives"),
        ("× Annual pregnancy rate",                 f"{bia_preg_rate:.3f}", "Input",   ""),
        ("= Annual pregnancies",                    f"{n_pregnant:,.0f}",  "Derived",
             f"~{n_pregnant/bia_covered:.2%} of covered lives"),
        ("× P(ED visit | pregnant)",                f"{bia_p_ed:.2f}",     "Input",   ""),
        ("= Pregnant ED visits (all payers)",       f"{n_ed_all:,.0f}",    "Derived", ""),
        ("× Payer fraction of ED volume",           f"{bia_payer_frac:.2f}","Input",  ""),
        ("= Payer-covered pregnant ED visits",      f"{n_ed_payer:,.0f}",  "Derived", ""),
        ("× P(unscreened antenatally | ED)",        f"{bia_p_unscr:.2f}",  "Input",   ""),
        ("= Eligible (unscreened) patients/yr",     f"{n_eligible:,.0f}",  "Derived",
             "← this is your annual denominator"),

        # ── Coverage ramp ───────────────────────────────────────────────────
        ("━━ COVERAGE RAMP",                        "",          "",                    ""),
        ("Usual-care coverage (sc_uc, strata-wtd)", f"{sc_uc_eff:.3f}",    "Derived", ""),
        ("Target coverage (sc_e)",                  f"{sc_e:.3f}",         "Input",   ""),
        ("Effective coverage — Year 1",             f"{cov_yr1:.3f}",      "Sigmoid", ""),
        (f"Effective coverage — Year {bia_n_yrs}",  f"{cov_mat:.3f}",      "Sigmoid", ""),

        # ── Incremental screened ────────────────────────────────────────────
        ("━━ INCREMENTAL SCREENED (Δ above usual care)",  "",     "",                  ""),
        ("Reached by usual care / yr (× p_id)",    f"{n_uc_yr1:,.1f}",    "Derived", ""),
        ("Reached by intervention yr 1 (× p_id)",  f"{n_intr_yr1:,.1f}",  "Derived", ""),
        ("Incremental screened — Year 1",           f"{n_incr_yr1:,.1f}",  "Derived",
             "← program's operative N in year 1"),
        (f"Incremental screened — Year {bia_n_yrs}",f"{n_incr_mat:,.1f}", "Derived", ""),

        # ── Syphilis cascade (year 1) ───────────────────────────────────────
        ("━━ SYPHILIS CASCADE — YEAR 1 (on incremental screened)", "", "",             ""),
        ("Active syphilis prevalence (p_act)",      f"{p_act:.4f}",        "Input",   ""),
        ("True positives detected (× sens)",        f"{n_tp_det_yr1:.2f}", "Derived", ""),
        ("True positives treated (× p_adeq × tx_eff)", f"{n_tp_tx_yr1:.2f}","Derived",""),
        ("Serofast detected",                       f"{n_sf_det_yr1:.2f}", "Derived", ""),
        ("False positives detected",                f"{n_fp_det_yr1:.2f}", "Derived", ""),

        # ── Outcomes averted (year 1) ───────────────────────────────────────
        ("━━ OUTCOMES AVERTED — YEAR 1",            "",          "",                    ""),
        ("CS cases averted",                        f"{n_cs_av_yr1:.3f}",  "Derived", ""),
        ("Stillbirths averted",                     f"{n_sb_av_yr1:.3f}",  "Derived", ""),
        ("Neonatal deaths averted",                 f"{n_nnd_av_yr1:.3f}", "Derived", ""),

        # ── Cost sanity check (year 1) ──────────────────────────────────────
        ("━━ COST SANITY CHECK — YEAR 1 (point estimates, no discount)", "", "",       ""),
        ("Screening tests (POC + RPR)",             f"${cost_tests_yr1:,.0f}", "Derived",""),
        ("Confirmatory tests (FTA)",                f"${cost_conf_yr1:,.0f}",  "Derived",""),
        ("Treatment (pen + soc work)",              f"${cost_tx_yr1:,.0f}",    "Derived",""),
        ("Staff / workflow",                        f"${cost_staff_yr1:,.0f}", "Derived",""),
        ("Total program cost — Year 1",             f"${cost_total_yr1:,.0f}", "Derived",
             "← does this match the BIA table?"),
    ]

    diag_df = pd.DataFrame(diag_rows, columns=["Step", "Value", "Source", "Note"])
    st.dataframe(diag_df, hide_index=True, width="stretch")

    # Flag implausibly small cascades
    if n_incr_yr1 < 1:
        st.error(
            f"⚠️ Incremental screened in year 1 = **{n_incr_yr1:.2f}** — less than 1 patient. "
            "Check covered lives, payer fraction, p_unscreened, and coverage gap (sc_e − sc_uc)."
        )
    if n_tp_tx_yr1 < 0.01:
        st.warning(
            f"⚠️ Treated true positives in year 1 = **{n_tp_tx_yr1:.4f}**. "
            "With this few treated cases, outcome savings will be negligible and "
            "the cost-effectiveness conclusion is driven entirely by program costs."
        )

    # ── 1. Ramp preview ──────────────────────────────────────────────────────
    st.subheader("Implementation Ramp Preview")

    fig_ramp, ax_ramp = plt.subplots(figsize=(8, 3.5))
    RAMP_COLORS = {
        "Conservative": "#e76f51",
        "Base case":    "steelblue",
        "Optimistic":   "#2a9d8f",
    }
    t_fine = np.linspace(0, bia_n_yrs, 300)

    for label, sp in SCENARIO_PARAMS.items():
        y = np.array([
            sigmoid_coverage_at_t(t, sp["t_half"], sp["t_ninety"], sc_uc_eff, sc_e)
            for t in t_fine
        ])
        ax_ramp.plot(t_fine, y * 100, color=RAMP_COLORS[label],
                     lw=2, label=label)
        # Year markers
        ramp_pts = sigmoid_ramp(
            sp["t_half"], sp["t_ninety"], bia_n_yrs, sc_uc_eff, sc_e)
        ax_ramp.scatter(
            list(ramp_pts.keys()),
            [v * 100 for v in ramp_pts.values()],
            color=RAMP_COLORS[label], s=35, zorder=5)

    ax_ramp.axhline(sc_e * 100, color="k", ls="--", lw=0.8,
                    label=f"Target coverage ({sc_e:.0%})")
    ax_ramp.set_xlim(0, bia_n_yrs + 0.3)
    ax_ramp.set_ylim(0, 105)
    ax_ramp.set_xlabel("Year of implementation")
    ax_ramp.set_ylabel("Effective coverage (%)")
    ax_ramp.set_title("S-curve implementation ramp — all scenarios",
                      fontweight="bold")
    ax_ramp.legend(fontsize=8, framealpha=0.7)
    ax_ramp.grid(alpha=0.15)
    ax_ramp.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig_ramp, width="stretch")

    buf = io.BytesIO()
    fig_ramp.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button("Download ramp chart", buf,
                       "bia_ramp.png", "image/png")

    # Ramp table — base case only for concision
    ramp_base = sigmoid_ramp(bia_t_half, bia_t_ninety, bia_n_yrs, sc_uc_eff, sc_e)
    ramp_tbl  = pd.DataFrame([
        {"Year": yr,
         "Effective coverage (base)": f"{cov:.1%}",
         "% of target": f"{(cov - sc_uc_eff) / max(sc_e - sc_uc_eff, 1e-9):.0%}"}
        for yr, cov in ramp_base.items()
    ])
    st.dataframe(ramp_tbl, hide_index=True, width="stretch")

    st.divider()

    # ── 2. Run all scenarios ─────────────────────────────────────────────────
    scenario_results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    with st.spinner("Computing BIA scenarios…"):
        for label, sp in SCENARIO_PARAMS.items():
            df_imp, df_fun = run_bia_scenario(
                pop        = sp["pop"],
                t_half     = sp["t_half"],
                t_ninety   = sp["t_ninety"],
                n_years    = bia_n_yrs,
                co         = co_bia,
                sc_e       = sc_e,
                sc_uc      = sc_uc_eff,
                p_act      = p_act,
                p_id       = p_id,
                sens       = sens,
                spec       = spec,
                p_adeq     = p_adeq,
                tx_eff     = tx_eff,
                prop_symp  = prop_symp,
                prop_late  = prop_late,
                p_sf       = p_sf,
                p_trepo_sf = p_trepo_sf,
                p_ux_sf    = p_ux_sf,
                treat_fp   = treat_fp,
            )
            scenario_results[label] = (df_imp, df_fun)

    # ── 3. KPI strip — base case ─────────────────────────────────────────────
    st.subheader("Key Metrics — Base Case")
    df_base_imp, df_base_fun = scenario_results["Base case"]

    cea_full_tp_treated = (
        int(cohort)
        * max(float(sc_e) - float(sc_uc_eff), 0.0)
        * float(p_id)
        * float(p_act)
        * float(sens)
        * float(p_adeq)
        * float(tx_eff)
    )

    df_cea_compare = df_base_imp[[
        "year",
        "n_incremental",
        "n_tp_detected",
        "n_tp_treated",
        "n_sf_detected",
        "n_sf_treated",
        "program_cost",
        "medical_savings",
        "net_impact",
    ]].copy()

    
    cum_net    = df_base_imp["cumulative_net"].iloc[-1]
    cum_prog   = df_base_imp["program_cost"].sum()
    cum_sav    = df_base_imp["medical_savings"].sum()
    cum_cs     = df_base_imp["n_cs_averted"].sum()
    cum_sb     = df_base_imp["n_sb_averted"].sum()
    cum_nnd    = df_base_imp["n_nnd_averted"].sum()
    pmpm       = cum_net / (bia_covered * 12 * bia_n_yrs)
    
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(f"{bia_n_yrs}-yr program cost",    f"${cum_prog/1e6:,.2f}M")
    k2.metric(f"{bia_n_yrs}-yr medical savings",  f"${cum_sav/1e6:,.2f}M")
    k3.metric(f"{bia_n_yrs}-yr net impact",
              f"${cum_net/1e6:,.2f}M",
              delta="Cost-saving ✓" if cum_net < 0 else "Net cost",
              delta_color="normal" if cum_net < 0 else "inverse")
    k4.metric("PMPM (over horizon)",
              f"${pmpm:,.4f}",
              help="Net budget impact per member per month over the full horizon.")
    k5.metric("CS cases averted (cumulative)",  f"{cum_cs:.1f}")
    k6.metric("Stillbirths averted (cumulative)", f"{cum_sb:.1f}")

    st.divider()

    # ── 4. Scenario comparison ───────────────────────────────────────────────
    st.subheader("Scenario Comparison")

    # Summary table
    scen_rows = []
    for label, (df_imp, _) in scenario_results.items():
        cn    = df_imp["cumulative_net"].iloc[-1]
        cp    = df_imp["program_cost"].sum()
        cs    = df_imp["medical_savings"].sum()
        pmpm_ = cn / (bia_covered * 12 * bia_n_yrs)
        scen_rows.append({
            "Scenario":                       label,
            f"{bia_n_yrs}-yr program cost":   f"${cp/1e6:,.2f}M",
            f"{bia_n_yrs}-yr savings":        f"${cs/1e6:,.2f}M",
            f"{bia_n_yrs}-yr net impact":     f"${cn/1e6:,.2f}M",
            "PMPM":                           f"${pmpm_:,.4f}",
            "CS averted":                     f"{df_imp['n_cs_averted'].sum():.1f}",
            "Stillbirths averted":            f"{df_imp['n_sb_averted'].sum():.1f}",
            "NNDs averted":                   f"{df_imp['n_nnd_averted'].sum():.1f}",
        })
    st.dataframe(pd.DataFrame(scen_rows), hide_index=True,
                 width="stretch")

    # Annual net impact chart — all scenarios
    fig_scen, ax_scen = plt.subplots(figsize=(8, 4))
    for label, (df_imp, _) in scenario_results.items():
        ax_scen.plot(df_imp["year"], df_imp["cumulative_net"] / 1e6,
                     "o-", color=RAMP_COLORS[label], lw=2, label=label)
    ax_scen.axhline(0, color="k", lw=0.8)
    ax_scen.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"${x:,.1f}M"))
    ax_scen.set_xlabel("Year")
    ax_scen.set_ylabel("Cumulative net impact ($M)")
    ax_scen.set_title("Cumulative Net Budget Impact — All Scenarios",
                      fontweight="bold")
    ax_scen.legend(fontsize=9, framealpha=0.7)
    ax_scen.grid(alpha=0.15)
    ax_scen.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig_scen, width="stretch")

    buf = io.BytesIO()
    fig_scen.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button("Download scenario chart", buf,
                       "bia_scenarios.png", "image/png")

    st.divider()

    # ── 5. Base-case detail tables ───────────────────────────────────────────
    st.subheader("Base Case — Annual Detail")

    tab_imp, tab_fun, tab_cost = st.tabs([
        "Budget impact", "Population funnel", "Cost breakdown"
    ])

    with tab_imp:
        disp_imp = pd.DataFrame({
            "Year":                     df_base_imp["year"],
            "Coverage (ramp)":          df_base_imp["eff_coverage"].map(lambda x: f"{x:.1%}"),
            "Incremental screened":     df_base_imp["n_incremental"].map(lambda x: f"{x:,.0f}"),

            "Active TP detected":       df_base_imp["n_tp_detected"].map(lambda x: f"{x:.1f}"),
            "Active TP treated":        df_base_imp["n_tp_treated"].map(lambda x: f"{x:.1f}"),

            "Serofast detected":        df_base_imp["n_sf_detected"].map(lambda x: f"{x:.1f}"),
            "Serofast treated":         df_base_imp["n_sf_treated"].map(lambda x: f"{x:.1f}"),

            "False positives":          df_base_imp["n_fp_detected"].map(lambda x: f"{x:.1f}"),
            "FP treated":               df_base_imp["n_fp_treated"].map(lambda x: f"{x:.1f}"),

            "Program cost ($)":         df_base_imp["program_cost"].map(lambda x: f"${x:,.0f}"),
            "Medical savings ($)":      df_base_imp["medical_savings"].map(lambda x: f"${x:,.0f}"),
            "Net impact ($)":           df_base_imp["net_impact"].map(lambda x: f"${x:,.0f}"),
            "Cumulative net ($)":       df_base_imp["cumulative_net"].map(lambda x: f"${x:,.0f}"),
        })
        st.dataframe(disp_imp, hide_index=True, width="stretch")

        # NNS
        total_screened  = df_base_imp["n_incremental"].sum()
        total_cs_averted = df_base_imp["n_cs_averted"].sum()
        nns = total_screened / max(total_cs_averted, 1e-6)
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Total incrementally screened ({bia_n_yrs} yr) | **{total_screened:,.0f}** |
        | CS cases averted ({bia_n_yrs} yr) | **{total_cs_averted:.1f}** |
        | Number needed to screen (NNS) to prevent 1 CS case | **{nns:,.0f}** |
        | PMPM over {bia_n_yrs}-year horizon | **${pmpm:,.4f}** |
        """)

    with tab_fun:
        st.caption(
            "Transparent population flow — each row auditable against "
            "published data sources. Payer fraction applied at the ED "
            "visit stage."
        )
        disp_fun = pd.DataFrame({
            "Year":                   df_base_fun["year"],
            "Repro-age females":      df_base_fun["n_repro"].map(
                                          lambda x: f"{x:,.0f}"),
            "Pregnancies/yr":         df_base_fun["n_pregnant"].map(
                                          lambda x: f"{x:,.0f}"),
            "ED visits (payer)":      df_base_fun["n_ed"].map(
                                          lambda x: f"{x:,.0f}"),
            "Unscreened antenatally": df_base_fun["n_unscreened"].map(
                                          lambda x: f"{x:,.0f}"),
            "Reached — usual care":   df_base_fun["n_uc"].map(
                                          lambda x: f"{x:,.0f}"),
            "Reached — intervention": df_base_fun["n_intr"].map(
                                          lambda x: f"{x:,.0f}"),
            "Incremental (Δ)":        df_base_fun["n_incremental"].map(
                                          lambda x: f"{x:,.0f}"),
        })
        st.dataframe(disp_fun, hide_index=True, width="stretch")

        st.caption(
            f"Population parameters: covered lives = {bia_covered:,} | "
            f"reproductive-age female fraction = {bia_frac_rf:.1%} | "
            f"annual pregnancy rate = {bia_preg_rate:.1%} | "
            f"P(ED visit | pregnant) = {bia_p_ed:.0%} | "
            f"P(unscreened antenatally | ED) = {bia_p_unscr:.0%} | "
            f"payer fraction = {bia_payer_frac:.0%}"
        )

    with tab_cost:
        st.caption(
            "Cost components use point estimates from the Costs dataclass "
            "(CPI-adjusted 2019→2025). No PSA draws; no discounting."
        )
        disp_cost = pd.DataFrame({
            "Year":                     df_base_imp["year"],
            "Screening tests ($)":      df_base_imp["cost_screening_tests"].map(lambda x: f"${x:,.0f}"),
            "Confirmatory tests ($)":   df_base_imp["cost_confirmatory"].map(lambda x: f"${x:,.0f}"),
            "Staff ($)":                df_base_imp["cost_staff"].map(lambda x: f"${x:,.0f}"),

            "Tx — active TP ($)":       df_base_imp["cost_tx_tp"].map(lambda x: f"${x:,.0f}"),
            "Tx — serofast ($)":        df_base_imp["cost_tx_sf"].map(lambda x: f"${x:,.0f}"),
            "Tx — false positive ($)":  df_base_imp["cost_tx_fp"].map(lambda x: f"${x:,.0f}"),
            "Serofast workup ($)":      df_base_imp["cost_sf_workup"].map(lambda x: f"${x:,.0f}"),

            "Total program ($)":        df_base_imp["program_cost"].map(lambda x: f"${x:,.0f}"),

            "CS savings ($)":           df_base_imp["sav_cs"].map(lambda x: f"${x:,.0f}"),
            "Preterm savings ($)":      df_base_imp["sav_preterm"].map(lambda x: f"${x:,.0f}"),
            "SB/IUFD savings ($)":      df_base_imp["sav_sb"].map(lambda x: f"${x:,.0f}"),
            "NND savings ($)":          df_base_imp["sav_nnd"].map(lambda x: f"${x:,.0f}"),
        })
        st.dataframe(disp_cost, hide_index=True, width="stretch")

        # Exclusions note — important for ISPOR compliance transparency
        with st.expander("ℹ️ Items excluded from BIA savings (and why)", expanded=False):
            st.markdown("""
            The following cost offsets are captured in the **CEA** but
            deliberately **excluded from the BIA** per ISPOR guidelines:

            | Item | Reason for exclusion |
            |------|----------------------|
            | Lifetime Markov CS sequelae costs (`mk_cst`) | Accrue over decades; outside the BIA horizon and outside the payer's book of business as the child ages out of coverage |
            | Low-birth-weight long-term costs (`lbw_hs`) | Extend into early childhood; cannot reliably be attributed within a 5-year payer window |
            | Term delivery cost (`term_del`) | Background cost incurred regardless of intervention; not an incremental saving |
            | Maternal morbidity costs | Accrue over 5–8 years after infection; mostly outside the perinatal episode captured in this BIA window |
            | Productivity losses | Societal perspective only; not a payer budget item |

            These exclusions make the BIA **conservative** — it understates
            long-run financial benefit to the health system. The CEA provides
            the full lifetime perspective.
            """)

    st.divider()

    # ── 6. Budget bar chart — base case ─────────────────────────────────────
    st.subheader("Annual Program Cost vs. Medical Savings — Base Case")

    fig_bars, ax_bars = plt.subplots(figsize=(8, 4))
    yrs = df_base_imp["year"].values
    ax_bars.bar(yrs - 0.2, df_base_imp["program_cost"]  / 1e6, 0.4,
                label="Program cost",    color="#4a90d9", alpha=0.85)
    ax_bars.bar(yrs + 0.2, df_base_imp["medical_savings"] / 1e6, 0.4,
                label="Medical savings", color="#2a9d8f", alpha=0.85)
    ax_bars.plot(yrs, df_base_imp["cumulative_net"] / 1e6,
                 "k--o", ms=5, lw=1.5, label="Cumulative net impact")
    ax_bars.axhline(0, color="k", lw=0.6)
    ax_bars.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"${x:,.1f}M"))
    ax_bars.set_xlabel("Year")
    ax_bars.set_title("Annual Budget Impact — Base Case",
                      fontweight="bold")
    ax_bars.legend(fontsize=8)
    ax_bars.grid(alpha=0.15)
    ax_bars.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig_bars, width="stretch")

    buf = io.BytesIO()
    fig_bars.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button("Download budget bar chart", buf,
                       "bia_bars.png", "image/png")

    # ── 7. Excel export ──────────────────────────────────────────────────────
    buf_xl = io.BytesIO()
    with pd.ExcelWriter(buf_xl, engine="openpyxl") as writer:
        for label, (df_imp, df_fun) in scenario_results.items():
            sname = label.replace(" ", "_")[:28]
            df_imp.to_excel(writer, sheet_name=f"{sname}_impact", index=False)
            df_fun.to_excel(writer, sheet_name=f"{sname}_funnel", index=False)
        # Parameter summary
        param_summary = pd.DataFrame([
            {"Parameter": "Covered lives",            "Value": bia_covered},
            {"Parameter": "Repro-age female fraction","Value": bia_frac_rf},
            {"Parameter": "Annual pregnancy rate",    "Value": bia_preg_rate},
            {"Parameter": "P(ED visit | pregnant)",   "Value": bia_p_ed},
            {"Parameter": "P(unscreened | ED)",       "Value": bia_p_unscr},
            {"Parameter": "Payer fraction",           "Value": bia_payer_frac},
            {"Parameter": "Target coverage (sc_e)",   "Value": sc_e},
            {"Parameter": "Usual-care coverage",      "Value": sc_uc_eff},
            {"Parameter": "Syphilis prevalence",      "Value": p_act},
            {"Parameter": "Projection horizon (yr)",  "Value": bia_n_yrs},
            {"Parameter": "t_half (base case)",       "Value": bia_t_half},
            {"Parameter": "t_ninety (base case)",     "Value": bia_t_ninety},
            {"Parameter": "Discount rate (BIA)",      "Value": "None (ISPOR)"},
        ])
        param_summary.to_excel(writer, sheet_name="Parameters", index=False)

    st.download_button(
        "⬇ Download full BIA (Excel — all scenarios)",
        buf_xl.getvalue(),
        "syphilis_ed_bia_v4.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 · Threshold Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Threshold Analysis")
    st.info("Deterministic, distribution means. Societal perspective available via toggle.")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        wtp_thresh = st.selectbox("WTP for NMB surface",
                                  [50_000, 100_000, 150_000, 200_000], index=1,
                                  format_func=lambda x: f"${x/1000:.0f}K/DALY")
    with tc2:
        prev_max = st.slider("Max prevalence on grid (%)", 1.0, 6.0, 4.0, 0.5)
    with tc3:
        thresh_societal = st.checkbox("Use societal perspective for NMB surface", value=False)

    mm_arg = MaternalMorbidity(p_cardio=mm_p_cardio, p_neuro=mm_p_neuro, p_hosp=mm_p_hosp,
                                cost_cardio=mm_cost_cardio, cost_neuro=mm_cost_neuro,
                                cost_hosp=mm_cost_hosp) if (use_mat_morb and thresh_societal) else None
    pl_arg = ProductivityLoss(bereavement_days=pl_bereavement_days,
                               wage_penalty_mild=pl_wage_mild, wage_penalty_severe=pl_wage_severe,
                               caregiver_hrs_wk=pl_caregiver_hrs,
                               caregiver_wage_frac=pl_caregiver_wage_frac,
                               friction_period_days=90.0 if use_friction else 0.0
                               ) if (use_prod_loss and thresh_societal) else None

    with st.spinner("Computing NMB surface…"):
        prev_g = np.arange(0.001, prev_max / 100 + 0.001, 0.001)
        tx_g   = np.arange(0.40, 1.01, 0.04)
        G = nmb_surface(prev_g, tx_g, p_sf, p_id, sc_uc_eff, sc_e, sens, spec,
                        prop_symp, prop_late, p_trepo_sf, p_ux_sf,
                        r_disc, LE, inc_lbw, inc_mat, int(cohort), int(wtp_thresh),
                        mm=mm_arg, pl=pl_arg, societal=thresh_societal,
                        inc_sb_yll=inc_sb_yll, inc_cs_yll=inc_cs_yll,
                        inc_misc_yld=inc_misc_yld, inc_mat_hosp_yld=inc_mat_hosp_yld,
                        inc_preterm_yld=inc_preterm_yld)
    st.pyplot(fig_nmb_surface(prev_g, tx_g, G, int(wtp_thresh)), width="stretch")
    fig = fig_nmb_surface(prev_g, tx_g, G, int(wtp_thresh))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(label="Download Figure", data=buf, file_name="nmb_surface.png", mime="image/png")
    st.subheader("ICER vs Prevalence")
    icer_vs_prev = []
    for pv_ in prev_g[::3]:
        ic_, dal_, ic_s, dal_s = _det_icost(pv_, p_sf, p_id, sc_uc_eff, sc_e, sens, spec,
                                             p_adeq, prop_symp, prop_late, p_trepo_sf, p_ux_sf,
                                             r_disc, LE, inc_lbw, inc_mat, int(cohort),
                                             mm_arg, pl_arg, None,
                                             inc_sb_yll=inc_sb_yll, inc_cs_yll=inc_cs_yll,
                                             inc_misc_yld=inc_misc_yld, inc_mat_hosp_yld=inc_mat_hosp_yld,
                                             inc_preterm_yld=inc_preterm_yld)
        icer_vs_prev.append(ic_ / max(dal_, 1e-6) if not thresh_societal
                            else ic_s / max(dal_s, 1e-6))

    fig_ip, ax_ip = plt.subplots(figsize=(7, 4))
    ax_ip.plot(prev_g[::3] * 100, icer_vs_prev, color="steelblue", lw=2)
    for th, col in [(50_000,"#2a9d8f"),(100_000,"#e9c46a"),(150_000,"#e76f51")]:
        ax_ip.axhline(th, ls="--", lw=1, color=col, label=f"${th/1000:.0f}K/DALY")
    ax_ip.axhline(0, color="k", lw=0.8)
    ax_ip.yaxis.set_major_formatter(ticker.FuncFormatter(dollar_fmt))
    ax_ip.set_xlabel("Active syphilis prevalence (%)")
    ax_ip.set_ylabel("ICER ($/DALY)")
    ax_ip.set_title("ICER vs Prevalence", fontweight="bold")
    ax_ip.legend(fontsize=8); ax_ip.grid(alpha=0.15)
    ax_ip.spines[["top","right"]].set_visible(False)
    st.pyplot(fig_ip, width="stretch")

    buf = io.BytesIO()
    fig_ip.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(label="Download Figure", data=buf, file_name="icer_prev.png", mime="image/png")

    prev_pct = prev_g[::3] * 100
    crossings = [i for i in range(1, len(icer_vs_prev))
                 if icer_vs_prev[i-1] > 100_000 and icer_vs_prev[i] <= 100_000]
    if crossings:
        be = prev_pct[crossings[0]]
        st.success(f"**Break-even at $100K/DALY: ≈ {be:.1f}% prevalence** "
                   f"(treatment rate {p_adeq:.0%}, P(ID) {p_id:.0%})")
    elif icer_vs_prev[0] < 100_000:
        st.success("**Cost-effective at $100K/DALY across all modelled prevalence values.**")
    else:
        st.warning("**Does not cross $100K/DALY threshold in this prevalence range.**")

    with st.expander("Interpretation of threshold analysis results", expanded=False):
        st.markdown("""
        ## Interpreting the Threshold Analysis Section

        The threshold analysis tab answers a core question: **under what conditions does ED syphilis screening become cost-effective?** It has two main components.

        ---

        ### 1. The NMB Surface (Contour Plot)

        This is a two-dimensional grid where:

        - **X-axis** = active syphilis prevalence (how common active syphilis is in your ED population)
        - **Y-axis** = same-day treatment rate (the `p_adeq` parameter — how often detected cases actually get treated)
        - **Color** = Net Monetary Benefit at your chosen WTP threshold

        The **black contour line** is the most important feature — it's the break-even boundary where NMB = 0. Everything to the **right/above** that line (green zone) is cost-effective at your chosen WTP; everything to the **left/below** (red zone) is not. You're essentially asking: "given your ED's prevalence and operational realities, where do you fall?"

        The WTP selector (50K, 100K, 150K, 200K per DALY) shifts where that black line sits — a higher WTP threshold makes the intervention cost-effective across a wider range of conditions.

        ---

        ### 2. The ICER vs. Prevalence Curve

        This takes a slice through that surface at your current treatment rate and shows how the ICER changes as prevalence varies. A few things to look for:

        - **Where the curve crosses the dashed WTP threshold lines** — this gives you the break-even prevalence. The model will flag this explicitly (e.g., "break-even at ~0.8% prevalence")
        - **The slope of the curve** — a steep downward slope means the program becomes dramatically more cost-effective as prevalence increases, which matters for targeting decisions (high-burden urban EDs vs. rural low-prevalence settings)
        - **Whether the curve goes negative** — a negative ICER means the program is cost-saving, not just cost-effective; this typically happens at higher prevalence values where outcome savings exceed program costs

        ---

        ### Practical Interpretation Example

        Say your base case is 0.75% prevalence and the break-even is at 0.5%. That tells you:

        1. You have a **safety margin** — prevalence would need to fall by about a third before the program stops being cost-effective at $100K/DALY
        2. The program is **robust to prevalence uncertainty**, which matters given that real-world ED syphilis prevalence estimates carry meaningful uncertainty

        If instead you're right at the break-even line, that signals the decision is sensitive to prevalence assumptions and you'd want to look closely at the PSA results and EVPPI to understand where the uncertainty is concentrated.

        ---

        ### The Societal Toggle

        Checking "use societal perspective" shifts the NMB surface upward (more favorable) because it adds productivity loss savings and maternal morbidity cost offsets to the benefit side. If the program only becomes cost-effective under the societal toggle, that's an important framing point — the payer-perspective case is marginal, but the broader social case is stronger.

        """)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 · Infant Markov
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Infant Lifetime Markov Module")
    st.markdown("""
    Four states: **Healthy** · **Mild sequelae** · **Severe sequelae** · **Dead**
    Background mortality: age-specific (US Life Table 2021).
    Annual mild→severe transition rate calibrated to user-specified lifetime target.
    """)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P(severe | CS comp)",  f"{p_sev_ui:.0%}")
    m2.metric("P(mild | CS comp)",    f"{p_mc_ui:.0%}")
    m3.metric("P(healthy | CS comp)", f"{max(1-p_sev_ui-p_mc_ui,0):.0%}")
    m4.metric("P(mild | CS uncomp)",  f"{p_mu_ui:.0%}")

    # q_progress calibration report
    with st.expander("q_progress Calibration Details", expanded=True):
        cal_ok = abs(implied_prog - q_target) < 0.01
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Target lifetime mild→severe progression | **{q_target:.0%}** |
        | Calibrated annual q_progress | **{q_cal:.5f}** |
        | Implied lifetime progression (with life table mortality) | **{implied_prog:.2%}** |
        | Discrepancy | **{abs(implied_prog - q_target)*100:.2f} pp** |
        | Status | {"✅ Within 1 pp of target" if cal_ok else "⚠️ Discrepancy >1 pp — check target feasibility"} |

        **Note:** The former fixed comment "~20% lifetime" in v3 reflected a target of 20%,
        but the actual implied rate under the life table at the default q=0.002 was ~{implied_lifetime_prog(0.002, max(int(LE),1)):.1%}.
        The calibration function now solves for q numerically so the implied rate exactly matches your target.
        """)

    mk_means_ui = {"p_severe_cs_comp": p_sev_ui, "p_mild_cs_comp": p_mc_ui,
                   "p_mild_cs_uncomp": p_mu_ui,
                   "mu_excess_mild": mu_x_mild_ui, "mu_excess_severe": mu_x_sev_ui}
    st.pyplot(fig_markov_states(float(r_disc), float(LE), mk_means_ui), width="stretch")

    st.subheader("PSA Distributions: Lifetime DALYs and Cost Savings")
    st.pyplot(fig_markov_daly_dist(df_psa), width="stretch")

    tot_hs = smry["dalys_hs"]["mean"]
    mk_mean = smry["dalys_markov"]["mean"]
    ncs_mean = smry["dalys_non_cs"]["mean"]
    st.markdown(f"""
    | DALY component | Mean |
    |----------------|------|
    | Non-CS infant + maternal grief | **{ncs_mean:,.1f}** |
    | CS lifetime YLD (Markov) | **{smry['dalys_markov_yld']['mean']:,.1f}** |
    | CS post-neonatal excess-mortality YLL (Markov) | **{smry['dalys_markov_yll']['mean']:,.1f}** |
    | CS lifetime total (Markov) | **{mk_mean:,.1f}** |
    | **Total (health sector)** | **{tot_hs:,.1f}** |
    | Markov share | **{mk_mean/max(tot_hs,1):.0%}** |
    | Lifetime medical cost saving (mean) | **${smry['mk_cst']['mean']:,.0f}** |
    """)

    if use_ltc and ltc_obj is not None:
        st.subheader("Long-Term Care Cost Decomposition — Mean Parameters")

        # Point-estimate decomposition for display
        def _ltc_point_decomp(p_h, p_m, p_s, ltc_o, n_cases):
            S = np.array([p_h, p_m, p_s, 0.0])
            sped_acc = cg_acc = 0.0
            for t in range(max(int(LE), 1)):
                disc = (1 + r_disc) ** (-t); mu_t = lt_qx(t)
                sped_active = float(ltc_o.sped_start_age <= t < ltc_o.sped_end_age)
                sped_acc += sped_active * ltc_o.cost_sped_ann * (
                    S[1] * ltc_o.p_sped_mild + S[2] * ltc_o.p_sped_severe) * disc
                cg_active = float(t < ltc_o.caregiver_end_age)
                cg_acc += cg_active * (S[1] * ltc_o.cost_cg_mild_ann
                                       + S[2] * ltc_o.cost_cg_severe_ann) * disc
                S_new = np.zeros(4)
                S_new[0] += S[0]*(1-mu_t); S_new[3] += S[0]*mu_t
                rm = max(1-mu_t-INFANT_MK["q_progress"], 0.0)
                S_new[1] += S[1]*rm; S_new[2] += S[1]*INFANT_MK["q_progress"]
                S_new[3] += S[1]*mu_t; S_new[2] += S[2]*(1-mu_t)
                S_new[3] += S[2]*mu_t; S_new[3] += S[3]; S = S_new
            return sped_acc * n_cases, cg_acc * n_cases

        n_comp   = smry["d_cs_comp"]["mean"]
        n_uncomp = smry["d_cs_uncomp"]["mean"]
        sped_c, cg_c = _ltc_point_decomp(
            max(1 - p_sev_ui - p_mc_ui, 0.0), p_mc_ui, p_sev_ui, ltc_obj, n_comp)
        sped_u, cg_u = _ltc_point_decomp(
            max(1 - p_mu_ui, 0.0), p_mu_ui, 0.0, ltc_obj, n_uncomp)

        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Special ed cost saving — CS comp",   f"${sped_c:,.0f}")
        l2.metric("Special ed cost saving — CS uncomp", f"${sped_u:,.0f}")
        l3.metric("Caregiver cost saving — CS comp",    f"${cg_c:,.0f}")
        l4.metric("Caregiver cost saving — CS uncomp",  f"${cg_u:,.0f}")
        st.caption(
            f"Special education: IDEA Part B, ages {ltc_obj.sped_start_age}–{ltc_obj.sped_end_age}  |  "
            f"Caregiver costs through age {ltc_obj.caregiver_end_age}  |  "
            f"Discounted at {r_disc:.1%}  |  "
            "Source: Chambers et al. (2010) SEEP; Genworth Cost of Care Survey (2023)."
        )

    with st.expander("CS Natural History — Calibration Cross-Check"):
        cr  = INFANT_MK["cs_early_cure_rate"]
        lm  = INFANT_MK["cs_late_manifest_rate"]
        nd  = INFANT_MK["cs_neuro_disorder_rate"]
        imp_sev  = lm * nd; imp_mild = lm * (1-nd); imp_h = 1 - imp_sev - imp_mild
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | P(cure \\| early treatment) | **{cr:.0%}** |
        | P(late complications \\| untreated) | **{lm:.0%}** |
        | P(neuro disorder \\| complications) | **{nd:.0%}** |

        | State | Implied (natural history) | Markov prior |
        |-------|--------------------------|--------------|
        | Healthy | {imp_h:.0%} | {max(1-p_sev_ui-p_mc_ui,0):.0%} |
        | Mild    | {imp_mild:.0%} | {p_mc_ui:.0%} |
        | Severe  | {imp_sev:.0%} | {p_sev_ui:.0%} |

        {"✅ Consistent (within ±10 pp)."
          if abs(imp_sev-p_sev_ui)<0.10 and abs(imp_mild-p_mc_ui)<0.10
          else "⚠️ Divergence >10 pp — review one or both parameter sets."}
        """)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6 · Serofast Detail
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Serofast / Prior-Treated Population")
    sf_mean    = smry["sf_cost"]["mean"]
    n_extra_screened = cohort * max(sc_e * p_id - sc_uc_eff * p_id, 0.0)
    n_sf_det   = n_extra_screened * p_sf * p_trepo_sf
    ic_with    = smry["inc_cost_hs"]["mean"]
    ic_without = ic_with - sf_mean
    icer_with  = ic_with   / max(smry["dalys_hs"]["mean"], 1e-9)
    icer_wo    = ic_without / max(smry["dalys_hs"]["mean"], 1e-9)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Serofast in cohort",        f"{cohort * p_sf:,.0f}")
    s2.metric("Incremental serofast detected", f"{n_sf_det:,.0f}")
    s3.metric("Mean serofast workup cost", f"${sf_mean:,.0f}")
    s4.metric("ICER impact",
              f"${icer_with:,.0f}",
              delta=f"${icer_with - icer_wo:+,.0f} vs naïve (no serofast modelling)",
              delta_color="inverse")
    fig_sf, ax_sf = plt.subplots(figsize=(7, 3.5))
    ax_sf.hist(df_psa["sf_cost"] / 1000, bins=60, color="darkorange", alpha=0.8, edgecolor="white")
    ax_sf.axvline(sf_mean/1000, color="k", lw=1.5, ls="--", label=f"Mean = ${sf_mean/1000:,.1f}K")
    ax_sf.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    ax_sf.set_xlabel("Serofast workup cost per cohort"); ax_sf.set_ylabel("Frequency")
    ax_sf.set_title("PSA Distribution: Serofast Workup Cost", fontweight="bold")
    ax_sf.legend(); ax_sf.grid(alpha=0.15); ax_sf.spines[["top","right"]].set_visible(False)
    st.pyplot(fig_sf, width="stretch")
    buf = io.BytesIO()
    fig_sf.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(label="Download Figure", data=buf, file_name="psa.png", mime="image/png")
    st.markdown(f"""
    | Component | Value |
    |-----------|-------|
    | Serofast workup & unnecessary treatment | **${sf_mean:,.0f}** |
    | ICER *with* serofast modelled | **${icer_with:,.0f}/DALY** |
    | ICER *without* serofast (naïve) | **${icer_wo:,.0f}/DALY** |
    | Serofast prevalence | **{p_sf:.1%}** |
    | P(treponemal+ \\| serofast) | **{p_trepo_sf:.0%}** |
    | P(unnecessarily treated) | **{p_ux_sf:.0%}** |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 7 · Assumptions & Citations
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("Model Assumptions & Parameters")

    with st.expander("v4 changes (this version)", expanded=True):
        st.markdown("""
        | Change | Detail |
        |--------|--------|
        | VSL/DALY separation | VSL now in a clearly-labelled separate WTP analysis; societal ICER uses productivity losses + maternal morbidity costs, not VSL subtraction |
        | Maternal morbidity module | Cardiovascular syphilis, neurosyphilis, pregnancy hospitalisation — DALYs and costs; toggle on/off |
        | DALY component corrections | Adds stillbirth YLL (IUFD ≥28w), optional CS excess-mortality YLL, miscarriage grief YLD, maternal hospitalisation YLD, and preterm infant YLD with an audit table |
        | Productivity loss (human capital) | BLS 2023 earnings by age band; bereavement, infant future earnings, CS wage penalties, caregiver time; friction-cost variant available |
        | Gestational-stratum prop_late | Stratum-specific early-loss fractions (0.80/0.55/0.25/0.05); weighted default replaces single scalar |
        | q_progress calibration | Brent's method solves for q matching user-specified lifetime target under US Life Table 2021 |
        | PSA convergence diagnostic | Rolling ICER mean ± 5% tolerance band |
        | CE-plane quadrant analysis | % of iterations in each quadrant with interpretation |
        | EVPI curve | E[max(NMB,0)] − max(E[NMB],0) plotted vs WTP |
        | OWSA table | Formal parameter/ICER-low/ICER-high table; downloadable Excel with OWSA + PSA sheets |
        | Credible interval labelling | "95% CrI (2.5th–97.5th)" replaces "2.5%/97.5%" throughout |
        """)

    with st.expander("CS conditional risk — derivation", expanded=False):
        st.markdown("""
        **Why a conditional hierarchy is used (reviewer note)**

        The model uses a sequential conditional outcome hierarchy:

        1. **Stillbirth** is evaluated first (pre-viability outcome).
        2. **Neonatal death** is conditional on liveborn: P(NND|syphilis+) = P(liveborn|syphilis+) × P(NND|liveborn, syphilis+).
        3. **CS** is conditional on neonatal survival: P(CS|syphilis+) = P(neonatal survivor|syphilis+) × P(CS|neonatal survivor, syphilis+).
        4. **Preterm / LBW** are independent conditional on livebirth (delivery timing is determined at birth, not influenced by post-neonatal survival).

        This avoids double-counting fetal deaths and neonatal deaths in the CS outcome pool.

        **Caveat:** CS risk is modelled as conditional on neonatal survival. In a small fraction
        of cases, CS can be present in infants who subsequently die in the neonatal period.
        This is a conservative approximation — if anything, it slightly underestimates CS
        prevalence. The published unconditional CS risk (CDC ~36% of untreated pregnancies)
        is used as a calibration anchor; the conditional adjustment is computed from the
        same sources' liveborn-conditional estimates.
        """)

    with st.expander("Gestational-age strata & prop_late derivation"):
        st.dataframe(pd.DataFrame(GES_STRATA).T.rename(
            columns=dict(w="Cohort weight", p_uc="P(screened | usual care)",
                         p_tx="P(tx complete)", prop_late="P(IUFD ≥28w | SB)")),
            width="stretch")
        eff_uc, eff_tx, eff_pl = ges_eff()
        st.caption(f"Strata-weighted: baseline coverage = **{eff_uc:.3f}** | "
                   f"tx completion = **{eff_tx:.3f}** | prop_late = **{eff_pl:.3f}**")

    with st.expander("Productivity loss parameters — BLS 2023 earnings"):
        st.dataframe(pd.DataFrame({
            "Age band": list(BLS_ANNUAL_EARNINGS.keys()),
            "Annual earnings ($)": [f"${v:,.0f}" for v in BLS_ANNUAL_EARNINGS.values()],
            "Maternal age dist.": [f"{MATERNAL_AGE_DIST[k]:.0%}" for k in BLS_ANNUAL_EARNINGS],
        }), width="stretch", hide_index=True)
        st.caption(f"Weighted maternal earnings: **${MATERNAL_WEIGHTED_EARNINGS:,.0f}/year**  "
                   f"(BLS CPS Table 5, 2023, all workers, both sexes)")

    with st.expander("Maternal morbidity module parameters"):
        
        mm_disp = MaternalMorbidity()
        st.json({
            "P(cardiovascular | untreated late latent)": mm_disp.p_cardio,
            "DW cardiovascular": mm_disp.dw_cardio,
            "Duration cardiovascular (years)": mm_disp.dur_cardio,
            "Annual cost cardiovascular ($)": mm_disp.cost_cardio,
            "P(neurosyphilis | tertiary)": mm_disp.p_neuro,
            "DW neurosyphilis": mm_disp.dw_neuro,
            "Duration neurosyphilis (years)": mm_disp.dur_neuro,
            "Annual cost neurosyphilis ($)": mm_disp.cost_neuro,
            "P(pregnancy hospitalisation | active infection)": mm_disp.p_hosp,
            "Cost per hospitalisation ($)": mm_disp.cost_hosp,
            "Lost work-days per episode": mm_disp.dur_hosp_days,
        })
    
    with st.expander("Infant Markov parameters"):
        mk_rows = []
        for k, v in INFANT_MK.items():
            if isinstance(v, dict):
                mk_rows.append({"Parameter": k, "Mean/Mode": v.get("m", v.get("mu","—")),
                                 "Lo": v.get("lo","—"), "Hi/SD": v.get("hi", v.get("sd","—"))})
            else:
                mk_rows.append({"Parameter": k, "Mean/Mode": v, "Lo":"—", "Hi/SD":"—"})
        st.dataframe(pd.DataFrame(mk_rows), width="stretch", hide_index=True)

    with st.expander("Cost parameters"):
        co = Costs(); d = asdict(co)
        rows_ = [{"Parameter": k, "Mean": f"${v:,.2f}", "SD": f"${d.get(k+'_sd',0):,.2f}"}
                 for k, v in d.items() if not k.endswith("_sd")]
        st.dataframe(pd.DataFrame(rows_), width="stretch", hide_index=True)
        st.caption(f"CPI factor (2019→2025): {CPI:.4f}×")

    with st.expander("Key citations"):
        st.markdown("""
        | Source | Used for |
        |--------|----------|
        | Chesson & Peterman (2023). STD. | CS workup cost; sequelae probabilities |
        | Sheffield et al. (2002). | TX RRs for CS outcomes |
        | CDC STI Surveillance Report (2023). | CS outcome distributions; prevalence anchors |
        | Walker et al. (2011). Lancet. | CS sequelae distribution for Markov |
        | Korenromp et al. (2018). PLOS ONE. | CS outcome severity split |
        | Veettil et al. (2023). Birth. | Stillbirth cost |
        | WHO GBD 2019. | LBW, maternal grief, neurosyphilis DWs |
        | CDC (2023). | Maternal morbidity probabilities |
        | DHHS ASPE (2023). | VSL = $13.7M |
        | BLS Current Population Survey Table 5 (2023). | Productivity loss earnings by age band |
        | US BLS CPI for Medical Care (2025). | 2019→2025 cost inflation |
        | CDC NCHS (2021). US Life Tables. | Age-specific mortality in Markov |
        | Brent (1973). Algorithms for Minimization. | q_progress calibration solver |
        """)

    with st.expander("Model limitations"):
        st.markdown("""
        - **Maternal morbidity**: probabilities are literature estimates with wide uncertainty;
          PSA propagates this via Beta distributions. Cardiovascular syphilis requires years
          of untreated late-latent infection — the 30% late-latent fraction is a conservative assumption.
        - **Productivity loss**: human-capital method does not account for substitution effects
          or household production losses. Friction-cost variant available but period (90 days)
          is calibrated to the Netherlands; US-specific estimates are not established.
        - **Partner reinfection** during pregnancy is not modelled → conservative (understates benefit).
        - **Treatment timing**: approximated via gestational strata; a continuous residual-risk
          function would be more precise.
        - **Markov mild→severe**: q_progress is sparsely parameterised in the literature;
          the calibration procedure ensures internal consistency but does not reduce
          the structural uncertainty.
        - **VSL**: derived from wage-risk studies primarily in working-age adults; applying
          to pregnancy outcomes involves methodological extrapolation acknowledged by DHHS.
        """)
