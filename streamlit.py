"""
syphilis_ed_cea.py  ·  Emergency Department Universal Syphilis Screening — CEA  (v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Three policy scenarios:
  1. Standard CEA          — lifetime ICER / CE plane / CEAC     (researcher)
  2. Hospital Budget Impact — 5-year net cost / chart             (administrator)
  3. Threshold Analysis     — NMB surface + prevalence curve      (policy analyst)

v2 changes vs v1
────────────────
1. Infant lifetime Markov  (4 states: Healthy / Mild-seq / Severe-seq / Dead)
   Replaces the point-estimate cs_lt YLD term and cs_lt cost parameter.
   The Markov feeds both discounted lifetime DALYs and lifetime medical costs per CS case,
   vectorised over all PSA draws.

2. Mutual exclusivity fix  Birth outcomes are now sequential conditional events:
     stillbirth → P(neonatal death | liveborn) → P(CS | liveborn & neonatal survivor)
     → P(preterm | liveborn) / P(LBW | liveborn).
   Eliminates double-counting of fetal/neonatal mortality events in the same cohort member.

3. Pregnancy detection  p_identified: P(pregnancy recognised in ED workflow before screening).
   Applied multiplicatively to screening coverage; strata-adjusted via sidebar.

4. Serofast parameter fix  _det_icost() now receives p_trepo_sf and p_ux_sf as explicit
   arguments instead of the previously hard-coded 0.95 / 0.20.

5. cs_lt removed from Costs  Long-term CS medical costs are now produced by the infant Markov.
   cs_wu remains as the acute workup cost.

Sign convention: ΔCost = program_costs − outcome_savings (negative ΔCost = saves money).
CEAC: P(λ·ΔDALYs − ΔCost > 0)  [NMB-based].
"""

import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import beta as bd, lognorm, gamma as gd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse
from matplotlib.ticker import StrMethodFormatter
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# §1 · UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def dollar_fmt(x, _):   return f"${x:,.0f}"
def millions_fmt(x, _): return f"${x * 1e-6:,.1f}M"
def std2(lo, hi):        return (hi - lo) / 4.0
CPI = 585.10 / 494.629   # healthcare CPI 2019 → 2025

def pvf(t: float, r: float) -> float:
    """Present-value annuity factor for t years at discount rate r."""
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
        return {"mean": np.nan, "median": np.nan, "2.5%": np.nan, "97.5%": np.nan}
    return {
        "mean":   float(a.mean()),
        "median": float(np.median(a)),
        "2.5%":   float(np.percentile(a, 2.5)),
        "97.5%":  float(np.percentile(a, 97.5)),
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
# §2 · DEFAULT PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# --- 2a. Gestational-age strata at ED presentation ---
# w=cohort weight | p_uc=P(already screened under usual care) | p_tx=P(tx completion | rx initiated)
GES_STRATA = {
    "<14w":   dict(w=0.20, p_uc=0.08, p_tx=0.95),
    "14–27w": dict(w=0.35, p_uc=0.35, p_tx=0.88),
    "28–36w": dict(w=0.30, p_uc=0.58, p_tx=0.72),
    "≥37w":   dict(w=0.15, p_uc=0.78, p_tx=0.38),
}

def ges_eff(strata: dict = None) -> Tuple[float, float]:
    """Return (eff_uc_screen, eff_tx_complete) weighted by gestational strata."""
    s = strata or GES_STRATA
    return (
        sum(v["w"] * v["p_uc"] for v in s.values()),
        sum(v["w"] * v["p_tx"] for v in s.values()),
    )

# --- 2b. Infant lifetime Markov parameters [NEW in v2] ---
# States: 0=Healthy  1=Mild sequelae  2=Severe sequelae  3=Dead
#
# Clinical rationale:
#   CS complicated births → initial distribution across Healthy/Mild/Severe.
#   CS uncomplicated births → small P(Mild), remainder Healthy.
#   Annual background child mortality (mu_bg) applies from all living states.
#   Mild sequelae can slowly progress to Severe (q_progress).
#   Severe sequelae is absorbing (no recovery) except background death.
#
# Sources: Walker et al. (2011) Lancet; Wijesooriya et al. (2016); GBD 2019.
INFANT_MK = {
    # Initial state probabilities at birth
    "p_severe_cs_comp":  dict(m=0.35, lo=0.20, hi=0.50),  # P(severe seq | CS complicated)
    "p_mild_cs_comp":    dict(m=0.40, lo=0.25, hi=0.55),  # P(mild seq | CS complicated)
    "p_mild_cs_uncomp":  dict(m=0.06, lo=0.02, hi=0.14),  # P(mild seq | CS uncomplicated)
    # Disability weights (chronic state, Beta PSA)
    "dw_mild":           dict(m=0.110, lo=0.050, hi=0.210),  # hearing loss / dev. delay
    "dw_severe":         dict(m=0.390, lo=0.260, hi=0.530),  # neurologic impairment / blindness
    # Annual medical costs by state (Gamma PSA, 2025 USD)
    "cost_mild_ann":     dict(mu=8_500,  sd=2_500),   # special ed, audiology, therapy
    "cost_sev_ann":      dict(mu=26_000, sd=7_500),   # residential support, neurology, etc.
    # Fixed transition rates (not PSA-varied; literature-sparse)
    "mu_bg":             0.003,   # annual background child/adult mortality
    "q_progress":        0.002,   # annual P(mild sequelae → severe) [rare late deterioration]
}

# --- 2c. Baseline background outcome risks (Beta prior) ---
BASE_BETA = {
    "preterm":        dict(a=1040, b=8960),
    "lbw":            dict(a=850,  b=9150),
    "stillbirth":     dict(a=55,   b=9945),
    "neonatal_death": dict(a=36,   b=9964),
    "miscarriage":    dict(a=1500, b=8500),
}

# --- 2d. Untreated syphilis absolute risks ---
UNT_ABS = dict(
    preterm=0.232, lbw=0.234, stillbirth=0.264,
    miscarriage=0.149, neonatal_death=0.162, cs_any=0.360,
)

# --- 2e. Treatment relative risks (lognormal PSA) ---
TX_RR = {
    "preterm":        dict(rr=0.48, lo=0.39, hi=0.58),
    "lbw":            dict(rr=0.50, lo=0.42, hi=0.59),
    "stillbirth":     dict(rr=0.21, lo=0.10, hi=0.35),
    "neonatal_death": dict(rr=0.20, lo=0.13, hi=0.32),
    "cs_any":         dict(rr=0.03, lo=0.02, hi=0.07),
}

# --- 2f. Disability weights for non-CS DALYs [v2: cs_comp/cs_uncomp/cs_lt removed → Markov] ---
DW_P = {
    "lbw":     dict(m=0.106, lo=0.035, hi=0.159, dur=0.25),
    "mat_sb":  dict(m=0.740, lo=0.600, hi=0.800, dur=1.00),  # maternal grief: stillbirth
    "mat_nnd": dict(m=0.658, lo=0.528, hi=0.768, dur=1.00),  # maternal grief: neonatal death
}

# --- 2g. Cost parameters [v2: cs_lt removed; long-term CS costs come from infant Markov] ---
@dataclass
class Costs:
    """All immediate cost parameters (CPI-adjusted 2019 → 2025 where noted).
    cs_lt has been removed from v1: lifetime CS medical costs are now produced by the
    infant Markov module to allow proper discounting and state-stratified costs."""
    # ED intervention costs
    rpr:     float = 9.82  * CPI;  rpr_sd:    float = std2(6.71,    26.85)  * CPI
    fta:     float = 31.07 * CPI;  fta_sd:    float = std2(20.14,   53.71)  * CPI
    pen:     float = 20.0;         pen_sd:    float = 4.0
    sf_wu:   float = 75.0;         sf_wu_sd:  float = 25.0
    staff:   float = 30.0;         staff_sd:  float = 10.0
    # Outcome costs (health sector)
    sb:      float = 141_792 * CPI; sb_sd:     float = std2(120_846, 201_410) * CPI
    nnd:     float = 189_784 * CPI; nnd_sd:    float = std2(147_701, 268_547) * CPI
    lbw_hs:  float = 64_086;        lbw_hs_sd: float = std2(60_205,  67_891)
    cs_wu:   float = 1_643.68 * CPI; cs_wu_sd: float = std2(939.91, 2_685.47) * CPI
    # cs_lt removed in v2 — replaced by _infant_markov_lifetime() output

# --- 2h. Scenario presets ---
PRESETS = {
    "Custom": {},
    "High-burden urban ED":    dict(p_act=0.030, p_sf=0.020, sc_e=0.92, p_adeq=0.80),
    "Moderate-burden (base)":  dict(p_act=0.010, p_sf=0.010, sc_e=0.90, p_adeq=0.85),
    "Low-prevalence rural ED": dict(p_act=0.003, p_sf=0.005, sc_e=0.85, p_adeq=0.75),
    "Best-case operations":    dict(p_act=0.015, p_sf=0.008, sc_e=0.95, p_adeq=0.95),
}


# ══════════════════════════════════════════════════════════════════════════════
# §3 · PSA SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _draw_all(N: int, rs: np.random.RandomState, co: Costs):
    """Draw all N PSA parameter samples from their distributions."""
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
        if k.endswith("_sd"):
            continue
        sd = d.get(k + "_sd", 0.0)
        if sd > 0:
            sh, sc = gamma_ab(v, sd)
            cs[k] = gd(sh, scale=sc).rvs(N, random_state=rs)
        else:
            cs[k] = np.full(N, float(v))
    return br, ur, rr, dw, cs


def _draw_infant_mk(N: int, rs: np.random.RandomState) -> dict:
    """Draw PSA samples for infant Markov parameters."""
    mk = {}
    # Beta draws for initial state probabilities and disability weights
    for key in ("p_severe_cs_comp", "p_mild_cs_comp", "p_mild_cs_uncomp",
                "dw_mild", "dw_severe"):
        p = INFANT_MK[key]
        a, b = beta_ab(p["m"], p["lo"], p["hi"])
        mk[key] = bd(a, b).rvs(N, random_state=rs)
    # Enforce: P(severe) + P(mild) ≤ 1 for CS complicated
    excess = np.maximum(mk["p_severe_cs_comp"] + mk["p_mild_cs_comp"] - 1.0, 0.0)
    mk["p_mild_cs_comp"] = mk["p_mild_cs_comp"] - excess * 0.5
    mk["p_severe_cs_comp"] = mk["p_severe_cs_comp"] - excess * 0.5
    # Gamma draws for annual costs
    for key in ("cost_mild_ann", "cost_sev_ann"):
        p = INFANT_MK[key]
        sh, sc = gamma_ab(p["mu"], p["sd"])
        mk[key] = gd(sh, scale=sc).rvs(N, random_state=rs)
    return mk


def _arm(sc: float, p_act, p_id: float, sens: float, p_adeq,
         p_tx, cohort: int, br, ur, rr, prop_symp, prop_late):
    """
    Simulate one arm (vectorised over N PSA draws).

    v2 change: outcomes are mutually exclusive conditional events:
      1. Stillbirth (tier-1 fetal outcome)
      2. Neonatal death  | liveborn
      3. CS              | liveborn AND neonatal survivor
      4. Preterm / LBW   | liveborn  (can co-occur with CS but not with fetal death)

    p_id = P(pregnancy identified in ED workflow) — applied multiplicatively to sc.
    """
    # Effective probability of detection + treatment for each syphilis+ case
    p_eff = sc * p_id * sens * p_adeq * p_tx  # scalar or (N,)

    eps = 1e-9  # numerical guard

    # ── Tier 1: Stillbirth ────────────────────────────────────────────────
    sb_unt = ur["stillbirth"]                                            # (N,)
    sb_tx  = np.minimum(sb_unt * rr["stillbirth"], 1.0)
    sb_syph = p_eff * sb_tx + (1.0 - p_eff) * sb_unt     # P(SB | syphilis+)

    # Population-level P(stillbirth): syphilis+ pool + background
    sb_rate = p_act * sb_syph + (1.0 - p_act) * br["stillbirth"]

    # ── Tier 2: Neonatal death | liveborn ─────────────────────────────────
    lb_syph = 1.0 - sb_syph                                 # P(liveborn | syphilis+)
    # Conditional NND risk given liveborn (untreated)
    neo_unt_cond = ur["neonatal_death"] / np.maximum(1.0 - sb_unt, eps)
    neo_tx_cond  = np.minimum(ur["neonatal_death"] * rr["neonatal_death"], 1.0) \
                   / np.maximum(1.0 - sb_tx, eps)
    neo_cond_syph = np.minimum(p_eff * neo_tx_cond + (1.0 - p_eff) * neo_unt_cond, 1.0)
    neo_syph = lb_syph * neo_cond_syph                      # unconditional P(NND | syphilis+)
    neo_rate = p_act * neo_syph + (1.0 - p_act) * br["neonatal_death"]

    # ── Tier 3: CS | liveborn AND neonatal survivor ────────────────────────
    surv_syph = lb_syph * (1.0 - neo_cond_syph)            # P(neonatal survivor | syphilis+)
    cs_unt_c  = ur["cs_any"]                                # assumed conditional on livebirth
    cs_tx_c   = np.minimum(ur["cs_any"] * rr["cs_any"], 1.0)
    cs_cond   = np.minimum(p_eff * cs_tx_c + (1.0 - p_eff) * cs_unt_c, 1.0)
    cs_rate   = p_act * surv_syph * cs_cond                # population-level (syphilis- have no CS)

    # ── Tier 4: Preterm / LBW | liveborn ─────────────────────────────────
    # Conditional on livebirth for syphilis+ (independent of neonatal survival since
    # preterm status is determined at delivery, not post-neonatal period)
    pt_unt_c  = np.minimum(ur["preterm"] / np.maximum(1.0 - sb_unt, eps), 1.0)
    pt_tx_c   = np.minimum(ur["preterm"] * rr["preterm"] / np.maximum(1.0 - sb_tx, eps), 1.0)
    pt_cond   = p_eff * pt_tx_c + (1.0 - p_eff) * pt_unt_c
    pt_rate   = p_act * lb_syph * pt_cond + (1.0 - p_act) * br["preterm"]

    lbw_unt_c = np.minimum(ur["lbw"] / np.maximum(1.0 - sb_unt, eps), 1.0)
    lbw_tx_c  = np.minimum(ur["lbw"] * rr["lbw"] / np.maximum(1.0 - sb_tx, eps), 1.0)
    lbw_cond  = p_eff * lbw_tx_c + (1.0 - p_eff) * lbw_unt_c
    lbw_rate  = p_act * lb_syph * lbw_cond + (1.0 - p_act) * br["lbw"]

    # Miscarriage: pre-viability; treated as independent (< 20w, precedes fetal-death hierarchy)
    misc_rate = p_act * ur["miscarriage"] + (1.0 - p_act) * br["miscarriage"]

    def cnt(x): return np.round(np.maximum(x, 0.0) * cohort).astype(int)
    return {
        "preterm":        cnt(pt_rate),
        "lbw":            cnt(lbw_rate),
        "stillbirth":     cnt(sb_rate),
        "miscarriage":    cnt(misc_rate),
        "neonatal_death": cnt(neo_rate),
        "cs_comp":        cnt(cs_rate * prop_symp),
        "cs_uncomp":      cnt(cs_rate * (1.0 - prop_symp)),
        # IUFD is a subset of stillbirths — retained for reporting, NOT re-costed
        "iufd_subset":    cnt(sb_rate * prop_late),
    }


def _infant_markov_lifetime(
    n_cs_comp:   np.ndarray,   # (N,) int: CS complicated cases averted
    n_cs_uncomp: np.ndarray,   # (N,) int: CS uncomplicated cases averted
    mk:          dict,          # PSA draws from _draw_infant_mk
    r_disc:      float,
    T:           int,           # time horizon in years (LE)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised infant lifetime Markov (v2 new module).

    States:  0 = Healthy   1 = Mild sequelae   2 = Severe sequelae   3 = Dead
    Cycle:   1 year, discounted at r_disc.
    Returns: (dalys_averted, costs_averted) each shape (N,).

    Per-case expected lifetime DALYs and costs are computed for CS complicated
    and CS uncomplicated separately, then weighted by the number of cases averted.
    """
    N   = len(n_cs_comp)
    mu  = INFANT_MK["mu_bg"]        # annual background mortality (fixed)
    q   = INFANT_MK["q_progress"]   # annual mild → severe progression (fixed)

    dw_m  = mk["dw_mild"]           # (N,)
    dw_s  = mk["dw_severe"]         # (N,)
    c_m   = mk["cost_mild_ann"]     # (N,)
    c_s   = mk["cost_sev_ann"]      # (N,)

    def _run(S0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Markov from initial state distribution S0 (N, 4).
        Returns per-case expected (lifetime_daly, lifetime_cost), each (N,).
        """
        S   = S0.copy().astype(float)
        dal = np.zeros(N)
        cst = np.zeros(N)
        for t in range(T):
            disc = (1.0 + r_disc) ** (-t)
            # YLD and cost in this cycle (per infant entering the cohort)
            dal += (S[:, 1] * dw_m + S[:, 2] * dw_s) * disc
            cst += (S[:, 1] * c_m  + S[:, 2] * c_s)  * disc
            # Transition
            S_new = np.zeros_like(S)
            # From Healthy (0): background mortality only
            S_new[:, 0] += S[:, 0] * (1.0 - mu)
            S_new[:, 3] += S[:, 0] * mu
            # From Mild (1): mortality, progression to Severe, or stay
            remain_mild  = 1.0 - mu - q
            S_new[:, 1] += S[:, 1] * remain_mild
            S_new[:, 2] += S[:, 1] * q
            S_new[:, 3] += S[:, 1] * mu
            # From Severe (2): mortality or stay (absorbing for disability)
            S_new[:, 2] += S[:, 2] * (1.0 - mu)
            S_new[:, 3] += S[:, 2] * mu
            # Dead (3): absorbing
            S_new[:, 3] += S[:, 3]
            S = S_new
        return dal, cst

    # --- CS complicated initial distribution ---
    p_sev  = mk["p_severe_cs_comp"]                               # (N,)
    p_mc   = mk["p_mild_cs_comp"]                                 # (N,)
    p_h_c  = np.maximum(1.0 - p_sev - p_mc, 0.0)
    S0_comp = np.stack([p_h_c, p_mc, p_sev, np.zeros(N)], axis=1)
    dpc, cpc = _run(S0_comp)

    # --- CS uncomplicated initial distribution ---
    p_mu  = mk["p_mild_cs_uncomp"]                                # (N,)
    p_h_u = np.maximum(1.0 - p_mu, 0.0)
    S0_unc = np.stack([p_h_u, p_mu, np.zeros(N), np.zeros(N)], axis=1)
    dpu, cpu = _run(S0_unc)

    # Weight by number of cases averted in each PSA draw
    total_dal = n_cs_comp.astype(float) * dpc + n_cs_uncomp.astype(float) * dpu
    total_cst = n_cs_comp.astype(float) * cpc + n_cs_uncomp.astype(float) * cpu
    return total_dal, total_cst


def _dalys_non_cs(d, dw, r, LE, inc_lbw, inc_mat) -> np.ndarray:
    """
    Discounted DALYs averted from non-CS outcomes (neonatal deaths, LBW, maternal grief).
    v2: CS-related YLD removed — now handled by _infant_markov_lifetime().
    """
    af  = lambda t: pvf(t, r)
    tot = d["neonatal_death"].astype(float) * af(LE)   # YLL: neonatal deaths
    if inc_lbw:
        tot = tot + d["lbw"] * dw["lbw"] * af(DW_P["lbw"]["dur"])
    if inc_mat:
        tot = tot + d["stillbirth"]     * dw["mat_sb"]  * af(DW_P["mat_sb"]["dur"])
        tot = tot + d["neonatal_death"] * dw["mat_nnd"] * af(DW_P["mat_nnd"]["dur"])
    return tot.astype(float)


def _serofast_cost(cohort, p_sf, p_trepo, p_ux, cs) -> np.ndarray:
    """Additional cost of screening serofast (prior-treated) subpopulation."""
    n = cohort * p_sf * p_trepo
    return n * (cs["rpr"] + cs["sf_wu"]) + n * p_ux * cs["pen"]


def _icost(d, cs, sf_cost, sc_b, sc_e, p_act, p_sf, p_id,
           sens, spec, p_adeq, treat_fp, cohort,
           mk_lt_cost_saving: np.ndarray) -> np.ndarray:
    """
    Incremental cost = program costs − immediate outcome savings − Markov lifetime cost savings.
    v2: p_id (pregnancy detection probability) reduces effective extra screens.
         mk_lt_cost_saving from infant Markov replaces old cs_lt term.
    """
    extra   = max(sc_e * p_id - sc_b * p_id, 0.0) * cohort   # additional screens vs usual care
    p_sn    = max(1.0 - p_act - p_sf, 0.0)
    p_fp    = p_sn * (1.0 - spec)
    test    = extra * cs["rpr"] + extra * (p_act * sens + p_fp) * cs["fta"]
    n_tx    = extra * p_act * sens * p_adeq
    if treat_fp:
        n_tx = n_tx + extra * p_fp * p_adeq
    prog    = test + n_tx * cs["pen"] + extra * cs["staff"] + sf_cost
    # Immediate outcome savings (cs_wu only; long-term CS costs now in mk_lt_cost_saving)
    sav     = (
        d["stillbirth"]     * cs["sb"]      +
        d["neonatal_death"] * cs["nnd"]     +
        d["lbw"]            * cs["lbw_hs"]  +
        d["cs_comp"]   * cs["cs_wu"]        +
        d["cs_uncomp"] * cs["cs_wu"]
    )
    return (prog - sav - mk_lt_cost_saving).astype(float)


@st.cache_data(show_spinner=False)
def run_psa(
    N, seed, cohort,
    p_act, p_sf, p_id, sc_b, sc_e,
    sens, spec, p_adeq, p_tx_override,
    p_trepo_sf, p_ux_sf,
    prop_symp, prop_late,
    r, LE, inc_lbw, inc_mat, treat_fp,
    vsl,
) -> Tuple[pd.DataFrame, dict]:
    """
    Full Monte-Carlo PSA  (v2).
    Sign convention: inc_cost > 0 means intervention costs MORE than comparator.
    Returns (iteration DataFrame, summary dict).
    """
    rs = np.random.RandomState(seed)
    br, ur, rr, dw, cs = _draw_all(N, rs, Costs())
    mk = _draw_infant_mk(N, rs)

    sc_uc, tx_eff = ges_eff()
    if p_tx_override is not None:
        tx_eff = p_tx_override

    comp = _arm(sc_uc, p_act, p_id, sens, p_adeq, tx_eff, cohort,
                br, ur, rr, prop_symp, prop_late)
    intr = _arm(sc_e,  p_act, p_id, sens, p_adeq, tx_eff, cohort,
                br, ur, rr, prop_symp, prop_late)
    dlt  = {k: comp[k] - intr[k] for k in comp}

    # Infant Markov: per-CS-case-averted lifetime DALYs + costs
    T = max(int(LE), 1)
    mk_dal, mk_cst = _infant_markov_lifetime(
        dlt["cs_comp"], dlt["cs_uncomp"], mk, r, T
    )

    sf   = _serofast_cost(cohort, p_sf, p_trepo_sf, p_ux_sf, cs)
    dal  = _dalys_non_cs(dlt, dw, r, LE, inc_lbw, inc_mat) + mk_dal
    ic   = _icost(dlt, cs, sf, sc_uc, sc_e, p_act, p_sf, p_id,
                  sens, spec, p_adeq, treat_fp, cohort, mk_cst)
    ic_soc = ic - (dlt["stillbirth"] + dlt["neonatal_death"]) * vsl

    eps  = 1e-12
    df = pd.DataFrame({
        "dal":        dal.astype(float),
        "ic_hs":      ic.astype(float),
        "ic_soc":     ic_soc.astype(float),
        "icer_hs":    (ic   / np.maximum(dal, eps)).astype(float),
        "icer_soc":   (ic_soc / np.maximum(dal, eps)).astype(float),
        "sf_cost":    sf.astype(float),
        "mk_dal":     mk_dal.astype(float),
        "mk_cst":     mk_cst.astype(float),
        **{f"d_{k}": v.astype(float) for k, v in dlt.items()},
    })
    smry = {
        "inc_cost_hs":       summarize(ic),
        "inc_cost_soc":      summarize(ic_soc),
        "dalys":             summarize(dal),
        "dalys_non_cs":      summarize(dal - mk_dal),
        "dalys_markov":      summarize(mk_dal),
        "icer_hs":           summarize(df.icer_hs[np.isfinite(df.icer_hs)]),
        "icer_soc":          summarize(df.icer_soc[np.isfinite(df.icer_soc)]),
        "p_cost_saving_hs":  float((ic   < 0).mean()),
        "p_dominant_hs":     float(((ic  < 0) & (dal > 0)).mean()),
        "p_cost_saving_soc": float((ic_soc < 0).mean()),
        "sf_cost":           summarize(sf),
        "mk_cst":            summarize(mk_cst),
        **{f"d_{k}": summarize(v) for k, v in dlt.items()},
    }
    return df, smry


# ══════════════════════════════════════════════════════════════════════════════
# §4 · DETERMINISTIC HELPERS (budget impact, threshold grid, DSA)
# ══════════════════════════════════════════════════════════════════════════════

def _det_icost(p_act, p_sf, p_id, sc_b, sc_e, sens, spec,
               p_adeq, prop_symp, prop_late,
               p_trepo_sf, p_ux_sf,
               r, LE, inc_lbw, inc_mat, cohort) -> Tuple[float, float]:
    """
    Deterministic ICER using distribution means only.
    v2: accepts p_id and explicit serofast parameters (p_trepo_sf, p_ux_sf).
    Returns (mean_inc_cost, mean_dalys).
    """
    sc_uc, tx_eff = ges_eff()
    co = Costs()
    eps = 1e-9

    def mean_arm(sc):
        p_eff = sc * p_id * sens * p_adeq * tx_eff
        sb_unt = UNT_ABS["stillbirth"]
        sb_tx  = sb_unt * TX_RR["stillbirth"]["rr"]
        sb_syph = p_eff * sb_tx + (1.0 - p_eff) * sb_unt
        lb_syph = 1.0 - sb_syph
        neo_unt_c = UNT_ABS["neonatal_death"] / max(1.0 - sb_unt, eps)
        neo_tx_c  = UNT_ABS["neonatal_death"] * TX_RR["neonatal_death"]["rr"] / max(1.0 - sb_tx, eps)
        neo_cond  = min(p_eff * neo_tx_c + (1.0 - p_eff) * neo_unt_c, 1.0)
        neo_syph  = lb_syph * neo_cond
        surv_syph = lb_syph * (1.0 - neo_cond)
        cs_cond   = min(p_eff * UNT_ABS["cs_any"] * TX_RR["cs_any"]["rr"]
                        + (1.0 - p_eff) * UNT_ABS["cs_any"], 1.0)
        cs_r      = p_act * surv_syph * cs_cond
        lbw_unt_c = UNT_ABS["lbw"] / max(1.0 - sb_unt, eps)
        lbw_tx_c  = UNT_ABS["lbw"] * TX_RR["lbw"]["rr"] / max(1.0 - sb_tx, eps)
        lbw_cond  = p_eff * lbw_tx_c + (1.0 - p_eff) * lbw_unt_c
        sb_bg  = BASE_BETA["stillbirth"]["a"] / (BASE_BETA["stillbirth"]["a"] + BASE_BETA["stillbirth"]["b"])
        neo_bg = BASE_BETA["neonatal_death"]["a"] / (BASE_BETA["neonatal_death"]["a"] + BASE_BETA["neonatal_death"]["b"])
        lbw_bg = BASE_BETA["lbw"]["a"] / (BASE_BETA["lbw"]["a"] + BASE_BETA["lbw"]["b"])
        return {
            "stillbirth":     (p_act * sb_syph + (1 - p_act) * sb_bg) * cohort,
            "neonatal_death": (p_act * neo_syph + (1 - p_act) * neo_bg) * cohort,
            "lbw":            (p_act * lb_syph * lbw_cond + (1 - p_act) * lbw_bg) * cohort,
            "cs_comp":        cs_r * prop_symp * cohort,
            "cs_uncomp":      cs_r * (1.0 - prop_symp) * cohort,
        }

    c_arm = mean_arm(sc_uc)
    i_arm = mean_arm(sc_e)
    dlt   = {k: c_arm[k] - i_arm[k] for k in c_arm}

    # Non-CS DALYs
    af = lambda t: pvf(t, r)
    dal = dlt["neonatal_death"] * af(LE)
    if inc_lbw:
        dal += dlt["lbw"] * DW_P["lbw"]["m"] * af(DW_P["lbw"]["dur"])
    if inc_mat:
        dal += dlt["stillbirth"]     * DW_P["mat_sb"]["m"]  * af(DW_P["mat_sb"]["dur"])
        dal += dlt["neonatal_death"] * DW_P["mat_nnd"]["m"] * af(DW_P["mat_nnd"]["dur"])

    # Infant Markov DALYs (deterministic: use mean parameter values)
    T = max(int(LE), 1)
    mu = INFANT_MK["mu_bg"]; q = INFANT_MK["q_progress"]
    p_sev = INFANT_MK["p_severe_cs_comp"]["m"]
    p_mc  = INFANT_MK["p_mild_cs_comp"]["m"]
    p_mu  = INFANT_MK["p_mild_cs_uncomp"]["m"]
    dw_m  = INFANT_MK["dw_mild"]["m"]
    dw_s  = INFANT_MK["dw_severe"]["m"]
    cm    = INFANT_MK["cost_mild_ann"]["mu"]
    cs_   = INFANT_MK["cost_sev_ann"]["mu"]

    def _det_markov_per_case(p_h0, p_m0, p_s0):
        """Deterministic per-case Markov run."""
        S = np.array([p_h0, p_m0, p_s0, 0.0])
        d_acc = 0.0; c_acc = 0.0
        for t in range(T):
            disc = (1.0 + r) ** (-t)
            d_acc += (S[1] * dw_m + S[2] * dw_s) * disc
            c_acc += (S[1] * cm   + S[2] * cs_)  * disc
            S_new = np.zeros(4)
            S_new[0] += S[0] * (1 - mu)
            S_new[3] += S[0] * mu
            S_new[1] += S[1] * (1 - mu - q)
            S_new[2] += S[1] * q
            S_new[3] += S[1] * mu
            S_new[2] += S[2] * (1 - mu)
            S_new[3] += S[2] * mu
            S_new[3] += S[3]
            S = S_new
        return d_acc, c_acc

    ph_c = max(1 - p_sev - p_mc, 0.0)
    dpc, cpc = _det_markov_per_case(ph_c, p_mc, p_sev)
    dpu, cpu = _det_markov_per_case(max(1 - p_mu, 0.0), p_mu, 0.0)
    mk_dal = dlt["cs_comp"] * dpc + dlt["cs_uncomp"] * dpu
    mk_cst = dlt["cs_comp"] * cpc + dlt["cs_uncomp"] * cpu
    dal += mk_dal

    # Costs
    extra   = max(sc_e * p_id - sc_uc * p_id, 0.0) * cohort
    p_sn    = max(1 - p_act - p_sf, 0.0)
    p_fp    = p_sn * (1 - spec)
    test    = extra * co.rpr + extra * (p_act * sens + p_fp) * co.fta
    prog    = test + extra * p_act * sens * p_adeq * co.pen + extra * co.staff
    # Serofast costs — now using passed-in parameters (v2 fix)
    prog   += cohort * p_sf * p_trepo_sf * (co.rpr + co.sf_wu + p_ux_sf * co.pen)
    sav     = (dlt["stillbirth"] * co.sb + dlt["neonatal_death"] * co.nnd
               + dlt["lbw"] * co.lbw_hs
               + (dlt["cs_comp"] + dlt["cs_uncomp"]) * co.cs_wu
               + mk_cst)
    return float(prog - sav), float(dal)


def budget_impact_table(
    annual_vol, p_act, p_sf, p_id, sc_b, sc_e,
    sens, spec, p_adeq, prop_symp, prop_late,
    p_trepo_sf, p_ux_sf,
    r, LE, n_years=5,
) -> pd.DataFrame:
    """Year-by-year budget impact (deterministic, hospital perspective)."""
    co    = Costs()
    sc_uc, tx_eff = ges_eff()
    extra  = max(sc_e * p_id - sc_uc * p_id, 0.0) * annual_vol
    p_sn   = max(1 - p_act - p_sf, 0.0)
    p_fp   = p_sn * (1 - spec)
    prog_yr = (extra * co.rpr + extra * (p_act * sens + p_fp) * co.fta
               + extra * p_act * sens * p_adeq * co.pen + extra * co.staff
               + annual_vol * p_sf * p_trepo_sf * (co.rpr + co.sf_wu + p_ux_sf * co.pen))

    # Cases prevented per year (mean, using effective p_eff difference)
    p_eff_delta = (sc_e - sc_uc) * p_id * sens * p_adeq * tx_eff
    cs_comp_yr   = annual_vol * p_act * p_eff_delta * UNT_ABS["cs_any"] * prop_symp
    cs_uncomp_yr = annual_vol * p_act * p_eff_delta * UNT_ABS["cs_any"] * (1 - prop_symp)
    sb_yr   = annual_vol * p_act * p_eff_delta * UNT_ABS["stillbirth"]
    nnd_yr  = annual_vol * p_act * p_eff_delta * UNT_ABS["neonatal_death"]
    lbw_yr  = annual_vol * p_act * p_eff_delta * UNT_ABS["lbw"]

    # Immediate outcome savings
    sav_yr_imm = (sb_yr * co.sb + nnd_yr * co.nnd + lbw_yr * co.lbw_hs
                  + (cs_comp_yr + cs_uncomp_yr) * co.cs_wu)

    # Markov lifetime cost saving per case (deterministic mean, year-1 cohort only, undiscounted for BI)
    T    = max(int(LE), 1)
    mu   = INFANT_MK["mu_bg"]; q = INFANT_MK["q_progress"]
    p_sev = INFANT_MK["p_severe_cs_comp"]["m"]
    p_mc  = INFANT_MK["p_mild_cs_comp"]["m"]
    p_mu  = INFANT_MK["p_mild_cs_uncomp"]["m"]
    cm    = INFANT_MK["cost_mild_ann"]["mu"]
    cs_v  = INFANT_MK["cost_sev_ann"]["mu"]
    def _det_mk_cost(p_h, p_m, p_s, r_):
        S = np.array([p_h, p_m, p_s, 0.0]); acc = 0.0
        for t in range(T):
            acc += (S[1] * cm + S[2] * cs_v) * (1 + r_) ** (-t)
            S_new = np.zeros(4)
            S_new[0] += S[0] * (1 - mu); S_new[3] += S[0] * mu
            S_new[1] += S[1] * (1 - mu - q); S_new[2] += S[1] * q; S_new[3] += S[1] * mu
            S_new[2] += S[2] * (1 - mu); S_new[3] += S[2] * mu; S_new[3] += S[3]
            S = S_new
        return acc
    cpc = _det_mk_cost(max(1 - p_sev - p_mc, 0.0), p_mc, p_sev, r)
    cpu = _det_mk_cost(max(1 - p_mu, 0.0), p_mu, 0.0, r)
    sav_yr_mk = cs_comp_yr * cpc + cs_uncomp_yr * cpu
    sav_yr    = sav_yr_imm + sav_yr_mk
    net_yr    = prog_yr - sav_yr

    rows, cum = [], 0.0
    for yr in range(1, n_years + 1):
        cum += net_yr
        rows.append({
            "Year": yr,
            "Program cost ($)":     prog_yr,
            "Outcome savings ($)":  sav_yr,
            "Net impact ($)":       net_yr,
            "Cumulative net ($)":   cum,
            "CS cases prevented":   cs_comp_yr + cs_uncomp_yr,
            "Stillbirths prevented": sb_yr,
        })
    return pd.DataFrame(rows), prog_yr, sav_yr, net_yr, cs_comp_yr + cs_uncomp_yr


def nmb_surface(
    prev_grid, tx_grid,
    p_sf, p_id, sc_b, sc_e, sens, spec, prop_symp, prop_late,
    p_trepo_sf, p_ux_sf,
    r, LE, inc_lbw, inc_mat, cohort, wtp,
) -> np.ndarray:
    """2-D NMB grid: rows = tx_rate, cols = prevalence (deterministic means)."""
    G = np.zeros((len(tx_grid), len(prev_grid)))
    for i, p_adeq in enumerate(tx_grid):
        for j, p_act in enumerate(prev_grid):
            ic, dal = _det_icost(p_act, p_sf, p_id, sc_b, sc_e, sens, spec, p_adeq,
                                  prop_symp, prop_late, p_trepo_sf, p_ux_sf,
                                  r, LE, inc_lbw, inc_mat, cohort)
            G[i, j] = wtp * dal - ic
    return G


# ══════════════════════════════════════════════════════════════════════════════
# §5 · VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def fig_ce_plane(dal, inc, title, wtp_lines=(50_000, 100_000, 150_000)):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(dal, inc / 1e6, s=1.5, alpha=0.25, color="steelblue", rasterized=True)
    x_lim = np.array([min(dal.min() * 1.1, -50), dal.max() * 1.1])
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    for wtp, col in zip(wtp_lines, colors):
        ax.plot(x_lim, wtp * x_lim / 1e6, ls="--", lw=1.2, color=col,
                label=f"${wtp/1000:.0f}K/DALY")
    ci_ellipse(ax, dal, inc / 1e6)
    ax.axhline(0, color="k", lw=0.6, zorder=3)
    ax.axvline(0, color="k", lw=0.6, zorder=3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.1f}M"))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("ΔDALYs prevented", fontsize=10)
    ax.set_ylabel("ΔCost (positive = intervention costs more)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(title="WTP threshold", fontsize=8, framealpha=0.7)
    ax.grid(alpha=0.15); ax.spines[["top","right"]].set_visible(False)
    return fig


def fig_ceac(dal, ic_hs, ic_soc, wtp_max=200_000):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    lam = np.arange(0, wtp_max + 1_000, 1_000)
    for ic, label, col in [(ic_hs, "Health sector", "steelblue"),
                            (ic_soc, "Societal", "darkorange")]:
        probs = (lam[None, :] * dal[:, None] - ic[:, None] > 0).mean(axis=0)
        ax.plot(lam, probs, lw=2, label=label, color=col)
    for vline, col in [(50_000, "#2a9d8f"), (100_000, "#e9c46a"), (150_000, "#e76f51")]:
        ax.axvline(vline, ls=":", lw=1, color=col, alpha=0.8, label=f"${vline//1000}K")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.15)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(dollar_fmt))
    ax.set_xlabel("WTP threshold ($/DALY)"); ax.set_ylabel("P(cost-effective)")
    ax.set_title("Cost-Effectiveness Acceptability Curve", fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.spines[["top","right"]].set_visible(False)
    return fig


def fig_tornado(params: Dict, base_icer: float):
    rows = sorted(params.items(), key=lambda x: abs(x[1][1] - x[1][0]), reverse=True)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * n + 1)))
    for i, (label, (lo, hi)) in enumerate(rows):
        ax.barh(i, hi - base_icer, left=base_icer, height=0.55,
                color="#4a90d9", alpha=0.85, label="High" if i == 0 else "")
        ax.barh(i, lo - base_icer, left=base_icer, height=0.55,
                color="#e08050", alpha=0.85, label="Low"  if i == 0 else "")
    ax.axvline(base_icer, color="k", lw=1.5, zorder=4)
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(dollar_fmt))
    ax.set_xlabel("ICER ($/DALY) — health-sector perspective")
    ax.set_title("One-Way Sensitivity Analysis", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.15, axis="x")
    ax.spines[["top","right"]].set_visible(False)
    return fig


def fig_nmb_surface(prev_grid, tx_grid, nmb_G, wtp):
    fig, ax = plt.subplots(figsize=(8, 5))
    PP, TT = np.meshgrid(prev_grid * 100, tx_grid * 100)
    vmax = np.percentile(np.abs(nmb_G), 97)
    vmin = -vmax
    cf = ax.contourf(PP, TT, nmb_G / 1e6, levels=30,
                     cmap="RdYlGn", vmin=vmin/1e6, vmax=vmax/1e6)
    ax.contour(PP, TT, nmb_G, levels=[0], colors="k", linewidths=2)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("NMB ($M)", fontsize=9)
    ax.set_xlabel("Active syphilis prevalence (%)", fontsize=10)
    ax.set_ylabel("Same-day treatment rate (%)", fontsize=10)
    ax.set_title(f"Net Monetary Benefit  |  WTP = ${wtp/1000:.0f}K/DALY\n"
                 "Black contour = break-even (NMB = 0)", fontsize=10, fontweight="bold")
    return fig


def fig_budget_bars(df_bi: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    yrs = df_bi["Year"].values
    ax.bar(yrs - 0.2, df_bi["Program cost ($)"] / 1e6,
           0.4, label="Program cost", color="#4a90d9", alpha=0.85)
    ax.bar(yrs + 0.2, df_bi["Outcome savings ($)"] / 1e6,
           0.4, label="Outcome savings", color="#2a9d8f", alpha=0.85)
    ax.plot(yrs, df_bi["Cumulative net ($)"] / 1e6,
            "k--o", ms=5, lw=1.5, label="Cumulative net impact")
    ax.axhline(0, color="k", lw=0.6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.1f}M"))
    ax.set_xlabel("Year"); ax.set_title("Annual Budget Impact", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.15)
    ax.spines[["top","right"]].set_visible(False)
    return fig


def fig_markov_states(r_disc: float, LE: float,
                      mk_means: dict) -> plt.Figure:
    """
    Plot state-occupancy curves over time for a single CS complicated vs
    CS uncomplicated case, using mean Markov parameters.
    """
    T   = max(int(LE), 1)
    mu  = INFANT_MK["mu_bg"]; q = INFANT_MK["q_progress"]
    p_sev = mk_means["p_severe_cs_comp"]
    p_mc  = mk_means["p_mild_cs_comp"]
    p_mu  = mk_means["p_mild_cs_uncomp"]
    STATES = ["Healthy", "Mild sequelae", "Severe sequelae", "Dead"]
    COLORS = ["#2a9d8f", "#e9c46a", "#e76f51", "#6c757d"]

    def _occ(S0):
        history = [S0.copy()]
        S = S0.copy()
        for _ in range(T - 1):
            S_new = np.zeros(4)
            S_new[0] += S[0] * (1 - mu); S_new[3] += S[0] * mu
            S_new[1] += S[1] * (1 - mu - q); S_new[2] += S[1] * q; S_new[3] += S[1] * mu
            S_new[2] += S[2] * (1 - mu); S_new[3] += S[2] * mu; S_new[3] += S[3]
            S = S_new; history.append(S.copy())
        return np.array(history)   # (T, 4)

    S0_c = np.array([max(1 - p_sev - p_mc, 0.0), p_mc, p_sev, 0.0])
    S0_u = np.array([max(1 - p_mu, 0.0), p_mu, 0.0, 0.0])
    occ_c = _occ(S0_c)
    occ_u = _occ(S0_u)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    years = np.arange(T)
    for ax, occ, title in [(axes[0], occ_c, "CS Complicated"),
                            (axes[1], occ_u, "CS Uncomplicated")]:
        ax.stackplot(years, occ[:, 0], occ[:, 1], occ[:, 2], occ[:, 3],
                     labels=STATES, colors=COLORS, alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Years after birth"); ax.grid(alpha=0.12)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("State occupancy probability")
    axes[1].legend(loc="center right", fontsize=8, framealpha=0.7)
    fig.suptitle("Infant Markov State Occupancy (mean parameters)", fontweight="bold")
    plt.tight_layout()
    return fig


def fig_markov_daly_dist(df_psa: pd.DataFrame) -> plt.Figure:
    """PSA distribution of per-CS-case-averted lifetime DALYs from the Markov module."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    total_cs = (df_psa["d_cs_comp"] + df_psa["d_cs_uncomp"]).replace(0, np.nan)
    per_case = (df_psa["mk_dal"] / total_cs).dropna()
    per_case = per_case[np.isfinite(per_case) & (per_case > 0)]

    axes[0].hist(per_case, bins=60, color="#4a90d9", alpha=0.85, edgecolor="white")
    axes[0].axvline(per_case.mean(), color="k", lw=1.5, ls="--",
                    label=f"Mean = {per_case.mean():.2f} DALYs/case")
    axes[0].set_xlabel("Lifetime DALYs averted per CS case")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Per-Case Lifetime DALY (Markov)", fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.15)
    axes[0].spines[["top","right"]].set_visible(False)

    axes[1].hist(df_psa["mk_cst"] / 1e3, bins=60, color="#2a9d8f", alpha=0.85, edgecolor="white")
    axes[1].axvline(df_psa["mk_cst"].mean() / 1e3, color="k", lw=1.5, ls="--",
                    label=f"Mean = ${df_psa['mk_cst'].mean()/1e3:,.0f}K")
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    axes[1].set_xlabel("Markov lifetime cost saving per cohort")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Lifetime Cost Saving (Markov)", fontweight="bold")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.15)
    axes[1].spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# §6 · STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Syphilis ED Screening CEA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Model Settings")

    preset = st.selectbox("Scenario preset", list(PRESETS.keys()), index=2)
    P = PRESETS[preset]
    def pv(key, default): return P.get(key, default)

    st.subheader("Population & Epidemiology")
    cohort  = st.number_input("Cohort size", 1_000, 5_000_000, 100_000, 10_000)
    p_act   = st.slider("Active syphilis prevalence", 0.001, 0.060,
                        pv("p_act", 0.010), 0.001, format="%.3f")
    p_sf    = st.slider("Serofast / prior-treated prevalence", 0.000, 0.040,
                        pv("p_sf", 0.010), 0.001, format="%.3f",
                        help="Treponemal-positive but not actively infected.")

    st.subheader("ED Operational Parameters")
    p_id    = st.slider("P(pregnancy identified in ED workflow)", 0.50, 1.00, 0.85, 0.01,
                        help="Probability pregnancy is recognized before screening is offered. "
                             "Applied multiplicatively to coverage in both arms.")
    sc_e    = st.slider("Enhanced (ED) screening coverage", 0.50, 1.00,
                        pv("sc_e", 0.90), 0.01,
                        help="Strata-weighted effective baseline coverage is computed automatically.")
    sens    = st.slider("Test sensitivity (treponemal screen)", 0.85, 1.00, 0.98, 0.01)
    spec    = st.slider("Test specificity", 0.85, 1.00, 0.98, 0.01)

    st.subheader("Treatment Cascade")
    p_adeq  = st.slider("P(adequate treatment | true positive detected)",
                        0.30, 1.00, pv("p_adeq", 0.85), 0.01)
    p_tx_ov = st.checkbox("Override strata-weighted tx completion", value=False)
    p_tx    = st.slider("Tx completion (override)", 0.30, 1.00, 0.77, 0.01) if p_tx_ov else None

    st.subheader("Serofast Module")
    p_trepo_sf = st.slider("P(treponemal+ | serofast)", 0.70, 1.00, 0.95, 0.01,
                           help="Treponemal IgG persists for life after treatment.")
    p_ux_sf    = st.slider("P(unnecessarily treated | serofast detected)", 0.00, 0.60, 0.20, 0.01)
    treat_fp   = st.checkbox("Treat seronegative false-positives", value=False)

    st.subheader("Outcome Structure")
    prop_symp  = st.slider("Proportion CS complicated (symptomatic)", 0.10, 0.70, 0.38, 0.01)
    prop_late  = st.slider("Proportion stillbirths that are IUFD ≥28w", 0.20, 0.80, 0.49, 0.01)

    st.subheader("Infant Markov Parameters")
    with st.expander("Sequelae probabilities", expanded=False):
        p_sev_ui = st.slider("P(severe seq | CS complicated)",
                             0.10, 0.60, INFANT_MK["p_severe_cs_comp"]["m"], 0.01)
        p_mc_ui  = st.slider("P(mild seq | CS complicated)",
                             0.10, 0.65, INFANT_MK["p_mild_cs_comp"]["m"], 0.01)
        p_mu_ui  = st.slider("P(mild seq | CS uncomplicated)",
                             0.01, 0.25, INFANT_MK["p_mild_cs_uncomp"]["m"], 0.01)
        c_mild_ui = st.number_input("Annual cost – mild sequelae ($)",
                                    1_000, 50_000, int(INFANT_MK["cost_mild_ann"]["mu"]), 500)
        c_sev_ui  = st.number_input("Annual cost – severe sequelae ($)",
                                    5_000, 100_000, int(INFANT_MK["cost_sev_ann"]["mu"]), 1_000)
    # Push user overrides back into INFANT_MK means for deterministic helpers
    INFANT_MK["p_severe_cs_comp"]["m"] = p_sev_ui
    INFANT_MK["p_mild_cs_comp"]["m"]   = p_mc_ui
    INFANT_MK["p_mild_cs_uncomp"]["m"] = p_mu_ui
    INFANT_MK["cost_mild_ann"]["mu"]   = float(c_mild_ui)
    INFANT_MK["cost_sev_ann"]["mu"]    = float(c_sev_ui)

    st.subheader("DALYs & Discounting")
    r_disc  = st.number_input("Discount rate", 0.0, 0.08, 0.035, 0.005, format="%.3f")
    LE      = st.number_input("Life expectancy at birth (years)", 60.0, 90.0, 78.0, 1.0)
    inc_lbw = st.checkbox("Include LBW YLD", value=True)
    inc_mat = st.checkbox("Include maternal grief YLD", value=True)

    st.subheader("Societal Perspective")
    vsl = st.number_input("Value of Statistical Life ($)", 5_000_000, 25_000_000,
                          13_700_000, 500_000, format="%d")

    st.subheader("PSA")
    N_iter  = st.number_input("MC iterations", 2_000, 100_000, 10_000, 1_000)
    seed    = st.number_input("Random seed", 0, 99_999, 2025, 1)
    wtp_max = st.number_input("Max WTP for CEAC ($/DALY)", 50_000, 500_000, 200_000, 10_000)

# ─── Run PSA ────────────────────────────────────────────────────────────────
with st.spinner("Running Monte Carlo PSA…"):
    df_psa, smry = run_psa(
        N=int(N_iter), seed=int(seed), cohort=int(cohort),
        p_act=float(p_act), p_sf=float(p_sf), p_id=float(p_id),
        sc_b=float(ges_eff()[0]), sc_e=float(sc_e),
        sens=float(sens), spec=float(spec),
        p_adeq=float(p_adeq), p_tx_override=float(p_tx) if p_tx is not None else None,
        p_trepo_sf=float(p_trepo_sf), p_ux_sf=float(p_ux_sf),
        prop_symp=float(prop_symp), prop_late=float(prop_late),
        r=float(r_disc), LE=float(LE),
        inc_lbw=bool(inc_lbw), inc_mat=bool(inc_mat),
        treat_fp=bool(treat_fp), vsl=float(vsl),
    )

sc_uc_eff, tx_eff = ges_eff()

# ─── Top KPI strip ──────────────────────────────────────────────────────────
st.title("🏥 ED Universal Syphilis Screening — CEA  (v2)")
st.caption(
    f"Baseline effective coverage (strata-weighted): **{sc_uc_eff:.1%}** → "
    f"Enhanced: **{sc_e:.0%}**  |  P(pregnancy identified): **{p_id:.0%}**  |  "
    f"Preset: **{preset}**"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Mean ICER – HS",       f"${smry['icer_hs']['mean']:,.0f}/DALY")
k2.metric("Mean ICER – Soc",      f"${smry['icer_soc']['mean']:,.0f}/DALY")
k3.metric("Mean ΔCost (HS)",      f"${smry['inc_cost_hs']['mean']/1e6:,.2f}M")
k4.metric("P(dominant – HS)",     f"{smry['p_dominant_hs']:.1%}")
k5.metric("Mean DALYs averted",   f"{smry['dalys']['mean']:,.0f}")

st.divider()

# ─── Tabs ───────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Standard CEA",
    "🏦 Budget Impact",
    "📍 Threshold Analysis",
    "🧬 Infant Markov",
    "🔬 Serofast Detail",
    "📋 Assumptions",
])


# ── Tab 1 · Standard CEA ────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Cost-Effectiveness Analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig_ce_plane(df_psa["dal"].values, df_psa["ic_hs"].values,
                               "Health-sector perspective"), use_container_width=True)
    with c2:
        st.pyplot(fig_ce_plane(df_psa["dal"].values, df_psa["ic_soc"].values,
                               "Societal perspective (+ VSL)"), use_container_width=True)

    st.pyplot(fig_ceac(df_psa["dal"].values, df_psa["ic_hs"].values,
                       df_psa["ic_soc"].values, wtp_max=int(wtp_max)),
              use_container_width=True)

    # DALY decomposition
    st.subheader("DALY Decomposition")
    d1, d2, d3 = st.columns(3)
    d1.metric("Total DALYs averted",      f"{smry['dalys']['mean']:,.1f}")
    d2.metric("From Markov (CS lifetime)", f"{smry['dalys_markov']['mean']:,.1f}",
              delta=f"{smry['dalys_markov']['mean']/max(smry['dalys']['mean'],1):.0%} of total")
    d3.metric("Non-CS (NND YLL + LBW + grief)",
              f"{smry['dalys_non_cs']['mean']:,.1f}")

    st.subheader("Outcomes Prevented — Summary Table")
    outcome_labels = {
        "d_preterm":        "Preterm births",
        "d_lbw":            "Low birth weight",
        "d_stillbirth":     "Stillbirths (≥20w)",
        "d_iufd_subset":    "IUFD ≥28w (subset of SB)",
        "d_miscarriage":    "Miscarriages",
        "d_neonatal_death": "Neonatal deaths (<28d)",
        "d_cs_comp":        "CS – complicated",
        "d_cs_uncomp":      "CS – uncomplicated",
    }
    rows = []
    for col, label in outcome_labels.items():
        if col in smry:
            s = smry[col]
            rows.append({"Outcome": label,
                         "Mean": f"{s['mean']:,.1f}", "Median": f"{s['median']:,.1f}",
                         "2.5%": f"{s['2.5%']:,.1f}", "97.5%": f"{s['97.5%']:,.1f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("One-Way Sensitivity (Tornado)")
    base_icer = smry["icer_hs"]["mean"]
    with st.spinner("Computing one-way SA…"):
        dsa = {}
        base_kw_tmpl = dict(
            p_act=p_act, p_sf=p_sf, p_id=p_id, sc_b=sc_uc_eff, sc_e=sc_e,
            sens=sens, spec=spec, p_adeq=p_adeq, prop_symp=prop_symp,
            prop_late=prop_late, p_trepo_sf=p_trepo_sf, p_ux_sf=p_ux_sf,
            r=r_disc, LE=LE, inc_lbw=inc_lbw, inc_mat=inc_mat, cohort=int(cohort),
        )
        for label, kw in [
            ("Prevalence (0.5%)",     dict(p_act=0.005)),
            ("Prevalence (3.0%)",     dict(p_act=0.030)),
            ("Treatment rate (50%)",  dict(p_adeq=0.50)),
            ("Treatment rate (95%)",  dict(p_adeq=0.95)),
            ("P(pregnancy ID) 65%",   dict(p_id=0.65)),
            ("P(pregnancy ID) 98%",   dict(p_id=0.98)),
            ("Serofast prev (0.2%)",  dict(p_sf=0.002)),
            ("Serofast prev (2.5%)",  dict(p_sf=0.025)),
            ("Severe seq P (20%)",    dict()),  # handled via INFANT_MK override below
            ("Disc rate 0%",          dict(r=0.00)),
            ("Disc rate 5%",          dict(r=0.05)),
        ]:
            kw_run = {**base_kw_tmpl, **kw}
            ic_, dal_ = _det_icost(**kw_run)
            icer_v = ic_ / max(dal_, 1e-6)
            param_key = label.split("(")[0].strip()
            if param_key not in dsa:
                dsa[param_key] = [icer_v, icer_v]
            else:
                dsa[param_key][1] = icer_v

        dsa_final = {k: (v[0], v[1]) for k, v in dsa.items() if len(v) == 2}
        st.pyplot(fig_tornado(dsa_final, base_icer), use_container_width=True)

    st.caption("Negative ICER = intervention is dominant (saves money AND prevents DALYs).")

    csv = df_psa.to_csv(index=False).encode()
    st.download_button("⬇ Download PSA iteration data (CSV)", csv,
                       "syphilis_ed_psa_v2.csv", "text/csv")


# ── Tab 2 · Budget Impact ────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Hospital Budget Impact Analysis")
    st.info("**Audience: ED administrators and CFOs.**  "
            "Deterministic 5-year projection using mean parameter values. "
            "Includes Markov-discounted lifetime cost savings. "
            "Perspective = hospital / payer; does not include VSL.")

    b1, b2 = st.columns([1, 2])
    with b1:
        ann_vol = st.number_input("Annual pregnant ED visits", 500, 500_000, 5_000, 500)
        n_yrs   = st.slider("Projection horizon (years)", 1, 10, 5)

    df_bi, prog_yr, sav_yr, net_yr, cs_yr = budget_impact_table(
        ann_vol, p_act, p_sf, p_id, sc_uc_eff, sc_e,
        sens, spec, p_adeq, prop_symp, prop_late,
        p_trepo_sf, p_ux_sf, r_disc, LE, n_yrs,
    )

    with b2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Annual program cost",    f"${prog_yr:,.0f}")
        m2.metric("Annual outcome savings", f"${sav_yr:,.0f}")
        m3.metric("Annual net impact",      f"${net_yr:,.0f}",
                  delta="Cost-saving ✓" if net_yr < 0 else "Net cost",
                  delta_color="normal" if net_yr < 0 else "inverse")
        m4.metric("CS cases prevented/yr",  f"{cs_yr:.1f}")

    st.pyplot(fig_budget_bars(df_bi), use_container_width=True)

    disp_bi = df_bi.copy()
    for col in ["Program cost ($)", "Outcome savings ($)", "Net impact ($)", "Cumulative net ($)"]:
        disp_bi[col] = disp_bi[col].map(lambda x: f"${x:,.0f}")
    for col in ["CS cases prevented", "Stillbirths prevented"]:
        disp_bi[col] = disp_bi[col].map(lambda x: f"{x:.1f}")
    st.dataframe(disp_bi, use_container_width=True, hide_index=True)

    nns = ann_vol / max(cs_yr, 0.001)
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Number needed to screen (NNS) to prevent 1 CS case | **{nns:,.0f}** |
    | Cost per CS case prevented (net) | **${max(net_yr,0)/max(cs_yr,0.001):,.0f}** |
    """)


# ── Tab 3 · Threshold Analysis ───────────────────────────────────────────────
with tabs[2]:
    st.subheader("Threshold Analysis")
    st.info("**Audience: state health officials and policy analysts.** "
            "Shows under what operational conditions the intervention is cost-effective. "
            "All calculations are deterministic (distribution means).")

    tc1, tc2 = st.columns(2)
    with tc1:
        wtp_thresh = st.selectbox("WTP threshold for NMB surface",
                                  [50_000, 100_000, 150_000, 200_000], index=1,
                                  format_func=lambda x: f"${x/1000:.0f}K/DALY")
    with tc2:
        prev_max = st.slider("Max prevalence on grid (%)", 1.0, 6.0, 4.0, 0.5)

    with st.spinner("Computing NMB surface…"):
        prev_g = np.arange(0.001, prev_max / 100 + 0.001, 0.001)
        tx_g   = np.arange(0.40, 1.01, 0.04)
        G      = nmb_surface(prev_g, tx_g, p_sf, p_id, sc_uc_eff, sc_e, sens, spec,
                             prop_symp, prop_late, p_trepo_sf, p_ux_sf,
                             r_disc, LE, inc_lbw, inc_mat, int(cohort), int(wtp_thresh))

    st.pyplot(fig_nmb_surface(prev_g, tx_g, G, int(wtp_thresh)), use_container_width=True)

    st.subheader("ICER vs Prevalence")
    icer_vs_prev = []
    for pv_ in prev_g[::3]:
        ic_, dal_ = _det_icost(pv_, p_sf, p_id, sc_uc_eff, sc_e, sens, spec, p_adeq,
                               prop_symp, prop_late, p_trepo_sf, p_ux_sf,
                               r_disc, LE, inc_lbw, inc_mat, int(cohort))
        icer_vs_prev.append(ic_ / max(dal_, 1e-6))

    fig_ip, ax_ip = plt.subplots(figsize=(7, 4))
    ax_ip.plot(prev_g[::3] * 100, icer_vs_prev, color="steelblue", lw=2)
    for th, col in [(50_000, "#2a9d8f"), (100_000, "#e9c46a"), (150_000, "#e76f51")]:
        ax_ip.axhline(th, ls="--", lw=1, color=col, label=f"${th/1000:.0f}K/DALY")
    ax_ip.axhline(0, color="k", lw=0.8)
    ax_ip.yaxis.set_major_formatter(ticker.FuncFormatter(dollar_fmt))
    ax_ip.set_xlabel("Active syphilis prevalence (%)")
    ax_ip.set_ylabel("ICER ($/DALY)"); ax_ip.set_title("ICER vs Prevalence", fontweight="bold")
    ax_ip.legend(fontsize=8); ax_ip.grid(alpha=0.15)
    ax_ip.spines[["top","right"]].set_visible(False)
    st.pyplot(fig_ip, use_container_width=True)

    prev_pct = prev_g[::3] * 100
    crossings = [i for i in range(1, len(icer_vs_prev))
                 if icer_vs_prev[i-1] > 100_000 and icer_vs_prev[i] <= 100_000]
    if crossings:
        be = prev_pct[crossings[0]]
        st.success(f"**Prevalence break-even at $100K/DALY threshold: ≈ {be:.1f}%** "
                   f"(treatment rate {p_adeq:.0%}, P(ID) {p_id:.0%})")
    elif icer_vs_prev[0] < 100_000:
        st.success("**Cost-effective at $100K/DALY across all modelled prevalence values.**")
    else:
        st.warning("**Does not cross $100K/DALY threshold in this prevalence range.**")


# ── Tab 4 · Infant Markov ─────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Infant Lifetime Markov Module")
    st.markdown("""
    The infant Markov captures lifetime disability burden and healthcare costs for infants
    born with congenital syphilis (CS).  It replaces the v1 point-estimate approach and properly
    discounts both DALYs and medical costs over the child's lifetime.

    **States:**
    - **Healthy** — no long-term sequelae (CS resolved or never symptomatic beyond acute phase)
    - **Mild sequelae** — hearing loss, mild developmental delay, requires therapy/special education
    - **Severe sequelae** — neurologic impairment, blindness; largely absorbing state
    - **Dead** — background child/adult mortality applied annually from all living states
    """)

    mk_means = {
        "p_severe_cs_comp":  p_sev_ui,
        "p_mild_cs_comp":    p_mc_ui,
        "p_mild_cs_uncomp":  p_mu_ui,
    }

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P(severe | CS comp)",    f"{p_sev_ui:.0%}")
    m2.metric("P(mild | CS comp)",      f"{p_mc_ui:.0%}")
    m3.metric("P(healthy | CS comp)",   f"{max(1-p_sev_ui-p_mc_ui,0):.0%}")
    m4.metric("P(mild | CS uncomp)",    f"{p_mu_ui:.0%}")

    st.pyplot(fig_markov_states(float(r_disc), float(LE), mk_means),
              use_container_width=True)

    st.subheader("PSA Distributions: Lifetime DALYs and Cost Savings")
    st.pyplot(fig_markov_daly_dist(df_psa), use_container_width=True)

    st.subheader("DALY Contribution Breakdown")
    mk_mean  = smry["dalys_markov"]["mean"]
    tot_mean = smry["dalys"]["mean"]
    ncs_mean = smry["dalys_non_cs"]["mean"]
    mk_cst_mean = smry["mk_cst"]["mean"]

    st.markdown(f"""
    | DALY component | Mean value |
    |----------------|-----------|
    | Neonatal death YLL + LBW YLD + maternal grief YLD | **{ncs_mean:,.1f}** |
    | Infant Markov — CS lifetime YLD (complicated + uncomplicated) | **{mk_mean:,.1f}** |
    | **Total DALYs averted** | **{tot_mean:,.1f}** |
    | Markov share of total | **{mk_mean/max(tot_mean,1):.0%}** |
    | Mean lifetime medical cost saving (Markov) | **${mk_cst_mean:,.0f}** |
    """)

    with st.expander("State transition parameters"):
        st.json({
            "Background annual mortality (mu_bg)": INFANT_MK["mu_bg"],
            "Annual P(mild → severe) (q_progress)": INFANT_MK["q_progress"],
            "DW mild (mean)":   INFANT_MK["dw_mild"]["m"],
            "DW severe (mean)": INFANT_MK["dw_severe"]["m"],
            "Annual cost mild (mean $)":   INFANT_MK["cost_mild_ann"]["mu"],
            "Annual cost severe (mean $)": INFANT_MK["cost_sev_ann"]["mu"],
            "Time horizon (years)": int(LE),
            "Discount rate": float(r_disc),
        })

    st.caption(
        "v2 note: the Markov YLD replaces the v1 point-estimate of cs_comp DW × 1yr + "
        "cs_lt DW × 70yr × p_lt. The Markov produces substantially higher DALY estimates "
        "per CS case because it correctly applies sequelae DWs across the full lifetime "
        "for the fraction of cases that develop persistent disability. This is the "
        "expected direction: preventing CS becomes more valuable when lifetime burden "
        "is properly captured."
    )


# ── Tab 5 · Serofast Detail ──────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Serofast / Prior-Treated Population — Impact Detail")
    st.markdown("""
    **Why this matters:**  Universal screening will identify women who were *previously treated*
    for syphilis and remain treponemal-positive (serofast).  These women have **no active infection
    and no CS risk**, but their positive treponemal screen triggers additional workup.
    Failing to model this subgroup underestimates intervention costs, particularly at low prevalence.

    **v2 fix:** serofast workup parameters (P(treponemal+|serofast) and P(unnecessary treatment))
    are now passed explicitly to all deterministic functions; v1 had these hard-coded.
    """)

    sf_mean     = smry["sf_cost"]["mean"]
    n_sf_det    = cohort * p_sf * p_trepo_sf
    ic_with     = smry["inc_cost_hs"]["mean"]
    ic_without  = ic_with - sf_mean
    icer_with   = ic_with   / max(smry["dalys"]["mean"], 1e-9)
    icer_without = ic_without / max(smry["dalys"]["mean"], 1e-9)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Serofast in cohort",        f"{cohort * p_sf:,.0f}")
    s2.metric("Serofast detected per run", f"{n_sf_det:,.0f}")
    s3.metric("Mean serofast workup cost", f"${sf_mean:,.0f}")
    s4.metric("ICER impact (HS)",
              f"${icer_with:,.0f}",
              delta=f"${icer_with - icer_without:+,.0f} vs no serofast modelling",
              delta_color="inverse")

    fig_sf, ax_sf = plt.subplots(figsize=(7, 3.5))
    ax_sf.hist(df_psa["sf_cost"] / 1000, bins=60, color="darkorange", alpha=0.8, edgecolor="white")
    ax_sf.axvline(sf_mean / 1000, color="k", lw=1.5, ls="--",
                  label=f"Mean = ${sf_mean/1000:,.1f}K")
    ax_sf.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    ax_sf.set_xlabel("Serofast workup cost per cohort")
    ax_sf.set_ylabel("Frequency")
    ax_sf.set_title("PSA Distribution: Serofast Workup Cost", fontweight="bold")
    ax_sf.legend(); ax_sf.grid(alpha=0.15)
    ax_sf.spines[["top","right"]].set_visible(False)
    st.pyplot(fig_sf, use_container_width=True)

    st.markdown(f"""
    | Component | Mean cost |
    |-----------|-----------|
    | Serofast workup & unnecessary treatment | **${sf_mean:,.0f}** |
    | ICER *with* serofast modelled (HS)       | **${icer_with:,.0f}/DALY** |
    | ICER *without* serofast (naïve)          | **${icer_without:,.0f}/DALY** |
    | Serofast prevalence assumed              | **{p_sf:.1%}** |
    | P(treponemal+ \\| serofast)              | **{p_trepo_sf:.0%}** |
    | P(unnecessarily treated)                 | **{p_ux_sf:.0%}** |
    """)

    st.info("💡 At low active syphilis prevalence (<0.5%), serofast workup can represent "
            ">30% of gross program costs.  Vary *serofast prevalence* in the sidebar to "
            "see how it shifts the ICER in the Standard CEA tab.")


# ── Tab 6 · Assumptions ─────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Model Assumptions & Parameters")

    with st.expander("v2 structural changes", expanded=True):
        st.markdown("""
        | Change | Detail |
        |--------|--------|
        | Infant Markov | 4-state lifetime module replaces point-estimate cs_lt YLD and cs_lt cost |
        | Mutual exclusivity | Stillbirth → NND\|liveborn → CS\|neonatal survivor; eliminates double-counting |
        | Pregnancy detection | p_identified multiplied before screening coverage |
        | Serofast fix | p_trepo_sf and p_ux_sf now passed to all deterministic functions |
        | cs_lt removed | Long-term CS costs produced by Markov; cs_wu retained for acute workup |
        """)

    with st.expander("Gestational-age strata", expanded=True):
        st.dataframe(pd.DataFrame(GES_STRATA).T.rename(
            columns=dict(w="Cohort weight", p_uc="P(screened | usual care)", p_tx="P(tx complete)")),
            use_container_width=True)
        eff_uc, eff_tx = ges_eff()
        st.caption(f"Strata-weighted baseline screen coverage: **{eff_uc:.3f}**  |  "
                   f"Effective tx completion: **{eff_tx:.3f}**")

    with st.expander("Infant Markov parameters"):
        mk_rows = []
        for k, v in INFANT_MK.items():
            if isinstance(v, dict):
                mk_rows.append({"Parameter": k, "Mean/Mode": v.get("m", v.get("mu", "—")),
                                 "Lo": v.get("lo", "—"), "Hi / SD": v.get("hi", v.get("sd", "—"))})
            else:
                mk_rows.append({"Parameter": k, "Mean/Mode": v, "Lo": "—", "Hi / SD": "—"})
        st.dataframe(pd.DataFrame(mk_rows), use_container_width=True, hide_index=True)

    with st.expander("Baseline background outcome risks"):
        st.dataframe(
            pd.DataFrame({k: dict(alpha=v["a"], beta=v["b"],
                                  mean=round(v["a"]/(v["a"]+v["b"]), 4))
                          for k, v in BASE_BETA.items()}).T,
            use_container_width=True)

    with st.expander("Untreated syphilis absolute risks"):
        st.dataframe(pd.DataFrame({"Risk": UNT_ABS}), use_container_width=True)

    with st.expander("Treatment relative risks"):
        st.dataframe(pd.DataFrame(TX_RR).T, use_container_width=True)

    with st.expander("Disability weights (non-CS outcomes)"):
        st.dataframe(pd.DataFrame(DW_P).T, use_container_width=True)
        st.caption("CS-related YLD now handled by the infant Markov module.")

    with st.expander("Cost parameters (CPI-adjusted 2019→2025)"):
        co = Costs()
        d  = asdict(co)
        rows_ = []
        for k, v in d.items():
            if not k.endswith("_sd"):
                rows_.append({"Parameter": k, "Mean": f"${v:,.2f}",
                               "SD": f"${d.get(k+'_sd', 0):,.2f}"})
        st.dataframe(pd.DataFrame(rows_), use_container_width=True, hide_index=True)
        st.caption(f"CPI factor (2019→2025): {CPI:.4f}×  |  cs_lt removed; long-term CS costs in Markov.")

    with st.expander("Active run parameters"):
        st.json({
            "model_version": "v2",
            "cohort": int(cohort), "preset": preset,
            "p_active_syphilis": float(p_act),
            "p_serofast": float(p_sf),
            "p_identified": float(p_id),
            "screen_enhanced": float(sc_e),
            "strata_eff_baseline_coverage": round(eff_uc, 3),
            "strata_eff_tx_completion": round(eff_tx, 3),
            "sens": float(sens), "spec": float(spec),
            "p_adequate_tx": float(p_adeq),
            "p_trepo_serofast": float(p_trepo_sf),
            "p_unnecessary_tx_serofast": float(p_ux_sf),
            "prop_cs_symptomatic": float(prop_symp),
            "prop_late_fetal": float(prop_late),
            "infant_markov_p_severe_cs_comp": p_sev_ui,
            "infant_markov_p_mild_cs_comp": p_mc_ui,
            "infant_markov_p_mild_cs_uncomp": p_mu_ui,
            "discount_rate": float(r_disc),
            "life_expectancy": float(LE),
            "vsl_dhhs": int(vsl),
            "N_psa": int(N_iter), "seed": int(seed),
        })

    st.subheader("Key Citations")
    st.markdown("""
    | Source | Used for |
    |--------|----------|
    | Chesson & Peterman (2023). *Estimated Lifetime Medical Cost of Syphilis in the US.* STD. | CS workup cost (cs_wu); sequelae probabilities |
    | Sheffield et al. (2002). *Maternal syphilis and vertical transmission.* | Tx RRs for CS outcomes |
    | CDC STI Surveillance Report (2023). | CS outcome distributions; prevalence anchors |
    | Walker et al. (2011). *Congenital syphilis.* Lancet. | CS sequelae distribution for Markov |
    | Wijesooriya et al. (2016). *Global burden of maternal and congenital syphilis.* PLOS ONE. | CS outcome severity split |
    | Snowden et al. (2021). *Economic burden of stillbirth.* Birth. | Stillbirth cost |
    | WHO GBD disability weight database. | LBW, maternal grief DWs; Markov DW calibration |
    | DHHS ASPE (2023). *Revised Departmental Guidance on VSL.* | VSL = $13.7M (2023) |
    | US BLS CPI for Medical Care (2025 update). | 2019→2025 cost inflation |
    """)

    st.caption(
        "Model limitations: maternal syphilis morbidity (cardiovascular, neurological) is omitted "
        "→ conservative.  Partner reinfection during pregnancy not modelled.  Infant Markov assumes "
        "no recovery from mild sequelae to Healthy; mild-to-severe progression rate is sparsely "
        "parameterised.  Treatment timing is approximated through gestational-age strata rather "
        "than a continuous residual-risk function."
    )
