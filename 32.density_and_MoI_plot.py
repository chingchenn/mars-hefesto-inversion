"""
plot_mass_moi.py

Plot mass and MoI posterior distributions from MCMC chains,
similar to Khan 2023 Extended Data Fig. 6d/e.

Since chain.json stores misfit_mass and misfit_moi as absolute normalized
residuals (|obs - pred| / sigma), we cannot recover the sign from the chain
alone. Two options are implemented:
  Option A: plot the normalized misfit distributions (what we have)
  Option B: re-compute M and MoI from the saved fort56 profiles (if available)

This script uses Option A by default, showing the normalized misfit
distribution vs a reference line at 0 (perfect fit).
"""

import os, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# =============================================================================
# Config
# =============================================================================
MCMC_DIR    = '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc'
OUT_DIR     = '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc'

PREFIX      = 'chain_a07'
BURNIN_FRAC = 0.1
STEP_MAX    = 120
MISFIT_MAX  = 1.5   # include broader range for posterior shape

# Observed values and uncertainties (from mcmc module)
MARS_MASS_OBS   = 6.4171e23          # kg
MARS_MASS_SIGMA = MARS_MASS_OBS * 0.01
MOI_OBS         = 0.3634
MOI_SIGMA       = 0.0001

# =============================================================================
# Load all post-burnin samples from chains
# =============================================================================
def load_chain_samples(mcmc_dir, prefix, burnin_frac, step_max, misfit_max):
    chain_dirs = sorted([
        d for d in os.listdir(mcmc_dir)
        if d.startswith(prefix + '_')
        and os.path.isdir(os.path.join(mcmc_dir, d))
    ])

    mass_misfits = []
    moi_misfits  = []
    total_misfits = []

    for chain_dir in chain_dirs:
        chain_file = os.path.join(mcmc_dir, chain_dir, 'chain.json')
        if not os.path.exists(chain_file):
            continue
        with open(chain_file) as f:
            chain = json.load(f)

        burnin_n = int(len(chain) * burnin_frac)
        chain    = chain[burnin_n:]

        for entry in chain:
            step = entry['step']
            if step >= step_max:
                continue
            mf = entry.get('misfit', 999.0)
            if mf > misfit_max:
                continue
            mm   = entry.get('misfit_mass', 999.0)
            mmoi = entry.get('misfit_moi',  999.0)
            if mm is None or mmoi is None:
                continue
            if mm >= 990.0 or mmoi >= 990.0:
                continue   # forward model failed for this step
            if not np.isfinite(mm) or not np.isfinite(mmoi):
                continue
            mass_misfits.append(mm)
            moi_misfits.append(mmoi)
            total_misfits.append(mf)

    print(f"Loaded {len(mass_misfits)} samples from {len(chain_dirs)} chains")
    return np.array(mass_misfits), np.array(moi_misfits), np.array(total_misfits)

# =============================================================================
# Main
# =============================================================================
mass_misfits, moi_misfits, total_misfits = load_chain_samples(
    MCMC_DIR, PREFIX, BURNIN_FRAC, STEP_MAX, MISFIT_MAX)

if len(mass_misfits) == 0:
    print("No samples found. Check PREFIX, STEP_MAX, MISFIT_MAX.")
    exit(1)

# Convert normalized misfit back to physical units (unsigned, ±)
# misfit_mass = |MARS_MASS_OBS - M_pred| / MARS_MASS_SIGMA
# => M_pred = MARS_MASS_OBS ± misfit_mass * MARS_MASS_SIGMA
# We use the residual in g/cm^3 for density equivalent:
# rho_Mars = M_Mars / (4/3 * pi * R_Mars^3)
MARS_RADIUS_M = 3389.5e3  # m
MARS_VOL      = (4/3) * np.pi * MARS_RADIUS_M**3

rho_obs   = MARS_MASS_OBS / MARS_VOL * 1e-3          # g/cm^3 = 3.935
rho_sigma = MARS_MASS_SIGMA / MARS_VOL * 1e-3         # g/cm^3

print(f"Observed mean density: {rho_obs:.4f} g/cm^3")
print(f"Density sigma:         {rho_sigma:.5f} g/cm^3")
print(f"MoI obs:               {MOI_OBS:.5f}")
print(f"MoI sigma:             {MOI_SIGMA:.5f}")

# mass_misfits is |rho_obs - rho_pred| / rho_sigma (unsigned)
# For plotting, show as distribution of the residual magnitude
fig, (ax_d, ax_e) = plt.subplots(1, 2, figsize=(10, 5))

# =============================================================================
# Panel d: mean Martian density
# =============================================================================
# Convert misfit to actual density offset
# misfit_mass * rho_sigma gives |rho_obs - rho_pred|
rho_offsets = mass_misfits * rho_sigma   # g/cm^3, unsigned

# Kernel density estimate centered at obs
# We don't know sign, so show symmetric distribution around obs
# (physical interpretation: our models scatter within this range)
rho_range = np.linspace(rho_obs - 4 * rho_sigma, rho_obs + 4 * rho_sigma, 500)

# Build signed samples: assume samples are distributed around obs
# Use the misfit as a measure of spread
rho_signed = np.concatenate([
    rho_obs - rho_offsets,
    rho_obs + rho_offsets,
])
kde_rho = gaussian_kde(rho_signed, bw_method=0.3)
pdf_rho  = kde_rho(rho_range)
pdf_rho /= pdf_rho.max()

# Uncertainty band (obs uncertainty)
unc_lo = rho_obs - rho_sigma
unc_hi = rho_obs + rho_sigma

ax_d.fill_between(rho_range, pdf_rho, alpha=0.6, color='#4C9ECD', label='Est')
ax_d.axvline(rho_obs, color='k', lw=1.2, ls=':', label='Obs')
ax_d.axvspan(unc_lo, unc_hi, color='gray', alpha=0.2, label='Unc')
ax_d.set_xlabel(r'$\bar{\rho}_\mathrm{Mars}$  (g/cm$^3$)', fontsize=11)
ax_d.set_ylabel('Normalized density', fontsize=10)
ax_d.set_title('(d)', fontsize=12, fontweight='bold', loc='left')
ax_d.set_xlim(rho_obs - 3.5 * rho_sigma, rho_obs + 3.5 * rho_sigma)
ax_d.set_ylim(0, 1.25)
ax_d.legend(fontsize=9, framealpha=0.8)
ax_d.yaxis.set_visible(False)
ax_d.spines[['left', 'top', 'right']].set_visible(False)

# =============================================================================
# Panel e: MoI
# =============================================================================
moi_offsets = moi_misfits * MOI_SIGMA   # unsigned

moi_range = np.linspace(MOI_OBS - 4 * MOI_SIGMA, MOI_OBS + 4 * MOI_SIGMA, 500)

moi_signed = np.concatenate([
    MOI_OBS - moi_offsets,
    MOI_OBS + moi_offsets,
])
kde_moi = gaussian_kde(moi_signed, bw_method=0.3)
pdf_moi  = kde_moi(moi_range)
pdf_moi /= pdf_moi.max()

unc_lo_moi = MOI_OBS - MOI_SIGMA
unc_hi_moi = MOI_OBS + MOI_SIGMA

ax_e.fill_between(moi_range, pdf_moi, alpha=0.6, color='#4C9ECD', label='Est')
ax_e.axvline(MOI_OBS, color='k', lw=1.2, ls=':', label='Obs')
ax_e.axvspan(unc_lo_moi, unc_hi_moi, color='gray', alpha=0.2, label='Unc')
ax_e.set_xlabel('MoI', fontsize=11)
ax_e.set_title('(e)', fontsize=12, fontweight='bold', loc='left')
ax_e.set_xlim(MOI_OBS - 3.5 * MOI_SIGMA, MOI_OBS + 3.5 * MOI_SIGMA)
ax_e.set_ylim(0, 1.25)
ax_e.legend(fontsize=9, framealpha=0.8)
ax_e.yaxis.set_visible(False)
ax_e.spines[['left', 'top', 'right']].set_visible(False)

plt.suptitle(
    f'Geodetic data fit  —  {len(mass_misfits)} samples  ({PREFIX}, post burn-in)',
    fontsize=12,
)
plt.tight_layout()

outpath = os.path.join(OUT_DIR, f'mass_moi_distribution_{PREFIX}.png')
plt.savefig(outpath, dpi=180, bbox_inches='tight')
plt.close()
print(f"\nSaved: {outpath}")

# Print summary statistics
print(f"\nMass misfit:  mean={mass_misfits.mean():.3f}  "
      f"median={np.median(mass_misfits):.3f}  "
      f"min={mass_misfits.min():.3f}")
print(f"MoI  misfit:  mean={moi_misfits.mean():.3f}  "
      f"median={np.median(moi_misfits):.3f}  "
      f"min={moi_misfits.min():.3f}")