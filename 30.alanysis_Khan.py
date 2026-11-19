# 30.alanysis_Khan.py
# LSL structure statistics of the Khan 2023 models:
# LSL top depth, true CMB depth, LSL thickness, and density stability of
#   upper interface: rho_lsl_top - rho_mantle_bottom (> 0 = stable, LSL heavier than mantle)
#   lower interface: rho_core_top - rho_lsl_bottom   (> 0 = stable, core heavier than LSL)
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mars_config import KHAN_MODEL_DIR

plt.rcParams['font.family'] = 'DejaVu Sans'

files = sorted(glob.glob(f'{KHAN_MODEL_DIR}/Model_*.txt'))
print(f"Found {len(files)} model files")

rows = []   # lsl_top, true_cmb, rho_mantle_bottom, rho_lsl_top, rho_lsl_bottom, rho_core_top
for fpath in files:
    data  = np.loadtxt(fpath, comments='#')
    depth = data[:, 0]
    Vs    = data[:, 2]
    rho   = data[:, 3]
    solid  = Vs > 0.01
    liquid = Vs < 0.01
    if solid.sum() < 5 or liquid.sum() < 5:
        continue

    # LSL top = first Vs = 0 point; mantle bottom = the solid point just above it
    first_liq = np.where(liquid)[0][0]
    if first_liq == 0:
        continue

    # true CMB = largest density jump inside the liquid part, preferring a jump > 1 g/cc
    # at (nearly) the same depth (|ddepth| < 1 km); otherwise the largest jump
    liq_depth = depth[liquid]
    liq_rho   = rho[liquid]
    drho      = np.diff(liq_rho)
    disc      = (np.abs(np.diff(liq_depth)) < 1.0) & (drho > 1.0)
    k = np.argmax(drho * disc) if disc.any() else np.argmax(drho)

    rows.append([liq_depth[0], liq_depth[k + 1], rho[first_liq - 1], liq_rho[0], liq_rho[k], liq_rho[k + 1]])

lsl_top, true_cmb, rho_man_bot, rho_lsl_t, rho_lsl_b, rho_core_t = np.array(rows, float).T
n = len(rows)
print(f"Successfully parsed: {n} models\n")

thickness    = true_cmb - lsl_top
upper_cont   = rho_lsl_t - rho_man_bot    # LSL top - mantle bottom
lower_cont   = rho_core_t - rho_lsl_b     # core top - LSL bottom
upper_stable = upper_cont > 0
lower_stable = lower_cont > 0
fully_stable = upper_stable & lower_stable

# ── summary ──────────────────────────────────────────────────────────────────
print("=" * 55)
print("LSL DEPTH STATISTICS")
print("=" * 55)
print(f"  LSL top depth   : {lsl_top.mean():.1f} ± {lsl_top.std():.1f} km  [{lsl_top.min():.0f} ~ {lsl_top.max():.0f}]")
print(f"  True CMB depth  : {true_cmb.mean():.1f} ± {true_cmb.std():.1f} km  [{true_cmb.min():.0f} ~ {true_cmb.max():.0f}]")
print(f"  LSL thickness   : {thickness.mean():.1f} ± {thickness.std():.1f} km  [{thickness.min():.0f} ~ {thickness.max():.0f}]")

print("\n" + "=" * 55)
print("DENSITY STATISTICS  [g/cm³]")
print("=" * 55)
print(f"  rho mantle bottom : {rho_man_bot.mean():.4f} ± {rho_man_bot.std():.4f}")
print(f"  rho LSL top       : {rho_lsl_t.mean():.4f} ± {rho_lsl_t.std():.4f}")
print(f"  rho LSL bottom    : {rho_lsl_b.mean():.4f} ± {rho_lsl_b.std():.4f}")
print(f"  rho core top      : {rho_core_t.mean():.4f} ± {rho_core_t.std():.4f}")

print("\n" + "=" * 55)
print("DENSITY STABILITY")
print("=" * 55)
print(f"  Upper interface (LSL_top - mantle_bottom):")
print(f"    mean contrast : {upper_cont.mean():.4f} g/cm³")
print(f"    stable (>0)   : {upper_stable.sum()} / {n} ({upper_stable.mean()*100:.1f}%)")
print(f"    unstable (<0) : {(~upper_stable).sum()} / {n} ({(~upper_stable).mean()*100:.1f}%)")
if (~upper_stable).any():
    print(f"    worst case    : {upper_cont.min():.4f} g/cm³")
print(f"\n  Lower interface (core_top - LSL_bottom):")
print(f"    mean contrast : {lower_cont.mean():.4f} g/cm³")
print(f"    stable (>0)   : {lower_stable.sum()} / {n} ({lower_stable.mean()*100:.1f}%)")
print(f"    unstable (<0) : {(~lower_stable).sum()} / {n} ({(~lower_stable).mean()*100:.1f}%)")
print(f"\n  Fully stable (both interfaces):")
print(f"    {fully_stable.sum()} / {n} ({fully_stable.mean()*100:.1f}%)")

print("\n" + "=" * 55)
print("SUGGESTED STABILITY THRESHOLD")
print("=" * 55)
print(f"  Upper contrast 5th  percentile: {np.percentile(upper_cont, 5):.4f} g/cm³")
print(f"  Upper contrast 25th percentile: {np.percentile(upper_cont, 25):.4f} g/cm³")
print(f"  -> If threshold = 0.0  : accept {upper_stable.mean()*100:.1f}% of Khan models")
for thr in [-0.05, -0.10, -0.15]:
    print(f"  -> If threshold = {thr:+.2f}: accept {(upper_cont > thr).mean()*100:.1f}% of Khan models")

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(lsl_top, bins=30, color='steelblue', edgecolor='white')
ax1.axvline(lsl_top.mean(), color='red', lw=2, label=f'mean={lsl_top.mean():.0f} km')
ax1.set_xlabel('LSL top depth [km]')
ax1.set_ylabel('Count')

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(thickness, bins=30, color='darkorange', edgecolor='white')
ax2.axvline(thickness.mean(), color='red', lw=2, label=f'mean={thickness.mean():.0f} km')
ax2.set_xlabel('LSL thickness [km]')

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(rho_lsl_t,   bins=30, color='seagreen', edgecolor='white', label='LSL top',       alpha=0.7)
ax3.hist(rho_man_bot, bins=30, color='brown',    edgecolor='white', label='Mantle bottom', alpha=0.7)
ax3.set_xlabel('Density [g/cm³]')

ax4 = fig.add_subplot(gs[1, 0])
neg = upper_cont < 0
ax4.hist(upper_cont[~neg], bins=30, color='steelblue', edgecolor='white', label=f'stable ({(~neg).sum()})')
ax4.hist(upper_cont[neg],  bins=15, color='salmon',    edgecolor='white', label=f'unstable ({neg.sum()})')
ax4.axvline(0, color='red', lw=2, ls='--')
ax4.set_xlabel('ρ_LSL_top − ρ_mantle_bottom [g/cm³]')

ax5 = fig.add_subplot(gs[1, 1])
neg2 = lower_cont < 0
ax5.hist(lower_cont[~neg2], bins=30, color='steelblue', edgecolor='white', label=f'stable ({(~neg2).sum()})')
if neg2.any():
    ax5.hist(lower_cont[neg2], bins=10, color='salmon', edgecolor='white', label=f'unstable ({neg2.sum()})')
ax5.axvline(0, color='red', lw=2, ls='--')
ax5.set_xlabel('ρ_core_top − ρ_LSL_bottom [g/cm³]')

# stability map: green = fully stable, red = unstable
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(upper_cont, lower_cont, c=np.where(fully_stable, 'green', 'red'), alpha=0.4, s=10)
ax6.axvline(0, color='black', lw=1, ls='--')
ax6.axhline(0, color='black', lw=1, ls='--')
ax6.set_xlabel('Upper contrast [g/cm³]')
ax6.set_ylabel('Lower contrast [g/cm³]')

for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.legend(fontsize=8)
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.tick_params(labelsize=14)
plt.show()
