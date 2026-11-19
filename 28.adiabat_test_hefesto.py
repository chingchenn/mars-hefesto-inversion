"""
benchmark_adiabat.py

比較兩種建絕熱溫度剖面的方法速度：
  Method A: 分段NPS (iensem=-2)，3段
  Method B: NPT scan（完整掃描，失敗點用插值，不跳過任何區間）

用法：
  python benchmark_adiabat.py
"""

import os, sys, shutil, time
import numpy as np
import subprocess
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script

# ── 把你的mcmc模組加進path ─────────────────────────────────
sys.path.insert(0, HERE)
from config import HEFESTO_MAIN, PAR_DIR
from mars_config import FIG_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

# ── 測試參數 ───────────────────────────────────────────────
TEST_PARAMS = {
    'T_lit':   1952.92,
    'P_lit':   6.0755,
    'Mg#':     0.7899,
    'T_bml':   2500.0,
    'Mg#_bml': 0.7,
}
P_MAX_GPA = 19.0
T_SURF    = 220.0

WORK_DIR = '/tmp/benchmark_adiabat'
os.makedirs(WORK_DIR, exist_ok=True)

# ── Composition ────────────────────────────────────────────
YM_BASE = {
    'Si': 4.01931, 'Mg': 4.08235, 'Fe': 1.08599,
    'Ca': 0.27259, 'Al': 0.37376, 'Na': 0.10105, 'Cr': 0.06146,
}
MGFE_TOTAL   = YM_BASE['Mg'] + YM_BASE['Fe']
FIXED_PARAMS = {
    'Si': YM_BASE['Si'], 'Ca': YM_BASE['Ca'], 'Al': YM_BASE['Al'],
    'Na': YM_BASE['Na'], 'Cr': YM_BASE['Cr'],
}

CONTROL_PHASES = """\
phase plg
0
an
ab
phase sp
1
sp
hc
smag
picr
phase opx
0
en
fs
mgts
odi
phase c2c
0
mgc2
fec2
phase cpx
1
di
he
cen
cats
jd
acm
phase wo
1
wo
phase pwo
1
pwo
phase gt
0
py
al
gr
mgmj
namj
andr
knor
phase cpv
0
capv
phase ol
1
fo
fa
phase wa
0
mgwa
fewa
phase ri
0
mgri
feri
phase il
0
mgil
feil
co
hem
esk
phase pv
0
mgpv
fepv
alpv
hepv
hlpv
fapv
crpv
phase ppv
0
mppv
fppv
appv
hppv
cppv
phase cf
0
mgcf
fecf
nacf
hmag
crcf
phase nal
0
mnal
fnal
nnal
phase mw
0
pe
wu
wuls
anao
mag
phase qtz
0
qtz
phase coes
0
coes
phase st
0
st
phase apbo
0
apbo
phase ky
0
ky
phase neph
0
neph
phase fea
0
fea
phase feg
0
feg
phase fee
0
fee
"""

def composition_from_params(params):
    mgnum = params['Mg#']
    Mg    = mgnum * MGFE_TOTAL
    Fe    = (1.0 - mgnum) * MGFE_TOTAL
    return {
        'Si': FIXED_PARAMS['Si'], 'Mg': Mg, 'Fe': Fe,
        'Ca': FIXED_PARAMS['Ca'], 'Al': FIXED_PARAMS['Al'],
        'Na': FIXED_PARAMS['Na'], 'Cr': FIXED_PARAMS['Cr'],
        'T_lit': params['T_lit'], 'P_lit': params['P_lit'],
    }

def compute_oxygen(p):
    return (2.0*p['Si'] + p['Mg'] + p['Fe'] + p['Ca'] +
            1.5*p['Al'] + 0.5*p['Na'] + 1.5*p['Cr'])

def make_control_lines(p, O, line1):
    return [
        line1, "8,2,4,2", "oxides",
        f"Si      {p['Si']:.5f}     {p['Si']:.5f}    0",
        f"Mg      {p['Mg']:.5f}     {p['Mg']:.5f}    0",
        f"Fe      {p['Fe']:.5f}     {p['Fe']:.5f}    0",
        f"Ca      {p['Ca']:.5f}     {p['Ca']:.5f}    0",
        f"Al      {p['Al']:.5f}     {p['Al']:.5f}    0",
        f"Na      {p['Na']:.5f}     {p['Na']:.5f}    0",
        f"Cr      {p['Cr']:.5f}     {p['Cr']:.5f}    0",
        f" O      {O:.5f}     {O:.5f}    0",
        "1,1,1,1", PAR_DIR, "73", CONTROL_PHASES,
    ]

def run_single(run_dir, control_lines, ad_in=None, timeout=600):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'control'), 'w') as f:
        f.write('\n'.join(control_lines))
    if ad_in:
        with open(os.path.join(run_dir, 'ad.in'), 'w') as f:
            f.write(ad_in)
    dst = os.path.join(run_dir, 'main')
    if not os.path.exists(dst):
        shutil.copy2(HEFESTO_MAIN, dst)
        os.chmod(dst, 0o755)
    with open(os.path.join(run_dir, 'hefesto.log'), 'w') as log:
        try:
            subprocess.run(['./main'], cwd=run_dir,
                           stdout=log, stderr=log, timeout=timeout)
        except Exception:
            return None
    f56 = os.path.join(run_dir, 'fort.56')
    if not os.path.exists(f56) or os.path.getsize(f56) == 0å:
        return None
    return f56

def read_f56(path):
    try:
        with open(path) as f:
            f.readline()
            cols = f.readline().split()
        df = pd.read_csv(path, sep=r'\s+', skiprows=2, names=cols)
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['P(GPa)', 'T(K)', 'S(J/g/K)'])
        if df.empty:
            return None
        return df
    except Exception:
        return None

# ============================================================
# Step 1: NPT @ (P_lit, T_lit) → S_lit
# ============================================================
p     = composition_from_params(TEST_PARAMS)
O     = compute_oxygen(p)
T_lit = TEST_PARAMS['T_lit']
P_lit = TEST_PARAMS['P_lit']

print("=" * 55)
print("Step 1: NPT single point")
t0    = time.time()
dir1  = os.path.join(WORK_DIR, 's1_npt')
line1 = f"{P_lit:.4f},{P_lit:.4f},1,{T_lit:.2f},{T_lit:.2f},0,0,0,0"
f56_1 = run_single(dir1, make_control_lines(p, O, line1))
t1_elapsed = time.time() - t0

if f56_1 is None:
    print("  Step 1 FAILED")
    sys.exit(1)
df1   = read_f56(f56_1)
S_lit = float(df1['S(J/g/K)'].iloc[0])
print(f"  S_lit = {S_lit:.6f} J/g/K  ({t1_elapsed:.1f}s)")

# ============================================================
# Method A: 分段NPS（3段，分段點在相變位置）
# ============================================================
print("\n" + "=" * 55)
print("Method A: Segmented NPS (3 segments)")
print(f"  Segments: {P_lit:.2f}→14.0, 14.0→18.0, 18.0→{P_MAX_GPA:.1f} GPa")
t0 = time.time()

segments    = [(P_lit, 14.0), (14.0, 18.0), (18.0, P_MAX_GPA)]
P_seg_list  = [np.array([P_lit])]
T_seg_list  = [np.array([T_lit])]
T_seg_start = T_lit
seg_ok      = True

for i, (P_start, P_end) in enumerate(segments):
    P_end    = min(P_end, P_MAX_GPA)
    dir_seg  = os.path.join(WORK_DIR, f'method_a_seg{i}')
    line_seg = (f"{P_start:.4f},{P_end:.4f},20,"
                f"{S_lit:.6f},{S_lit:.6f},0,-2,"
                f"{T_seg_start:.2f},0")
    t_seg_0      = time.time()
    f56_seg      = run_single(dir_seg, make_control_lines(p, O, line_seg))
    t_seg_elapsed = time.time() - t_seg_0

    if f56_seg is None:
        print(f"  Seg{i} ({P_start:.1f}→{P_end:.1f} GPa) FAILED"
              f"  ({t_seg_elapsed:.1f}s)")
        seg_ok = False
        break

    df_seg = read_f56(f56_seg)
    if df_seg is None or len(df_seg) < 3:
        print(f"  Seg{i} ({P_start:.1f}→{P_end:.1f} GPa) read failed"
              f"  ({t_seg_elapsed:.1f}s)")
        seg_ok = False
        break

    P_s = df_seg['P(GPa)'].values
    T_s = df_seg['T(K)'].values
    print(f"  Seg{i} ({P_start:.1f}→{P_end:.1f} GPa): "
          f"T={T_s[0]:.1f}→{T_s[-1]:.1f} K  "
          f"({len(P_s)} pts, {t_seg_elapsed:.1f}s)")

    P_seg_list.append(P_s[1:])
    T_seg_list.append(T_s[1:])
    T_seg_start = float(T_s[-1])

t_seg = time.time() - t0

if seg_ok:
    P_seg_out = np.concatenate(P_seg_list)
    T_seg_out = np.concatenate(T_seg_list)
    print(f"  Segmented NPS OK  ({t_seg:.1f}s total)")
    print(f"  T: {T_seg_out[0]:.1f}K @ {P_seg_out[0]:.2f}GPa"
          f"  →  {T_seg_out[-1]:.1f}K @ {P_seg_out[-1]:.2f}GPa")
else:
    P_seg_out, T_seg_out = None, None
    print(f"  Segmented NPS FAILED  ({t_seg:.1f}s)")

# ============================================================
# Method B: NPT scan（完整掃描，失敗點用插值）
# ============================================================
print("\n" + "=" * 55)
print("Method B: NPT scan (full range, interpolate failed points)")
t0 = time.time()

P_scan         = np.array([P_lit, 9.0, 12.0, 15.0, 17.0, P_MAX_GPA])
T_guess        = T_lit
P_adiab_list   = [P_lit]
T_adiab_list   = [T_lit]
n_npt_calls    = 0
n_interpolated = 0
scan_ok        = True

for i_p, P_target in enumerate(P_scan[1:]):

    # 嘗試不同dT，從大到小
    d_lo, d_hi, T_lo, T_hi = None, None, None, None
    for dT in [20.0, 10.0, 5.0, 2.0]:
        T_lo = T_guess - dT
        T_hi = T_guess + dT

        dir_lo  = os.path.join(WORK_DIR, f'scan_{i_p}_lo_dT{dT:.0f}')
        dir_hi  = os.path.join(WORK_DIR, f'scan_{i_p}_hi_dT{dT:.0f}')
        line_lo = (f"{P_target:.4f},{P_target:.4f},1,"
                   f"{T_lo:.2f},{T_lo:.2f},0,0,0,0")
        line_hi = (f"{P_target:.4f},{P_target:.4f},1,"
                   f"{T_hi:.2f},{T_hi:.2f},0,0,0,0")

        f_lo = run_single(dir_lo, make_control_lines(p, O, line_lo))
        f_hi = run_single(dir_hi, make_control_lines(p, O, line_hi))
        n_npt_calls += 2

        d_lo = read_f56(f_lo) if f_lo else None
        d_hi = read_f56(f_hi) if f_hi else None

        if d_lo is not None and d_hi is not None:
            print(f"    P={P_target:.2f} GPa: dT={dT:.0f}K OK")
            break
        else:
            print(f"    P={P_target:.2f} GPa: dT={dT:.0f}K failed, retry...")

    if d_lo is None or d_hi is None:
        # 所有dT都失敗 → 用前一個點的梯度外插，繼續往下跑
        if len(T_adiab_list) >= 2:
            dT_dP     = ((T_adiab_list[-1] - T_adiab_list[-2]) /
                         (P_adiab_list[-1] - P_adiab_list[-2]))
            T_adiab_p = (T_adiab_list[-1] +
                         dT_dP * (P_target - P_adiab_list[-1]))
        else:
            T_adiab_p = T_guess
        T_adiab_p = float(np.clip(T_adiab_p, 800, 4000))
        print(f"    P={P_target:.2f} GPa: all dT failed → "
              f"interpolated T={T_adiab_p:.1f}K")
        P_adiab_list.append(P_target)
        T_adiab_list.append(T_adiab_p)
        T_guess = T_adiab_p
        n_interpolated += 1
        continue

    S_lo = float(d_lo['S(J/g/K)'].iloc[0])
    S_hi = float(d_hi['S(J/g/K)'].iloc[0])

    if abs(S_hi - S_lo) < 1e-8:
        T_adiab_p = T_guess
    else:
        T_adiab_p = (T_lo + (S_lit - S_lo) *
                     (T_hi - T_lo) / (S_hi - S_lo))

    T_adiab_p = float(np.clip(T_adiab_p, 800, 4000))
    P_adiab_list.append(P_target)
    T_adiab_list.append(T_adiab_p)
    T_guess = T_adiab_p

t_scan = time.time() - t0

P_scan_out = np.array(P_adiab_list)
T_scan_out = np.array(T_adiab_list)
print(f"  NPT scan done  ({t_scan:.1f}s,  "
      f"{n_npt_calls} NPT calls, "
      f"{n_interpolated} interpolated pts)")
print(f"  T: {T_scan_out[0]:.1f}K @ {P_scan_out[0]:.2f}GPa"
      f"  →  {T_scan_out[-1]:.1f}K @ {P_scan_out[-1]:.2f}GPa")

# ============================================================
# 結果比較
# ============================================================
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  Step1 NPT:         {t1_elapsed:.1f}s")
print(f"  Method A seg NPS:  {t_seg:.1f}s"
      + (" (FAILED)" if P_seg_out is None else ""))
print(f"  Method B NPT scan: {t_scan:.1f}s  "
      f"({n_npt_calls} calls, {n_interpolated} interpolated)")

if P_seg_out is not None:
    ratio = t_seg / t_scan if t_scan > 0 else float('inf')
    if ratio > 1:
        print(f"\n  → NPT scan is {ratio:.1f}x FASTER than segmented NPS")
    else:
        print(f"\n  → Segmented NPS is {1/ratio:.1f}x FASTER than NPT scan")

# ── 畫圖 ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
if P_seg_out is not None:
    ax.plot(P_seg_out, T_seg_out, 'b-o', ms=4,
            label=f'Seg NPS ({t_seg:.0f}s)')
ax.plot(P_scan_out, T_scan_out, 'r-s', ms=4,
        label=f'NPT scan ({t_scan:.0f}s)')

# 標記插值點
if n_interpolated > 0:
    interp_mask = []
    # 找出哪些點是插值的（在P_adiab_list裡但對應dT全失敗的點）
    # 用顏色標記
    ax.scatter(P_scan_out, T_scan_out, c='r', s=20, zorder=5)

ax.set_xlabel('Pressure (GPa)')
ax.set_ylabel('Temperature (K)')
ax.set_title('Adiabatic T(P)')
ax.legend()
ax.grid(True, alpha=0.3)

ax2 = axes[1]
if P_seg_out is not None:
    T_seg_interp = np.interp(P_scan_out, P_seg_out, T_seg_out)
    dT_diff      = T_scan_out - T_seg_interp
    ax2.plot(P_scan_out, dT_diff, 'g-o', ms=4)
    ax2.axhline(0, color='k', lw=1, ls='--')
    ax2.set_xlabel('Pressure (GPa)')
    ax2.set_ylabel('ΔT = NPT_scan − Seg_NPS (K)')
    ax2.set_title(f'T difference\n'
                  f'mean={dT_diff.mean():.1f}K  '
                  f'max={dT_diff.max():.1f}K  '
                  f'min={dT_diff.min():.1f}K')
    ax2.grid(True, alpha=0.3)
    print(f"\n  T difference (NPT scan - Seg NPS):")
    print(f"    mean = {dT_diff.mean():.2f} K")
    print(f"    max  = {dT_diff.max():.2f} K")
    print(f"    min  = {dT_diff.min():.2f} K")
else:
    ax2.text(0.5, 0.5, 'Seg NPS failed\nno comparison',
             ha='center', va='center', transform=ax2.transAxes,
             fontsize=12)

out_fig = os.path.join(FIG_DIR, 'adiabat_comparison.png')
plt.tight_layout()
if SAVE_FIG: plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f"\n  Figure saved: {out_fig}")