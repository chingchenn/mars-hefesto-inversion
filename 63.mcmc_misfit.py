#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 22:19:03 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 22:19:03 2026
@author: chingchen

misfit 與 BML 狀態的關係:
  圖A  misfit vs 三態(h_liquid 散點 + total/tt 直方圖)
  圖L  misfit 箱型圖,依 melt state 與 thermal state 分組
  圖F  misfit 分量拆解(molten 相對 layered 的優劣)
讀檔支援 chain.jsonl(a34 起)與 chain.json(舊版,含 salvage)。
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR

# ── 設定 ─────────────────────────────────────────────────────────────────────
CHAIN_DIR = MCMC_DIR
OUT_DIR   = FIG_DIR
PREFIX    = 'chain_a52A'

plt.rcParams['font.family'] = 'DejaVu Sans'

BURNIN_FRAC = 0.3
DROP_LAST   = 1
MIN_STEPS   = 100        # 設 400 會剔掉 chain_13(396 步)
RA_C        = 1.0e5      # 跟 59 的 _RA_C 一致
SAVE_FIG    = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

# a34 先驗。這支腳本並未實際用到(圖A/L/F 都不需要先驗範圍),
# 保留純粹作為該輪 run 的紀錄,避免各腳本之間的值不一致。
PRIOR = {
    'T_lit':         (1000.0, 2600.0),
    'P_lit':         (   1.4,    9.0),
    'Mg#':           (  0.50,   0.90),
    'T_core':        (1800.0, 3600.0),
    'Mg#_bulk_bml':  (  0.20,   0.80),
    'BML_thickness': (   0.0,  400.0),
}

PARAMS = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml', 'BML_thickness']
PARAMS = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml', 'BML_thickness',
          'w_S', 'R_cmb']

# 這支腳本實際用到的 record 欄位
POST_KEYS = ['misfit', 'misfit_tt', 'misfit_solidus', 'mass_sigma', 'moi_sigma',
             'h_solid_km', 'h_liquid_km']

STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}
TPAL        = {'molten': '#CD5C5C', 'conductive': '#4E6E8E',
               'convective': '#7B9E87', '': '#BBBBBB'}
bwith, fontsize = 1.5, 12

os.makedirs(OUT_DIR, exist_ok=True)
TAG = f'{PREFIX}_' if PREFIX else ''


# ── chain 載入: chain.jsonl(逐行) + chain.json(salvage) ─────────────────────
def load_chain_json(path):
    if not os.path.exists(path):
        return None, 'missing'
    with open(path) as fh:
        text = fh.read()
    if not text.strip():
        return None, 'empty'
    try:
        return json.loads(text), 'ok'
    except json.JSONDecodeError:
        pass
    dec, i, recs = json.JSONDecoder(), text.find('[') + 1, []
    while True:
        while i < len(text) and text[i] in ' \t\r\n,':
            i += 1
        if i >= len(text) or text[i] == ']':
            break
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        recs.append(obj)
    return (recs, 'salvaged') if recs else (None, 'unparseable')


def load_chain_jsonl(path):
    """逐行讀;只有最後一行可能寫到一半,壞掉就丟棄該行"""
    if not os.path.exists(path):
        return None, 'missing'
    recs, n_bad = [], 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                n_bad += 1
                break
    if not recs:
        return None, 'empty'
    return recs, ('ok' if n_bad == 0 else 'truncated_last_line')


def load_chain_dir(chain_dir):
    jsonl_path = os.path.join(chain_dir, 'chain.jsonl')
    if os.path.exists(jsonl_path):
        return load_chain_jsonl(jsonl_path)
    return load_chain_json(os.path.join(chain_dir, 'chain.json'))


chain_dirs = sorted(glob.glob(os.path.join(CHAIN_DIR, f'{PREFIX}*')))
all_chains, chain_names, load_report = [], [], []

for dd in chain_dirs:
    name = os.path.basename(dd)
    ch, status = load_chain_dir(dd)
    n_raw = 0 if ch is None else len(ch)
    if ch is not None and DROP_LAST > 0:
        ch = ch[:-DROP_LAST] if len(ch) > DROP_LAST else []
    if ch is None or len(ch) < MIN_STEPS:
        load_report.append((name, status, n_raw, 'DROPPED'))
        continue
    all_chains.append(ch)
    chain_names.append(name)
    load_report.append((name, status, n_raw, f'kept {len(ch)}'))

print(f"{'chain':<20s} {'status':<18s} {'n_raw':>7s}  action")
for name, status, n_raw, action in load_report:
    print(f"{name:<20s} {status:<18s} {n_raw:>7d}  {action}")
print(f"\nusable chains: {len(all_chains)}")
if not all_chains:
    raise SystemExit(
        f"no usable chains for prefix '{PREFIX}'.\n"
        f"  檢查: (1) {CHAIN_DIR} 下是否有 chain.jsonl\n"
        f"        (2) MIN_STEPS={MIN_STEPS} 是否高於實際鏈長")


# ── 後驗(逐鏈套 burn-in,長度不必相同) ───────────────────────────────────────
samples, chain_id = [], []
for c, ch in enumerate(all_chains):
    b = int(len(ch) * BURNIN_FRAC)
    samples.extend(ch[b:])
    chain_id.extend([c] * (len(ch) - b))
chain_id = np.array(chain_id)
print(f"post burn-in samples: {len(samples)}")
if not samples:
    raise SystemExit("burn-in 之後沒有樣本,降低 BURNIN_FRAC 或等鏈跑更長")


def col(key):
    return np.array([np.nan if s.get(key) is None else float(s.get(key, np.nan))
                     for s in samples])


post = {p: np.array([s['params'][p] for s in samples], dtype=float) for p in PARAMS}
for k in POST_KEYS:
    post[k] = col(k)
post['thermal_state'] = np.array([s.get('thermal_state', '') for s in samples])

h_sol, h_liq = post['h_solid_km'], post['h_liquid_km']
state = np.full(len(h_sol), 'fail', dtype='<U8')
state[(h_sol >= 1.0) & (h_liq >= 1.0)] = 'layered'
state[(h_sol <  1.0) & (h_liq >= 1.0)] = 'molten'
state[(h_sol >= 1.0) & (h_liq <  1.0)] = 'solid'
post['state'] = state
print("state counts:", {s: int((state == s).sum())
                        for s in ['solid', 'layered', 'molten', 'fail']})

mis, tt = post['misfit'], post['misfit_tt']


# ── 圖A: misfit vs 三態 ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for name, c in STATE_COLOR.items():
    sel = (state == name) & np.isfinite(mis)
    if sel.sum() == 0:
        continue
    axes[1].axvline(np.median(mis[sel]), color=c, ls='--', lw=1.5)
    axes[0].scatter(h_liq[sel], mis[sel], s=8, alpha=0.4, color=c,
                    label=f'{name} (n={int(sel.sum())})')
    axes[1].hist(mis[sel], bins=30, histtype='step', lw=2, color=c,
                 density=True, label=name)
    s2 = sel & np.isfinite(tt)
    if s2.sum():
        axes[2].hist(tt[s2], bins=30, histtype='step', lw=2, color=c,
                     density=True, label=name)
axes[0].set_xlabel('h_liquid (km)', fontsize=14)
axes[0].set_ylabel('total misfit', fontsize=14)
axes[1].set_xlabel('total misfit', fontsize=14)
axes[1].set_ylabel('density', fontsize=14)
axes[2].set_xlabel('travel-time misfit only', fontsize=14)
axes[2].set_ylabel('density', fontsize=14)
for a in axes:
    a.tick_params(labelsize=14); a.legend(fontsize=10)
plt.tight_layout()
if SAVE_FIG: plt.savefig(os.path.join(OUT_DIR, TAG + 'A_misfit_state.png'), dpi=150)
plt.show()

print(f"\n{'state':8s} {'n':>6s} {'min':>8s} {'q25':>8s} {'median':>8s} "
      f"{'tt_min':>8s} {'tt_med':>8s}")
for name in STATE_COLOR:
    sel = (state == name) & np.isfinite(mis)
    if sel.sum() == 0:
        continue
    v = mis[sel]
    u = tt[sel & np.isfinite(tt)]
    tt_min = f'{np.min(u):8.3f}' if u.size else f"{'--':>8s}"
    tt_med = f'{np.median(u):8.3f}' if u.size else f"{'--':>8s}"
    print(f"{name:8s} {int(sel.sum()):6d} {np.min(v):8.3f} "
          f"{np.percentile(v,25):8.3f} {np.median(v):8.3f} {tt_min}{tt_med}")


# ── 圖L: misfit 箱型圖(melt state / thermal state)────────────────────────────
def _boxplot(ax, data, labels, **kw):
    # tick_labels 是 matplotlib >= 3.9 的名稱,舊版是 labels
    try:
        return ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kw)


ts = post['thermal_state']
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, key, pal, ttl in [
        (axes[0], state, STATE_COLOR, 'melt state (h_solid / h_liquid)'),
        (axes[1], ts, TPAL, f'thermal state (Ra vs Ra_c={RA_C:.0e})')]:
    labs = [s for s in pal if ((key == s) & np.isfinite(mis)).sum() > 5]
    data = [mis[(key == s) & np.isfinite(mis)] for s in labs]
    if not data:
        ax.text(0.5, 0.5, 'no group with n > 5', transform=ax.transAxes,
                ha='center', va='center', fontsize=fontsize, color='gray')
    else:
        bp = _boxplot(ax, data,
                      [f'{s or "other"}\n(n={len(d)})' for s, d in zip(labs, data)],
                      patch_artist=True, showfliers=False, widths=.55)
        for patch, s in zip(bp['boxes'], labs):
            patch.set_facecolor(pal[s]); patch.set_alpha(.75)
        for i, d in enumerate(data):
            ax.scatter(np.random.normal(i+1, .06, len(d)), d, s=2, alpha=.12, c='k')
    ax.set_ylabel('total misfit', fontsize=fontsize)
    ax.set_title(ttl, fontsize=fontsize)
    ax.tick_params(labelsize=14)
    for sp in ['top', 'bottom', 'left', 'right']:
        ax.spines[sp].set_linewidth(bwith)
plt.tight_layout()
if SAVE_FIG: plt.savefig(os.path.join(OUT_DIR, TAG + 'L_misfit_vs_state.png'), dpi=150)
plt.show()

print(f"\n{'thermal_state':<14s}{'n':>7s}{'misfit med':>12s}{'tt med':>10s}")
for s in TPAL:
    m_ = (ts == s) & np.isfinite(mis)
    if m_.sum():
        print(f'{s or "other":<14s}{int(m_.sum()):7d}'
              f'{np.nanmedian(mis[m_]):12.4f}{np.nanmedian(tt[m_]):10.4f}')
print('   兩組 misfit 若完全重疊 -> Ra_C=1e5 這個有爭議的選擇對擬合無影響（好論證）')


# ── misfit 分量拆解 + 圖F ────────────────────────────────────────────────────
COMPS = ['misfit_tt', 'misfit_solidus', 'mass_sigma', 'moi_sigma', 'misfit']
n_layered = int((state == 'layered').sum())
n_molten  = int((state == 'molten').sum())

print(f"\n{'component':<16s}{'layered':>10s}{'molten':>10s}{'layered adv':>13s}")
if n_layered == 0 or n_molten == 0:
    print(f'   layered n={n_layered}, molten n={n_molten} -> 無法比較')
else:
    for name in COMPS:
        v = post[name]
        row = {}
        for s in ('layered', 'molten'):
            sel = (state == s) & np.isfinite(v)
            row[s] = np.nanmean(v[sel]) if sel.sum() else np.nan
        adv = row['molten'] - row['layered']
        mark = '  <<<' if abs(adv) > 0.3 else ''
        print(f"{name:<16s}{row['layered']:10.4f}{row['molten']:10.4f}{adv:+13.4f}{mark}")
    print("   positive = layered fits better;  tt is the primary data, "
          "MoI carries a hardcoded 2.7-sigma bias correction")

    names = ['misfit_tt', 'misfit_solidus', 'mass_sigma', 'moi_sigma']
    advs = []
    for name in names:
        v = post[name]
        advs.append(np.nanmean(v[(state == 'molten') & np.isfinite(v)]) -
                    np.nanmean(v[(state == 'layered') & np.isfinite(v)]))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, advs, color=['tab:red' if a < 0 else 'tab:green' for a in advs])
    ax.axvline(0, color='k', lw=1.2)
    ax.set_xlabel('mean(molten) - mean(layered)   [positive = layered better]',
                  fontsize=fontsize)
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    if SAVE_FIG: plt.savefig(os.path.join(OUT_DIR, TAG + 'F_misfit_decomp.png'), dpi=150)
    plt.show()