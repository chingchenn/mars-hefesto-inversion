#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 22:02:56 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 22:02:56 2026
@author: chingchen

重力穩定性與 Ra 診斷:
  上/下密度對比散點(依 misfit_tt / misfit / 四態 三種上色)
  h_liquid vs BML_thickness
  Ra 直方圖(雙門檻 Ra_c = 1e3 與 1e5)+ k 重標表 + 四態圓餅

讀檔支援 chain.jsonl(a34 起)與 chain.json(舊版,含 salvage)。
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR

# ── 設定 ─────────────────────────────────────────────────────────────────────
CHAIN_DIR = MCMC_DIR
OUT_DIR   = FIG_DIR
PREFIX    = 'chain_a52B'

plt.rcParams['font.family'] = 'DejaVu Sans'

BURNIN_FRAC = 0.3
DROP_LAST   = 1
MIN_STEPS   = 100        
SAVE_FIG    = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

D_NO_BML = 2.0           # km, D 小於此值視為「沒有 BML」(待辦 ⑥ 的污染比例)
H_MIN    = 1.0           # km, 三態分類門檻

# Ra 門檻。1e3 是古典定黏度 Rayleigh-Benard 值,1e5 是 Samuel 2021 SM S4 的下界。
# 修正 rho 後 Ra 落在 1e2-1e4 量級,已經橫跨 1e3 -> Ra_c 的選擇不再無關,兩者都要報。
RA_C_LIST = [1.0e3, 1.0e5]

# Ra ∝ 1/k 精確成立,故 k 的敏感度是純後處理: Ra(k) = Ra_stored × K_SOLID_STORED / k
K_SOLID_STORED = 4.0                    # 跑鏈時 59 用的值
K_SOLID_LIST   = [1.0, 1.6, 2.5, 4.0]   # 1.6 = Drilleau 2026 kd 後驗中位

PARAMS = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml', 'BML_thickness',
          'w_S', 'R_cmb']
POST_KEYS = ['misfit', 'misfit_tt', 'h_solid_km', 'h_liquid_km',
             'T_interface', 'upper_contrast', 'lower_contrast', 'Ra']

STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87',
               'molten': '#CD5C5C', 'no_bml': '#97795D'}
STATES = ['no_bml', 'solid', 'layered', 'molten', 'fail']
fontsize = 12

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


# ── 四態分類 ─────────────────────────────────────────────────────────────────
# a34 的 BML_thickness 先驗下界為 0,D 極小的模型會讓 h_solid 與 h_liquid 都 < 1 km
# 而被誤判成 solid,必須先切出 no_bml(待辦 ⑥)。
h_sol, h_liq, D_bml = post['h_solid_km'], post['h_liquid_km'], post['BML_thickness']
state = np.full(len(h_sol), 'fail', dtype='<U8')
state[(h_sol >= H_MIN) & (h_liq >= H_MIN)] = 'layered'
state[(h_sol <  H_MIN) & (h_liq >= H_MIN)] = 'molten'
state[(h_sol >= H_MIN) & (h_liq <  H_MIN)] = 'solid'
state[D_bml < D_NO_BML]                    = 'no_bml'
post['state'] = state

print("\nstate counts:")
for s in STATES:
    n = int((state == s).sum())
    print(f"   {s:<10s}{n:7d}  {100*n/len(state):6.2f}%")

# 待辦 ⑥: D < 2 km 若沒有單獨切出來,會有多少被誤歸成 solid
naive_solid = int(((h_sol >= H_MIN) & (h_liq < H_MIN)).sum())
true_solid  = int((state == 'solid').sum())
print(f"\n  待辦 ⑥ 污染檢查: 不切 no_bml 時 solid = {naive_solid},"
      f" 切了之後 = {true_solid}"
      f"  (污染 {naive_solid - true_solid} 個樣本)")


# ── 圖: h_liquid vs BML_thickness ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(post['h_liquid_km'], post['BML_thickness'],
                c=post['T_interface'], cmap='magma', s=4)
cb = plt.colorbar(sc, ax=ax)
cb.set_label('T_interface (K)', fontsize=fontsize)
cb.ax.tick_params(labelsize=14)
ax.set_xlabel('h_liquid (km)', fontsize=fontsize)
ax.set_ylabel('BML thickness (km)', fontsize=fontsize)
ax.tick_params(labelsize=14)
plt.tight_layout()
if SAVE_FIG: plt.savefig(os.path.join(OUT_DIR, TAG + '07_hliq_vs_D.png'), dpi=150)
plt.show()


# ── 圖: 上/下密度對比,三種上色(原本是三段幾乎相同的複製貼上)──────────────
uc, lc = post['upper_contrast'], post['lower_contrast']
msk_c = np.isfinite(uc) & np.isfinite(lc)
print(f"\ngravitational contrast: {int(msk_c.sum())}/{len(uc)} samples finite")


def plot_contrast(color_key, cbar_label, cmap='magma', suffix='', discrete=False):
    fig, ax = plt.subplots(figsize=(7, 6))
    if discrete:
        for s in ['molten', 'layered', 'solid', 'no_bml']:
            m = msk_c & (state == s)
            if m.sum():
                ax.scatter(uc[m], lc[m], c=STATE_COLOR[s], s=5, alpha=0.6, label=s)
        ax.legend(fontsize=11)
    else:
        sc = ax.scatter(uc[msk_c], lc[msk_c], c=post[color_key][msk_c],
                        cmap=cmap, s=5)
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(cbar_label, fontsize=fontsize)
        cb.ax.tick_params(labelsize=14)
    ax.axvline(0, color='k', lw=1.5, ls='--')
    ax.axhline(0, color='k', lw=1.5, ls='--')
    ax.set_xlabel('upper contrast (g/cc)', fontsize=fontsize)
    ax.set_ylabel('lower contrast (g/cc)', fontsize=fontsize)
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    if SAVE_FIG and suffix:
        plt.savefig(os.path.join(OUT_DIR, TAG + suffix + '.png'), dpi=150)
    plt.show()


if msk_c.sum():
    plot_contrast('misfit_tt', 'seismic misfit', suffix='C1_contrast_tt')
    plot_contrast('misfit',    'total misfit',   suffix='C2_contrast_total')
    plot_contrast(None,        'BML state',      suffix='C3_contrast_state',
                  discrete=True)
    # Kono 2025: BML 必須比上方地函重、比下方核輕,否則 << 4.5 Gyr 內翻轉
    print(f"  upper_contrast > 0: {100*np.mean(uc[msk_c] > 0):.1f}%"
          f"   lower_contrast > 0: {100*np.mean(lc[msk_c] > 0):.1f}%")
    both = np.mean((uc[msk_c] > 0) & (lc[msk_c] > 0))
    print(f"  兩者皆正(重力穩定): {100*both:.1f}%")
else:
    print('  沒有有限的密度對比值,跳過圖 C1-C3')


# ── 圖: Ra 直方圖(雙門檻)+ 四態圓餅 ───────────────────────────────────────
Ra = post['Ra']
ok_ra = np.isfinite(Ra) & (Ra > 0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
if ok_ra.sum():
    axes[0].hist(np.log10(Ra[ok_ra]), bins=40, color='steelblue', alpha=0.85)
    for rc, ls in zip(RA_C_LIST, ['-', '--']):
        axes[0].axvline(np.log10(rc), color='r', ls=ls, lw=1.6,
                        label=f'Ra_c = {rc:.0e}')
    axes[0].legend(fontsize=11)
else:
    axes[0].text(0.5, 0.5, 'no finite positive Ra', transform=axes[0].transAxes,
                 ha='center', va='center', fontsize=fontsize, color='gray')
axes[0].set_xlabel('log10(Ra), solid BML', fontsize=fontsize)
axes[0].set_ylabel('count', fontsize=fontsize)
axes[0].tick_params(labelsize=14)

sizes = [int((state == s).sum()) for s in STATES]
keep = [i for i, v in enumerate(sizes) if v > 0]
axes[1].pie([sizes[i] for i in keep],
            labels=[f'{STATES[i]}\n({sizes[i]})' for i in keep],
            colors=[STATE_COLOR.get(STATES[i], '#BBBBBB') for i in keep],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
axes[1].set_title('posterior BML state', fontsize=fontsize)
plt.tight_layout()
if SAVE_FIG: plt.savefig(os.path.join(OUT_DIR, TAG + '08_rayleigh_state.png'), dpi=150)
plt.show()

if ok_ra.sum():
    print(f"\nRa: n={int(ok_ra.sum())}  "
          f"range {np.nanmin(Ra[ok_ra]):.2e} - {np.nanmax(Ra[ok_ra]):.2e}  "
          f"median {np.nanmedian(Ra[ok_ra]):.2e}")
    for rc in RA_C_LIST:
        n_ab = int((Ra[ok_ra] > rc).sum())
        print(f"   above Ra_c = {rc:.0e}: {n_ab:6d}  "
              f"{100*n_ab/ok_ra.sum():6.2f}%")

    # Ra ∝ 1/k -> k 的敏感度是純後處理,不需重跑鏈
    print(f"\nk 重標(Ra ∝ 1/k, 鏈中儲存值對應 k = {K_SOLID_STORED:.1f}):")
    print(f"{'k':>6s}{'Ra median':>13s}" +
          ''.join(f'{f"> {rc:.0e}":>12s}' for rc in RA_C_LIST))
    for k in K_SOLID_LIST:
        Ra_k = Ra[ok_ra] * K_SOLID_STORED / k
        row = ''.join(f'{100*np.mean(Ra_k > rc):11.1f}%' for rc in RA_C_LIST)
        print(f'{k:6.1f}{np.median(Ra_k):13.2e}{row}')
    print('   若某個 k 讓「超過 Ra_c」的比例明顯改變 -> 該組合會翻轉 thermal_state,')
    print('   進而影響 likelihood,那就不是後處理,必須另跑一條鏈。')
else:
    print('\n沒有有限的正 Ra 值(可能全部是 molten 或 no_bml,無固態次層)')
