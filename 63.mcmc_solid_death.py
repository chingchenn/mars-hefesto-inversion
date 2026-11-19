#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 23:02:14 2026

@author: chingchen
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 23:02:14 2026
@author: chingchen

遷移診斷:每條鏈的起始/結束狀態、造訪各態的次數、最後一次出現 solid 的 step。
用途是判斷「初始化在 solid 的鏈是否遷出、遷出後是否再回來」。

刻意使用完整鏈、不套 burn-in —— burn-in 期間的遷移正是要觀察的對象。
讀檔支援 chain.jsonl(a34 起)與 chain.json(舊版,含 salvage)。
"""

import json, glob, os
import numpy as np
from mars_config import MCMC_DIR

# ── 設定 ─────────────────────────────────────────────────────────────────────
CHAIN_DIR = MCMC_DIR
PREFIX    = 'chain_a36A'

DROP_LAST = 1
MIN_STEPS = 100      # 設 400 會剔掉 chain_13(396 步)

D_NO_BML = 2.0       # km, D 小於此值視為「沒有 BML」
H_MIN    = 1.0       # km, 三態分類門檻

STATES = ['no_bml', 'solid', 'layered', 'molten', 'fail']


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
    """優先 chain.jsonl(a34+),沒有才退回 chain.json。
    注意:不能用 'chain.jso*' 這種通配符丟給 os.path.exists,它不做 glob 展開。"""
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


def classify(ch):
    """四態分類。a34 的 BML_thickness 先驗下界為 0,D 極小的模型會讓
    h_solid 與 h_liquid 都 < 1 km 而被誤判成 solid,故必須先切出 no_bml,
    否則 last_solid 這個診斷會出現假訊號。"""
    def arr(key):
        return np.array([np.nan if s.get(key) is None else float(s[key]) for s in ch])

    hs, hl = arr('h_solid_km'), arr('h_liquid_km')
    D = np.array([float(s['params']['BML_thickness']) for s in ch])

    st = np.full(len(hs), 'fail', dtype='<U8')
    st[(hs >= H_MIN) & (hl >= H_MIN)] = 'layered'
    st[(hs <  H_MIN) & (hl >= H_MIN)] = 'molten'
    st[(hs >= H_MIN) & (hl <  H_MIN)] = 'solid'
    st[D < D_NO_BML]                  = 'no_bml'
    return st


# ── 逐鏈摘要 ─────────────────────────────────────────────────────────────────
hdr = (f"{'chain':<20s}{'n':>6s}{'start':>9s}{'end':>9s}" +
       ''.join(f'{s[:7]:>8s}' for s in STATES) +
       f"{'last_solid':>12s}{'trans':>7s}")
print('\n' + '=' * len(hdr))
print(hdr)
print('=' * len(hdr))

started_solid, left_solid, returned_solid = [], [], []
total_trans = 0

for name, ch in zip(chain_names, all_chains):
    st = classify(ch)
    steps = np.array([s['step'] for s in ch])

    counts = [int((st == s).sum()) for s in STATES]
    idx = np.where(st == 'solid')[0]
    last_solid = int(steps[idx[-1]]) if len(idx) else -1
    n_trans = int((st[1:] != st[:-1]).sum())
    total_trans += n_trans

    print(f"{name:<20s}{len(st):6d}{st[0]:>9s}{st[-1]:>9s}" +
          ''.join(f'{c:8d}' for c in counts) +
          f"{last_solid:12d}{n_trans:7d}")

    if st[0] == 'solid':
        started_solid.append(name)
        first_non = np.where(st != 'solid')[0]
        if len(first_non):
            left_solid.append((name, int(steps[first_non[0]])))
            # 遷出之後是否又回到 solid
            if (st[first_non[0]:] == 'solid').any():
                back = first_non[0] + np.where(st[first_non[0]:] == 'solid')[0][0]
                returned_solid.append((name, int(steps[back])))

print('=' * len(hdr))
print(f"  last_solid = 最後一次為 solid 的 step 編號;-1 表示從未造訪 solid")
print(f"  total transitions across all chains: {total_trans}")


# ── 遷移摘要 ─────────────────────────────────────────────────────────────────
print(f"\n{'-' * 60}")
print('solid 態遷移摘要')
print('-' * 60)
if not started_solid:
    print('  沒有任何鏈以 solid 態起始')
else:
    print(f'  以 solid 起始的鏈: {len(started_solid)}  '
          f'({", ".join(started_solid)})')
    if left_solid:
        st_steps = [s for _, s in left_solid]
        print(f'  其中遷出 solid: {len(left_solid)}  '
              f'(最早 step {min(st_steps)}, 最晚 step {max(st_steps)})')
        for nm, s in left_solid:
            print(f'      {nm:<20s} 於 step {s} 首次離開 solid')
    else:
        print('  沒有任何鏈遷出 solid(全程停留)')

if returned_solid:
    print(f'\n  遷出後又回到 solid: {len(returned_solid)}')
    for nm, s in returned_solid:
        print(f'      {nm:<20s} 於 step {s} 回到 solid')
else:
    print('\n  遷出 solid 的鏈都沒有再回來')

# ── 全體狀態計數 ─────────────────────────────────────────────────────────────
pooled = np.concatenate([classify(ch) for ch in all_chains])
print(f"\n{'-' * 60}")
print(f'全體(完整鏈,未套 burn-in) n = {len(pooled)}')
print('-' * 60)
for s in STATES:
    n = int((pooled == s).sum())
    print(f'  {s:<10s}{n:8d}  {100*n/len(pooled):6.2f}%')