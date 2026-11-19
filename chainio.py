#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chainio.py  —  把 MCMC chain.jsonl / chain.json 讀成 numpy array

用法
----
    from chainio import load_chains

    D = load_chains(MCMC_DIR,
                    'chain_a36A')

    D['R_cmb']          # (N,) float  參數
    D['misfit']         # (N,) float  頂層純量欄位
    D['state']          # (N,) str    四態分類
    D['chain']          # (N,) int    來自第幾條鏈(索引)
    D['chain_name']     # (N,) str    鏈目錄名
    D.keys()            # 全部可用欄位

所有陣列長度相同,可以直接互相遮罩:

    m = (D['state'] == 'layered') & (D['R_cmb'] < 1700)
    np.median(D['T_core'][m])

檔名故意不帶數字前綴,這樣才能 import(63.xxx.py 不是合法模組名)。
把它跟其他 script 放同一個資料夾即可。
"""

import os
import glob
import json

import numpy as np
from mars_config import MCMC_DIR

MARS_RADIUS        = 3389.5    # km
FALLBACK_CMB_DEPTH = 1743.3    # km,舊鏈沒有 R_cmb 時退回
H_MIN              = 1.0       # km,三態分類門檻
D_NO_BML           = 2.0       # km,BML 薄於此值視為沒有 BML


# ── 單一 chain 檔的讀取 ───────────────────────────────────────────────────────
def _read_jsonl(path):
    """逐行讀;只有最後一行可能寫到一半,壞掉就丟棄該行之後的內容"""
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return recs


def _read_json(path):
    """舊版整檔 chain.json;被寫到一半時盡量救回完整的紀錄"""
    with open(path) as fh:
        text = fh.read()
    if not text.strip():
        return []
    try:
        return json.loads(text)
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
    return recs


def read_chain_dir(chain_dir):
    """優先 chain.jsonl(a34+),沒有才退回 chain.json(a31-a33)"""
    p = os.path.join(chain_dir, 'chain.jsonl')
    if os.path.exists(p):
        return _read_jsonl(p)
    p = os.path.join(chain_dir, 'chain.json')
    return _read_json(p) if os.path.exists(p) else []


# ── 主函式 ───────────────────────────────────────────────────────────────────
def load_chains(chain_dir, prefix,
                burnin_frac=0.3, min_steps=300, drop_last=1,
                derived=True, verbose=True):
    """讀 {chain_dir}/{prefix}* 下所有鏈,回傳 {欄位名: (N,) ndarray}

    burnin_frac  逐鏈丟掉前面這個比例(長度不必相同)
    min_steps    短於此長度的鏈整條丟掉
    drop_last    丟掉每條鏈最後幾步(最後一行可能寫到一半)
    derived      是否附加 state / cmb_depth_km 等推導欄位
    """
    if prefix in ('chain_a36', 'chain_a36_'):
        raise SystemExit(
            "prefix='chain_a36' 會同時抓到 a36A 與 a36B,兩者是不同的核 EoS,"
            "後驗不可合併。請指定 chain_a36A 或 chain_a36B。")

    dirs = sorted(glob.glob(os.path.join(chain_dir, f'{prefix}*')))
    if not dirs:
        raise SystemExit(f"找不到任何 {prefix}* 目錄於 {chain_dir}")

    samples, chain_idx, chain_name, report = [], [], [], []
    n_kept = 0                      # 已保留的鏈數,當作 chain 索引
    for d in dirs:
        name = os.path.basename(d)
        recs = read_chain_dir(d)
        n_raw = len(recs)
        if drop_last > 0:
            recs = recs[:-drop_last] if n_raw > drop_last else []
        if len(recs) < min_steps:
            report.append((name, n_raw, 'DROPPED'))
            continue
        b = int(len(recs) * burnin_frac)
        keep = recs[b:]
        samples.extend(keep)
        chain_idx.extend([n_kept] * len(keep))
        chain_name.extend([name] * len(keep))
        n_kept += 1
        report.append((name, n_raw, f'kept {len(keep)} / {len(recs)}'))

    if verbose:
        print(f"{'chain':<20s}{'n_raw':>8s}   action")
        for name, n_raw, act in report:
            print(f'{name:<20s}{n_raw:8d}   {act}')
    if not samples:
        raise SystemExit(f"{prefix}: burn-in 之後沒有樣本"
                         f"(min_steps={min_steps} 是否太高?)")

    n_chain = n_kept

    # ── 欄位盤點: params 展平 + 頂層純量;dict 欄位(tt_miss_by_event)跳過 ──
    num_keys, str_keys, skipped = set(), set(), set()
    for s in samples:
        for k, v in s.get('params', {}).items():
            num_keys.add(k)
        for k, v in s.items():
            if k == 'params':
                continue
            if isinstance(v, dict) or isinstance(v, (list, tuple)):
                skipped.add(k)
            elif isinstance(v, str):
                str_keys.add(k)
            elif isinstance(v, (bool, int, float)) or v is None:
                num_keys.add(k)
    str_keys -= num_keys
    param_keys = sorted({k for s in samples for k in s.get('params', {})})

    out = {}
    for k in sorted(num_keys):
        if k in param_keys:
            out[k] = np.array([_f(s.get('params', {}).get(k)) for s in samples])
        else:
            out[k] = np.array([_f(s.get(k)) for s in samples])
    for k in sorted(str_keys):
        out[k] = np.array([str(s.get(k, '')) for s in samples])

    out['chain']      = np.asarray(chain_idx, dtype=int)
    out['chain_name'] = np.asarray(chain_name)

    if derived:
        _add_derived(out, len(samples))

    if verbose:
        print(f"\n{len(samples)} samples from {n_chain} chains, "
              f"{len(out)} fields")
        if skipped:
            print(f"  非純量欄位已跳過: {sorted(skipped)}")
        _report_duplicates(out, verbose=True)

    return out


def _f(v):
    """None / 缺欄位 -> NaN;bool -> 0/1"""
    if v is None:
        return np.nan
    if isinstance(v, bool):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _add_derived(out, n):
    """幾何與三/四態分類。缺欄位時該項填 NaN / 'unknown',不會炸。"""
    # CMB 深度: a36 起 R_cmb 是自由參數,舊鏈退回常數
    if 'R_cmb' in out and np.isfinite(out['R_cmb']).any():
        out['cmb_depth_km'] = MARS_RADIUS - out['R_cmb']
    else:
        out['cmb_depth_km'] = np.full(n, FALLBACK_CMB_DEPTH)
    if 'BML_thickness' in out:
        out['bml_top_depth_km'] = out['cmb_depth_km'] - out['BML_thickness']
        out['bml_top_r_km']     = MARS_RADIUS - out['bml_top_depth_km']

    # 四態分類。no_bml 必須先切,否則極薄層會被誤判成 solid。
    if 'h_solid_km' in out and 'h_liquid_km' in out:
        hs, hl = out['h_solid_km'], out['h_liquid_km']
        st = np.full(n, 'fail', dtype='<U8')
        st[(hs >= H_MIN) & (hl >= H_MIN)] = 'layered'
        st[(hs <  H_MIN) & (hl >= H_MIN)] = 'molten'
        st[(hs >= H_MIN) & (hl <  H_MIN)] = 'solid'
        if 'BML_thickness' in out:
            st[out['BML_thickness'] < D_NO_BML] = 'no_bml'
        out['state']   = st
        out['f_solid'] = hs / np.maximum(hs + hl, 1e-9)


def _report_duplicates(out, verbose=True):
    """MCMC 拒絕時會重複寫上一個狀態,所以樣本數 >> 相異點數。
    重複率高時,以樣本數估出來的誤差棒是假的。"""
    keys = [k for k in ('T_lit', 'T_core', 'BML_thickness', 'R_cmb', 'misfit')
            if k in out]
    if not keys:
        return None
    m = np.column_stack([out[k] for k in keys])
    n_uniq = len(np.unique(np.round(m, 9), axis=0))
    n = len(m)
    if verbose:
        print(f"  相異點 {n_uniq} / {n} 筆 ({100*n_uniq/n:.1f}%)"
              + ("   <-- 重複率高,樣本數不等於有效樣本數"
                 if n_uniq < 0.5 * n else ""))
    return n_uniq


def to_struct(data, fields=None):
    """把 load_chains 的 dict 轉成 numpy structured array(要 D['x'] 以外的
    用法時方便,例如 np.savetxt / pandas.DataFrame(arr))"""
    fields = list(data) if fields is None else list(fields)
    dt = [(k, data[k].dtype) for k in fields]
    arr = np.empty(len(data[fields[0]]), dtype=dt)
    for k in fields:
        arr[k] = data[k]
    return arr


# ── 直接執行時印摘要 ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('prefix', nargs='?', default='chain_a36A')
    ap.add_argument('--dir', default=MCMC_DIR)
    ap.add_argument('--burnin', type=float, default=0.3)
    ap.add_argument('--min-steps', type=int, default=300)
    a = ap.parse_args()

    D = load_chains(a.dir, a.prefix, burnin_frac=a.burnin,
                    min_steps=a.min_steps)

    if 'state' in D:
        u, c = np.unique(D['state'], return_counts=True)
        print('\nstate:', dict(zip(u.tolist(), c.tolist())))

    print(f"\n{'field':<20s}{'median':>12s}{'2.5%':>12s}{'97.5%':>12s}")
    for k in sorted(D):
        v = D[k]
        if v.dtype.kind not in 'fi':
            continue
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        lo, md, hi = np.percentile(v, [2.5, 50, 97.5])
        print(f'{k:<20s}{md:12.4g}{lo:12.4g}{hi:12.4g}')
