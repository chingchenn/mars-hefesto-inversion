#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三態分類是怎麼算出來的 —— 從頭走一遍

這支不是拿來出圖的,是拿來搞懂邏輯的。從上往下讀,每一段都可以自己改。
"""

import json, glob, os
import numpy as np
from mars_config import MCMC_DIR

CHAIN_DIR = MCMC_DIR
PREFIX    = 'chain_a36A'
BURNIN    = 0.3
MIN_STEPS = 300


# ═══════════════════════════════════════════════════════════════════════
# 1. 讀檔
# ═══════════════════════════════════════════════════════════════════════
# chain.jsonl 是「一行一個 dict」。每一行就是 MCMC 的一步。
# 注意:被拒絕的步也會寫一行,內容跟上一步一模一樣。所以樣本數 ≠ 相異點數。

samples = []
for d in sorted(glob.glob(os.path.join(CHAIN_DIR, PREFIX + '*'))):
    recs = []
    for line in open(os.path.join(d, 'chain.jsonl')):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            break                       # 最後一行可能寫到一半
    recs = recs[:-1]
    if len(recs) < MIN_STEPS:
        continue
    samples += recs[int(len(recs) * BURNIN):]      # 丟掉 burn-in

print(f'讀到 {len(samples)} 個樣本')


# ═══════════════════════════════════════════════════════════════════════
# 2. 分類只需要三個量
# ═══════════════════════════════════════════════════════════════════════
# D  = BML 總厚度,這是 MCMC 的自由參數(我們提議的)
# hs = 其中固態次層的厚度   ┐ 這兩個是 forward model 算出來的,
# hl = 其中液態次層的厚度   ┘ 不是參數。它們決定了這個模型是什麼狀態。

D  = np.array([s['params']['BML_thickness'] for s in samples])
hs = np.array([s['h_solid_km']  for s in samples])
hl = np.array([s['h_liquid_km'] for s in samples])

# 先確認一件事:固態 + 液態 應該就等於總厚度。
# 如果這裡對不上,後面所有分類都不用看了。
residual = np.abs(hs + hl - D)
print(f'檢查 hs + hl == D :  最大殘差 = {residual.max():.3g} km')


# ═══════════════════════════════════════════════════════════════════════
# 3. 分類
# ═══════════════════════════════════════════════════════════════════════
# 概念上只有三種:
#   全固   -> hl 是 0        -> solid
#   全液   -> hs 是 0        -> molten
#   兩者都有 -> 分層          -> layered
#
# 但浮點數不會剛好是 0,而且「0.3 km 的液態層」在地震學上也不算一層。
# 所以要一個門檻 H_MIN:薄於它就當作不存在。
#
# H_MIN 是我們選的,不是資料給的。改它會改變結論,所以它必須有理由
# (例如:走時資料能分辨的最小層厚),不能隨手填。

H_MIN = 1.0        # km  <- 自己改改看

state = np.full(len(D), 'fail', dtype='<U8')          # 先全部標成 fail
state[(hs >= H_MIN) & (hl >= H_MIN)] = 'layered'
state[(hs <  H_MIN) & (hl >= H_MIN)] = 'molten'
state[(hs >= H_MIN) & (hl <  H_MIN)] = 'solid'
# 剩下沒被蓋掉的還是 'fail' = hs 和 hl 都小於門檻 = 整層都太薄

print(f'\nH_MIN = {H_MIN} km 之下的三態:')
for k in ['solid', 'layered', 'molten', 'fail']:
    n = int((state == k).sum())
    print(f'   {k:<8s} {n:6d}   {100*n/len(D):5.2f}%')


# ═══════════════════════════════════════════════════════════════════════
# 4. 第四態:no_bml
# ═══════════════════════════════════════════════════════════════════════
# 上面有個漏洞。假設 D = 3 km 而且整層固態,那 hs = 3、hl = 0,
# 它會被標成 'solid' —— 也就是「存在一層固態 BML」。
# 可是 3 km 的固態層跟「根本沒有 BML」在物理上分不出來。
#
# 這很重要,因為整篇論文要回答的就是「有沒有 BML」。如果把
# 「沒有」跟「有一層薄固態」放進同一個 solid 桶子裡,BF 的分子分母
# 就不是你以為的那個假設了。
#
# 所以要再切一刀。注意順序:這一刀必須放在最後,才蓋得掉前面的標籤。

D_NO_BML = 2.0     # km  <- 自己改改看

state_4 = state.copy()
state_4[D < D_NO_BML] = 'no_bml'

print(f'\n加上 D_NO_BML = {D_NO_BML} km 之後:')
for k in ['no_bml', 'solid', 'layered', 'molten', 'fail']:
    n = int((state_4 == k).sum())
    print(f'   {k:<8s} {n:6d}   {100*n/len(D):5.2f}%')


# ═══════════════════════════════════════════════════════════════════════
# 5. 門檻選多少有差嗎?
# ═══════════════════════════════════════════════════════════════════════
# 與其爭論門檻該是多少,不如直接看結論對它敏不敏感。
# 如果 layered:solid 的比值在合理範圍內幾乎不動,那門檻就不是重點。

print('\nD_NO_BML 掃描:')
print(f"   {'門檻':>6}{'no_bml':>9}{'solid':>8}{'layered':>9}{'lay/sol':>10}")
for t in [0, 1, 2, 5, 10, 20, 50]:
    st = state.copy()
    st[D < t] = 'no_bml'
    ns, nl = int((st == 'solid').sum()), int((st == 'layered').sum())
    nb = int((st == 'no_bml').sum())
    print(f'   {t:6.0f}{nb:9d}{ns:8d}{nl:9d}{nl/ns if ns else np.nan:10.3f}')


# ═══════════════════════════════════════════════════════════════════════
# 6. 牆邊到底有沒有樣本?
# ═══════════════════════════════════════════════════════════════════════
# 上面那個掃描只能告訴你「已經抽到的樣本」怎麼重新貼標籤。
# 但如果 D = 0 附近根本沒有樣本,那不管門檻怎麼設,
# 「沒有 BML」這個假設都沒有被真正測試過。
#
# Savage-Dickey 要的是 D=0 這一點的後驗密度。沒有樣本 = 沒辦法估。

print(f'\nD 的最小值 = {D.min():.3f} km')
for a, b in [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 50)]:
    m = (D >= a) & (D < b)
    print(f'   {a:3d} - {b:3d} km : n = {int(m.sum()):5d}   {100*m.mean():6.3f}%')

# 用相異值看,才不會被重複列騙
u = np.unique(np.round(D, 6))
print(f'\n相異 D 值共 {len(u)} 個,其中小於 20 km 的只有 {int((u < 20).sum())} 個')
print(f'最小的五個: {np.sort(u)[:5].round(2).tolist()}')


# ═══════════════════════════════════════════════════════════════════════
# 接下來可以自己玩的
# ═══════════════════════════════════════════════════════════════════════
# * 改 H_MIN 從 0.1 到 10,看三態比例怎麼變
# * 把 state 跟別的量交叉看:
#       T_core = np.array([s['params']['T_core'] for s in samples])
#       np.median(T_core[state == 'layered'])
# * 逐鏈看,檢查是不是某幾條鏈主導了某一態:
#       (需要在第 1 段順便記下每個樣本來自哪條鏈)
