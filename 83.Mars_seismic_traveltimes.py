#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 85.tt_dataset_audit.py
#
# InSight 體波走時資料的逐筆清點。輸出:
#   (0) 事件層級:三套定位並排 (IPGP / ETH / MQS-V14) + 是否在現行 code
#   (1) 逐筆走時:每個事件每個相位,各目錄並排,含去留判定
#   (2) 剔除清單與理由
#   (3) 跨鏈判讀分歧 sigma_sys
#   (4) 最終 TT_DATA
#   (5) Bi 2025 的四個核相位 (獨立區塊,不進 likelihood)
#
# 三條獨立的 pick 鏈:
#   IPGP : MQS -> Drilleau 2022 -> Drilleau 2024 -> Samuel 2023 / Drilleau 2026
#   ETH  : MQS -> Duran 2022 / Stahler 2021 / Kawamura 2022 -> Khan 2023 / 2026
#   Bi   : MQS V14 -> Bi 2025 (只有核相位)
# Irving 2023 由兩系合著,SS-PP / SKS-PP 兩系相同。

import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'PingFang TC'

# ── 設定 ──────────────────────────────────────────────────────────────────
CHAIN       = 'IPGP'      # 'IPGP' 或 'ETH'。走時與定位必須來自同一條鏈。
SIGMA_MODE  = 'scs_only'  # 'raw' | 'scs_only' | 'all'
SIG_FLOOR   = 2.0
SAVE_FIG    = True
fontsize    = 14

# sigma_pick 相位種類固定值
# Drilleau 2024 SM Table S1.1 caption == Samuel 2023 SM Table 1 caption。
# Drilleau 2024 Eq. S3.2 把這些數字直接當 differential sigma 用,不做傳播。
SIG_DR = {'S-P':5.0, 'pP-P':3.0, 'sP-P':5.0, 'PP-P':8.0, 'PPP-P':12.0,
          'sS-S':5.0, 'SS-S':5.0, 'SSS-S':8.0, 'ScS-S':12.0,
          'SS-PP':10.0, 'SKS-PP':10.0}
PHASES = ['S-P','pP-P','sP-P','PP-P','PPP-P','sS-S','SS-S','SSS-S','ScS-S',
          'SS-PP','SKS-PP']

# ══════════════════════════════════════════════════════════════════════════
# IPGP 鏈  Drilleau 2024 GRL SM Table S1.1 (走時) + Table S7.1 (定位,1D 欄)
# ══════════════════════════════════════════════════════════════════════════
DR_GEOM = {   # Delta, sDelta, depth, sDepth
 'S0154a': (29.9,2.2, 60.6,24.1), 'S0173a': (30.2,1.5, 41.7,13.0),
 'S0185a': (54.7,1.6, 12.5, 9.6), 'S0235b': (29.2,1.5, 21.5, 9.9),
 'S0325a': (41.5,2.0, 44.4, 9.2), 'S0407a': (27.9,1.6, 33.7,10.1),
 'S0409d': (30.3,2.2, 24.0, 7.2), 'S0474a': (19.7,2.5, 55.9,18.7),
 'S0484b': (30.9,1.4, 45.0,15.6), 'S0784a': (30.6,1.9, 18.7, 8.4),
 'S0802a': (27.9,2.3, 17.0, 8.9), 'S0809a': (29.4,2.8, 20.0,10.0),
 'S0820a': (29.4,2.3, 18.6, 8.1), 'S0861a': (55.1,3.0, 38.5,28.4),
 'S0864a': (29.5,3.2, 38.3,15.5), 'S0916d': (29.4,1.4, 30.8,20.0),
 'S0918a': (16.8,2.2, 56.5,26.2), 'S0976a': (143.6,2.6, 52.5,29.5),
 'S1000a': (126.09,0.0, 0.0,0.0), 'S1012d': (37.6,1.9, 15.6, 5.4),
 'S1015f': (30.1,2.1, 26.1, 8.8), 'S1022a': (29.4,2.2, 21.3, 6.0),
 'S1039b': (30.1,2.8, 38.1,13.1), 'S1048d': (30.8,1.7, 35.1,12.1),
 'S1094b': (58.5,0.0, 0.0,0.0),   'S1102a': (75.7,2.9, 57.1,23.1),
 'S1133c': (29.5,2.1, 18.3, 4.5), 'S1153a': (82.2,2.7, 95.9,42.4),
 'S1157a': (36.9,2.1, 30.3, 7.7), 'S1197a': (32.2,2.0, 35.7,10.3),
 'S1222a': (37.6,2.4, 61.7,25.0),
}
DR = {
 'S0154a': {'S-P':174.4,'SS-S':25.3,'SSS-S':35.0},
 'S0173a': {'S-P':178.8,'sP-P':9.43,'PP-P':19.9,'PPP-P':34.4,'sS-S':13.2,
            'SS-S':24.4,'SSS-S':40.5,'ScS-S':345.2},
 'S0185a': {'S-P':327.28,'pP-P':4.0,'PP-P':22.47,'PPP-P':49.3,'sS-S':10.0,
            'SS-S':30.9,'SSS-S':55.4,'ScS-S':152.3},
 'S0235b': {'S-P':171.4,'PP-P':18.6,'PPP-P':32.0,'sS-S':9.2,'SS-S':23.2,
            'SSS-S':33.3,'ScS-S':343.9},
 'S0325a': {'S-P':229.3,'pP-P':9.8,'PP-P':21.1,'PPP-P':34.4,'sS-S':13.8,
            'SS-S':26.1,'SSS-S':50.3,'ScS-S':220.4},
 'S0407a': {'S-P':170.7,'pP-P':6.77,'PP-P':23.38,'sS-S':13.3,'SS-S':21.1,
            'SSS-S':33.1,'ScS-S':370.0},
 'S0409d': {'S-P':163.2,'pP-P':8.3,'PP-P':27.6,'PPP-P':36.94,'sS-S':8.4,
            'SS-S':20.9,'SSS-S':39.8,'ScS-S':320.1},
 'S0474a': {'S-P':121.6,'PP-P':13.4,'PPP-P':24.8,'SS-S':15.8,'SSS-S':32.4},
 'S0484b': {'S-P':173.1,'pP-P':5.5,'PP-P':19.73,'sS-S':13.0,'SS-S':17.4,
            'ScS-S':322.3},
 'S0784a': {'S-P':179.3,'pP-P':6.5,'PP-P':13.7,'PPP-P':22.4,'sS-S':7.2,
            'SS-S':19.6,'SSS-S':28.0},
 'S0802a': {'S-P':180.3,'pP-P':4.0,'PP-P':25.6,'PPP-P':33.9,'sS-S':9.3,
            'SS-S':22.4,'SSS-S':36.5,'ScS-S':387.6},
 'S0809a': {'S-P':191.95,'pP-P':4.5,'PP-P':16.25,'PPP-P':29.65,'sS-S':8.1,
            'SS-S':23.8,'SSS-S':39.3,'ScS-S':373.5},
 'S0820a': {'S-P':174.1,'PP-P':21.9,'PPP-P':32.1,'sS-S':8.5},
 'S0861a': {'S-P':319.3,'PP-P':19.6,'PPP-P':47.6,'SS-S':41.1},
 'S0864a': {'S-P':171.4,'PP-P':18.0,'PPP-P':27.9,'sS-S':17.3,'SS-S':26.4},
 'S0916d': {'S-P':170.8,'pP-P':3.9,'PP-P':19.3,'PPP-P':36.1,'SS-S':19.0,
            'SSS-S':42.9,'ScS-S':342.8},
 'S0918a': {'S-P':102.4,'PP-P':12.8,'PPP-P':22.5,'SS-S':21.2,'SSS-S':35.0},
 'S0976a': {'SS-PP':854.4,'SKS-PP':303.9},
 'S1000a': {'SS-PP':749.0,'SKS-PP':339.3},
 'S1012d': {'S-P':221.7,'pP-P':6.7,'PP-P':18.0,'PPP-P':27.7,'sS-S':11.0,
            'SS-S':24.5,'SSS-S':40.65},
 'S1015f': {'S-P':177.7,'PP-P':20.47,'PPP-P':35.37,'sS-S':11.2},
 'S1022a': {'S-P':177.0,'pP-P':9.0,'PP-P':19.3,'PPP-P':27.5,'sS-S':11.0,
            'SS-S':25.0,'SSS-S':55.0},
 'S1039b': {'S-P':176.1,'pP-P':7.8,'PP-P':12.3,'PPP-P':28.7,'SS-S':14.9},
 'S1048d': {'S-P':179.19,'pP-P':7.49,'PP-P':18.05,'SS-S':24.3},
 'S1094b': {'S-P':343.0},
 'S1102a': {'S-P':443.7},
 'S1133c': {'S-P':175.3,'pP-P':5.8,'PP-P':16.7,'PPP-P':28.46,'sS-S':9.1,
            'SS-S':17.4,'SSS-S':35.4},
 'S1153a': {'S-P':467.95},
 'S1157a': {'S-P':215.8,'PP-P':18.7,'sS-S':14.6,'SS-S':39.6},
 'S1197a': {'S-P':191.2,'pP-P':8.4,'PP-P':15.4,'sS-S':14.7},
 'S1222a': {'S-P':216.0,'PP-P':28.0,'PPP-P':37.8,'ScS-S':258.0},
}

# ══════════════════════════════════════════════════════════════════════════
# ETH 鏈
# ══════════════════════════════════════════════════════════════════════════
ST_A = {'S0235b':(167,511,3), 'S0407a':(168,510,10), 'S0484b':(172,513,20),
        'S0173a':(173,512,3), 'S0409d':(177,510,5),  'S0325a':(230,500,20)}
ST_B = {'S0185a':(360.4,444.9,5.6), 'S0183a':(305.6,459.9,3.0),
        'S0325a':(230.2,500.3,9.0), 'S0484b':(173.4,524.9,7.9),
        'S0173a':(172.7,517.9,37.5),'S0235b':(171.1,517.2,90.1),
        'S0407a':(170.9,523.8,8.6), 'S0409d':(164.1,514.2,6.7)}
KH_T0 = {'S0167b':2,'S0173a':3,'S0183a':3,'S0185a':3,'S0235b':2,'S0325a':2,
         'S0407a':3,'S0409d':4,'S0484b':3,'S0784a':5,'S0802a':4,'S0809a':3,
         'S0820a':3,'S0864a':5,'S0976a':8,'S1000a':5,'S1094b':1,
         'S1022a':3,'S1048d':4,'S1133c':4,'S1222a':4}
KH = {
 'S0167b': {'PP':(37.0,3),'S':(414.5,2),'SSS':(468.0,3)},
 'S0173a': {'pP':(11.1,3),'S':(174.8,2),'sS':(184.8,3),'SS':(197.9,2),'ScS':(515.0,5)},
 'S0183a': {'PP':(24.5,4),'PPP':(43.0,7)},
 'S0185a': {'S':(360.2,2),'sS':(379.5,4),'SSS':(412.5,6)},
 'S0235b': {'PP':(17.4,2),'PPP':(31.1,7),'S':(166.0,3),'sS':(178.7,3),
            'SS':(193.1,3),'ScS':(512.0,8)},
 'S0325a': {'pP':(11.3,3),'S':(230.7,3),'SS':(260.7,4),'SSS':(281.0,6)},
 'S0407a': {'PP':(17.8,5),'PPP':(33.0,7),'S':(172.2,2),'sS':(183.4,3),'SS':(196.4,5)},
 'S0409d': {'S':(162.5,3),'SS':(184.9,4),'SSS':(207.7,6)},
 'S0484b': {'PP':(18.1,5),'S':(170.2,1),'sS':(184.0,3),'SS':(196.8,3)},
 'S0784a': {'PP':(15.2,4),'PPP':(29.5,7),'S':(173.0,4),'sS':(182.2,4),
            'SS':(196.9,5),'SSS':(221.9,6)},
 'S0802a': {'PP':(17.3,5),'S':(176.3,5),'SS':(201.5,5),'SSS':(222.1,6)},
 'S0809a': {'PP':(15.5,5),'PPP':(29.3,7),'S':(175.0,1),'SS':(197.5,3)},
 'S0820a': {'PP':(15.7,5),'S':(176.5,2),'sS':(186.2,3),'SS':(201.4,5)},
 'S0864a': {'PP':(18.0,5),'S':(169.0,3),'sS':(181.2,4),'SS':(194.0,3),
            'SSS':(216.1,6),'ScS':(505.0,8)},
 'S1094b': {'S':(342.8,20)},
 'S0976a': {'SS_PP':(859.0,20)},       # 該兩事件 T0 = PP 到時
 'S1000a': {'SS_PP':(746.0,8)},
}
# Khan 2026 Rev. Geophys. Table 6 —— 與 KH23 是同一系但不是同一張表:
#   S0167b / S0183a / S0185a 已移除
#   S0325a 的 pP 不再列出
#   S1000a 的 SS-PP 由 746.0+-8 改為 727.0+-8   (19 s 修訂)
#   新增 S1022a / S1048d / S1133c / S1222a
KH26 = dict(KH)
for _e in ('S0167b','S0183a','S0185a'):
    KH26.pop(_e, None)
KH26['S0325a'] = {k:v for k,v in KH['S0325a'].items() if k != 'pP'}
KH26['S1000a'] = {'SS_PP':(727.0,8)}
KH26['S1022a'] = {'S':(171.0,3),'SS':(197.0,4)}
KH26['S1048d'] = {'S':(175.8,3)}
KH26['S1133c'] = {'S':(172.3,3),'SS':(194.2,3)}
KH26['S1222a'] = {'S':(214.4,3)}       # 該事件無 ScS —— 與 Kawamura 2022 不同
KH_GEOM = {  # Khan 2026 Table 6: Delta, sDelta, quality
 'S0173a':(29.5,1.7,'A'), 'S0235b':(28.1,1.8,'A'), 'S0325a':(39.4,2.1,'B'),
 'S0407a':(29.1,1.8,'B'), 'S0409d':(27.5,1.1,'B'), 'S0484b':(28.8,1.3,'B'),
 'S0784a':(29.3,1.8,'B'), 'S0802a':(29.9,2.1,'B'), 'S0809a':(29.6,1.9,'A'),
 'S0820a':(29.9,2.6,'A'), 'S0864a':(28.6,2.2,'A'), 'S0976a':(143.9,6.7,'A'),
 'S1000a':(126.1,0.0,'A'),'S1022a':(28.7,1.7,'A'), 'S1048d':(30.2,2.3,'A'),
 'S1094b':(58.5,0.0,'A'), 'S1133c':(29.4,1.6,'A'), 'S1222a':(35.9,2.3,'A'),
}
KA = {'S1222a': {'S-P':216.0, 'ScS-S':258.0}}     # Kawamura 2022 GRL
IR = {'S0976a': {'SS-PP':(854.4,5.2), 'SKS-PP':(303.9,7.1)},
      'S1000a': {'SS-PP':(749.0,4.7), 'SKS-PP':(339.3,4.7)}}
IR_UNCONFIDENT = {('S0976a','SKS-PP')}

# ══════════════════════════════════════════════════════════════════════════
# Bi et al. 2025 Nature  (第三套定位 MQS V14;只有核相位)
# ══════════════════════════════════════════════════════════════════════════
BI_GEOM = {
 'S1015f':(27.5,2.0,'A'),  'S0918a':(27.8,6.7,'B'),  'S0235b':(28.7,1.5,'A'),
 'S0864a':(28.7,3.5,'A'),  'S0474a':(29.1,17.5,'B'), 'S0916d':(29.3,5.9,'B'),
 'S0802a':(30.0,3.5,'B'),  'S0173a':(30.0,1.4,'A'),  'S0820a':(30.2,2.4,'A'),
 'S1048d':(30.2,1.3,'A'),  'S1133c':(30.2,1.3,'A'),  'S0290b':(30.4,8.2,'B'),
 'S1039b':(30.4,17.5,'B'), 'S1022a':(30.7,2.0,'A'),  'S0484b':(31.8,5.9,'B'),
 'S1197a':(32.0,1.5,'B'),  'S0105a':(32.5,8.2,'B'),  'S0189a':(33.4,8.2,'B'),
 'S0784a':(34.5,3.6,'B'),  'S1157a':(35.6,2.5,'B'),  'S1222a':(37.0,1.6,'A'),
 'S1012d':(38.3,3.3,'B'),  'S0325a':(39.7,6.1,'B'),  'S1102a':(73.3,4.6,'A'),
 'S1153a':(84.8,10.5,'B'), 'S1415a':(88.2,9.6,'B'),
}
# Supplementary Tables 3-6,filter-bank 值 (sigma 最小的一欄)
BI_CORE = {
 'PKiKP':    {'S0235b':(603.8,1.6),'S0802a':(596.4,1.8),'S0864a':(602.4,2.2),
              'S1015f':(608.4,2.7),'S1039b':(606.4,3.0),'S1197a':(586.5,1.2),
              'S1012d':(548.0,1.5)},
 'PKKP':     {'S0235b':(1347.8,1.9),'S0474a':(1343.7,1.3),'S0484b':(1326.5,1.8),
              'S0802a':(1330.7,2.6),'S0918a':(1347.6,1.4),'S1012d':(1274.9,2.5),
              'S1048d':(1320.4,2.4)},
 "P'P'r_ab": {'S0173a':(1829.1,1.7),'S0235b':(1831.5,2.2),'S0802a':(1821.4,2.8),
              'S0864a':(1836.6,3.0),'S1015f':(1855.2,2.5),'S1022a':(1799.7,2.1),
              'S1048d':(1809.8,2.0),'S1222a':(1741.7,1.9),'S0918a':(1846.6,2.5)},
 "P'P'n":    {'S0105a':(1990.0,2.5),'S0474a':(2007.8,2.6),'S0802a':(2003.8,2.6),
              'S0864a':(2007.7,2.2),'S1133c':(2005.9,2.1)},
}
BI_STACK = {'PKiKP':(604,2), 'PKKP':(1341,5), "P'P'r_ab":(1835,4), "P'P'n":(2008,3)}

# ══════════════════════════════════════════════════════════════════════════
# 現行 code 的 SAMUEL_DATA
# ══════════════════════════════════════════════════════════════════════════
MINE = {
 'S0154a': {'SS-S':25.3,'SSS-S':35.0},
 'S0173a': {'S-P':178.8,'sP-P':9.43,'PP-P':19.9,'PPP-P':34.4,'sS-S':13.2,
            'SS-S':24.4,'SSS-S':40.5,'ScS-S':345.2},
 'S0185a': {'S-P':327.28,'pP-P':4.0,'PP-P':22.47,'PPP-P':49.3,'sS-S':10.0,
            'SS-S':30.9,'SSS-S':55.4,'ScS-S':152.3},
 'S0235b': {'S-P':171.4,'PP-P':18.6,'PPP-P':32.0,'sS-S':9.2,'SS-S':23.2,
            'SSS-S':33.3,'ScS-S':343.9},
 'S0325a': {'S-P':229.3,'pP-P':9.8,'PP-P':21.1,'PPP-P':34.4,'sS-S':13.8,
            'SS-S':26.1,'SSS-S':50.3,'ScS-S':220.4},
 'S0407a': {'S-P':170.7,'pP-P':6.77,'PP-P':23.38,'sS-S':13.3,'SS-S':21.1,
            'SSS-S':33.1,'ScS-S':370.0},
 'S0409d': {'S-P':163.2,'pP-P':8.3,'PP-P':27.6,'PPP-P':36.94,'sS-S':8.4,
            'SS-S':20.9,'SSS-S':39.8,'ScS-S':320.1},
 'S0474a': {'S-P':121.6,'PP-P':13.4,'sS-S':9.7,'SS-S':16.9,'SSS-S':25.8,
            'ScS-S':455.0},
 'S0484b': {'S-P':151.0,'PP-P':16.0,'PPP-P':29.4,'sS-S':11.6,'SS-S':19.6,
            'SSS-S':33.4,'ScS-S':360.0},
 'S0820a': {'S-P':437.5,'SS-S':48.4,'SSS-S':82.3,'SS-PP':22.6},
 'S1000a': {'S-P':490.4,'SS-S':54.9,'SSS-S':93.3,'SKS-PP':87.0,
            'PP-PbdiffPcP':22.5},
 'S1222a': {'S-P':216.0,'ScS-S':258.0},
}

# ══════════════════════════════════════════════════════════════════════════
# 轉成統一格式
# ══════════════════════════════════════════════════════════════════════════
_REF  = {'pP':'P','sP':'P','PP':'P','PPP':'P','sS':'S','SS':'S','SSS':'S','ScS':'S'}
_NAME = {'pP':'pP-P','sP':'sP-P','PP':'PP-P','PPP':'PPP-P',
         'sS':'sS-S','SS':'SS-S','SSS':'SSS-S','ScS':'ScS-S'}

def kh_diff(src):
    # T0 = P 到時 (S0976a/S1000a 為 PP 到時)。
    #   X-P  = T_X,          sigma = sqrt(sX^2 + sT0^2)
    #   X-S  = T_X - T_S,    sigma = sqrt(sX^2 + sS^2)
    out = {}
    for ev, tab in src.items():
        S, s0, d = tab.get('S'), KH_T0.get(ev, 3), {}
        for ph, (v, s) in tab.items():
            if ph == 'SS_PP':
                d['SS-PP'] = (v, float(np.hypot(s, s0)))
            elif ph == 'S':
                d['S-P'] = (v, float(np.hypot(s, s0)))
            elif _REF.get(ph) == 'P':
                d[_NAME[ph]] = (v, float(np.hypot(s, s0)))
            elif _REF.get(ph) == 'S' and S is not None:
                d[_NAME[ph]] = (v - S[0], float(np.hypot(s, S[1])))
        out[ev] = d
    return out

def st_diff(raw, mode):
    # ScS-S = (tScS - tP) - (tS - tP)
    out = {}
    for ev, r in raw.items():
        if mode == 'a':
            sp, scsp, sig = r
            out[ev] = {'S-P':(float(sp),5.0), 'ScS-S':(float(scsp-sp), float(sig))}
        else:
            sp, scsp, _ = r
            out[ev] = {'S-P':(sp,2.5), 'ScS-S':(scsp-sp, float(np.hypot(5.0,2.5)))}
    return out

SRC = {'DR24':  {e:{p:(v,SIG_DR[p]) for p,v in t.items()} for e,t in DR.items()},
       'ST21a': st_diff(ST_A,'a'),
       'ST21b': st_diff(ST_B,'b'),
       'KH23':  kh_diff(KH),
       'KH26':  kh_diff(KH26),
       'IR23':  IR,
       'KA22':  {e:{p:(v,SIG_DR[p]) for p,v in t.items()} for e,t in KA.items()}}
SHOW = ['DR24','ST21a','ST21b','KH23','KH26','IR23','KA22']

CHAIN_SRC = {'IPGP': ['DR24','IR23','KA22'],
             'ETH':  ['KH26','KH23','ST21b','ST21a','IR23','KA22']}[CHAIN]
IPGP_S, ETH_S = ['DR24'], ['ST21a','ST21b','KH26','KA22']

# DR24 明確轉載自其他來源的筆,不算獨立判讀
# Samuel 2023 SM: S0976a/S1000a 來自 Irving 2023;S1094b/S1222a 來自
#                 Posiolova 2022 / Kawamura 2022
DERIVED = {('S0976a','DR24'),('S1000a','DR24'),('S1222a','DR24'),('S1094b','DR24')}

ALL_EV = sorted(set(DR)|set(SRC['KH23'])|set(SRC['KH26'])
                |set(SRC['ST21b'])|set(MINE)|set(BI_GEOM))

# ══════════════════════════════════════════════════════════════════════════
# (0) 事件層級
# ══════════════════════════════════════════════════════════════════════════
print('=' * 112)
print('(0) 事件層級:三套獨立定位   n = 各鏈的走時筆數')
print('=' * 112)
print(f"{'event':<9}{'IPGP Delta':>15}{'ETH Delta':>15}{'Bi(V14) Delta':>16}"
      f"{'IPGP depth':>15}{'Q':>3}{'nIP':>5}{'nETH':>6}{'nBi':>5}{'MINE':>6}")
for ev in ALL_EV:
    g1, g2, g3 = DR_GEOM.get(ev), KH_GEOM.get(ev), BI_GEOM.get(ev)
    c1 = f'{g1[0]:9.1f}+-{g1[1]:.1f}' if g1 else f'{"--":>15}'
    c2 = f'{g2[0]:9.1f}+-{g2[1]:.1f}' if g2 else f'{"--":>15}'
    c3 = f'{g3[0]:10.1f}+-{g3[1]:.1f}' if g3 else f'{"--":>16}'
    dp = f'{g1[2]:9.1f}+-{g1[3]:.1f}' if g1 else f'{"--":>15}'
    q  = g2[2] if g2 else (g3[2] if g3 else '-')
    n1 = len(DR.get(ev, {}))
    n2 = len(set(SRC['KH26'].get(ev,{})) | set(SRC['KH23'].get(ev,{}))
             | set(SRC['ST21b'].get(ev,{})) | set(SRC['ST21a'].get(ev,{})))
    n3 = sum(1 for ph in BI_CORE if ev in BI_CORE[ph])
    print(f'{ev:<9}{c1}{c2}{c3}{dp}{q:>3}{n1:>5}{n2:>6}{n3:>5}'
          f'{("Y" if ev in MINE else ""):>6}')

# ══════════════════════════════════════════════════════════════════════════
# (1) 逐筆走時 + 去留
# ══════════════════════════════════════════════════════════════════════════
print()
print('=' * 112)
print(f'(1) 逐筆走時。CHAIN = {CHAIN}(走時與定位同源)。格式 值(sigma)')
print('=' * 112)

records, dropped, changed = [], [], []
for ev in ALL_EV:
    lines = []
    for ph in PHASES:
        cells, ipgp, eth = '', [], []
        for s in SHOW:
            e = SRC[s].get(ev, {}).get(ph)
            cells += f'{"--":>13}' if e is None else f'{e[0]:>8.2f}({e[1]:>3.1f})'
            if e is None or (ev, s) in DERIVED:
                continue
            (ipgp if s in IPGP_S else eth).append(e[0])
        mine = MINE.get(ev, {}).get(ph)
        chain_e, chain_src = None, ''
        for s in CHAIN_SRC:
            e = SRC[s].get(ev, {}).get(ph)
            if e is not None:
                chain_e, chain_src = e, s
                break
        if chain_e is None and mine is None:
            continue
        # sigma_sys: 兩條鏈各取中位數。Var(v1 - v2) = 2 sigma^2
        s_sys = abs(np.median(ipgp) - np.median(eth))/np.sqrt(2.0) \
                if (ipgp and eth) else np.nan
        if chain_e is None:
            act, why = 'DROP', f'{CHAIN} 鏈無此筆(現行 code 有,六家皆無來源)'
            dropped.append((ev, ph, mine, why))
        elif (ev, ph) in IR_UNCONFIDENT:
            act, why = 'DROP', 'Irving 2023 標 * insufficiently confident'
            dropped.append((ev, ph, chain_e[0], why))
        else:
            act, why = 'KEEP', ''
            if mine is not None and abs(mine - chain_e[0]) > 0.005:
                why = f'值改 {mine} -> {chain_e[0]}'
                changed.append((ev, ph, mine, chain_e[0], chain_src))
            elif mine is None:
                why = '新增'
            records.append(dict(ev=ev, ph=ph, val=chain_e[0], sig=chain_e[1],
                                src=chain_src, s_sys=s_sys, mine=mine))
        ss = f'{s_sys:5.1f}' if np.isfinite(s_sys) else f'{"--":>5}'
        mv = f'{mine:>9.2f}' if mine is not None else f'{"--":>9}'
        lines.append(f'  {ph:<7}{cells}{mv}{ss:>7}  {act:<5}{why}')
    if lines:
        print(f'\n{ev}  ' + '-' * 112)
        print(f'  {"phase":<7}{"DR24":>13}{"ST21a":>13}{"ST21b":>13}'
              f'{"KH23":>13}{"KH26":>13}{"IR23":>13}{"KA22":>13}'
              f'{"MINE":>9}{"sysig":>7}')
        for l in lines:
            print(l)

for ev, tt in MINE.items():
    for ph in tt:
        if ph not in PHASES:
            dropped.append((ev, ph, tt[ph],
                            'Pdiff 辨識未定 (Khan 給三個值 -235/-177/-113);'
                            'Russell 2026 質疑'))

# ══════════════════════════════════════════════════════════════════════════
# (2) 剔除與修正
# ══════════════════════════════════════════════════════════════════════════
print()
print('=' * 112)
print('(2a) 剔除清單')
print('=' * 112)
print(f"{'event':<9}{'phase':<15}{'現行值':>10}   理由")
for ev, ph, v, why in sorted(dropped):
    vv = f'{v:10.2f}' if isinstance(v, (int, float)) else f'{"--":>10}'
    print(f'{ev:<9}{ph:<15}{vv}   {why}')

print()
print('=' * 112)
print('(2b) 值被修正的筆')
print('=' * 112)
print(f"{'event':<9}{'phase':<9}{'現行':>10}{'改為':>10}{'差':>9}   來源")
for ev, ph, old, new, src in sorted(changed):
    print(f'{ev:<9}{ph:<9}{old:>10.2f}{new:>10.2f}{new-old:>+9.2f}   {src}')

# ══════════════════════════════════════════════════════════════════════════
# (3) 跨鏈判讀分歧
# ══════════════════════════════════════════════════════════════════════════
print()
print('=' * 112)
print('(3) 跨鏈判讀分歧   sigma_sys = |median(IPGP) - median(ETH)| / sqrt(2)')
print('=' * 112)
print(f"{'phase':<10}{'n':>4}{'sig_sys med':>13}{'max':>8}{'sig_pick':>10}{'ratio':>8}")
RATIO = {}
for ph in PHASES:
    ss = [r['s_sys'] for r in records if r['ph'] == ph and np.isfinite(r['s_sys'])]
    if not ss:
        continue
    RATIO[ph] = float(np.median(ss)/SIG_DR[ph])
    print(f'{ph:<10}{len(ss):>4}{np.median(ss):>13.1f}{max(ss):>8.1f}'
          f'{SIG_DR[ph]:>10.1f}{RATIO[ph]:>8.2f}')
print('\n  ratio > 1 = 標示的 pick 誤差沒有涵蓋兩條鏈的判讀差異')

# ══════════════════════════════════════════════════════════════════════════
# (4) 最終 TT_DATA
#     sigma = sqrt(sigma_pick^2 + sigma_sys^2),範圍由 SIGMA_MODE 決定
# ══════════════════════════════════════════════════════════════════════════
_scs_r = RATIO.get('ScS-S', 0.0)
TT = {}
for r in records:
    sig = r['sig']
    if SIGMA_MODE != 'raw' and (SIGMA_MODE == 'all' or r['ph'] == 'ScS-S'):
        ss = r['s_sys'] if np.isfinite(r['s_sys']) else r['sig']*_scs_r
        sig = float(np.hypot(r['sig'], ss))
    TT.setdefault(r['ev'], {})[r['ph']] = (r['val'], round(max(sig, SIG_FLOOR), 1))

GEOM = DR_GEOM if CHAIN == 'IPGP' else \
       {e:(g[0], g[1], DR_GEOM.get(e,(0,0,30.0,15.0))[2],
           DR_GEOM.get(e,(0,0,30.0,15.0))[3]) for e,g in KH_GEOM.items()}

print()
print('=' * 112)
print(f'(4) TT_DATA   CHAIN={CHAIN}  SIGMA_MODE={SIGMA_MODE}')
print('=' * 112)
print(f'# 走時與定位皆取自 {CHAIN} 鏈 —— 每條鏈的 pick 綁著自己的定位解,不可混用。')
print('# sigma_pick: Drilleau 2024 SM Table S1.1 caption (= Samuel 2023 Table 1)')
print('# sigma_sys : IPGP 與 ETH 兩條獨立 pick 鏈的判讀差異,見第 (3) 節')
print('TT_DATA = {')
n_ph = 0
for ev in sorted(TT):
    if ev not in GEOM:
        continue
    D, sD, h, sh = GEOM[ev]
    print(f"    '{ev}': {{")
    print(f"        'delta': {D}, 'sigma_delta': {sD}, "
          f"'depth': {h}, 'sigma_depth': {sh},")
    for ph in PHASES:
        if ph in TT[ev]:
            v, s = TT[ev][ph]
            print(f"        '{ph}':{' '*(8-len(ph))}({v}, {s}),")
            n_ph += 1
    print('    },')
print('}')
n_ev = len([e for e in TT if e in GEOM])
print(f'\n# {n_ev} events, {n_ph} phases   (現行 code: 12 events, 72 phases)')
n_scs = sum(1 for e in TT for p in TT[e] if p == 'ScS-S')
print(f'# ScS-S: {n_scs} 筆   (現行 code: 9 筆)')

# 給 86.sigma_from_location.py 用。sigma 拆成 pick / sys 兩項分開存,
# 86 會再加上定位項後重新平方和。
_dump = {}
for r in records:
    if r['ev'] not in GEOM:
        continue
    D, sD, h, sh = GEOM[r['ev']]
    _dump.setdefault(r['ev'], {'delta':D, 'sigma_delta':sD,
                               'depth':h, 'sigma_depth':sh, 'tt':{}})
    # 從未有兩鏈同時觀測的相位 (sP-P),用全體 ratio 中位數外推,不可留 0
    _rmed = float(np.median(list(RATIO.values()))) if RATIO else 0.0
    ss = r['s_sys'] if np.isfinite(r['s_sys']) else r['sig']*RATIO.get(r['ph'], _rmed)
    _dump[r['ev']]['tt'][r['ph']] = dict(value=r['val'], sigma_pick=r['sig'],
                                         sigma_sys=float(ss), src=r['src'])
json.dump(_dump, open('85_tt_data.json','w'), indent=1)
print("\n# 已寫出 85_tt_data.json")

# ══════════════════════════════════════════════════════════════════════════
# (5) Bi 2025 核相位
# ══════════════════════════════════════════════════════════════════════════
print()
print('=' * 112)
print('(5) Bi et al. 2025 核相位 (相對 P,filter-bank 值)。建議只做 posterior predictive。')
print('=' * 112)
for ph in BI_CORE:
    st, ss = BI_STACK[ph]
    print(f'\n  {ph}   vespagram 疊加 {st} +- {ss} s   n = {len(BI_CORE[ph])}')
    print(f'{"":<6}{"event":<9}{"t-tP":>9}{"sig":>6}{"Bi Delta":>14}'
          f'{"IPGP Delta":>15}{"d(Delta)":>10}{"in TT_DATA":>12}')
    for ev, (v, s) in sorted(BI_CORE[ph].items(), key=lambda x: -x[1][0]):
        gb, gd = BI_GEOM.get(ev), DR_GEOM.get(ev)
        cb = f'{gb[0]:8.1f}+-{gb[1]:.1f}' if gb else f'{"--":>14}'
        cd = f'{gd[0]:9.1f}+-{gd[1]:.1f}' if gd else f'{"--":>15}'
        dd = f'{gb[0]-gd[0]:+10.1f}' if (gb and gd) else f'{"--":>10}'
        print(f'{"":<6}{ev:<9}{v:>9.1f}{s:>6.1f}{cb}{cd}{dd}'
              f'{("Y" if ev in TT else ""):>12}')
print('\n  PKiKP 反射於 ICB —— 唯一直接約束內核的資料。')
print('  警告:Bi 的 Delta 來自 MQS V14,與 IPGP 鏈不同(S0474a 差 9.4 度),')
print('        混用會讓同一事件出現兩個位置。')

# ══════════════════════════════════════════════════════════════════════════
# 圖
# ══════════════════════════════════════════════════════════════════════════
if SAVE_FIG:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ph_plot = [p for p in PHASES if p in RATIO]
    x = np.arange(len(ph_plot))
    med = [np.median([r['s_sys'] for r in records
                      if r['ph'] == p and np.isfinite(r['s_sys'])]) for p in ph_plot]
    ax1.bar(x-0.2, [SIG_DR[p] for p in ph_plot], 0.4,
            label=r'$\sigma_{pick}$  Drilleau 2024', color='#849DAB')
    ax1.bar(x+0.2, med, 0.4, label=r'$\sigma_{sys}$  IPGP vs ETH', color='#CD5C5C')
    ax1.set_xticks(x); ax1.set_xticklabels(ph_plot, rotation=45, ha='right')
    ax1.set_ylabel('s', fontsize=fontsize)
    ax1.tick_params(labelsize=14)
    ax1.legend(loc='upper left', fontsize=12)

    evs = sorted(set(DR_GEOM) & set(KH_GEOM) & set(BI_GEOM))
    xx = np.arange(len(evs))
    for lab, G, c, m, off in [('IPGP (Drilleau 2024)', DR_GEOM, '#CD5C5C','o',-0.25),
                              ('ETH (Khan 2026)',      KH_GEOM, '#35838D','s', 0.0),
                              ('Bi 2025 (MQS V14)',    BI_GEOM, '#414F67','^', 0.25)]:
        ax2.errorbar(xx+off, [G[e][0] for e in evs], yerr=[G[e][1] for e in evs],
                     fmt=m, color=c, ms=7, capsize=3, lw=1.2, label=lab)
    ax2.set_xticks(xx); ax2.set_xticklabels(evs, rotation=45, ha='right')
    ax2.set_ylabel('epicentral distance (deg)', fontsize=fontsize)
    ax2.tick_params(labelsize=14)
    ax2.legend(loc='lower right', fontsize=11)

    fig.tight_layout()
    fig.savefig('85_tt_dataset_audit.png', dpi=150)
    plt.show()