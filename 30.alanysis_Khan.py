#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:03:26 2026

@author: chingchen
"""

"""
analyze_khan_lsl.py

分析1000個Khan模型的LSL結構統計：
1. LSL頂部深度分佈
2. 真正CMB深度分佈
3. LSL厚度分佈
4. 密度穩定性：
   - rho_lsl_top vs rho_mantle_bottom（上界面）
   - rho_lsl_bottom vs rho_core_top（下界面）
5. LSL頂部密度分佈
"""

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


KHAN_MODEL_DIR = "/Users/chingchen/Desktop/Lunar/Mars_Khan_2023/LSL_Models/"   

# ============================================================
def parse_model(fpath):
    """
    解析單一Model_*.txt，回傳LSL和CMB的統計資訊。
    回傳 dict 或 None（解析失敗）。
    """
    try:
        data  = np.loadtxt(fpath, comments='#')
        depth = data[:, 0]
        Vp    = data[:, 1]
        Vs    = data[:, 2]
        rho   = data[:, 3]
    except Exception:
        return None

    solid_mask  = Vs > 0.01
    liquid_mask = Vs < 0.01

    if solid_mask.sum() < 5 or liquid_mask.sum() < 5:
        return None

    # ── LSL頂部：第一個Vs=0的深度 ────────────────────────
    lsl_top_depth   = float(depth[liquid_mask][0])
    rho_lsl_top     = float(rho[liquid_mask][0])

    # 緊鄰LSL頂部的固態地幔底部密度
    # 找liquid_mask第一個True之前的最後一個solid點
    first_liq_idx = np.where(liquid_mask)[0][0]
    if first_liq_idx == 0:
        return None
    rho_mantle_bottom = float(rho[first_liq_idx - 1])

    # ── 真正CMB：液態區裡密度最大跳變 ───────────────────
    liq_depth = depth[liquid_mask]
    liq_rho   = rho[liquid_mask]
    drho      = np.diff(liq_rho)
    ddepth    = np.diff(liq_depth)

    # 同一介面（depth幾乎不變）且密度大跳(>1 g/cm³)
    disc_mask = (np.abs(ddepth) < 1.0) & (drho > 1.0)
    if not disc_mask.any():
        # fallback
        best_idx = np.argmax(drho)
    else:
        best_idx = np.argmax(drho * disc_mask)

    true_cmb_depth = float(liq_depth[best_idx + 1])
    rho_core_top   = float(liq_rho[best_idx + 1])

    # LSL底部密度（CMB之前的最後一個液態點）
    rho_lsl_bottom = float(liq_rho[best_idx])

    lsl_thickness = true_cmb_depth - lsl_top_depth

    # ── 密度穩定性 ───────────────────────────────────────
    # 上界面：rho_lsl_top - rho_mantle_bottom
    # 正值 = 穩定（LSL比地幔重），負值 = 不穩定
    upper_contrast = rho_lsl_top - rho_mantle_bottom

    # 下界面：rho_core_top - rho_lsl_bottom
    # 正值 = 穩定（核心比LSL重），負值 = 不穩定
    lower_contrast = rho_core_top - rho_lsl_bottom

    return {
        'file':               os.path.basename(fpath),
        'lsl_top_depth':      lsl_top_depth,
        'true_cmb_depth':     true_cmb_depth,
        'lsl_thickness':      lsl_thickness,
        'rho_mantle_bottom':  rho_mantle_bottom,
        'rho_lsl_top':        rho_lsl_top,
        'rho_lsl_bottom':     rho_lsl_bottom,
        'rho_core_top':       rho_core_top,
        'upper_contrast':     upper_contrast,   # lsl_top - mantle_bottom
        'lower_contrast':     lower_contrast,   # core_top - lsl_bottom
        'upper_stable':       upper_contrast > 0,
        'lower_stable':       lower_contrast > 0,
        'fully_stable':       (upper_contrast > 0) and (lower_contrast > 0),
    }

# ============================================================
def main():
    files = sorted(glob.glob(os.path.join(KHAN_MODEL_DIR, 'Model_*.txt')))
    print(f"Found {len(files)} model files")

    results = []
    for fpath in files:
        r = parse_model(fpath)
        if r is not None:
            results.append(r)

    print(f"Successfully parsed: {len(results)} models\n")

    if len(results) == 0:
        print("No models parsed. Check KHAN_MODEL_DIR path.")
        return

    # ── 轉成 numpy arrays ───────────────────────────────
    lsl_top      = np.array([r['lsl_top_depth']     for r in results])
    true_cmb     = np.array([r['true_cmb_depth']    for r in results])
    thickness    = np.array([r['lsl_thickness']     for r in results])
    rho_man_bot  = np.array([r['rho_mantle_bottom'] for r in results])
    rho_lsl_t    = np.array([r['rho_lsl_top']       for r in results])
    rho_lsl_b    = np.array([r['rho_lsl_bottom']    for r in results])
    rho_core_t   = np.array([r['rho_core_top']      for r in results])
    upper_cont   = np.array([r['upper_contrast']    for r in results])
    lower_cont   = np.array([r['lower_contrast']    for r in results])

    upper_stable = np.array([r['upper_stable'] for r in results])
    lower_stable = np.array([r['lower_stable'] for r in results])
    fully_stable = np.array([r['fully_stable'] for r in results])

    # ── 統計摘要 ─────────────────────────────────────────
    print("=" * 55)
    print("LSL DEPTH STATISTICS")
    print("=" * 55)
    print(f"  LSL top depth   : {lsl_top.mean():.1f} ± {lsl_top.std():.1f} km"
          f"  [{lsl_top.min():.0f} ~ {lsl_top.max():.0f}]")
    print(f"  True CMB depth  : {true_cmb.mean():.1f} ± {true_cmb.std():.1f} km"
          f"  [{true_cmb.min():.0f} ~ {true_cmb.max():.0f}]")
    print(f"  LSL thickness   : {thickness.mean():.1f} ± {thickness.std():.1f} km"
          f"  [{thickness.min():.0f} ~ {thickness.max():.0f}]")

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
    print(f"    stable (>0)   : {upper_stable.sum()} / {len(results)}"
          f" ({upper_stable.mean()*100:.1f}%)")
    print(f"    unstable (<0) : {(~upper_stable).sum()} / {len(results)}"
          f" ({(~upper_stable).mean()*100:.1f}%)")
    if (~upper_stable).any():
        print(f"    worst case    : {upper_cont.min():.4f} g/cm³")

    print(f"\n  Lower interface (core_top - LSL_bottom):")
    print(f"    mean contrast : {lower_cont.mean():.4f} g/cm³")
    print(f"    stable (>0)   : {lower_stable.sum()} / {len(results)}"
          f" ({lower_stable.mean()*100:.1f}%)")
    print(f"    unstable (<0) : {(~lower_stable).sum()} / {len(results)}"
          f" ({(~lower_stable).mean()*100:.1f}%)")

    print(f"\n  Fully stable (both interfaces):")
    print(f"    {fully_stable.sum()} / {len(results)}"
          f" ({fully_stable.mean()*100:.1f}%)")

    # ── 建議的穩定性判斷閾值 ─────────────────────────────
    print("\n" + "=" * 55)
    print("SUGGESTED STABILITY THRESHOLD")
    print("=" * 55)
    pct_5  = np.percentile(upper_cont, 5)
    pct_25 = np.percentile(upper_cont, 25)
    print(f"  Upper contrast 5th  percentile: {pct_5:.4f} g/cm³")
    print(f"  Upper contrast 25th percentile: {pct_25:.4f} g/cm³")
    print(f"  → If threshold = 0.0  : accept {upper_stable.mean()*100:.1f}% of Khan models")
    for thr in [-0.05, -0.10, -0.15]:
        frac = (upper_cont > thr).mean() * 100
        print(f"  → If threshold = {thr:+.2f}: accept {frac:.1f}% of Khan models")

    # ── 圖 ───────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(lsl_top, bins=30, color='steelblue', edgecolor='white')
    ax1.axvline(lsl_top.mean(), color='red', lw=2,
                label=f'mean={lsl_top.mean():.0f} km')
    ax1.set_xlabel('LSL top depth [km]')
    ax1.set_ylabel('Count')
    ax1.set_title('LSL Top Depth')
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(thickness, bins=30, color='darkorange', edgecolor='white')
    ax2.axvline(thickness.mean(), color='red', lw=2,
                label=f'mean={thickness.mean():.0f} km')
    ax2.set_xlabel('LSL thickness [km]')
    ax2.set_title('LSL Thickness')
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(rho_lsl_t, bins=30, color='seagreen', edgecolor='white',
             label='LSL top', alpha=0.7)
    ax3.hist(rho_man_bot, bins=30, color='brown', edgecolor='white',
             label='Mantle bottom', alpha=0.7)
    ax3.set_xlabel('Density [g/cm³]')
    ax3.set_title('LSL top vs Mantle bottom ρ')
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(upper_cont, bins=40,
             color=['salmon' if v < 0 else 'steelblue' for v in
                    np.histogram(upper_cont, bins=40)[1][:-1]],
             edgecolor='white')
    # 重新畫（histogram顏色需要分段處理，這裡簡化）
    ax4.cla()
    neg_mask = upper_cont < 0
    ax4.hist(upper_cont[~neg_mask], bins=30, color='steelblue',
             edgecolor='white', label=f'stable ({(~neg_mask).sum()})')
    ax4.hist(upper_cont[neg_mask],  bins=15, color='salmon',
             edgecolor='white', label=f'unstable ({neg_mask.sum()})')
    ax4.axvline(0, color='red', lw=2, ls='--')
    ax4.set_xlabel('ρ_LSL_top − ρ_mantle_bottom [g/cm³]')
    ax4.set_title('Upper Interface Stability')
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[1, 1])
    neg_mask2 = lower_cont < 0
    ax5.hist(lower_cont[~neg_mask2], bins=30, color='steelblue',
             edgecolor='white', label=f'stable ({(~neg_mask2).sum()})')
    if neg_mask2.any():
        ax5.hist(lower_cont[neg_mask2], bins=10, color='salmon',
                 edgecolor='white', label=f'unstable ({neg_mask2.sum()})')
    ax5.axvline(0, color='red', lw=2, ls='--')
    ax5.set_xlabel('ρ_core_top − ρ_LSL_bottom [g/cm³]')
    ax5.set_title('Lower Interface Stability')
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(upper_cont, lower_cont,
                c=['green' if r['fully_stable'] else 'red' for r in results],
                alpha=0.4, s=10)
    ax6.axvline(0, color='black', lw=1, ls='--')
    ax6.axhline(0, color='black', lw=1, ls='--')
    ax6.set_xlabel('Upper contrast [g/cm³]')
    ax6.set_ylabel('Lower contrast [g/cm³]')
    ax6.set_title(f'Stability Map\n'
                  f'green=fully stable ({fully_stable.sum()}), '
                  f'red=unstable ({(~fully_stable).sum()})')

    fig.suptitle('Khan Model LSL Density Analysis', fontsize=14, y=1.01)
    plt.savefig('khan_lsl_analysis.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved: khan_lsl_analysis.png")
    plt.show()


if __name__ == '__main__':
    main()