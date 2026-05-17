"""
plot_raypath_final.py

正確版 ray path 圖（座標已驗證）：
- InSight 在頂部中央 (0, R)
- dist=0 → 震源，dist=delta → InSight
- theta_from_InSight = radians(delta) - dist
- x = r * sin(theta), y = r * cos(theta)
- LineCollection 的 alpha 反映 ray density
- 右側 radial sensitivity bar

修改 CONFIG 區塊即可套用到你的 Mars 模型。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import importlib.util
from obspy.taup import TauPyModel

# ============================================================
# CONFIG：修改這裡
# ============================================================

MCMC_MODULE_PATH = "/home/jcchen2/mars-hefesto-inversion/21.mcmc_NPS.py"
TAUP_MODEL_PATH  = ("/net/beno3/data1/jcchen2/"
                    "mars-hefesto-runs/taup_models/best_fit_BML_plot.npz")
OUTPUT_PATH      = ("/net/beno3/data1/jcchen2/"
                    "mars-hefesto-runs/mcmc/mars_raypaths_final.png")

R_MARS        = 3389.5   # km
CMB_DEPTH     = 1752.5   # km，修改成你的模型值
BML_TOP_DEPTH = 1582.5   # km = CMB_DEPTH - 150

# density grid 解析度（越大越精細但越慢）
N_GRID = 300

# ============================================================
# 震相設定
# ============================================================

PHASE_CONFIG = {
    'P':    {'color': '#FFAA77', 'lw': 0.8,  'label': 'P'},
    'PP':   {'color': '#FF5500', 'lw': 1.0,  'label': 'PP'},
    'S':    {'color': '#AACCFF', 'lw': 0.8,  'label': 'S'},
    'SS':   {'color': '#4488EE', 'lw': 1.0,  'label': 'SS'},
    'SSS':  {'color': '#4488EE', 'lw': 0.7,  'label': 'SSS'},
    'ScS':  {'color': '#1144BB', 'lw': 1.2,  'label': 'ScS'},
    'SKS':  {'color': '#001188', 'lw': 1.5,  'label': 'SKS'},
    'Pdiff':{'color': '#CC1111', 'lw': 2.0,  'label': 'Pdiff'},
}

# ============================================================
# 核心函數
# ============================================================

def to_xy(dist_rad, depth_km, delta_deg, R_planet):
    """
    TauP 座標 → 圖面 xy。
    dist_rad: TauP path['dist']，0=震源，delta=InSight
    翻轉後 InSight 在 theta=0（頂部）。
    """
    r     = R_planet - depth_km
    theta = np.radians(delta_deg) - dist_rad  # 翻轉
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return x, y


def collect_paths_and_density(taup, samuel_data, phase_list, R_planet, n_grid):
    """
    同時收集 ray path segments 和累積 density grid。
    回傳:
      all_segs: {phase: [(segs, alphas_placeholder), ...]}
      density:  ndarray (n_r, n_t)，已正規化
    """
    density  = np.zeros((n_grid, n_grid))
    all_segs = {p: [] for p in phase_list}

    for event, obs in samuel_data.items():
        delta = obs['delta']
        depth = obs.get('depth', 10.0)

        try:
            paths = taup.get_ray_paths(
                source_depth_in_km=depth,
                distance_in_degree=delta,
                phase_list=phase_list,
            )
        except Exception as e:
            print(f"  {event}: TauP error: {e}")
            continue

        seen = set()
        for path in paths:
            name = path.name
            if name not in all_segs or name in seen:
                continue
            seen.add(name)

            dist_arr  = path.path['dist']    # rad, 0=震源
            depth_arr = path.path['depth']
            r_arr     = R_planet - depth_arr
            theta_arr = np.radians(delta) - dist_arr  # 翻轉

            # 累積 density
            for t, r in zip(theta_arr, r_arr):
                if 0 <= t <= np.pi and 0 < r <= R_planet:
                    i_r = int(np.clip(r / R_planet * (n_grid - 1), 0, n_grid - 1))
                    i_t = int(np.clip(t / np.pi   * (n_grid - 1), 0, n_grid - 1))
                    density[i_r, i_t] += 1

            # 收集 LineCollection segments
            xs, ys = to_xy(dist_arr, depth_arr, delta, R_planet)
            pts  = np.stack([xs, ys], axis=1)
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            all_segs[name].append({
                'segs':      segs,
                'theta_arr': theta_arr,
                'r_arr':     r_arr,
            })

    
    if density.max() > 0:
        density /= density.max()

    return all_segs, density


def draw_background(ax, R_planet, cmb_depth, bml_top_depth):
    theta = np.linspace(-np.pi / 2, np.pi / 2, 400)

    # 行星表面
    ax.plot(R_planet * np.sin(theta), R_planet * np.cos(theta),
            'k-', lw=1.8, zorder=10)

    # BML top（橘色虛線）
    r_bml = R_planet - bml_top_depth
    ax.plot(r_bml * np.sin(theta), r_bml * np.cos(theta),
            color='#CC8844', lw=1.0, ls='--', zorder=2, alpha=0.7)

    # CMB（深橘虛線）
    r_cmb = R_planet - cmb_depth
    ax.plot(r_cmb * np.sin(theta), r_cmb * np.cos(theta),
            color='#884400', lw=1.3, ls='--', zorder=2, alpha=0.8)

    # 參考同心圓
    step = 500
    d = step
    while d < R_planet:
        r = R_planet - d
        ax.plot(r * np.sin(theta), r * np.cos(theta),
                color='#CCCCCC', lw=0.3, ls=':', zorder=0)
        d += step


def draw_raypaths(ax, all_segs, density, phase_config, R_planet, n_grid):
    for phase, seg_list in all_segs.items():
        if phase not in phase_config or not seg_list:
            continue
        cfg  = phase_config[phase]
        base = mcolors.to_rgb(cfg['color'])
        lw   = cfg['lw']

        for item in seg_list:
            segs      = item['segs']
            theta_arr = item['theta_arr']
            r_arr     = item['r_arr']

            # 每個 segment 的 alpha = f(local density)
            alphas = []
            for t, r in zip(theta_arr[:-1], r_arr[:-1]):
                if 0 <= t <= np.pi and 0 < r <= R_planet:
                    i_r = int(np.clip(r / R_planet * (n_grid-1), 0, n_grid-1))
                    i_t = int(np.clip(t / np.pi   * (n_grid-1), 0, n_grid-1))
                    d_val = density[i_r, i_t]
                else:
                    d_val = 0.0
                # 密度高 → 不透明；密度低 → 半透明
                alphas.append(float(np.clip(d_val * 1.5 + 0.25, 0.15, 0.95)))

            colors_rgba = [(*base, a) for a in alphas]
            lc = LineCollection(segs, colors=colors_rgba,
                                linewidths=lw, zorder=3,
                                capstyle='round')
            ax.add_collection(lc)


def draw_events(ax, samuel_data, R_planet):
    for event, obs in samuel_data.items():
        delta = obs['delta']
        depth = obs.get('depth', 10.0)
        r     = R_planet - depth
        theta = np.radians(delta)
        x = r * np.sin(theta)
        y = r * np.cos(theta)

        ax.plot(x, y, '*', color='white', markersize=11,
                markeredgecolor='black', markeredgewidth=0.8, zorder=18)

        if delta > 50:
            ax.annotate(event, (x, y),
                        xytext=(10, 4), textcoords='offset points',
                        fontsize=7.5, color='#111111', zorder=19)


def draw_sensitivity_bar(ax, density, R_planet, n_grid,
                         bar_frac=0.06):
    """
    在圖右側畫垂直 radial sensitivity bar。
    sensitivity(r) = sum over theta（該半徑的總 ray coverage）。
    """
    sensitivity = density.sum(axis=1)
    if sensitivity.max() > 0:
        sensitivity /= sensitivity.max()

    bar_x0 = R_planet * 1.08
    bar_x1 = R_planet * (1.08 + bar_frac)
    r_vals  = np.linspace(0, R_planet, n_grid)

    cmap = plt.cm.Purples
    norm = mcolors.Normalize(0, 1)

    for i in range(n_grid - 1):
        c = cmap(norm(sensitivity[i]))
        ax.fill_between(
            [bar_x0, bar_x1],
            [r_vals[i],   r_vals[i]],
            [r_vals[i+1], r_vals[i+1]],
            color=c, linewidth=0, zorder=4
        )

    # 邊框
    ax.plot([bar_x0, bar_x1, bar_x1, bar_x0, bar_x0],
            [0, 0, R_planet, R_planet, 0],
            'k-', lw=0.6, zorder=5)

    # 標籤
    ax.text((bar_x0 + bar_x1) / 2, -R_planet * 0.07,
            'Radial\nsensitivity', ha='center', fontsize=8, color='#333333')

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.25, pad=0.01,
                        location='right', label='Ray density')
    cbar.ax.tick_params(labelsize=7)


def build_legend(ax, phase_config, cmb_depth, bml_depth):
    elems = []
    for ph, cfg in phase_config.items():
        elems.append(Line2D([0], [0], color=cfg['color'],
                            lw=max(cfg['lw'], 1.2), label=cfg['label']))
    elems += [
        Line2D([0],[0], color='#CC8844', lw=1.0, ls='--',
               label=f'BML top ({bml_depth:.0f} km)'),
        Line2D([0],[0], color='#884400', lw=1.3, ls='--',
               label=f'CMB ({cmb_depth:.0f} km)'),
    ]
    ax.legend(handles=elems, loc='upper left',
              fontsize=8, ncol=2, framealpha=0.88)

# ============================================================
# Main
# ============================================================

def main():
    # ── 載入 ──────────────────────────────────────────────────
    print("Loading MCMC module...")
    spec = importlib.util.spec_from_file_location("mcmc", MCMC_MODULE_PATH)
    mcmc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mcmc)
    mcmc.load_gravity_profile()
    samuel_data = mcmc.KHAN_DATA

    print(f"Loading TauP model: {TAUP_MODEL_PATH}")
    taup = TauPyModel(model=TAUP_MODEL_PATH)

    phase_list = list(PHASE_CONFIG.keys())
    print(f"Computing ray paths for {len(samuel_data)} events, "
          f"phases: {phase_list}")

    all_segs, density = collect_paths_and_density(
        taup, samuel_data, phase_list, R_MARS, N_GRID)

    # ── 繪圖 ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_aspect('equal')
    ax.axis('off')

    draw_background(ax, R_MARS, CMB_DEPTH, BML_TOP_DEPTH)
    draw_raypaths(ax, all_segs, density, PHASE_CONFIG, R_MARS, N_GRID)
    draw_events(ax, samuel_data, R_MARS)
    draw_sensitivity_bar(ax, density, R_MARS, N_GRID)
    build_legend(ax, PHASE_CONFIG, CMB_DEPTH, BML_TOP_DEPTH)

    # InSight
    ax.plot(0, R_MARS, '^', color='black', markersize=13, zorder=20)
    ax.text(0, R_MARS + 130, 'InSight', ha='center',
            fontsize=10, fontweight='bold')

    ax.set_xlim(-R_MARS * 1.05, R_MARS * 1.22)
    ax.set_ylim(-R_MARS * 0.12, R_MARS * 1.12)
    ax.set_title('Mars body wave ray paths\n(best-fit BML model)',
                 fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()