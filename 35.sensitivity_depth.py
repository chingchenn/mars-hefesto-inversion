"""
plot_event_phase_summary.py

對每個有 observation data 的 event，選最好的 N 個 TauP models，
畫一張三列 subplots 的 PNG：
  - 左欄: 所有 phases 的 ray path 總覽，10 個 models 半透明疊加
  - 中欄: 每個 differential phase 一列，phase_a vs phase_b ray path 疊加比較
           （標題標出 Khan/Samuel 觀測 ΔT）
  - 右欄: scatter summary（X=diff phase，Y=深度差中位數，點大小=|深度差|，顏色=Samuel−Khan）

用法: python plot_event_phase_summary.py
"""

import os, json, re, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import importlib.util

from obspy.taup import TauPyModel
from mars_config import FIG_DIR, MCMC_DIR, TAUP_WORK_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script

# ─────────────────────────────────────────────────
# 載入 mcmc module（取 OBS_DATA 等設定）
# ─────────────────────────────────────────────────
MCMC_MODULE_PATH = f"{HERE}/21.mcmc_NPS.py"

spec = importlib.util.spec_from_file_location("mcmc", MCMC_MODULE_PATH)
mcmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcmc)

# ─────────────────────────────────────────────────
# config
# ─────────────────────────────────────────────────
TAUP_DIR  = TAUP_WORK_DIR
OUT_DIR   = f'{FIG_DIR}/event_phase_summary'
os.makedirs(OUT_DIR, exist_ok=True)

PREFIX      = "chain_a07"
BURNIN_FRAC = 0.1
STEP_MAX    = 150
N_BEST      = 20

# 兩套觀測資料
KHAN_DATA   = mcmc.KHAN_DATA
SAMUEL_DATA = mcmc.SAMUEL_DATA

# 你提供的 Khan/Samuel 差異表格
OBS_CSV_TEXT = """\
Event,Phase,Khan,Samuel,Samuel_minus_Khan_s
S0173a,S-P,174.8,178.8,4.0
S0173a,SS-S,23.099999999999994,24.4,1.3000000000000043
S0173a,ScS-S,340.2,345.2,5.0
S0173a,sS-S,10.0,13.2,3.1999999999999993
S0185a,S-P,360.2,327.28,-32.920000000000016
S0185a,SSS-S,52.30000000000001,55.4,3.099999999999987
S0185a,sS-S,19.30000000000001,10.0,-9.300000000000011
S0235b,PP-P,17.4,18.6,1.2000000000000028
S0235b,PPP-P,31.1,32.0,0.8999999999999986
S0235b,S-P,166.0,171.4,5.400000000000006
S0235b,SS-S,27.099999999999994,23.2,-3.899999999999995
S0235b,ScS-S,346.0,343.9,-2.1000000000000227
S0235b,sS-S,12.699999999999989,9.2,-3.4999999999999893
S0325a,S-P,230.7,229.3,-1.3999999999999773
S0325a,SS-S,30.0,26.1,-3.8999999999999986
S0325a,SSS-S,50.30000000000001,50.3,-1.4210854715202004e-14
S0325a,pP-P,11.3,9.8,-1.5
S0407a,PP-P,17.8,23.38,5.579999999999998
S0407a,S-P,172.2,170.7,-1.5
S0407a,SS-S,24.200000000000017,21.1,-3.1000000000000156
S0407a,sS-S,11.200000000000017,13.3,2.0999999999999837
S0409d,S-P,162.5,163.2,0.6999999999999886
S0409d,SS-S,22.400000000000006,20.9,-1.500000000000007
S0409d,SSS-S,45.19999999999999,39.8,-5.3999999999999915
S0484b,PP-P,18.1,19.73,1.629999999999999
S0484b,S-P,170.2,173.1,2.9000000000000057
S0484b,SS-S,26.600000000000023,17.4,-9.200000000000024
S0484b,sS-S,13.800000000000011,13.0,-0.8000000000000114
S0784a,PP-P,15.2,13.7,-1.5
S0784a,PPP-P,29.5,22.4,-7.100000000000001
S0784a,S-P,173.0,179.3,6.300000000000011
S0784a,SS-S,23.900000000000006,19.6,-4.300000000000004
S0784a,SSS-S,48.900000000000006,28.0,-20.900000000000006
S0784a,sS-S,9.199999999999989,7.2,-1.9999999999999885
S0802a,PP-P,17.3,25.6,8.3
S0802a,S-P,176.3,180.3,4.0
S0802a,SS-S,25.19999999999999,22.4,-2.79999999999999
S0802a,SSS-S,45.79999999999998,36.5,-9.299999999999983
S0809a,PP-P,15.5,16.25,0.75
S0809a,PPP-P,29.3,29.65,0.34999999999999787
S0809a,S-P,175.0,191.95,16.94999999999999
S0809a,SS-S,22.5,23.8,1.3000000000000007
S0820a,PP-P,15.7,21.9,6.199999999999999
S0820a,S-P,176.5,174.1,-2.4000000000000057
S0820a,sS-S,9.699999999999989,8.5,-1.1999999999999886
S0864a,PP-P,18.0,18.0,0.0
S0864a,S-P,169.0,171.4,2.4000000000000057
S0864a,SS-S,25.0,26.4,1.3999999999999986
S0864a,sS-S,12.199999999999989,17.3,5.100000000000012
"""

# phases 清單（ray path 要算的）
ALL_PHASES = ["P", "S", "PP", "PPP", "SS", "SSS", "ScS", "sS", "pP"]

# diff phases（對應 obs CSV 裡的 Phase 欄位）
DIFF_PHASES = {
    "S-P":    ("S",   "P"),
    "PP-P":   ("PP",  "P"),
    "PPP-P":  ("PPP", "P"),
    "SS-S":   ("SS",  "S"),
    "SSS-S":  ("SSS", "S"),
    "ScS-S":  ("ScS", "S"),
    "sS-S":   ("sS",  "S"),
    "pP-P":   ("pP",  "P"),
}

# phase 配色（ray path 圖使用）
PHASE_COLORS = {
    "P":   "#4C72B0",
    "S":   "#DD8452",
    "PP":  "#55A868",
    "PPP": "#C44E52",
    "SS":  "#8172B2",
    "SSS": "#937860",
    "ScS": "#DA8BC3",
    "sS":  "#8C8C8C",
    "pP":  "#CCB974",
}

# ─────────────────────────────────────────────────
# 解析 OBS CSV
# ─────────────────────────────────────────────────
obs_df = pd.read_csv(io.StringIO(OBS_CSV_TEXT))
obs_df["Samuel_minus_Khan_s"] = obs_df["Samuel_minus_Khan_s"].round(2)
EVENTS_WITH_OBS = obs_df["Event"].unique().tolist()


# ─────────────────────────────────────────────────
# 選 best models
# ─────────────────────────────────────────────────
def load_best_models():
    selected = []
    chain_dirs = sorted([
        d for d in os.listdir(MCMC_DIR)
        if d.startswith(PREFIX + "_")
        and os.path.isdir(os.path.join(MCMC_DIR, d))
    ])
    for chain_dir in chain_dirs:
        chain_file = os.path.join(MCMC_DIR, chain_dir, "chain.json")
        if not os.path.exists(chain_file):
            continue
        with open(chain_file) as f:
            chain = json.load(f)
        burnin_n = int(len(chain) * BURNIN_FRAC)
        chain = chain[burnin_n:]
        m = re.search(r"_(\d+)$", chain_dir)
        if m is None:
            continue
        cid = int(m.group(1))
        for entry in chain:
            step = entry["step"]
            if step >= STEP_MAX:
                continue
            model_name = f"mcmc_c{cid:02d}_s{step:05d}"
            npz_path = os.path.join(TAUP_DIR, model_name + ".npz")
            if os.path.exists(npz_path):
                selected.append({
                    "model_name": model_name,
                    "misfit": entry["misfit"],
                    "npz_path": npz_path,
                })
    selected = sorted(selected, key=lambda x: x["misfit"])
    return selected[:N_BEST]


# ─────────────────────────────────────────────────
# 計算 ray paths + phase depths
# ─────────────────────────────────────────────────
def get_ray_paths_and_depths(model_info, event_name):
    """
    Returns:
        ray_paths: dict {phase: list of (dist_deg_array, depth_array)}
                   每個 model 的 path 都加進去（最多 1 條 first arrival）
        phase_info: dict {phase: {"max_depth": float, "time": float}}
    """
    obs = KHAN_DATA.get(event_name) or SAMUEL_DATA.get(event_name)
    if obs is None:
        return {}, {}

    delta = obs["delta"]
    depth = obs.get("depth", 10.0)

    try:
        taup = TauPyModel(model=model_info["npz_path"])
    except Exception as e:
        print(f"  [WARN] Cannot load model {model_info['model_name']}: {e}")
        return {}, {}

    ray_paths = {}
    phase_info = {}

    for ph in ALL_PHASES:
        try:
            arrivals = taup.get_ray_paths(
                source_depth_in_km=depth,
                distance_in_degree=delta,
                phase_list=[ph],
            )
        except Exception:
            continue

        if not arrivals:
            continue

        arr = min(arrivals, key=lambda a: a.time)
        dist_deg = np.rad2deg(arr.path["dist"])
        dep = arr.path["depth"]
        deep_path = compute_deep_path_length(arr, depth_threshold=1600)

        if ph not in ray_paths:
            ray_paths[ph] = []
        ray_paths[ph].append((dist_deg, dep))

        phase_info[ph] = {
            "deep_path": deep_path,
            "time": arr.time,
        }

    return ray_paths, phase_info

def compute_deep_path_length(arr, depth_threshold=1000):

    dep  = arr.path["depth"]
    dist = arr.path["dist"]   # radians

    if len(dep) < 2:
        return 0.0

    # Mars radius
    R = 3389.5

    deep_length = 0.0

    for i in range(len(dep)-1):

        dmid = 0.5 * (dep[i] + dep[i+1])

        if dmid < depth_threshold:
            continue

        r1 = R - dep[i]
        r2 = R - dep[i+1]

        dr = r2 - r1
        ds = 0.5 * (r1 + r2) * (dist[i+1] - dist[i])

        seglen = np.sqrt(dr**2 + ds**2)

        deep_length += abs(seglen)

    return float(deep_length)

# ─────────────────────────────────────────────────
# 畫一個事件的三欄 subplot
# ─────────────────────────────────────────────────
def plot_event(event_name, best_models):
    print(f"  Plotting {event_name} ...")

    # 觀測資料
    ev_obs = obs_df[obs_df["Event"] == event_name].copy()
    diff_phase_list = ev_obs["Phase"].tolist()

    # 累積各 model 的 ray paths 和 depth info
    all_ray_paths = {}   # {phase: [(dist, dep), ...]}
    all_depth_info = []  # list of dict {diff_phase: extra_depth}

    for model_info in best_models:
        ray_paths, phase_info = get_ray_paths_and_depths(model_info, event_name)

        # 合併 ray paths
        for ph, paths in ray_paths.items():
            if ph not in all_ray_paths:
                all_ray_paths[ph] = []
            all_ray_paths[ph].extend(paths)

        # 記錄每個 phase 的 max depth（per model）
        row = {}
        for ph, info in phase_info.items():
            row[ph] = info["deep_path"]
        all_depth_info.append(row)

    # ── 聚合：每個 phase 的 max depth 分布（across 10 models）
    phase_path_stats = {}
    all_phases_seen = set(ph for row in all_depth_info for ph in row)
    for ph in all_phases_seen:
        vals = [r[ph] for r in all_depth_info if ph in r]
        if vals:
            phase_path_stats[ph] = {
                "median": float(np.median(vals)),
                "q25":    float(np.percentile(vals, 25)),
                "q75":    float(np.percentile(vals, 75)),
            }

    # ── 對齊 diff phases：只顯示 obs_df 裡有的，且兩個 phase 都有深度資料
    obs_phases_present = [
        p for p in diff_phase_list
        if p in DIFF_PHASES
        and DIFF_PHASES[p][0] in phase_path_stats
        and DIFF_PHASES[p][1] in phase_path_stats
    ]

    # ─────────────────────────────────────────────
    # 動態 layout：
    #   第 0 欄：所有 phases 的 ray path 總覽（跨所有 rows）
    #   第 1 欄：每個 diff phase 一個 ax（phase_a vs phase_b ray path）
    #   第 2 欄：scatter summary（跨所有 rows）
    # ─────────────────────────────────────────────
    from matplotlib.gridspec import GridSpec

    n_diff = max(len(diff_phase_list), 1)
    fig = plt.figure(figsize=(6 + n_diff * 3.0 + 5, max(n_diff * 2.8, 6)))
    fig.suptitle(f"Event {event_name}  —  top {N_BEST} models", fontsize=14, fontweight="bold", y=1.01)

    gs = GridSpec(n_diff, 3, figure=fig, wspace=0.38, hspace=0.50,
                  width_ratios=[2.5, 2.5, 2.0])

    ax1  = fig.add_subplot(gs[:, 0])
    ax3  = fig.add_subplot(gs[:, 2])
    ax2s = [fig.add_subplot(gs[i, 1]) for i in range(n_diff)]

    # ══════════════════════════════════════════════
    # 左圖: 所有 phases 的 ray path 總覽
    # ══════════════════════════════════════════════
    plotted_phases = set()

    for ph, paths in all_ray_paths.items():
        color = PHASE_COLORS.get(ph, "gray")
        for dist_deg, dep in paths:
            label = ph if ph not in plotted_phases else "_nolegend_"
            ax1.plot(dist_deg, dep, color=color, alpha=0.35, lw=1.2, label=label)
            plotted_phases.add(ph)

    ax1.invert_yaxis()
    ax1.set_xlabel("Angular distance along ray (°)", fontsize=10)
    ax1.set_ylabel("Depth (km)", fontsize=10)
    ax1.set_title("All ray paths", fontsize=11)
    ax1.legend(fontsize=8, ncol=2, loc="lower right")
    ax1.grid(True, ls="--", alpha=0.3)

    # ══════════════════════════════════════════════
    # 中欄: 每個 diff phase — phase_a vs phase_b ray path
    # ══════════════════════════════════════════════
    COLOR_A = "#C44E52"   # phase_a（較深的那個）
    COLOR_B = "#4C72B0"   # phase_b（reference）

    for i, dph in enumerate(diff_phase_list):
        ax2 = ax2s[i]
        pair = DIFF_PHASES.get(dph)
        if pair is None:
            ax2.text(0.5, 0.5, f"{dph}\n(unknown)", transform=ax2.transAxes,
                     ha="center", va="center", fontsize=9, color="gray")
            continue

        phase_a, phase_b = pair
        for ph_name, color in [(phase_a, COLOR_A), (phase_b, COLOR_B)]:
            paths = all_ray_paths.get(ph_name, [])
            labeled = False
            for dist_deg, dep in paths:
                label = ph_name if not labeled else "_nolegend_"
                ax2.plot(dist_deg, dep, color=color, alpha=0.4, lw=1.3, label=label)
                labeled = True

        ax2.invert_yaxis()
        ax2.grid(True, ls="--", alpha=0.25)

        # 標題含觀測值
        obs_row = ev_obs[ev_obs["Phase"] == dph]
        if len(obs_row):
            kv = obs_row["Khan"].values[0]
            sv = obs_row["Samuel"].values[0]
            dv = obs_row["Samuel_minus_Khan_s"].values[0]
            title_str = f"{dph}    Khan={kv:.1f}s  Samuel={sv:.1f}s  (Δ={dv:+.1f}s)"
        else:
            title_str = dph
        ax2.set_title(title_str, fontsize=8.5, pad=3)
        ax2.legend(fontsize=8, loc="lower right")
        ax2.set_ylabel("Depth (km)", fontsize=8)
        if i == len(diff_phase_list) - 1:
            ax2.set_xlabel("Angular distance (°)", fontsize=8)
        else:
            ax2.set_xticklabels([])

    # ══════════════════════════════════════════════
    # 右圖: depth range bar
    #   X 軸 = diff phases
    #   Y 軸 = depth（inverted）
    #   每個 diff phase 畫兩條垂直色塊：
    #     - phase_b 的 max depth 範圍（淺色，IQR）
    #     - phase_a 的 max depth 範圍（深色，IQR）
    #     - 兩者之間（phase_b max ~ phase_a max）= 「額外深度區間」，顏色 = Samuel−Khan
    # ══════════════════════════════════════════════
    BAR_W = 0.35   # 每個色塊的寬度

    # Samuel−Khan colormap（RdBu）
    all_diffs = []
    for p in obs_phases_present:
        row = ev_obs[ev_obs["Phase"] == p]
        if len(row):
            all_diffs.append(float(row["Samuel_minus_Khan_s"].values[0]))
    vmax = max(max(abs(v) for v in all_diffs), 1.0) if all_diffs else 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = cm.RdBu_r   # 紅=Samuel 比 Khan 大，藍=Samuel 比 Khan 小

    if obs_phases_present:
        for xi, dph in enumerate(obs_phases_present):
            phase_a, phase_b = DIFF_PHASES[dph]
            stats_a = phase_path_stats[phase_a]
            stats_b = phase_path_stats[phase_b]

            # Samuel−Khan 差值 → 顏色
            obs_row = ev_obs[ev_obs["Phase"] == dph]
            dv = float(obs_row["Samuel_minus_Khan_s"].values[0]) if len(obs_row) else np.nan
            color_extra = cmap(norm(dv)) if not np.isnan(dv) else "gray"

            # phase_b 的 max depth（IQR 範圍，淺藍色）
            ax3.bar(xi, stats_b["q75"] - stats_b["q25"],
                    bottom=stats_b["q25"], width=BAR_W,
                    color="#B5D4F4", alpha=0.7, zorder=2)
            ax3.hlines(stats_b["median"], xi - BAR_W/2, xi + BAR_W/2,
                       color="#185FA5", lw=1.5, zorder=3)

            # phase_a 的 max depth（IQR 範圍，淺紅色）
            ax3.bar(xi, stats_a["q75"] - stats_a["q25"],
                    bottom=stats_a["q25"], width=BAR_W,
                    color="#F5C4B3", alpha=0.7, zorder=2)
            ax3.hlines(stats_a["median"], xi - BAR_W/2, xi + BAR_W/2,
                       color="#993C1D", lw=1.5, zorder=3)

            # 「額外深度」區間：phase_b median → phase_a median，顏色 = Samuel−Khan
            d_lo = min(stats_b["median"], stats_a["median"])
            d_hi = max(stats_b["median"], stats_a["median"])
            if d_hi > d_lo:
                ax3.bar(xi, d_hi - d_lo, bottom=d_lo, width=BAR_W * 0.55,
                        color=color_extra, alpha=0.85, zorder=4,
                        edgecolor="k", linewidth=0.5)

            # Samuel−Khan 數值標在最深處下方
            if not np.isnan(dv):
                ax3.text(xi, max(stats_a["q75"], stats_b["q75"]) + 20,
                         f"{dv:+.1f}s", ha="center", va="top",
                         fontsize=8, fontweight="bold")

        ax3.set_xticks(range(len(obs_phases_present)))
        ax3.set_xticklabels(obs_phases_present, rotation=35, ha="right", fontsize=9)
        ax3.invert_yaxis()
        ax3.set_ylabel("Deep path length below 1600 km (km)", fontsize=9)

        # colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax3, pad=0.02, shrink=0.6)
        cbar.set_label("Samuel − Khan (s)", fontsize=8)

        # 圖例
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_els = [
            Patch(facecolor="#B5D4F4", edgecolor="none", label="phase_b deep path (IQR)"),
            Patch(facecolor="#F5C4B3", edgecolor="none", label="phase_a deep path (IQR)"),
            Patch(facecolor="gray",    edgecolor="k", lw=0.5, label="extra depth (color=ΔT)"),
        ]
        ax3.legend(handles=legend_els, fontsize=7, loc="lower right", framealpha=0.8)

    else:
        ax3.text(0.5, 0.5, "No matching diff phases", transform=ax3.transAxes,
                 ha="center", va="center", fontsize=11, color="gray")

    ax3.set_title("path comparison (color = Samuel−Khan ΔT)", fontsize=10)
    ax3.grid(True, axis="y", ls="--", alpha=0.3)

    # ── save
    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, f"{event_name}_summary_deep_path.png")
    if SAVE_FIG: plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {outpath}")


# ─────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading best models ...")
    best_models = load_best_models()
    print(f"  Selected {len(best_models)} models:")
    for m in best_models:
        print(f"    {m['model_name']}  misfit={m['misfit']:.4f}")

    for event_name in EVENTS_WITH_OBS:
        plot_event(event_name, best_models)

    print("\nAll done.")