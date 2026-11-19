"""
讀取 MCMC chain 資料夾中每個 step 的 fort.56 和 ad.in
路徑結構: <base_dir>/chain_a07_XX/step_000XX/s3_final/{fort.56, ad.in}

輸出:
  - fort56_all  : dict, key=(chain, step), value=pd.DataFrame
  - adin_all    : dict, key=(chain, step), value=pd.DataFrame
  - fort56_df   : 所有資料合併成一個大 DataFrame（含 chain, step 欄位）
  - adin_df     : 同上
"""

import os
import re
import pandas as pd
from mars_config import MCMC_DIR, FIG_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

BASE_DIR = MCMC_DIR
ADIN_COLUMNS = ["pressure", "col2", "temperature"]


def read_fort56(filepath: str) -> pd.DataFrame:
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_line = lines[1].strip()
    columns = header_line.split()

    records = []
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 0:
            continue
        n_numeric = len(columns) - 1
        try:
            row = [float(x) for x in parts[:n_numeric]]
            row.append(parts[n_numeric] if len(parts) > n_numeric else "")
            records.append(row)
        except ValueError:
            continue  

    df = pd.DataFrame(records, columns=columns)
    return df


def read_adin(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=ADIN_COLUMNS)
    return df


def load_all(base_dir: str):
    fort56_all = {}
    adin_all   = {}
    failed     = []  # 新增

    chain_pattern = re.compile(r"^chain_a07_\d+$")
    step_pattern  = re.compile(r"^step_\d+$")

    chains = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and chain_pattern.match(d)
    ])

    for chain in chains:
        chain_dir = os.path.join(base_dir, chain)
        steps = sorted([
            d for d in os.listdir(chain_dir)
            if os.path.isdir(os.path.join(chain_dir, d)) and step_pattern.match(d)
        ])

        for step in steps:
            s3_dir = os.path.join(chain_dir, step, "s3_final")
            if not os.path.isdir(s3_dir):
                continue

            key = (chain, step)

            fort56_path = os.path.join(s3_dir, "fort.56")
            if os.path.isfile(fort56_path):
                try:
                    df = read_fort56(fort56_path)
                    if not df.empty:
                        fort56_all[key] = df
                    else:
                        failed.append((key, "fort.56", "empty file"))
                except Exception as e:
                    failed.append((key, "fort.56", str(e)))

            adin_path = os.path.join(s3_dir, "ad.in")
            if os.path.isfile(adin_path):
                try:
                    adin_all[key] = read_adin(adin_path)
                except Exception as e:
                    failed.append((key, "ad.in", str(e)))

    return fort56_all, adin_all, failed  




fort56_all, adin_all, failed = load_all(BASE_DIR)

print(f"成功 fort.56: {len(fort56_all)}, ad.in: {len(adin_all)}")
print(f"失敗: {len(failed)} 筆")
for (chain, step), filetype, reason in failed:
    print(f"  {chain}/{step}/{filetype}: {reason}")

fort56_failed = [(k, f, r) for k, f, r in failed if f == "fort.56"]
adin_failed   = [(k, f, r) for k, f, r in failed if f == "ad.in"]

total_fort56 = len(fort56_all) + len(fort56_failed)
total_adin   = len(adin_all)   + len(adin_failed)

print(f"fort.56 成功: {len(fort56_all)}/{total_fort56}  失敗率: {len(fort56_failed)/total_fort56*100:.1f}%")
print(f"ad.in   成功: {len(adin_all)}/{total_adin}    失敗率: {len(adin_failed)/total_adin*100:.1f}%")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 10))

for (chain, step), df in fort56_all.items():
    ax.plot(df["VS(km/s)"], df["P(GPa)"], color="steelblue", alpha=0.1, linewidth=0.5)
    

ax.set_xlabel("Vs (km/s)")
ax.set_ylabel("Pressure (GPa)")
ax.set_title("All chains & steps: Vs vs Pressure")
ax.invert_yaxis()  
plt.tight_layout()
if SAVE_FIG: plt.savefig(os.path.join(FIG_DIR, "test_vs_pressure_all.png"), dpi=150)