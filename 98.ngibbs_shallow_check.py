#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 98: nGibbs (old July 9 model and new Nov model) vs HeFESTo at shallow, cold conditions,
#     to decide whether Moho to 100 km can be computed with nGibbs instead of the Samuel column.
#
#   beno3, in the repo root (~/mars-hefesto-inversion):
#     git clone https://github.com/benjthyer/nGibbs ~/nGibbs_new      (once; the old ~/nGibbs is not touched)
#     python 98.ngibbs_shallow_check.py
#
# What it does:
#   1. composition: a posterior sample of chain CHAIN (YM20), via the main program's
#      composition_from_params / compute_oxygen (same as the forward model)
#   2. P-T grid: P_LIST x T_LIST, covering Moho (~0.5-0.7 GPa) to 100 km (~1.2 GPa) and colder
#      than the current lithosphere at 100 km (~650 K)
#   3. HeFESTo single-point NPT at every grid node (parallel, N_JOBS at a time)
#   4. old nGibbs: the main program's _ng_forward / _ng_props (July 9, 'Vp' isomorphic)
#      new nGibbs: ~/nGibbs_new, 'Vp' and 'Vp_fast' (fort.56 VP includes order-disorder relaxation)
#   5. T_freeze test (new model): assemblage at max(T, T_FREEZE), properties at T (Stixrude's
#      suggestion to Ben, June 2026); reported as frozen - equilibrium
# Output: table to terminal, 98_ngibbs_shallow_check.csv next to this script.
import os, sys, json, glob, re, shutil, subprocess
import importlib.util
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mars_config import MCMC_DIR, RUN_BASE, PAR_DIR, HEFESTO_MAIN

MAIN_SCRIPT = os.path.join(HERE, '26fall_main_mcmc_a72.py')     # old nGibbs (NGIBBS_ROOT inside)
SCRIPT_50   = os.path.join(HERE, '50.mcmc_main.py')             # only read for CONTROL_PHASES
NEW_NGIBBS  = os.path.expanduser('~/nGibbs_new')
CHAIN       = 'chain_a74_YM20_00'
WORK_DIR    = os.path.join(RUN_BASE, 'hef_shallow_98')
P_LIST      = [0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0]          # GPa
T_LIST      = [300.0, 500.0, 700.0, 900.0, 1100.0]          # K
T_FREEZE    = 900.0                                        # K
N_JOBS      = 12
HEF_TIMEOUT = 900                                          # s per point
CSV_OUT     = os.path.join(HERE, '98_ngibbs_shallow_check.csv')

# ---------------------------------------------------------------- old nGibbs via the main program
spec = importlib.util.spec_from_file_location('main_a72', MAIN_SCRIPT)
NG = importlib.util.module_from_spec(spec)
sys.modules['main_a72'] = NG
spec.loader.exec_module(NG)
print('old nGibbs from', NG.NGIBBS_ROOT)

# ---------------------------------------------------------------- new nGibbs: drop the old package, import the new one
for k in [k for k in sys.modules if k == 'ngibbs' or k.startswith('ngibbs.')]:
    del sys.modules[k]
sys.path = [p for p in sys.path if not p.startswith(str(NG.NGIBBS_ROOT))]
sys.path.insert(0, os.path.join(NEW_NGIBBS, 'src'))
from ngibbs import HeFESToMarsEmulatorCPU as EMN          # new layout (Sep 2026): models at package level
print('new nGibbs from', NEW_NGIBBS, ' training P bounds:', EMN.training_pressure_bounds())

# ---------------------------------------------------------------- composition from a posterior sample
d = os.path.join(MCMC_DIR, CHAIN)
npz = sorted(glob.glob(os.path.join(d, 'profile_s*.npz')))
step = int(os.path.basename(npz[len(npz)//2])[9:14])
params = None
for line in open(os.path.join(d, 'chain.jsonl')):
    line = line.strip()
    if line:
        r = json.loads(line)
        if r.get('step') == step:
            params = r['params']
p  = NG.composition_from_params(params)
O  = NG.compute_oxygen(p)
cv = NG._comp_values(p, O)                                  # Si Mg Fe Ca Al Na Cr O (moles)
print(f'composition: {CHAIN} step {step}  Mg# {params.get("Mg#")}')
print('  ' + '  '.join(f'{k} {v:.4f}' for k, v in zip(['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O'], cv)))

PP, TT = [a.ravel() for a in np.meshgrid(P_LIST, T_LIST, indexing='ij')]
n = len(PP)

# ---------------------------------------------------------------- old nGibbs
o = NG._ng_forward(PP, TT, cv, NG.NG_T_HEADERS, ['component_moles'])
pr = NG._ng_props(o['component_moles'], PP, TT, ['rho', 'Vp', 'Vs'])
old = {k: np.asarray(pr[k].detach().cpu() if torch.is_tensor(pr[k]) else pr[k], float) for k in ['rho', 'Vp', 'Vs']}

# ---------------------------------------------------------------- new nGibbs, equilibrium and frozen assemblage
def new_props(P, T_assemblage, T_eval):
    x = np.zeros((len(P), 2 + len(cv)), dtype=np.float32)
    x[:, 0], x[:, 1], x[:, 2:] = P, T_assemblage, cv
    with torch.no_grad():
        out = EMN.ForwardMB(x, headers=NG.NG_T_HEADERS, outputs=['component_moles'])
        PT = torch.tensor(np.column_stack([P, T_eval]), dtype=torch.float64)
        pr = EMN.get_property_hefesto_vectorized_from_assemblage(
            out['component_moles'].double(), PT, property_names=['rho', 'Vp', 'Vp_fast', 'Vs'])
    return {k: np.asarray(pr[k].detach().cpu() if torch.is_tensor(pr[k]) else pr[k], float)
            for k in ['rho', 'Vp', 'Vp_fast', 'Vs']}
new = new_props(PP, TT, TT)
frz = new_props(PP, np.maximum(TT, T_FREEZE), TT)

# ---------------------------------------------------------------- HeFESTo single points
src = open(SCRIPT_50).read()
CONTROL_PHASES = re.search(r'CONTROL_PHASES\s*=\s*"""\\?\n(.*?)"""', src, re.S).group(1)
el = dict(zip(['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr'], cv[:7]))
def run_hef(i):
    rd = os.path.join(WORK_DIR, f'pt_{i:02d}')
    os.makedirs(rd, exist_ok=True)
    lines = [f'{PP[i]:.4f},{PP[i]:.4f},1,{TT[i]:.2f},{TT[i]:.2f},0,0,0,0', '8,2,4,2', 'oxides']
    lines += [f'{k:>2}      {v:.5f}     {v:.5f}    0' for k, v in el.items()]
    lines += [f' O      {cv[7]:.5f}     {cv[7]:.5f}    0', '1,1,1,1', PAR_DIR, '73', CONTROL_PHASES]
    open(os.path.join(rd, 'control'), 'w').write('\n'.join(lines))
    if not os.path.exists(os.path.join(rd, 'main')):
        shutil.copy2(HEFESTO_MAIN, os.path.join(rd, 'main'))
        os.chmod(os.path.join(rd, 'main'), 0o755)
    f56 = os.path.join(rd, 'fort.56')
    if os.path.exists(f56):
        os.remove(f56)
    try:
        with open(os.path.join(rd, 'hefesto.log'), 'w') as log:
            subprocess.run(['./main'], cwd=rd, stdout=log, stderr=log, timeout=HEF_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    if not os.path.exists(f56) or os.path.getsize(f56) == 0:
        return None
    with open(f56) as f:
        f.readline()
        cols = f.readline().split()
    row = pd.read_csv(f56, sep=r'\s+', skiprows=2, names=cols).iloc[0]
    return {'rho': float(row['rho(g/cm^3)']), 'Vp': float(row['VP(km/s)']), 'Vs': float(row['VS(km/s)'])}

print(f'\nHeFESTo: {n} points, {N_JOBS} at a time, work dir {WORK_DIR}')
with ThreadPoolExecutor(N_JOBS) as ex:
    hef = list(ex.map(run_hef, range(n)))
H = {k: np.array([h[k] if h else np.nan for h in hef]) for k in ['rho', 'Vp', 'Vs']}
print(f'HeFESTo ok: {np.sum(np.isfinite(H["rho"]))}/{n}')

# ---------------------------------------------------------------- table
rel = lambda a, b: 100*(a - b)/b
df = pd.DataFrame({'P_GPa': PP, 'T_K': TT,
                   'rho_hef': H['rho'], 'Vp_hef': H['Vp'], 'Vs_hef': H['Vs'],
                   'rho_old': old['rho'], 'Vp_old': old['Vp'], 'Vs_old': old['Vs'],
                   'rho_new': new['rho'], 'Vp_new': new['Vp'], 'Vpfast_new': new['Vp_fast'], 'Vs_new': new['Vs'],
                   'rho_frz': frz['rho'], 'Vpfast_frz': frz['Vp_fast'], 'Vs_frz': frz['Vs']})
df.to_csv(CSV_OUT, index=False)

print('\nrelative difference, %  (old/new vs HeFESTo;  frz-eq = frozen at T_FREEZE minus equilibrium, new model)')
print(f"{'P':>5}{'T':>6} | {'rho old':>8}{'rho new':>8}{'Vp old':>8}{'Vpf new':>8}{'Vs old':>8}{'Vs new':>8}"
      f" | {'rho f-e':>8}{'Vpf f-e':>8}{'Vs f-e':>8}")
for i in range(n):
    print(f'{PP[i]:5.1f}{TT[i]:6.0f} | {rel(old["rho"][i], H["rho"][i]):+8.3f}{rel(new["rho"][i], H["rho"][i]):+8.3f}'
          f'{rel(old["Vp"][i], H["Vp"][i]):+8.3f}{rel(new["Vp_fast"][i], H["Vp"][i]):+8.3f}'
          f'{rel(old["Vs"][i], H["Vs"][i]):+8.3f}{rel(new["Vs"][i], H["Vs"][i]):+8.3f}'
          f' | {rel(frz["rho"][i], new["rho"][i]):+8.3f}{rel(frz["Vp_fast"][i], new["Vp_fast"][i]):+8.3f}'
          f'{rel(frz["Vs"][i], new["Vs"][i]):+8.3f}')

print('\nsummary over the Moho-100 km box (P <= 1.2 GPa), mean / max |rel diff| %')
m = (PP <= 1.2) & np.isfinite(H['rho'])
for lab, a, b in [('rho old', old['rho'], H['rho']), ('rho new', new['rho'], H['rho']),
                  ('Vp old (iso)', old['Vp'], H['Vp']), ('Vp new (iso)', new['Vp'], H['Vp']),
                  ('Vp new (fast)', new['Vp_fast'], H['Vp']),
                  ('Vs old', old['Vs'], H['Vs']), ('Vs new', new['Vs'], H['Vs']),
                  ('rho frozen-eq', frz['rho'], new['rho']), ('Vs frozen-eq', frz['Vs'], new['Vs'])]:
    r = np.abs(rel(a[m], b[m]))
    print(f'  {lab:<15} {np.mean(r):6.3f} / {np.max(r):6.3f}')
# shell 55-100 km: 4 pi r^2 dr rho = 12.57 * (3.312e6 m)^2 * 4.5e4 m * 3400 kg/m3 = 2.1e22 kg = 3.3 % of M
print(f'\n  the Moho-100 km shell is ~3.3 % of the mass of Mars: a mean density error e % there is ~0.033*e % of M')
print(f'  target mass sigma ~0.045 %  ->  tolerable mean density error in this shell ~{0.045/0.033:.1f} %')
print(f'\nwrote {CSV_OUT}')
