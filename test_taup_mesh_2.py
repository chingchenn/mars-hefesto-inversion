# 95.branch_diag.py
import os, sys, importlib.util
import numpy as np



CHAIN_DIR = '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc'

import json
recs = [json.loads(l) for l in open(f"{CHAIN_DIR}/chain_a47B_13/chain.jsonl")]
stuck = recs[1465] # step 1466, 0-indexed
print("step:", stuck['step'])
print("accepted:", stuck['accepted'])
print("misfit:", stuck['misfit'])
print("params:", stuck['params'])

json.dump(stuck['params'], open("/tmp/chain13_stuck.json", "w"))
print("\nsaved to /tmp/chain13_stuck.json")
