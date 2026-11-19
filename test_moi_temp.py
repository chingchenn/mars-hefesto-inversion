import json, glob, numpy as np
from mars_config import MCMC_DIR
T = []
for fn in glob.glob(f'{MCMC_DIR}/chain_a36*/chain.jsonl'):
    for line in open(fn):
        line = line.strip()
        if not line: continue
        T.append(json.loads(line)['params']['T_core'])
T = np.array(T)
print(f"min={T.min():.0f} max={T.max():.0f} median={np.median(T):.0f}")
print(f"within 50K of max: {100*np.mean(T > T.max()-50):.1f}%")
print(f"within 50K of min: {100*np.mean(T < T.min()+50):.1f}%")