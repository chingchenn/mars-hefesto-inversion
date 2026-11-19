import json, glob, numpy as np
T = []
for fn in glob.glob('/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc/chain_a36*/chain.jsonl'):
    for line in open(fn):
        line = line.strip()
        if not line: continue
        T.append(json.loads(line)['params']['T_core'])
T = np.array(T)
print(f"min={T.min():.0f} max={T.max():.0f} median={np.median(T):.0f}")
print(f"within 50K of max: {100*np.mean(T > T.max()-50):.1f}%")
print(f"within 50K of min: {100*np.mean(T < T.min()+50):.1f}%")