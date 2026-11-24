#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 97: Wieczorek et al. 2022 (JGR Planets) uniform-density crust models, digitized from the
#      vector paths of their Fig. 3 (left panel) and Fig. 4 (left and center panels).
#
#   python 97.wieczorek2022_crust.py           (SAVE_FIG=1 to save the check figure)
#
# Fig. 4 plots mean (left) and minimum (center) crustal thickness against the density contrast
# across the Moho, drho = rho_m - rho_c, for InSight anchors 20, 25, ..., 45 km. The ten mantle
# models collapse onto one curve per anchor (their Sec. 3), so one fit per anchor serves any rho_m:
#     h_avg(drho; a)  = c0 + c1*x + c2*x^2,   x = 1000/drho     (Moho relief ~ 1/drho)
#     h_min(drho; a)  = same form;  h_min = 0 at drho0(a)  ->  rho_c,max = rho_m - drho0(a)
# Anchors in between: linear interpolation in a; above 45 km: linear extrapolation from 40-45.
# Fig. 3 (anchor 39 km, rho_c on a 50 kg/m3 grid, each model at its own rho_m from Table 2) is used
# only as an independent check of the Fig. 4 fits.
#
# First run extracts the curves from the PDF (needs PyMuPDF: pip install pymupdf) into CSV_OUT.
# Later runs only read the CSV. Extraction: every curve in these figures is a filled outline
# polygon; the centerline is the mean of the upper and lower outline crossings at each x.
# Axis calibration from the tick marks (page coordinates, pt), found by hand on 2026-11-24.
import os
import numpy as np
import matplotlib.pyplot as plt

PDF     = os.path.expanduser('~/Desktop/paper/Wieczorek 2022 JGR Planets .pdf')
CSV_OUT = 'wieczorek2022_fig3_fig4.csv'
ANCH    = [20, 25, 30, 35, 40, 45]
# Table 2: uppermost-mantle density rho_m of each interior model
RHO_M = {'DWThotCrust1r': 3493, 'EH45ThotCrust2r': 3462, 'EH45Tcold': 3459, 'SANAK': 3450,
         'LFAK': 3400, 'ZG_DW': 3389, 'Khan2022': 3382, 'DWAK': 3380, 'TAYAK': 3376,
         'YOTHotRc1810': 3352}
# fill colour (rounded RGB) -> model, from the figure legend
COLOR = {(0.12, 0.47, 0.71): 'DWThotCrust1r', (1.0, 0.5, 0.05): 'EH45Tcold',
         (0.17, 0.63, 0.17): 'EH45ThotCrust2r', (0.84, 0.15, 0.16): 'LFAK',
         (0.58, 0.4, 0.74): 'SANAK', (0.55, 0.34, 0.29): 'TAYAK', (0.89, 0.47, 0.76): 'DWAK',
         (0.5, 0.5, 0.5): 'ZG_DW', (0.74, 0.74, 0.13): 'YOTHotRc1810', (0.0, 0.0, 0.0): 'Khan2022'}
# panel: (page index, frame x0 y0 x1 y1, x tick pt/value pair, y tick pt/value pair, x step)
PANELS = {'fig3avg': (8,  (80.4, 92.3, 205.7, 245.5),  (92.205, 2600, 115.855, 2700), (95.345, 60, 230.07, 30), 50),
          'fig4avg': (10, (80.4, 92.3, 205.7, 245.5),  (93.595, 400, 113.205, 500),   (93.59, 70, 232.01, 30), 10),
          'fig4min': (10, (249.2, 92.3, 374.5, 245.5), (262.365, 400, 281.975, 500),  (245.495, 0, 103.685, 25), 10)}
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'
FS = 14
plt.rcParams['font.family'] = 'DejaVu Sans'

# ---------------------------------------------------------------- extraction (first run only)
if not os.path.exists(CSV_OUT):
    import pymupdf
    doc = pymupdf.open(PDF)
    rows = []                                   # (panel, anchor, model, x, y)
    for pan, (ipg, box, xc, yc, dx) in PANELS.items():
        segs = []
        for d in doc[ipg].get_drawings():
            r = d['rect']
            if d['type'] != 'f' or not d['fill'] or len(d['items']) < 8:
                continue
            if not (box[0] - 2 < r.x0 < box[2] and box[1] - 2 < r.y0 < box[3] and r.x1 < box[2] + 2):
                continue
            key = tuple(round(v, 2) for v in d['fill'])
            if key not in COLOR or r.width > box[2] - box[0] - 3:       # skip frame
                continue
            P = []
            for it in d['items']:
                if it[0] == 'l':
                    P += [it[1], it[2]]
                elif it[0] == 'c':
                    P += [it[1], it[4]]
            P = np.array([(p.x, p.y) for p in P])
            # page pt <-> data: x = v0 + (pt - p0)*(v1 - v0)/(p1 - p0)
            v_lo = xc[1] + (r.x0 + 0.8 - xc[0])*(xc[3] - xc[1])/(xc[2] - xc[0])
            v_hi = xc[1] + (r.x1 - 0.8 - xc[0])*(xc[3] - xc[1])/(xc[2] - xc[0])
            grid = np.arange(np.ceil(v_lo/dx)*dx, v_hi + 1e-6, dx)
            ys = []
            for v in grid:
                xp = xc[0] + (v - xc[1])*(xc[2] - xc[0])/(xc[3] - xc[1])
                cut = []
                for i in range(len(P)):
                    (xa, ya), (xb, yb) = P[i], P[(i + 1) % len(P)]
                    if (xa - xp)*(xb - xp) <= 0 and xa != xb:
                        cut.append(ya + (xp - xa)*(yb - ya)/(xb - xa))
                yp = 0.5*(min(cut) + max(cut)) if len(cut) >= 2 else np.nan
                ys.append(yc[1] + (yp - yc[0])*(yc[3] - yc[1])/(yc[2] - yc[0]))
            ys = np.array(ys)
            ok = np.isfinite(ys)
            if ok.sum() < 3 or np.nanstd(ys) < 0.2:                     # outlined text glyphs
                continue
            # text glyphs ("40 km" labels) are not smooth: drop segments a cubic cannot follow
            if np.sqrt(np.mean((np.polyval(np.polyfit(grid[ok], ys[ok], 3), grid[ok]) - ys[ok])**2)) > 0.15:
                continue
            segs.append((COLOR[key], grid, ys))
        if pan == 'fig3avg':                     # keep only the 39 km (solid) group: h > 40
            for m, x, y in segs:
                if np.nanmean(y) > 40:
                    rows += [(pan, 39, m, a, b) for a, b in zip(x, y) if np.isfinite(b)]
            continue
        # Fig. 4: six tight groups, one per anchor; cluster by the quadratic extrapolation to drho 830
        v830 = np.array([np.polyval(np.polyfit(x[np.isfinite(y)], y[np.isfinite(y)], 2), 830) for m, x, y in segs])
        order = np.argsort(v830)
        cuts = sorted(np.argsort(np.diff(v830[order]))[-5:])
        g, lab = 0, np.zeros(len(segs), int)
        for i, k in enumerate(order):
            lab[k] = g
            if i in cuts:
                g += 1
        for (m, x, y), l in zip(segs, lab):
            rows += [(pan, ANCH[l], m, a, b) for a, b in zip(x, y) if np.isfinite(b)]
    with open(CSV_OUT, 'w') as f:
        f.write('# Wieczorek et al. 2022 JGR Planets, Fig. 3 left (fig3avg: x = rho_c, anchor 39 km) and\n')
        f.write('# Fig. 4 left/center (fig4avg, fig4min: x = rho_m - rho_c). y = thickness in km.\n')
        f.write('# Extracted from PDF vector paths by 97.wieczorek2022_crust.py\n')
        f.write('panel,anchor_km,model,x,y_km\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]},{r[2]},{r[3]:.1f},{r[4]:.3f}\n')
    print(f'wrote {CSV_OUT}: {len(rows)} points')

# ---------------------------------------------------------------- fits
D = np.genfromtxt(CSV_OUT, delimiter=',', dtype=None, encoding=None, skip_header=3, names=True)
C_AVG, C_MIN, DRHO0 = {}, {}, {}
print('\nanchor  h_avg fit: c0, c1, c2 (x = 1000/drho)   rms   model spread   drho0 (h_min = 0)')
for a in ANCH:
    for pan, store in [('fig4avg', C_AVG), ('fig4min', C_MIN)]:
        s = (D['panel'] == pan) & (D['anchor_km'] == a)
        x = 1e3/D['x'][s]
        store[a] = np.linalg.lstsq(np.c_[np.ones_like(x), x, x**2], D['y_km'][s], rcond=None)[0]
    s = (D['panel'] == 'fig4avg') & (D['anchor_km'] == a)
    rms = np.sqrt(np.mean((np.polyval(C_AVG[a][::-1], 1e3/D['x'][s]) - D['y_km'][s])**2))
    spread = max(np.ptp(D['y_km'][s][D['x'][s] == v]) for v in np.unique(D['x'][s]) if np.sum(D['x'][s] == v) > 3)
    g = np.linspace(250.0, 1000.0, 7501)
    DRHO0[a] = g[np.argmin(np.abs(np.polyval(C_MIN[a][::-1], 1e3/g)))]
    print(f'  {a:2d}    {C_AVG[a][0]:8.3f} {C_AVG[a][1]:8.3f} {C_AVG[a][2]:8.3f}      {rms:.2f}    {spread:.2f} km      {DRHO0[a]:.0f}')

# h_avg(rho_c; rho_m, anchor): evaluate each anchor's fit at drho, interpolate (extrapolate) in anchor
# drho0(anchor) likewise; rho_c,max = rho_m - drho0
def h_avg(rho_c, rho_m, anchor):
    v = np.array([np.polyval(C_AVG[a][::-1], 1e3/(rho_m - np.asarray(rho_c, float))) for a in ANCH])
    w = np.interp(anchor, ANCH, np.arange(len(ANCH)))        # fractional index, clipped at 20 / 45
    if anchor > ANCH[-1]:
        w = len(ANCH) - 1 + (anchor - ANCH[-1])/5.0
    i = min(int(np.floor(w)), len(ANCH) - 2)
    return v[i] + (w - i)*(v[i + 1] - v[i])
def rho_c_max(rho_m, anchor):
    d = [DRHO0[a] for a in ANCH]
    d0 = np.interp(anchor, ANCH, d) if anchor <= ANCH[-1] else d[-1] + (anchor - ANCH[-1])*(d[-1] - d[-2])/5.0
    return rho_m - d0

print('\nCheck against Fig. 3 (anchor 39 km): h_fig3 - h_fit, km')
res3 = []
for m in RHO_M:
    s = (D['panel'] == 'fig3avg') & (D['model'] == m)
    if not s.any():
        continue
    dif = D['y_km'][s] - h_avg(D['x'][s], RHO_M[m], 39)
    res3 += list(dif)
    print(f'  {m:<16} rho_m {RHO_M[m]}  rho_c {D["x"][s].min():.0f}-{D["x"][s].max():.0f}   '
          f'mean {np.mean(dif):+.2f}  max |d| {np.max(np.abs(dif)):.2f}   last plotted rho_c {D["x"][s].max():.0f}'
          f'  vs rho_c,max fit {rho_c_max(RHO_M[m], 39):.0f}')
print(f'  all: rms {np.sqrt(np.mean(np.square(res3))):.2f} km')

print('\nrho_c,max and h_avg range for candidate uppermost-mantle densities')
for rm, nm in [(3382, 'Khan2022 (Table 2)'), (3446, 'Samuel 2023 column 66.7-100 km'), (3459, 'EH45Tcold')]:
    for a in [31, 39, 47]:
        rmax = rho_c_max(rm, a)
        print(f'  rho_m {rm} ({nm:<30}) anchor {a}: rho_c,max {rmax:.0f}   '
              f'h_avg {h_avg(2550, rm, a):.1f} km at 2550 -> {h_avg(rmax, rm, a):.1f} km at rho_c,max')

# ---------------------------------------------------------------- figure
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
cols = plt.cm.viridis(np.linspace(0, 0.9, len(ANCH)))
for k, a in enumerate(ANCH):
    for pan, ax, cc in [('fig4avg', axs[0], C_AVG), ('fig4min', axs[1], C_MIN)]:
        s = (D['panel'] == pan) & (D['anchor_km'] == a)
        ax.plot(D['x'][s], D['y_km'][s], '.', color=cols[k], ms=3)
        g = np.linspace(D['x'][s].min() - 20, 950, 200)
        ax.plot(g, np.polyval(cc[a][::-1], 1e3/g), '-', color=cols[k], lw=1.5, label=f'{a} km')
    axs[1].axvline(DRHO0[a], color=cols[k], ls=':', lw=1)
for m in RHO_M:
    s = (D['panel'] == 'fig3avg') & (D['model'] == m)
    l, = axs[2].plot(D['x'][s], D['y_km'][s], 'o', ms=4)
    g = np.linspace(2550, rho_c_max(RHO_M[m], 39), 100)
    axs[2].plot(g, h_avg(g, RHO_M[m], 39), '-', color=l.get_color(), lw=1, label=f'{m} ({RHO_M[m]})')
g = np.linspace(2550, 3050, 10)
axs[2].plot(g, 48 + 11*(g - 2550)/500, 'k--', lw=2, label='40: linear 48-59 km')
axs[0].set_xlabel(r'$\rho_m - \rho_c$ [kg/m$^3$]', fontsize=FS)
axs[1].set_xlabel(r'$\rho_m - \rho_c$ [kg/m$^3$]', fontsize=FS)
axs[2].set_xlabel(r'crust density $\rho_c$ [kg/m$^3$]', fontsize=FS)
axs[0].set_ylabel('mean crustal thickness [km]', fontsize=FS)
axs[1].set_ylabel('minimum crustal thickness [km]', fontsize=FS)
axs[2].set_ylabel('mean crustal thickness [km], anchor 39 km', fontsize=FS)
axs[0].set_title('Fig. 4 left: points and fit per anchor', fontsize=FS)
axs[1].set_title(r'Fig. 4 center: $h_{min}=0$ gives $\rho_{c,max}$', fontsize=FS)
axs[2].set_title('Fig. 3 points vs Fig. 4 fit', fontsize=FS)
axs[1].set_ylim(0, 27)
for ax in axs:
    ax.tick_params(labelsize=14)
axs[0].legend(loc='lower right', fontsize=FS - 4, title='InSight anchor')
axs[2].legend(loc='lower right', fontsize=FS - 6)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig('97_wieczorek2022_crust.png', dpi=200)
plt.show()
