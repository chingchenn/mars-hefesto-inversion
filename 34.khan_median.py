import numpy as np
import glob
import os

KHAN_MODEL_DIR = "/net/beno3/data1/jcchen2/Mars_Khan_2023/LSL_Models"   
OUTPUT_PATH    = "/net/beno3/data1/jcchen2/Mars_Khan_2023/LSL_Models/khan_median.npz"
MARS_RADIUS    = 3389.5

files = sorted(glob.glob(os.path.join(KHAN_MODEL_DIR, 'Model_*.txt')))

crust_z   = np.linspace(0, 100, 200)
core_z    = np.linspace(1500, MARS_RADIUS, 200)
gravity_z = np.linspace(0, MARS_RADIUS, 400)

crust_vp_all  = []; crust_vs_all = []; crust_rho_all = []
core_vp_all   = []; core_rho_all = []
grav_all      = []
lsl_top_depths  = []
true_cmb_depths = []

for fpath in files:
    try:
        data  = np.loadtxt(fpath, comments='#')
        depth = data[:, 0]
        Vp    = data[:, 1]
        Vs    = data[:, 2]
        rho   = data[:, 3]
        g_col = data[:, 5]

        solid_mask  = Vs > 0.01
        liquid_mask = Vs < 0.01
        if solid_mask.sum() < 5 or liquid_mask.sum() < 5:
            continue

        # LSL top & true CMB
        lsl_top = depth[liquid_mask][0]
        lsl_top_depths.append(lsl_top)

        liq_depth = depth[liquid_mask]
        liq_rho   = rho[liquid_mask]
        drho      = np.diff(liq_rho)
        ddepth    = np.diff(liq_depth)
        disc_mask = (np.abs(ddepth) < 1.0) & (drho > 1.0)
        if disc_mask.any():
            true_cmb = liq_depth[np.argmax(drho * disc_mask) + 1]
        else:
            true_cmb = liq_depth[np.argmax(drho) + 1]
        true_cmb_depths.append(true_cmb)

        # crust velocity
        sd = depth[solid_mask]
        if sd.max() > crust_z.max():
            crust_vp_all.append(np.interp(crust_z, sd, Vp[solid_mask]))
            crust_vs_all.append(np.interp(crust_z, sd, Vs[solid_mask]))
            crust_rho_all.append(np.interp(crust_z, sd, rho[solid_mask]))

        # core velocity
        core_liq_mask = liquid_mask & (depth >= true_cmb)
        cd = depth[core_liq_mask]
        if len(cd) > 0 and cd.max() >= core_z.max() * 0.9:
            core_vp_all.append(np.interp(core_z, cd, Vp[core_liq_mask]))
            core_rho_all.append(np.interp(core_z, cd, rho[core_liq_mask]))

        # gravity
        _, uniq = np.unique(depth, return_index=True)
        if depth[uniq].max() >= MARS_RADIUS * 0.95:
            grav_all.append(np.interp(gravity_z, depth[uniq], g_col[uniq]))

    except Exception as e:
        print(f"  {os.path.basename(fpath)}: {e}")
        continue

lsl_top_median  = float(np.median(lsl_top_depths))
true_cmb_median = float(np.median(true_cmb_depths))

print(f"LSL top  : {lsl_top_median:.1f} km")
print(f"True CMB : {true_cmb_median:.1f} km")
print(f"LSL thickness:                 {true_cmb_median - lsl_top_median:.1f} km")

np.savez(
    OUTPUT_PATH,
    crust_z        = crust_z,
    crust_vp       = np.nanmedian(crust_vp_all,  axis=0),
    crust_vs       = np.nanmedian(crust_vs_all,  axis=0),
    crust_rho      = np.nanmedian(crust_rho_all, axis=0),
    core_z         = core_z,
    core_vp        = np.nanmedian(core_vp_all,   axis=0),
    core_vs        = np.zeros(len(core_z)),
    core_rho       = np.nanmedian(core_rho_all,  axis=0),
    gravity_z      = gravity_z,
    gravity_g      = np.nanmedian(grav_all,       axis=0),
    lsl_top_depth  = np.array([lsl_top_median]),
    true_cmb_depth = np.array([true_cmb_median]),
)
print(f" {OUTPUT_PATH}")