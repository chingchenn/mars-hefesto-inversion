# 24.melting_solidus.py
# Duncan 2018 mantle solidus (deg C) vs pressure.
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
P = np.linspace(0, 26, 50)        # GPa

# Duncan 2018 solidus (deg C)
# P <= 10 : -4.877 P^2 + 120.2 P + 1088
# P <= 23 : -1.323 (P-10)^2 + 38.18 (P-10) + 1802
# P >  23 : 77.75 (P-23) + 2075
T_solidus = np.where(P <= 10, -4.877 * P**2 + 120.2 * P + 1088,
            np.where(P <= 23, -1.323 * (P - 10)**2 + 38.18 * (P - 10) + 1802,
                              77.75 * (P - 23) + 2075))

fig, ax = plt.subplots(figsize=(5, 7))
ax.scatter(T_solidus, P, color='#414F67')
ax.set_xlabel('Solidus temperature (deg C)', fontsize=14)
ax.set_ylabel('Pressure (GPa)', fontsize=14)
ax.set_ylim(26, 0)
ax.tick_params(labelsize=14)
plt.tight_layout()
plt.show()
