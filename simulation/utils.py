import numpy as np

# constants
NA = 6.022e23               # particles/mol
w0 = 3e-7                   # m
axialFactor = 3
w_z = axialFactor * w0      # m
Lbox = 5 * max(w0, w_z)     # m
resFactor = 10
Lres = resFactor * Lbox     # m
area_yz = (2*Lbox)**2       # m^2
Vres = (2*Lres) * area_yz   # m^3


def power_law(Rp, Nres, alpha, rng):
    xmin_raw = 1e5/Rp
    if alpha > 0:
        xmax_raw = 1.6e7/Rp
    else:
        xmax_raw = 1e7/Rp

    u = rng.random(Nres)
    if np.isclose(alpha, 1.0):
        b = xmin_raw * (xmax_raw / xmin_raw) ** u
    else:
        a = 1.0 - alpha
        b = (u * (xmax_raw**a - xmin_raw**a) + xmin_raw**a) ** (1.0 / a)

    return b

def compute_underlying_dist(Rp, dist_type, sigma_b, alpha, rng, conc=None, Nres=None):
    if conc:
        Nres = max(1, rng.poisson(conc*1e3 * NA * Vres))
    if dist_type == 'pl':
        # xmin_raw, xmax_raw = 1e5/(Rp), 1.6e7/(Rp)
        # c = xmax_raw/xmin_raw
        # b = truncpareto.rvs(alpha, c=c, loc=0, scale = xmin_raw, size=Nres, random_state=rng)
        b = power_law(Rp, Nres, alpha, rng)
    else:
        b = np.exp(sigma_b * rng.normal(size=Nres) - 0.5 * sigma_b**2)
    if conc:
        return b, Nres
    return b
