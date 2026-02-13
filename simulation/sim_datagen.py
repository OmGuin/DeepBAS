#imports
import time
import numpy as np
import os
import json
from numba import njit
import multiprocessing as mp
import gc
from ..CNNp1a.utils import compute_underlying_dist

#directory
data_dir = os.path.join("data_gen", "pchdata")
os.makedirs(data_dir, exist_ok=True)
metadata = {
    "pch_edges":"np.logspace(np.log10(1), np.log10(8000), 25)"
}
with open(os.path.join(data_dir, f"metadata.json"), "w") as f:
    json.dump(metadata, f)

sims_per_env = 10

# 1a) Geometry & Constraints
NA = 6.022e23               # particles/mol
w0 = 3e-7                   # m
axialFactor=3
w_z = axialFactor * w0      # m
Lbox = 5 * max(w0, w_z)     # m
resFactor = 10
Lres = resFactor * Lbox     # m

# 2a) Reservoir initialization
area_yz = (2*Lbox)**2       # m^2
Vres = (2*Lres) * area_yz   # m^3


def SimPhotDiffFlowGL6(C_molar, Rp, b, Nres, D, totalTime, binDt, w0, axialFactor, includeBg, bgRate, beamShape, vFlow, rng, resFactor=10):
    # 1b) Concentration
    C_m3 = C_molar * 1e3
    
    # 2b) Reservoir initialization
    pos = np.empty((Nres, 3))
    pos[:, 0] = (rng.random(Nres)-0.5)*2*Lres
    pos[:, 1] = (rng.random(Nres)-0.5)*2*Lbox
    pos[:, 2] = (rng.random(Nres)-0.5)*2*Lbox

    # 3) Time step and diffusion
    dt = binDt
    sigma = np.sqrt(2 * D * dt)
    nSteps = int(np.ceil(totalTime/dt))
    if vFlow > 0:
        stepsPerSweep = int(np.ceil((2*Lres) / (vFlow * dt)))
    else:
        stepsPerSweep = int(1e9)
    
    Rp_i = Rp * b
    Rp_i = np.minimum(Rp_i, 1.6e7)
    # poisson_bg = rng.poisson(bgRate * dt, nSteps) if includeBg else np.zeros(nSteps, dtype=np.int64)

    perm_indices_x = rng.integers(0, Nres, (nSteps//stepsPerSweep + 2, Nres))
    perm_indices_y = rng.integers(0, Nres, (nSteps//stepsPerSweep + 2, Nres))
    perm_indices_z = rng.integers(0, Nres, (nSteps//stepsPerSweep + 2, Nres))

    # 4) Preallocate photon times
    Veff = np.pi ** (3/2) * w0**2 * w_z
    Navg = C_m3 * NA * Veff
    expCount = int(np.ceil((Navg*Rp + bgRate) * totalTime * 3.0))
    arrivalTimes = np.empty(expCount, dtype=np.float64)

    # JIT-compiled simulation loop
    arrivalTimes, idx = simulation_loop_jit(
        arrivalTimes,
        pos, Rp_i, w0, w_z, Nres, nSteps, stepsPerSweep, dt, sigma, vFlow, Lres, Lbox,
        includeBg, beamShape, bgRate, perm_indices_x, perm_indices_y, perm_indices_z
    )
    arrivalTimes = arrivalTimes[:idx]

    return arrivalTimes, Rp_i

@njit
def simulation_loop_jit(arrivalTimes, pos, Rp_i, w0, w_z, Nres, nSteps, stepsPerSweep, dt, sigma, vFlow, Lres, Lbox,
                       includeBg, beamShape, bgRate, perm_indices_x, perm_indices_y, perm_indices_z):
    idx = 0
    perm_counter = 0
    for k in range(1, nSteps+1):
        t0 = (k-1) * dt
        # Advect
        pos[:, 0] += vFlow * dt
        pos[:, 0] = np.mod(pos[:, 0] + Lres, 2*Lres) - Lres
        # Diffuse
        inBox = np.abs(pos[:, 0]) <= Lbox
        # n_inBox = np.sum(inBox)
        for i in range(Nres):
            if inBox[i]:
                # Apply diffusion step
                for d in range(3):
                    pos[i, d] += sigma * np.random.standard_normal()

                # Reflect in each axis
                if pos[i, 0] > Lbox:
                    pos[i, 0] = 2 * Lbox - pos[i, 0]
                elif pos[i, 0] < -Lbox:
                    pos[i, 0] = -2 * Lbox - pos[i, 0]
                if pos[i, 1] > Lbox:
                    pos[i, 1] = 2 * Lbox - pos[i, 1]
                elif pos[i, 1] < -Lbox:
                    pos[i, 1] = -2 * Lbox - pos[i, 1]
                if pos[i, 2] > Lbox:
                    pos[i, 2] = 2 * Lbox - pos[i, 2]
                elif pos[i, 2] < -Lbox:
                    pos[i, 2] = -2 * Lbox - pos[i, 2]
        
        # Photon emission
        xy2 = pos[:, 0]**2 + pos[:, 1]**2
        z2 = pos[:, 2]**2
        # if beamShape == 'gaussian':
        #     W = np.exp(-2 * xy2/w0**2 - 2*z2/w_z**2)
        # else:
        Wlat = np.exp(-2*xy2/w0**2)
        Wax = 1 / (1 + z2/w_z**2)
        W = Wlat * Wax
        Rtot = np.sum(Rp_i * W)
        mean_photons = Rtot * dt
        Nph = np.random.poisson(mean_photons)
        Nbg = np.random.poisson(bgRate * dt) if includeBg else 0
        NtotEv = Nph + Nbg
        if NtotEv > 0:
            need = idx + NtotEv
            if need > arrivalTimes.size:
                newcap = arrivalTimes.size
                while need > newcap:
                    newcap*=2
                newArr = np.empty(newcap, dtype=np.float64)
                newArr[:idx] = arrivalTimes[:idx]
                arrivalTimes = newArr

            for j in range(NtotEv):
                arrivalTimes[idx] = t0 + np.random.random() * dt
                idx += 1

        # Permute
        if stepsPerSweep < 1e8 and k % stepsPerSweep == 0:
            perm_x = perm_indices_x[perm_counter]
            perm_y = perm_indices_y[perm_counter]
            perm_z = perm_indices_z[perm_counter]
            pos[:, 0] = pos[perm_x, 0]
            pos[:, 1] = pos[perm_y, 1]
            pos[:, 2] = pos[perm_z, 2]
            perm_counter += 1
    
    return arrivalTimes, idx

def run_one_env(env_num, num_species, amps, conc, widths, D):
    #fixed variables
    D = D                       #m^2/s
    dt = 5e-6                   #s
    vF = 5e-4                   #m/s
    tt = 60                     #s
    setDt = 500e-6              #s
    bgRate = 1e2                #counts/s

    #ENV specific ops
    seed = env_num + int(time.time()) % 10000
    rng = np.random.default_rng(seed)
    start_sim = time.time()
    env_dir = os.path.join(data_dir, f"sim_{env_num:07}")
    os.makedirs(env_dir, exist_ok=True)

    #ENV ground truth
    num_species = num_species   
    total_conc = conc              # M
    
    if num_species == 0: #power law; fractions aren't relevant
        fractions = np.array([1.0], dtype=float)
        dist_type = 'pl'
    else:
        dist_type = 'ln'
        match num_species:
            case 1:
                fractions = np.array([1.0], dtype=float)
                sigma_1 = widths
                AmpS1 = amps
                sigma_b2 = sigma_b3 = sigma_2 = sigma_3 = None
                AmpS2 = AmpS3 = None
                sigma_b1 = float(np.sqrt(np.log((sigma_1 / AmpS1)**2 + 1)))
            case 2:
                fractions = np.array([0.5, 0.5], dtype=float)
                sigma_1, sigma_2 = widths
                AmpS1, AmpS2 = amps
                sigma_b3 = sigma_3 = None
                AmpS3 = None
                sigma_b1 = float(np.sqrt(np.log((sigma_1 / AmpS1)**2 + 1)))
                sigma_b2 = float(np.sqrt(np.log((sigma_2 / AmpS2)**2 + 1)))
            case 3:
                fractions = np.array([0.6, 0.3, 0.1], dtype=float)
                sigma_1, sigma_2, sigma_3 = widths
                AmpS1, AmpS2, AmpS3 = amps
                sigma_b1 = float(np.sqrt(np.log((sigma_1 / AmpS1)**2 + 1)))
                sigma_b2 = float(np.sqrt(np.log((sigma_2 / AmpS2)**2 + 1)))
                sigma_b3 = float(np.sqrt(np.log((sigma_3 / AmpS3)**2 + 1)))

    Frac1 = fractions[0]
    Frac2 = fractions[1] if num_species >= 2 else 0.0
    Frac3 = fractions[2] if num_species == 3 else 0.0
    conc1 = Frac1 * total_conc
    conc2 = Frac2 * total_conc
    conc3 = Frac3 * total_conc

    truedist1 = np.array([])
    truedist2 = np.array([])
    truedist3 = np.array([])
    at1 = np.array([])
    at2 = np.array([])
    at3 = np.array([])
    alpha = (rng.uniform(1.0, 2.5) if rng.integers(0, 2) == 0 else rng.uniform(-3.5, 0)) if num_species < 1 else None

    # precompute underlying dist.
    if num_species >= 1:
        b1, Nres1 = compute_underlying_dist(Rp=AmpS1/500e-6, dist_type=dist_type, sigma_b=sigma_b1, alpha=alpha, rng=rng, conc=conc1)
    else:
        b1, Nres1 = compute_underlying_dist(Rp=AmpS1/500e-6, dist_type=dist_type, sigma_b=sigma_b1, alpha=alpha, rng=rng, conc=conc1)
    if num_species >= 2:
        b2, Nres2 = compute_underlying_dist(Rp=AmpS2/500e-6, dist_type=dist_type, sigma_b=sigma_b2, alpha=alpha, rng=rng, conc=conc2)
    if num_species >= 3:
        b3, Nres3 = compute_underlying_dist(Rp=AmpS3/500e-6, dist_type=dist_type, sigma_b=sigma_b3, alpha=alpha, rng=rng, conc=conc3)

    #run several simulations with identical parameters
    for sim in range(sims_per_env):
        if num_species >= 1:
            at1, truedist1 = SimPhotDiffFlowGL6(C_molar = conc1,
                                                Rp = AmpS1 / setDt,
                                                b = b1,
                                                Nres = Nres1,
                                                D = D, 
                                                totalTime = tt, 
                                                binDt = dt, 
                                                w0 = w0, 
                                                axialFactor = axialFactor, 
                                                includeBg = True, 
                                                bgRate = bgRate, 
                                                beamShape = 'gl',
                                                vFlow=vF,
                                                rng = rng)
        else:
            at1, truedist1 = SimPhotDiffFlowGL6(C_molar = conc1,
                                                Rp = AmpS1 / setDt,
                                                b = b1,
                                                Nres = Nres1,
                                                D = D, 
                                                totalTime = tt, 
                                                binDt = dt, 
                                                w0 = w0, 
                                                axialFactor = axialFactor, 
                                                includeBg = True, 
                                                bgRate = bgRate, 
                                                beamShape = 'gl',
                                                vFlow=vF,
                                                rng = rng)

        if num_species >= 2:
            at2, truedist2 = SimPhotDiffFlowGL6(C_molar = conc2,
                                                Rp = AmpS2 / setDt, 
                                                b = b2,
                                                Nres = Nres2,
                                                D = D, 
                                                totalTime = tt, 
                                                binDt = dt, 
                                                w0 = w0, 
                                                axialFactor = axialFactor, 
                                                includeBg = False, 
                                                bgRate = 0, 
                                                beamShape = 'gl',
                                                vFlow = vF,
                                                rng = rng)
            
        if num_species == 3:
            at3, truedist3 = SimPhotDiffFlowGL6(C_molar = conc3,
                                                Rp = AmpS3 / setDt,
                                                b = b3,
                                                Nres = Nres3,
                                                D = D, 
                                                totalTime = tt, 
                                                binDt = dt, 
                                                w0 = w0, 
                                                axialFactor = axialFactor, 
                                                includeBg = False, 
                                                bgRate = 0, 
                                                beamShape = 'gl',
                                                vFlow=vF,
                                                rng = rng)

        fullBrightDist = np.concatenate([truedist1.flatten()*setDt, truedist2.flatten()*setDt, truedist3.flatten()*setDt])
        
        fullTOAs = np.concatenate([at1, at2, at3])
        fullTOAs = np.sort(fullTOAs)
        bins_hist = np.linspace(0, tt, int((tt)/(setDt)) + 1)
        histA, _ = np.histogram(fullTOAs, bins_hist)
        PCHedges = np.logspace(np.log10(1), np.log10(8000), 25)
        PCHbins, _ = np.histogram(histA, bins=PCHedges)

        GT = {
            "Amplitudes":{
                "AmpS1":AmpS1,
                "AmpS2":AmpS2,
                "AmpS3":AmpS3
            },
            "ActualFractions": {
                "Frac1": float(Frac1),
                "Frac2": float(Frac2),
                "Frac3": float(Frac3)
            },
            "ActualConcentrations":{
                "Species1":float(conc1),
                "Species2":float(conc2),
                "Species3":float(conc3),
                "Total":float(total_conc)
            },
            "ActualSigmas":{
                "Species1":sigma_1,
                "Species2":sigma_2,
                "Species3":sigma_3
            },
            "SimulationInputs":{
                "D":D,
                "totaltime":tt,
                "binDt":dt,
                "w0":w0,
                "axialFactor":axialFactor,
                "vFlow":vF,
                "bgRate":bgRate,
                "disttype":dist_type,
                "sigma_bS1":sigma_b1,
                "sigma_bS2":sigma_b2,
                "sigma_bS3":sigma_b3,
                "alpha":alpha,
                "num_species":int(num_species)
            },
            "Data":{
                "pch_bins":PCHbins.tolist(),
                "raw_particle_amps":fullBrightDist.tolist()
            },
            "Other":{
                "Seed":seed
            }
        }

        if sim == 0:
            np.save(os.path.join(env_dir, f"arrivalTimes_{sim + 1}"), fullTOAs)

        with open(os.path.join(env_dir, f"GT{sim:02}.json"), "w") as f:
            json.dump(GT, f)


    print(f"Finished env {env_num}, Time: {time.time() - start_sim}")
    gc.collect()
    return

if __name__ == '__main__':
    I_L = 1000
    I_Lp = 1500
    
    I_Mm = 2000
    I_M = 2500
    I_Mp = 3000
    
    I_Hm = 3500
    I_H = 4500
    
    N_L = 1.4e-11
    N_M = 4.2e-11
    N_H = 7e-11
    
    W_N = 200
    W_M = 500
    W_H = 1000

    D_L = 1e-11
    D_H = 4e-10

    jobs = [
        (0, 1, I_L, N_L, W_N, D_L),
        (1, 1, I_L, N_H, W_N, D_L),
        (2, 1, I_L, N_L, W_M, D_L),
        (3, 1, I_L, N_H, W_M, D_L),

        (4, 1, I_H, N_L, W_N, D_L),
        (5, 1, I_H, N_H, W_N, D_L),
        (6, 1, I_H, N_L, W_M, D_L),
        (7, 1, I_H, N_H, W_M, D_L),

        (8, 2, (I_L, I_H), N_L, (W_M, W_M), D_L),
        (9, 2, (I_Mp, I_Hm), N_L, (W_M, W_M), D_L),
        (10, 2, (I_Mm, I_Mp), N_L, (W_M, W_M), D_L),

        (11, 2, (I_L, I_H), N_H, (W_M, W_M), D_L),
        (12, 2, (I_Mp, I_Hm), N_H, (W_M, W_M), D_L),
        (13, 2, (I_Mm, I_Mp), N_H, (W_M, W_M), D_L),

        (14, 3, (I_L, I_M, I_H), N_H, (W_M, W_M, W_M), D_L),

        (15, 1, I_M, N_L, W_M, D_L),
        (16, 1, I_M, N_M, W_M, D_L),
        (17, 1, I_M, N_H, W_M, D_L),
        (18, 1, I_M, N_L, W_M, D_H),
        (19, 1, I_M, N_M, W_M, D_H),
        (20, 1, I_M, N_H, W_M, D_H),
    ]

    mp.set_start_method('spawn', force=True)

    num_workers = len(jobs)

    with mp.Pool(processes=num_workers) as pool:
        pool.starmap(run_one_env, jobs)

