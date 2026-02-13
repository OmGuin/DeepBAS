#imports
import time
import numpy as np
import os
import json
from numba import njit
import multiprocessing as mp
import gc
import argparse
from utils import compute_underlying_dist

#directory
data_dir = os.path.join("data_gen", "pchdata")
os.makedirs(data_dir, exist_ok=True)
metadata = {
    "D":1e-11,
    "binDt":5e-6,
    "w0":3e-7,
    "axialFactor":3,
    "vFlow":5e-4,
    "bgRate":1e2,
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

def run_one_env(env_num, time_variable_analysis):
    #fixed variables
    D = 1e-11                                            #m^2/s
    dt = 5e-6                                            #s
    vF = 5e-4                                            #m/s
    tt = 120 if time_variable_analysis else 60           #s
    setDt = 500e-6                                       #s
    bgRate = 1e2                                         #counts/s

    #ENV specific ops
    seed = env_num + int(time.time()) % 10000
    rng = np.random.default_rng(seed)
    start_sim = time.time()
    env_dir = os.path.join(data_dir, f"sim_{env_num:07}")
    os.makedirs(env_dir, exist_ok=True)

    #ENV ground truth
    num_species = rng.integers(0,4) #[0,3]   
    total_conc = rng.uniform(9e-12, 7e-11)              # M
    
    if num_species == 0: #power law; fractions aren't relevant
        fractions = np.array([1.0], dtype=float)
        dist_type = 'pl'
    else:
        dist_type = 'ln'
        min_frac = 0.15
        while True:
            fractions = rng.dirichlet([1.5] * num_species)
            if np.all(fractions > min_frac):
                break

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

    amps = np.sort(rng.integers(50, 5000, size=num_species))
    while num_species > 1 and not np.all(np.diff(amps) > 400):
        amps = np.sort(rng.integers(50, 5000, size=num_species))

    AmpS1 = int(amps[0]) if num_species >= 1 else float(rng.integers(50, 5000))
    AmpS2 = int(amps[1]) if num_species >= 2 else None
    AmpS3 = int(amps[2]) if num_species >= 3 else None

    sigmas = rng.uniform(200, 1000, size=num_species)
    sigma_1 = float(sigmas[0]) if num_species >= 1 else None
    sigma_2 = float(sigmas[1]) if num_species >= 2 else None
    sigma_3 = float(sigmas[2]) if num_species >= 3 else None
    sigma_b1 = float(np.sqrt(np.log((sigma_1 / AmpS1)**2 + 1))) if num_species >= 1 else None
    sigma_b2 = float(np.sqrt(np.log((sigma_2 / AmpS2)**2 + 1))) if num_species >= 2 else None
    sigma_b3 = float(np.sqrt(np.log((sigma_3 / AmpS3)**2 + 1))) if num_species >= 3 else None
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
                                                vFlow=vF,
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
        PCH60, _ = np.histogram(histA[:120000], bins=PCHedges)

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
                "disttype":dist_type,
                "sigma_bS1":sigma_b1,
                "sigma_bS2":sigma_b2,
                "sigma_bS3":sigma_b3,
                "alpha":alpha,
                "num_species":int(num_species)
            },
            "Data":{
                "pch_bins":PCH60.tolist(),
                "raw_particle_amps":fullBrightDist.tolist()
            },
            "Other":{
                "Seed":seed
            }
        }

        if time_variable_analysis:
            PCH30, _ = np.histogram(histA[:60000], bins=PCHedges)
            PCH120, _ = np.histogram(histA, bins=PCHedges)
            GT['DifferentEventCount'] = {
                "pch30": PCH30.tolist(),
                "pch120":PCH120.tolist()
            }


        with open(os.path.join(env_dir, f"GT{sim:02}.json"), "w") as f:
            json.dump(GT, f)


    print(f"Finished env {env_num}, Time: {time.time() - start_sim}")
    gc.collect()
    return 


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='parallel monte carlo simulations')
    parser.add_argument('--num_workers', type=int, default=200)
    parser.add_argument('--time_variable_analysis', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    
    num_sims = 1000000
    num_workers = args.num_workers

    with mp.Pool(processes=num_workers) as pool:
        pool.map(run_one_env, range(num_sims), args.time_variable_analysis)

