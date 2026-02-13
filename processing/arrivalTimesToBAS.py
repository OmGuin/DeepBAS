#this script takes in a file of single channel photon arrival times (arrivalTimes.npy)
#and creates .par files for both preprocessing code and BAS code


import numpy as np
import os
import argparse
from datetime import datetime


parser = argparse.ArgumentParser(description='convert simulated data to format needed for preprocessing -> bas')
parser.add_argument('arrivalTimesPath', type=str, help='path to arrivalTimes numpy array')
args = parser.parse_args()

aTpath, aTfile = os.path.split(args.arrivalTimesPath)
arrivalTimes = np.load(args.arrivalTimesPath)

txtFile = aTfile.removesuffix('.npy')+".txt"
txtPath = os.path.join(aTpath, txtFile)
parPath = txtPath.replace('.txt', '.par')
basPath = txtPath[:-4]+'_BAS.par'

with open(txtPath, 'w') as f:
    f.write(f'{datetime.now().strftime("%d%b%y-%H%M%S")}\n\n')
    f.write('Total acquisition time: 60 sec \n')
    f.write('MT Clock: 1 sec\n')
    f.write('Bin width for correlator intensity data: 500.00 usec \n')
    f.write('Scan diamater: 0.40 mm \n')
    f.write('Scan speed: 0.50 mm/sec \n')
    f.write('Lasers Active: 488 @ 0.050 mW \n')
    f.write('Laser modulation not active \n')
    f.write('Notes: \n')
    f.write('Simulated data\n')
    f.write('***end header*** \n\n')
    f.write('I_A  I_B \n')
    for timeStamp in arrivalTimes:
        f.write(str(timeStamp)+'\n')

with open(parPath, 'w') as f:
    f.write('%%%% (maintain 5 header lines) \n')
    f.write('These data represent simulated data with 3 independent populations \n')
    f.write('These header lines can be used to explain the point of the data collection \n')
    f.write('and point to other informative files with additional information. \n')
    f.write('%%%% \n')
    f.write('\n')
    f.write(txtFile)
    f.write('\n\n')
    f.write('datatype TotTime MTclock bintime driftwinA driftwinB driftwinC threshA threshB threshC corrt dofit colA colB colC\n')
    formatSpec = '%d %f %.2e %.6f %d %d %d %d %d %d %.3f %d %d %d %d\n'
    f.write(formatSpec % (1, 60, 1, 500e-6, 0, 0, 0, 1, 1, 1, 0.003, 0, 1, 0, 0))
    f.write('\n')


with open(basPath, 'w') as f:
    f.write('%%%%    (maintain 5 header lines)\n')
    f.write('Here there be parameters associated with \n')
    f.write('BAS processing (use= 1 or 0). Seems better to include in a separate parameter file from\n')
    f.write('Preprocessing; Offset 0 = none, 1 = mean , 2 = median, 3 = rms\n')
    f.write('%%%%\n\n')

    f.write(txtFile.replace('.txt', '.par') + '\n\n')

    f.write('useA useB useC OffsetA  OffsetB OffsetC smooth showfig\n')
    f.write('1 1 1 0 0 0 1 1\n')


