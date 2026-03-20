Please email me if you find any bugs at om.guin@gmail.com

## Dependencies

All dependencies can be installed with `pip install -r requirements.txt`

## Datasets

Make sure to download `pch_val.zip` and `pch_train.zip`, not the `pchdata` subfolder. These `.zip` files should be in the same directory as the notebooks/scripts.<br>
Download training dataset here (637 mb): https://drive.google.com/file/d/18CvMuYGEKRgVnylyroAIdB9wS38fvBDw/view?usp=sharing (Last updated 3/19/26)<br>
Download testing dataset here (27.8 mb): https://drive.google.com/file/d/1XE1UYzCkCGqOBiRjT28pyeLry37ZwDZr/view?usp=sharing (Last updated 3/19/26)

It is best to run the commands in the first cell of  `val_runs.ipynb` for loading these datasets instead of unzipping manually because those commands will handle the removal of the incomplete files.

## Simulations

All simulations generate a folder with a subfolder called `pchdata`. For use with deep learning, download the subfolder, not the folder.

`sim_datagen_train.py`: Training dataset generation. Arguments: num_workers
- Example usage: `python sim_datagen_train.py --num_workers=200`
- Simulation will parallelize across `num_workers` processes. Ensure sufficient cpu count. Each worker requires <10 mb of RAM, so it should not be a bottleneck.

`sim_dataMCval.py`: Testing dataset generation. Arguments: num_workers, time_vary_analysis
- Example usage: `python sim_dataMCval_1a.py --num_workers=200 --no-time_variable_analysis`
- Example usage: `python sim_dataMCval_1a.py --num_workers=200 --time_variable_analysis`
- The second example turns on the variable time analysis. If this is on, each simulation will run for 120 (virtual) seconds instead of 60 and save the photon counting histogram at 30 seconds, 60 seconds, and 120 seconds. This allows for analysis across differnet event counts

`sim_datagen.py`: Testing set generation for specific test cases.
- Manually edit test cases at bottom of file.
- For example, the following tuple:  (X, 2, (I_Mm, I_Mp), N_L, (W_M, W_M), D_L), describes a distribution with two species, one at 2000 counts/500μs and another at 3000 counts/500μs, low concentration ($1.4*10^{-11}$ M), both medium width ($\sigma$ = 500), and diffusion constant of $10^{-11}$. X is the ID for test case; it can be any number as long as all IDs are exclusive so they dont overwrite each other.
- Example usage: `python sim_datagen.py --diff_event_counts`
- Example usage: `python sim_datagen.py --no-diff_event_counts`
- `diff_event_counts` does the same thing as `time_variable_analysis` as sim_dataMCval_1a.py
- If using different_event_counts, the default time is 120 seconds, and it will save the 30 second and 60 second PCHs. This should be manually changed near the bottom if different times are desired.
- saves arrival times in numpy array

`utils.py`: Contains utility functions for generating distributions for all simulation scripts. 

## Deep Learning

`train.py`: Train linear/log model. Arguments: learning rate, batch size, epochs, linear/log, output bins, inital beta, save, progress bar
- Example usage: `python train.py --lr=2e-4 --batch_size=32 --num_epochs=100 --bin_type='lin' --output_bins=50 --init_beta=0.1`
- Example usage: `python train.py --no-save --no-prog_bar --bin_type='log'`
- Saves model in `models/`, loss curve figure, beta curve figure, and `.pkl` of loss/beta curves in `loss_curves/` 

`models.py`, `datasets.py`: Contain classes for models and datasets, respectively.

`val_runs.ipynb`: Notebooks for validation/testing for models; also has linux commands for unpacking datasets.
- This notebook has some irrelevant code; will update soon.

## Processing Scripts

`TAMUPhotonsToPCH.py`: Converts TAMU photon timestamp `.txt` files to numpy array of photon counting histograms.
- Example usage: `python TAMUPhotonsToPCH.py "C:\path" --num_channels`
- Outputs `num_channels` arrays of photons for each file

`arrivalTimesToBAS.py`: Converts numpy array of arrival times (`.npy`) OR `.txt` to TAMU photon timestamps and `.par` files.
- Outputs `.par` files needed for BAS & BAS preprocessing
- Example usage:
  + `.npy`: `python arrivalTimesToBAS.py path\to\arrivalTimes.npy`
  + Single `.txt`: `python arrivalTimesToBAS.py path\to\arrivalTimes.txt`
  + Folder of `.txt`: `python arrivalTimesToBAS.py path\to\data_folder --pre-name Pre_name.par --bas-name BAS_name.par`