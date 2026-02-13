Please email me if you find any bugs at om.guin@gmail.com

## Dependencies

All dependencies can be installed with `pip install -r requirements.txt`

## Datasets

Make sure to download `pch_val.zip` and `pch_train.zip`, not the `pchdata` subfolder. These .zip files should be in the same directory as the notebooks/scripts.<br>
Download training dataset here (104.6 mb): https://drive.google.com/file/d/1NZdXhGq6dqxh_PTQscNXchs6OcnoIdER/view?usp=sharing (Last updated 2/12/26)<br>
Download testing dataset here (29.2 mb): https://drive.google.com/file/d/1w7wPlNZVNjjvFcUDsCCKjYkz2hxxJbaj/view?usp=sharing (Last updated 2/12/26)

## Simulations

All simulations generate a folder with a subfolder called `pchdata`. Download the folder, not the subfolder.

`sim_datagen_1a.py`: Training dataset generation. Arguments: num_workers
- Example usage: `python sim_datagen_1a.py --num_workers=200`
- Simulation will parallelize across `num_workers` processes. Ensure sufficient cpu count. Each worker requires <10 mb of RAM, so it should not be a bottleneck.

`sim_dataMCval.py`: Testing dataset generation. Arguments: num_workers, time_vary_analysis
- Example usage: `python sim_dataMCval_1a.py --num_workers=200, --no-time_vary_analysis`
- Example usage: `python sim_dataMCval_1a.py --num_workers=200, --time_variable_analysis`
- The second example turns on the variable time analysis. If this is on, each simulation will run for 120 (virtual) seconds instead of 60 and save the photon counting histogram at 30 seconds, 60 seconds, and 120 seconds.


`sim_datagen.py`: Testing set generation for specific test cases.
- Manually edit test cases at bottom of file.
- For example, the following tuple:  (X, 2, (I_Mm, I_Mp), N_L, (W_M, W_M), D_L), describes a distribution with two species, one at 2000 counts/500μs and another at 3000 counts/500μs, low concentration ($1.4*10^{-11}$ M), both medium width ($\sigma$ = 500), and diffusion constant of $10^{-11}$. X is the ID for test case; it can be any number as long as all IDs are exclusive so they dont overwrite each other.
- saves arrival times in numpy array

`utils.py`: Contains utility functions for generating distributions for all simulation scripts. 

## Deep Learning

`train.py`: Train linear/log model. Arguments: learning rate, batch size, epochs, linear/log, output bins, inital beta, save, progress bar
- Example usage: `python train.py --lr=2e-4 --batch_size=32 --num_epochs=100 --bin_type='lin' --output_bins=50 --init_beta=0.1`
- Example usage: `python train.py --no-save --no-prog_bar --bin_type='log'`
- Saves model in `models/`, loss curve figure, beta curve figure, and `.pkl` of loss/beta curves in `loss_curves/` 

`models.py`, `datasets.py`: Contain classes for models and datasets, respectively.

`CNN_LINEAR.ipynb`, `CNN_LOG.ipynb`: Notebooks for validation/testing for linear and log models, respectively

## Processing Scripts

`tamu_photons_to_pch.py`: Converts TAMU photon timestamp `.txt` files to numpy array of photon counting histograms.
- Example usage: `python tamu_photons_to_pch.py "C:\path" --num_channels`
- Outputs `num_channels` arrays of photons for each file

`arrivalTimesToBAS.py`: Converts numpy array of arrival times to TAMU photon timestamps
- Example usage: `python arrivalTimesToBAS.py "C:\path"`
- Outputs `.par` files needed for BAS & BAS preprocessing