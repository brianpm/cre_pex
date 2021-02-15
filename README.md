# Cloud-radiative effects & Precipitation Extremes

## Introduction

This repository contains the code used to perform the analysis and make the figures for:

Medeiros, B., A. C. Clement, J. J. Benedict, & B. Zhang, 2021: Investigating the impact of cloud radiative feedbacks on tropical precipitation extremes, npj Climate and Atmospheric Science, _in press_.

The code is mainly contained in Jupyter notebooks. The figures are each made in a separate notebook, except Figures 2 and 3 are both made in one. This `README` provides an annotated list of what is included in the repository.

## Notebooks for figures
The notebooks that are used to create all the figures are provided as:
- `Figure1_precip_zonal_with_histogram_aqua_amip_locking.ipynb`
- `Figure2_and_3_extreme_precip_connections.ipynb`
- `Figure4_extreme_precip_lw_vs_4K.ipynb`
- `Figure5_wavenumber_frequency_testing.ipynb`
- `Figure6_extreme_precip_tracking_summary.ipynb`


## Additional custom modules
Two additional modules are included to support the analysis:
- `wavenumber_frequency_functions.py` contains code to calculate the wavenumber-frequency spectra; this is a copy of the same file from my [wavnumber_frequency](https://github.com/brianpm/wavenumber_frequency) repository. The file here will stay static; the one in the other repo will be maintained going forward (unless otherwise noted there).
- `detect_and_track.py` provides code that uses [`scipy.ndimage`](https://docs.scipy.org/doc/scipy/reference/ndimage.html) to provide a simple detection and tracking method. Some of this code is related to the methods we have been developing for the detecting [temperature extremes](https://github.com/brianpm/TemperatureExtremes), but here we have added the tracking. We are evaluating the same code for other purposes, including detecting mesoscale convective systems. 

## Dependencies
The dependencies are fairly minimal, they are mainly the standard python scientific ecosystem. All the code should run with python 3.6 and up; but no guarantees. My environments included python 3.8 and 3.9. In the following lists, I do not include imports from the standard library.

### Figure 1
- xarray
- numpy
- matplotlib
- cycler (included with matplotlib, I think)
- palettable (some nice color tables)
- colorcet (some more nice color tables)

### Figure 2 & 3
- xarray
- numpy
- matplotlib
- colorcet
- esmlab (this is used for easy weighted averages, but could be replaced without much hassle)
- scipy (the stats module)

### Figure 4
- xarray
- numpy
- matplotlib
- colorcet
- esmlab
- scipy (stats)

### Figure 5
- `wavenumber_frequency_functions`
  - xarray, numpy, scipy.signal are the dependencies in there.
  - After import, you can run the `helper` function to see a list of the included functions.
- numpy
- xarray
- matplotlib
- palettable

### Figure 6
- numpy
- xarray
- pandas
- esmlab
- scipy (stats)
- matplotlib
- colorcet
- palettable

It looks like the Notebook for Figure 6 directly implements functions from `detect_and_track.py`. It would have been better to import the module; I include it in the repository since that is the more portable form of the code. There may be conflicts between that module and the notebook, so be cautious. 

Another notebook called `global_precipitation_energy_constraint.ipynb` has code that produces Table 1 from the paper. It was also used for some other analysis of the surface and TOA energy balances across simulations. It would need a little care before it should be used to make sure sign conventions are correct.


## Additional analyses
A few additional files are included. 
- `check_prw_locking.ipynb` : a notebook that investigates the change in precipitable water with cloud locking. This analysis is incomplete, but was partially developed as part of our response to reviews. 
- `cre_pex_regional_extremes.ipynb` : a notebook that does some analysis to find out where most extreme precipitation occurs. Divides the world into tropical ocean, tropical land, northern extratropics, and southern extratropics. 
- `esgf_download_cmip6.ipynb` : a notebook that has some functions that search/download CMIP6 data. This notebook is in a messy state, but the functions do work, and may be useful. 
- `extreme_precip_prw_amon.ipynb` : a quick look at the zonal mean changes in integrated water vapor.
- `extreme_precip_track_viz.ipynb` : some preliminary visual analysis to validate the detection and tracking method.
- `extreme_precip_wap_profiles.ipynb` : som analysis of the vertical velocity profiles. Mentioned in the paper, but CMIP files have so few levels that these profiles do not seem very useful beyond the obvious reduction in ascent in convective regimes.
- `extreme_precipitation_functions.py` : an incomplete module; has a function to get duration and size of events.
- `mean_stability.ipynb` : an incomplete analysis of atmospheric stability, used as part of the discussion of the TTL stability and the 4xCO2 experiments. 
- `preliminary_lwoff_precipitation_response.ipynb` : this was probably the first notebook that started the project following initial conversations among the authors. Already in this notebook a lot of the main conclusions are being sketched out, and some of the figures come out of this first look at the LWoff experiments.
- `revision_check_fig3_confidence.ipynb` : this is where we did bootstrapped confidence intervals to make sure that Figure 3 is significant.
- `tracker_explainer.ipynb` : some tests of the tracking method; in particular, I was exploring how different structure matrices impact the tracking to approximate merging/splitting.

## Data files
I saved zonal means, quantiles, and histograms as intermediate files to speed up some analysis. I'm including them in the repository because they are small. The file names provide some indication of processing (e.g., region), but I **strongly advise** to use these only for testing purposes, and re-calculate them from the original CMIP files. 

## Contact
Contact me for additional details regarding analysis. This is **not a software package**, and I provide it with no support. The intention is to document the code used for the paper for purposes of transparency. Even if you are attempting to reproduce the results, I advise that you develop your own codes to do it, and compare with the included codes and the paper. I can try to help with reasonable inquiries. CMIP data can be downloaded from the ESGF [https://esgf-node.llnl.gov/search/cmip6/]. The cloud-locking data is available on [Zenodo](https://www.zenodo.org) (doi: `10.5281/zenodo.3591996`). 