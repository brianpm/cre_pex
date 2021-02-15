README.md

This repository contains the code used to perform the analysis and make the figures for:

Medeiros, B., A. C. Clement, J. J. Benedict, & B. Zhang, 2021: Investigating the impact of cloud radiative feedbacks on tropical precipitation extremes, npj Climate and Atmospheric Science, _in press_.

The code is mainly contained in Jupyter notebooks. The figures are each made in a separate notebook, except Figures 2 and 3 are both made in one.

Two additional modules are included to support the analysis:
- `wavenumber_frequency_functions.py` contains code to calculate the wavenumber-frequency spectra; this is a copy of the same file from my [wavnumber_frequency](https://github.com/brianpm/wavenumber_frequency) repository. The file here will stay static; the one in the other repo will be maintained going forward (unless otherwise noted there).
- `detect_and_track.py` provides code that uses [`scipy.ndimage`](https://docs.scipy.org/doc/scipy/reference/ndimage.html) to provide a simple detection and tracking method. Some of this code is related to the methods we have been developing for the detecting [temperature extremes](https://github.com/brianpm/TemperatureExtremes), but here we have added the tracking. We are evaluating the same code for other purposes, including detecting mesoscale convective systems. 

