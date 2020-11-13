import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.ndimage import label, generate_binary_structure
import detect_and_track as dt
from tqdm import tqdm


def get_tropics_quantiles(da, threshold=None, q=0.99):
    da_tropics = da.sel(lat=slice(-30, 30))  # SHOULD BE ADJUSTABLE
    da_tropics_val = np.nanquantile(da_tropics, q)
    if threshold is not None:
        da_climo = da.mean(dim="time")
        da_process = np.where(
            (da_climo >= threshold).sel(lat=slice(-30, 30)).broadcast_like(da_tropics),
            da_tropics,
            np.nan,
        )
        da_threshold_val = np.nanquantile(da_process, q)
        print(
            f"Compare the tropics region ({da_tropics_val}) to the rainy tropics ({da_threshold_val})"
        )
        return da_threshold_val
    else:
        print(f"Quantile value: {da_tropics_val}")
        return da_tropics_val
    

def cesm_rmland(d):
    land_f = xr.open_dataset("/Users/brianpm/Dropbox/Data/cesm2_f09_land.nc")
    landx = land_f['LANDFRAC'].squeeze()
    land0, _ = xr.broadcast(landx, d)
    return d.where(land0 <= 0)

    
def cesm_fix_timedim(ds):
    nt = ds['time_bnds'].mean(dim='nbnd')
    nt.attrs = ds['time'].attrs
    ds['time'] = nt
    return xr.decode_cf(ds)


def cesm_pr_label_workflow(
    pr0,
    pr1,
    remove_land=False,
    label="2d",
    latitude=None,
    lon_wrap=True,
):
    """Uses daily precipitation data for 2 simulations, 
    calculates the 99th percentile of tropical precipitation,
    detects precipitation features.
    Options:
        - remove_land : loads land fraction for the simulation and removes land.
        - label : if '2d' do detection on time slices independently, 
                  if '3d' do detection in time (cheap tracking).
        - latitude : use a subset of latitudes, typically provide a slice object.
        - lon_wrap : if true run the code to connect events across longitude boundary.
        
    Commented out the `get_simple_size` lines... do those afterward.
    
    No longer return number of features. Depends on whether tracking or not, also changes
    if lon_wrap is True, so need to re-calculate either way. 
    """
    if remove_land:
        print("REMOVE LAND")
        pr0 = cesm_rmland(pr0)
        pr1 = cesm_rmland(pr1)
        print(f"SHAPE OF MASKED: {pr0.shape}")
    m0_threshold = get_tropics_quantiles(pr0, q=0.99)
    m1_threshold = get_tropics_quantiles(pr1, q=0.99)
    
    if label == "2d":
        track = False
    elif label == "3d":
        track = True

    m0_labels, m0_nfeatures = dt.pr_labeler(
        pr0, m0_threshold, track=track, latitude=latitude
    )


    m1_labels, m1_nfeatures = dt.pr_labeler(
        pr1, m1_threshold, track=track, latitude=latitude
    )

    if lon_wrap:
        m0_transformer = dt.wrap_events(m0_labels)
        m1_transformer = dt.wrap_events(m1_labels)
        m0_labels = dt.event_wrapper(m0_labels, m0_transformer)
        m1_labels = dt.event_wrapper(m1_labels, m1_transformer)

    return m0_labels, m1_labels


if __name__ == "__main__":
    ## LOCKING RUNS ARE NOT IN CMIP FORMAT
    flock = xr.open_dataset('/Volumes/Samsung_T5/F1850JJB_c201_CLOCK.cam.h2.ncrcat.PRECT.nc')
    flock = cesm_fix_timedim(flock)
    fc = xr.open_dataset('/Volumes/Samsung_T5/F1850JJB_c201_CTL.cam.h2.ncrcat.PRECT.nc')
    fc = cesm_fix_timedim(fc)
    p_cntl = fc['PRECT']*86400.*1000.  # m/s -> mm/day
    p_lock = flock['PRECT']*86400.*1000.

    ## Coupled runs
    clock = xr.open_dataset('/Volumes/Samsung_T5/B1850_c201_CLOCK/daily/B1850_c201_CLOCK.cam.h2.ncrcat.PRECT.nc')
    clock = cesm_fix_timedim(clock)
    cc = xr.open_dataset('/Volumes/Samsung_T5/B1850_c201_CTL/daily/B1850_c201_CTL.cam.h2.ncrcat.PRECT.nc')
    cc = cesm_fix_timedim(cc)
    p_cc = cc['PRECT']*86400.*1000.  # m/s -> mm/day
    p_clock = clock['PRECT']*86400.*1000.

    cntl_labels, lock_labels = cesm_pr_label_workflow(
        p_cntl,
        p_lock,
        remove_land=True,
        label="3d",
        latitude=slice(-30,30),
        lon_wrap=True,
    )
    summarize_events(cntl_labels, "CESM2", "F-control")
    summarize_events(lock_labels, "CESM2", "F-lock")
    bc_labels, clock_labels = cesm_pr_label_workflow(
        p_cc,
        p_clock,
        remove_land=True,
        label="3d",
        latitude=slice(-30,30),
        lon_wrap=True,
    )
    summarize_events(bc_labels, "CESM2", "C-control")
    summarize_events(clock_labels, "CESM2", "C-lock")  # "C" stands for coupled

    #
    # Save labeled events so as to enable easier filtering
    #
    cntl_labels.name = "precip_events"
    lock_labels.name = "precip_events"
    bc_labels.name = "precip_events"
    clock_labels.name = "precip_events"
    outloc = Path("/Volumes/Glyph6TB/cloud_locking/pr_events/")

    if 'type' in cntl_labels.coords:
        if cntl_labels['type'].shape == ():
            cntl_labels = cntl_labels.drop('type')

    if 'type' in lock_labels.coords:
        if lock_labels['type'].shape == ():
            lock_labels = lock_labels.drop('type')

    if 'type' in bc_labels.coords:
        if bc_labels['type'].shape == ():
            bc_labels = bc_labels.drop('type')

    if 'type' in clock_labels.coords:
        if clock_labels['type'].shape == ():
            clock_labels = clock_labels.drop('type')

    # Order doesn't matter here
    # F-case LOCKED
    oname0 = outloc / "pr_events_F1850JJB_c201_CLOCK.nc"
    lock_labels.to_netcdf(oname0)

    # F-case CONTROL
    oname1 = outloc / "pr_events_F1850JJB_c201_CTL.nc"
    cntl_labels.to_netcdf(oname1)

    # B-case LOCKED
    oname2 = outloc / "pr_events_B1850_c201_CLOCK.nc"
    clock_labels.to_netcdf(oname2)

    # B-case CONTROL
    oname3 = outloc / "pr_events_B1850_c201_CTL.nc"
    bc_labels.to_netcdf(oname3)



