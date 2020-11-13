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
    land_f = xr.open_dataset("/project/amp02/brianpm/cesm_landfrac.nc")
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
    connect=None
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
        pr0, m0_threshold, track=track, latitude=latitude, connect=connect
    )


    m1_labels, m1_nfeatures = dt.pr_labeler(
        pr1, m1_threshold, track=track, latitude=latitude, connect=connect
    )

    if lon_wrap:
        m0_transformer = dt.wrap_events(m0_labels)
        m1_transformer = dt.wrap_events(m1_labels)
        m0_labels = dt.event_wrapper(m0_labels, m0_transformer)
        m1_labels = dt.event_wrapper(m1_labels, m1_transformer)

    return m0_labels, m1_labels


if __name__ == "__main__":
    ## LOCKING RUNS ARE NOT IN CMIP FORMAT
    dloc = Path("/project/amp02/brianpm")
    ctl_ds = xr.open_dataset(dloc / 'B1850_c201_CTL.cam.h4.PRECT.nc', decode_times=False)
    lck_ds = xr.open_dataset(dloc / 'B1850_c201_CLOCK.cam.h4.PRECT.nc', decode_times=False)
    ctl_ds = cesm_fix_timedim(ctl_ds)
    lck_ds = cesm_fix_timedim(lck_ds)
    p_cntl = ctl_ds['PRECT']*86400.*1000.  # m/s -> mm/day
    p_lock = lck_ds['PRECT']*86400.*1000.

    cntl_labels, lock_labels = cesm_pr_label_workflow(
        p_cntl,
        p_lock,
        remove_land=True,
        label="3d",
        latitude=slice(-30,30),
        lon_wrap=True,
        connect=1
    )
    dt.summarize_events(cntl_labels, "CESM2", "control")
    dt.summarize_events(lock_labels, "CESM2", "lock")
    #
    # Save labeled events so as to enable easier filtering
    #
    cntl_labels.name = "precip_events"
    lock_labels.name = "precip_events"
    outloc = Path("/project/amp02/brianpm/")

    if 'type' in cntl_labels.coords:
        if cntl_labels['type'].shape == ():
            cntl_labels = cntl_labels.drop('type')

    if 'type' in lock_labels.coords:
        if lock_labels['type'].shape == ():
            lock_labels = lock_labels.drop('type')

    # Order doesn't matter here
    # B-case LOCKED
    oname2 = outloc / "pr_events_6hr_B1850_c201_CLOCK.nc"
    lock_labels.to_netcdf(oname2)

    # B-case CONTROL
    oname3 = outloc / "pr_events_6hr_B1850_c201_CTL.nc"
    cntl_labels.to_netcdf(oname3)



