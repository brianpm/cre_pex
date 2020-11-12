import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.ndimage import label, generate_binary_structure

from tqdm import tqdm

def quick_load_var(loc, var, table, model, experiment, member):
    """Convenience for loading CMIP data."""
    fils = sorted(
        list(Path(loc).glob("_".join([var, table, model, experiment, member, "*.nc"])))
    )
    if len(fils) == 0:
        raise IOError("Can not find the files.")
    elif len(fils) == 1:
        ds = xr.open_dataset(fils[0])
    else:
        ds = xr.open_mfdataset(fils, combine="by_coords")
    return ds[var].compute()


#
# --- Detection and tracking by labeling events ---
# 
def get_labels_np(da, crit):
    """Labels contiguous blobs (da >= crit) in 2d numpy arrays.
    
    The 2D version can be used to develop different tracking methods.
    Assumes da has (time, lat, lon), and does identification in (lat, lon),
    returns lbarr (time, lat, lon) of labels
            and
            nfarr the number of labels at each time.
    
    """
    mask = np.where(da >= crit, 1, 0)
    assert len(mask.shape) == 3  # has to be time,lat,lon
    labeled_arrays = []
    n_features = []
    for t in range(mask.shape[0]):
        la, nf = label(mask[t, :, :], structure=generate_binary_structure(2, 2))
        labeled_arrays.append(la)
        n_features.append(nf)
    nfarr = np.array(n_features)
    lbarr = np.stack(labeled_arrays)
    return lbarr, nfarr


def get_labels3d_np(da, crit):
    """Labels contiguous blobs in 3D numpy arrays.
       Uses a structure/connectivity matrix that keeps
       blobs together if they move by 1 non-diagonal point
       in time. Thus prefers long-lived systems.
    """
    mask = np.where(da >= crit, 1, 0)
    assert len(mask.shape) == 3  # has to be time,lat,lon
    la, nf = label(mask, structure=generate_binary_structure(3, 2))
    # nf is just an integer
    return la, nf


def get_labels(da, crit):
    """Same as get_labels_np, but assumes an xarray DataArray as input and returns a DataArray."""
    assert "time" in da.dims
    assert "lat" in da.dims
    assert "lon" in da.dims
    mask = np.where(da >= crit, 1, 0)
    assert len(mask.shape) == 3  # has to be time,lat,lon
    labeled_arrays = []
    n_features = []
    for t in range(mask.shape[0]):
        la, nf = label(mask[t, :, :], structure=generate_binary_structure(2, 2))
        la = xr.DataArray(
            la, dims=("lat", "lon"), coords={"lat": da.lat, "lon": da.lon}
        )
        labeled_arrays.append(la)
        n_features.append(nf)
    nfarr = xr.DataArray(np.array(n_features), dims="time", coords={"time": da.time})
    lbarr = xr.concat(labeled_arrays, dim="time")
    lbarr = lbarr.assign_coords({"time": da.time})
    return lbarr, nfarr


def get_labels3d(da, crit):
    """Label events through time. Assumes DataArray, otherwise as in get_labels3d_np."""
    assert "time" in da.dims
    assert "lat" in da.dims
    assert "lon" in da.dims
    mask = np.where(da >= crit, 1, 0)
    assert len(mask.shape) == 3  # has to be time,lat,lon
    la, nf = label(mask, structure=generate_binary_structure(3, 2))
    la = xr.DataArray(la, dims=da.dims, coords=da.coords)
    # nf is just an integer
    return la, nf


def alt_get_labels3d(da, crit):
    """Label events through time. (numpy array or xarray) 
    
    This version uses a specific structure matrix that
    is equivalent to specifying overlap in time.
    The difference from using a default generate_binary_structure(3, 2)
    matrix is that it handles merging/splitting a little more aggressively.
    """
    mask = np.where(da >= crit, 1, 0)
    assert len(mask.shape) == 3  # has to be time,lat,lon
    structure_matrix = np.array([[[False,  False, False],
        [ False,  True,  False],
        [False,  False, False]],

       [[ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True]],

       [[False,  False, False],
        [ False,  True,  False],
        [False,  False, False]]])
    la, nf = label(mask, structure=structure_matrix)
    if isinstance(da, xr.DataArray):
        la = xr.DataArray(la, dims=da.dims, coords=da.coords)
    # nf is just an integer
    return la, nf


def pr_labeler(p, pcrit, track=False, latitude=None, connect=None):
    """Control the flow to the different labeling functions.
    
        If connect is None, assumes neighbor connectivity in time,
        otherwise tracks using strict temporal overlap.
        
        Input
        -----
        p       : the data
        pcrit   : the threshold to use for detecting blobs/features/events
        track   : if True apply temporal tracking (else label events in each time slice)
        latitude: a `slice` object to subset latitudes.
        connect : When track is True, if connect is not None, use strict temporal overlap.
        
        Returns
        -------
        labels    : array of labeled events
        nfeatures : number of labeled events (total if track is True, in time if track is False)
        
    """
    if latitude is not None:
        pw = p.sel(lat=latitude)
    else:
        pw = p
    if track:
        if connect is None:
            labels, nfeatures = get_labels3d(pw, pcrit)
        else:
            labels, nfeatures = alt_get_labels3d(pw, pcrit)
    else:
        labels, nfeatures = get_labels(pw, pcrit)
    return labels, nfeatures


# We should account for wrapping around longitude
# * https://stackoverflow.com/questions/55953353/how-to-specify-a-periodic-connection-for-features-of-scipy-ndimage-label
# if this is fast enough, it will also provide a vector of features per time

# Performance notes:
# This is potentially very slow.
# Naive approach would loop over all times and all latitudes, which is already a bit slow.
# Pretty easy to reduce that by checking whether there are any values to check. 
# Even reducing the loops to what I think is minimal, it is slow if we 
# try to update the array with boolean indexing for each identified wrapped feature.
# For example, doing new_labels[new_labels == new_labels[t, y, 0]] = older_feature
# I'm not sure exactly why this is so slow.
# MUCH, MUCH faster is to make a list of changes that need to be made.
# Here, `transformer` is the list, just a dictionary with keys being the feature to change
# and values being the corrected value. Creating this dictionary is very fast.
# Then a separate step loops through the dictionary and actually does the change,
# for example using labels = np.where(labels==dictkey, dictvalue, labels)
# In my test case, my initial attempts were going to run for ~2.5 hours,
# but in the transformer/where steps, that was reduced to ~5 minutes.

def wrap_events(input_labels):
    if isinstance(input_labels, xr.DataArray):
        labels = input_labels.values
    else:
        labels = input_labels
    new_labels = labels.copy()
    transformer = dict()
    for t in tqdm(range(new_labels.shape[0])):
        if np.all(new_labels[t,:,0] == 0) or np.all(new_labels[t,:,-1] == 0):
            continue  # no reason to check if no events on edges
        check_lats = np.intersect1d(np.nonzero(new_labels[t, :, 0]), np.nonzero(new_labels[t, :, -1]))
        if len(check_lats) == 0:
            continue
        for y in check_lats:
            if new_labels[t,y,0] != new_labels[t,y,-1]:
                # now we know that [t, y, 0] and [t, y, -1] should be part of same feature, and the label should be the smaller of the two
                older_feature = np.min([new_labels[t, y, 0], new_labels[t, y, -1]])
                if new_labels[t, y, 0] < new_labels[t, y, -1]:
                    transformer[new_labels[t, y, -1]] = older_feature
                else:
                    transformer[new_labels[t, y, 0]] = older_feature
    return transformer


def event_wrapper(labels, transformer_dict):
    if isinstance(labels, xr.DataArray):
        lll = labels.values
    else:
        lll = labels
    new_labels = lll.copy()
    for k in tqdm(transformer_dict):
        new_labels = np.where(new_labels==k, transformer_dict[k], new_labels)
    if isinstance(labels, xr.DataArray):
        new_labels = xr.DataArray(new_labels, dims=labels.dims, coords=labels.coords)
    return new_labels


#
#  --- Processing labeled events ---
#


def get_events_per_time(larr):
    return [len(np.unique(larr[t,:,:]))-1 for t in range(larr.shape[0])]


def get_simple_size(larr):
    """larr is set of labeled arrays. 
    Count how many points have each label, 
    return average and standard deviation 
    of that count for each time."""
    a_size = []
    a_sizestd = []
    for i, tim in enumerate(larr["time"]):
        ctr = np.bincount(larr[i, :, :].values.flatten())
        a_size.append(ctr[1::].mean())
        a_sizestd.append(ctr[1::].std())
    a_size = np.array(a_size)
    a_sizestd = np.array(a_sizestd)
    return a_size, a_sizestd

def get_initial_tindex(labels, ev):
    if isinstance(labels, np.ndarray):
        labv = labels
    else:
        labv = labels.values  # converts to numpy array
    for t in range(labv.shape[0]):
        tmp = labv[t,:,:] # temporary time slice
        present_events = set(tmp[tmp != 0]) # uniq events in this time
        if ev in present_events:
            return t
    return None

def get_duration(lab):
    """Returns a dictionary with keys that are event labels and values that are duration (days)."""
    dur = dict()
    if isinstance(lab, np.ndarray):
        labv = lab
    else:
        labv = lab.values  # converts to numpy array
    for t in range(lab.shape[0]):
        tmp = labv[t,:,:] # temporary time slice
        present_events = set(tmp[tmp != 0]) # uniq events in this time
        if len(present_events) > 0:
            for ev in present_events:
                if ev in dur:
                    dur[ev] += 1 # continuing
                else:
                    dur[ev] = 1 # new event
    return dur


def get_duration_and_size(lab, rettype='df'):
    """Returns either a Pandas DataFrame (default) or dictionary with keys that are event labels and values that are duration (days)."""
    dur = dict()
    # put eventID as key, values are dict: {duration: D, size: [s_0, ..., s_d]}
    if isinstance(lab, np.ndarray):
        labv = lab
    else:
        labv = lab.values  # converts to numpy array
    for t in range(lab.shape[0]):
        tmp = labv[t,:,:] # temporary time slice
        present_events = set(tmp[tmp != 0]) # uniq events in this time
        if len(present_events) > 0:
            for ev in present_events:
                if ev in dur:
                    dur[ev]['duration'] += 1 # continuing
                    dur[ev]['size'].append(np.count_nonzero(tmp==ev))
                else:
                    dur[ev] = dict() # new event
                    dur[ev]['duration'] = 1
                    dur[ev]['size'] = [np.count_nonzero(tmp==ev)]
    if rettype == 'df':
        # convert to dataFrame
        df = pd.DataFrame.from_dict(dur, orient ='index')
        # compute average size and include as a column
        asiz = []
        msiz = []
        for s in df['size']:
            asiz.append(np.mean(s))
            msiz.append(np.max(s))
        df['AvgSiz'] = asiz
        df['MaxSiz'] = msiz
        # get distribution of average size
        return df
    else:
        return dur  # AvgSiz and MaxSiz not included in output dict.


def summarize_events(labels, mstr, estr):
    """Provide text summary of events via simple print."""
    X = get_events_per_time(labels)
    Y = get_duration_and_size(labels)  # Y is a pandas DF with columns: duration, size, AvgSiz, MaxSiz
    tot_number_events = len(Y)
#     dur = []
#     avgsiz = []
#     mxsiz = []
#     for ev in Y:
#         dur.append(Y[ev]["duration"])
#         avgsiz.append(np.mean(Y[ev]["size"]))
#         mxsiz.append(np.max(Y[ev]["size"]))
    avg_duration = np.mean(Y['duration'])
    std_duration = np.std(Y['duration'])
    avg_size = np.mean(Y['AvgSiz'])
    std_size = np.std(Y['AvgSiz'])
    avg_mx_size = np.mean(Y['MaxSiz'])
    std_mx_size = np.std(Y['MaxSiz'])
    print(
        f"{mstr} {estr}: [N_events] = {np.mean(X):.2f} ± {np.std(X):.2f}, Ntot: {tot_number_events:.2f}, [D] = {avg_duration:.2f} ± {std_duration:.2f}, [S] = {avg_size:.2f} ± {std_size:.2f}, [Smx] = {avg_mx_size:.2f} ± {std_mx_size:.2f}"
    )