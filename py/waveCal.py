# Collection of all wavelength related functions so far
import os
from glob import glob
import numpy as np
from numpy.polynomial.polynomial import polyvander2d, polyval2d
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.constants import c
from scipy.optimize import curve_fit, least_squares
from scipy.signal import argrelmin
from scipy.interpolate import UnivariateSpline, interp1d, LSQUnivariateSpline
from scipy.io import readsav
from sklearn.decomposition import TruncatedSVD

from tqdm import tqdm

###########################################################
# Functions for Reading in Data
###########################################################

def readParams(file_name):
    """
    Given the file name of a check_point file,
    load in all relevant data into 1D vectors
    
    Returns vectors for line center in pixel (x),
    order (y), error in line center fit in pixels (e),
    and wavelength of line (w)
    """
    info = np.load(file_name,allow_pickle=True)[()]
    # Assemble information into "fit-able" form
    lines = [p[:,1] for p in info['params'] if p is not None]
    errs = [np.sqrt(cov[:,1,1]) for cov in info['cov'] if cov is not None]
    ordrs = [o for o in np.arange(86) if info['params'][o] is not None]
    waves = [w for w in info['wvln'] if w is not None]
    # I believe, but am not sure, that the wavelengths are multiplied by order
    # to separate them from when orders overlap at the edges
    waves = [wvln for order, wvln in zip(ordrs,waves)]
    ordrs = [np.ones_like(x) * m for m,x in zip(ordrs, lines)]

    x = np.concatenate(lines)
    y = np.concatenate(ordrs)
    e = np.concatenate(errs)
    w = np.concatenate(waves)
    # Note: default of pipeline includes ThAr lines, which we're not including here
    
    return (x,y,e,w)

def readThid(thid_file):
    # Extract information from a thid file
    # And then very _very_ painfully order values and remove duplicates
    thid = readsav(thid_file)['thid']
    pixl = np.array(thid.pixel[0])
    ordr = np.array(160-thid.order[0])
    wave = np.array(thid.wvac[0])
    
    sorted_pixl = []
    sorted_ordr = []
    sorted_wave = []
    for r in range(86):
        # Identify lines in the given order
        ord_mask = ordr==r
        sort_ordr = ordr[ord_mask]
        
        # Sort by wavelength along order
        ord_sort = np.argsort(pixl[ord_mask])
        sort_pixl = pixl[ord_mask][ord_sort]
        sort_wave = wave[ord_mask][ord_sort]
        
        # Remove duplicate pixel values
        sorted_ordr.append([sort_ordr[0]])
        sorted_pixl.append([sort_pixl[0]])
        sorted_wave.append([sort_wave[0]])
        duplo = np.logical_and(np.diff(sort_pixl)!=0, np.diff(sort_wave)!=0)
        
        # Append to array
        sorted_ordr.append(sort_ordr[1:][duplo])
        sorted_pixl.append(sort_pixl[1:][duplo])
        sorted_wave.append(sort_wave[1:][duplo])
    return np.concatenate(sorted_pixl), np.concatenate(sorted_ordr), np.concatenate(sorted_wave)

###########################################################
# Functions for Reading in Data
###########################################################

def lfcSet(file_list):
    """
    Bunch LFC file list into sets of consecutive observations
    Never Forget! Closeness in obsid is only _usually_ correlated
        with closeness in time.
    """
    # Just out of paranoia, order the LFC files
    pseudo_time = np.empty_like(file_list,dtype='float')
    file_date   = np.empty_like(file_list,dtype='float')
    file_obsn   = np.empty_like(file_list,dtype='float')
    for i,file_name in enumerate(file_list):
        file_id = os.path.basename(file_name).split('_')[-1][:-5]
        pseudo_time[i] = file_id
        file_date[i], file_obsn[i] = file_id.split('.')
    time_sort = np.argsort(pseudo_time)
    file_list = np.array(file_list)[time_sort]
    file_date = file_date[time_sort]
    file_obsn = file_obsn[time_sort]
    
    sets = []
    consecutive = []
    date = file_date[0]
    for i in range(1,len(file_list)):
        if date != file_date[i]:
            date = file_date[i]
            sets.append(consecutive)
            consecutive=[file_list[i]]
        elif file_obsn[i] != file_obsn[i-1]+1:
            sets.append(consecutive)
            consecutive=[file_list[i]]
        else:
            consecutive.append(file_list[i])
    return sets

###########################################################
# Polynomial Fitting Code
###########################################################

# Constructing a design matrix for polynomial fitting instead
def mkBlob(x, m, deg):
    """
    x: pixel
    m: order
    deg: degree of polynomial
    """
    # shift the data to center around the mean and have lower values
    xshift = np.mean(x)
    mshift = np.mean(m)
    xt = (x - xshift)
    mt = (m - mshift)
    scales = []
    for i in range(deg+1):
        for j in range(deg+1-i):
            vec = xt ** i * mt ** j
            # Scale the data so they cover about the same range of values
            scales.append(np.sqrt(vec.dot(vec)))
    # Values of shift and scale must be catalogue
    # in order to keep the fitted coefficients interpretable
    return (deg, xshift, mshift, scales)
            
def mkDesignMatrix(x, m, blob):
    """
    blob: output of mkBlob()
    BUG: DUPLICATED CODE WITH mkBlob()
    """
    deg, xshift, mshift, scales = blob
    xt = (x - xshift)
    mt = (m - mshift)
    matrix = []
    k = 0
    for i in range(deg+1):
        for j in range(deg+1-i):
            vec = xt ** i * mt ** j
            matrix.append(vec / scales[k])
            k += 1
    return np.array(matrix).T

def fit(data, M, weights):
    """
    return coefficients of the linear fit!
    """
    MTM = M.T.dot(weights[:,None] * M)
    print("fit(): condition number: {:.2e}".format(np.linalg.cond(MTM)))
    MTy = M.T.dot(weights * data)
    return np.linalg.solve(MTM, MTy)

def predict(newx, newm, blob, coeffs):
    """
    use coefficients to predict new wavelengths
    """
    Mnew = mkDesignMatrix(newx, newm, blob)
    return Mnew.dot(coeffs)

def poly_train_and_predict(newx, newm, x, m, data, weights, deg):
    blob = mkBlob(x, m, deg)
    M = mkDesignMatrix(x, m, blob)
    coeffs = fit(data, M, weights)
    return predict(newx, newm, blob, coeffs)

###########################################################
# Interpolation Code
###########################################################

def interp_train_and_predict(newx, newm, x, m, data, e=None, orders=range(86),nknot=70):
    prediction = np.zeros_like(newx)
    for r in orders:
        Inew = newm == r
        I = m==r
        if (np.sum(Inew)>0) and (np.sum(I)>0):
            assert np.all(np.diff(x[I]) > 0.),print(r)
            #prediction[Inew] = np.interp(newx[Inew], x[I], data[I],
            #                             left=np.nan,right=np.nan)
            #f = interp1d(x[I], data[I],kind='quadratic',bounds_error=False)
            s = UnivariateSpline(x[I],x[I],s=0)
            t = s.get_knots()[1:-1]
            #t = np.linspace(min(x[I])+1,max(x[I])-1,nknot)
            if e is not None:
                f = LSQUnivariateSpline(x[I],data[I],t,w=1/e[I]**2)
            else:
                f = LSQUnivariateSpline(x[I],data[I],t)
            prediction[Inew] = f(newx[Inew])
    return prediction

###########################################################
# PCA Noise
###########################################################

def pcaSetup(file_list, x_range=(500,7000), m_range=(45,75),
             allow_file_error=True, vet_pxls=True, vet_exps=True,
             verbose=False):
    # Construct wavelength "grids"
    x_range = np.arange(*x_range).astype(float)
    m_range = np.arange(*m_range).astype(float)
    x_grid, m_grid = np.meshgrid(x_range,m_range)
    x_grid = x_grid.flatten()
    m_grid = m_grid.flatten()
    
    # Load in all wavelength solutions
    w_fit_array = np.empty((len(file_list),len(x_grid)))
    if file_list[0].split('.')[-1] == 'thid':
        line_requirement = 0
        def readFunc(file_name):
            x,m,w = readThid(file_name)
            e = None
            return x,m,e,w
    else:
        line_requirement = 15000
        line_requirement = 0
        def readFunc(file_name):
             return readParams(file_name)
    
    used_files = []
    for i in range(len(file_list)):
        file_name = file_list[i]
        try:
            x,m,e,w = readFunc(file_name)
            if len(e) < line_requirement:
                # THIS LIMIT IS HARD CODED
                # WHICH IS DUMB
                # SHOULD BE SOMETHING LIKE LINES PER ORDER
                # ALSO ONLY WORKS ON LFCs
                if verbose:
                    print(f'File {file_name} has too few lines')
                w_fit_array[i,:] = np.nan
            else:
                w_fit_array[i] = interp_train_and_predict(x_grid,m_grid,x,m,w,e)
                used_files.append(os.path.basename(file_name))
        except ValueError as err:
            if not allow_file_error:
                raise err
            w_fit_array[i,:] = np.nan
    
    # Bad lines/exposure
    good = np.isfinite(w_fit_array)
    bad  = np.logical_not(good)
    if vet_exps:
        exp_okay = np.sum(good, axis=1) > 3
        w_fit_array = w_fit_array[exp_okay,:]
        if verbose:
            print(f"Not okay Exposures: {np.sum(~exp_okay)}")
            print(np.array(file_list)[~exp_okay])
        used_files = np.array(file_list)[exp_okay]
    if vet_pxls:
        pxl_okay = np.sum(good, axis=0) > 3
        w_fit_array = w_fit_array[:,pxl_okay]
        if verbose:
            print(f"Not okay Pixels: {np.sum(~pxl_okay)}")
        x_grid = x_grid[pxl_okay]
        m_grid = m_grid[pxl_okay]
    good = np.isfinite(w_fit_array)
    bad = np.logical_not(good)
    
    # Find mean wavelength pixel by pixel
    mean_w_fit = np.empty(w_fit_array.shape[1])
    for i in range(w_fit_array.shape[1]):
        mean_w_fit[i] = np.nanmean(w_fit_array[:,i])
    
    # Replace bad pixels with mean value
    # THIS IS TERRIBLE
    for i in range(w_fit_array.shape[0]):
        w_fit_array[i][bad[i]] = mean_w_fit[bad[i]]
    
    return w_fit_array, mean_w_fit, used_files

def eigenFun(w_fit_array, mean_w_fit, n_svd=5,verbose=False):
    # Find eigenvectors
    svd = TruncatedSVD(n_components=5,n_iter=7,random_state=42)
    uu = svd.fit_transform(w_fit_array - mean_w_fit[None, :])
    ss = svd.singular_values_
    vv = svd.components_
    
    return uu, ss, vv