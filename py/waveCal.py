# Collection of all wavelength related functions so far
import os
from glob import glob
import numpy as np
from numpy.polynomial.polynomial import polyvander2d, polyval2d
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.constants import c
from astropy.time import Time
from scipy import interpolate, optimize
from scipy.io import readsav
# BUG: SKLEARN NOT INSTALLED IN EXPRES_PIPELINE ENVIRONMENT
#from sklearn.decomposition import TruncatedSVD
import pickle

#import pcp
from tqdm import tqdm

# LFC Constants
rep_rate = 14e9 # magic
lfc_offset = 6.19e9 # magic


###########################################################
# Bandaids
###########################################################

# A function for finding all calibration files between specified dates
# There is a lot of tweaking to avoid LFCs where the LFC was left on for > ~3 minutes
def findFiles(start_date='000000', end_date='999999', cal_source='lfc'):
    line_files = []
    fits_files = []
    times = []
    
    if cal_source == 'thar':
        line_dir = '/expres/extracted/thidfiles/ThAr_*.thid'
    else:
        line_dir = '/expres/extracted/lfc_cal/lfc_id/LFC_*.npy'
        #line_dir = '/expres/extracted/lfc_cal/checkpoint/LFC_*.npy'
    
    for file_name in tqdm(glob(line_dir)):
        base = os.path.basename(file_name)
        if cal_source == 'thar':
            file_date = base[5:11]
        else:
            file_date = base[4:10]
        if (file_date >= start_date) and (file_date <= end_date):
            file_year = file_date[:2]
            if cal_source == 'thar':
                file_base = base[:-5]
                _, file_onum = file_base.split('.')
                if file_date=='191017':
                    if (int(file_onum) >= 1065) and (int(file_onum) < 1067):
                        # Engineering
                        continue
            else:
                file_base = base[:-4]
                _, file_onum = file_base.split('.')
                # Vet for consecutive and bad LFCs
                # Will eventually be replaced with a check for database status
                if file_date=='190828':
                    if int(file_onum) < 1219:
                        # consecutive LFCs
                        continue
                elif file_date=='190825':
                    if (int(file_onum) >= 1065) and (int(file_onum) < 1153):
                        # consecutive LFCs
                        continue
                elif file_date=='190824':
                    if (int(file_onum) < 1288):
                        # consecutive LFCs
                        continue
                elif file_date in ['190808', '190809']:
                    # CTI test days
                    continue
                elif file_date=='191007' and file_onum=='1117':
                    # bad, over-exposed file
                    continue
                elif file_date=='200124' and file_onum=='1170':
                    # bad, over-exposed file
                    continue
                elif file_date=='191223' and file_onum in ['1062','1090']:
                    # Not sure?
                    continue
                elif file_date=='200102' and file_onum in['1062','1063','1064','1069', '1098']:
                    # Not sure?
                    continue
            fits_file = f'/expres/extracted/fitspec/20{file_year}/{file_date}/{file_base}.fits'
            if os.path.isfile(fits_file): # original file exists
                try:
                    hdus = fits.open(fits_file)
                    times.append(Time(hdus[0].header['MIDPOINT'],format='isot').mjd)
                except KeyError:
                    hdus.close()
                    continue
                hdus.close()
                line_files.append(file_name)
                fits_files.append(fits_file)
    
    assert len(line_files)==len(fits_files)
    line_files = np.array(line_files)[np.argsort(times)]
    fits_files = np.array(fits_files)[np.argsort(times)]
    times = np.array(times)[np.argsort(times)]
    
    return line_files, fits_files, times

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
    try:
        info = np.load(file_name,allow_pickle=True)[()]
    except FileNotFoundError:
        if file_name.split('/')[-2] == 'checkpoint':
            lfc_id_dir = '/expres/extracted/lfc_cal/lfc_id/'
            file_name = lfc_id_dir + os.path.basename(file_name)
            info = np.load(file_name,allow_pickle=True)[()]
        else:
            raise FileNotFoundError
    # Assemble information into "fit-able" form
    num_orders = len(info['params'])
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
    
    return (x,y,w,e)

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

def readFile(file_name):
    """
    Pick appropriate reading function based on file name
    Normalize output (thid doesn't have errors)
    """
    if file_name.split('.')[-1] == 'thid':
        x,m,w = readThid(file_name)
        e = np.empty_like(x)
        e[:] = np.nan
        return x,m,w,e
    else:
        return readParams(file_name)

def sortFiles(file_list,get_mjd=False):
    # Find file extension length
    cut = len(os.path.basename(file_list[0]).split('.')[-1])+1
    # Sort files by date and file_num
    file_times = np.empty_like(file_list,dtype='float')
    for i in range(len(file_list)):
        file_times[i] = os.path.basename(file_list[i]).split('_')[-1][:-cut]
    file_list = np.array(file_list)[np.argsort(file_times)]

    if get_mjd: # Find MJD of each exposure from header
        for i, file_name in enumerate(file_list):
            hdus = fits.open(file_name)
            file_times[i] = Time(hdus[0].header['MIDPOINT'],format='isot').mjd
            hdus.close()

        return file_list, file_times

    else: # Just return sorted list of files
        return file_list

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
    #print("fit(): condition number: {:.2e}".format(np.linalg.cond(MTM)))
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

def interp_train_and_predict(newx, newm, x, m, data, e=None,
                             orders=range(86), interp_deg='pchip'):
    # Make all the searches match-able
    newm = np.array(newm,dtype=int)
    m = np.array(m,dtype=int)
    orders = np.array(orders,dtype=int)
    
    prediction = np.zeros_like(newx)
    prediction[:] = np.nan
    for r in orders:
        Inew = newm == r
        I = m==r
        if (np.sum(Inew)>0) and (np.sum(I)>0):
            wave_sort = np.argsort(data[I])
            ord_xs = x[I][wave_sort]
            ord_data = data[I][wave_sort]
            assert np.all(np.diff(ord_xs) > 0.),print(r,ord_xs[1:][np.argmin(np.diff(ord_xs))])
        
            # Interpolate
            if interp_deg==1:
                prediction[Inew] = np.interp(newx[Inew], ord_xs, ord_data,
                                             left=np.nan,right=np.nan,k=interp_deg)
            elif interp_deg == 'pchip':
                f = interpolate.PchipInterpolator(ord_xs,ord_data,extrapolate=False)
                prediction[Inew] = f(newx[Inew])
            elif interp_deg == 'inverse':
                f = interpolate.interp1d(ord_data, ord_xs, kind='cubic',
                                         bounds_error=False,fill_value=0)
                inv_f = lambda x, a: f(x)-a
                
                f0 = interpolate.UnivariateSpline(ord_xs,ord_data,ext=1)
                
                predict = np.zeros(np.sum(Inew),dtype=float)
                for i,pix in enumerate(newx[Inew]):
                    if (pix <= ord_xs.min()) or (pix >= ord_xs.max()): # No extrapolation
                        predict[i] = np.nan
                    else:
                        try:
                            x0 = f0(pix)
                            predict[i] = optimize.newton(inv_f,x0,args=(pix,))
                        except RuntimeError:
                            predict[i] = np.nan
                prediction[Inew] = predict
            else:
                tck = interpolate.splrep(ord_xs, ord_data, k=interp_deg)
                predict = interpolate.splev(newx[Inew],tck,ext=1)
                predict[predict==0] = np.nan
                prediction[Inew] = predict
    return prediction

###########################################################
# PCA Noise
###########################################################

def mkMeanWave(file_list, x_range=(500,7000), m_range=(45,75),
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
    used_files = []
    for i in range(len(file_list)):
        file_name = file_list[i]
        try:
            x,m,w,e = readFile(file_name)
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


###########################################################
# PCA Patching
###########################################################

def mode2wave(modes):
    """
    Return true wavelengths given LFC modes
    """
    # True Wavelengths
    freq = modes * rep_rate + lfc_offset  # true wavelength
    waves = c.value / freq * 1e10 # magic
    return np.array(waves)

def wave2mode(waves):
    """
    Return LFC mode number given wavelengths
    """
    freq = c.value * 1e10 / waves
    modes = np.round((freq - lfc_offset) / rep_rate)
    return np.array(modes).astype(int)

def buildLineDB(file_list, flatten=True, verbose=False, use_orders=None,
                hand_mask=True):
    """ 
    Find all observed modes in specified file list
    
    Eventually we'll record mode number while lines are being fitted
        and this will be simplified like a whole bunch
    
    hand_mask: if True, remove lines we've determined are bad
        (stop gap measure)
    """
    # Load in all observed modes into a big dictionary
    name_dict = {}
    wave_dict = {}
    order_list = []
    
    for file_name in tqdm(file_list):
        try:
            x,m,w,e = readFile(file_name)
            m = m.astype(int)
        except ValueError as err:
            if verbose:
                print(f'{os.path.basename(file_name)} threw error: {err}')
            continue
        orders = np.unique(m)
        for nord in orders:
            I = m==nord # Mask for an order
            n = [(nord,"{0:09.3f}".format(wave))for wave in w[I]]
            if nord not in name_dict.keys():
                name_dict[nord] = np.array(n,dtype=str)
                wave_dict[nord] = np.array(w[I],dtype=float)
                continue
            # Get identifying names: "(nord, wavelength string)"
            name_dict[nord], unq_indx = np.unique(np.concatenate([name_dict[nord],n]),
                                                  return_index=True,axis=0)
            wave_dict[nord] = np.concatenate([wave_dict[nord],w[I]])[unq_indx]
        order_list = np.unique(np.concatenate([order_list,orders])).astype(int)
    name_keys = []
    name_waves = []
    if use_orders is None:
        for nord in order_list:
            # Combine all added names and waves into one long list
            name_keys.append(name_dict[nord])
            name_waves.append(wave_dict[nord])
    else:
        for nord in use_orders:
            try:
                # Combine all added names and waves into one long list
                name_keys.append(name_dict[nord])
                name_waves.append(wave_dict[nord])
            except KeyError:
                continue
    name_keys = np.concatenate(name_keys)
    name_waves = np.concatenate(name_waves)
    
    if hand_mask:
        # Lines that are too close to be properly identified by thid.pro
        bad_orders = ['15','15','24','24','26','26','55','55','60','60','60']
        bad_wavels = [4231.05549,4231.05984, # 15
                      4482.07139,4482.08631, # 24
                      4545.13508,4545.14291, # 26
                      5841.11525,5841.10176, # 55
                      6116.61576,6116.61924,6140.35449] # 60
        for i in range(len(bad_orders)):
            bad_mask = np.logical_and(name_keys[:,0]==bad_orders[i],
                                      name_keys[:,1]=="{0:09.3f}".format(bad_wavels[i]))
            if np.sum(bad_mask)>0:
                mask = np.ones(len(name_keys),dtype=bool)
                bad_indx = np.where(bad_mask)[0][0]
                mask[bad_indx] = False
                name_keys  = name_keys[mask]
                name_waves = name_waves[mask]
    if flatten:
        # Separate out names and orders
        orders, names = name_keys.T
        return orders.astype(int), names, name_waves
    else:
        return name_keys, name_waves

def getLineMeasures(file_list, orders, names, err_cut=0):
    """
    Find line center (in pixels) to match order/mode lines
    """
    # Load in x values to match order/mode lines
    x_values = np.empty((len(file_list),len(orders)))
    x_values[:] = np.nan # want default empty to be nan
    x_errors = np.empty((len(file_list),len(orders)))
    x_errors[:] = np.nan
    
    pd_keys = pd.DataFrame({'orders':orders.copy().astype(int),
                            'names':names.copy().astype(str)})
    for file_num in tqdm(range(len(file_list))):
        # Load in line fit information
        file_name = file_list[file_num]
        try:
            x,m,w,e = readFile(file_name)
            m = m.astype(int)
            if err_cut > 0:
                mask = e < err_cut
                x = x[mask]
                m = m[mask]
                w = w[mask]
                e = e[mask]
        except ValueError:
            continue
        
        # Identify which lines this exposure has
        for nord in np.unique(m):
            I = m==nord # Mask for an order
            # Get identifying names: "(nord, wavelength string)"
            n = ["{0:09.3f}".format(wave) for wave in w[I]]
            xvl_dict = dict(zip(n,x[I]))
            err_dict = dict(zip(n,e[I]))
            ord_xval = np.array(pd_keys[pd_keys.orders==nord].names.map(xvl_dict))
            ord_errs = np.array(pd_keys[pd_keys.orders==nord].names.map(err_dict))
            x_values[file_num,pd_keys.orders==nord] = ord_xval
            x_errors[file_num,pd_keys.orders==nord] = ord_errs
                
    return x_values, x_errors

def pcaPatch(x_values, mask, K=2, num_iters=50,
             fast_pca=False, sparse_pca=False, return_iters=0):
    """
    Iterative PCA patching
    """
    K = int(K)
    if return_iters > 0:
        # Initialize arrays to store info from each iteration
        iter_x_values = np.zeros((int(num_iters//return_iters),*x_values.shape))
        if fast_pca:
            iter_vvs = np.zeros((int(num_iters//return_iters),K,x_values.shape[1]))
        else:
            iter_vvs = np.zeros((int(num_iters//return_iters),*x_values.shape))
    
    for i in tqdm(range(num_iters)):
        # There should be no more NaN values in x_values
        assert np.sum(np.isnan(x_values)) == 0
        # Redefine mean
        mean_x_values = np.mean(x_values,axis=0)
        
        # Run PCA
        """
        # NO OTHER PCA OPTIONS RIGHT NOW
        # PACKAGES NOT INSTALLED IN EXPRES_PIPELINE ENVIRONMENT
        if fast_pca:
            svd = TruncatedSVD(n_components=K, n_iter=7, random_state=42)
            uu = svd.fit_transform(x_values - mean_x_values)
            ss = svd.singular_values_
            vv = svd.components_
        elif sparse_pca:
            L, S, (uu, ss, vv) =  pcp.pcp(x_values-mean_x_values, lam=1.,
                                   delta=1e-6, mu=None, maxiter=500,
                                   verbose=False, missing_data=True,
                                   svd_method="exact")
        else:
            uu,ss,vv = np.linalg.svd(x_values-mean_x_values, full_matrices=False)
        """
        uu,ss,vv = np.linalg.svd(x_values-mean_x_values, full_matrices=False)

        # Repatch bad data with K PCA reconstruction
        denoised_xs = mean_x_values + np.dot((uu*ss)[:,:K],vv[:K])
        x_values[mask] = denoised_xs[mask]
        
        if return_iters > 0:
            if (i%return_iters)==0:
                # Populate array with information from this iteration
                iter_vvs[int(i//return_iters)] = vv.copy()
                iter_x_values[int(i//return_iters)] = x_values.copy()
    if return_iters:
        return x_values, mean_x_values, denoised_xs, uu, ss, vv, iter_vvs, iter_x_values
    else:
        return x_values, mean_x_values, denoised_xs, uu, ss, vv

def patchAndDenoise(file_list, file_times=None, K=2, use_orders=None,
             num_iters=50, return_iters=0, running_window=0,
             line_cutoff=0.5, file_cutoff=0.5, outlier_cut=0, err_cut=0,
             fast_pca=False,sparse_pca=False,verbose=False):
    """
    Vet for bad lines/exposures
    Initial patch of bad data with running mean
    Iterative patch with PCA until "convergence"
    (convergence limit hard coded but maybe doesn't need to be?)
    
    file_times : mjd time each file was observed (for interpolation later)
    line_cutoff: required fraction of files to have a line
    file_cutoff: required fraction of lines a file must have
    outlier_cut: for finding line centers that are off, 0 then don't do anything
    err_cut: don't use lines with erros over this value
    
    plot: NOT YET IMPLEMENTED
    Idea is to make plots of what lines are missing
    (Scatter plots would take very long time, though)
    """
    if file_times is None:
        file_times = np.zeros_like(file_list)
    
    ### Gather calibration information
    # Find all observed lines in each order and their wavlengths
    if verbose:
        print('Finding all observed modes')
    orders, names, waves = buildLineDB(file_list, use_orders=use_orders)

    # Find x-values of observed lines
    if verbose:
        print('Finding line center for each mode')
    x_values, x_errors = getLineMeasures(file_list, orders, names, err_cut=0)
    
    
    ### Vetting
    # Find where there is no line information
    x_values[x_values < 1] = np.nan # This will throw a warning
    
    # Mask out of order lines
    out_of_order = np.zeros_like(x_values,dtype=bool)
    for m in np.unique(orders):
        I = orders==m
        wave_sort = np.argsort(waves[I])
        for i, exp in enumerate(x_values):
            exp_sort = exp[I][wave_sort]
            exp_diff = np.diff(exp_sort)
            left_diff = np.insert(exp_diff<0,0,False)
            right_diff = np.append(exp_diff<0,False)
            exp_mask = np.logical_or(left_diff,right_diff)
            out_of_order[i,I] = exp_mask.copy()
    x_values[out_of_order] = np.nan
    if verbose:
        num_bad = np.sum(out_of_order)
        num_total = out_of_order.size
        print('{:.3}% of lines masked'.format(
             (num_bad)/num_total*100))
            
    # Get rid of bad lines
    good_lines = np.mean(np.isnan(x_values),axis=0) < line_cutoff
    # Trim everything
    names  = names[good_lines]
    orders = orders[good_lines]
    waves  = waves[good_lines]
    x_values = x_values[:,good_lines]
    x_errors = x_errors[:,good_lines]
    if verbose:
        num_good = np.sum(good_lines)
        num_total = good_lines.size
        print('{} of {} lines cut ({:.3}%)'.format(
            (num_total - num_good),num_total,
            (num_total - num_good)/num_total*100))
    
    # Get rid of bad files
    good_files = np.mean(np.isnan(x_values),axis=1) < file_cutoff
    # Trim everything
    x_values = x_values[good_files]
    x_errors = x_errors[good_files]
    exp_list = file_list[good_files]
    file_times = file_times[good_files]
    if verbose:
        num_good = np.sum(good_files)
        num_total = good_files.size
        print('{} of {} files cut ({:.3}%)'.format(
            (num_total - num_good),num_total,
            (num_total - num_good)/num_total*100))
        print('Files that were cut:')
        print(file_list[~good_files])
    
    ### Patching
    # Initial patch of bad data with mean
    bad_mask = np.isnan(x_values) # mask to identify patched x_values
    if running_window > 0:
        half_size = int(running_window//2)
        for i in range(x_values.shape[0]):
            # Identify files in window
            file_range = [max((i-half_size,0)), min((i+half_size+1,x_values.shape[1]))]
            # Find mean of non-NaN values
            run_med = np.nanmean(x_values[file_range[0]:file_range[1],:],axis=0)
            # Patch NaN values with mean for center file
            x_values[i][bad_mask[i,:]] = run_med[bad_mask[i,:]]
        counter = 5
        while np.sum(np.isnan(x_values)) > 0:
            for i in range(x_values.shape[0]):
                # Identify files in window
                file_range = [max((i-half_size,0)), min((i+half_size+1,x_values.shape[1]))]
                # Find mean of non-NaN values
                run_med = np.nanmean(x_values[file_range[0]:file_range[1],:],axis=0)
                # Patch NaN values with mean for center file
                x_values[i][bad_mask[i,:]] = run_med[bad_mask[i,:]]
            counter -= 1
            if counter < 0:
                print("Persistant NaNs with running mean.")
                print("Replacing remaining NaNs with global mean.")
                tot_mean = np.nanmean(x_values,axis=0)[None,...]*np.ones_like(x_values)
                x_values[np.isnan(x_values)] = tot_mean[np.isnan(x_values)]
                break
    else: # don't bother with running mean
        mean_values = np.nanmean(x_values,axis=0)
        mean_patch = np.array([mean_values for _ in range(x_values.shape[0])])
        x_values[bad_mask] = mean_patch[bad_mask]
    
    # Iterative PCA
    pca_results = pcaPatch(x_values, bad_mask, K=K, num_iters=num_iters,
                           fast_pca=fast_pca, sparse_pca=sparse_pca, return_iters=return_iters)
    x_values, mean_x_values, denoised_xs, uu, ss, vv = pca_results[:6]
    
    # Mask line center outliers
    if outlier_cut > 0:
        x_resids  = x_values-denoised_xs
        out_mask  = abs(x_resids-np.mean(x_resids)) > (outlier_cut*np.nanstd(x_resids))
        if verbose:
            num_out = np.sum(out_mask)
            num_total = out_mask.size
            num_bad = np.sum(np.logical_and(out_mask,bad_mask))
            print('{:.3}% of lines marked as Outliers'.format(
                 (num_out)/num_total*100))
            print('{:.3}% of lines marked as Outliers that were PCA Patched'.format(
                 (num_bad)/num_total*100))
        pca_results = pcaPatch(x_values, np.logical_or(bad_mask,out_mask),
                               K=K, num_iters=num_iters,
                               fast_pca=fast_pca, return_iters=return_iters)
        x_values, mean_x_values, denoised_xs, uu, ss, vv = pca_results[:6]
    
    
    patch_dict = {}
    patch_dict['K'] = K
    # Exposure Information
    patch_dict['files']  = exp_list.copy()
    patch_dict['times']  = file_times.copy()
    # Line Information
    patch_dict['names']  = names.copy()
    patch_dict['orders'] = orders.copy()
    patch_dict['waves']  = waves.copy()
    patch_dict['errors'] = None # Is there such a thing?
    # Line Measurement Information
    patch_dict['x_values'] = x_values.copy()
    patch_dict['x_errors'] = x_errors.copy()
    patch_dict['denoised_xs'] = denoised_xs.copy()
    patch_dict['mean_xs']  = mean_x_values.copy()
    patch_dict['bad_mask'] = bad_mask.copy()
    # PCA Information
    patch_dict['u'] = uu.copy()
    patch_dict['s'] = ss.copy()
    patch_dict['v'] = vv.copy()
    patch_dict['ec'] = (uu*ss)[:,:K]
    # Information by Iteration
    if return_iters > 0:
        patch_dict['iter_vs'] = pca_results[6].copy()
        patch_dict['iter_x_values'] = pca_results[7].copy()
    # Outlier Information
    if outlier_cut > 0:
        patch_dict['out_mask'] = out_mask.copy()
    
    return patch_dict

def evalWaveSol(new_interps, patch_dict, intp_deg=1, interp_key='times'):
    """
    Interpolate eigen coefficients against time
    """
    try:
        len(new_interps)
    except TypeError:
        new_interps = [new_interps]
    K  = patch_dict['K']
    vv = patch_dict['v']
    # Interpolate eigen coefficients
    new_ecs = np.empty((len(new_interps),K),dtype=float)
    for i in range(K):
        # Interpolating one by one seems right, right?
        if intp_deg==0:
            # Find nearest time for each time
            # IS THERE A WAY TO DO THIS NOT ONE BY ONE???
            for i_idx, i in enumerate(new_interps):
                idx = np.abs(patch_dict[interp_key]-i).argmin()
                new_ecs[i_idx,i] = patch_dict['ec'][idx,i]
            
        elif intp_deg==1: # Default
            f = interpolate.interp1d(patch_dict[interp_key],patch_dict['ec'][:,i],kind='linear',
                                 bounds_error=False,fill_value=np.nan)
            new_ecs[:,i] = f(new_interps)
            
        elif intp_deg==3:
            f = interpolate.interp1d(patch_dict[interp_key],patch_dict['ec'][:,i],kind='cubic',
                                 bounds_error=False,fill_value=np.nan)
            new_ecs[:,i] = f(new_interps)
            
        else:
            tck = interpolate.splrep(patch_dict[interp_key],patch_dict['ec'][:,i],k=interp_deg)
            new_ecs[:,i] = interpolate.splev(new_interps,tck)
            
    # Construct x values for that period of time
    denoised_xs = np.dot(new_ecs,vv[:K]) + patch_dict['mean_xs']
    
    return denoised_xs


###########################################################
# PCA Patching Tests
###########################################################

def findBestK(file_list, save_file, file_times=None, k_values=[2],
              use_orders=None, num_iters=50, running_window=9,
              line_cutoff=0.5, file_cutoff=0.5, outlier_cut=3, err_cut=0,
              fast_pca=False,sparse_pca=False,verbose=False):
    """
    Vet for bad lines/exposures
    Initial patch of bad data with running mean
    Iterative patch with PCA until "convergence"
    (convergence limit hard coded but maybe doesn't need to be?)
    
    file_times : mjd time each file was observed (for interpolation later)
    k_values: list of k_values to test
    line_cutoff: required fraction of files to have a line
    file_cutoff: required fraction of lines a file must have
    outlier_cut: for finding line centers that are off, 0 then don't do anything
    err_cut: don't use lines with erros over this value
    
    plot: NOT YET IMPLEMENTED
    Idea is to make plots of what lines are missing
    (Scatter plots would take very long time, though)
    """
    if file_times is None:
        file_times = np.zeros_like(file_list)
    k_values = np.array(k_values).astype(int)
    
    ### Gather calibration information
    # Find all observed lines in each order and their wavlengths
    if verbose:
        print('Finding all observed modes')
    orders, names, waves = buildLineDB(file_list, use_orders=use_orders)

    # Find x-values of observed lines
    if verbose:
        print('Finding line center for each mode')
    x_values, x_errors = getLineMeasures(file_list, orders, names, err_cut=0)
    
    
    ### Vetting
    # Find where there is no line information
    x_values[x_values < 1] = np.nan # This will throw a warning
    
    # Mask out of order lines
    out_of_order = np.zeros_like(x_values,dtype=bool)
    for m in np.unique(orders):
        I = orders==m
        wave_sort = np.argsort(waves[I])
        for i, exp in enumerate(x_values):
            exp_sort = exp[I][wave_sort]
            exp_diff = np.diff(exp_sort)
            left_diff = np.insert(exp_diff<0,0,False)
            right_diff = np.append(exp_diff<0,False)
            exp_mask = np.logical_or(left_diff,right_diff)
            out_of_order[i,I] = exp_mask.copy()
    x_values[out_of_order] = np.nan
    if verbose:
        num_bad = np.sum(out_of_order)
        num_total = out_of_order.size
        print('{:.3}% of lines masked'.format(
             (num_bad)/num_total*100))
            
    # Get rid of bad lines
    good_lines = np.mean(np.isnan(x_values),axis=0) < line_cutoff
    # Trim everything
    names  = names[good_lines]
    orders = orders[good_lines]
    waves  = waves[good_lines]
    x_values = x_values[:,good_lines]
    x_errors = x_errors[:,good_lines]
    if verbose:
        num_good = np.sum(good_lines)
        num_total = good_lines.size
        print('{} of {} lines cut ({:.3}%)'.format(
            (num_total - num_good),num_total,
            (num_total - num_good)/num_total*100))
    
    # Get rid of bad files
    good_files = np.mean(np.isnan(x_values),axis=1) < file_cutoff
    # Trim everything
    x_values = x_values[good_files]
    x_errors = x_errors[good_files]
    exp_list = file_list[good_files]
    file_times = file_times[good_files]
    if verbose:
        num_good = np.sum(good_files)
        num_total = good_files.size
        print('{} of {} files cut ({:.3}%)'.format(
            (num_total - num_good),num_total,
            (num_total - num_good)/num_total*100))
        print('Files that were cut:')
        print(file_list[~good_files])
    
    ### Patching
    # Initial patch of bad data with mean
    bad_mask = np.isnan(x_values) # mask to identify patched x_values
    if running_window > 0:
        half_size = int(running_window//2)
        for i in range(x_values.shape[0]):
            # Identify files in window
            file_range = [max((i-half_size,0)), min((i+half_size+1,x_values.shape[1]))]
            # Find mean of non-NaN values
            run_med = np.nanmean(x_values[file_range[0]:file_range[1],:],axis=0)
            # Patch NaN values with mean for center file
            x_values[i][bad_mask[i,:]] = run_med[bad_mask[i,:]]
        counter = 5
        while np.sum(np.isnan(x_values)) > 0:
            for i in range(x_values.shape[0]):
                # Identify files in window
                file_range = [max((i-half_size,0)), min((i+half_size+1,x_values.shape[1]))]
                # Find mean of non-NaN values
                run_med = np.nanmean(x_values[file_range[0]:file_range[1],:],axis=0)
                # Patch NaN values with mean for center file
                x_values[i][bad_mask[i,:]] = run_med[bad_mask[i,:]]
            counter -= 1
            if counter < 0:
                print("Persistant NaNs with running mean.")
                print("Replacing remaining NaNs with global mean.")
                tot_mean = np.nanmean(x_values,axis=0)[None,...]*np.ones_like(x_values)
                x_values[np.isnan(x_values)] = tot_mean[np.isnan(x_values)]
                break
    else: # don't bother with running mean
        mean_values = np.nanmean(x_values,axis=0)
        mean_patch = np.array([mean_values for _ in range(x_values.shape[0])])
        x_values[bad_mask] = mean_patch[bad_mask]
    
    # Load in information independent of K value
    patch_dict = {}
    # Exposure Information
    patch_dict['files']  = exp_list.copy()
    patch_dict['times']  = file_times.copy()
    # Line Information
    patch_dict['names']  = names.copy()
    patch_dict['orders'] = orders.copy()
    patch_dict['waves']  = waves.copy()
    patch_dict['errors'] = None # Is there such a thing?
    # File information for K tests
    patch_dict['K_tests'] = k_values
    patch_dict['K_files'] = []
    
    # Run PCA with different K values
    x_values0 = x_values.copy()
    for K in k_values:
        patch_dict[f'K{K}'] = save_file+f'_K{K}.pkl'
        # Iterative PCA
        pca_results = pcaPatch(x_values0, bad_mask, K=K, num_iters=num_iters,
                               fast_pca=fast_pca, sparse_pca=sparse_pca)
        x_values, mean_x_values, denoised_xs, uu, ss, vv = pca_results[:6]

        # Mask line center outliers
        if outlier_cut > 0:
            x_resids  = x_values-denoised_xs
            out_mask  = abs(x_resids-np.mean(x_resids)) > (outlier_cut*np.nanstd(x_resids))
            if verbose:
                num_out = np.sum(out_mask)
                num_total = out_mask.size
                num_bad = np.sum(np.logical_and(out_mask,bad_mask))
                print('{:.3}% of lines marked as Outliers'.format(
                     (num_out)/num_total*100))
                print('{:.3}% of lines marked as Outliers that were PCA Patched'.format(
                     (num_bad)/num_total*100))
            pca_results = pcaPatch(x_values, np.logical_or(bad_mask,out_mask),
                                   K=K, num_iters=num_iters,
                                   fast_pca=fast_pca)
            x_values, mean_x_values, denoised_xs, uu, ss, vv = pca_results[:6]
        
        kval_dict = {}
        kval_dict['K'] = K
        # Line Measurement Information
        kval_dict['x_values'] = x_values.copy()
        kval_dict['x_errors'] = x_errors.copy()
        kval_dict['denoised_xs'] = denoised_xs.copy()
        kval_dict['mean_xs']  = mean_x_values.copy()
        kval_dict['bad_mask'] = bad_mask.copy()
        # PCA Information
        kval_dict['u'] = uu.copy()
        kval_dict['s'] = ss.copy()
        kval_dict['v'] = vv.copy()
        kval_dict['ec'] = (uu*ss)[:,:K]
        # Outlier Information
        if outlier_cut > 0:
            kval_dict['out_mask'] = out_mask.copy()
        
        pickle.dump(kval_dict,open(save_file+f'_K{K}.pkl','wb'))
        patch_dict['K_files'].append(save_file+f'_K{K}.pkl')
    
    return patch_dict