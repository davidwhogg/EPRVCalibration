# Collection of all wavelength related functions so far
import os
from glob import glob
import numpy as np
from numpy.polynomial.polynomial import polyvander2d, polyval2d
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.constants import c
from astropy.time import Time
from scipy.optimize import curve_fit, least_squares
from scipy.signal import argrelmin
from scipy.interpolate import UnivariateSpline, interp1d, LSQUnivariateSpline, splrep, splev
from scipy.io import readsav
from sklearn.decomposition import TruncatedSVD

from tqdm import tqdm

# LFC Constants
rep_rate = 14e9 # magic
lfc_offset = 6.19e9 # magic

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

def readFile(file_name):
    """
    Pick appropriate reading function based on file name
    Normalize output (thid doesn't have errors)
    """
    if file_name.split('.')[-1] == 'thid':
        x,m,w = readThid(file_name)
        e = None
        return x,m,e,w
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

def interp_train_and_predict(newx, newm, x, m, data, e=None,
                             orders=range(86), nknot=70):
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
            x,m,e,w = readFile(file_name)
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

def findModes(file_list, order_list=range(42,76),
              flatten=False, verbose=False):
    """ 
    Find all observed modes in specified file list
    
    Eventually we'll record mode number while lines are being fitted
        and this will be simplified like a whole bunch
    """
    # Load in all observed modes into a big dictionary
    mode_dict = {}
    
    for file_name in file_list:
        try:
            x,m,e,w = readFile(file_name)
        except ValueError as err:
            if verbose:
                print(f'{os.path.basename(file_name)} threw error: {err}')
            continue
        for nord in order_list:
            n = wave2mode(w[m==nord])
            if nord not in mode_dict.keys():
                mode_dict[nord] = np.array([]).astype(int)
            mode_dict[nord] = np.unique(np.concatenate([mode_dict[nord],n]))
    
    if flatten:
        # Reformat mode dictionary into a flat vector
        modes = np.array([]).astype(int)
        orders = np.array([]).astype(int)
        for m in mode_dict.keys():
            modes = np.concatenate((modes, mode_dict[m]))
            orders = np.concatenate((orders, (np.zeros_like(mode_dict[m])+m)))
            
        return modes, orders
    
    else:
        return mode_dict

def mode2pixl(file_list, modes, orders, waves,
              use_mode_number=True):
    """
    Find line center (in pixels) to match order/mode lines
    """
    # Load in x values to match order/mode lines
    x_values = np.empty((len(file_list),len(modes)))
    x_values[:] = np.nan # want default empty to be nan
    
    for i in tqdm(range(len(file_list))):
        file_name = file_list[i]
        try:
            x,m,e,w = readFile(file_name)
        except ValueError:
            continue
        for line in range(len(modes)):
            I = m==orders[line] # Identify all lines in given order
            try:
                if use_mode_number: # We have LFC mode numbers to identify lines
                    ord_modes = wave2mode(w[I])
                    if modes[line] in ord_modes:
                        x_values[i,line] = x[I][ord_modes==modes[line]]
                else: # Identify lines by shared wavelength
                    if waves[line] in w[I]:
                        x_values[i,line] = x[I][w[I]==waves[line]] # hogg hates this line
            except ValueError:
                # Happens when a peak is mistakenly identified between LFC lines
                # Means two peaks profess to have the same mode number
                # Ignore since this will have to be fixed in line fitting stage
                x_values[i,line] = np.nan
    
    return x_values

def pcaPatch(file_list, file_times=None, order_range=range(45,76), K=2,
             running_window=9, num_iters=45, return_iters=False,
             line_cutoff=0.5, file_cutoff=0.5, 
             fast_pca=False, plot=False, verbose=False):
    """
    Vet for bad lines/exposures
    Initial patch of bad data with running mean
    Iterative patch with PCA until "convergence"
    (convergence limit hard coded but maybe doesn't need to be?)
    
    file_times : mjd time each file was observed (for interpolation later)
    line_cutoff: required fraction of files to have a line
    file_cutoff: required fraction of lines a file must have
    
    plot: NOT YET IMPLEMENTED
    Idea is to make plots of what lines are missing
    (Scatter plots would take very long time, though)
    """
    # Find all observed modes
    if verbose:
        print('Finding all observed modes')
    modes, orders = findModes(file_list, order_list=order_range, flatten=True)

    # True Wavelengths
    waves = mode2wave(modes)

    # Find x-values of observed modes
    if verbose:
        print('Finding line center for each mode')
    x_values = mode2pixl(file_list, modes, orders, waves)
    if plot:
        init_x_values = x_values.copy() # Save pre-vet for plotting
    
    # Vetting
    x_values[x_values < 1] = np.nan # Where there is no line information, this will thow a warning
    # Get rid of bad lines
    good_lines = np.mean(np.isnan(x_values),axis=0) < line_cutoff
    # Trim everything
    modes  = modes[good_lines]
    orders = orders[good_lines]
    waves  = waves[good_lines]
    x_values = x_values[:,good_lines]
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
    
    if plot:
        """ # I'm just not sure if any of this works yet
        # But I am sure it's just not a priority
        date = Time.now().isot.split('T')[0].replace('-','')[2:] # YYMMDD
        for nord in order_range:
            init_ord_mask = init_orders==nord
            ord_mask = orders==nord

            plt.figure()
            plt.title(f'Order {nord}')
            plt.xlabel('Mode Number')
            plt.ylabel('Exposure Number-ish');
            for f in range(len(ckpt_files)):
                plt.plot(init_x_values[f][init_ord_mask],
                         np.zeros_like(init_ord_mask,dtype=int)+f,'o',label='All Lines')
                plt.plot(init_x_values[f][init_ord_mask][~good_lines[init_ord_mask]],
                         np.zeros(np.sum(~good_lines),dtype=int)+f,'.',label='Vetted Lines')
                plt.plot(init_x_values[f][init_ord_mask][good_lines][~good_files[init_ord_mask[good_lines]]],
                         np.zeros(np.sum(~good_files),dtype=int)+f,'.',label='Vetted Files')
                nan_mask = np.isnan(x_values[ord_mask])
                plt.plot(x_values[ord_mask][nan_mask],
                         np.zeros_like(nan_mask,dtype=int)+f,'r.',label='No Info.')
                if f==0:
                    plt.legend(loc=1)
            plt.tight_layout()
            plt.savefig(f'./Figures/{date}_ord{nord}Lines.png')
            plt.close()
        """
    
    # Initial patch of bad data with running mean
    bad_mask = np.isnan(x_values)
    half_size = int(running_window//2)
    for i in range(x_values.shape[0]):
        # Identify files in window
        file_range = [max((i-half_size,0)), min((i+half_size+1,x_values.shape[1]))]
        # Find mean of non-NaN values
        run_med = np.nanmean(x_values[file_range[0]:file_range[1],:],axis=0)
        # Patch NaN values with mean for center file
        x_values[i][bad_mask[i,:]] = run_med[bad_mask[i,:]]
    
    # Iterative PCA
    if return_iters:
        # Initialize arrays to store info from each iteration
        iter_x_values = np.zeros((num_iters,*x_values.shape))
        if fast_pca:
            iter_vvs = np.zeros((num_iters,K,x_values.shape[1]))
        else:
            iter_vvs = np.zeros((num_iters,*x_values.shape))
    
    # Dimension of PCA reconstruction
    K = int(K)
    
    for i in tqdm(range(num_iters)):
        assert np.sum(np.isnan(x_values)) == 0
        # Redefine mean
        mean_x_values = np.mean(x_values,axis=0)
        
        # Run PCA
        if fast_pca:
            svd = TruncatedSVD(n_components=K, n_iter=7, random_state=42)
            uu = svd.fit_transform(x_values - mean_x_values)
            ss = svd.singular_values_
            vv = svd.components_
        else:
            uu,ss,vv = np.linalg.svd(x_values-mean_x_values, full_matrices=False)

        # Repatch bad data with K PCA reconstruction
        pca_patch = np.dot((uu*ss)[:,:K],vv[:K])
        x_values[bad_mask] = (pca_patch+mean_x_values)[bad_mask]
        
        if return_iters:
            # Populate array with information from this iteration
            iter_vvs[i] = vv.copy()
            iter_x_values[i] = x_values.copy()
    
    patch_dict = {}
    patch_dict['K'] = K
    patch_dict['files']  = exp_list.copy()
    if file_times is not None:
        patch_dict['times']  = file_times.copy()
    patch_dict['modes']  = modes.copy()
    patch_dict['orders'] = orders.copy()
    patch_dict['waves']  = waves.copy()
    #patch_dict['errors'] ? Is there such a thing?
    patch_dict['x_values'] = x_values.copy()
    patch_dict['mean_x_values']  = mean_x_values.copy()
    patch_dict['bad_mask'] = bad_mask.copy()
    patch_dict['u'] = uu.copy()
    patch_dict['s'] = ss.copy()
    patch_dict['v'] = vv.copy()
    patch_dict['ec'] = (uu.dot(np.diag(ss)))
    if return_iters:
        patch_dict['iter_vs'] = iter_vvs
        patch_dict['iter_x_values'] = iter_x_values
    
    return patch_dict

def interp_coefs_and_predict(new_time, patch_dict, interp_deg=1,
                             new_x=None, new_m=None,
                             x_range=(500,7000), m_range=(45,75)):
    """
    Interpolate eigen coefficients against time
    """
    K  = patch_dict['K']
    vv = patch_dict['v']
    # Interpolate eigen coefficients
    new_ec = np.empty(K,dtype=float)
    for i in range(K):
        # Interpolating one by one seems right, right?
        if interp_deg==0:
            # Find nearest time
            idx = np.abs(patch_dict['times']-new_time).argmin()
            new_ec[i] = patch_dict['ec'][idx,i]
        elif interp_deg==1:
            new_ec[i] = np.interp(new_time,patch_dict['times'],patch_dict['ec'][:,i])
        else:
            tck = splrep(patch_dict['times'],patch_dict['ec'][:,i],k=interp_deg)
            new_ec[i] = splev(new_time,tck)
            
    # Construct x values for that period of time
    x = np.dot(new_ec,vv[:K]) + patch_dict['mean_x_values']
    m = patch_dict['orders']
    w = patch_dict['waves']
    
    # Construct wavelength "grids"
    if new_x is None or new_m is None: # use specified x ranges
        x_range = np.arange(*x_range).astype(float)
        m_range = np.arange(*m_range).astype(float)
        x_grid, m_grid = np.meshgrid(x_range,m_range)
        new_x = x_grid.flatten()
        new_m = m_grid.flatten()
    else: # use specific values provided
        new_x = np.sort(new_x)
        new_m = np.sort(new_m)
    
    if new_x[0] < min(x):
        print('WARNING: Interpolation range in pixel direction lower than training set.')
    if new_x[-1] > max(x):
        print('WARNING: Interpolation range in pixel direction higher than training set.')
    if new_m[0] < min(m):
        print('WARNING: Interpolation range in order direction lower than training set.')
    if new_m[-1] > max(m):
        print('WARNING: Interpolation range in order direction higher than training set.')
    
    w_fit = interp_train_and_predict(new_x,new_m,x,m,w)

    return w_fit