import os

import numpy as np
import WaveSpace.Utils.HelperFuns as hf
import WaveSpace.Utils.WaveData as wd

def find_wave_activity(waveData, dataBucketName=None, dataInd = None, nBases=3, use_bases=None):
    """Identifies dominant traveling wave patterns (spatial bases) in complex-valued waveData
    via singular value decomposition on the data covariance. Converts complex data to normalized (unit-length) phase representation.

    Inputs:
    waveData : WaveData object        
    dataBucketName : str, optional
        which data bucket to use
    dataInd : tuple or None, optional
        Optional index tuple to select a subset of the data (e.g., (slice(0,1), slice(10,12), ...)). 
        Needs to result in the same number of dimensions as full data. So use slices for singular dimensions. 
        If None, uses all data.
    nBases : int, optional
        The number of spatial bases to extract (default is 3).

    Adds data buckets to waveData:
    Bases : Complex spatial bases (channels x bases).
    Fit :   fit per trial and time ((freq) trials x time).
    betas : weights ((freq) trials x time x bases).

    References
    ----------
    - https://github.com/ScaleSymmetry/Traveling-wave-analysis https://doi.org/10.1371/journal.pone.0148413
          https://doi.org/10.1371/journal.pcbi.1007316"""
    
    #sanity checks:
    if  dataBucketName == "":
        dataBucketName =  waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)

    hf.assure_consistency(waveData)
    complexData = waveData.get_data(dataBucketName)
    if dataInd:
        complexData = complexData[dataInd]
    origDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    origShape = complexData.shape
    desiredDimord = "trl_chan_time"
    hasBeenReshaped, complexData =  hf.force_dimord(complexData, origDimord , desiredDimord)
    
    if dataBucketName == "complexData":
        # Make complex valued Phase/magnitude Timeseries per freq
        complexData = np.exp(1j*np.angle(complexData))
        #reshape to (trial, time, channel)
        phi= np.transpose(complexData, (0, 2, 1))
        bases, fit, betas, model_cts, variance_explained = c_TW_bases_betas(phi,nBases=nBases, use_bases=use_bases)
        
    elif dataBucketName == "UnwrappedPhase":
        #reshape to (trial, time, channel)
        uphase= np.transpose(complexData, (0, 2, 1))
        bases, fit, betas, model_cts, variance_explained = R_TW_bases_betas(uphase,nBases=nBases, use_bases=use_bases)
        
    chan_names = waveData.get_channel_names()
    if hasBeenReshaped:
        splitDimensions = origDimord.split("_")
        if "chan" in splitDimensions:
            nGroupDimensions = splitDimensions.index("chan")
            channelShape = (origShape[nGroupDimensions],) #origShape[nGroupDimensions] # modified by Yue
        elif "posx" in splitDimensions:
            nGroupDimensions = splitDimensions.index("posx")
            channelShape = origShape[nGroupDimensions:nGroupDimensions+2]

        groupDimensions = splitDimensions[0:nGroupDimensions]
        groupDimSizes = origShape[:len(groupDimensions)]
        multi_indices  = np.array(np.unravel_index(np.arange(complexData.shape[0]), groupDimSizes)).T
        
        bases = np.reshape(bases, (*channelShape, bases.shape[-1]))
        
        # added by Yue
        if "chan" in splitDimensions:
            dimord = "chan_base"
        elif "posx" in splitDimensions:
            dimord = "posx_posy_base"
        
        basesBucket = wd.DataBucket(bases,"Bases", dimord, chanNames=chan_names) # fixed hardcode by Yue
        time = waveData.DataBuckets[dataBucketName].get_time()
        fit = np.reshape(fit,(*groupDimSizes, fit.shape[-1]))
        fitBucket = wd.DataBucket(fit,"Fit", ("_").join(groupDimensions) +"_time", time=time ,chanNames=chan_names)
        betas = np.reshape(betas,(*groupDimSizes, betas.shape[-2], betas.shape[-1]))
        betasBucket = wd.DataBucket(betas,"betas",("_").join(groupDimensions) +"_time_basis",time=time ,chanNames= chan_names)
        model_cts = np.reshape(model_cts, (*groupDimSizes, model_cts.shape[-2], model_cts.shape[-1]))
        modelBucket = wd.DataBucket(model_cts,"model",("_").join(groupDimensions) + "_time_chan",time=time,chanNames=chan_names)
    else:
        basesBucket = wd.DataBucket(bases,"Bases","chan_base", chanNames= chan_names)
        time = waveData.DataBuckets[dataBucketName].get_time()
        fitBucket = wd.DataBucket(fit,"Fit","trl_time", time=time ,chanNames= chan_names)
        betasBucket = wd.DataBucket(betas,"betas","trl_time_basis",time=time ,chanNames= chan_names)
        modelBucket = wd.DataBucket(model_cts,"model","trl_time_chan",time=time ,chanNames= chan_names)
        
    waveData.add_data_bucket(basesBucket)
    waveData.add_data_bucket(fitBucket)
    waveData.add_data_bucket(betasBucket)
    waveData.add_data_bucket(modelBucket)
    
    return variance_explained

def c_TW_bases_betas(phi_cts,nBases=3, use_bases=None):
    #phi complex-valued phase, c trials, t times, s sensors
    phi_Cs = np.asarray(phi_cts.reshape(-1,phi_cts.shape[-1]))
    phi_cent = phi_Cs - phi_Cs.mean(0)
    if use_bases is None:
        COV = phi_cent.T.conj()@phi_cent
        u,s,vh = np.linalg.svd(COV)
        variance_explained = 100.0*s[:nBases]/s.sum() # Yue added this
        bases_sb = vh[:nBases].T
    else:
        #variance_explained = None # Yue added this
        bases_sb = use_bases
        
        # despite unorthogonal bases, we can still calculate variance explained by each basis
        basis_norms_sq = np.sum(np.abs(bases_sb)**2, axis=0)
        var_total = np.linalg.norm(phi_cent, 'fro')**2
        betas_Cb = phi_cent.dot(bases_sb.conj()) # Yue updated this
        variance_explained = 100.0 * np.linalg.norm(betas_Cb, axis=0)**2 / (basis_norms_sq * var_total)
        
        # to calculate variance explained by all bases together
        # Solve bases_sb @ coeff ≈ phi_cent.T in least-squares sense
        #coeff, _, _, _ = np.linalg.lstsq(bases_sb, phi_cent.T, rcond=None)
        #phi_proj = (bases_sb @ coeff).T          # C×s reconstruction
        #var_total = np.linalg.norm(phi_cent, 'fro')**2
        #variance_explained = 100.0 * np.linalg.norm(phi_proj, 'fro')**2 / var_total

    betas_Cb = phi_cent.dot(bases_sb.conj()) # Yue updated this
    # betas_Cb = phi_Cs.dot(bases_sb)
    model_Cs = np.exp(1j*np.angle(bases_sb.dot(betas_Cb.T).T))
    model_cts = model_Cs.reshape(phi_cts.shape)
    fit_C = (phi_Cs/model_Cs).mean(-1).real
    fit_ct = fit_C.reshape(phi_cts.shape[0],phi_cts.shape[1])
    betas_ctb = betas_Cb.reshape(phi_cts.shape[0],-1,bases_sb.shape[1])
    return bases_sb,fit_ct,betas_ctb,model_cts,variance_explained

def R_TW_bases_betas(uphase_cts, nBases=3, use_bases=None):
    #uphase real-valued unwrapped phase, c trials, t times, s sensors
   
    # Reshape to (combined cases/times, sensors)
    uphase_Cs = uphase_cts.reshape(-1, uphase_cts.shape[-1])
    # Center the data
    uphase_cent = uphase_Cs - uphase_Cs.mean(0)
    
    if use_bases is None:
        # Compute SVD on covariance matrix
        COV = uphase_cent.T @ uphase_cent
        u, s, vt = np.linalg.svd(COV)
        # Calculate variance explained
        variance_explained = 100.0 * s[:nBases] / s.sum()
        # Extract bases (top nBases right singular vectors)
        bases_sb = vt[:nBases].T
    else:
        bases_sb = use_bases
        variance_explained = None # Yue added this

    # Project data onto bases to get weights
    betas_Cb = uphase_cent @ bases_sb
    # Reconstruct data from bases
    model_Cs = betas_Cb @ bases_sb.T
    # Reshape model back to original shape
    model_cts = model_Cs.reshape(uphase_cts.shape)
    
    # Calculate R-squared fit (correlation between data and model)
    # Covariance between centered data and centered model
    model_cent = model_Cs - model_Cs.mean(0)
    covariance_um = (uphase_cent * model_cent).sum(1)
    
    # Normalizing factor (product of standard deviations)
    norming_um = np.sqrt(
        (uphase_cent**2).sum(1) * (model_cent**2).sum(1)
    )
    
    # Pearson correlation coefficient
    rs = covariance_um / norming_um
    
    # R-squared (coefficient of determination)
    Rs = rs**2
    fit_ct = Rs.reshape(uphase_cts.shape[0], uphase_cts.shape[1])
    
    # Reshape betas back to original dimensions
    betas_ctb = betas_Cb.reshape(uphase_cts.shape[0], -1, bases_sb.shape[1])
    
    return bases_sb,fit_ct,betas_ctb,model_cts,variance_explained


def spatial_unwrapping(waveData, edgelist_file, dataBucketName=None, dataInd=None):
    """Performs spatial phase unwrapping on complex-valued wave data to resolve 
    phase ambiguities across sensor locations.
    
    Uses a spatial unwrapping algorithm that propagates phase values across 
    contiguous sensors while minimizing phase jumps at sensor boundaries.
    
    Parameters
    ----------
    waveData : WaveData object
        The wave data object containing complex-valued phase data
    edgelist_file : str
        Path to CSV file containing sensor edge connectivity (Nx2 array of sensor pairs)
    dataBucketName : str, optional
        Which data bucket to use. If None, uses the active data bucket.
    dataInd : tuple or None, optional
        Optional index tuple to select a subset of the data (e.g., (slice(0,1), slice(10,12), ...)). 
        Must result in the same number of dimensions as full data. Use slices for singular dimensions.
        If None, uses all data.
    
    Adds data buckets to waveData:
    UnwrappedPhase : Spatially unwrapped phase data (same dimensions as input)
    
    References
    ----------
    - Phase unwrapping algorithm using nearest neighbor phase differences
    
    Usage
    ----------
    wa.spatial_unwrapping(waveData, edgelist_file, dataBucketName="complexData", dataInd=None)
    
    """
    
    # Load edge list from file
    print(f'Reading edge coordinates from {edgelist_file}')
    edges = np.loadtxt(edgelist_file, delimiter=',').astype(int)
    print(f'Loaded {edges.shape[0]} edges')
    
    # Sanity checks
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    
    hf.assure_consistency(waveData)
    complexData = waveData.get_data(dataBucketName)
    
    if dataInd:
        complexData = complexData[dataInd]
    
    origDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    origShape = complexData.shape
    desiredDimord = "trl_time_chan"
    
    hasBeenReshaped, complexData = hf.force_dimord(complexData, origDimord, desiredDimord)
    
    # Extract unit period phase representation 
    unit_period = np.angle(complexData) / (2.0 * np.pi) # shape: cts "trl_time_chan"
    
    # Perform spatial unwrapping
    print(f'Performing spatial unwrapping on {complexData.shape[0]} trials...')
    uphase_data = _unwrap_spatial_phase(unit_period, edges)
    
    # Reshape back to original dimensions if needed
    chan_names = waveData.get_channel_names()
    time = waveData.DataBuckets[dataBucketName].get_time()
    
    if hasBeenReshaped:
        uphase_data = np.reshape(uphase_data, origShape)
        uphaseBucket = wd.DataBucket(uphase_data, "UnwrappedPhase", origDimord, 
                                     time=time, chanNames=chan_names)
    else:
        uphaseBucket = wd.DataBucket(uphase_data, "UnwrappedPhase", "trl_time_chan", 
                                     time=time, chanNames=chan_names)
    
    waveData.add_data_bucket(uphaseBucket)
    
    
def _unwrap_spatial_phase(unit_period, edges):
    """Helper function to perform the actual phase unwrapping.
    
    Parameters
    ----------
    unit_period : ndarray
        Phase data in unit cycles (trials x time x channels)
    edges : ndarray
        Nx2 array of sensor connectivity (edge list)
    
    Returns
    -------
    uphase : ndarray
        Spatially unwrapped phase (same shape as input)
    """
    uphase = np.zeros(unit_period.shape, dtype=float)
    
    n_trials, n_times, n_sensors = unit_period.shape
    
    for trial in range(n_trials):
        if trial % max(1, n_trials // 20) == 0:  # Print progress every 5%
            print('.', end='', flush=True)
        
        for time_idx in range(n_times):
            # PhaseSpread returns: (max_spread, max_delta, unwrapped_phase)
            _, _, uphase[trial, time_idx, :] = PhaseSpread(
                unit_period[trial, time_idx, :], 
                edges
            )
    print()
    
    return uphase

# Include the PhaseSpread function (both C and Python versions)
# Try to use the C-version
from ctypes import cdll
try:
    _ud = cdll.LoadLibrary('libcunwrapping.so')  # Or appropriate library name
except: 
    _ud = None

if _ud and 0:  # Set to 1 to enable C version
    def PhaseSpread(unit_period, edgelist):
        """C-accelerated version of phase spreading algorithm."""
        from numpy import empty, argsort
        from ctypes import c_long, c_double, byref
        
        print('phasespread unit_period', unit_period.min(), unit_period.max())
        phase_diff = -((unit_period[edgelist[:,1]] - unit_period[edgelist[:,0]] + 0.5) % 1.0 - 0.5)
        sort_phase_diff = argsort(abs(phase_diff))
        N = len(unit_period)
        Ne = c_long(edgelist.shape[0])
        unwrapped_data = empty((N,), 'f8')
        err = c_double(0.0)
        
        _ud.phase_spread(c_long(N), unit_period.ctypes,
                         Ne, edgelist.ctypes, 
                         phase_diff.ctypes, sort_phase_diff.ctypes,
                         unwrapped_data.ctypes, byref(err))
        
        unwrapped_data = unwrapped_data - min(unwrapped_data)
        return max(unwrapped_data), abs(phase_diff).max(), unwrapped_data
        
else:
    def PhaseSpread(unit_period, edgelist):
        from numpy import argsort, abs, zeros, ones, c_, fabs
        #unwrap the nodes from lowest magnitude edge to highest magnitude edge
        
        nElectrodes = len(unit_period)
        phase_diff = -( (unit_period[edgelist[:,1]] - unit_period[edgelist[:,0]] + 0.5)%1.0 - 0.5 )
        abs_phase_diff = abs(phase_diff)
        sort_phase_diff = argsort(abs_phase_diff)
        
        unwrapped_data = -100*ones((nElectrodes), 'f8')
        nodegroup = -ones((nElectrodes), 'i4')
        errors = []
        
        #for the first edge
        s = sort_phase_diff[0]
        D, E = edgelist[s]
        unwrapped_data[[D, E]] = 0.0, -phase_diff[s]
        nodegroup[[D, E]] = 0
        
        group = 1
        for s in sort_phase_diff[1:]:
            D, E = edgelist[s]
            # print('DE', D, E, unwrapped_data[D], unwrapped_data[E])
            if unwrapped_data[D] == -100:
                if unwrapped_data[E] == -100:  # unwrapped_data[D] == -100 and unwrapped_data[E] == -100:
                    unwrapped_data[D] = 0.0
                    unwrapped_data[E] = -phase_diff[s]
                    nodegroup[D] = group
                    nodegroup[E] = group
                    group += 1
                else: # unwrapped_data[D] == -100 and unwrapped_data[E] != -100:
                    unwrapped_data[D] = unwrapped_data[E] + phase_diff[s]
                    nodegroup[D] = nodegroup[E]
            else:
                if unwrapped_data[E] == -100: # unwrapped_data[D] != -100 and unwrapped_data[E] == -100:
                    unwrapped_data[E] = unwrapped_data[D] - phase_diff[s]
                    nodegroup[E] = nodegroup[D]
                else: # unwrapped_data[D] != -100 and unwrapped_data[E] != -100:
                    # assign an error
                    if nodegroup[D] == nodegroup[E]:
                        errors.append(unwrapped_data[D] - unwrapped_data[E] - phase_diff[s])
                    #harmonize the two groups
                    elif nodegroup[D] != nodegroup[E]:
                        harmonize_to_this_group = nodegroup[D]
                        harmonize_from_this_group = nodegroup[E]
                        #calculate the correction
                        correction = unwrapped_data[D]-unwrapped_data[E]-phase_diff[s]
                        for l in range(nElectrodes):
                            if nodegroup[l] == harmonize_from_this_group:
                                #correction group associated with E
                                unwrapped_data[l] = unwrapped_data[l]+correction
                                #assign both groups to the same node group
                                nodegroup[l] = harmonize_to_this_group
    
        #print(sum(abs(errors)))
        unwrapped_data = unwrapped_data-min(unwrapped_data)
        #return unwrapped_data and phase leading electrode
        return max(unwrapped_data),fabs(phase_diff).max(),unwrapped_data
