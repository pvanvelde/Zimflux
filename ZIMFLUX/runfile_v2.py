from photonpy.simflux import ZimfluxProcessor
from photonpy import Gauss3D_Calibration
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from plot_figure_nanorulers import plot_nanorulerclusters

# -------------------------------- User inputs --------------------------------------
cfg = {
   'roisize':16,
   'spotDetectSigma': 3,
   'maxSpotsPerFrame': 2000,
   'detectionThreshold': 6,  # 6!
    'patternFrames': [[0,1,2]],
   'gain': 0.36,
   'offset': 100,
   'pixelsize' : 65, # nm/pixel
   'startframe':0,
   'maxframes': 0,
    'use3D': True,
    'chisq_threshold': 3,
    'ref_index': 1.33,
    'wavelength': 640,
    'debugMode' :False,
    'usecuda': True,
    'depth': 60
}

# calibration for astigmatic PSF to make initial guess
calib_fn = 'E:/SIMFLUX Z/astigmatism_cal_100mlam_1/astigmatist_cal_100mlam_1_MMStack_Default.ome_gausscalib20nm.yaml'
calib = Gauss3D_Calibration.from_file(calib_fn)
cfg['psf_calib'] = calib_fn

# folder name with data
folder = 'nanorulers_simflux_10ms_originalanglejan13_2_firstpattern'

# absolute path to data
path = 'E:/SIMFLUX Z/nanorulers_simflux_10ms_originalanglejan13_2_firstpattern/nanorulers_simflux_10ms_originalanglejan13_2_firstpattern_MMStack_Default.ome.tif'

# Output path for figures/results
output_folder = 'E:/results_zimflux/'

# absolute path to gain and offset data
cfg['gain'] = 'E:/SIMFLUX Z/bright_10ms_oriangle/bright_2_MMStack_Default.ome.tif'
cfg['offset'] = 'E:/SIMFLUX Z/dark_10ms_oriangle/dark_1_MMStack_Default.ome.tif'

# intial wavevector in axial direction and number of iterations refinement
init_kz = 13.2
num_iterations = 10

# -------------------------------- end --------------------------------------

# Detect spots
sp = ZimfluxProcessor(path, cfg)
sp.detect_rois(ignore_cache=False, roi_batch_size=200)

# fit vectorial PSF and find the drift
sumfit = sp.spot_fitting( vectorfit=True, check_pixels=False)
sp.drift_correct(framesPerBin=100, display=True,
                 outputfn=sp.resultsdir + 'drift')
plt.show()

# find pitch and direction pattern
sp.estimate_angles(1, pitch_minmax_nm=[400,500], dft_peak_search_range=0.004)

# refine axial pitch

sp.refine_kz(init_kz,num_iterations)

# estimate phases
sp.estimate_phases(10, iterations=10)

# zimflux estimation
e, traces, iterss = sp.gaussian_vector_2_simflux(lamda=0.1, iter=40, pick_percentage=1, depth=sp.depth)

# apply drift
sp.zf_ds_undrift = sp.zf_ds[:]
sp.zf_ds_undrift.applyDrift(sp.drift)
sp.zf_ds_undrift.save(sp.resultsdir + "zimflux_undrift" +".hdf5")
sp.sum_ds_filtered_undrift = sp.sum_ds_filtered[:]
sp.sum_ds_filtered_undrift.applyDrift(sp.drift)
sp.sum_ds_filtered_undrift.save(sp.resultsdir + "smlm_undrift" +".hdf5")

# find cluster data
plot_data= sp.cluster_picassopicksv2(pattern=0, depth=sp.depth,drift_corrected=True, fit_gaussians=False)
clusterdata = open(sp.resultsdir+'clusterdata' , 'wb')
pickle.dump(plot_data, clusterdata)
clusterdata.close()

# plot clusterdata
plot_nanorulerclusters(clusterdata,sp.resultsdir)

