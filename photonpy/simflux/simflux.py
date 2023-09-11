"""
Main SIMFLUX data processing pipeline.

photonpy - Single molecule localization microscopy library
© Jelmer Cnossen 2018-2021
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from photonpy.simflux.spotlist import SpotList
import math
import os,pickle
from photonpy import Context, Dataset
import sys
import time
import tqdm
from photonpy.cpp.estimator import Estimator
from photonpy.cpp.estim_queue import EstimQueue,EstimQueue_Results
from photonpy.cpp.roi_queue import ROIQueue
from photonpy.cpp.gaussian import Gaussian, Gauss3D_Calibration
import photonpy.cpp.spotdetect as spotdetect
from photonpy.cpp.simflux import SIMFLUX, CFSFEstimator
from scipy.interpolate import InterpolatedUnivariateSpline

import photonpy.utils.multipart_tiff as read_tiff
import photonpy.smlm.process_movie as process_movie
from photonpy.smlm.util import plot_traces
from photonpy.smlm import blinking_spots
from photonpy.cpp.postprocess import PostProcessMethods
from photonpy.utils import multipart_tiff

from .simflux_dataset import SFDataset

from photonpy.utils.caching import equal_cache_cfg, save_cache_cfg, load_cache_cfg
import photonpy.utils.fitpoints2D as fitpoints2D

figsize=(9,7)

ModDType = SIMFLUX.modulationDType

#mpl.use('svg')

# Make sure the angles dont wrap around, so you can plot them and take mean
# TODO: loop through this..
def unwrap_angle(ang):
    r = ang * 1
    ang0 = ang.flatten()[0]
    r[ang > ang0 + math.pi] -= 2 * math.pi
    r[ang < ang0 - math.pi] += 2 * math.pi
    return r


# Pattern angles wrap at 180 degrees
def unwrap_pattern_angle(ang):
    r = ang * 1
    ang0 = ang.flatten()[0]
    r[ang > ang0 + math.pi / 2] -= math.pi
    r[ang < ang0 - math.pi / 2] += math.pi
    return r


def print_phase_info(mod):
    for axis in [0, 1]:
        steps = np.diff(mod[axis::2, 3])
        steps[steps > np.pi] = -2 * np.pi + steps[steps > np.pi]
        print(f"axis {axis} steps: {-steps*180/np.pi}")



def result_dir(path):
    dir, fn = os.path.split(path)
    return dir + "/results/" + os.path.splitext(fn)[0] + "/"



def load_mod(tiffpath):
    with open(os.path.splitext(tiffpath)[0]+"_mod.pickle", "rb") as pf:
        mod = pickle.load(pf)['mod']
        assert(mod.dtype == ModDType)
        return mod




def print_mod(reportfn, mod, pattern_frames, pixelsize):
    k = mod['k']
    phase = mod['phase']
    depth = mod['depth']
    ri = mod['relint']

    for i in range(len(mod)):
        reportfn(f"Pattern {i}: kx={k[i,0]:.4f} ky={k[i,1]:.4f} Phase {phase[i]*180/np.pi:8.2f} Depth={depth[i]:5.2f} "+
               f"Power={ri[i]:5.3f} ")

    for ang in range(len(pattern_frames)):
        pat=pattern_frames[ang]
        d = np.mean(depth[pat])
        phases = phase[pat]
        shifts = (np.diff(phases[-1::-1]) % (2*np.pi)) * 180/np.pi
        shifts[shifts > 180] = 360 - shifts[shifts>180]

        with np.printoptions(precision=3, suppress=True):
            reportfn(f"Angle {ang} shifts: {shifts} (deg) (patterns: {pat}). Depth={d:.3f}")



class SimfluxProcessor:

    """
    Simflux processing.
    """
    def __init__(self, src_fn, cfg, debugMode=False):
        src_fn = os.path.abspath(src_fn)
        """
        chi-square threshold: Real threshold = chisq_threshold*roisize^2
        """
        self.pattern_frames = np.array(cfg['patternFrames'])
        self.src_fn = src_fn
        self.rois_fn = os.path.splitext(src_fn)[0] + "_rois.npy"
        self.debugMode = debugMode
        self.cfg = cfg
        if type(self.cfg['psf_calib']) != str and np.isscalar(self.cfg['psf_calib']):
            s = self.cfg['psf_calib']
            self.cfg['psf_calib'] = np.array([s,s])
        self.psf_calib = self.cfg['psf_calib']
        self.roisize = cfg['roisize']
        self.pixelsize = cfg['pixelsize']
        self.threshold = cfg['detectionThreshold']
        self.maxframes = cfg['maxframes'] if 'maxframes' in cfg else -1
        self.chisq_threshold = cfg['chisq_threshold'] if 'chisq_threshold' in cfg else 4

        self.mod_fn = os.path.splitext(self.src_fn)[0]+"-mod.pickle"
        self.numrois = None

        dir, fn = os.path.split(src_fn)
        self.resultsdir = dir + "/results/" + os.path.splitext(fn)[0] + "/"
        os.makedirs(self.resultsdir, exist_ok=True)
        self.resultprefix = self.resultsdir
        self.sum_ds_fn = self.resultprefix+'g2d_fits.hdf5'

        self.reportfile = self.resultprefix + "report.txt"
        with open(self.reportfile,"w") as f:
            f.write("")

        self.sum_ds = None
        self.sum_ds_spotfiltered=None
        self.g_undrifted=None
        self.sf_undrifted=None

        # note that 'phase' is later drift-corrected and a function of time
        self.mod = np.zeros(self.pattern_frames.size,dtype=ModDType)

    @staticmethod
    def load(src_fn):
        rois_fn = os.path.splitext(src_fn)[0] + "_rois.npy"
        cfg = load_cache_cfg(rois_fn)
        s = SimfluxProcessor(src_fn, cfg)
        s.sum_ds = Dataset.load(s.sum_ds_fn)
        s.imgshape = s.sum_ds.imgshape
        s.gaussian_fitting()
        s.load_mod()
        return s

    def _camera_calib(self, ctx):
        calib = process_movie.create_calib_obj(self.cfg['gain'], self.cfg['offset'],self.imgshape, ctx)
        return calib

    def detect_rois(self, ignore_cache=False, roi_batch_size=100, background_img=None):

        self.imgshape = read_tiff.tiff_get_image_size(self.src_fn)

        spotDetector = spotdetect.SpotDetector(self.cfg['spotDetectSigma'], self.roisize, self.threshold, backgroundImage=background_img)

        if not equal_cache_cfg(self.rois_fn, self.cfg, self.src_fn) or ignore_cache:
            with Context(debugMode=self.debugMode) as ctx:
                process_movie.detect_spots_slow(spotDetector, self._camera_calib(ctx),
                                   read_tiff.tiff_read_file(self.src_fn, self.cfg['startframe'], self.maxframes),
                                   self.pattern_frames.size, self.rois_fn,
                                   batch_size = roi_batch_size,
                                   ctx=ctx)#,numThreads=1)
            save_cache_cfg(self.rois_fn, self.cfg, self.src_fn)

        self.numrois = int(np.sum([len(ri) for ri,px in self._load_rois_iterator()]))
        print(f"Num ROIs: {self.numrois}")

        if self.numrois == 0:
            raise ValueError('No spots detected')

    def close(self):
        ...

    def bias_plot2D(self,  tag, iter=False, shift_input=None, pattern=0):
        g2d_d = self.sum_ds_filtered

        pos = g2d_d.pos[:, :3]

        filter = True
        if filter:
            filter_arr = pos[:,0]>5000/65
        else:
            filter_arr = np.ones[np.size(pos,0)]
        pos = pos[filter_arr,:]

        if shift_input is None:
            sf_d = self.sf_ds
            shift =  g2d_d.pos[filter_arr, :3] - sf_d.pos[filter_arr, :3]
        else:
            shift = shift_input

        cx = np.polyfit(pos[:, 0], shift[:, 0], 1)
        cy = np.polyfit(pos[:, 1], shift[:, 1], 1)
        cz = np.polyfit(pos[:, 2], shift[:, 2], 1)
        rms = np.sqrt(np.mean(shift ** 2, 0))
        print(f"RMS x shift={rms[0] * self.pixelsize}, y shift={rms[1] * self.pixelsize}")

        w = 25

        print(f"running 2D bias fit over {len(pos)} points")

        gridshape = [w, w]

        #        statfn = 'median'

        def median_mincount(x, minsize):
            if len(x) < minsize:
                return np.nan
            return np.median(x)

        def rms_mincount(x, minsize):
            if len(x) < minsize:
                return np.nan
            return np.sqrt(np.mean(x ** 2))

        statfn = lambda x: median_mincount(x, 20)
        metric = 'Median bias'
        #       statfn = lambda x: rms_mincount(x,20)
        #        metric='RMS'
        scatter = True

        clustergrid = [w, w]
        shape = [256, 256]

        shift = shift[ :]

        xysx = fitpoints2D.hist2d_statistic_points(pos[:, 0], pos[:, 1], shift[:, 0], shape, clustergrid,
                                                   statistic='median')
        xysy = fitpoints2D.hist2d_statistic_points(pos[:, 0], pos[:, 1], shift[:, 1], shape, clustergrid,
                                                   statistic='median')
        xysz = fitpoints2D.hist2d_statistic_points(pos[:, 0], pos[:, 1], shift[:, 2], shape, clustergrid,
                                                   statistic='median')
        rms_x = np.sqrt(np.mean(xysx[:, 2] ** 2))
        rms_y = np.sqrt(np.mean(xysy[:, 2] ** 2))
        rms_z = np.sqrt(np.mean(xysz[:, 2] ** 2))
        print(f"RMS clusters x shift={rms_x * self.pixelsize}, y shift={rms_y * self.pixelsize}, z shift = {rms_z } um")

        # print(f"2D bias plot: plotting {len(xysx)} clusters")
        # figsize = (8, 6)
        # plt.figure(figsize=figsize)
        # plt.set_cmap('jet')
        #
        # #        plt.imshow(img_x, origin='lower',extent=[0,shape[1],0,shape[0]])
        # s = self.pixelsize
        # sc =self.pixelsize
        #
        # if scatter: im = plt.scatter(s * xysx[:, 0], s * xysx[:, 1], c=sc * xysx[:, 2])
        # cb = plt.colorbar()
        # cb.set_label('Bias in X [nm]')
        # #plt.clim(-20, 20)
        #
        # #cbar.set_clim([-20, 20])
        # plt.xlabel(u'X position in FOV [um]')
        # plt.ylabel(u'Y position in FOV [um]')
        # plt.title(f'{metric} in X over FOV')
        #
        # # plt.savefig(self.outdir + tag + "x-bias-fov.png")
        # # plt.savefig(self.outdir + tag + "x-bias-fov.svg")
        #
        # plt.savefig(self.resultprefix + "x-bias-fov_"+ tag+ ".png")
        #
        # plt.figure(figsize=figsize)
        # #       plt.imshow(img_y, origin='lower',  extent=[0,shape[1],0,shape[0]])
        # if scatter: plt.scatter(s * xysy[:, 0], s * xysy[:, 1], c=sc * xysy[:, 2])
        # cb = plt.colorbar()
        # cb.set_label('Bias in Y [nm]')
        # #plt.clim(-20, 20)
        # plt.xlabel(u'X position in FOV [um]')
        # plt.ylabel(u'Y position in FOV [um]')
        # plt.title(f'{metric} in Y over FOV')
        # # plt.savefig(self.outdir + tag + "y-bias-fov.png")
        # # plt.savefig(self.outdir + tag + "y-bias-fov.svg")
        # plt.savefig(self.resultprefix + " y-bias-fov_"+ tag+ ".png")
        # plt.figure(figsize=figsize)
        # if scatter: plt.scatter(s * xysz[:, 0], s * xysz[:, 1], c= xysz[:, 2])
        # cb = plt.colorbar()
        # cb.set_label('Bias in Z [um]')
        # #plt.clim(-0.1, 0.1)
        # plt.xlabel(u'X position in FOV [um]')
        # plt.ylabel(u'Y position in FOV [um]')
        # plt.title(f'{metric} in Z over FOV')
        # plt.savefig(self.resultprefix + "z-bias-fov_"+ tag+ ".png")


        if True:
            # plot a random subset of the points
            np.random.seed(0)
            indices = np.arange(len(pos))  # np.nonzero(np.min( self.IBg[:,:,0],1) > min_intensity)[0]
            np.random.shuffle(indices)
            indices = indices[:10000]

            w, h = self.imgshape

            fig, axes = plt.subplots(3, 1, figsize=[12,12])
            px = np.polyfit(pos[indices, 0], shift[indices, 0], 1)
            py = np.polyfit(pos[indices, 1], shift[indices, 1], 1)
            pz = np.polyfit(pos[indices, 2], shift[indices, 2], 1)
            poszvsx = np.polyfit( pos[indices,0], pos[indices, 2], 1)
            sx = self.pixelsize
            sy = self.pixelsize
            sz = 1
            kx, ky, kz = np.mean([self.mod[0][0][0], self.mod[1][0][0]]), np.mean([self.mod[0][0][1], self.mod[1][0][1]]), np.mean([self.mod[0][0][2], self.mod[1][0][2]])
            axes[0].scatter(sx * pos[indices, 0], sy * shift[indices, 0], s=3)
            axes[0].plot(np.linspace(0, w) * sx, np.polyval(px, np.linspace(0, w)) * sy, 'g-')
            axes[0].set_title(f'X slope ({cx[0] * kx:.5f} rad/pixel), offset ({cx[1]:.3f})')

            axes[0].set_title('Difference between SIMFLUX and SMLM in X')
            axes[0].set_xlabel('X position in FOV [nm]')
            axes[0].set_ylabel('Difference [nm]')
            axes[0].set_ylim(-60, 60)
            #axes[0].set_ylim(-50, 50)
            axes[1].scatter(sx * pos[indices, 1], sy * shift[indices, 1], s=3)
            axes[1].plot(np.linspace(0, h) * sx, np.polyval(py, np.linspace(0, h)) * sy, 'g-')
            axes[1].set_title(f'Y shift ({cy[0] * ky:.5f} rad/pixel), offset ({cy[1]:.3f}))')
            axes[1].set_title('Difference between SIMFLUX and SMLM in Y')
            axes[1].set_xlabel('Y position in FOV [nm]')
            axes[1].set_ylabel('Difference [nm]')
            axes[1].set_ylim(-60, 60)
            #axes[1].set_xlim(2500, 3000)
            #
            axes[2].scatter(sz * pos[indices, 2], sz * shift[indices, 2], s=3)
            axes[2].plot(np.linspace(-0.3, 0.3) * sz, np.polyval(pz, np.linspace(-0.5, 0.5)) * sz, 'g-')
            axes[2].set_title(f'Z slope ({cz[0] * kz:.5f} rad/pixel), offset ({cz[1]:.3f})')

            axes[2].set_title('Difference between SIMFLUX and SMLM in Z')
            axes[2].set_xlabel('Z position in FOV [um]')
            axes[2].set_ylabel('Difference [um]')

            axes[2].set_ylim(-0.2, 0.2)
            axes[2].set_xlim(-0.3, 0.3)
            # axes[3].scatter(sx* pos[indices, 0], sz * pos[indices, 2], s=3)
            # axes[3].plot(np.linspace(0, w) * sx, np.polyval(poszvsx, np.linspace(0, w)) , 'g-')
            # axes[3].set_title(f'Z slope ({poszvsx[0] :.5f} rad/pixel), offset ({poszvsx[0]:.3f})')
            #
            # axes[3].set_title('Position Z based on SMLM as function of X')
            # axes[3].set_xlabel('X position in FOV [nm]')
            # axes[3].set_ylabel('Z [um]')

            #axes[3].set_ylim(-0.5, 0.5)

            fig.tight_layout()
            fig.savefig(self.resultprefix +  "fov-shift" +tag+ ".png")

        if False:
            # plot a random subset of the points


            np.random.seed(0)
            indices = np.arange(len(pos))  # np.nonzero(np.min( self.IBg[:,:,0],1) > min_intensity)[0]
            np.random.shuffle(indices)
            indices = indices[:10000]

            w, h = self.imgshape

            fig, axes = plt.subplots(2, 1, figsize=[12, 12])
            pxz = np.polyfit(pos[indices, 0], shift[indices, 0], 1)
            pyz = np.polyfit(pos[indices, 1], shift[indices, 1], 1)

            poszvsx = np.polyfit(pos[indices, 0], pos[indices, 2], 1)
            sx = self.pixelsize
            sy = self.pixelsize
            sz = 1
            kx, ky, kz = np.mean([self.mod[0][0][0], self.mod[1][0][0]]), np.mean(
                [self.mod[0][0][1], self.mod[1][0][1]]), np.mean([self.mod[0][0][2], self.mod[1][0][2]])
            axes[0].scatter(sx * pos[indices, 0], sz * shift[indices, 2], s=3)
            axes[0].plot(np.linspace(0, w) * sx, np.polyval(pxz, np.linspace(0, w)) * 0.001, 'g-')
            axes[0].set_title(f'X slope ({cx[0] * kx:.5f} rad/pixel), offset ({cx[1]:.3f})')

            axes[0].set_title('Difference between SIMFLUX and SMLM in X of zbias')
            axes[0].set_xlabel('X position in FOV [nm]')
            axes[0].set_ylabel('Z Difference [um]')

            # axes[0].set_ylim(-50, 50)
            axes[1].scatter(sx * pos[indices, 1], sz * shift[indices, 2], s=3)
            axes[1].plot(np.linspace(0, h) * sx, np.polyval(pyz, np.linspace(0, h)) * 0.001, 'g-')
            axes[1].set_title(f'Y shift ({cy[0] * ky:.5f} rad/pixel), offset ({cy[1]:.3f}))')
            axes[1].set_title('Difference between SIMFLUX and SMLM in Y of z bias')
            axes[1].set_xlabel('Y position in FOV [nm]')
            axes[1].set_ylabel('Difference of z [um]')
            # axes[1].set_ylim(-50, 50)


            # axes[3].scatter(sx* pos[indices, 0], sz * pos[indices, 2], s=3)
            # axes[3].plot(np.linspace(0, w) * sx, np.polyval(poszvsx, np.linspace(0, w)) , 'g-')
            # axes[3].set_title(f'Z slope ({poszvsx[0] :.5f} rad/pixel), offset ({poszvsx[0]:.3f})')
            #
            # axes[3].set_title('Position Z based on SMLM as function of X')
            # axes[3].set_xlabel('X position in FOV [nm]')
            # axes[3].set_ylabel('Z [um]')

            # axes[3].set_ylim(-0.5, 0.5)

            fig.tight_layout()
            fig.savefig(self.resultprefix + "fov-shift" + tag + ".png")
        flag = 0
        if iter:
            if pattern == 1:
                loop_array = self.pattern_frames[0,:][...,None]
            elif pattern == 2:
                loop_array = self.pattern_frames[1,:][...,None]
            else:
                loop_array = self.pattern_frames
            for ii in range(np.size(loop_array,0)):
                patmod = loop_array[ii, :]

                print(f"Adjust kx: {cx[0] * kx:.4f}, ky: {cy[0] * ky:.4f}, kz: {cz[0]*kz:.4f}. phase shift x: {cx[1] * kx:.4f}, phase shift y: {cy[1] * ky:.4f}, phase shift z: {cz[1]*kz:.4f}")
                for jj in range(len(patmod)):


                    self.mod[patmod[jj]][0][0] += cx[0] * self.mod[patmod[jj]][0][0]
                    self.mod[patmod[jj]][0][1] += cy[0] * self.mod[patmod[jj]][0][1]
                    # if self.mod[patmod[jj]][0][2] < 0:
                    #     self.mod[patmod[jj]][0][2] -= 0.2*cz[0] * self.mod[patmod[jj]][0][2]
                    # else:
                    #     print('slope: ', cz[0],' mod: ' , self.mod[patmod[jj]][0][2])
                    #     self.mod[patmod[jj]][0][2] += 0.2*cz[0] * self.mod[patmod[jj]][0][2]

                    # self.mod[patmod[jj]][2] += cx[1] * self.mod[patmod[jj]][2]
                    # self.mod[patmod[jj]][2] += cy[1] * self.mod[patmod[jj]][2]
                    # self.mod[patmod[jj]][2] += cz[1] * self.mod[patmod[jj]][2]


            if abs(cz[0] * self.mod[patmod[jj]][0][2]) < 0.005:
                flag = 1
        return cx[0],cy[0],cz[0]



    def estimate_sigma(self, initialSigma=2, plot=True):

        with Context(debugMode=self.debugMode) as ctx:
            pos_estim = Gaussian(ctx).CreatePSF_XYIBg(self.roisize, initialSigma, True)
            r = process_movie.localize_rois(self._summed_rois_iterator(), pos_estim, total=self.numrois)

            sigma_estim = Gaussian(ctx).CreatePSF_XYIBgSigmaXY(self.roisize, initialSigma, True)

            initial = np.zeros((self.numrois, 6))
            initial[:,:4] = r.estim
            initial[:,4:] = initialSigma

            r = process_movie.localize_rois(self._summed_rois_iterator(), sigma_estim, initial_estim=initial)

        if plot:
            plt.figure()
            plt.hist(r.estim [:,4], range=[1, 4],bins=200)
            plt.title('Sigma X')

            plt.figure()
            plt.hist(r.estim[:,5], range=[1, 4],bins=200)
            plt.title('Sigma Y')

        best = np.median(r.estim [:,[4,5]],0)
        print(f'Now using estimated sigma: {best}')

        self.cfg['psf_calib'] = best
        self.psf_calib = best
        return best

    def view_rois(self, ids, indices=None, summed=False, fits=None):
        from photonpy.utils.ui.show_image import array_view

        ri, pixels = process_movie.load_rois(self.rois_fn)

        if self.sum_ds.roi_id is not None:
            #px = pixels[self.roi_indices]
            indcs = np.where(np.in1d(self.sum_ds.roi_id,ids))

            px = pixels[self.sum_ds.roi_id[indcs],:,:,:]
        else:
            px = pixels

        if indices is not None:
            px = px[indices]

        if summed:
            px = np.sum(px, axis=1)

        array_view(px)

        """
        if fits is not None:
            #points = np.array([[100, 100], [200, 200], [300, 100]])
            
            for data, kwargs in fits:
                coords = np.zeros((len(data),3))
                coords[:,0] = np.arange(len(data))
                coords[:,[2,1]] = data[:,:2]

                viewer.add_points(coords, size=0.1, **kwargs)

        return viewer
        """




    def load_all_rois(self):
        ri, pixels = process_movie.load_rois(self.rois_fn)
        return pixels

    def gaussian_fitting(self, max_iter=500,  vectorfit=False,  max_iter_vec = 20, depth=0,check_pixels=False, alternative_load = False):
        """
        Make sure self.IBg and self.sum_fits are known
        """
        sys.path.insert(1, 'C:/Development/simflux3d/projects/vectorpsf')
        # region configure pupil
        import torch
        from vectorize_torch import  get_pupil_matrix
        from config import sim_params
        paramsss = sim_params(depth)
        dev = 'cuda'
        zstack = False
        # region get values from paramsss
        NA = paramsss.NA
        refmed = paramsss.refmed
        refcov = paramsss.refcov
        refimm = paramsss.refimm
        refimmnom = paramsss.refimmnom
        Lambda = paramsss.Lambda
        Npupil = paramsss.Npupil
        abberations = torch.from_numpy(paramsss.abberations).to(dev)
        zvals = torch.from_numpy(paramsss.zvals).to(dev)
        ztype = paramsss.ztype
        wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix = get_pupil_matrix(NA, zvals,
                                                                                              refmed,
                                                                                              refcov,
                                                                                              refimm,
                                                                                              refimmnom,
                                                                                              Lambda,
                                                                                              Npupil,
                                                                                              abberations,
                                                                                              dev)
        # endregion
        mu_est_list = []
        rois_info = []
        sum_fit = []
        ibg = []
        iteration_tot = []
        iterations_vector_tot = []
        iterations_ibg_tot = []
        sum_crlb = []
        sum_chisq = []
        sum_fit_vector = []
        traces_all = np.empty((max_iter_vec + 1, 0, 5))
        print('2D Gaussian fitting...',flush=True)
        with Context(debugMode=self.debugMode) as ctx:
            with self.create_psf(ctx, modulated=False) as psf, tqdm.tqdm(total=self.numrois) as pb:


                flag = 0
                for ri, pixels in self._load_rois_iterator(alt_load = alternative_load):

                    if alternative_load:
                        flag = flag+ len(ri)
                        ri2 = np.load(self.rois_fn[:-4] + 'info.npy')
                        pixels2 = np.load(self.rois_fn[:-4] + 'pixels.npy')
                        self.imgshape = read_tiff.tiff_get_image_size(self.src_fn)
                        ri_new = ri2[flag-len(ri):flag]
                        pixels_new = pixels2[flag-len(ri):flag,:,:,:]

                        summed = pixels_new.sum(1)
                    else:
                        summed = pixels.sum(1)

                    fixed_sigma_psf = Gaussian(ctx).CreatePSF_XYIBg(self.roisize, 2, cuda=True)
                    fixed_sigma_psf.SetLevMarParams(np.array([1e3, 1e3, 1e3, 1e3]), max_iter)

                    xyIbg = fixed_sigma_psf.Estimate(summed)[0]

                    guess = np.zeros((len(summed), 5))
                    guess[:, [0, 1, 3, 4]] = xyIbg
                    guess[:, 2] = 0
                    try:
                        params = self.levmarparams_astig
                    except:
                        raise NameError('Define sp.levmarparams_astig')

                    psf.SetLevMarParams(params, max_iter)

                    e, diag, traces = psf.Estimate(summed, initial=guess)

                    if vectorfit:
                        e[:,[0,1]] = e[:,[1,0]]
# 						e[:, [2]] = e[:, [2]]
                        if check_pixels:
                            e_vector, traces_vector, theta_min, theta_max, mu_est = self.gaussian_vector_2(
                                torch.tensor(summed).to('cuda'), torch.tensor(e).to('cuda'),
                                wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix, iter=max_iter_vec,
                                depth=depth, check_pixels=True)
                        else:
                            e_vector, traces_vector, theta_min, theta_max = self.gaussian_vector_2(
                                torch.tensor(summed).to('cuda'), torch.tensor(e).to('cuda'),
                                wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix, iter=max_iter_vec,
                                depth=depth)

                        iterations_vector = np.zeros(np.size(e_vector,0))


                        for loc in range(np.size(e_vector,0)):
                            try:
                                iterations_vector[loc] = np.where(traces_vector[:,loc,0]==0)[0][0]
                            except:
                                iterations_vector[loc] = max_iter_vec
                        iterations_vector_tot.append(iterations_vector)

                    else:
                        iterations_vector_tot.append(0)


                    iterations = [len(tr) for tr in traces]
                    # psf.ChiSquare()
                    sum_crlb.append(psf.CRLB(e))
                    sum_chisq.append(psf.ChiSquare(e, summed))
                    iteration_tot.append(iterations)
                    #iterations_vector
                    if alternative_load:
                        rois_info.append(ri_new)
                    else:
                        rois_info.append(ri)

                    sh = pixels.shape # numspots, numpatterns, roisize, roisize
                    pixels_rs = pixels.reshape((sh[0]*sh[1],sh[2],sh[3]))
                    params = np.repeat(e, sh[1], axis=0)

                    params[:,-1] = 0
                    params[:, -2] =1
                    ibg_, ibg_crlb_ = psf.EstimateIntensityAndBackground(params, pixels_rs, cuda=True)
                    ibg_vec = []
                    ibg_traces_vec = []
                    if vectorfit:
                        e_vector_ibg = np.repeat(e_vector, sh[1], axis=0)



                        # intial guess bg = mean outer pixels
                        temp_guess = np.mean(np.mean(pixels_rs[:, [0, int(self.roisize - 1)], :], axis=-1),
                                                      axis=1)
                        e_vector_ibg[:, -1] = (temp_guess + np.mean(np.mean(pixels_rs[:,:,  [0, int(self.roisize - 1)]], axis=-1),
                                                      axis=1))/2

                        # fix bg at expeted value
                        # e_vector_ibg[:, -1] = np.repeat(e_vector[:,-1]/8, sh[1], axis=0)
                        e_vector_ibg[:, -2] = np.clip(
                            np.sum(np.sum(pixels_rs, axis=-1), axis=-1) - e_vector_ibg[:, -1] * self.roisize * self.roisize,
                            1, np.inf)
                        pixels_rs_split = np.array_split(pixels_rs, np.size(self.pattern_frames), axis=0)
                        e_vector_ibg_split = np.array_split(e_vector_ibg,np.size(self.pattern_frames),axis=0)


                        for split in range(np.size(self.pattern_frames)):
                            iterations_ibg = max_iter_vec*1

                            temp1, temp2 = self.gaussian_vector_2_ibg_only(torch.tensor(pixels_rs_split[split]).to('cuda'),
                                                                           torch.tensor(e_vector_ibg_split[split]).to('cuda'),
                                                                           wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix, iter = iterations_ibg)
                            iterations_vector_ibg = np.zeros(np.size(temp1, 0))
                            for loc in range(np.size(temp1,0)):

                                try:
                                    if iterations_vector_ibg[loc] < np.where(temp2[:,loc,0]==0)[0][0]:
                                        iterations_vector_ibg[loc] = np.where(temp2[:,loc,0]==0)[0][0]
                                except:
                                    iterations_vector_ibg[loc] = iterations_ibg


                            ibg_vec.append(temp1)
                            ibg_traces_vec.append(temp2)
                        ibg_vec = np.concatenate(ibg_vec)
                        ibg_traces_vec = np.concatenate(ibg_traces_vec)
                        iterations_ibg_tot.append(iterations_vector_ibg)
                    ic = np.zeros((len(e)*sh[1],4))
                    if vectorfit:
                        ic[:, [0, 1]] = ibg_vec[:, 3:5]
                    else:

                        ic [:,[0,1]] = ibg_ #for normal !!!!!


                    ic [:,[2,3]] = ibg_crlb_
                    ibg.append(ic.reshape((sh[0],sh[1],4)))
                    if vectorfit:
                        traces_all= np.concatenate((traces_all, traces_vector), axis=1)
                        sum_fit_vector.append(e_vector)
                        if check_pixels:
                            mu_est_list.append(mu_est)
                    sum_fit.append(e)

                    pb.update(len(pixels))


                param_names = psf.param_names
        print(flush=True)

        sum_fit = np.concatenate(sum_fit)
        if vectorfit:
            sum_fit_vector= np.concatenate(sum_fit_vector)

        # sum_fit[sum_fit[:,2]<0,2] = sum_fit[sum_fit[:,2]<0,2] * np.polyval(np.array([0.00167957, 1]),sum_fit[sum_fit[:,2]<0,2]*1000)
        # sum_fit[sum_fit[:,2]>0, 2] = sum_fit[sum_fit[:,2]>0, 2] * np.polyval(np.array([-0.00167957, 1]),
        #                                                                sum_fit[sum_fit[:,2]>0, 2] * 1000)

        ####
        indices = np.ones(np.shape(sum_fit)[0]) == 1
        sum_fit = sum_fit[indices]
        IBg = np.concatenate(ibg)[indices]

        iteration_tot= np.zeros(np.shape(indices))
        sum_chisq = np.concatenate(sum_chisq)[indices]
        sum_crlb = np.concatenate(sum_crlb)[indices]
        rois_info = np.concatenate(rois_info)[indices]
        if vectorfit:
            sum_fit_vector[:, 0:2] = sum_fit_vector[:, 0:2] / 65 + self.roisize/2
            sum_fit_vector[:, 2] = -sum_fit_vector[:, 2] / 1000
            theta_min[0:2] =  theta_min[0:2]/ 65 + self.roisize/2
            theta_max[0:2] = theta_max[0:2] / 65 + self.roisize / 2
            theta_max[2] = theta_max[ 2] / 1000
            theta_min[2] = theta_min[ 2] / 1000
            theta_min = theta_min.detach().cpu().numpy()
            theta_max = theta_max.detach().cpu().numpy()
        #self._store_IBg_fits(sum_fit_vector, IBg, sum_chisq, sum_crlb, rois_info, param_names, iteration_tot, 0.4, max_iterations =max_iter)
        if vectorfit:
            # roipos = sp.sum_ds.pos - sp.sum_ds.local_pos
            # sp.sum_ds.pos_old = sp.sum_ds.pos * 1
            # sp.sum_ds.pos = sumfit[sp.sum_ds.roi_id, 0:3] * 1 + roipos[:, [1, 0, 2]]
            # sp.sum_ds.pos = sp.sum_ds.pos[:, [1, 0, 2]]
            sum_fit_vector[:,[0,1]] = sum_fit_vector[:,[1,0]]
            self._store_IBg_fits_vector(sum_fit_vector, IBg, sum_chisq, sum_crlb, rois_info, param_names, np.concatenate(iterations_vector_tot),np.concatenate(iterations_ibg_tot),theta_min, theta_max,
                                 max_iterations=max_iter_vec)
            if check_pixels:
                return sum_fit_vector, traces_all, ibg_vec, ibg_traces_vec, np.concatenate(iterations_vector_tot), \
                       np.concatenate(iterations_ibg_tot), np.concatenate(mu_est_list)
            else:
                return sum_fit_vector, traces_all, ibg_vec, ibg_traces_vec, np.concatenate(
                    iterations_vector_tot), np.concatenate(iterations_ibg_tot)

        else:
            self._store_IBg_fits(sum_fit, IBg, sum_chisq, sum_crlb, rois_info, param_names, iteration_tot, 0.4,
                                 max_iterations=max_iter)
            return sum_fit

    def gaussian_fitting_zstack_vdec(self, pixels2,max_iter_vec = 40):
        sys.path.insert(1, 'C:/Development/simflux3d/projects/vectorpsf')
        # region configure pupil
        import torch
        from vectorize_torch import poissonrate, get_pupil_matrix, initial_guess, thetalimits, compute_crlb, \
            Model_vectorial_psf_simflux, Model_vectorial_psf, LM_MLE_simflux, LM_MLE

        from config import sim_params
        iterations_vector_tot = []

        # fit the rois, get phase and mod, filter
        spots_ = torch.tensor(pixels2.sum(1), device='cuda').type(torch.float)
        params = sim_params(100, astig=0)
        dev = 'cuda'
        zstack = False
        # region get values from params
        NA = params.NA
        refmed = params.refmed
        refcov = params.refcov
        refimm = params.refimm
        refimmnom = params.refimmnom
        Lambda = params.Lambda
        Npupil = params.Npupil
        abberations = torch.from_numpy(params.abberations).to(dev)
        zvals = torch.from_numpy(params.zvals).to(dev)
        zspread = torch.tensor(params.zspread).to(dev)
        numparams = params.numparams
        numparams_fit = 5
        K = params.K
        Mx = params.Mx
        My = params.My
        Mz = params.Mz

        pixelsize = params.pixelsize
        Ax = torch.tensor(params.Axmt).to(dev)
        Bx = torch.tensor(params.Bxmt).to(dev)
        Dx = torch.tensor(params.Dxmt).to(dev)
        Ay = torch.tensor(params.Aymt).to(dev)
        By = torch.tensor(params.Bymt).to(dev)
        Dy = torch.tensor(params.Dymt).to(dev)

        N = params.cztN
        M = params.cztM
        L = params.cztL

        Nitermax = params.Nitermax
        wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix = get_pupil_matrix(NA, zvals, refmed,
                                                                                              refcov, refimm, refimmnom,
                                                                                              Lambda,
                                                                                              Npupil, abberations, dev)
        thetamin, thetamax = thetalimits(abberations, Lambda, Mx, My, pixelsize, zspread, dev, zstack=False)

        theta = initial_guess(spots_, dev).type(torch.float)
        theta[:, 0:2] = (theta[:, [1, 0]] - Mx / 2) * pixelsize
        theta[:, 2] = 0  # (theta[:, 2])*1000

        thetaretry = theta * 1

        param_range = torch.concat((thetamin[..., None], thetamax[..., None]), dim=1).type(torch.float)
        model = Model_vectorial_psf()

        mle = LM_MLE(model, lambda_=1e-6, iterations=40,
                     param_range_min_max=param_range, tol=torch.tensor([1e-3, 1e-3, 1e-3, 1e-1, 1e-1]).to(dev))
        mle = torch.jit.script(mle)

        e_vector, traces_vector, _ = mle.forward(spots_, NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil,
                                              abberations,
                                              0,
                                              0, K, N,
                                              M, L, Ax, Bx,
                                              Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, theta, dev, zstack,
                                              wavevector, wavevectorzimm, all_zernikes, PupilMatrix)
        mu_est, dmu = poissonrate(NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, 0,
                                    0, K, N,
                                    M, L, Ax, Bx,
                                    Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, e_vector, dev, False,
                                    wavevector, wavevectorzimm, all_zernikes, PupilMatrix)

        iterations_vector = np.zeros(np.size(e_vector, 0))
        for loc in range(np.size(e_vector, 0)):
            try:
                iterations_vector[loc] = np.where(traces_vector[:, loc, 0] == 0)[0][0]
            except:
                iterations_vector[loc] = max_iter_vec
        iterations_vector_tot.append(iterations_vector)

        sh = pixels2.shape  # numspots, numpatterns, roisize, roisize
        pixels_rs = pixels2.reshape((sh[0] * sh[1], sh[2], sh[3]))
        e_vector = e_vector.detach().cpu().numpy()
        params = np.repeat(e_vector, sh[1], axis=0)

        params[:, -1] = 0
        params[:, -2] = 1

        ibg_vec = []
        ibg_traces_vec = []
        e_vector_ibg = np.repeat(e_vector, sh[1], axis=0)

        # intial guess bg = mean outer pixels
        temp_guess = np.mean(np.mean(pixels_rs[:, [0, int(self.roisize - 1)], :], axis=-1),
                             axis=1)
        e_vector_ibg[:, -1] = (temp_guess + np.mean(np.mean(pixels_rs[:, :, [0, int(self.roisize - 1)]], axis=-1),
                                                    axis=1)) / 2

        # fix bg at expeted value
        # e_vector_ibg[:, -1] = np.repeat(e_vector[:,-1]/8, sh[1], axis=0)
        e_vector_ibg[:, -2] = np.clip(
            np.sum(np.sum(pixels_rs, axis=-1), axis=-1) - e_vector_ibg[:, -1] * self.roisize * self.roisize,
            1, np.inf)
        pixels_rs_split = np.array_split(pixels_rs, np.size(self.pattern_frames), axis=0)
        e_vector_ibg_split = np.array_split(e_vector_ibg, np.size(self.pattern_frames), axis=0)

        for split in range(np.size(self.pattern_frames)):
            iterations_ibg = max_iter_vec * 1

            temp1, temp2 = self.gaussian_vector_2_ibg_only(torch.tensor(pixels_rs_split[split]).to('cuda'),
                                                           torch.tensor(e_vector_ibg_split[split]).to('cuda'),
                                                           wavevector, wavevectorzimm, Waberration, all_zernikes,
                                                           PupilMatrix, iter=iterations_ibg)
            iterations_vector_ibg = np.zeros(np.size(temp1, 0))
            for loc in range(np.size(temp1, 0)):

                try:
                    if iterations_vector_ibg[loc] < np.where(temp2[:, loc, 0] == 0)[0][0]:
                        iterations_vector_ibg[loc] = np.where(temp2[:, loc, 0] == 0)[0][0]
                except:
                    iterations_vector_ibg[loc] = iterations_ibg
            ibg_vec.append(temp1)
            ibg_traces_vec.append(temp2)
        ibg_vec = np.concatenate(ibg_vec)
        ibg_traces_vec = np.concatenate(ibg_traces_vec)
        ic = np.zeros((len(e_vector) * sh[1], 2))

        ic[:, [0, 1]] = ibg_vec[:, 3:5]
        ic = ic.reshape(int(np.size(ic,0)/np.size(self.pattern_frames)),np.size(self.pattern_frames),2)

        return e_vector, ic, iterations_vector, mu_est.detach().cpu().numpy()


    def gaussian_fitting_zstack(self,pixels2, max_iter=500,  vectorfit=False,  max_iter_vec = 30, depth=0,check_pixels=False, roi_inf = 0):
        """
        Make sure self.IBg and self.sum_fits are known
        """

        sys.path.insert(1, 'C:/Development/simflux3d/projects/vectorpsf')
        # region configure pupil
        import torch
        from vectorize_torch import  get_pupil_matrix
        from config import sim_params

        paramsss = sim_params(depth)
        dev = 'cuda'
        zstack = False
        # region get values from paramsss
        NA = paramsss.NA
        refmed = paramsss.refmed
        refcov = paramsss.refcov
        refimm = paramsss.refimm
        refimmnom = paramsss.refimmnom
        Lambda = paramsss.Lambda
        Npupil = paramsss.Npupil
        abberations = torch.from_numpy(paramsss.abberations).to(dev)
        zvals = torch.from_numpy(paramsss.zvals).to(dev)
        ztype = paramsss.ztype
        wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix = get_pupil_matrix(NA, zvals,
                                                                                              refmed,
                                                                                              refcov,
                                                                                              refimm,
                                                                                              refimmnom,
                                                                                              Lambda,
                                                                                              Npupil,
                                                                                              abberations,
                                                                                              dev)
        # endregion
        mu_est_list = []
        rois_info = []
        sum_fit = []
        ibg = []
        iteration_tot = []
        iterations_vector_tot = []
        iterations_ibg_tot = []
        sum_crlb = []
        sum_chisq = []
        sum_fit_vector = []
        traces_all = np.empty((max_iter_vec + 1, 0, 5))
        print('2D Gaussian fitting...',flush=True)
        with Context(debugMode=self.debugMode) as ctx:
            with self.create_psf(ctx, modulated=False) as psf, tqdm.tqdm(total=self.numrois) as pb:

                batchsize= int(np.ceil(np.size(pixels2,0)/500))
                flag = 0
                pixel_batch = np.array_split(pixels2 ,batchsize)

                for pixels in pixel_batch:


                    summed = pixels.sum(1)
                    fixed_sigma_psf = Gaussian(ctx).CreatePSF_XYIBg(self.roisize, 2, cuda=True)
                    fixed_sigma_psf.SetLevMarParams(np.array([1e3, 1e3, 1e3, 1e3]), max_iter)
                    xyIbg = fixed_sigma_psf.Estimate(summed)[0]
                    guess = np.zeros((len(summed), 5))
                    guess[:, [0, 1, 3, 4]] = xyIbg
                    guess[:, 2] = 0
                    try:
                        params = self.levmarparams_astig
                    except:
                        raise NameError('Define sp.levmarparams_astig')

                    psf.SetLevMarParams(params, max_iter)

                    e, diag, traces = psf.Estimate(summed, initial=guess)
                    e[:, 2] = 0
                    if vectorfit:
                        e[:,[0,1]] = e[:,[1,0]]
                        e[:, [2]] = -1*e[:, [2]]
                        if check_pixels:
                            e_vector, traces_vector, theta_min, theta_max, mu_est = self.gaussian_vector_2(
                                torch.tensor(summed).to('cuda'), torch.tensor(e).to('cuda'),
                                wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix, iter=max_iter_vec,
                                depth=depth, check_pixels=True)
                        else:
                            e_vector, traces_vector, theta_min, theta_max = self.gaussian_vector_2(
                                torch.tensor(summed).to('cuda'), torch.tensor(e).to('cuda'),
                                wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix, iter=max_iter_vec,
                                depth=depth)

                        iterations_vector = np.zeros(np.size(e_vector,0))


                        for loc in range(np.size(e_vector,0)):
                            try:
                                iterations_vector[loc] = np.where(traces_vector[:,loc,0]==0)[0][0]
                            except:
                                iterations_vector[loc] = max_iter_vec
                        iterations_vector_tot.append(iterations_vector)

                    else:
                        iterations_vector_tot.append(0)


                    iterations = [len(tr) for tr in traces]
                    # psf.ChiSquare()
                    sum_crlb.append(psf.CRLB(e))
                    sum_chisq.append(psf.ChiSquare(e, summed))
                    iteration_tot.append(iterations)
                    #iterations_vector


                    sh = pixels.shape # numspots, numpatterns, roisize, roisize
                    pixels_rs = pixels.reshape((sh[0]*sh[1],sh[2],sh[3]))
                    params = np.repeat(e, sh[1], axis=0)

                    params[:,-1] = 0
                    params[:, -2] =1
                    ibg_, ibg_crlb_ = psf.EstimateIntensityAndBackground(params, pixels_rs, cuda=True)
                    ibg_vec = []
                    ibg_traces_vec = []
                    if vectorfit:
                        e_vector_ibg = np.repeat(e_vector, sh[1], axis=0)



                        # intial guess bg = mean outer pixels
                        temp_guess = np.mean(np.mean(pixels_rs[:, [0, int(self.roisize - 1)], :], axis=-1),
                                                      axis=1)
                        e_vector_ibg[:, -1] = (temp_guess + np.mean(np.mean(pixels_rs[:,:,  [0, int(self.roisize - 1)]], axis=-1),
                                                      axis=1))/2

                        # fix bg at expeted value
                        # e_vector_ibg[:, -1] = np.repeat(e_vector[:,-1]/8, sh[1], axis=0)
                        e_vector_ibg[:, -2] = np.clip(
                            np.sum(np.sum(pixels_rs, axis=-1), axis=-1) - e_vector_ibg[:, -1] * self.roisize * self.roisize,
                            1, np.inf)
                        pixels_rs_split = np.array_split(pixels_rs, np.size(self.pattern_frames), axis=0)
                        e_vector_ibg_split = np.array_split(e_vector_ibg,np.size(self.pattern_frames),axis=0)


                        for split in range(np.size(self.pattern_frames)):
                            iterations_ibg = max_iter_vec*1

                            temp1, temp2 = self.gaussian_vector_2_ibg_only(torch.tensor(pixels_rs_split[split]).to('cuda'),
                                                                           torch.tensor(e_vector_ibg_split[split]).to('cuda'),
                                                                           wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix, iter = iterations_ibg)
                            iterations_vector_ibg = np.zeros(np.size(temp1, 0))
                            for loc in range(np.size(temp1,0)):

                                try:
                                    if iterations_vector_ibg[loc] < np.where(temp2[:,loc,0]==0)[0][0]:
                                        iterations_vector_ibg[loc] = np.where(temp2[:,loc,0]==0)[0][0]
                                except:
                                    iterations_vector_ibg[loc] = iterations_ibg


                            ibg_vec.append(temp1)
                            ibg_traces_vec.append(temp2)
                        ibg_vec = np.concatenate(ibg_vec)
                        ibg_traces_vec = np.concatenate(ibg_traces_vec)
                        iterations_ibg_tot.append(iterations_vector_ibg)
                    ic = np.zeros((len(e)*sh[1],4))
                    if vectorfit:
                        ic[:, [0, 1]] = ibg_vec[:, 3:5]
                    else:

                        ic [:,[0,1]] = ibg_ #for normal !!!!!


                    ic [:,[2,3]] = ibg_crlb_
                    ibg.append(ic.reshape((sh[0],sh[1],4)))
                    if vectorfit:
                        traces_all= np.concatenate((traces_all, traces_vector), axis=1)
                        sum_fit_vector.append(e_vector)
                        if check_pixels:
                            mu_est_list.append(mu_est)
                    sum_fit.append(e)

                    pb.update(len(pixels))


                param_names = psf.param_names
        print(flush=True)

        sum_fit = np.concatenate(sum_fit)
        if vectorfit:
            sum_fit_vector= np.concatenate(sum_fit_vector)

        # sum_fit[sum_fit[:,2]<0,2] = sum_fit[sum_fit[:,2]<0,2] * np.polyval(np.array([0.00167957, 1]),sum_fit[sum_fit[:,2]<0,2]*1000)
        # sum_fit[sum_fit[:,2]>0, 2] = sum_fit[sum_fit[:,2]>0, 2] * np.polyval(np.array([-0.00167957, 1]),
        #                                                                sum_fit[sum_fit[:,2]>0, 2] * 1000)

        ####
        indices = np.ones(np.shape(sum_fit)[0]) == 1
        sum_fit = sum_fit[indices]
        IBg = np.concatenate(ibg)[indices]

        iteration_tot= np.zeros(np.shape(indices))
        sum_chisq = np.concatenate(sum_chisq)[indices]
        sum_crlb = np.concatenate(sum_crlb)[indices]

        if vectorfit:
            sum_fit_vector[:, 0:2] = sum_fit_vector[:, 0:2] / 65 + self.roisize/2
            sum_fit_vector[:, 2] = sum_fit_vector[:, 2] / 1000
            theta_min[0:2] =  theta_min[0:2]/ 65 + self.roisize/2
            theta_max[0:2] = theta_max[0:2] / 65 + self.roisize / 2
            theta_max[2] = theta_max[ 2] / 1000
            theta_min[2] = theta_min[ 2] / 1000
            theta_min = theta_min.detach().cpu().numpy()
            theta_max = theta_max.detach().cpu().numpy()
        if roi_inf == 0:
            rois_info,_ = process_movie.load_rois(self.rois_fn)
        else:
            rois_info, _ = process_movie.load_rois(roi_inf)
        rois_info = rois_info[0:np.size(sum_fit_vector,0)]
        #self._store_IBg_fits(sum_fit_vector, IBg, sum_chisq, sum_crlb, rois_info, param_names, iteration_tot, 0.4, max_iterations =max_iter)
        if vectorfit:
            # roipos = sp.sum_ds.pos - sp.sum_ds.local_pos
            # sp.sum_ds.pos_old = sp.sum_ds.pos * 1
            # sp.sum_ds.pos = sumfit[sp.sum_ds.roi_id, 0:3] * 1 + roipos[:, [1, 0, 2]]
            # sp.sum_ds.pos = sp.sum_ds.pos[:, [1, 0, 2]]
            sum_fit_vector[:,[0,1]] = sum_fit_vector[:,[1,0]]
            self._store_IBg_fits_vector(sum_fit_vector, IBg, sum_chisq, sum_crlb, rois_info, param_names, np.concatenate(iterations_vector_tot),np.concatenate(iterations_ibg_tot),theta_min, theta_max,
                                 max_iterations=max_iter_vec)
            if check_pixels:
                return sum_fit_vector, traces_all, ibg_vec, ibg_traces_vec, np.concatenate(iterations_vector_tot), \
                       np.concatenate(iterations_ibg_tot), np.concatenate(mu_est_list)
            else:
                return sum_fit_vector, traces_all, ibg_vec, ibg_traces_vec, np.concatenate(
                    iterations_vector_tot), np.concatenate(iterations_ibg_tot)

        else:
            self._store_IBg_fits(sum_fit, IBg, sum_chisq, sum_crlb, rois_info, param_names, iteration_tot, 0.4,
                                 max_iterations=max_iter)
            return sum_fit

    def gaussian_fitting_vtorch(self,  calib, zrange = np.inf):
        """
        Make sure self.IBg and self.sum_fits are knownd
        """

        # import shit
        from photonpy.simflux.torch_mle import gauss_psf_2D_astig, lm_mle, SIMFLUXModel, Gaussian2DAstigmaticPSF
        import photonpy.cpp.gaussian as gaussian
        import torch





        rois_info = []
        sum_fit = []
        ibg = []
        sum_crlb = []
        sum_chisq = []

        print('2D Gaussian torch fitting...',flush=True)

        with Context(debugMode=self.debugMode) as ctx:
            with self.create_psf(ctx, modulated=False) as psf, tqdm.tqdm(total=self.numrois) as pb:
                dev = torch.device('cuda')


                for ri, pixels in self._load_rois_iterator():

                    summed = pixels.sum(1)
                    fixed_sigma_psf = Gaussian(ctx).CreatePSF_XYIBg(self.roisize, 2, cuda=True)
                    fixed_sigma_psf.SetLevMarParams(np.array([1e3, 1e3, 1e3, 1e3]), 300)

                    xyIbg = fixed_sigma_psf.Estimate(summed)[0]

                    guess = np.zeros((len(summed), 5))
                    guess[:, [0, 1, 3, 4]] = xyIbg
                    guess[:, 2] = 0
                    param_range = np.array([
                        [2, self.roisize - 3],
                        [2, self.roisize - 3],
                        [-3, 3],
                        [1, 1e9],
                        [0.5, 1000],
                    ])

                    guess = torch.from_numpy(guess).to(dev)
                    summed = torch.from_numpy(summed).to(dev)
                    param_range = torch.from_numpy(param_range).to(dev)

                    model = lambda x: gauss_psf_2D_astig(x, self.roisize, [calib.x, calib.y])
                    e = lm_mle(model, guess, summed, param_range, iterations=100, lambda_=0.01).cpu().numpy()
                    # psf.ChiSquare()
                    sum_crlb.append(np.zeros(np.shape(e)))
                    sum_chisq.append(np.zeros(np.size(e,1)))

                    rois_info.append(ri)

                    sh = pixels.shape # numspots, numpatterns, roisize, roisize
                    pixels_rs = pixels.reshape((sh[0]*sh[1],sh[2],sh[3]))
                    params = np.repeat(e, sh[1], axis=0)
                    params[:,-2] = 1
                    params[:,-1] = 0


                    ibg_, ibg_crlb_ = psf.EstimateIntensityAndBackground(params, pixels_rs, cuda=True)

                    ic = np.zeros((len(e)*sh[1],4))
                    ic [:,[0,1]] = ibg_
                    ic [:,[2,3]] = ibg_crlb_
                    ibg.append(ic.reshape((sh[0],sh[1],4)))

                    sum_fit.append(e)

                    pb.update(len(pixels))

                param_names = psf.param_names
        print(flush=True)

        sum_fit = np.concatenate(sum_fit)

        indices = np.ones(np.shape(sum_fit)[0]) == 1
        sum_fit = sum_fit[indices]
        IBg = np.concatenate(ibg)[indices]

        #sum_chisq = np.concatenate(sum_chisq)[indices]
        sum_chisq =np.zeros(np.shape(indices))
        sum_crlb = np.zeros(np.shape(indices))
        #sum_crlb = np.concatenate(sum_crlb)[indices]
        rois_info = np.concatenate(rois_info)[indices]
        sum_crlb = None
        sum_chisq=np.ones(np.size(IBg,0))
        self.chisq_threshold = 0
        self._store_IBg_fits(sum_fit, IBg, sum_chisq, sum_crlb, rois_info, param_names,np.ones(np.size(IBg,0)), zrange)


    def gaussian_vector_2(self,  summed, guess, wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix,iter=40, depth=0, check_pixels=False):
        import sys
        # caution: path[0] is reserved for script path (or '' in REPL)
        sys.path.insert(1, 'C:/Development/simflux3d/projects/vectorpsf')

        import torch
        from vectorize_torch import poissonrate, get_pupil_matrix, initial_guess, thetalimits, compute_crlb, \
            Model_vectorial_psf, LM_MLE
        from config import sim_params

        paramsss = sim_params(depth)
        dev = 'cuda'
        zstack = False
        # region get values from paramsss
        NA = paramsss.NA
        refmed = paramsss.refmed
        refcov = paramsss.refcov
        refimm = paramsss.refimm
        refimmnom = paramsss.refimmnom
        Lambda = paramsss.Lambda
        Npupil = paramsss.Npupil
        abberations = torch.from_numpy(paramsss.abberations).to(dev)
        zvals = torch.from_numpy(paramsss.zvals).to(dev)
        zspread = torch.tensor(paramsss.zspread).to(dev)

        numparams_fit = 5
        K = paramsss.K
        Mx = paramsss.Mx
        My = paramsss.My


        pixelsize = paramsss.pixelsize
        Ax = torch.tensor(paramsss.Axmt).to(dev)
        Bx = torch.tensor(paramsss.Bxmt).to(dev)
        Dx = torch.tensor(paramsss.Dxmt).to(dev)
        Ay = torch.tensor(paramsss.Aymt).to(dev)
        By = torch.tensor(paramsss.Bymt).to(dev)
        Dy = torch.tensor(paramsss.Dymt).to(dev)

        N = paramsss.cztN
        M = paramsss.cztM
        L = paramsss.cztL

        thetamin, thetamax = thetalimits(abberations, Lambda, Mx, My, pixelsize, zspread, dev, zstack=False)
        param_range = torch.concat((thetamin[..., None], thetamax[..., None]), dim=1)

        guess[:, 0:2] = (guess[:, 0:2] - self.roisize / 2) * pixelsize
        guess[:, 2] =        guess[:, 2] * 1000
        guess[:, 3] = guess[:, 3] * 1.7
        model = Model_vectorial_psf()
        mle = LM_MLE(model, lambda_=1e-3, iterations=iter,
                     param_range_min_max=param_range, tol=torch.tensor([1e-3,1e-3,1e-6,1e-3,1e-3]).to(dev))
        mle = torch.jit.script(mle)
        summed = summed.type(torch.float)
        e, traces, _ = mle.forward(summed, NA, zvals, refmed, refcov, refimm, refimmnom, Lambda,
                                   Npupil, abberations, 0,
                                   0, K, N,
                                   M, L, Ax, Bx,
                                   Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, guess, dev,
                                   zstack,
                                   wavevector, wavevectorzimm, all_zernikes, PupilMatrix)
        if check_pixels:
            mu_est, dmu = model.forward(NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, 0,
                                        0, K, N,
                                        M, L, Ax, Bx,
                                        Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, e, dev, zstack,
                                        wavevector, wavevectorzimm, all_zernikes, PupilMatrix)
        e = e.detach().cpu().numpy()
        traces = traces.detach().cpu().numpy()
        if check_pixels:
            return e, traces, thetamin, thetamax, mu_est.detach().cpu().numpy()
        else:
            return e, traces, thetamin, thetamax
    def gaussian_vector_2_ibg_only(self,  summed, guess, wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix,iter=15):
        import sys
        # caution: path[0] is reserved for script path (or '' in REPL)
        sys.path.insert(1, 'C:/Development/simflux3d/projects/vectorpsf')

        import torch
        from vectorize_torch import poissonrate, get_pupil_matrix, initial_guess, thetalimits, compute_crlb, \
            Model_vectorial_psf_IBg, LM_MLE
        from config import sim_params

        paramsss = sim_params()
        dev = 'cuda'
        zstack = False
        # region get values from paramsss
        NA = paramsss.NA
        refmed = paramsss.refmed
        refcov = paramsss.refcov
        refimm = paramsss.refimm
        refimmnom = paramsss.refimmnom
        Lambda = paramsss.Lambda
        Npupil = paramsss.Npupil
        abberations = torch.from_numpy(paramsss.abberations).to(dev)
        zvals = torch.from_numpy(paramsss.zvals).to(dev)
        zspread = torch.tensor(paramsss.zspread).to(dev)
        numparams = paramsss.numparams
        numparams_fit = 5
        K = paramsss.K
        Mx = paramsss.Mx
        My = paramsss.My
        Mz = paramsss.Mz

        pixelsize = paramsss.pixelsize
        Ax = torch.tensor(paramsss.Axmt).to(dev)
        Bx = torch.tensor(paramsss.Bxmt).to(dev)
        Dx = torch.tensor(paramsss.Dxmt).to(dev)
        Ay = torch.tensor(paramsss.Aymt).to(dev)
        By = torch.tensor(paramsss.Bymt).to(dev)
        Dy = torch.tensor(paramsss.Dymt).to(dev)

        N = paramsss.cztN
        M = paramsss.cztM
        L = paramsss.cztL

        thetamin, thetamax = thetalimits(abberations, Lambda, Mx, My, pixelsize, zspread, dev, zstack=False)
        param_range = torch.concat((thetamin[..., None], thetamax[..., None]), dim=1)
        #
        # guess[:, 0:2] = (guess[:, 0:2] - self.roisize / 2) * pixelsize
        # guess[:, 2] =        guess[:, 2] * 1000
        model = Model_vectorial_psf_IBg()
        mle = LM_MLE(model, lambda_=1e-4, iterations=iter,
                     param_range_min_max=param_range, tol=0.1)
        mle = torch.jit.script(mle)
        summed = summed.type(torch.float)
        e, traces, _ = mle.forward(summed, NA, zvals, refmed, refcov, refimm, refimmnom, Lambda,
                                   Npupil, abberations, 0,
                                   0, K, N,
                                   M, L, Ax, Bx,
                                   Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, guess, dev,
                                   zstack,
                                   wavevector, wavevectorzimm, all_zernikes, PupilMatrix, ibg_only= True)
        e = e.detach().cpu().numpy()
        traces = traces.detach().cpu().numpy()
        return e, traces

    def gaussian_vector_2_simflux(self, lamda = 0.01,iter=40, pick_percentage=1, pattern=0, depth=0):
        import sys
        # caution: path[0] is reserved for script path (or '' in REPL)
        sys.path.insert(1, 'C:/Development/simflux3d/projects/vectorpsf')

        import torch
        from vectorize_torch import poissonrate, get_pupil_matrix, initial_guess, thetalimits, compute_crlb, \
            Model_vectorial_psf_simflux, LM_MLE_simflux
        from config import sim_params
        estimation = np.empty((0, 5))
        paramsss = sim_params(depth=depth)
        traces_all = np.empty((iter + 1, 0, 5))
        dev = 'cuda'
        zstack = False
        # region get values from paramsss
        NA = paramsss.NA
        refmed = paramsss.refmed
        refcov = paramsss.refcov
        refimm = paramsss.refimm
        refimmnom = paramsss.refimmnom
        Lambda = paramsss.Lambda
        Npupil = paramsss.Npupil
        abberations = torch.from_numpy(paramsss.abberations).to(dev)
        zvals = torch.from_numpy(paramsss.zvals).to(dev)
        zspread = torch.tensor(paramsss.zspread).to(dev)
        numparams = paramsss.numparams
        numparams_fit = 5
        K = paramsss.K
        Mx = paramsss.Mx
        My = paramsss.My
        Mz = paramsss.Mz

        pixelsize = paramsss.pixelsize
        Ax = torch.tensor(paramsss.Axmt).to(dev)
        Bx = torch.tensor(paramsss.Bxmt).to(dev)
        Dx = torch.tensor(paramsss.Dxmt).to(dev)
        Ay = torch.tensor(paramsss.Aymt).to(dev)
        By = torch.tensor(paramsss.Bymt).to(dev)
        Dy = torch.tensor(paramsss.Dymt).to(dev)

        N = paramsss.cztN
        M = paramsss.cztM
        L = paramsss.cztL

        thetamin, thetamax = thetalimits(abberations, Lambda, Mx, My, pixelsize, zspread, dev, zstack=False)
        param_range = torch.concat((thetamin[..., None], thetamax[..., None]), dim=1)


        model = Model_vectorial_psf_simflux(self.roisize)
        mle = LM_MLE_simflux(model, lambda_=lamda, iterations=iter,
                     param_range_min_max=param_range, tol=torch.tensor(([1e-3,1e-3,1e-5,1e-2,1e-2])).to(dev))
        #mle = torch.jit.script(mle)
        wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix = get_pupil_matrix(NA, zvals,
                                                                                              refmed,
                                                                                              refcov,
                                                                                              refimm,
                                                                                              refimmnom,
                                                                                              Lambda,
                                                                                              Npupil,
                                                                                              abberations,
                                                                                              dev)

        indices, moderr = self.filter_spots(self.moderror_threshold)
        temp = np.zeros(np.shape(indices))
        select = np.linspace(0, indices.size-1, int(len(indices) * pick_percentage)).astype(int)
        # for itter in range(len(temp)):
        #     if itter in select:
        #         temp[iter] = 1
        #     else:
        #         temp[iter] = 0


        #select = np.random.choice(np.arange(indices.size), size=int(len(indices) * pick_percentage), replace=False)
        indices = indices[select]
        moderr = moderr[select]
        numblocks = np.ceil(len(indices) / 400)
        block_indices = np.array_split(indices, numblocks)
        print(f"# filtered spots left: {len(indices)}. median moderr:{np.median(moderr):.2f}")


        mod = self.mod_at_frame(self.sum_ds.frame)

        print('\n Perform Simflux localization via pytorch... ')
        iterations_vector_tot = []
        for blocks in tqdm.trange(int(numblocks)):
            indices_single = block_indices[blocks]
            mod_single = mod[indices_single, :]
            mod_torch = np.zeros([len(indices_single), np.size(self.pattern_frames), 6])

            for i in range(np.size(mod_torch, 0)):
                for j in range(np.size(self.pattern_frames)):
                    mod_torch[i, j, 0:3] = mod_single[i, j][0]
                    mod_torch[i, j, 3] = mod_single[i, j][1]
                    mod_torch[i, j, 4] = mod_single[i, j][2]
                    mod_torch[i, j, 5] = mod_single[i, j][3]
            roi_idx = self.sum_ds.roi_id[indices_single]

            roi_info, pixels = process_movie.load_rois(self.rois_fn)
            pixels = pixels[roi_idx]

            if pattern == 1:
                mod_torch = mod_torch[:, [0, 2, 4], :]
                pixels = pixels[:, [0, 2, 4], :]
            if pattern == 2:
                mod_torch = mod_torch[:, [1, 3, 5], :]
                pixels = pixels[:, [1, 3, 5], :]
            roi_pos = np.vstack((roi_info['x'], roi_info['y'], np.zeros(np.shape(roi_info['y'])))).T
            roi_pos = roi_pos[roi_idx, :]
            with torch.no_grad():
                dev = torch.device('cuda')

                smp_ = torch.from_numpy(pixels).to(dev)  # summed for modulated model
                mod_ = torch.from_numpy(np.asarray(mod_torch)).to(dev)
                roi_pos_= torch.from_numpy(np.asarray(roi_pos[:,:2])).to(dev)
                initial = np.zeros([len(indices_single), 5])
                dims = 3

                # errror in local pos correction
                local_pos = self.sum_ds.local_pos[indices_single] + roi_pos
                #local_pos[:, [0, 1, 2]] = local_pos[:, [1, 0, 2]]
                initial[:, :dims] = local_pos - roi_pos


                initial[:, -1] = self.sum_ds.background[
                                     indices_single]  # divide by numpat?????????????????????????????????????
                initial[:, -2] = self.sum_ds.photons[indices_single]

                initial_ = torch.from_numpy(initial).to(dev)


                e, traces, _ = mle.forward(smp_, NA, zvals, refmed, refcov, refimm, refimmnom, Lambda,
                                           Npupil, abberations, 0,
                                           0, K, N,
                                           M, L, Ax, Bx,
                                           Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, initial_, 'cuda',
                                           zstack,
                                           wavevector, wavevectorzimm, all_zernikes, PupilMatrix, mod_, roi_pos_)
                traces = traces.cpu().numpy()

                iterations_vector = np.zeros(np.size(e, 0))
                for loc in range(np.size(e, 0)):
                    try:
                        iterations_vector[loc] = np.where(traces[:, loc, 0] == 0)[0][0]
                    except:
                        iterations_vector[loc] = iter
                iterations_vector_tot.append(iterations_vector)

                estim = e.cpu().numpy()
                estim[:,2] = -estim[:,2]*1
                estimation = np.append(estimation, estim, axis=0)
                traces_all = np.concatenate((traces_all, traces),axis=1)


        import copy



        #
        # ds = Dataset.fromEstimates(estimation, ['x', 'y', 'z', 'I', 'bg'], self.sum_ds.frame[indices],
        #                            self.imgshape,
        #                            roipos=self.sum_ds.data['roipos'][indices],
        #                            chisq=np.zeros(np.shape(self.sum_ds.frame[indices])))
        iterations_vector_tot = np.concatenate(iterations_vector_tot)
        border = 2.1
        sel = ((estimation[:, 0] > border) & (estimation[:, 0] < self.roisize - border - 1) &
               (estimation[:, 1] > border) & (estimation[:, 1] < self.roisize - border - 1) &
               (estimation[:, 2] > -0.5) & (estimation[:, 2] < 0.5))
        sel = np.logical_and(sel,(iterations_vector_tot < iter))
        print(f'Filtering on position in ROI and iterations: {np.shape(estimation)[0] - sel.sum()}/{np.shape(estimation)[0]} spots removed.')


        indices = indices[sel]

        iterations_vector_tot = iterations_vector_tot[sel]
        self.sum_ds_filtered = self.sum_ds[indices]
        ds = copy.deepcopy(self.sum_ds_filtered)
        pos = estimation * 1

        roipos_add = self.sum_ds.data['roipos'][indices]
        pos[sel , 0:2] += roipos_add[:, [-1, -2]]
        ds.pos = pos[sel , :3]
        ds.photons = estimation[sel , 3]
        ds.background = estimation[sel , 4]

        #filter crap
        #for visual
        #ds.pos[: , 2]= -ds.pos[: , 2]
        #self.sum_ds_filtered.pos[: , 2] = -self.sum_ds_filtered.pos[: , 2]
        self.sf_ds = ds
        self.sf_ds.save(self.resultprefix + "simflux" + ".hdf5")
        self.sum_ds_filtered.save(self.resultprefix + "g2d-filtered" + ".hdf5")


        return e, traces_all, iterations_vector_tot


    def gaussian_vector(self, zrange=np.inf, iter=40):
        """
        Make sure self.IBg and self.sum_fits are known
        """

        # import shit
        import sys
        # caution: path[0] is reserved for script path (or '' in REPL)
        sys.path.insert(1, 'C:/Development/simflux3d/projects/vectorpsf')

        import torch
        from vectorize_torch import poissonrate, get_pupil_matrix, initial_guess, thetalimits, compute_crlb, \
            Model_vectorial_psf, LM_MLE
        from config import sim_params
        rois_info = []
        sum_fit = []
        traces_all = np.empty((iter+1,0,5))
        ibg = []
        sum_crlb = []
        sum_chisq = []

        print('2D Gaussian torch fitting...', flush=True)

        with Context(debugMode=self.debugMode) as ctx:


            with self.create_psf(ctx, modulated=False) as psf, tqdm.tqdm(total=self.numrois) as pb:
                dev = torch.device('cuda')
                for ri, pixels in self._load_rois_iterator():


                    paramsss = sim_params()
                    dev = 'cuda'
                    zstack = False
                    # region get values from paramsss
                    NA = paramsss.NA
                    refmed = paramsss.refmed
                    refcov = paramsss.refcov
                    refimm = paramsss.refimm
                    refimmnom = paramsss.refimmnom
                    Lambda = paramsss.Lambda
                    Npupil = paramsss.Npupil
                    abberations = torch.from_numpy(paramsss.abberations).to(dev)
                    zvals = torch.from_numpy(paramsss.zvals).to(dev)
                    zspread = torch.tensor(paramsss.zspread).to(dev)
                    numparams = paramsss.numparams
                    numparams_fit = 5
                    K = paramsss.K
                    Mx = paramsss.Mx
                    My = paramsss.My
                    Mz = paramsss.Mz

                    pixelsize = paramsss.pixelsize
                    Ax = torch.tensor(paramsss.Axmt).to(dev)
                    Bx = torch.tensor(paramsss.Bxmt).to(dev)
                    Dx = torch.tensor(paramsss.Dxmt).to(dev)
                    Ay = torch.tensor(paramsss.Aymt).to(dev)
                    By = torch.tensor(paramsss.Bymt).to(dev)
                    Dy = torch.tensor(paramsss.Dymt).to(dev)

                    N = paramsss.cztN
                    M = paramsss.cztM
                    L = paramsss.cztL

                    summed = pixels.sum(1)
                    fixed_sigma_psf = Gaussian(ctx).CreatePSF_XYIBg(self.roisize, 2, cuda=True)
                    fixed_sigma_psf.SetLevMarParams(np.array([1e3, 1e3, 1e3, 1e3]), 300)

                    xyIbg = fixed_sigma_psf.Estimate(summed)[0]

                    guess = np.zeros((len(summed), 5))
                    guess[:, [0, 1, 3, 4]] = xyIbg
                    guess[:, 2] = 0
                    param_range = np.array([
                        [2, self.roisize - 3],
                        [2, self.roisize - 3],
                        [-3, 3],
                        [1, 1e9],
                        [0.5, 1000],
                    ])

                    guess = torch.from_numpy(guess).to(dev)
                    summed = torch.from_numpy(summed).to(dev)
                    param_range = torch.from_numpy(param_range).to(dev)
                    thetamin, thetamax = thetalimits(abberations, Lambda, Mx, My, pixelsize, zspread, dev, zstack=False)
                    param_range = torch.concat((thetamin[..., None], thetamax[..., None]), dim=1)

                    guess[:, 0:2] = (guess[:, 0:2] - self.roisize / 2) * pixelsize

                    model = Model_vectorial_psf()
                    mle = LM_MLE(model, lambda_=1e-1, iterations=iter,
                                 param_range_min_max=param_range, tol=1e-3)
                    mle = torch.jit.script(mle)
                    wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix = get_pupil_matrix(NA, zvals,
                                                                                                          refmed,
                                                                                                          refcov,
                                                                                                          refimm,
                                                                                                          refimmnom,
                                                                                                          Lambda,
                                                                                                          Npupil,
                                                                                                          abberations,
                                                                                                           dev)
                    e, traces, _ = mle.forward(summed, NA, zvals, refmed, refcov, refimm, refimmnom, Lambda,
                                                      Npupil, abberations, 0,
                                                      0, K, N,
                                                      M, L, Ax, Bx,
                                                      Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, guess, dev,
                                                      zstack,
                                                      wavevector, wavevectorzimm, all_zernikes, PupilMatrix)
                    e = e.detach().cpu().numpy()
                    traces = traces.detach().cpu().numpy()
                    # psf.ChiSquare()
                    sum_crlb.append(np.zeros(np.shape(e)))
                    sum_chisq.append(np.zeros(np.size(e, 1)))

                    rois_info.append(ri)

                    sh = pixels.shape  # numspots, numpatterns, roisize, roisize
                    pixels_rs = pixels.reshape((sh[0] * sh[1], sh[2], sh[3]))



                    sum_fit.append(e)
                    traces_all = np.concatenate((traces_all,traces), axis =1)

                    pb.update(len(pixels))

            param_names = psf.param_names


        sum_fit = np.concatenate(sum_fit)

        return sum_fit, traces_all

    def fix_rois_laeteral(self, rois_ori):

        newrois = rois_ori[rois_ori['id'] == 0]

        maxframes = np.floor(read_tiff.tiff_get_movie_size(self.src_fn)[1] / np.size(self.pattern_frames))
        for i in np.arange(1, maxframes):
            insert_array = rois_ori[rois_ori['id'] == 0]
            insert_array['id'] = i
            newrois = np.hstack((insert_array, newrois))

        def save_rois_vp(f, rois_info, pixels):
            np.save(f[:-4] + 'info.npy', rois_info)
            np.save(f[:-4] + 'pixels.npy', pixels)
            self.numrois = len(rois_info)

        # create roi in format
        image = read_tiff.tiff_read_all(self.src_fn)[0:int(maxframes * np.size(self.pattern_frames))]
        image2 = np.reshape(image, (
            int(maxframes), np.size(self.pattern_frames), np.size(image, 1),
            np.size(image, 2)))
        pxstack = np.zeros([len(newrois), np.size(self.pattern_frames), self.roisize, self.roisize])
        pxstack_summedpat1 = np.zeros(
            [len(rois_ori), int(np.size(self.pattern_frames) / len(self.pattern_frames)), self.roisize, self.roisize])

        for jj in range(len(newrois)):
            frame = newrois['id'][jj]
            x = newrois['x'][jj]
            y = newrois['y'][jj]
            pxstack[jj, :, :, :] = image2[frame, :, y:(y + self.roisize), x:(x + self.roisize)]
            pxstack_summedpat1[None,jj % len(rois_ori), :, :, :] += image2[frame, self.pattern_frames[0],
                                                               y:(y + self.roisize), x:(x + self.roisize)]


        staticroi = np.tile(np.arange(len(rois_ori) - 1, -1, -1), int(np.size(image, 0) / np.size(self.pattern_frames)))
        save_rois_vp(self.rois_fn, newrois, pxstack)

        return staticroi, pxstack, newrois, (pxstack_summedpat1 / len(rois_ori) - 100) * 0.4

    def fix_rois_through_frames(self, rois_ori):

        newrois = rois_ori[rois_ori['id'] == 0]

        maxframes = np.floor(read_tiff.tiff_get_movie_size(self.src_fn)[1] / np.size(self.pattern_frames))
        for i in np.arange(1, maxframes):
            insert_array = rois_ori[rois_ori['id'] == 0]
            insert_array['id'] = i
            newrois = np.hstack((insert_array, newrois))

        def save_rois_vp(f, rois_info, pixels):
            np.save(f[:-4] + 'info.npy', rois_info)
            np.save(f[:-4] + 'pixels.npy', pixels)
            self.numrois = len(rois_info)

        # create roi in format
        image = read_tiff.tiff_read_all(self.src_fn)[0:int(maxframes*np.size(self.pattern_frames))]
        image2 = np.reshape(image, (
        int(maxframes), np.size(self.pattern_frames), np.size(image, 1),
        np.size(image, 2)))
        pxstack = np.zeros([len(newrois), np.size(self.pattern_frames), self.roisize, self.roisize])
        pxstack_summedpat1 = np.zeros([len(rois_ori), int(np.size(self.pattern_frames)/len(self.pattern_frames)), self.roisize, self.roisize])
        pxstack_summedpat2 = np.zeros([len(rois_ori), int(np.size(self.pattern_frames)/len(self.pattern_frames)), self.roisize, self.roisize])
        for jj in range(len(newrois)):
            frame = newrois['id'][jj]
            x = newrois['x'][jj]
            y = newrois['y'][jj]
            pxstack[jj, :, :, :] = image2[frame, :, y:(y + self.roisize), x:(x + self.roisize)]
            pxstack_summedpat1[jj%len(rois_ori),:,:,:] +=image2[frame, self.pattern_frames[0], y:(y + self.roisize), x:(x + self.roisize)]
            pxstack_summedpat2[jj%len(rois_ori),:,:,:] +=image2[frame, self.pattern_frames[1], y:(y + self.roisize), x:(x + self.roisize)]

        staticroi = np.tile(np.arange(len(rois_ori) - 1, -1, -1), int(np.size(image, 0) / np.size(self.pattern_frames)))
        save_rois_vp(self.rois_fn, newrois, pxstack)

        return staticroi, pxstack, newrois, (pxstack_summedpat1/len(rois_ori)-100)*0.4, (pxstack_summedpat2/len(rois_ori)-100)*0.4

    def _store_IBg_fits_vector(self, sum_fit, ibg, sum_chisq, sum_crlb, rois_info,psf_param_names, iterations, iterations_ibg,theta_min, theta_max, max_iterations=500, torchv = False):
        """
        roi_indices: The indices into the list of ROIs
        """
        roipos = np.zeros((len(rois_info),2), dtype=np.int32)
        roipos[:,0] = rois_info['y']
        roipos[:,1] = rois_info['x']
        if torchv:

            ds = SFDataset.fromEstimates(sum_fit, psf_param_names, rois_info['id'],
                                         self.imgshape, crlb=sum_crlb, chisq=sum_chisq, iterations=np.ones(np.size(roipos,0)),
                                         roipos=roipos,
                                         numPatterns=ibg.shape[1])

        else:

            ds = SFDataset.fromEstimates(sum_fit, psf_param_names, rois_info['id'],
                                  self.imgshape, crlb=sum_crlb, chisq=sum_chisq,iterations=iterations,
                                  roipos=roipos,
                                  numPatterns=ibg.shape[1])

        ds.data.IBg = ibg[:,:,:2]
        ds.data.IBg_crlb = ibg[:,:,2:]

        if False:
            threshold = self.roisize**2 * self.chisq_threshold

            if False:
                plt.figure()
                plt.hist(ds.chisq, bins=50, range=[0,4000])
                plt.title('Non-simflux fit chi-square')
                plt.axvline(threshold, label='Threshold')

            ok = np.logical_and(sum_chisq < threshold)
            print(f"Accepted {np.sum(ok)}/{self.numrois} spots (chi-square threshold={threshold:.1f}")
        else:
            ok = np.ones(sum_chisq.shape, dtype=np.bool)

        border = self.roisize/4
        ok = ok & (
            (sum_fit[:,0] > theta_min[0]) &
            (sum_fit[:,1] > theta_min[1]) &
            (sum_fit[:,0] < theta_max[0]) &
            (sum_fit[:,1] < theta_max[1]) &
            (abs(sum_fit[:,2]) < theta_max[2]) &
            (iterations < max_iterations)&
            (iterations_ibg < max_iterations))

        ds = ds[ok]
        print('Keep ' + str(sum((iterations < max_iterations)&
            (iterations_ibg < max_iterations))) + ' spots, from ' + str(len(iterations)) + ' detections, based on iterations Gaussian estimator')
        print('Keep ' + str(sum (abs(sum_fit[:,2]) < theta_max[2])) + ' spots, from ' + str(len(iterations)) + ' detections, based on z_range')

        self.sum_ds = ds
        self.sum_ds.save(self.sum_ds_fn)

        self.spotlist = SpotList(self.sum_ds, self.selected_roi_source, pixelsize=self.cfg['pixelsize'],
                            outdir=self.resultsdir, IBg=ds.data.IBg[:,:], IBg_crlb=ds.data.IBg_crlb[:,:])

        median_crlb_x = np.median(self.sum_ds.crlb.pos[:,0])
        median_I = np.median(self.sum_ds.photons)

        self.report(f"g2d mean I={median_I:.1f}. mean crlb x {median_crlb_x:.4f}")

    def _store_IBg_fits(self, sum_fit, ibg, sum_chisq, sum_crlb, rois_info, psf_param_names, iterations,
                        zrange=np.inf, max_iterations=500, torchv=False):
        """
        roi_indices: The indices into the list of ROIs
        """
        roipos = np.zeros((len(rois_info), 2), dtype=np.int32)
        roipos[:, 0] = rois_info['y']
        roipos[:, 1] = rois_info['x']
        if torchv:

            ds = SFDataset.fromEstimates(sum_fit, psf_param_names, rois_info['id'],
                                         self.imgshape, crlb=sum_crlb, chisq=sum_chisq,
                                         iterations=np.ones(np.size(roipos, 0)),
                                         roipos=roipos,
                                         numPatterns=ibg.shape[1])

        else:

            ds = SFDataset.fromEstimates(sum_fit, psf_param_names, rois_info['id'],
                                         self.imgshape, crlb=sum_crlb, chisq=sum_chisq, iterations=iterations,
                                         roipos=roipos,
                                         numPatterns=ibg.shape[1])

        ds.data.IBg = ibg[:, :, :2]
        ds.data.IBg_crlb = ibg[:, :, 2:]

        if self.chisq_threshold > 0:
            threshold = self.roisize ** 2 * self.chisq_threshold

            if False:
                plt.figure()
                plt.hist(ds.chisq, bins=50, range=[0, 4000])
                plt.title('Non-simflux fit chi-square')
                plt.axvline(threshold, label='Threshold')

            ok = np.logical_and(sum_chisq < threshold, abs(sum_fit[:, 2]) < zrange)
            print(f"Accepted {np.sum(ok)}/{self.numrois} spots (chi-square threshold={threshold:.1f}")
        else:
            ok = np.ones(sum_chisq.shape, dtype=np.bool)

        border = self.roisize / 4
        ok = ok & (
                (sum_fit[:, 0] > border) &
                (sum_fit[:, 1] > border) &
                (sum_fit[:, 0] < self.roisize - border - 1) &
                (sum_fit[:, 1] < self.roisize - border - 1) &
                (abs(sum_fit[:, 2]) < zrange) &
                (iterations < max_iterations))

        ds = ds[ok]
        print('Keep ' + str(sum((iterations < max_iterations))) + ' spots, from ' + str(
            len(iterations)) + ' detections, based on iterations Gaussian estimator')
        print('Keep ' + str(sum(abs(sum_fit[:, 2]) < zrange)) + ' spots, from ' + str(
            len(iterations)) + ' detections, based on z_range')

        self.sum_ds = ds
        self.sum_ds.save(self.sum_ds_fn)

        self.spotlist = SpotList(self.sum_ds, self.selected_roi_source, pixelsize=self.cfg['pixelsize'],
                                 outdir=self.resultsdir, IBg=ds.data.IBg[:, :], IBg_crlb=ds.data.IBg_crlb[:, :])

        median_crlb_x = np.median(self.sum_ds.crlb.pos[:, 0])
        median_I = np.median(self.sum_ds.photons)

        self.report(f"g2d mean I={median_I:.1f}. mean crlb x {median_crlb_x:.4f}")

    def save_pickle(self, fn):
        with open(fn, "wb") as f:
            pickle.dump(self.sum_ds, f)

    def load_pickle(self, fn):

        with open(fn, "rb") as f:
            self.sum_ds = pickle.load(f)

        self.spotlist = SpotList(self.sum_ds, self.selected_roi_source, pixelsize=self.cfg['pixelsize'],
                    outdir=self.resultsdir, IBg=self.sum_ds.data.IBg[:,:], IBg_crlb=self.sum_ds.data.IBg_crlb[:,:])

    def remove_indices(self, indc):
        indc = ~indc

        self.sum_ds.IBg_filtered =  self.sum_ds.IBg[indc,:,:]
        self.sum_ds.background_filtered = self.sum_ds.background[indc]
        self.sum_ds.crlb_filtered = self.sum_ds.crlb[indc]
        self.sum_ds.data_filtered = self.sum_ds.data[indc]
        self.sum_ds.frame_filtered = self.sum_ds.frame[indc]
        self.sum_ds.group_filtered = self.sum_ds.group[indc]
        self.sum_ds.local_pos_filtered = self.sum_ds.local_pos[indc, :]
        self.sum_ds.photons_filtered = self.sum_ds.photons[indc]
        self.sum_ds.pos_filtered = self.sum_ds.pos[indc, :]
        self.sum_ds.roi_id_filtered = self.sum_ds.roi_id[indc]
        self.sum_ds.chi_filtered = self.sum_ds.chi[indc]
        self.sum_ds.modulation_filtered = self.sum_ds.modulation[indc]
        self.sum_ds.params_sine_filtered = self.sum_ds.params_sine[indc,:]

    def estimate_angles(self, num_angle_bins=1, pitch_minmax_nm=[300,1000],
                          dft_peak_search_range=0.02, debug_images=False):
        freq_minmax = 2*np.pi / (np.array(pitch_minmax_nm[::-1]) / self.pixelsize)
        nframes = self.sum_ds.numFrames
        fr = np.arange(nframes)

        with Context(debugMode=self.debugMode) as ctx:
            angles, pitch = self.spotlist.estimate_angle_and_pitch(
                self.pattern_frames,
                frame_bins=np.array_split(fr, num_angle_bins),
                ctx=ctx,
                freq_minmax=freq_minmax,
                debug_images=debug_images,
                dft_peak_search_range=dft_peak_search_range
            )

        print("Pitch and angle estimation: ")
        for k in range(len(self.pattern_frames)):
            angles[angles[:, k] > 0.6 * np.pi] -= np.pi  # 180 deg to around 0
            angles[:, k] = unwrap_pattern_angle(angles[:, k])
            angles_k = angles[:, k]
            pitch_k = pitch[:, k]
            self.report(f"Angle {k}: { np.rad2deg(np.mean(angles_k)) :7.5f} [deg]. Pitch: {np.mean(pitch_k)*self.pixelsize:10.5f} ({2*np.pi/np.mean(pitch_k):3.3f} [rad/pixel])")

            freq = 2 * np.pi / np.mean(pitch_k)
            kx = np.cos(np.mean(angles_k)) * freq
            ky = np.sin(np.mean(angles_k)) * freq
            self.mod['k'][self.pattern_frames[k], :2] = kx,ky
            self.mod['k'][self.pattern_frames[k], 2] = 0

            if 'use3D' in self.cfg and self.cfg['use3D']:
                """
                wavelen = self.cfg['wavelength']
            
                beamangle = np.arcsin(self.cfg['wavelength'] / (self.cfg['ref_index'] * pitch_k * self.pixelsize))
                if p_ax != 0:
                    p_axial = (wavelen / (self.cfg['ref_index'] * (1 - np.cos(beamangle)))) * p_ax
                else:
                    p_axial = (wavelen / (self.cfg['ref_index'] * (1 - np.cos(beamangle))))

                # k = self.mod['k'] / self.cfg['pixelsize'] # k is now in rad/nm
                # kr = np.sqrt( (k[:,:2]**2).sum(-1) )
                # beamangle = np.arcsin(self.cfg['wavelength'] / (self.cfg['ref_index'] * np.mean(pitch_k) * self.pixelsize))
                # beamAngle = beamangle
                """

                if k== 0:
                    self.mod['k'][self.pattern_frames[k], 2] = 4.33

                elif k==1:
                    self.mod['k'][self.pattern_frames[k], 2] = 4.73

                self.excDims = 3
                print(self.excDims)
            else:
                self.excDims = 3

    def report_pattern(self):
        ...

    def estimate_kz(self, kz0_range, kz1_range, initial_filter, numpoints=10):

        kz0 = np.linspace(kz0_range[0], kz0_range[1], numpoints)
        kz1 = np.linspace(kz1_range[0], kz1_range[1], numpoints)
        mean_depth_pat0 = np.zeros(np.shape(kz0))
        mean_depth_pat1 = np.zeros(np.shape(kz0))
        mean_error_pat0 = np.zeros(np.shape(kz0))
        mean_error_pat1 = np.zeros(np.shape(kz1))

        for i in range(len(kz0)):
            k_iter = kz0[i]
            self.estimate_patterns_iter(first_iteration=True, kz=np.array([k_iter, 5.05]))
            self.estimate_phases(num_phase_bins=10, iterations=10, accept_percentile=100)
            errpat0, _ = self.compute_mean_moderr(initial_filter)
            mean_error_pat0[i] = np.mean(errpat0)
            mean_depth_pat0[i] = np.mean(self.mod['depth'][self.pattern_frames[0]])
        for i in range(len(kz1)):
            k_iter = kz1[i]
            self.estimate_patterns_iter(first_iteration=True, kz=np.array([5.05, k_iter]))
            self.estimate_phases(num_phase_bins=10, iterations=10, accept_percentile=100)
            _, errpat1 = self.compute_mean_moderr(initial_filter)
            mean_error_pat1[i] = np.mean(errpat1)
            mean_depth_pat1[i] = np.mean(self.mod['depth'][self.pattern_frames[1]])
        plt.figure()
        plt.scatter(kz0, mean_error_pat0, label='data')

        p = np.polyfit(kz0, mean_error_pat0, 4)
        plt.plot(np.linspace(min(kz0), max(kz0), 100), np.polyval(p, np.linspace(min(kz0), max(kz0), 100)), color='red',
                 linewidth=1, label='4th order polynomial')

        # plt.vlines(5.05, min(mean_error_pat1)-0.05*max(mean_error_pat1), max(mean_error_pat1), 'red')
        plt.xlabel('kz0')
        plt.ylabel('Mean modulation error for direction 0 [(I_exp - I_found)/I_tot]')
        plt.title('kz = [ kz0, _ ]')
        plt.legend()
        plt.tight_layout()
        plt.show()

        minima = np.roots(np.polyder(p))
        minimumkz0 = minima[np.argmin(np.polyval(p, minima))]

        plt.figure()
        plt.scatter(kz1, mean_error_pat1, label='data')

        p = np.polyfit(kz1, mean_error_pat1, 4)
        plt.plot(np.linspace(min(kz1), max(kz1), 100), np.polyval(p, np.linspace(min(kz1), max(kz1), 100)), color='red',
                 linewidth=1, label='4th order polynomial')

        # plt.vlines(5.05, min(mean_error_pat1)-0.05*max(mean_error_pat1), max(mean_error_pat1), 'red')
        plt.xlabel('kz1')
        plt.ylabel('Mean modulation error for direction 1 [(I_exp - I_found)/I_tot]')
        plt.title('kz = [ _ , kz1]')
        plt.legend()
        plt.tight_layout()
        plt.show()

        minima = np.roots(np.polyder(p))
        minimumkz1 = minima[np.argmin(np.polyval(p, minima))]

        return minimumkz0, minimumkz1,  mean_depth_pat0, mean_depth_pat1

    def set_kz(self, k_patz):
        for ii in range(np.size(self.pattern_frames, 0)):
            patmod = self.pattern_frames[ii, :]
            for jj in range(len(patmod)):
                self.mod[patmod[jj]][0][2] = k_patz[ii]

    def set_kx(self, k_patx):
        for ii in range(np.size(self.pattern_frames, 0)):
            patmod = self.pattern_frames[ii, :]
            for jj in range(len(patmod)):
                self.mod[patmod[jj]][0][0] = k_patx[ii]

    def set_ky(self, k_patz):
        for ii in range(np.size(self.pattern_frames, 0)):
            patmod = self.pattern_frames[ii, :]
            for jj in range(len(patmod)):
                self.mod[patmod[jj]][0][1] = k_patz[ii]

    def estimate_kz_depth(self, kz0_range, kz1_range, initial_filter, numpoints=10):

        kz0 = np.linspace(kz0_range[0], kz0_range[1], numpoints)
        kz1 = np.linspace(kz1_range[0], kz1_range[1], numpoints)
        mean_depth_pat0 = np.zeros(np.shape(kz0))
        mean_depth_pat1 = np.zeros(np.shape(kz0))
        mean_error_pat0 = np.zeros(np.shape(kz0))
        mean_error_pat1 = np.zeros(np.shape(kz1))

        for i in range(len(kz0)):
            k_iter = kz0[i]
            self.set_kz([k_iter, 4.5])
            self.estimate_phases(num_phase_bins=10, iterations=5, accept_percentile=100)
            errpat0, _ = self.compute_mean_moderr(initial_filter)
            mean_error_pat0[i] = np.mean(errpat0)
            mean_depth_pat0[i] = np.mean(self.mod['depth'][self.pattern_frames[0]])
        for i in range(len(kz1)):
            k_iter = kz1[i]
            self.set_kz([ 4.5, k_iter])

            self.estimate_phases(num_phase_bins=10, iterations=5, accept_percentile=100)
            _, errpat1 = self.compute_mean_moderr(initial_filter)
            mean_error_pat1[i] = np.mean(errpat1)
            mean_depth_pat1[i] = np.mean(self.mod['depth'][self.pattern_frames[1]])
        plt.figure()
        plt.scatter(kz0, mean_error_pat0, label='data')

        p = np.polyfit(kz0, mean_error_pat0, 5)
        fig1 = plt.plot(np.linspace(min(kz0), max(kz0), 100), np.polyval(p, np.linspace(min(kz0), max(kz0), 100)), color='red',
                 linewidth=1, label='4th order polynomial')

        #plt.vlines(5.05, min(mean_depth_pat0)-0.05*max(mean_depth_pat0), max(mean_depth_pat0), 'red')
        plt.xlabel('kz0')
        plt.ylabel('Mean modulation error for direction 0 ')
        plt.title('kz = [ kz0, _ ]')
        plt.legend()
        plt.tight_layout()
        plt.show()

        minima = np.roots(np.polyder(p))
        minima = np.real(minima)
        minimumkz0 = minima[np.argmin(np.polyval(p, minima))]

        fig2 = plt.figure()
        plt.scatter(kz1, mean_error_pat1, label='data')

        p = np.polyfit(kz1, mean_error_pat1, 5)
        plt.plot(np.linspace(min(kz1), max(kz1), 100), np.polyval(p, np.linspace(min(kz1), max(kz1), 100)), color='red',
                 linewidth=1, label='4th order polynomial')

        #plt.vlines(5.05, min(mean_depth_pat1)-0.05*max(mean_depth_pat1), max(mean_depth_pat1), 'red')
        plt.xlabel('kz1')
        plt.ylabel('Mean modulation error for direction 1 ')
        plt.title('kz = [ _ , kz1]')
        plt.legend()
        plt.tight_layout()
        plt.show()

        minima = np.roots(np.polyder(p))
        minima = np.real(minima)
        minimumkz1 = minima[np.argmin(np.polyval(p, minima))]

        return minimumkz0, minimumkz1,  mean_depth_pat0, mean_depth_pat1, [fig1, fig2]

    def estimate_phases(self, num_phase_bins=10,
                          fix_phase_shifts=None,
                          fix_depths=None,
                          show_plots=False,
                          accept_percentile=30,
                          iterations=10, old=False):

        if len(self.sum_ds)<300:
            raise ValueError(f'too few spots: {len(self.sum_ds)}')

        if num_phase_bins<3:
            raise ValueError('num phase bins should be >= 3, so a spline can be fitted')

        fr = np.arange(self.sum_ds.numFrames)
        frame_bins = np.array_split(fr, num_phase_bins)
        frame_bins = [b for b in frame_bins if len(b)>0]

        k = self.mod['k'][:,:self.excDims]
        phase, depth, power = self.spotlist.estimate_phase_and_depth(k, self.pattern_frames, frame_bins,
                                                                     percentile=accept_percentile,
                                                                     iterations=iterations)
        
        phase_all, depth_all, power_all = self.spotlist.estimate_phase_and_depth(k, self.pattern_frames, [fr],
                                                                                 percentile=accept_percentile,
                                                                                 iterations=iterations)
        cm = 1 / 2.54
        # store interpolated phase for every frame
        frame_bin_t = [np.mean(b) for b in frame_bins]
        self.phase_interp = np.zeros((len(fr),self.pattern_frames.size))
        for qq in range(self.pattern_frames.size):
            phase[:,qq] = unwrap_angle(phase[:, qq])
            spl = InterpolatedUnivariateSpline(frame_bin_t, phase[:,qq], k=2)
            self.phase_interp[:,qq] = spl(fr)

        if True:
            import matplotlib
            from matplotlib import rc, font_manager
            fontProperties = {'family': 'sans-serif', 'sans-serif': ['Helvetica'],
                              'weight': 'normal', 'size': 12}
            rc('text', usetex=True)
            rc('font', **fontProperties)
            fig = plt.figure(figsize=(8*cm,6*cm))
            styles = ['o', "x", "*", 'd']
            for ax, idx in enumerate(self.pattern_frames):
                for k in range(len(idx)):
                    p=plt.plot(fr, self.phase_interp[:,idx[k]] * 180/np.pi,ls='-')
                    plt.plot(frame_bin_t, phase[:,idx[k]] * 180 / np.pi,ls='', c=p[0].get_color(), marker=styles[ax%len(styles)], label=f"Phase {idx[k]} (axis {ax})")
            plt.legend(fontsize=10)
            #plt.title(f"Phases for {self.src_fn}")
            plt.xlabel("Frame number"); plt.ylabel("Phase [deg]")
            #plt.grid()

            plt.tight_layout(pad=0.1)
            fig.savefig(self.resultprefix + "phases.png", dpi=600)
            if not show_plots: plt.close(fig)

            fig = plt.figure(figsize=(8*cm,6*cm))
            for ax, idx in enumerate(self.pattern_frames):
                for k in range(len(idx)):
                    plt.plot(frame_bin_t, depth[:, idx[k]], styles[ax%len(styles)], ls='-', label=f"Depth {idx[k]} (axis {ax})")
            plt.legend(fontsize=10)
            #plt.title(f"Depths for {self.src_fn}")
            plt.xlabel("Frame number"); plt.ylabel("Modulation Depth")
            plt.ylim(0.7,1)
            #plt.grid()
            plt.tight_layout(pad=0.1)
            fig.savefig(self.resultprefix + "depths.png", dpi=600)
            if not show_plots: plt.close(fig)

            fig = plt.figure(figsize=(8*cm,6*cm))
            for ax, idx in enumerate(self.pattern_frames):
                for k in range(len(idx)):
                    plt.plot(frame_bin_t, power[:, idx[k]], styles[ax%len(styles)], ls='-', label=f"Power {idx[k]} (axis {ax})")
            plt.legend(fontsize=10)
            #plt.title(f"Power for {self.src_fn}")
            plt.xlabel("Frame number"); plt.ylabel("Modulation Power")
            plt.ylim(0.25, 0.4)
            #plt.grid()
            plt.tight_layout(pad=0.1)
            fig.savefig(self.resultprefix + "power.png", dpi=600)
            if not show_plots: plt.close(fig)

        # Update mod
        phase_std = np.zeros(len(self.mod))
        for k in range(len(self.mod)):
            ph_k = unwrap_angle(phase[:, k])
            self.mod['phase'][k] = phase_all[0, k]
            self.mod['depth'][k] = depth_all[0, k]
            self.mod['relint'][k] = power_all[0, k]
            phase_std[k] = np.std(ph_k)

        s=np.sqrt(num_phase_bins)
        for k in range(len(self.mod)):
            self.report(f"Pattern {k}: Phase {self.mod[k]['phase']*180/np.pi:8.2f} (std={phase_std[k]/s*180/np.pi:6.2f}) "+
                   f"Depth={self.mod[k]['depth']:5.2f} (std={np.std(depth[:,k])/s:5.3f}) "+
                   f"Power={self.mod[k]['relint']:5.3f} (std={np.std(power[:,k])/s:5.5f}) ")

        #mod=self.spotlist.refine_pitch(mod, self.ctx, self.spotfilter, plot=True)[2]

        for angIndex in range(len(self.pattern_frames)):
            self.mod[self.pattern_frames[angIndex]]['relint'] = np.mean(self.mod[self.pattern_frames[angIndex]]['relint'])
            # Average modulation depth
            self.mod[self.pattern_frames[angIndex]]['depth'] = np.mean(self.mod[self.pattern_frames[angIndex]]['depth'])

        self.mod['relint'] /= np.sum(self.mod['relint'])

        if fix_depths:
            self.report(f'Fixing modulation depth to {fix_depths}' )
            self.mod['depth']=fix_depths

        self.report("Final modulation pattern parameters:")
        print_mod(self.report, self.mod, self.pattern_frames, self.pixelsize)

        with open(self.mod_fn,"wb") as f:
            pickle.dump((self.mod,self.phase_interp),f)

        med_sum_I = np.median(self.sum_ds.data.IBg[:,:,0].sum(1))
        lowest_power = np.min(self.mod['relint'])
        depth = self.mod[np.argmin(self.mod['relint'])]['depth']
        median_intensity_at_zero = med_sum_I * lowest_power * (1-depth)
        self.report(f"Median summed intensity: {med_sum_I:.1f}. Median intensity at pattern zero: {median_intensity_at_zero:.1f}")

    # def set_beam_angle(self, beamAngle):
    #     wavelen = self.cfg['wavelength']
    #     kz = 2*np.pi/wavelen *np.cos(beamAngle) * 1000 # rad/um
    #     self.mod['k'][:,2] = kz


    def draw_pattern(self):
        ...

    def pattern_plots(self, spotfilter, svg=True, **kwargs):
        self.load_mod()
        self.report(f"Generating pattern plots using spot filter: {spotfilter}. ")
        for k in range(len(self.mod)):
            png_file= f"{self.resultprefix}patternspots{k}.png"
            print(f"Generating {png_file}...")
            src_name = os.path.split(self.src_fn)[1]
            self.spotlist.draw_spots_in_pattern(png_file, self.mod,
                                       k, tiffname=src_name, spotfilter=spotfilter, **kwargs)
            if svg:
                self.spotlist.draw_spots_in_pattern(f"{self.resultprefix}patternspots{k}.svg", self.mod,
                                           k, tiffname=src_name,spotfilter=spotfilter, **kwargs)

        self.spotlist.draw_axis_intensity_spread(self.pattern_frames, self.mod, spotfilter)


    def draw_mod(self, showPlot=False):
        allmod = self.mod
        filename = self.resultprefix+'patterns.png'
        fig,axes = plt.subplots(1,2)
        fig.set_size_inches(*figsize)
        for axis in range(len(self.pattern_frames)):
            axisname = ['X', 'Y']
            ax = axes[axis]
            indices = self.pattern_frames[axis]
            freq = np.sqrt(np.sum(allmod[indices[0]]['k']**2))
            period = 2*np.pi/freq
            x = np.linspace(0, period, 200)
            sum = x*0
            for i in indices:
                mod = allmod[i]
                q = (1+mod['depth']*np.sin(x*freq-mod['phase']) )*mod['relint']
                ax.plot(x, q, label=f"Pattern {i}")
                sum += q
            ax.plot(x, sum, label=f'Summed {axisname[axis]} patterns')
            ax.legend()
            ax.set_title(f'{axisname[axis]} modulation')
            ax.set_xlabel('Pixels');ax.set_ylabel('Modulation intensity')
        fig.suptitle('Modulation patterns')
        if filename is not None: fig.savefig(filename)
        if not showPlot: plt.close(fig)
        return fig


    def plot_ffts(self):
        with Context(debugMode=self.debugMode) as ctx:
            self.spotlist.generate_projections(self.mod, 4,ctx)
            self.spotlist.plot_proj_fft()

    def load_mod(self):
        with open(self.mod_fn, "rb") as f:
            self.mod, self.phase_interp = pickle.load(f)

    def set_mod_array(self, mod):
        """
        Assign mod array and phase_interp for simulation purposes
        """
        self.mod = mod

        if self.sum_ds is not None:
            nf = self.sum_ds.numFrames
            self.phase_interp = np.zeros((nf, self.pattern_frames.size))
            for i in range(self.pattern_frames.size):
                self.phase_interp[:,i] = self.mod['phase'][i] # constant phase

    def set_mod(self, pitch_nm, angle_deg, depth, z_angle=None):
        """
        Assign mod array and phase_interp for simulation purposes
        """
        freq = 2*np.pi/np.array(pitch_nm)*self.pixelsize
        angle = np.deg2rad(angle_deg)
        mod = np.zeros(self.pattern_frames.size, dtype=ModDType)
        if z_angle is None:
            z_angle = angle*0
        else:
            z_angle = np.deg2rad(z_angle)
        for i,pf in enumerate(self.pattern_frames):
            mod['k'][pf,0] = np.cos(angle[i]) * freq[i] * np.cos(z_angle[i])
            mod['k'][pf,1] = np.sin(angle[i]) * freq[i] * np.cos(z_angle[i])
            mod['k'][pf,2] = freq[i] * np.sin(z_angle[i])
            mod['phase'][pf] = np.linspace(0,2*np.pi,len(pf),endpoint=False)

        mod['depth'] = depth
        mod['relint'] = 1/self.pattern_frames.size
        self.set_mod_array(mod)


    def mod_at_frame(self, framenums):
        """
        Return modulation patterns with spline interpolated phases
        """
        print(framenums.shape)

        mod_ = np.zeros((len(framenums), self.pattern_frames.size), dtype=ModDType)
        mod_[:] = self.mod[None]

        for k in range(len(self.mod)):
            mod_['phase'][:,k] = self.phase_interp[framenums][:,k]

        return mod_
        #return np.reshape(mod_.view(np.float32), (len(framenums), 6*len(self.mod)))

    def create_psf(self, ctx, modulated=False):
        psf_calib = self.psf_calib
        if type(psf_calib) == str:
            psf_calib = Gauss3D_Calibration.from_file(psf_calib)



        if modulated:
            # This will return a 3D astigmatic PSF if psf_calib is a Gauss2D_Calibration
            psf = SIMFLUX(ctx).CreateEstimator_Gauss2D(psf_calib,len(self.mod),
                                                        self.roisize,len(self.mod),
                                                        simfluxEstim=True)

            return psf
        else:
            if isinstance(psf_calib, Gauss3D_Calibration):
                psf = Gaussian(ctx).CreatePSF_XYZIBg(self.roisize, psf_calib, True)
            else:
                psf = Gaussian(ctx).CreatePSF_XYIBg(self.roisize, psf_calib, True)


            return psf

    def _intensity_model(self, theta, mod):
        dims = theta.shape[1]-1
        intensity = theta[:,-1]
        pos = theta[:,:dims]
        
        k = mod['k']
        phaseshift = mod['phase']
        depth = mod['depth']
        relint = mod['relint']

        p = (pos[:,None] * k[:,:,:dims]).sum(2) - phaseshift
        deriv_I = relint * ( 1 + depth * np.sin(p) )
        expectedIntensity = intensity[:,None] * deriv_I
        deriv_x = intensity[:,None] * relint * k[:,:,0] * depth * np.cos(p)
        deriv_y = intensity[:,None] * relint * k[:,:,1] * depth * np.cos(p)
        deriv_z = intensity[:,None] * relint * k[:,:,2] * depth * np.cos(p)
        return expectedIntensity, np.array([deriv_x, deriv_y, deriv_z, deriv_I])


    def compute_moderr(self):
        # g2d_results are the same set of spots used for silm, for fair comparison
        mod_ = self.mod_at_frame(self.sum_ds.frame)

        ds = self.sum_ds
        dims = np.size(ds.pos, 1)
        theta = np.zeros((len(ds), 4))
        theta[:,:dims] = ds.pos[:,:dims]
        theta[:,dims] = ds.photons

        expected_intensities = self._intensity_model(theta, mod_)[0]
        err = np.abs(expected_intensities - ds.IBg[:,:,0]) / expected_intensities.sum(1)[:,None]
        return np.max(err, 1)

    def compute_mean_moderr(self, initial_filter):
        # g2d_results are the same set of spots used for silm, for fair comparison
        mod_ = self.mod_at_frame(self.sum_ds.frame)

        ds = self.sum_ds
        dims = np.size(ds.pos, 1)
        theta = np.zeros((len(ds), 4))
        theta[:,:dims] = ds.pos[:,:dims]
        filter = initial_filter
        # filter = ds.photons > 10000
        # filter = np.logical_and(filter, initial_filter)
        theta[:,dims] = ds.photons
        theta = theta[filter]
        patterns = self.pattern_frames
        expected_intensities = self._intensity_model(theta, mod_[filter])[0]
        IBg = ds.IBg[filter]
        errpat0 = np.abs(expected_intensities[:,patterns[0]] - IBg[:,patterns[0],0]) / expected_intensities[:,patterns[0]].sum(1)[:,None]
        errpat1 = np.abs(expected_intensities[:, patterns[1]] - IBg[:, patterns[1], 0]) / expected_intensities[:,patterns[1]].sum(1)[:,None]
        #err = np.abs(expected_intensities - ds.IBg[:,:,0]) / expected_intensities.sum(1)[:,None]
        return np.mean(errpat0, 1), np.mean(errpat1,1)

    def filter_spots(self, moderror_threshold):
        moderr = self.compute_moderr()
        return np.nonzero(moderr < moderror_threshold)[0], moderr

    def process(self, fn_idx=0, maxiter= 500, filter_spots = True, zrange=np.inf):
        """
        Perform SIMFLUX localization
        """

        with Context(debugMode=self.debugMode) as ctx:

            #if len(self.pattern_frames)==2: # assume XY modulation
            #    self.draw_mod()

            indices, moderr = self.filter_spots(self.moderror_threshold)


            print(f"# filtered spots left: {len(indices)}. median moderr:{np.median(moderr):.2f}")

            mod_ = self.mod_at_frame(self.sum_ds.frame)

            psf = self.create_psf(ctx, modulated=True)
            try:
                params = self.levmarparams_sf
            except:
                raise NameError('Define sp.levmarparams_astig')

            psf.SetLevMarParams(params, maxiter, normalizeWeights=True)
            dims = psf.numparams-2

            print(f"Constants per ROI: {psf.numconst}")
            
            indices_org = indices*1
            testmask = np.zeros(len(self.sum_ds),dtype=np.bool)

            with EstimQueue(psf, batchSize=2048, numStreams=1, keepSamples=False) as queue:
                idx = 0
                for roipos, pixels, block_indices in self.selected_roi_source(indices):

                    roi_mod = mod_[block_indices]
#
                    roi_mod_float = np.reshape(roi_mod.view(np.float32), (len(roi_mod), 6*len(self.mod)))

                    initial = np.zeros((len(block_indices), psf.numparams))
                    initial[:,:dims] = self.sum_ds.local_pos[block_indices]
                    initial[:,-1] = self.sum_ds.background[block_indices]/len(self.mod)
                    initial[:,-2] = self.sum_ds.photons[block_indices]
                    
                    testmask[block_indices] = True

                    queue.Schedule(pixels, roipos=roipos,initial=initial,
                                   constants=roi_mod_float, ids=np.arange(len(roipos))+idx)
                    idx += len(roipos)

                queue.WaitUntilDone()
                qr = queue.GetResults()
                
            assert testmask.sum() == len(indices_org)

            qr.SortByID(isUnique=True)


            qr.ids = self.sum_ds.frame[indices]
            ds = Dataset.fromQueueResults(qr, self.imgshape)

            if filter_spots:
                print(f'Filtering on iterations: { (ds.iterations>maxiter).sum() }/{len(ds)} spots removed. Max iterations={maxiter}')

                indices = indices[ds.iterations < maxiter]

                ds = ds[ds.iterations < maxiter]
                # indices = indices[ds.photons > 2000]
                # ds = ds[ds.photons > 2000]
                border = 2.1
                sel = ((ds.local_pos[:,0] > border) & (ds.local_pos[:,0] < self.roisize-border-1) &
                    (ds.local_pos[:,1] > border) & (ds.local_pos[:,1] < self.roisize-border-1) &
                       (ds.local_pos[:,2] > -zrange) & (ds.local_pos[:,2] < zrange))

                print(f'Filtering on position in ROI: {len(ds)-sel.sum()}/{len(ds)} spots removed.')
                ds = ds[sel]
                indices = indices[sel]
            
            self.sf_ds = ds
            self.sum_ds_filtered = self.sum_ds[indices]

            self.sf_ds.save(self.resultprefix+"simflux"+str(fn_idx)+".hdf5")
            self.sum_ds_filtered.save(self.resultprefix+"g2d-filtered"+str(fn_idx)+".hdf5")


    def simflux_estimator_vector(self):

        """
        Perform SIMFLUX localization
        """
        from photonpy.simflux.torch_mle import gauss_psf_2D_astig, lm_mle, ModulatedIntensitiesModel, \
            Gaussian2DAstigmaticPSF, SIMFLUXModel, SIMFLUXModel_vector
        import photonpy.smlm.process_movie as process_movie
        import torch
        sys.path.insert(1, 'C:/Development/simflux3d/projects/vectorpsf')

        import torch
        from vectorize_torch import LM_MLE

        from torch import Tensor

        with Context(debugMode=False) as ctx:
            # if len(self.pattern_frames)==2: # assume XY modulation
            #    self.draw_mod()

            indices, moderr = self.filter_spots(self.moderror_threshold)

            numblocks = np.ceil(len(indices) / 100)
            block_indices = np.array_split(indices, numblocks)
            print(f"# filtered spots left: {len(indices)}. median moderr:{np.median(moderr):.2f}")
            estimation = np.empty((0, 5))

            mod = self.mod_at_frame(self.sum_ds.frame)

            print('\n Perform Simflux localization via pytorch... ')

            for blocks in tqdm.trange(1):  # ))int(numblocks)): # int(numblocks
                indices_single = block_indices[blocks]
                mod_single = mod[indices_single, :]
                mod_torch = np.zeros([len(indices_single), 8, 6])

                for i in range(np.size(mod_torch, 0)):
                    for j in range(8):
                        mod_torch[i, j, 0:3] = mod_single[i, j][0]
                        mod_torch[i, j, 3] = mod_single[i, j][1]
                        mod_torch[i, j, 4] = mod_single[i, j][2]
                        mod_torch[i, j, 5] = mod_single[i, j][3]
                roi_idx = self.sum_ds.roi_id[indices_single]

                roi_info, pixels = process_movie.load_rois(self.rois_fn)
                pixels = pixels[roi_idx]
                roi_pos = np.vstack((roi_info['x'], roi_info['y'])).T
                roi_pos = roi_pos[roi_idx, :]
                with torch.no_grad():
                    dev = torch.device('cuda')

                    smp_ = torch.from_numpy(pixels).to(dev)  # summed for modulated model
                    mod_ = torch.from_numpy(np.asarray(mod_torch)).to(dev)

                    initial = np.zeros([len(indices_single), 5])
                    dims = 3
                    initial[:, :dims] = self.sum_ds.local_pos[indices_single]
                    initial[:, -1] = self.sum_ds.background[
                                         indices_single] / 8  # divide by numpat?????????????????????????????????????
                    if only_illu:
                        initial[:, -1] = initial[:, -1] * 16 * 16
                    initial[:, -2] = self.sum_ds.photons[indices_single]

                    initial_ = torch.from_numpy(initial).to(dev)

                    psf_model = Gaussian2DAstigmaticPSF(self.roisize, [calib.x, calib.y])
                    roi_pos_ = torch.from_numpy(roi_pos).to(dev)



    def process_torch(self, calib, fn_idx=0, only_illu= False, lamda = 1, vector=True, pick_percentage=1, pattern = 0):
        """
        Perform SIMFLUX localization
        """
        from photonpy.simflux.torch_mle import gauss_psf_2D_astig, lm_mle, ModulatedIntensitiesModel, Gaussian2DAstigmaticPSF, SIMFLUXModel, SIMFLUXModel_vector
        import photonpy.smlm.process_movie as process_movie
        import torch
        sys.path.insert(1, 'C:/Development/simflux3d/projects/vectorpsf')

        import torch
        from vectorize_torch import LM_MLE_simflux

        from torch import Tensor

        with Context(debugMode=False) as ctx:
            # if len(self.pattern_frames)==2: # assume XY modulation
            #    self.draw_mod()

            indices, moderr = self.filter_spots(self.moderror_threshold)
            select = np.random.choice(np.arange(indices.size),size=int(len(indices)*pick_percentage), replace=False)
            indices = indices[select]
            moderr = moderr[select]
            numblocks = np.ceil(len(indices) / 80000)
            block_indices = np.array_split(indices, numblocks)
            print(f"# filtered spots left: {len(indices)}. median moderr:{np.median(moderr):.2f}")
            estimation = np.empty((0,5))

            mod = self.mod_at_frame(self.sum_ds.frame)

            print('\n Perform Simflux localization via pytorch... ')

            for blocks in tqdm.trange(int(numblocks)): # int(numblocks
                indices_single = block_indices[blocks]
                mod_single = mod[indices_single,:]
                mod_torch  =np.zeros([len(indices_single), 3,6])


                for i in range(np.size(mod_torch,0)):
                    for j in range(3):
                        mod_torch[i,j,0:3] = mod_single[i,j][0]
                        mod_torch[i, j, 3] = mod_single[i, j][1]
                        mod_torch[i, j, 4] = mod_single[i, j][2]
                        mod_torch[i, j, 5] = mod_single[i, j][3]



                roi_idx = self.sum_ds.roi_id[indices_single]

                roi_info, pixels = process_movie.load_rois(self.rois_fn)
                pixels = pixels[roi_idx]
                roi_pos = np.vstack((roi_info['x'],roi_info['y'])).T
                roi_pos = roi_pos[roi_idx,:]
                if pattern == 1:
                    mod_torch = mod_torch[:, [0, 2, 4], :]
                    pixels = pixels[:,[0, 2, 4],:]
                if pattern == 2:
                    mod_torch = mod_torch[:, [1, 3, 5], :]
                    pixels = pixels[:, [1, 3, 5], :]
                with torch.no_grad():
                    dev = torch.device('cuda')

                    smp_ = torch.from_numpy(pixels).to(dev) # summed for modulated model
                    mod_ = torch.from_numpy(np.asarray(mod_torch)).to(dev)

                    initial = np.zeros([len(indices_single), 5])
                    dims = 3
                    initial[:,:dims] = self.sum_ds.local_pos[indices_single]
                    initial[:,-1] = self.sum_ds.background[indices_single] # divide by numpat?????????????????????????????????????
                    if only_illu:
                        initial[:, -1] =  initial[:, -1]*self.roisize
                    initial[:,-2] = self.sum_ds.photons[indices_single]

                    initial_ = torch.from_numpy(initial).to(dev)

                    psf_model = Gaussian2DAstigmaticPSF(self.roisize, [calib.x, calib.y])
                    roi_pos_ = torch.from_numpy(roi_pos).to(dev)
                    if only_illu:

                        sf_model_ = ModulatedIntensitiesModel(roi_pos_,bg_factor=self.roisize**2)
                    elif vector:
                        sf_model_ = SIMFLUXModel_vector(psf_model, roi_pos_, divide_bg=False, roisize=self.roisize)
                    else:
                        sf_model_ = SIMFLUXModel(psf_model, roi_pos_, divide_bg=False)
                    param_range = np.array([
                        [2, self.roisize - 3],
                        [2, self.roisize - 3],
                        [-2, 2],
                        [1, 1e9],
                        [0.5, 1e5],
                    ])
                    param_range = torch.tensor(param_range).to(dev)
                    if vector:

                        mle = LM_MLE_simflux(sf_model_, lambda_=lamda, iterations=40,
                                     param_range_min_max=param_range, tol=1e-3)
                        #mle = torch.jit.script(mle)

                        e, traces_, _ = mle.forward(smp_,initial_, dev,mod_, False)
                        traces = traces_.cpu().numpy()

                        estim = e.cpu().numpy()
                        estimation = np.append(estimation, estim, axis=0)
                    else:
                        #sf_model= lambda x: sf_model_.forward(x,roi_pos_, mod_)
                        #sf_model = lambda x: sf_model_.forward(x, mod_)


                        if only_illu:
                            estim_, traces_ = lm_mle(sf_model_, initial_, (smp_.sum(axis=-1)).sum(axis=-1), param_range,roi_pos_,mod_,
                                                     iterations=50, lambda_=lamda,
                                                     store_traces=True)

                        else:

                            estim_, traces_ = lm_mle(sf_model, initial_, smp_, param_range, iterations=200, lambda_=lamda,
                                                     store_traces=True)
                        traces = traces_.cpu().numpy()
                        #estim_[:,[0,1]] = estim_[:,[1,0]]
                        estim = estim_.cpu().numpy()
                        estimation = np.append(estimation ,estim, axis=0)


            #filter crap
            # border = 2.1
            # sel = ((ds.local_pos[:, 0] > border) & (ds.local_pos[:, 0] < self.roisize - border - 1) &
            #        (ds.local_pos[:, 1] > border) & (ds.local_pos[:, 1] < self.roisize - border - 1) &
            #        (ds.local_pos[:, 2] > -0.5) & (ds.local_pos[:, 2] < 0.5))
            #
            # print(f'Filtering on position in ROI: {len(ds) - sel.sum()}/{len(ds)} spots removed.')
            # ds = ds[sel]
            # indices = indices[sel]

            import copy

            self.sum_ds_filtered = self.sum_ds[indices]

            #
            ds = Dataset.fromEstimates(estimation, ['x', 'y', 'z', 'I', 'bg'], self.sum_ds.frame[indices],
                                       self.imgshape,
                                       roipos=self.sum_ds.data['roipos'][indices],
                                       chisq=np.zeros(np.shape(self.sum_ds.frame[indices])))


            ds = copy.deepcopy(self.sum_ds_filtered)
            pos = estimation*1
            roipos_add= self.sum_ds.data['roipos'][indices]
            pos[:, 0:2] += roipos_add[:,[-1,-2]]
            ds.pos = pos[:,:3]
            ds.photons = estimation[:,3]
            ds.background = estimation[:,4]


            self.sf_ds = ds
            self.sf_ds.save(self.resultprefix + "simflux"  + ".hdf5")
            self.sum_ds_filtered.save(self.resultprefix + "g2d-filtered"  + ".hdf5")
            return traces

    def plot_sine(self, pat=0, spot=0, itnum=0):
        import scipy
        def sinfunc(t, A, p, c):
            return A * np.sin(-2*np.pi/8 * t + p) + c

        i = pat
        pattern = self.pattern_frames[i]
        x = np.arange(len(pattern))

        y = self.sum_ds.IBg_filtered[:, pattern, 0]/np.array([np.max(self.sum_ds.IBg_filtered[:,pattern,0], axis=1)]).T
        plt.figure(1)


        guess = np.array([np.std(y, axis=1) * 2. ** 0.5, np.ones(np.size(y, 0))*np.pi,
                          np.mean(y, axis=1)])
        popt_mat = np.zeros([np.size(y, 0), 3])

        j = spot
        plt.scatter(x, self.sum_ds.IBg_filtered[j, pattern, 0]/np.max(self.sum_ds.IBg_filtered[j, pattern, 0]))
        try:
            popt_mat[j, :], pcov = scipy.optimize.curve_fit(sinfunc, np.arange(len(pattern)),
                                                            self.sum_ds.IBg_filtered[j, pattern, 0]/np.max(self.sum_ds.IBg_filtered[j, pattern, 0]),
                                                            method='trf', p0=guess[:, j], bounds=  ([0, 0, -np.inf], [np.inf, 2*np.pi, np.inf]))
            plt.plot(np.linspace(0,len(pattern),100), sinfunc(np.linspace(0,len(pattern),100), *popt_mat[j,:]), alpha=0.5)
            chi,p = scipy.stats.chisquare(y[j, :], sinfunc(x, *popt_mat[j, :]))
            #chi = np.sum((y[j, :] - sinfunc(x, *popt_mat[j, :])) ** 2 / sinfunc(x, *popt_mat[j, :]))
            #plt.title('R^2 = ' + str(chi) + ', p =  ' + str(p))
            plt.xlabel('Pattern')
            plt.ylabel('Intensity [photons]')
        except:
            print('no fit possible')




    def fit_sine(self,lateral=False, display=False):
        if lateral:
            def sinfunc(t, freq, A, p, c,slope):  return A * np.sin(-2 * np.pi*freq *t+ p) + c + slope*t
        else:
            def sinfunc(t, A, p, c):  return A * np.sin(2*np.pi/len(self.pattern_frames) * t + p) + c
        import scipy
        chi_test_mat = np.zeros([np.size(self.sum_ds.IBg, 0)])
        modulation_mat = np.zeros([np.size(self.sum_ds.IBg, 0)])
        p = np.zeros([np.size(self.sum_ds.IBg, 0)])


        if lateral:
            pattern = self.pattern_frames[0]
            x = np.arange(len(pattern[0]))
        else:
            pattern = self.pattern_frames
            x = np.arange(len(pattern))

        y = self.sum_ds.IBg[:,:,0]/np.array([np.max(self.sum_ds.IBg[:,:,0], axis=1)]).T


        guess = np.array([np.std(y, axis=1) * 2. ** 0.5, np.ones(np.size(y,0))*np.pi, np.mean(y,axis=1)])
        bounds = ([0, 0, -np.inf], [np.inf, 2 * np.pi, np.inf])
        if lateral:
            guess = np.array([np.ones(np.size(y,0))*1/10,np.std(y, axis=1) * 2. ** 0.5, np.zeros(np.size(y,0)), np.mean(y, axis=1), np.zeros(np.size(y,0))])
            bounds = ([-np.inf,0, -2*np.pi, -np.inf,-1], [np.inf,np.inf, 2 * np.pi, np.inf,1])
        popt_mat = np.zeros([np.size(self.sum_ds.IBg, 0), np.size(guess,0)])
        for j in range(np.size(y,0)):
            try:
                popt_mat[j,:], pcov = scipy.optimize.curve_fit(sinfunc, x, y[j,:],
                                              method='trf', p0=guess[:,j], bounds= bounds)

                #chi_test_mat[j,i] = np.sum((y[j,:] - sinfunc(x, *popt_mat[j,:])) ** 2 / sinfunc(x, *popt_mat[j,:]))
                chi_test_mat[j], p[j] = scipy.stats.chisquare(y[j, :], sinfunc(x, *popt_mat[j, :]))
                modulation_mat[j] = (max(y[j,:]) - min(y[j,:])) / (max(y[j,:]) + min(y[j,:]))
            except:
                chi_test_mat[j] = np.nan
                modulation_mat[j] = np.nan

            if display:
                cm = 1 / 2.54

                from matplotlib import rc, font_manager
                fontProperties = {'family': 'sans-serif', 'sans-serif': ['Helvetica'],
                                  'weight': 'normal', 'size': 12}
                rc('text', usetex=True)
                rc('font', **fontProperties)
                fig, ax = plt.subplots(figsize=(8 * cm, 5 * cm))
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.set_prop_cycle(color=[
                    '#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e'
                ])
                # plt.rcParams['text.usetex'] = True

                plt.scatter(x*64, y[j], label='Data')
                plt.plot(np.linspace(0,x[-1],100)*64,sinfunc(np.linspace(0,x[-1],100),*popt_mat[j,:]),label='Sinusoidal fit')

                xlabel = 'Distance [nm]'
                ylabel = 'Intensity [au]'
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # ax.tick_params(axis='both', which='major', labelsize=10)
                # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
                # stylize_axes(ax)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                fontProperties = {'family': 'sans-serif', 'sans-serif': ['Helvetica'],
                                  'weight': 'normal', 'size': 10}
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), frameon=False)

                #ax.set_xticks(np.arange(1.0, 9.0, 1))
                rc('font', **fontProperties)
                #fig.tight_layout()

                # plt.savefig('C:/Users/Gebruiker/Desktop/inkscape/fig_lateral_pitch/sinewave.png', bbox_inches='tight',
                #             pad_inches=0.01, dpi=600)

                plt.show()

        self.sum_ds.chi = chi_test_mat
        self.sum_ds.params_sine = popt_mat
        self.sum_ds.pvalues = p
        self.sum_ds.modulation = modulation_mat



        #self.filter_onR()
        return chi_test_mat, p, modulation_mat

    def filter_onR(self, filter = 0.05, min_mod = 0.7, display = False):


        maxp = self.sum_ds.pvalues


        pindices = maxp < (1-filter)
        print('filtered '+ str(int(sum(pindices))) + '/'+str(int(len(pindices))) + ' based on p value'  )
        sorted_indc = np.argsort(-1*maxp)
        test = pindices[sorted_indc]

        best_denied = sorted_indc[test]
        maxmod = self.sum_ds.modulation
        mod_ind = maxmod<min_mod
        print('filtered ' + str(int(sum(mod_ind))) + '/' + str(int(len(mod_ind))) + ' based on modulation')
        if display:
            for i in range(np.min([5,len(best_denied)])):
                pattern = np.argmin(self.sum_ds.pvalues[best_denied[i], :])
                self.plot_sine(pattern,best_denied[i], i)
        print('filtered ' + str(int(sum(np.logical_or(mod_ind,pindices)))) + '/' + str(int(len(mod_ind))) + ' in total')
        self.remove_indices(np.logical_or(mod_ind,pindices))

    def selected_roi_source(self, indices):
        """
        Yields roipos,pixels,idx for the selected ROIs.
        'indices' indexes into the set of ROIs selected earlier by gaussian_fitting(), stored in roi_indices
        idx is the set of indices in the block, indexing into self.sum_ds
        """
        roi_idx = self.sum_ds.roi_id[indices]

        if self.numrois is None:
            self.numrois = int(np.sum([len(ri) for ri,px in self._load_rois_iterator()]))

        # The mask is to quickly select the required ROIs from the block that is currently loaded
        mask = np.zeros(self.numrois, dtype=np.bool)
        mask[roi_idx] = True

        idx = 0

        # The index map is to find the index of the ROIs in the sum_ds dataset
        indexmap = np.zeros(self.numrois,dtype=np.int32)
        indexmap[self.sum_ds.roi_id[indices]] = indices

        for rois_info, pixels in self._load_rois_iterator():
            block_mask = mask[idx:idx+len(pixels)]
            block_roi_indices = indexmap[idx:idx+len(pixels)][block_mask]
            idx += len(pixels)

            if np.sum(block_mask) > 0:
                roipos = np.zeros((len(rois_info),3), dtype=np.int32)
                roipos[:,0] = 0
                roipos[:,1] = rois_info['y']
                roipos[:,2] = rois_info['x']

                yield roipos[block_mask], pixels[block_mask], block_roi_indices


    def crlb_map(self, intensity=None, bg=None, sample_area_width=0.2):
        """

        """
        if intensity is None:
            intensity = np.median(self.sum_ds.photons)

        if bg is None:
            bg = np.median(self.sum_ds.background)

        #pitchx = 2*np.pi / np.max(np.abs(self.mod['k'][:,0]))
        #pitchy = 2*np.pi / np.max(np.abs(self.mod['k'][:,1]))

        W = 100
        xr = np.linspace((0.5-sample_area_width/2)*self.roisize,(0.5+sample_area_width/2)*self.roisize,W)
        yr = np.linspace((0.5-sample_area_width/2)*self.roisize,(0.5+sample_area_width/2)*self.roisize,W)

        X,Y = np.meshgrid(xr,yr)

        with Context(debugMode=self.debugMode) as ctx:
            sf_psf = self.create_psf(ctx, modulated=True)

            coords = np.zeros((W*W,sf_psf.numparams))
            coords[:,0] = X.flatten()
            coords[:,1] = Y.flatten()
            coords[:,-2] = intensity
            coords[:,-1] = bg

            coords_ = coords*1
            coords_[:,-1] /= sf_psf.sampleshape[0] # bg should be distributed over all frames
            mod_ = np.repeat([self.mod.view(np.float32)], len(coords), 0)
            sf_crlb = sf_psf.CRLB(coords_, constants=mod_)

            psf = self.create_psf(ctx, modulated=False)
            g2d_crlb = psf.CRLB(coords)

        IFmap = g2d_crlb/sf_crlb

        fig,ax = plt.subplots(2,1,sharey=True)
        im = ax[0].imshow(IFmap[:,0].reshape((W,W)))
        ax[0].set_title('Improvement Factor X')

        ax[1].imshow(IFmap[:,1].reshape((W,W)))
        ax[1].set_title('Improvement Factor Y')

        fig.colorbar(im, ax=ax)

        IF = np.mean(g2d_crlb/sf_crlb,0)
        print(f"SF CRLB: {np.mean(sf_crlb,0)}")
        print(f"SMLM CRLB: {np.mean(g2d_crlb,0)}")

        print(f"Improvement factor X: {IF[0]:.3f}, Y: {IF[1]:.3f}")


    def modulation_error(self, spotfilter):
        self.load_mod()
        return self.spotlist.compute_modulation_error(self.mod)

    def draw_projections(self):
        for ep in tqdm.trange(len(self.mod), desc='Generating projections'):
            self.draw_projection(ep)
    def draw_projection(self, ep):
            if ep%2 == 0:
                kx = 0.5040475
                ky = 0.44295067
            else:
                kx = 0.50422263
                ky = -0.44322073

            ds = self.sum_ds

            proj = ds.pos_filtered[:, 0] * (kx) + ds.pos_filtered[:, 1] * (ky)
            projkxy = proj % (np.sqrt(kx ** 2 + ky ** 2))

            projz = ds.pos_filtered[:, 2]
            normI = ds.IBg_filtered[:, ep, 0] / ds.photons_filtered


            binxinp = np.linspace(min(projz), max(projz), 50).tolist()
            binyinp = np.linspace(min(projkxy), max(projkxy), 50).tolist()

            #clustergrid = [binxinp, binyinp]
            clustergrid = (50, 50)

            import scipy
            medians, binx, biny, binnr = scipy.stats.binned_statistic_2d(projkxy, projz, normI, bins=clustergrid,
                                                                         statistic='median')

            clustergrid = (50, 50)
            medians_non, binx_non, biny_non, binnr = scipy.stats.binned_statistic_2d(proj, projz, normI, bins=clustergrid,
                                                                         statistic='median')
            X, Y = np.meshgrid((binx[1:] + binx[:-1]) / 2, (biny[1:] + biny[:-1]) / 2)
            X_non, Y_non = np.meshgrid((binx_non[1:] + binx_non[:-1]) / 2, (biny_non[1:] + biny_non[:-1]) / 2)

            plt.figure()
            plt.pcolormesh(X, Y, np.clip(medians,0,1))
            plt.colorbar(label='Normalized intensity Ipat/Itot')
            plt.title('pattern ' + str(ep))
            plt.xlabel('Projection on kxy')
            plt.ylabel('Z (um)')
            plt.savefig(self.resultprefix + f"projection, scaled to 1kxy{ep}.png")

            plt.figure()
            plt.pcolormesh(X_non, Y_non, np.clip(medians_non,0,1))
            plt.colorbar(label='Normalized intensity Ipat/Itot')
            plt.title('pattern ' + str(ep))
            plt.xlabel('Projection on kxy')
            plt.ylabel('Z (um)')
            plt.savefig(self.resultprefix + f"projectionpat{ep}.png")

    def draw_patterns(self, dims, only_dim3=False):
        for ep in tqdm.trange(len(self.mod), desc='Generating modulation pattern plots'):
            self.draw_pattern(ep, dims, only_dim3)

    def draw_pattern(self, ep, dims, only_dim3=False):
        import matplotlib.pyplot as plt
        import seaborn as sns
        # normalize
        if only_dim3 == False:
            ds = self.sum_ds

            mod = self.mod[ep]
            k = mod['k']
            k = k[:dims]

            sel = np.arange(len(ds))
            numpts = 5000

            np.random.seed(0)
            indices = np.arange(len(sel))
            np.random.shuffle(indices)
            sel = sel[:np.minimum(numpts, len(indices))]

            accepted = self.compute_moderr()[sel] < self.moderror_threshold
            rejected = np.logical_not(accepted)
            # plt.figure(1)
            # plt.hist(ds.photons[rejected],alpha=0.5 ,label='rejected')
            # plt.hist(ds.photons[accepted],alpha=0.5 , label='accepted')
            # plt.legend()
            # plt.show()
            # Correct for phase drift
            spot_phase = self.phase_interp[ds.frame[sel]][:, ep]
            spot_phase -= np.mean(spot_phase)
            #spot_phase = np.zeros(np.shape(spot_phase))
            normI = ds.IBg[sel][:, ep, 0] / ds.photons[sel]

            proj = (k[None] * ds.pos[sel][:, :dims]).sum(1) - spot_phase

            x = proj % (np.pi * 2)

            cm = 1 / 2.54
            import matplotlib
            from matplotlib import rc, font_manager
            fontProperties = {'family': 'sans-serif', 'sans-serif': ['Helvetica'],
                              'weight': 'normal', 'size': 12}
            rc('text', usetex=True)
            rc('font', **fontProperties)
            plt.figure(figsize=(8 * cm, 6 * cm))
            plt.scatter(x[accepted], normI[accepted], marker='.', c='b', label='Accepted',s=5)
            plt.scatter(x[rejected], normI[rejected], marker='.', c='r', label='Rejected',s=5)

            sigx = np.linspace(0, 2 * np.pi, 400)
            exc = mod['relint'] * (1 + mod['depth'] * np.sin(sigx - mod['phase']))
            plt.plot(sigx, exc, 'r', linewidth=4, label='Estimated P')

            plt.ylim([-0.01, 0.8])
            plt.xlabel('Phase [radians]')
            plt.ylabel(r'Normalized intensity ($I_k$)')
            lenk = np.sqrt(np.sum(k ** 2))
            #plt.title(f': Pattern {ep}. K={lenk:.4f} ' + f" Phase={self.mod[ep]['phase'] * 180 / np.pi:.3f}).")
            #plt.colorbar()
            plt.legend(fontsize=10)
            plt.tight_layout(pad=0.1)
            plt.savefig(self.resultprefix + f"pattern{ep}.png",dpi=600)
        #return ds.photons[rejected], ds.roi_id[rejected], ds.local_pos[rejected], ds.frame[rejected]

    def continuous_frc(self,maxdistance, freq=10):
        if self.g_undrifted is not None:
            g_data = self.g_undrifted
            sf_data = self.sf_undrifted
            print('Running continuous FRC on drift-corrected data')
        else:
            g_data = self.sum_ds_filtered
            sf_data = self.sf_ds

        if np.isscalar(freq):
            freq = np.linspace(0,freq,200)
        sys.stdout.write(f'Computing continuous FRC for gaussian fits...')
        frc_g2d,val_g2d = self._cfrc(maxdistance,g_data.get_xy(),freq)
        print(f"{self.pixelsize/val_g2d:.1f} nm")
        sys.stdout.write(f'Computing continuous FRC for modulation enhanced fits...')
        frc_sf,val_sf = self._cfrc(maxdistance,sf_data.get_xy(),freq)
        print(f"{self.pixelsize/val_sf:.1f} nm")

        plt.figure()
        plt.plot(freq, frc_g2d, label=f'2D Gaussian (FRC={self.pixelsize/val_g2d:.1f} nm)')
        plt.plot(freq, frc_sf, label=f'Modulated (FRC={self.pixelsize/val_sf:.1f} nm)')
        plt.xlabel('Spatial Frequency (pixel^-1)')
        plt.ylabel('FRC')
        IF =  val_sf / val_g2d
        plt.title(f'Continuous FRC with localization pairs up to {maxdistance*self.pixelsize:.1f} nm distance. Improvement={IF:.2f}')
        plt.legend()
        plt.savefig(self.resultsdir+"continuous-frc.png")
        print(f"Improvement: {IF}")
        return IF

    def _cfrc(self, maxdist,xy,freq):
        with Context(debugMode=self.debugMode) as ctx:
            frc=PostProcessMethods(ctx).ContinuousFRC(xy, maxdist, freq, 0,0)

        c = np.where(frc<1/7)[0]
        v = freq[c[0]] if len(c)>0 else freq[0]
        return frc, v

    def _load_rois_iterator(self, alt_load=False):
            return process_movie.load_rois_iterator(self.rois_fn)

    def _summed_rois_iterator(self):
        for info, pixels in process_movie.load_rois_iterator(self.rois_fn):
            yield info, pixels.sum(1)

    def report(self, msg):
        with open(self.reportfile,"a") as f:
            f.write(msg+"\n")
        print(msg)

    def simulate(self, nframes, bg, gt: Dataset, output_fn, blink_params=None,
                 em_drift=None, exc_drift=None, debugMode=False):
        """
        Simulate SMLM dataset with excitation patterns. Very basic atm,
        spots are either off or on during the entire series of modulation patterns (6 frames for simflux)
        """
        with Context(debugMode=debugMode) as ctx, self.create_psf(ctx, True) as psf,  \
              multipart_tiff.MultipartTiffSaver(output_fn) as tif, tqdm.tqdm(total=nframes) as pb:

            sampleframes = psf.sampleshape[0] if len(psf.sampleshape)==3 else 1

            for f, spot_ontimes in enumerate(blinking_spots.blinking(len(gt), nframes // sampleframes,
                                    **(blink_params if blink_params is not None else {}), subframe_blink=1)):


                which = np.where(spot_ontimes > 0)[0]

                params = np.zeros((len(which), gt.dims+2),dtype=np.float32)
                params[:,:gt.dims] = gt.pos[which]
                params[:,-2] = gt.photons[which]

                roiposYX = (params[:,[1,0]] - psf.sampleshape[-2:]/2).astype(np.int32)
                params[:,:2] -= roiposYX[:,[1,0]]

                roipos = np.zeros((len(which),3),dtype=np.int32)
                roipos[:,1:] = roiposYX

                # this way the modulation pattern will automatically be broadcasted
                # into the required (spots, npatterns, 6) shape
                rois = psf.ExpectedValue(params, roipos=roipos, 
                                         constants=self.mod.view(np.float32)) 
                
                                
                for i in range(sampleframes):
                    img = np.zeros(gt.imgshape, dtype=np.float32)
                    ctx.smlm.DrawROIs(img, rois[:,i], roiposYX)
                    img += bg
                    tif.save(np.random.poisson(img))
                    pb.update()


    def drift_correct(self, **kwargs):
        self.drift, est_ = self.sum_ds.estimateDriftMinEntropy(
            coarseSigma=self.sum_ds.crlb.pos.mean(0)*4,
            pixelsize=self.cfg['pixelsize'], **kwargs)

        self.sum_ds_undrift = self.sum_ds[:]
        self.sum_ds_undrift.applyDrift(self.drift)
        self.sum_ds_undrift.save(self.resultsdir + "g2d_dme.hdf5")

    def drift_correct_sum_ds_filtered(self, **kwargs):
        self.drift, est_ = self.sum_ds.estimateDriftMinEntropy(
            coarseSigma=self.sum_ds.crlb.pos.mean(0) * 4,
            pixelsize=self.cfg['pixelsize'], **kwargs)

        self.sum_ds_undrift = self.sum_ds[:]
        self.sum_ds_undrift.applyDrift(self.drift)
        self.sum_ds_undrift.save(self.resultsdir + "g2d_filtered_dme.hdf5")

    def drift_correctsf(self, **kwargs):
        self.drift, est_ = self.sf_ds.estimateDriftMinEntropy(
            coarseSigma=self.sf_ds.crlb.pos.mean(0)*4,
            pixelsize=self.cfg['pixelsize'], **kwargs)

        self.sf_ds_undrift = self.sf_ds[:]
        self.sf_ds_undrift.applyDrift(self.drift)
        self.sf_ds_undrift.save(self.resultsdir + "sf_dme.hdf5")


    def phase_undrift(self):
        for step in [0]:
            ds = self.get_phase_ds(step)

    def get_phase_ds(self,step):
        k = self.mod['k'][step][:2]

        print('hi')

        M = 1
        ds = Dataset(len(self.sum_ds), 2, (1, M*2*np.pi))

        normI = self.sum_ds.IBg[:, step, 0] / self.sum_ds.photons

        proj = (k[None] * self.sum_ds.pos).sum(1) # - spot_phase
        ds.pos[:,0] = proj % (np.pi*2)
        ds.pos[:,1] = normI

        ds.frame = self.sum_ds.frame
        ds.crlb.pos[:,0] = 0.1 #self.sum_ds.crlb.pos[:,0]
        ds.crlb.pos[:,1] = 0.1# self.sum_ds.crlb.photons / self.sum_ds.photons
        return ds


    def cluster_plot(self):
        ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def refine_phases(self):
        ...


    def intensity_fits(self):
        ...

    def estimate_patterns_iter(self, num_angle_bins=1,
                          num_phase_bins=10,
                          pitch_minmax_nm=[300, 1500],
                          p_ax=0,
                          fix_phase_shifts=None,
                          fix_depths=None,
                          show_plots=True,
                          phase_method=1,
                          dft_peak_search_range=0.02,
                               first_iteration =False, kz = np.array([])):

        if first_iteration == True:
            kx0 = 0.5
            ky0 = 0.5
            kz0 = 5
            #kx0 =  self.mod['k'][self.pattern_frames[0], 0][0]
            #ky0 = self.mod['k'][self.pattern_frames[0], 1][0]
            # kz0 = 5.05
            self.excDims =3

            kx1 = 0.5
            ky1 = -0.5
            kz1 = 5
            #kx1 = self.mod['k'][self.pattern_frames[1], 0][0]
            #ky1 = self.mod['k'][self.pattern_frames[1], 1][0]
            # kz1 = 5.05

        if kz.size:
            kz0 = kz[0]
            kz1 = kz[1]

        flag = 0*1
        for k in range(len(self.pattern_frames)):
            if flag ==0:
                self.mod['k'][self.pattern_frames[k], :3] = kx0, ky0, kz0
            if flag ==1:
                self.mod['k'][self.pattern_frames[k], :3] = kx1, ky1, kz1
            flag += 1

    def drawUpdate(self, mapping, centers):
        plt.figure(figsize=(30, 30))
        plt.scatter(pts[:, 0], pts[:, 1], c=mapping, s=1, marker='o')

        for k in range(len(centers)):
            circle = plt.Circle((centers[k, 0], centers[k, 1]), dist, color='r', fill=False)
            plt.gca().add_artist(circle)

        print(np.max(mapping))

    def cluster_picassopicks(self, pattern=0, depth=0, drift_corrected=False, fit_gaussians=False, prefix='0'):
        import yaml
        from yaml.loader import SafeLoader
        from sklearn.cluster import KMeans, SpectralClustering
        import scipy
        from scipy.optimize import curve_fit
        import scipy.stats
        def gauss(x, a, x0, sigma, c):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2) + c)

        folder = 'nanorulers_simflux_10ms_originalanglejan13_2_firstpattern'
        if pattern != 0:
            kzname = str(np.round(self.mod[pattern - 1][0][2], 4))
            filename = "E:/SIMFLUX Z/" + folder + "/results/" + folder + "_MMStack_Default.ome/all_picks_v2.yaml"
            #filename = "C:/data/SIMFLUX Z/" + folder + "/results/" + folder + "_MMStack_Default.ome/pickregions.yaml"

        else:
            kzname = str(np.round(self.mod[0][0][2], 4)) + '_' + str(np.round(self.mod[1][0][2], 4))
            filename = "E:/SIMFLUX Z/" + folder + "/results/" + folder + "_MMStack_Default.ome/all_picks_v2.yaml"
            #filename = "C:/data/SIMFLUX Z/" + folder + "/results/" + folder + "_MMStack_Default.ome/pickregions.yaml"


        with open(filename) as f:
            data = yaml.load(f, Loader=SafeLoader)

        centers = data['Centers']
        diameter = data['Diameter']
        if drift_corrected:
            pos_smlm = self.sum_ds_filtered_undrift.pos
            pos_sf = self.sf_ds_undrift.pos
            photons_sf = self.sf_ds.photons
            bg_sf = self.sf_ds.background
        else:

            pos_smlm = self.sum_ds_filtered.pos
            pos_sf = self.sf_ds.pos
            photons_sf = self.sf_ds.photons
            bg_sf = self.sf_ds.background
        bias_array = np.empty((0,3))
        mean_array = np.empty((0,3))

        distance_array_sf = np.array([])
        angle_array_sf = np.array([])
        distance_array_smlm = np.array([])
        angle_array_smlm = np.array([])
        std_smlm = []
        std_sf = []

        full_array_photons_cluster0 = np.empty((0))
        full_array_photons_cluster1 = np.empty((0))
        full_array_bg_cluster0 = np.empty((0))
        full_array_bg_cluster1 = np.empty((0))


        for i in range(len(centers)):
            center = np.array(centers[i])
            array = (pos_smlm[:,0]-center[0])**2 + (pos_smlm[:,1]-center[1])**2< (diameter/2)**2

            full_array_smlm = np.concatenate((pos_smlm[array,0, None] * 65, pos_smlm[array,1, None] * 65,pos_smlm[array,2, None]*1000), axis=-1)
            full_array_sf = np.concatenate(
                (pos_sf[array, 0, None] * 65, pos_sf[array, 1, None] * 65, pos_sf[array, 2, None] * 1000),
                axis=-1)
            photons = photons_sf[array]
            photons_bg = bg_sf[array]
            if len(full_array_smlm)>50 and len(full_array_sf)>50 :
                kmeans_smlm = KMeans(n_clusters=2).fit(full_array_smlm)
                kmeans_sf = KMeans(n_clusters=2).fit(full_array_sf)
                kmeans_sf.cluster_centers_[0,:] = np.mean(full_array_sf[kmeans_smlm.labels_ == 0], axis=0)
                kmeans_sf.cluster_centers_[1, :] = np.mean(full_array_sf[kmeans_smlm.labels_ == 1], axis=0)


                if np.sum(kmeans_smlm.labels_ == 0) > 30 and np.sum(kmeans_smlm.labels_ == 1) > 30 and \
                    np.array([np.linalg.norm(kmeans_sf.cluster_centers_[0, :] - kmeans_sf.cluster_centers_[1, :])])<150 and \
                        kmeans_smlm.cluster_centers_[0,0]>5000:
                    if False:
                        fig = plt.figure()
                        ax = fig.add_subplot(projection='3d')
                        ax.scatter(full_array_sf[kmeans_smlm.labels_ == 0][:,0],full_array_sf[kmeans_smlm.labels_ == 0][:,1],full_array_sf[kmeans_smlm.labels_ == 0][:,2])
                        ax.scatter(full_array_sf[kmeans_smlm.labels_ == 1][:,0],full_array_sf[kmeans_smlm.labels_ == 1][:,1], full_array_sf[kmeans_smlm.labels_ == 1][:,2])
                        plt.show()
                    #if np.sum(kmeans_sf.labels_ == 0) > 30 and np.sum(kmeans_sf.labels_ == 1) > 30:
                    # full_array_photons_cluster0 = np.append(full_array_photons_cluster0,np.mean(photons[kmeans_sf.labels_ == 0]))
                    # full_array_photons_cluster1 = np.append(full_array_photons_cluster1,np.mean(photons[kmeans_sf.labels_ == 1]))
                    #
                    # full_array_bg_cluster0 = np.append(full_array_bg_cluster0,np.mean(photons_bg[kmeans_sf.labels_ == 0]))
                    # full_array_bg_cluster1 = np.append(full_array_bg_cluster1,np.mean(photons_bg[kmeans_sf.labels_ == 1]))

                    bias = kmeans_smlm.cluster_centers_-kmeans_sf.cluster_centers_
                    mean_value = kmeans_smlm.cluster_centers_
                    bias_array= np.concatenate((bias_array,bias))
                    mean_array = np.concatenate((mean_array,mean_value))
                    mean_array_cluster0  = mean_array
                    distance_array_sf =  np.concatenate((distance_array_sf, np.array([np.linalg.norm(kmeans_sf.cluster_centers_[0,:] - kmeans_sf.cluster_centers_[1, :])])))

                    distance_array_smlm = np.concatenate((distance_array_smlm, np.array([np.linalg.norm(kmeans_smlm.cluster_centers_[0,:] - kmeans_smlm.cluster_centers_[1, :])])))

                    diff_sf = kmeans_sf.cluster_centers_[0, :] - kmeans_sf.cluster_centers_[1, :]
                    angle_array_sf=np.concatenate((angle_array_sf, np.array([np.arctan(np.sqrt(diff_sf[0]**2 + diff_sf[1]**2)/abs(diff_sf[2]))]) ))
                    diff_smlm = kmeans_smlm.cluster_centers_[0, :] - kmeans_smlm.cluster_centers_[1, :]
                    angle_array_smlm = np.concatenate((angle_array_smlm, np.array([np.arctan(np.sqrt(diff_smlm[0]**2 + diff_smlm[1]**2)/abs(diff_smlm[2])) ])))

                    if fit_gaussians:

                        # stuff for plotting things:
                        binwidth = 5
                        plt.hist(full_array_smlm[:,2], label='SMLM', alpha=0.5,
                                 facecolor='blue',
                                 bins=np.arange(min(full_array_smlm[:,2]), max(full_array_smlm[:,2]) + binwidth, binwidth))
                        plt.hist(full_array_sf[:,2], label='Simflux', alpha=0.5,
                                 facecolor='red',
                                 bins=np.arange(min(full_array_sf[:,2]), max(full_array_sf[:,2]) + binwidth, binwidth))
                        for kmean in range(2):
                            init_fit_smlm = scipy.stats.norm.fit(full_array_smlm[:,2][kmeans_smlm.labels_ == kmean])
                            init_fit_sf = scipy.stats.norm.fit(full_array_sf[:,2][kmeans_smlm.labels_ == kmean])

                            n_smlm, bins_smlm = np.histogram(full_array_smlm[:,2][kmeans_smlm.labels_ == kmean],
                                                   bins=np.arange(min(full_array_smlm[:,2]), max(full_array_smlm[:,2]) + binwidth, binwidth))
                            n_sf, bins_sf = np.histogram(full_array_sf[:,2][kmeans_smlm.labels_ == kmean],
                                                             bins=np.arange(min(full_array_sf[:,2]),
                                                                            max(full_array_sf[:,2]) + binwidth, binwidth))

                            popt_smlm, pcov_smlm = curve_fit(gauss, (bins_smlm[1:] + bins_smlm[:-1]) / 2, n_smlm,
                                                             p0=[max(n_smlm), init_fit_smlm[0], init_fit_smlm[1], 0],
                                                             maxfev=2000)
                            popt_sf, pcov_sf = curve_fit(gauss, (bins_sf[1:] + bins_sf[:-1]) / 2, n_sf,
                                                             p0=[max(n_sf), init_fit_sf[0], init_fit_sf[1], 0],
                                                             maxfev=2000)
                            plt.plot(np.linspace(min(full_array_smlm[:,2][kmeans_smlm.labels_ == kmean]),
                                                 max(full_array_smlm[:,2][kmeans_smlm.labels_ == kmean]), 50),
                                     gauss(np.linspace(min(full_array_smlm[:,2][kmeans_smlm.labels_ == kmean]),
                                                       max(full_array_smlm[:,2][kmeans_smlm.labels_ == kmean]), 50), *popt_smlm),
                                     c='blue', label=str(popt_smlm[1].round(1)) + ' +/-' + str(popt_smlm[2].round(1)))
                            plt.plot(np.linspace(min(full_array_sf[:,2][kmeans_sf.labels_ == kmean]),
                                                 max(full_array_sf[:,2][kmeans_sf.labels_ == kmean]), 50),
                                     gauss(np.linspace(min(full_array_sf[:,2][kmeans_sf.labels_ == kmean]),
                                                       max(full_array_sf[:,2][kmeans_sf.labels_ == kmean]), 50), *popt_sf),
                                     c='red', label=str(popt_sf[1].round(1)) + ' +/-' + str(popt_sf[2].round(1)))

                        # n, bins, patches = plt.hist(data_plot, label=filename, alpha=0.5,
                        #                             facecolor=color[i],
                        #                             bins=np.arange(min(data_plot), max(data_plot) + binwidth, binwidth))
                        plt.title('detection cluster %i' % (i))
                        plt.xlabel('z (nm)')
                        plt.ylabel('count')
                        plt.legend()
                        import os
                        folder_results = "E:/SIMFLUX Z/" + folder + "/results/results10msnov/test" + prefix +str(depth)
                        isExist = os.path.exists("E:/SIMFLUX Z/" + folder + "/results/results10msnov/")
                        # folder_results = 'C:/Users/Gebruiker/Desktop/results10msnov/test' + prefix +str(depth)
                        # isExist = os.path.exists('C:/Users/Gebruiker/Desktop/results10msnov/')


                        if not isExist:
                            #os.mkdir('C:/Users/GrunwaldLab/Desktop/results10msnov/')
                            os.mkdir("E:/SIMFLUX Z/" + folder + "/results/results10msnov/")

                        isExist = os.path.exists(folder_results)
                        if not isExist:
                            os.mkdir(folder_results)
                        plt.savefig(
                            folder_results + "/detection cluster %i.png" % (
                                i))
                        plt.close()
                        # plot clusters seperately
                        for kmean in range(2):
                            binwidth = 8
                            plt.hist(full_array_smlm[:, 2][kmeans_smlm.labels_ == kmean], label='SMLM', alpha=0.5,
                                     facecolor='blue',
                                     bins=np.arange(min(full_array_smlm[:, 2][kmeans_smlm.labels_ == kmean]), max(full_array_smlm[:, 2][kmeans_smlm.labels_ == kmean]) + binwidth,
                                                    binwidth))
                            plt.hist(full_array_sf[:, 2][kmeans_smlm.labels_ == kmean], label='Simflux', alpha=0.5,
                                     facecolor='red',
                                     bins=np.arange(min(full_array_sf[:, 2][kmeans_smlm.labels_ == kmean]), max(full_array_sf[:, 2][kmeans_smlm.labels_ == kmean]) + binwidth,
                                                    binwidth))
                            init_fit_smlm = scipy.stats.norm.fit(full_array_smlm[:, 2][kmeans_smlm.labels_ == kmean])
                            init_fit_sf = scipy.stats.norm.fit(full_array_sf[:, 2][kmeans_smlm.labels_ == kmean])

                            n_smlm, bins_smlm = np.histogram(full_array_smlm[:, 2][kmeans_smlm.labels_ == kmean],
                                                             bins=np.arange(min(full_array_smlm[:, 2]),
                                                                            max(full_array_smlm[:, 2]) + binwidth,
                                                                            binwidth))
                            n_sf, bins_sf = np.histogram(full_array_sf[:, 2][kmeans_smlm.labels_ == kmean],
                                                         bins=np.arange(min(full_array_sf[:, 2]),
                                                                        max(full_array_sf[:, 2]) + binwidth, binwidth))

                            popt_smlm, pcov_smlm = curve_fit(gauss, (bins_smlm[1:] + bins_smlm[:-1]) / 2, n_smlm,
                                                             p0=[max(n_smlm), init_fit_smlm[0], init_fit_smlm[1], 0],
                                                             maxfev=2000, bounds=[[0,-np.inf,0,0],[np.inf,np.inf,np.inf,np.inf]])
                            std_smlm.append(popt_smlm[2])
                            popt_sf, pcov_sf = curve_fit(gauss, (bins_sf[1:] + bins_sf[:-1]) / 2, n_sf,
                                                         p0=[max(n_sf), init_fit_sf[0], init_fit_sf[1], 0],
                                                         maxfev=2000, bounds=[[0,-np.inf,0,0],[np.inf,np.inf,np.inf,np.inf]])
                            std_sf.append(popt_sf[2])
                            plt.plot(np.linspace(min(full_array_smlm[:, 2][kmeans_smlm.labels_ == kmean]),
                                                 max(full_array_smlm[:, 2][kmeans_smlm.labels_ == kmean]), 50),
                                     gauss(np.linspace(min(full_array_smlm[:, 2][kmeans_smlm.labels_ == kmean]),
                                                       max(full_array_smlm[:, 2][kmeans_smlm.labels_ == kmean]), 50),
                                           *popt_smlm),
                                     c='blue', label=str(popt_smlm[1].round(1)) + ' +/-' + str(popt_smlm[2].round(1)))
                            plt.plot(np.linspace(min(full_array_sf[:, 2][kmeans_sf.labels_ == kmean]),
                                                 max(full_array_sf[:, 2][kmeans_sf.labels_ == kmean]), 50),
                                     gauss(np.linspace(min(full_array_sf[:, 2][kmeans_sf.labels_ == kmean]),
                                                       max(full_array_sf[:, 2][kmeans_sf.labels_ == kmean]), 50),
                                           *popt_sf),
                                     c='red', label=str(popt_sf[1].round(1)) + ' +/-' + str(popt_sf[2].round(1)))
                            plt.title('binding site '+str(kmean)+ ' detection cluster %i' % (i))
                            plt.xlabel('z (nm)')
                            plt.ylabel('count')
                            plt.legend()
                            plt.savefig(
                                folder_results + "/binding site "+str(kmean)+ "detection cluster %i.png" % (
                                    i))
                            plt.close()
        cm = 1 / 2.54
        import matplotlib
        from matplotlib import rc, font_manager
        import scipy.interpolate
        fontProperties = {'family': 'sans-serif', 'sans-serif': ['Helvetica'],
                          'weight': 'normal', 'size': 12}
        rc('text', usetex=True)
        rc('font', **fontProperties)
        filter_min = np.logical_and(bias_array[:,2] < (np.mean(bias_array[:,2]) + np.std(bias_array[:,2]))
                                    , bias_array[:,2] > (np.mean(bias_array[:,2]) - np.std(bias_array[:,2])))
        slope_z, offset = np.polyfit(-(mean_array[:, 2]+2*self.depth)[filter_min], bias_array[filter_min, 2], 1)
        plt.figure(figsize=(10*cm, 8*cm))
        plt.scatter(-(mean_array[:, 2]+2*self.depth), bias_array[:, 2],s=8)
        plt.plot(-(mean_array[:, 2]+2*self.depth), slope_z * -(mean_array[:, 2]+2*self.depth) + offset, color='red')
        plt.xlim([min(-(mean_array[:, 2]+2*self.depth)), max(-(mean_array[:, 2]+2*self.depth))])
        plt.ylim(-100, 100)
        plt.xlabel(r'$z_{pos}$ SMLM [nm]')
        plt.ylabel('Bias in z [nm]')
        plt.tight_layout(pad=0.1)
        #plt.title('pattern = '+ str(pattern) + ', kz =' + kzname )

        import os
        if drift_corrected:
             path_folder = "E:/SIMFLUX Z/biasplots/"
        else:
            path_folder = "E:/SIMFLUX Z/biasplots/"


        # Check whether the specified path exists or not
        isExist = os.path.exists(path_folder)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path_folder)

        plt.savefig(path_folder + 'bias_pat' + str(pattern) + '_kz='+kzname+'.png', dpi=600)

        fig, axs = plt.subplots(2,figsize=(12*cm,8*cm))
        #fig.suptitle('x and y bias')
        axs[0].scatter(mean_array[:, 0], bias_array[:, 0],s=8)
        axs[0].set_xlabel('x position SMLM [nm]')
        axs[0].set_ylabel('x bias [nm]')
        axs[0].set_ylim(-100, 100)
        axs[1].scatter(mean_array[:, 1], bias_array[:, 1],s=8)
        axs[1].set_xlabel('y position SMLM [nm]')
        axs[1].set_ylabel('y bias [nm]')
        axs[1].set_ylim(-100, 100)
        fig.tight_layout(pad=0.1)
        plt.savefig(path_folder + 'xy_bias' + str(pattern) + '_kz=' + kzname + '.png', dpi=600)

        plt.figure()
        plt.scatter(mean_array[:,0],mean_array[:,1],c=bias_array[:,2])
        plt.xlabel('x position [nm]')
        plt.ylabel('y position [nm]')
        #plt.title('z-bias (SMLM-SF) over FOV')
        cbar = plt.colorbar()
        cbar.set_label('z-bias [nm]')
        fig.tight_layout(pad=0.1)
        plt.savefig(
        path_folder + 'biasfov' + str(pattern) + '_kz=' + kzname + '.png')

        def gauss(x, a, x0, sigma, c):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2) + c)

        # from scipy.optimize import curve_fit
        # plt.figure()
        # n, bins, _ = plt.hist(bias_array[:, 2])
        # plt.xlabel('bias in z [nm]')
        # plt.ylabel('counts')
        #
        # popt_bias, pcov_bias = curve_fit(gauss, (bins[1:] + bins[:-1]) / 2, n, p0=[max(n), np.mean(bias_array[:,2]),
        #                                                                            np.std(bias_array[:,2]), 0],maxfev=2000)
        # plt.plot(np.linspace(min(bias_array[:, 2]), max(bias_array[:, 2]), 50),
        #          gauss(np.linspace(min(bias_array[:, 2]), max(bias_array[:, 2]), 50), *popt_bias), c='red')
        # plt.title('pattern = ' + str(pattern) + ', kz =' + kzname + ', mean = ' + str(np.round(popt_bias[1],2)))
        #
        # plt.savefig(
        #     path_folder + 'biasoverall' + str(pattern) + '_kz=' + kzname + '.png')


        plt.figure(figsize=(10*cm, 8*cm))
        slopex, offset = np.polyfit(mean_array[:, 0], bias_array[:, 2], 1)
        plt.scatter(mean_array[:, 0], bias_array[:, 2],s=8)
        plt.plot(mean_array, slopex * mean_array + offset, color='red')
        plt.xlim([min(mean_array[:, 0]), max(mean_array[:, 0])])
        plt.ylim(-100, 100)
        plt.xlabel('x-postion SMLM [nm]')
        plt.ylabel('Bias in z [nm]')
        #plt.title('pattern = '+ str(pattern) + ', kz =' + kzname )
        plt.tight_layout(pad=0.1)
        plt.savefig(
            path_folder + 'bias_z_dep_x' + str(pattern) + '_kz=' + kzname + '.png', dpi=600)

        plt.figure(figsize=(10*cm, 8*cm))
        slopey, offset = np.polyfit(mean_array[:, 1], bias_array[:, 2], 1)
        plt.scatter(mean_array[:, 1], bias_array[:, 2],s=8)
        plt.plot(mean_array, slopey * mean_array + offset, color='red')
        plt.xlim([min(mean_array[:, 1]), max(mean_array[:, 1])])
        plt.ylim(-100, 100)
        plt.xlabel('y-postion SMLM [nm]')
        plt.ylabel('Bias in z [nm]')
        #plt.title('pattern = ' + str(pattern) + ', kz =' + kzname)
        plt.tight_layout(pad=0.1)
        plt.savefig(
            path_folder + 'bias_z_dep_y' + str(pattern) + '_kz=' + kzname + '.png', dpi=600)

        plt.figure(figsize=(6,6))
        import scipy.stats
        # filter_dist_sf = distance_array_sf>45
        # distance_array_smlm =distance_array_smlm[filter_dist_sf]
        # distance_array_sf = distance_array_sf[filter_dist_sf]
        fit_sf = scipy.stats.norm.fit(distance_array_sf)
        fit_smlm = scipy.stats.norm.fit(np.array(distance_array_smlm))
        binwidth_cluster = 5
        n, bins, _ = plt.hist(distance_array_sf, label='simflux', alpha=0.5, facecolor='red',
                    bins= np.arange(min(distance_array_sf), max(distance_array_sf) + binwidth_cluster, binwidth_cluster)
)
        #filter based on dsitance


        try:

            popt_sf, pcov_sf = curve_fit(gauss, (bins[1:] + bins[:-1]) / 2, n, p0=[max(n), fit_sf[0], fit_sf[1], 0],
                                     maxfev=2000)

            plt.plot(np.linspace(min(distance_array_sf), max(distance_array_sf), 50),
                     gauss(np.linspace(min(distance_array_sf), max(distance_array_sf), 50), *popt_sf), c='red',
                     label='SF fit')
        except:
            _ = 0
        n, bins, _ = plt.hist(np.array(distance_array_smlm), label='SMLM', facecolor='blue', alpha=0.5,
                  bins=np.arange(min(distance_array_smlm), max(distance_array_smlm) + binwidth_cluster, binwidth_cluster))

        try:
            popt_smlm, pcov_smlm = curve_fit(gauss, (bins[1:] + bins[:-1]) / 2, n, p0=[max(n), fit_smlm[0], fit_smlm[1], 0],
                                             maxfev=2000)
            plt.plot(np.linspace(min(distance_array_smlm), 150, 50),
                     gauss(np.linspace(min(distance_array_smlm), 150, 50), *popt_smlm), c='blue', label='SMLM fit')
        except:
            _=0
        # except:
        # n, bins, _ = plt.hist(cart_distance_smlm, label='SMLM', facecolor='blue', alpha=0.5, bins=20)
        # print('do nothing')
        plt.xlabel('Euclidean distance [nm]')
        plt.ylabel('Counts')
        #plt.legend()
        if 'popt_smlm' in locals() and 'popt_sf' in locals() :
            plt.title('mean smlm pos = ' + str(np.round(popt_smlm[1],2)) + ' sf = ' + str(np.round(popt_sf[1],2)) )

            plt.title('mean smlm = ' + str(popt_smlm[1]))
            plt.title('mean smlm = ' + str(popt_smlm[1]))
        plt.tight_layout(pad=0)
        plt.savefig(path_folder + 'clusters' + str(pattern) + '_kz=' + kzname + '.png')

        plt.figure()
        import scipy.stats
        #angle_array_smlm = angle_array_smlm[filter_dist_sf]
        #angle_array_sf = angle_array_sf[filter_dist_sf]
        fit_sf = scipy.stats.norm.fit(angle_array_sf)
        fit_smlm = scipy.stats.norm.fit(np.array(angle_array_smlm))
        binwidth_cluster = 0.3
        n, bins, _ = plt.hist(angle_array_sf, label='simflux', alpha=0.5, facecolor='red',
                              bins=np.arange(min(angle_array_sf), max(angle_array_sf) + binwidth_cluster,
                                             binwidth_cluster)
                              )
        # filter based on dsitance

        #try:

        # popt_sf, pcov_sf = curve_fit(gauss, (bins[1:] + bins[:-1]) / 2, n, p0=[max(n), fit_sf[0], fit_sf[1], 0],
        #                              maxfev=2000)
        #
        # plt.plot(np.linspace(min(angle_array_sf), max(angle_array_sf), 50),
        #          gauss(np.linspace(min(angle_array_sf), max(angle_array_sf), 50), *popt_sf), c='red',
        #          label='SF fit')
        #except:
            #_ = 0
        n, bins, _ = plt.hist(np.array(angle_array_smlm), label='SMLM', facecolor='blue', alpha=0.5,
                              bins=np.arange(min(angle_array_smlm), max(angle_array_smlm) + binwidth_cluster,
                                             binwidth_cluster))

        #try:
        # popt_smlm, pcov_smlm = curve_fit(gauss, (bins[1:] + bins[:-1]) / 2, n,
        #                                  p0=[max(n), fit_smlm[0], fit_smlm[1], 0],
        #                                  maxfev=2000)
        # plt.plot(np.linspace(min(angle_array_smlm), 150, 50),
        #          gauss(np.linspace(min(angle_array_smlm), 150, 50), *popt_smlm), c='blue', label='SMLM fit')
        # #except:
            #_ = 0
        # except:
        # n, bins, _ = plt.hist(cart_distance_smlm, label='SMLM', facecolor='blue', alpha=0.5, bins=20)
        # print('do nothing')
        plt.xlabel('Angle [rad]')
        plt.ylabel('Counts')
        plt.legend()
        #plt.title('mean smlm pos = ' + str(np.round(popt_smlm[1], 2)) + ' sf = ' + str(np.round(popt_sf[1], 2)))
        plt.savefig(path_folder + 'angle' + str(pattern) + '_kz=' + kzname + '.png')
        # plot angle vs length
        plt.figure(figsize=(6,6))
        import scipy.stats

        plt.scatter(angle_array_sf, np.array(distance_array_sf), label='simflux', alpha=0.5, facecolor='red')
        slope_angle_sf = np.polyfit(angle_array_sf,distance_array_sf,1)
        plt.plot(np.array([0,1.6]), slope_angle_sf[0]* np.array([0,1.6])+ slope_angle_sf[1], color='red', linestyle=':',markersize=0)
        plt.scatter(angle_array_smlm, distance_array_smlm, label='SMLM', facecolor='blue', alpha=0.5)

        slope_angle_smlm = np.polyfit(angle_array_smlm, distance_array_smlm, 1)
        plt.plot(np.array([0,1.6]), slope_angle_smlm[0] *np.array([0,1.6]) + slope_angle_smlm[1], color='blue', linestyle=':',markersize=0)
        # try:
        # popt_smlm, pcov_smlm = curve_fit(gauss, (bins[1:] + bins[:-1]) / 2, n,
        #                                  p0=[max(n), fit_smlm[0], fit_smlm[1], 0],
        #                                  maxfev=2000)
        # plt.plot(np.linspace(min(angle_array_smlm), 150, 50),
        #          gauss(np.linspace(min(angle_array_smlm), 150, 50), *popt_smlm), c='blue', label='SMLM fit')
        # #except:
        # _ = 0
        # except:
        # n, bins, _ = plt.hist(cart_distance_smlm, label='SMLM', facecolor='blue', alpha=0.5, bins=20)
        # print('do nothing')
        plt.xlabel('Angle [rad]')
        plt.ylabel('Length [nm]')
        #plt.legend()
        # plt.title('mean smlm pos = ' + str(np.round(popt_smlm[1], 2)) + ' sf = ' + str(np.round(popt_sf[1], 2)))
        plt.tight_layout(pad=0)
        plt.savefig(path_folder + 'length_vs_angle' + str(pattern) + '_kz=' + kzname + '.png', dpi=300)
        #plt.show()
        plt.close('all')

        return slope_z, std_smlm, std_sf, angle_array_smlm,angle_array_sf, distance_array_smlm,distance_array_sf,mean_array, \
               full_array_photons_cluster0, full_array_photons_cluster1,full_array_bg_cluster0,full_array_bg_cluster1


    def cluster_picassopicksv2(self, pattern=0, depth=0, drift_corrected=False, fit_gaussians=False, prefix='0'):
        import yaml
        from yaml.loader import SafeLoader
        from sklearn.cluster import KMeans, SpectralClustering
        import scipy
        from scipy.optimize import curve_fit
        import scipy.stats
        def gauss(x, a, x0, sigma, c):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2) + c)

        folder = 'nanorulers_simflux_10ms_originalanglejan13_2_firstpattern'
        if pattern != 0:
            kzname = str(np.round(self.mod[pattern - 1][0][2], 4))
            filename = "E:/SIMFLUX Z/" + folder + "/results/" + folder + "_MMStack_Default.ome/all_picks_v2.yaml"
            # filename = "C:/data/SIMFLUX Z/" + folder + "/results/" + folder + "_MMStack_Default.ome/pickregions.yaml"

        else:
            kzname = str(np.round(self.mod[0][0][2], 4)) + '_' + str(np.round(self.mod[1][0][2], 4))
            filename = "E:/SIMFLUX Z/" + folder + "/results/" + folder + "_MMStack_Default.ome/all_picks_v2.yaml"
            # filename = "C:/data/SIMFLUX Z/" + folder + "/results/" + folder + "_MMStack_Default.ome/pickregions.yaml"

        with open(filename) as f:
            data = yaml.load(f, Loader=SafeLoader)

        centers = data['Centers']
        diameter = data['Diameter']
        if drift_corrected:
            pos_smlm = self.sum_ds_filtered_undrift.pos
            pos_sf = self.sf_ds_undrift.pos
            photons_sf = self.sf_ds_undrift.photons
            bg_sf = self.sf_ds_undrift.background
            photons_all_smlm = self.sum_ds_filtered_undrift.photons
            bg_all_smlm = self.sum_ds_filtered_undrift.background
        else:

            pos_smlm = self.sum_ds_filtered.pos
            pos_sf = self.sf_ds.pos
            photons_sf = self.sf_ds.photons
            bg_sf = self.sf_ds.background
            photons_all_smlm = self.sum_ds_filtered.photons
            bg_all_smlm = self.sum_ds_filtered.background
        bias_array = np.empty((0, 3))
        mean_array = np.empty((0, 3))
        mean_array_cluster0 = np.empty((0, 3))
        mean_array_cluster1 = np.empty((0, 3))
        mean_array_cluster0_sf = np.empty((0, 3))
        mean_array_cluster1_sf = np.empty((0, 3))
        distance_array_sf = np.array([])
        angle_array_sf = np.array([])
        distance_array_smlm = np.array([])
        angle_array_smlm = np.array([])
        std_smlm = []
        std_sf = []
        stdx_smlm = []
        stdx_sf = []
        stdy_smlm = []
        stdy_sf = []

        full_array_photons_cluster0 = np.empty((0))
        full_array_photons_cluster1 = np.empty((0))
        full_array_bg_cluster0 = np.empty((0))
        full_array_bg_cluster1 = np.empty((0))

        full_array_photons_cluster0_smlm = np.empty((0))
        full_array_photons_cluster1_smlm = np.empty((0))
        full_array_bg_cluster0_smlm = np.empty((0))
        full_array_bg_cluster1_smlm = np.empty((0))

        for i in range(len(centers)):
            try:
                center = np.array(centers[i])
                array = (pos_smlm[:, 0] - center[0]) ** 2 + (pos_smlm[:, 1] - center[1]) ** 2 < (diameter / 2) ** 2

                full_array_smlm = np.concatenate(
                    (pos_smlm[array, 0, None] * 65, pos_smlm[array, 1, None] * 65, pos_smlm[array, 2, None] * 1000),
                    axis=-1)
                full_array_sf = np.concatenate(
                    (pos_sf[array, 0, None] * 65, pos_sf[array, 1, None] * 65, pos_sf[array, 2, None] * 1000),
                    axis=-1)
                photons = photons_sf[array]
                photons_bg = bg_sf[array]
                photons_smlm = photons_all_smlm[array]
                photons_bg_smlm = bg_all_smlm[array]
                if len(full_array_smlm) > 50 and len(full_array_sf) > 50:
                    kmeans_smlm = KMeans(n_clusters=2, random_state=2, algorithm='elkan').fit(full_array_smlm)
                    kmeans_sf = KMeans(n_clusters=2, random_state=2, algorithm='elkan').fit(full_array_sf)
                    kmeans_sf.cluster_centers_[0, :] = np.mean(full_array_sf[kmeans_smlm.labels_ == 0], axis=0)
                    kmeans_sf.cluster_centers_[1, :] = np.mean(full_array_sf[kmeans_smlm.labels_ == 1], axis=0)

                    if np.sum(kmeans_smlm.labels_ == 0) > 30 and np.sum(kmeans_smlm.labels_ == 1) > 30 and \
                            np.array([np.linalg.norm(
                                kmeans_sf.cluster_centers_[0, :] - kmeans_sf.cluster_centers_[1, :])]) < 150 and \
                            kmeans_smlm.cluster_centers_[0, 0] > 5000:
                        full_array_photons_cluster0 = np.append(full_array_photons_cluster0,np.mean(photons[kmeans_sf.labels_ == 0]))
                        full_array_photons_cluster1 = np.append(full_array_photons_cluster1,np.mean(photons[kmeans_sf.labels_ == 1]))

                        full_array_bg_cluster0 = np.append(full_array_bg_cluster0,np.mean(photons_bg[kmeans_sf.labels_ == 0]))
                        full_array_bg_cluster1 = np.append(full_array_bg_cluster1,np.mean(photons_bg[kmeans_sf.labels_ == 1]))

                        full_array_photons_cluster0_smlm = np.append(full_array_photons_cluster0_smlm,np.mean(photons_smlm[kmeans_sf.labels_ == 0]))
                        full_array_photons_cluster1_smlm = np.append(full_array_photons_cluster1_smlm,np.mean(photons_smlm[kmeans_sf.labels_ == 1]))

                        full_array_bg_cluster0_smlm = np.append(full_array_bg_cluster0_smlm,np.mean(photons_bg_smlm[kmeans_sf.labels_ == 0]))
                        full_array_bg_cluster1_smlm = np.append(full_array_bg_cluster1_smlm,np.mean(photons_bg_smlm[kmeans_sf.labels_ == 1]))

                        bias = kmeans_smlm.cluster_centers_ - kmeans_sf.cluster_centers_
                        mean_value = kmeans_smlm.cluster_centers_
                        bias_array = np.concatenate((bias_array, bias))
                        mean_array = np.concatenate((mean_array, mean_value))
                        mean_array_cluster0 =  np.concatenate((mean_array_cluster0, mean_value[0,:][None,...]))
                        mean_array_cluster1 =  np.concatenate((mean_array_cluster1, mean_value[1,:][None,...]))
                        mean_array_cluster0_sf = np.concatenate((mean_array_cluster0_sf, kmeans_sf.cluster_centers_[0,:][None,...]))
                        mean_array_cluster1_sf = np.concatenate((mean_array_cluster1_sf, kmeans_sf.cluster_centers_[1,:][None,...]))

                        distance_array_sf = np.concatenate((distance_array_sf, np.array(
                            [np.linalg.norm(kmeans_sf.cluster_centers_[0, :] - kmeans_sf.cluster_centers_[1, :])])))

                        distance_array_smlm = np.concatenate((distance_array_smlm, np.array(
                            [np.linalg.norm(kmeans_smlm.cluster_centers_[0, :] - kmeans_smlm.cluster_centers_[1, :])])))

                        diff_sf = kmeans_sf.cluster_centers_[0, :] - kmeans_sf.cluster_centers_[1, :]
                        angle_array_sf = np.concatenate((angle_array_sf, np.array(
                            [np.arctan(np.sqrt(diff_sf[0] ** 2 + diff_sf[1] ** 2) / abs(diff_sf[2]))])))
                        diff_smlm = kmeans_smlm.cluster_centers_[0, :] - kmeans_smlm.cluster_centers_[1, :]
                        angle_array_smlm = np.concatenate((angle_array_smlm, np.array(
                            [np.arctan(np.sqrt(diff_smlm[0] ** 2 + diff_smlm[1] ** 2) / abs(diff_smlm[2]))])))

                        for kmean in range(2):

                            # for z
                            binwidth = 8
                            init_fit_smlm = scipy.stats.norm.fit(full_array_smlm[:, 2][kmeans_smlm.labels_ == kmean])
                            init_fit_sf = scipy.stats.norm.fit(full_array_sf[:, 2][kmeans_smlm.labels_ == kmean])

                            n_smlm, bins_smlm = np.histogram(full_array_smlm[:, 2][kmeans_smlm.labels_ == kmean],
                                                             bins=np.arange(min(full_array_smlm[:, 2]),
                                                                            max(full_array_smlm[:, 2]) + binwidth,
                                                                            binwidth))
                            n_sf, bins_sf = np.histogram(full_array_sf[:, 2][kmeans_smlm.labels_ == kmean],
                                                         bins=np.arange(min(full_array_sf[:, 2]),
                                                                        max(full_array_sf[:, 2]) + binwidth, binwidth))

                            popt_smlm, pcov_smlm = curve_fit(gauss, (bins_smlm[1:] + bins_smlm[:-1]) / 2, n_smlm,
                                                             p0=[max(n_smlm), init_fit_smlm[0], init_fit_smlm[1], 0],
                                                             maxfev=2000,
                                                             bounds=[[0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]])

                            popt_sf, pcov_sf = curve_fit(gauss, (bins_sf[1:] + bins_sf[:-1]) / 2, n_sf,
                                                         p0=[max(n_sf), init_fit_sf[0], init_fit_sf[1], 0],
                                                         maxfev=2000,
                                                         bounds=[[0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]])


                            # for x
                            binwidth = 8
                            init_fit_smlm = scipy.stats.norm.fit(full_array_smlm[:, 0][kmeans_smlm.labels_ == kmean])
                            init_fit_sf = scipy.stats.norm.fit(full_array_sf[:, 0][kmeans_smlm.labels_ == kmean])

                            n_smlm, bins_smlm = np.histogram(full_array_smlm[:, 0][kmeans_smlm.labels_ == kmean],
                                                             bins=np.arange(min(full_array_smlm[:, 0]),
                                                                            max(full_array_smlm[:, 0]) + binwidth,
                                                                            binwidth))
                            n_sf, bins_sf = np.histogram(full_array_sf[:, 0][kmeans_smlm.labels_ == kmean],
                                                         bins=np.arange(min(full_array_sf[:, 0]),
                                                                        max(full_array_sf[:, 0]) + binwidth, binwidth))

                            popt_smlm, pcov_smlm = curve_fit(gauss, (bins_smlm[1:] + bins_smlm[:-1]) / 2, n_smlm,
                                                             p0=[max(n_smlm), init_fit_smlm[0], init_fit_smlm[1], 0],
                                                             maxfev=2000,
                                                             bounds=[[0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]])

                            popt_sf, pcov_sf = curve_fit(gauss, (bins_sf[1:] + bins_sf[:-1]) / 2, n_sf,
                                                         p0=[max(n_sf), init_fit_sf[0], init_fit_sf[1], 0],
                                                         maxfev=2000,
                                                         bounds=[[0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]])

                            # for y
                            binwidth = 8
                            init_fit_smlm = scipy.stats.norm.fit(full_array_smlm[:, 1][kmeans_smlm.labels_ == kmean])
                            init_fit_sf = scipy.stats.norm.fit(full_array_sf[:, 1][kmeans_smlm.labels_ == kmean])

                            n_smlm, bins_smlm = np.histogram(full_array_smlm[:, 0][kmeans_smlm.labels_ == kmean],
                                                             bins=np.arange(min(full_array_smlm[:, 1]),
                                                                            max(full_array_smlm[:, 1]) + binwidth,
                                                                            binwidth))
                            n_sf, bins_sf = np.histogram(full_array_sf[:, 0][kmeans_smlm.labels_ == kmean],
                                                         bins=np.arange(min(full_array_sf[:, 1]),
                                                                        max(full_array_sf[:, 1]) + binwidth, binwidth))

                            popt_smlm, pcov_smlm = curve_fit(gauss, (bins_smlm[1:] + bins_smlm[:-1]) / 2, n_smlm,
                                                             p0=[max(n_smlm), init_fit_smlm[0], init_fit_smlm[1], 0],
                                                             maxfev=2000,
                                                             bounds=[[0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]])

                            popt_sf, pcov_sf = curve_fit(gauss, (bins_sf[1:] + bins_sf[:-1]) / 2, n_sf,
                                                         p0=[max(n_sf), init_fit_sf[0], init_fit_sf[1], 0],
                                                         maxfev=2000,
                                                         bounds=[[0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]])



                            #double, but with append
                            # for z
                            binwidth = 8
                            init_fit_smlm = scipy.stats.norm.fit(full_array_smlm[:, 2][kmeans_smlm.labels_ == kmean])
                            init_fit_sf = scipy.stats.norm.fit(full_array_sf[:, 2][kmeans_smlm.labels_ == kmean])

                            n_smlm, bins_smlm = np.histogram(full_array_smlm[:, 2][kmeans_smlm.labels_ == kmean],
                                                             bins=np.arange(min(full_array_smlm[:, 2]),
                                                                            max(full_array_smlm[:, 2]) + binwidth,
                                                                            binwidth))
                            n_sf, bins_sf = np.histogram(full_array_sf[:, 2][kmeans_smlm.labels_ == kmean],
                                                         bins=np.arange(min(full_array_sf[:, 2]),
                                                                        max(full_array_sf[:, 2]) + binwidth, binwidth))

                            popt_smlm, pcov_smlm = curve_fit(gauss, (bins_smlm[1:] + bins_smlm[:-1]) / 2, n_smlm,
                                                             p0=[max(n_smlm), init_fit_smlm[0], init_fit_smlm[1], 0],
                                                             maxfev=2000,
                                                             bounds=[[0, -np.inf, 0, 0],
                                                                     [np.inf, np.inf, np.inf, np.inf]])
                            std_smlm.append(popt_smlm[2])
                            popt_sf, pcov_sf = curve_fit(gauss, (bins_sf[1:] + bins_sf[:-1]) / 2, n_sf,
                                                         p0=[max(n_sf), init_fit_sf[0], init_fit_sf[1], 0],
                                                         maxfev=2000,
                                                         bounds=[[0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]])
                            std_sf.append(popt_sf[2])

                            # for x
                            binwidth = 8
                            init_fit_smlm = scipy.stats.norm.fit(full_array_smlm[:, 0][kmeans_smlm.labels_ == kmean])
                            init_fit_sf = scipy.stats.norm.fit(full_array_sf[:, 0][kmeans_smlm.labels_ == kmean])

                            n_smlm, bins_smlm = np.histogram(full_array_smlm[:, 0][kmeans_smlm.labels_ == kmean],
                                                             bins=np.arange(min(full_array_smlm[:, 0]),
                                                                            max(full_array_smlm[:, 0]) + binwidth,
                                                                            binwidth))
                            n_sf, bins_sf = np.histogram(full_array_sf[:, 0][kmeans_smlm.labels_ == kmean],
                                                         bins=np.arange(min(full_array_sf[:, 0]),
                                                                        max(full_array_sf[:, 0]) + binwidth, binwidth))

                            popt_smlm, pcov_smlm = curve_fit(gauss, (bins_smlm[1:] + bins_smlm[:-1]) / 2, n_smlm,
                                                             p0=[max(n_smlm), init_fit_smlm[0], init_fit_smlm[1], 0],
                                                             maxfev=2000,
                                                             bounds=[[0, -np.inf, 0, 0],
                                                                     [np.inf, np.inf, np.inf, np.inf]])
                            stdx_smlm.append(popt_smlm[2])
                            popt_sf, pcov_sf = curve_fit(gauss, (bins_sf[1:] + bins_sf[:-1]) / 2, n_sf,
                                                         p0=[max(n_sf), init_fit_sf[0], init_fit_sf[1], 0],
                                                         maxfev=2000,
                                                         bounds=[[0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]])
                            stdx_sf.append(popt_sf[2])

                            # for y
                            binwidth = 8
                            init_fit_smlm = scipy.stats.norm.fit(full_array_smlm[:, 1][kmeans_smlm.labels_ == kmean])
                            init_fit_sf = scipy.stats.norm.fit(full_array_sf[:, 1][kmeans_smlm.labels_ == kmean])

                            n_smlm, bins_smlm = np.histogram(full_array_smlm[:, 0][kmeans_smlm.labels_ == kmean],
                                                             bins=np.arange(min(full_array_smlm[:, 1]),
                                                                            max(full_array_smlm[:, 1]) + binwidth,
                                                                            binwidth))
                            n_sf, bins_sf = np.histogram(full_array_sf[:, 0][kmeans_smlm.labels_ == kmean],
                                                         bins=np.arange(min(full_array_sf[:, 1]),
                                                                        max(full_array_sf[:, 1]) + binwidth, binwidth))

                            popt_smlm, pcov_smlm = curve_fit(gauss, (bins_smlm[1:] + bins_smlm[:-1]) / 2, n_smlm,
                                                             p0=[max(n_smlm), init_fit_smlm[0], init_fit_smlm[1], 0],
                                                             maxfev=2000,
                                                             bounds=[[0, -np.inf, 0, 0],
                                                                     [np.inf, np.inf, np.inf, np.inf]])
                            stdy_smlm.append(popt_smlm[2])
                            popt_sf, pcov_sf = curve_fit(gauss, (bins_sf[1:] + bins_sf[:-1]) / 2, n_sf,
                                                         p0=[max(n_sf), init_fit_sf[0], init_fit_sf[1], 0],
                                                         maxfev=2000,
                                                         bounds=[[0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]])
                            stdy_sf.append(popt_sf[2])
            except:
                _ = 1




        return [std_smlm, std_sf, stdx_smlm, stdx_sf, stdy_smlm,stdy_sf], \
               [angle_array_smlm,distance_array_smlm, angle_array_sf, distance_array_sf],\
               [mean_array_cluster0, mean_array_cluster1,  full_array_photons_cluster0_smlm, full_array_photons_cluster1_smlm, full_array_bg_cluster0_smlm, full_array_bg_cluster1_smlm],\
               [mean_array_cluster0_sf, mean_array_cluster1_sf,  full_array_photons_cluster0, full_array_photons_cluster1, full_array_bg_cluster0, full_array_bg_cluster1]

    def clustercrap(self,ds, ds2, dist_factor=4,minpoints = 30, torch=False):
        from photonpy import Context, PostProcessMethods

        with Context() as ctx:
            crlb = ds.crlb['pos']
            if torch == True:
                crlb = crlb + 0.15
            crlb = crlb[:, 0:2] *dist_factor
            pts = ds.pos[:, 0:2]
            dist = np.mean(crlb) * dist_factor

            clusterpos, cl_crlb, mapping = PostProcessMethods(ctx).ClusterLocs(pts, crlb, dist)

            plt.scatter(clusterpos[:, 0], clusterpos[:, 1], s=1, marker='o', label='Cluster mean')

            plt.legend()

            values = np.histogram(mapping, bins=np.size(clusterpos, 0))
            final_clusters = clusterpos[values[0] > minpoints]
            plt.figure(figsize=(30, 30))
            plt.scatter(pts[:, 0], pts[:, 1], c=mapping, s=1, marker='o')

            for k in range(len(final_clusters)):
                circle = plt.Circle((final_clusters[k, 0], final_clusters[k, 1]), dist, color='r', fill=False)
                plt.gca().add_artist(circle)
        list_ds = []
        pos3d = ds.pos
        avg_z = np.zeros(np.size(final_clusters, 0))
        avg_xy = np.zeros((np.size(final_clusters, 0),2))
        for qq in range(np.size(final_clusters, 0)):
            center = final_clusters[qq, :]
            temp = (pos3d[:, 0] - center[0]) ** 2 + (pos3d[:, 1] - center[1]) ** 2
            array_filter = temp <= dist ** 2
            avg_z[qq] = np.mean(pos3d[array_filter, 2])
            avg_xy[qq] = np.mean(pos3d[array_filter, 0:2], axis=0)
            list_ds.append(pos3d[array_filter,:])
        if ds2 != None:
            pos3d_ds2 = ds2.pos
            avg_z_ds2 = np.zeros(np.size(final_clusters, 0))
            avg_xy_ds2 = np.zeros((np.size(final_clusters, 0), 2))
            for qq in range(np.size(final_clusters, 0)):
                center = final_clusters[qq, :]
                temp = (pos3d_ds2[:, 0] - center[0]) ** 2 + (pos3d_ds2[:, 1] - center[1]) ** 2
                array_filter = temp <= dist ** 2
                avg_z_ds2[qq] = np.mean(pos3d_ds2[array_filter, 2])
                avg_xy_ds2[qq] = np.mean(pos3d_ds2[array_filter, 0:2],axis=0)
            z_bias = avg_z - avg_z_ds2
            xybias = avg_xy - avg_xy_ds2
            nan = np.isnan(z_bias)
            z_bias = z_bias[np.logical_not(nan)]
            avg_z = avg_z[np.logical_not(nan)]
            return avg_z, final_clusters, z_bias, avg_xy, xybias, list_ds
        else:
            return avg_z, final_clusters, 0, 0, 0, list_ds

    def extract_phase_pos_fixedroi(self, static_roi):

        pos1 = np.zeros((np.max(static_roi), 7))
        map_sumds1 = []
        phase1 = []

        for i in range(max(static_roi)):
            roi_id = self.sum_ds.roi_id_filtered
            set1 = static_roi == i
            joe = np.where(set1 == True)[0]
            # true index for non-filtered roi through frames.
            true_index = np.where(np.in1d(roi_id, joe))
            map_sumds1.append(true_index)
            pos = self.sum_ds.pos_filtered[true_index]
            phase = self.sum_ds.params_sine_filtered[true_index, 1]
            num_spots = np.size(pos, 1)
            if num_spots != 0:
                phase1.append(phase)

                pos1[i, 0] = num_spots
                pos1[i, 1:4] = np.mean(pos, axis=1)
                pos1[i, 4:7] = np.std(pos, axis=1)
            else:
                phase1.append(0)

        return phase1, pos1, np.array(map_sumds1)

    def compute_phase_shift_zstack(path, path2, cfg, maxlinking=3):
        from photonpy.simflux import SimfluxProcessor
        import trackpy
        import numpy as np
        import pandas as pd
        sp = SimfluxProcessor(path, cfg)
        sp2 = SimfluxProcessor(path2, cfg)

        sp.detect_rois(ignore_cache=False)
        sp2.detect_rois(ignore_cache=False)
        sp.gaussian_fitting()
        sp2.gaussian_fitting()
        # sp.view_rois()
        import matplotlib.pyplot as plt
        import scipy
        sp.fit_sine()
        sp2.fit_sine()

        sp.filter_onR(filter=0.05, min_mod=0.5)
        sp2.filter_onR(filter=0.05, min_mod=0.5)
        plt.show()
        params = np.append(np.append(sp.sum_ds.local_pos_filtered, sp.sum_ds.photons_filtered[:, None], axis=1),
                           sp.sum_ds.background_filtered[:, None], axis=1)
        trackdata = pd.DataFrame()
        trackdata['x'] = sp.sum_ds.pos_filtered[:, 0]
        trackdata['y'] = sp.sum_ds.pos_filtered[:, 1]
        trackdata['frame'] = sp.sum_ds.frame_filtered[:]
        trackdata['phase_pat1'] = sp.sum_ds.params_sine_filtered[:, 1, 0]
        trackdata['phase_pat2'] = sp.sum_ds.params_sine_filtered[:, 1, 1]
        trackdata['zpos'] = sp.sum_ds.pos_filtered[:, 2]
        trackdata['roid'] = sp.sum_ds.roi_id_filtered[:]

        trackdata2 = pd.DataFrame()
        trackdata2['x'] = sp2.sum_ds.pos_filtered[:, 0]
        trackdata2['y'] = sp2.sum_ds.pos_filtered[:, 1]
        trackdata2['zpos'] = sp2.sum_ds.pos_filtered[:, 2]
        trackdata2['frame'] = sp2.sum_ds.frame_filtered[:]
        trackdata2['phase_pat1'] = sp2.sum_ds.params_sine_filtered[:, 1, 0]
        trackdata2['phase_pat2'] = sp2.sum_ds.params_sine_filtered[:, 1, 1]
        trackdata2['roid'] = sp2.sum_ds.roi_id_filtered[:]

        # sp1 are even frames, sp2 are uneven frames
        trackdata['frame'] = trackdata['frame'] * 2
        trackdata2['frame'] = trackdata2['frame'] * 2 + 1
        combined_data = trackdata.append(trackdata2, ignore_index=True)
        trackpy.quiet()
        trackdata_comb = trackpy.link(combined_data, maxlinking, memory=3)

        # trackpy.plot_traj(trackdata_comb)
        # plt.show()
        shift_data = pd.DataFrame()
        for i in range(max(trackdata_comb['particle'] + 1)):
            particle = trackdata_comb[trackdata_comb['particle'] == i]
            odd_frames = particle[particle['frame'] % 2 != 0]
            even_frames = particle[particle['frame'] % 2 == 0]
            if even_frames.shape[0] != 0 and odd_frames.shape[0] != 0:
                temp = pd.DataFrame()
                temp['xdrift'] = [np.mean(odd_frames['x']) - np.mean(even_frames['x'])]
                temp['ydrift'] = [np.mean(odd_frames['y']) - np.mean(even_frames['y'])]
                temp['particle'] = [i]
                temp['zpos1_avg'] = np.mean(even_frames['zpos'])
                temp['zpos2_avg'] = np.mean(odd_frames['zpos'])
                temp['deltaz'] = np.mean(odd_frames['zpos']) - np.mean(even_frames['zpos'])
                temp['roi_ids_firstslice'] = [np.array(even_frames['roid'])]
                temp['roi_ids_secondslice'] = [np.array(odd_frames['roid'])]

                for name in ['phase_pat1', 'phase_pat2']:
                    if odd_frames.shape[0] < even_frames.shape[0]:
                        meanshift = np.zeros(odd_frames.shape[0])
                        for qq in range(odd_frames.shape[0]):
                            phase_odd = np.array(odd_frames[name])[qq]

                            shift = np.zeros(even_frames.shape[0])
                            for ww in range(even_frames.shape[0]):
                                phase_even = np.array(even_frames[name])[ww]
                                shift[ww] = phase_odd - phase_even
                                if abs(shift[ww]) > np.pi:
                                    if shift[ww] < 0:
                                        shift[ww] = shift[ww] + 2 * np.pi
                                    elif shift[ww] > 0:
                                        shift[ww] = shift[ww] - 2 * np.pi
                                    else:
                                        print('ERROR')
                            meanshift[qq] = np.mean(shift)
                        temp[name + '_shift'] = np.mean(meanshift)
                    if odd_frames.shape[0] >= even_frames.shape[0]:
                        meanshift = np.zeros(even_frames.shape[0])
                        for qq in range(even_frames.shape[0]):
                            phase_even = np.array(even_frames[name])[qq]

                            shift = np.zeros(odd_frames.shape[0])
                            for ww in range(odd_frames.shape[0]):
                                phase_odd = np.array(odd_frames[name])[ww]
                                shift[ww] = phase_odd - phase_even
                                if abs(shift[ww]) > np.pi:
                                    if shift[ww] < 0:
                                        shift[ww] = shift[ww] + 2 * np.pi
                                    elif shift[ww] > 0:
                                        shift[ww] = shift[ww] - 2 * np.pi
                                    else:
                                        print('ERROR')
                            meanshift[qq] = np.mean(shift)
                        temp[name + '_shift'] = np.mean(meanshift)
                temp['num_spots_even'] = even_frames.shape[0]
                temp['num_spots_odd'] = odd_frames.shape[0]
                shift_data = shift_data.append(temp)
        return shift_data, sp, sp2

    def view_rois_concatv2(self, pxstack, map1, calib):
        from photonpy.cpp.gaussian import GaussianPSFMethods, Gauss3D_Calibration

        pxstack = np.sum(pxstack, axis=1)

        pxconcat = np.zeros([len(map1), 2 * np.size(pxstack, 1), np.size(pxstack, 2)])
        for i in range(len(map1)):
            px = pxstack[self.sum_ds.roi_id_filtered[map1[i][0][0]]]

            gpm = GaussianPSFMethods(ctx=Context(debugMode=False))
            calib2 = Gauss3D_Calibration(calib['x'], calib['y'])
            p = np.zeros(5)
            p[:3] = self.sum_ds.local_pos_filtered[map1[i][0][0], :]
            p[3], p[4] = self.sum_ds.photons_filtered[map1[i][0][0]], self.sum_ds.background_filtered[map1[0][0][0]]

            psf_astig = gpm.CreatePSF_XYZIBg(self.roisize, calib2, cuda=True)

            pxconcat_optimalpsf = psf_astig.ExpectedValue(p)[0, :, :]
            pxconcat[i] = np.append(px, pxconcat_optimalpsf, axis=0)
        return pxconcat


def view_napari(mov):
    import napari

    with napari.gui_qt():
        napari.view_image(mov)


def set_plot_fonts():
    import matplotlib as mpl
    new_rc_params = {
    #    "font.family": 'Times',
        "font.size": 15,
    #    "font.serif": [],
        "svg.fonttype": 'none'} #to store text as text, not as path
    mpl.rcParams.update(new_rc_params)


def estimate_patterns(self, num_angle_bins=1,
                      num_phase_bins=10,
                      pitch_minmax_nm=[300, 1500],
                      p_ax=0,
                      fix_phase_shifts=None,
                      fix_depths=None,
                      show_plots=True,
                      phase_method=1,
                      dft_peak_search_range=0.02):
    self.estimate_angles(num_angle_bins, pitch_minmax_nm, dft_peak_search_range, debug_images=False, p_ax=p_ax)

    fr = np.arange(self.sum_ds.numFrames)
    frame_bins = np.array_split(fr, num_phase_bins)
    frame_bins = [b for b in frame_bins if len(b) > 0]

    k = self.mod['k'][:, :self.excDims]
    phase, depth, power = self.spotlist.estimate_phase_and_depth(k, self.pattern_frames, frame_bins,
                                                                 method=phase_method)
    phase_all, depth_all, power_all = self.spotlist.estimate_phase_and_depth(k, self.pattern_frames, [fr],
                                                                             method=phase_method)

    # store interpolated phase for every frame
    frame_bin_t = [np.mean(b) for b in frame_bins]
    self.phase_interp = np.zeros((len(fr), self.pattern_frames.size))
    for k in range(self.pattern_frames.size):
        phase[:, k] = unwrap_angle(phase[:, k])
        spl = InterpolatedUnivariateSpline(frame_bin_t, phase[:, k], k=2)
        self.phase_interp[:, k] = spl(fr)

    fig = plt.figure(figsize=figsize)
    styles = ['o', "x", "*", 'd']
    for ax, idx in enumerate(self.pattern_frames):
        for k in range(len(idx)):
            p = plt.plot(fr, self.phase_interp[:, idx[k]] * 180 / np.pi, ls='-')
            plt.plot(frame_bin_t, phase[:, idx[k]] * 180 / np.pi, ls='', c=p[0].get_color(),
                     marker=styles[ax % len(styles)], label=f"Phase {idx[k]} (axis {ax})")
    plt.legend()
    plt.title(f"Phases for {self.src_fn}")
    plt.xlabel("Frame number");
    plt.ylabel("Phase [deg]")
    plt.grid()
    plt.tight_layout()
    fig.savefig(self.resultprefix + "phases.png")
    if not show_plots: plt.close(fig)

    fig = plt.figure(figsize=figsize)
    for ax, idx in enumerate(self.pattern_frames):
        for k in range(len(idx)):
            plt.plot(frame_bin_t, depth[:, idx[k]], styles[ax % len(styles)], ls='-',
                     label=f"Depth {idx[k]} (axis {ax})")
    plt.legend()
    plt.title(f"Depths for {self.src_fn}")
    plt.xlabel("Frame number");
    plt.ylabel("Modulation Depth")
    plt.grid()
    plt.tight_layout()
    fig.savefig(self.resultprefix + "depths.png")
    if not show_plots: plt.close(fig)

    fig = plt.figure(figsize=figsize)
    for ax, idx in enumerate(self.pattern_frames):
        for k in range(len(idx)):
            plt.plot(frame_bin_t, power[:, idx[k]], styles[ax % len(styles)], ls='-',
                     label=f"Power {idx[k]} (axis {ax})")
    plt.legend()
    plt.title(f"Power for {self.src_fn}")
    plt.xlabel("Frame number");
    plt.ylabel("Modulation Power")
    plt.grid()
    plt.tight_layout()
    fig.savefig(self.resultprefix + "power.png")
    if not show_plots: plt.close(fig)

    # Update mod
    phase_std = np.zeros(len(self.mod))
    for k in range(len(self.mod)):
        ph_k = unwrap_angle(phase[:, k])
        self.mod['phase'][k] = phase_all[0, k]
        self.mod['depth'][k] = depth_all[0, k]
        self.mod['relint'][k] = power_all[0, k]
        phase_std[k] = np.std(ph_k)

    s = np.sqrt(num_phase_bins)
    for k in range(len(self.mod)):
        self.report(
            f"Pattern {k}: Phase {self.mod[k]['phase'] * 180 / np.pi:8.2f} (std={phase_std[k] / s * 180 / np.pi:6.2f}) " +
            f"Depth={self.mod[k]['depth']:5.2f} (std={np.std(depth[:, k]) / s:5.3f}) " +
            f"Power={self.mod[k]['relint']:5.3f} (std={np.std(power[:, k]) / s:5.5f}) ")

    # mod=self.spotlist.refine_pitch(mod, self.ctx, self.spotfilter, plot=True)[2]

    for angIndex in range(len(self.pattern_frames)):
        self.mod[self.pattern_frames[angIndex]]['relint'] = np.mean(self.mod[self.pattern_frames[angIndex]]['relint'])
        # Average modulation depth
        self.mod[self.pattern_frames[angIndex]]['depth'] = np.mean(self.mod[self.pattern_frames[angIndex]]['depth'])

    self.mod['relint'] /= np.sum(self.mod['relint'])

    if fix_depths:
        self.report(f'Fixing modulation depth to {fix_depths}')
        self.mod['depth'] = fix_depths

    self.report("Final modulation pattern parameters:")
    print_mod(self.report, self.mod, self.pattern_frames, self.pixelsize)

    with open(self.mod_fn, "wb") as f:
        pickle.dump((self.mod, self.phase_interp), f)

    med_sum_I = np.median(self.sum_ds.data.IBg[:, :, 0].sum(1))
    lowest_power = np.min(self.mod['relint'])
    depth = self.mod[np.argmin(self.mod['relint'])]['depth']
    median_intensity_at_zero = med_sum_I * lowest_power * (1 - depth)
    self.report(
        f"Median summed intensity: {med_sum_I:.1f}. Median intensity at pattern zero: {median_intensity_at_zero:.1f}")

