
import numpy as np
import matplotlib.pyplot as plt
import pickle
from photonpy import Context, Dataset, GaussianPSFMethods, Gauss3D_Calibration, EstimQueue
import time
import tqdm
import photonpy.cpp.spotdetect as spotdetect
from photonpy.cpp.simflux import SIMFLUX, CFSFEstimator

import photonpy.utils.multipart_tiff as read_tiff
import photonpy.smlm.process_movie as process_movie

from .simflux import SimfluxProcessor, equal_cache_cfg, save_cache_cfg


class CombinedFrameSFProcessor (SimfluxProcessor):
    def __init__(self, src_fn, cfg, **args):
        super().__init__(src_fn, cfg, **args)
        
        self.patternsPerFrame = cfg['patternsPerFrame']
        self.use3D = cfg['use3D'] if 'use3D' in cfg else False
        if self.use3D:
            self.wavelength = cfg['wavelength']
        
        if 'gauss3DCalib' in cfg:
            self.psfCalib = Gauss3D_Calibration.from_file(cfg['gauss3DCalib'])
        
    def cfsf_create_estimator(self, useModulation, ctx) -> CFSFEstimator:
        psf = self.create_psf(ctx)
            
        sfe = SIMFLUX(ctx).CreateCFSFEstimator(psf, self.offsets, self.patternsPerFrame,
                                                    simfluxMode=useModulation)
        return sfe

    def create_psf(self, ctx, modulated=False): #override SimfluxProcessor.create_psf
        gpm = GaussianPSFMethods(ctx)
        if self.use3D:
            psf = gpm.CreatePSF_XYZIBg(self.roisize, self.psfCalib, True)
        else:
            psf = gpm.CreatePSF_XYIBg(self.roisize, self.sigma, True)

        if modulated:
            return SIMFLUX(ctx).CreateCFSFEstimator(psf, self.offsets, self.patternsPerFrame,
                                                        simfluxMode=True)
        
        return psf
    
    def cfsf_template_image(self,ctx):
        """
        Generate template for spot detection. 
        Currently Z position is just set to 0. 
        TODO: Z Stack template for better 3D detection in longer Z ranges
        """
        with self.cfsf_create_estimator(False,ctx) as sfe:
            if self.use3D:
                params = [self.roisize/2,self.roisize/2,0,1,0]
            else:
                params = [self.roisize/2,self.roisize/2,1,0]

            p = sfe.ExpandIntensities([params])
            ev = sfe.ExpectedValue(p)
            return ev[0]

    def cfsf_view_rois(self, maxspots=1000, indices=None):
        ri, pixels = process_movie.load_rois(self.rois_fn)
        
        if indices is not None:
            indices = self.roi_indices[indices]
        else:
            indices = self.roi_indices
            
        n = min(maxspots, len(indices))
        px = pixels[indices[:n]]
            
        with Context(debugMode=self.debugMode) as ctx:
            with self.cfsf_create_estimator(useModulation=False,ctx=ctx) as sfe:
                ev = sfe.ExpectedValue(self.sfsf_qr.estim[indices[:n]])

        import napari
        with napari.gui_qt():
            napari.view_image( np.concatenate([px, ev],-1))
            
        return px, ev
    
    def process(self, spotfilter, chisq_threshold):
        idx = 0
        last_done = 0
        
        sel_indices = self.spotlist.get_filtered_spots(spotfilter, self.mod)

        print(flush=True)        
        with Context(debugMode=self.debugMode) as ctx:
            with self.cfsf_create_estimator(True,ctx) as sfe, tqdm.tqdm(total=len(sel_indices))  as pb:
                #sfe.SetLevMarParams(lambda_=-1, iterations=30)
                
                param_names = sfe.param_names
                
                with EstimQueue(sfe, batchSize=1024, numStreams=3, 
                                maxQueueLenInBatches=5, keepSamples=False) as q:
                    
                    #all = np.arange(len(self.roi_indices))
                    for roipos, pixels, block_indices  in self.selected_roi_source(sel_indices):
                        initial = np.zeros((len(pixels),sfe.numparams))
                        if True:
                            initial[:,:self.sum_ds.dims] = self.sum_ds.local_pos[block_indices]
                            initial[:,-2] = self.sum_ds.photons[block_indices]
                            initial[:,-1] = self.sum_ds.background[block_indices]
                        else:
                            initial[:,:2] = self.roisize/2
                            initial[:,-2] = pixels.reshape((len(pixels),-1)).sum(1)
                            initial[:,-1] = 0
                        
                        roi_mod = self.mod_per_spot(self.sum_ds.frame[block_indices])
                        #roi_mod = np.repeat([self.mod.view(np.float32)],len(pixels),0)
                        
                        q.Schedule(pixels, 
                                   ids=np.arange(len(pixels))+idx,
                                   initial=initial,
                                   constants=roi_mod,
                                   roipos=roipos)
                        idx += len(pixels)
                        
                        new_done = q.GetResultCount()
                        pb.update(new_done-last_done)
                        last_done = new_done
        
                    q.Flush()
                    while True:
                        new_done = q.GetResultCount()
                        pb.update(new_done-last_done)
                        last_done = new_done
                        
                        if last_done == len(sel_indices):
                            break
                        time.sleep(0.1)
                        
                    qr = q.GetResults()
                    qr.SortByID()
                    
        # not sure yet if i need to allow dataset to have roipos with more than 2 dimensions
        qr.roipos = qr.roipos[:,-2:]

        ds = Dataset.fromQueueResults(qr, self.imgshape)
        ds.frame = self.sum_ds.frame[sel_indices]

        border= self.roisize/4
        unfilteredLen = len(qr.estim)
        ok_indices = qr.FilterXY(border, border, self.roisize-1-border, self.roisize-1-border)
        atborder = unfilteredLen - len(ok_indices)
        if atborder>0: print(f"Removing {atborder} unconverged spots. (Out of {unfilteredLen})")
        
        ds = ds[ok_indices]
        
        chisq_threshold *= sfe.samplecount
        print(f"Median Chi-square value: {np.median(ds.chisq):.1f}")
        print(f"Chisq filter ({chisq_threshold}) removes {(ds.chisq >= chisq_threshold).sum()} spots")
        self.sf_ds = ds[ds.chisq<chisq_threshold]
                
        fn = self.resultprefix+'sfsf_fits.hdf5'
        self.sf_ds.save(fn)
        
    def load_ds(self):
        fn = self.resultprefix+'sfsf_fits.hdf5'
        self.sf_ds = Dataset.load(fn)
        
    def set_offsets(self, offsets):
        if type(offsets) == str:
            with open(offsets, "rb") as f:
                _,self.offsets = pickle.load(f)
        else:
            self.offsets = offsets
                        
        # create a template to detect with
        self.cfg['dmpatOffsets'] = self.offsets
        
        
        

    def detect_rois(self, offsets, ignore_cache=False, roi_batch_size=20000,numStreams=3,
                    numSpotDetectThreads=1,chisq_threshold=4):
        self.chisq_threshold = chisq_threshold
        
        self.set_offsets(offsets)
        self.imgshape = read_tiff.tiff_get_image_size(self.src_fn)
                
        bgimg = np.zeros(self.imgshape)
        
        with Context(debugMode=self.debugMode) as ctx:
            templateImage = self.cfsf_template_image(ctx)
            
            plt.figure()
            plt.imshow(templateImage[0])    
            plt.title('Template image used for spot detection')
    
            sd = spotdetect.PSFCorrelationSpotDetector(templateImage, bgimg, self.threshold,
                                                       maxFilterSizeXY=self.cfg['sdXYFilterSize'],
                                                       bgFilterSize=self.cfg['sdBackgroundSigma'])
            
            # how many frames does a single spot consist of?
            sumframes = len(offsets)//self.patternsPerFrame
    
            if not equal_cache_cfg(self.rois_fn, self.cfg, self.src_fn) or ignore_cache:
                process_movie.detect_spots(sd, self._camera_calib(ctx), 
                                   read_tiff.tiff_read_file(self.src_fn, self.cfg['startframe'], self.maxframes), 
                                   sumframes, self.rois_fn, 
                                   batch_size = roi_batch_size, ctx=ctx,
                                   numThreads=numSpotDetectThreads)
                save_cache_cfg(self.rois_fn, self.cfg, self.src_fn)
    
            self.numrois = sum([len(ri) for ri,px in self._load_rois_iterator()])
            print(f"Num ROIs: {self.numrois}", flush=True)
            
            rois_info = []
    
            idx = 0
            last_done = 0

            with self.cfsf_create_estimator(True, ctx) as psf:
                param_names = psf.param_names
                nparams = psf.numparams  #number of params for modulated PSF

            with self.cfsf_create_estimator(False,ctx) as sfe, tqdm.tqdm(total=self.numrois)  as pb:
                #sfe.SetLevMarParams(lambda_=-1, iterations=30)
                                
                with EstimQueue(sfe, batchSize=1024, numStreams=numStreams, 
                                maxQueueLenInBatches=5, keepSamples=False) as q:
                    
                    for ri, pixels in self._load_rois_iterator():
                        initial = np.zeros((len(ri), nparams))
                        initial[:,:2] = self.roisize/2
                        initial[:,-2] = pixels.reshape((len(pixels),-1)).sum(1)
                        initial[:,-1] = 0
                        
                        sfe_initial = sfe.ExpandIntensities(initial)
                        
                        q.Schedule(pixels, ids=np.arange(len(ri))+idx,initial=sfe_initial)
                        idx += len(pixels)
                        
                        new_done = q.GetResultCount()
                        pb.update(new_done-last_done)
                        last_done = new_done
                        rois_info.append(ri)
    
                    q.Flush()
                    while True:
                        new_done = q.GetResultCount()
                        pb.update(new_done-last_done)
                        last_done = new_done
                        
                        if last_done == self.numrois:
                            break
                        time.sleep(0.1)
                        
                    qr = q.GetResults()
                    qr.SortByID()
                    
                    estim_crlb, IBg_crlb = sfe.SeparateIntensities(qr.CRLB())
                    estim, IBg = sfe.SeparateIntensities(qr.estim)
                    rois_info = np.concatenate(rois_info)
                    
        roi_indices = None
        
        """
        if self.use3D:
            n = len(estim)
            roi_indices = (estim[:,2] > self.psfCalib.zrange[0]*0.9) & (estim[:,2] < self.psfCalib.zrange[1]*0.9)
            print(f"Filtering Z position: {n-roi_indices.sum()} spots removed")
        """

        self.sfsf_qr = qr
        self._store_IBg_fits(estim, np.concatenate([IBg,IBg_crlb],-1), qr.chisq, estim_crlb, rois_info, param_names)

    def crlb_map_Z(self, zrange=[-1,1], intensity=None, bg=None):
        self.load_mod()
        
        if intensity is None:
            intensity = np.median(self.sum_ds.photons)
            
        if bg is None:
            bg = np.median(self.sum_ds.background)
            
        print(f"Computing CRLB map for intensity={intensity:.1f}, bg={bg:.1f}")

        with Context() as ctx:
            W = 100

            sf_psf = self.cfsf_create_estimator(useModulation=True, ctx=ctx)
            #zrange = self.psfCalib.zrange

            xr = np.linspace(self.roisize/4,3*self.roisize/4,W)
            zr = np.linspace(zrange[0], zrange[1],W)

            X,Z = np.meshgrid(xr,zr)

            coords = np.zeros((W*W,sf_psf.numparams))
            coords[:,0] = X.flatten()
            coords[:,1] = self.roisize/2
            coords[:,2] = Z.flatten()
            coords[:,-2] = intensity
            coords[:,-1] = bg

            coords_ = coords*1
            coords_[:,-1] /= sf_psf.sampleshape[0] # bg should be distributed over all frames
            mod_ = np.repeat([self.mod.view(np.float32)], len(coords), 0)
            sf_crlb = sf_psf.CRLB(coords_, constants=mod_)
            
            psf = self.create_psf(ctx)
            um_crlb = psf.CRLB(coords)
        
        scale = [self.pixelsize, self.pixelsize, 1000, 1, 1]

        fig,ax = plt.subplots(1,2,sharey=True)
        #plt.figure()
        im = ax[0].imshow(sf_crlb[:,2].reshape((W,W)) * 1000, 
                   extent=(xr[0]*scale[0],xr[-1]*scale[0],zr[0]*scale[2],zr[-1]*scale[2]), 
                   origin='lower',
                   aspect= 'auto')
        ax[0].set_title('CRLB Z [nm]')

        ax[1].imshow(sf_crlb[:,0].reshape((W,W)) * 1000, 
                   extent=(xr[0]*scale[0],xr[-1]*scale[0],zr[0]*scale[2],zr[-1]*scale[2]), 
                   origin='lower',
                   aspect= 'auto')
        ax[1].set_title('CRLB X [nm]')

        fig.colorbar(im, ax=ax)

        #ax[1].imshow(IFmap[:,1].reshape((W,W)))
        #ax[1].set_title('Improvement Factor Y')
        
        #IF = np.mean(g2d_crlb/sf_crlb,0)
        scale = [self.pixelsize, self.pixelsize, 1000, 1, 1]
        crlb = sf_crlb.mean(0) * scale
        print(f"SF CRLB: X:{crlb[0]:.1f} Y:{crlb[1]:.1f} Z:{crlb[2]:.1f}")
        crlb = um_crlb.mean(0) * scale
        print(f"SMLM CRLB: X:{crlb[0]:.1f} Y:{crlb[1]:.1f} Z:{crlb[2]:.1f}")
        
        #print(f"Improvement factor X: {IF[0]:.3f}, Y: {IF[1]:.3f}")

    def test_estimator_3D(self, intensities=[1000], bg=30, zsteps = 100, perzstep=400,
                          show_napari=False, zrange=[-0.5, 0.5]):
        """
        Test if the estimator can reach its CRLB
        """
        with Context() as ctx:
            
            sfe = self.cfsf_create_estimator(True, ctx)
            
            zsmp = perzstep # samples per z value
            errs = []
            crlbs = []
            zrange_ = np.linspace(zrange[0],zrange[1], zsteps)
            for i,intensity in tqdm.tqdm(enumerate(intensities), total=len(intensities)):
                W = self.roisize/4
                params = np.zeros((zsteps*zsmp, sfe.numparams))
                params[:,:2] = self.roisize/2 + np.random.uniform(-W/2,W/2,size=(zsteps*zsmp,2))
                params[:,2] = zrange_.repeat(zsmp)
                params[:,3] = intensity
                params[:,4] = bg/sfe.sampleshape[0]
                mod = np.repeat([self.mod.view(np.float32)], len(params), 0)
                
                smp = sfe.GenerateSample(params, constants=mod)
                crlb = sfe.CRLB(params, constants=mod)
            
                initial = params*np.random.uniform(0.8,1.2,size=params.shape)
                #initial[:,:2] = self.roisize/2
                #initial[:,2] = 0
                initial[:,3] = smp.reshape((len(smp),-1)).sum(1)
                initial[:,4] = 0
                estim = sfe.Estimate(smp, constants=mod, initial=initial)[0]
                
                if show_napari:
                    fitted = sfe.ExpectedValue(estim, constants=mod)
                    
                    import napari
                    with napari.gui_qt():
                        napari.view_image( np.concatenate([smp, fitted],-1))

                err = (estim-params).reshape((zsteps, zsmp, -1))
                errs.append(err.std(1))
                crlbs.append(crlb.reshape(zsteps,zsmp,-1).mean(1))
            
            scale = [self.pixelsize, self.pixelsize, 1000, 1, 1]
            colors = ['m', 'b', 'g']
            plt.figure(figsize=(8,6))
            for i in range(len(intensities)):
                I = intensities[i]
                
                for k in range(3):
                    plt.plot(zrange_, errs[i][:,k] * scale[k], 'o', ms=4, color=colors[k], label=f'$\sigma_{sfe.param_names[k]}$ - I={I}')
                    plt.plot(zrange_, crlbs[i][:,k] * scale[k], '--', color=colors[k], label=f'CRLB {sfe.param_names[k]} - I={I}')
            plt.xlabel('Z Position [$\mu$m]')
            plt.ylabel('Precision [nm]')
            plt.legend()
            plt.title(f'Comparing estimation precision to CRLB (Samples per Z step = {zsmp})')
            plt.savefig('crlb-zrange.png')

    def projectZ(self, freq_minmax_nm):# period_minmax_nm = [200, 1000]):
        with Context() as ctx:
            g = GaussianPSFMethods(ctx)
            
            #height = np.max(self.sum_ds.pos[:,2]) - np.min(self.sum_ds.pos[:,2])
            xyscale = 0.05
            width = int(np.max(self.sum_ds.imgshape)*1 * self.pixelsize * xyscale)
            
            zscale = 0.1
            
            pos = self.sum_ds.pos

            n = len(pos)
            indices = (pos[:,2] > self.psfCalib.zrange[0]*0.9) & (pos[:,2] < self.psfCalib.zrange[1]*0.9)
            print(f"Filtering Z position: {n-indices.sum()}/{n} spots removed")
            
            pos = pos[indices]
            
            xyfreq = np.fft.fftshift(np.fft.fftfreq(width) * 2*np.pi)
            zfreq = np.fft.fftshift(np.fft.fftfreq(width) * 2*np.pi)
            XYFreq, ZFreq = np.meshgrid(xyfreq,zfreq)
            
            #freq_minmax_nm = np.array(period_minmax_nm)[::-1]
            
            print(freq_minmax_nm)

            mask = ((np.abs(XYFreq) >= freq_minmax_nm[0]) & (np.abs(XYFreq) < freq_minmax_nm[1]) & 
                    (np.abs(ZFreq) >= freq_minmax_nm[0]) & (np.abs(ZFreq) < freq_minmax_nm[1]))
                
            plt.imsave(self.resultsdir + "xyz mask.png", mask)
    
            for i, pflist in enumerate(self.pattern_frames):
                
                f_sum = np.zeros((width,width))
                for j, pf in enumerate(pflist):
                    kxy = self.mod['k'][pf,:2]
                    kxy /= np.sqrt((kxy**2).sum())
                    print(kxy)

                    img = np.zeros((width,width),  dtype=np.float64)

                    spots = np.ones((len(pos), 5), dtype=np.float32)
                    spots[:, 0] = (pos[:,:2] * self.pixelsize * kxy[None]).sum(1) * xyscale
                    spots[:, 1] = pos[:,2] * 1000 * zscale
                    spots[:, 4] = self.IBg[indices][:,pf,0]
                    spots[:, :2] = spots[:,:2] - np.mean(spots[:,:2],0) + width/2
                    g.Draw(img, spots)
                    
                    plt.imsave(self.resultsdir + f"XYZ_render-ep{pf}.png", img/np.max(img))
                    
                    f_img = ctx.smlm.FFT2(img)
                    f_sum += np.abs(f_img)

                f_sum = np.fft.fftshift(f_sum)
                f_sum = f_sum / np.sum(f_sum) 
                f_sum[~mask] = 0
                plt.imsave(self.resultsdir + f"XYZ_fft-angle{i}.png", f_sum)
                            
            return xyfreq
