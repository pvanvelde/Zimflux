# -*- coding: utf-8 -*-


import ctypes
import numpy as np

import numpy.ctypeslib as ctl

from . import gaussian
from .context import Context
from .estimator import Estimator
from .calib import sCMOS_Calib

Theta = ctypes.c_float * 4
FisherMatrix = ctypes.c_float * 16
Modulation = ctypes.c_float * 4



class CFSFEstimator(Estimator):
    def __init__(self, ctx, inst, psf:Estimator, offsetsXYZ, patternsPerFrame):
        super().__init__(ctx,inst)
        self.psf = psf
        self.offsetsXYZ = offsetsXYZ
        self.patternsPerFrame = patternsPerFrame
    
    def SumIntensities(self, params):
        """
        Translate from parameters for simfluxMode=False to simfluxMode=True.
        """
        params = np.ascontiguousarray(params,dtype=np.float32)
        assert np.array_equal(params.shape, [len(params), self.psf.numparams - 1 + len(self.offsetsXYZ)])
        nparams = np.zeros((len(params), self.psf.NumParams()))
        numcoords = nparams.shape[1] - 2  # xyz or xy

        nparams[:,:numcoords] = params[:,:numcoords]
        nparams[:,-1] = params[:,-1]  # bg
        nparams[:,numcoords] = np.sum(params[:,numcoords:-1], 1)
        return nparams
    
    def ExpandIntensities(self, params):
        """
        Opposite of SumIntensities. This takes a parameter vector with 1 intensity 
        and returns one with len(mod) intensities
        """
        params = np.ascontiguousarray(params,dtype=np.float32)
        assert np.array_equal(params.shape, [len(params), self.psf.numparams])
        numcoords = self.psf.numparams-2
        nparams = np.zeros((len(params), numcoords+1+len(self.offsetsXYZ)))
        nparams[:,:numcoords] = params[:,:numcoords]
        nparams[:,-1] = params[:,-1]
        nparams[:,numcoords:-1] = params[:,numcoords,None]
        return nparams
    
    def SeparateIntensities(self, params):
        numcoords = self.psf.numparams-2
        IBg = np.zeros((len(params),len(self.offsetsXYZ),2))
        IBg[:,:,1] = params[:,-1,None]
        IBg[:,:,0] = params[:,numcoords:-1]
        return self.SumIntensities(params), IBg

class SIMFLUX:
    
    modulationDType = np.dtype([('k', '<f4', (3,)), 
                                ('depth','<f4'),
                                ('phase','<f4'),
                                ('relint','<f4')
                                ])
    
    def __init__(self, ctx:Context):
        self.ctx = ctx
        smlmlib = ctx.smlm.lib
       # CDLL_EXPORT void SIMFLUX_DFT2D_Points(const Vector3f* xyI, int numpts, const Vector2f* k, 
       # int numk, Vector2f* output, bool useCuda);

        self._SIMFLUX_DFT2D_Points = smlmlib.SIMFLUX_DFT2D_Points
        self._SIMFLUX_DFT2D_Points.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # xyI
            ctypes.c_int32,  # numpts
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # k
            ctypes.c_int32,  # numk
            ctl.ndpointer(np.complex64, flags="aligned, c_contiguous"),  # output
            ctypes.c_bool # useCuda
        ]

        # CDLL_EXPORT void FFT(const cuFloatComplex* src, cuFloatComplex* dst, int batchsize, int siglen, int forward)
        self._FFT = smlmlib.FFT
        self._FFT.argtypes = [
            ctl.ndpointer(np.complex64, flags="aligned, c_contiguous"),  # src
            ctl.ndpointer(np.complex64, flags="aligned, c_contiguous"),  # dst
            ctypes.c_int32,  # batchsize
            ctypes.c_int32,  # numsigA
            ctypes.c_int32,  # forward
        ]
        
        self._SIMFLUX_ProjectPointData = smlmlib.SIMFLUX_ProjectPointData
        self._SIMFLUX_ProjectPointData.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # xyI
            ctypes.c_int32,  # numpts
            ctypes.c_int32,  # projectionWidth
            ctypes.c_float,  # scale
            ctypes.c_int32,  # numProjAngles
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # projectionAngles
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # output
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # output
        ]
        
        
#CDLL_EXPORT PSF* SIMFLUX2D_Gauss2D_PSF_Create(SIMFLUX_Modulation* mod, int num_patterns, 
# float sigma, int roisize, int numframes, bool simfluxMode, Context* ctx);
        
        self._SIMFLUX_Gauss2D_CreateEstimator = smlmlib.SIMFLUX_Gauss2D_CreateEstimator
        self._SIMFLUX_Gauss2D_CreateEstimator.argtypes= [
                ctypes.c_int, # numpatterns
                ctypes.c_float, # sigma_x
                ctypes.c_float, # sigma_y
                ctypes.c_int, # roisize
                ctypes.c_int, # nframes
                ctypes.c_void_p # context
                ]
        self._SIMFLUX_Gauss2D_CreateEstimator.restype = ctypes.c_void_p


        
        self._SIMFLUX_Gauss2DAstig_CreateEstimator = smlmlib.SIMFLUX_Gauss2DAstig_CreateEstimator
        self._SIMFLUX_Gauss2DAstig_CreateEstimator.argtypes= [
                ctypes.c_int, # numpatterns
                ctypes.POINTER(gaussian.Gauss3D_Calibration), 
                ctypes.c_int, # roisize
                ctypes.c_int, # nframes
                ctypes.c_void_p # context
                ]
        self._SIMFLUX_Gauss2DAstig_CreateEstimator.restype = ctypes.c_void_p


        self._SIMFLUX_GenerateROIExtractionList = smlmlib.SIMFLUX_GenerateROIExtractionList
        self._SIMFLUX_GenerateROIExtractionList.argtypes= [
                ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # startframes
                ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # ontime
                ctypes.c_int, #maxresults
                ctypes.c_int, # numlinked
                ctypes.c_int, # numpatterns
                ctl.ndpointer(np.int32, flags="aligned, c_contiguous")  # results
                ]
        


    def GenerateROIExtractionList(self, startframes, ontime, numpatterns):
        """
        returns linkedIndex, numroi and firstframe
        """
        maxresults = np.sum(ontime)//numpatterns
        numlinked = len(startframes)
        startframes=  np.ascontiguousarray(startframes,dtype=np.int32)
        ontime = np.ascontiguousarray(ontime, dtype=np.int32)
        results = np.zeros((maxresults,3),dtype=np.int32)
        resultcount = self._SIMFLUX_GenerateROIExtractionList(startframes,ontime,maxresults,numlinked,numpatterns,results)
        results =results[:resultcount]
        return results[:,0],results[:,1],results[:,2]

    
    def CreateEstimator_Gauss2D(self, calib, num_patterns, roisize, 
                                    numframes, simfluxEstim=True) -> Estimator:
        """
        Each spot requires 6 * numPatterns constants describing the modulation patterns:
        a matrix with shape [numExcitationPatterns, 6].
        
        if calib is Gauss3D_Calibration, this functions returns a 3D estimator using 2D Astigmatic gaussian fit
        """            
        if type(calib) == gaussian.Gauss3D_Calibration:
            inst = self._SIMFLUX_Gauss2DAstig_CreateEstimator(num_patterns, calib, roisize, numframes, self.ctx.inst)
        else:
            if np.isscalar(calib):
                sigma_x, sigma_y = calib,calib
            else:
                sigma_x, sigma_y = calib
            
            inst = self._SIMFLUX_Gauss2D_CreateEstimator(num_patterns, sigma_x, sigma_y,
                                                      roisize, numframes, 
                                                      self.ctx.inst if self.ctx else None)
        return Estimator(self.ctx,inst)
        
            
    
    def CreateTIRFScanEstimator(self, calib, num_patterns, roisize) -> Estimator:
        """
        Each spot requires 2 * numPatterns constants describing the modulation patterns:
        a matrix with shape [numExcitationPatterns, 2].
        """            
        
        assert type(calib) == gaussian.Gauss3D_Calibration
        inst = self._TIRFScan_CreateEstimator(num_patterns, calib, roisize, self.ctx.inst)
        return Estimator(self.ctx,inst)
        

    # Convert an array of phases to an array of alternating XY modulation parameters
    def phase_to_mod(self, phases, omega, depth=1):
        mod = np.zeros((*phases.shape, 5), dtype=np.float32)
        mod[..., 0::2, 0] = omega  # kx
        mod[..., 1::2, 1] = omega  # ky
        mod[..., 2] = depth
        mod[..., 3] = phases
        mod[..., 4] = 1/len(mod)
        return mod

    
    def SIMFLUX_DFT2D_Points(self, xyI, k, useCuda=True):
        xyI = np.ascontiguousarray(xyI, dtype=np.float32)
        numpts = len(xyI)
        k = np.ascontiguousarray(k, dtype=np.float32)
        output = np.zeros( len(k), dtype=np.complex64)
        self._SIMFLUX_DFT2D_Points(xyI, numpts, k, len(k), output, useCuda)
        return output

    # CDLL_EXPORT void SIMFLUX_ProjectPointData(const Vector3f *xyI, int numpts, int projectionWidth,
    # 	float scale, int numProjAngles, const float *projectionAngles, float* output)
    def ProjectPoints(self, xyI, projectionWidth, scale, projectionAngles):
        numProjAngles = len(projectionAngles)
        assert xyI.shape[1] == 3
        xyI = np.ascontiguousarray(xyI, dtype=np.float32)
        output = np.zeros((numProjAngles, projectionWidth), dtype=np.float32)
        shifts = np.zeros((numProjAngles), dtype=np.float32)

        self._SIMFLUX_ProjectPointData(
            xyI,
            len(xyI),
            projectionWidth,
            scale,
            numProjAngles,
            np.array(projectionAngles, dtype=np.float32),
            output,
            shifts,
        )
        return output, shifts

    ##CDLL_EXPORT void FFT(const cuFloatComplex* src, cuFloatComplex* dst, int batchsize, int siglen, int forward)

    def FFT(self, src, forward=True):
        batchsize = len(src)
        src = np.ascontiguousarray(src, dtype=np.complex64)
        dst = np.zeros(src.shape, dtype=np.complex64)
        self._FFT(src, dst, batchsize, src.shape[1], forward)
        return dst
