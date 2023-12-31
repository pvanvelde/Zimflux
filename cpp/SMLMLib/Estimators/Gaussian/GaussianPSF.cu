// 2D Gaussian PSF models
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "MemLeakDebug.h"
#include "dllmacros.h"
#include "palala.h"
#include "Estimators/Estimation.h"
#include "StringUtils.h"
#include "MathUtils.h"

#include "CameraCalibration.h"

#include "GaussianPSF.h"
#include "GaussianPSFModels.h"

#include "Estimators/Estimator.h"
#include "Estimators/EstimatorImpl.h"



template<typename TModel> 
Estimator* CreateGaussianPSF(int roisize, bool cuda, TModel::Calibration calib, param_limits_t limits, int numconst) {
	if (cuda) {
		auto* cuda_psf = new SimpleCalibrationPSF< cuPSFImpl< TModel, decltype(calib) > >
			(calib, { roisize,roisize }, limits, numconst);
		return new cuEstimatorWrapper(cuda_psf);
	}
	else {
		return new SimpleCalibrationPSF< PSFImpl< TModel, decltype(calib) > >
			(calib, { roisize,roisize }, limits, numconst);
	}
}

CDLL_EXPORT Estimator* Gauss2D_CreatePSF_XYZIBg(int roisize, const Gauss3D_Calibration& calib, bool cuda, Context* ctx)
{
	try {
		Estimator* psf;
		typedef Gauss2D_Model_XYZIBg Model;

		param_limits_t limits = {
			std::vector<float>{Gauss2D_Border, Gauss2D_Border, calib.minz, 10.0f, 1.0f},
			std::vector<float>{roisize - 1 - Gauss2D_Border, roisize - 1 - Gauss2D_Border, calib.maxz, 1e9f, 1e9f}
		};

		psf = CreateGaussianPSF<Model>(roisize, cuda, calib, limits, Model::NumConstants);

		if (ctx) psf->SetContext(ctx);
		return psf;
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}


CDLL_EXPORT Estimator* Gauss2D_CreatePSF_XYITiltedBg(int roisize, bool cuda, Context* ctx)
{
	try {
		Estimator* psf;

		param_limits_t limits = {
			std::vector<float>{Gauss2D_Border, Gauss2D_Border, 10.0f, 1.0f, -1e3f, -1e3f},
			std::vector<float>{roisize - 1 - Gauss2D_Border, roisize - 1 - Gauss2D_Border, 1e9f, 1e9f, 1e3f, 1e3f}
		};
		psf = CreateGaussianPSF<Gauss2D_Model_XYITiltedBg>(roisize, cuda, {}, limits, Gauss2D_Model_XYITiltedBg::NumConstants);

		if (ctx) psf->SetContext(ctx);
		return psf;
	}
	catch (const std::runtime_error& e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}

CDLL_EXPORT Estimator* Gauss2D_CreatePSF_XYIBg(int roisize, float sigmaX, float sigmaY, bool cuda, Context* ctx)
{
	try {
		Estimator* psf;

		param_limits_t limits = {
			std::vector<float>{Gauss2D_Border, Gauss2D_Border, 10.0f, 1.0f},
			std::vector<float>{roisize - 1 - Gauss2D_Border, roisize - 1 - Gauss2D_Border, 1e9f, 1e9f}
		};

		psf = CreateGaussianPSF<Gauss2D_Model_XYIBg>(roisize, cuda, { sigmaX,sigmaY }, limits, Gauss2D_Model_XYIBg::NumConstants);

		if (ctx) psf->SetContext(ctx);
		return psf;
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}

CDLL_EXPORT Estimator* Gauss2D_CreatePSF_XYIBgConstSigma(int roisize, bool cuda, Context* ctx)
{
	try {
		Estimator* psf;

		param_limits_t limits = {
			std::vector<float>{Gauss2D_Border, Gauss2D_Border, 10.0f, 1.0f},
			std::vector<float>{roisize - 1 - Gauss2D_Border, roisize - 1 - Gauss2D_Border, 1e9f, 1e9f}
		};

		psf = CreateGaussianPSF<Gauss2D_Model_XYIBgConstSigma>(roisize, cuda, {}, limits, 
			Gauss2D_Model_XYIBgConstSigma::NumConstants);

		if (ctx) psf->SetContext(ctx);
		return psf;
	}
	catch (const std::runtime_error& e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}



CDLL_EXPORT Estimator* Gauss2D_CreatePSF_XYIBgSigma(int roisize, float initialSigma, bool cuda, Context* ctx)
{
	try {
		Estimator* psf;

		param_limits_t limits = {
			std::vector<float>{Gauss2D_Border, Gauss2D_Border, 10.0f, 1.0f, Gauss2D_MinSigma},
			std::vector<float>{roisize-1-Gauss2D_Border, roisize - 1 - Gauss2D_Border, 1e9f, 1e9f,roisize * 2.0f}
		};
		psf = CreateGaussianPSF<Gauss2D_Model_XYIBgSigma>(roisize, cuda, initialSigma, limits,
			Gauss2D_Model_XYIBgSigma::NumConstants);

		if (ctx) psf->SetContext(ctx);
		return psf;
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}



CDLL_EXPORT Estimator* Gauss2D_CreatePSF_XYIBgSigmaXY(int roisize, float initialSigmaX, float initialSigmaY, bool cuda, Context* ctx)
{
	try {
		Estimator* psf;
		Vector2f initialSigma = { initialSigmaX,initialSigmaY };

		param_limits_t limits = {
			std::vector<float>{Gauss2D_Border, Gauss2D_Border, 10.0f, 1.0f, Gauss2D_MinSigma,Gauss2D_MinSigma},
			std::vector<float>{roisize - 1 - Gauss2D_Border, roisize - 1 - Gauss2D_Border, 1e9f, 1e9f, roisize * 2.0f, roisize * 2.0f}
		};
		psf = CreateGaussianPSF<Gauss2D_Model_XYIBgSigmaXY>(roisize, cuda, initialSigma, limits, 0);

		if (ctx) psf->SetContext(ctx);
		return psf;
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}

CDLL_EXPORT Estimator* Gauss2D_CreatePSF_XYITiltedBgSigmaXY(int roisize, float initialSigmaX, float initialSigmaY, bool cuda, Context* ctx)
{
	try {
		Estimator* psf;
		Vector2f initialSigma = { initialSigmaX,initialSigmaY };

		param_limits_t limits = {
			std::vector<float>{Gauss2D_Border, Gauss2D_Border, 10.0f, 1.0f,-1e6f,-1e6f, Gauss2D_MinSigma,Gauss2D_MinSigma},
			std::vector<float>{roisize - 1 - Gauss2D_Border, roisize - 1 - Gauss2D_Border, 1e9f, 1e9f, 1e6f,1e6f, roisize * 2.0f, roisize * 2.0f}
		};
		psf = CreateGaussianPSF<Gauss2D_Model_XYITiltedBgSigmaXY>(roisize, cuda, initialSigma, limits, 0);

		if (ctx) psf->SetContext(ctx);
		return psf;
	}
	catch (const std::runtime_error& e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}



CDLL_EXPORT void Gauss2D_EstimateIntensityBg(const float* imageData, Vector2f *IBg, Vector2f* IBg_crlb,
	int numspots, const Vector2f* xy, const Int2* roipos, const float* sigma, int imgw, int maxIterations, bool cuda)
{
	std::vector<float> psf_space(imgw*imgw*numspots);
	LevMarSettings<2> settings({ 1.0f,0.0f }, { 1e9,1e9 }, { 1,1 }, maxIterations, false);

	palala_for(numspots, cuda, 
		PLL_FN(int i, float* psf_space, const Int2* roipos, const float* imageData, Vector2f *IBg, Vector2f* IBg_crlb, const Vector2f* xy, const float* sigma) {
		const float* spot_img = &imageData[imgw*imgw*i];

		float *psf = &psf_space[imgw*imgw*i];
		IntensityBgModel::ComputeGaussianPSF({ sigma[i],sigma[i] }, xy[i][0], xy[i][1], imgw, psf);
		IntensityBgModel model({ imgw,imgw }, psf);

		auto r = LevMarOptimize(spot_img, { 1,0 }, model, settings);
		IBg[i] = r.estimate;
		IBg_crlb[i] = ComputeCRLB(ComputeFisherMatrix(model, r.estimate));
	}, psf_space,
		const_array(roipos, numspots),
		const_array(imageData, numspots*imgw*imgw),
		out_array(IBg, numspots),
		out_array(IBg_crlb, numspots),
		const_array(xy, numspots),
		const_array(sigma, numspots));
}





