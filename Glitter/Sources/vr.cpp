#include "vr.h"
#include "SRanipal.h"
#include "SRanipal_Eye.h"
#include "SRanipal_Enums.h"

VRApplication::VRApplication()
{
	m_pHMD = NULL;
	m_fNearClip = 1.0f;
	m_fFarClip = 40000.0f;
}

bool VRApplication::Init()
{
	// Loading the SteamVR Runtime
	vr::EVRInitError eError = vr::VRInitError_None;
	m_pHMD = vr::VR_Init(&eError, vr::VRApplication_Scene);

	if (eError != vr::VRInitError_None)
	{
		m_pHMD = NULL;
		printf("Unable to init VR runtime: %s", vr::VR_GetVRInitErrorAsEnglishDescription(eError));
		return false;
	}

	// Init compositor
	vr::EVRInitError peError = vr::VRInitError_None;
	if (!vr::VRCompositor())
	{
		printf("Compositor initialization failed. See log file for details\n");
		return false;
	}

	// Init eye tracking
	int error = ViveSR::anipal::Initial(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE, NULL);
	if (error == ViveSR::Error::WORK) { mEnableEye = true; printf("Successfully initialize Eye engine.\n"); }
	else if (error == ViveSR::Error::RUNTIME_NOT_FOUND) printf("please follows SRanipal SDK guide to install SR_Runtime first\n");
	else printf("Fail to initialize Eye engine. please refer the code %d of ViveSR::Error.\n", error);

	return true;
}

void VRApplication::Shutdown()
{
	if (m_pHMD)
	{
		vr::VR_Shutdown();
		m_pHMD = NULL;

		ViveSR::anipal::Release(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE);
	}
}

bool VRApplication::setRenderTargetSize(uint32_t* width, uint32_t* height)
{
	if (!m_pHMD) return false;

	m_pHMD->GetRecommendedRenderTargetSize(width, height);
	return true;
}

glm::mat4 VRApplication::getProjectionMatrix(vr::Hmd_Eye nEye)
{
	if (!m_pHMD) return glm::mat4();

	vr::HmdMatrix44_t mat = m_pHMD->GetProjectionMatrix(nEye, m_fNearClip, m_fFarClip);

	return glm::mat4(
		mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0],
		mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1],
		mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2],
		mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]
	);
}

glm::mat4 VRApplication::getMatrixPoseEye(vr::Hmd_Eye nEye)
{
	if (!m_pHMD) return glm::mat4();

	vr::HmdMatrix34_t mat = m_pHMD->GetEyeToHeadTransform(nEye);
	glm::mat4 matrixObj(
		mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.0f,
		mat.m[0][1], mat.m[1][1], mat.m[2][1], 0.0f,
		mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.0f,
		mat.m[0][3], mat.m[1][3], mat.m[2][3], 1.0f
	);

	return glm::inverse(matrixObj);
}

bool VRApplication::getMatrixPoseHead(glm::mat4& mat)
{
	if (!m_pHMD) return false;

	vr::VRCompositor()->WaitGetPoses(m_rTrackedDevicePose, vr::k_unMaxTrackedDeviceCount, NULL, 0);

	if (m_rTrackedDevicePose[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid)
	{
		vr::HmdMatrix34_t hmdMat = m_rTrackedDevicePose[vr::k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking;

		glm::mat4 matrixObj(
			hmdMat.m[0][0], hmdMat.m[1][0], hmdMat.m[2][0], 0.0f,
			hmdMat.m[0][1], hmdMat.m[1][1], hmdMat.m[2][1], 0.0f,
			hmdMat.m[0][2], hmdMat.m[1][2], hmdMat.m[2][2], 0.0f,
			hmdMat.m[0][3], hmdMat.m[1][3], hmdMat.m[2][3], 1.0f
		);

		mat = glm::inverse(matrixObj);
		return true;
	}

	return false;
}

void VRApplication::submitFrame(GLuint leftTexutre, GLuint rightTexture)
{
	if (m_pHMD)
	{
		vr::Texture_t leftEyeTexture = { (void*)(uintptr_t)leftTexutre, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
		vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);
		vr::Texture_t rightEyeTexture = { (void*)(uintptr_t)rightTexture, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
		vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);
	}
}

bool VRApplication::getGaze(glm::vec3& gaze, double& newTimestamp) {
	const auto flag = 1 << ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_DIRECTION_VALIDITY;
	if (!mEnableEye) return false;

	ViveSR::anipal::Eye::EyeData eye_data;
	int result = ViveSR::anipal::Eye::GetEyeData(&eye_data);
	if (result == ViveSR::Error::WORK) {
		auto bitMask = eye_data.verbose_data.combined.eye_data.eye_data_validata_bit_mask;
		if ((bitMask & flag) != flag) return false;
		newTimestamp = eye_data.timestamp / 1000.0;
		float* data = eye_data.verbose_data.combined.eye_data.gaze_direction_normalized.elem_;
		gaze = glm::vec3{ -data[0], data[1], -data[2] };
		return true;
	}

	return false;
}
