#include <openvr.h>
#include "glitter.hpp"

class VRApplication {
public:
	VRApplication();
	bool Init();
	void Shutdown();
	bool setRenderTargetSize(uint32_t* width, uint32_t* height);

	glm::mat4 getProjectionMatrix(vr::Hmd_Eye nEye);
	glm::mat4 getMatrixPoseEye(vr::Hmd_Eye nEye);
	bool getMatrixPoseHead(glm::mat4& mat);
	void submitFrame(GLuint leftTexutre, GLuint rightTexture);
	bool getGaze(glm::vec3& gaze, double& newTimestamp);

	vr::IVRSystem* m_pHMD;
	float m_fNearClip;
	float m_fFarClip;
	bool mEnableEye = false;

private:
	vr::TrackedDevicePose_t m_rTrackedDevicePose[vr::k_unMaxTrackedDeviceCount];
};
