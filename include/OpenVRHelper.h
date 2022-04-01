
#pragma once

#include <string>
#include <vector>

#include "openvr.h"
#include <glm/glm.hpp>

using std::string;
using std::vector;
using glm::dmat4;

using namespace vr;

struct Pose {
	bool valid;
	dmat4 transform;
};

struct UV {
	float u;
	float v;
};

struct DistortionMap {
	vector<UV> red;
	vector<UV> green;
	vector<UV> blue;
};

class OpenVRHelper {

public:

	IVRSystem *system = nullptr;
	string driver = "No Driver";
	string display = "No Display";

	// bitmask of pressed buttons for each device
	uint64_t buttonMap[k_unMaxTrackedDeviceCount];

	// poses for all tracked devices
	TrackedDevicePose_t trackedDevicePose[k_unMaxTrackedDeviceCount];

	vector<dmat4> previousDevicePose = vector<dmat4>(k_unMaxTrackedDeviceCount);
	vector<dmat4> devicePose = vector<dmat4>(k_unMaxTrackedDeviceCount);

	vector<vr::VRControllerState_t> controllerStates = vector<vr::VRControllerState_t>(k_unMaxTrackedDeviceCount);

	dmat4 hmdPose;

	static OpenVRHelper *_instance;

	OpenVRHelper() {

	}

	static OpenVRHelper *instance() {
		return OpenVRHelper::_instance;
	}

	bool start();

	void stop();

	bool isActive();

	void processEvents();

	void updatePose();

	void postPresentHandoff();

	Pose getPose(int deviceID);

	Pose getLeftControllerPose();
	Pose getRightControllerPose();

	vr::VRControllerState_t getLeftControllerState();
	vr::VRControllerState_t getRightControllerState();

	vector<unsigned int> getRecommmendedRenderTargetSize();

	void submit(unsigned int left, unsigned int right);

	void submit(unsigned int texture, EVREye eye);

	void submitDistortionApplied(unsigned int left, unsigned int right);

	void submitDistortionApplied(unsigned int texture, EVREye eye);

	DistortionMap computeDistortionMap(EVREye eye, int width, int height);

	//-----------------------------------------------------------------------------
	// code taken from hellovr_opengl_main.cpp
	//-----------------------------------------------------------------------------
	string getTrackedDeviceString(TrackedDeviceIndex_t device, TrackedDeviceProperty prop, TrackedPropertyError *peError = nullptr);

	//-----------------------------------------------------------------------------
	// from hellovr_opengl_main.cpp CMainApplication::GetHMDMatrixProjectionEye()
	//-----------------------------------------------------------------------------
	dmat4 getProjection(EVREye eye, float nearClip, float farClip);

	float getFOV();

	//-----------------------------------------------------------------------------
	// from hellovr_opengl_main.cpp CMainApplication::GetHMDMatrixPoseEye()
	//-----------------------------------------------------------------------------
	dmat4 getEyePose(Hmd_Eye nEye);

	//-----------------------------------------------------------------------------
	// code taken from hellovr_opengl_main.cpp
	//-----------------------------------------------------------------------------
	dmat4 steamToGLM(const HmdMatrix34_t &mSteam);
};