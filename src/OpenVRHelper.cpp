
#include "OpenVRHelper.h"

#include <iostream>
#include <vector>
#include <algorithm>


using std::vector;
using std::cout;
using std::endl;

OpenVRHelper *OpenVRHelper::_instance = new OpenVRHelper();


bool OpenVRHelper::start() {

	if(system){
		return false;
	}

	EVRInitError error;
	system = VR_Init(&error, EVRApplicationType::VRApplication_Scene);

	if (error != VRInitError_None) {
		auto errorMsg = vr::VR_GetVRInitErrorAsEnglishDescription(error);
		cout << "ERROR: could not start VR. (" << error << "): " << errorMsg << endl;

		return false;
	}

	driver = getTrackedDeviceString(k_unTrackedDeviceIndex_Hmd, Prop_TrackingSystemName_String);
	display = getTrackedDeviceString(k_unTrackedDeviceIndex_Hmd, Prop_SerialNumber_String);

	if (!vr::VRCompositor()) {
		cout << "ERROR: failed to initialize compositor." << endl;
		return false;
	}

	return true;
}

void OpenVRHelper::stop() {
	if(system){
		VR_Shutdown();
		system = nullptr;
	}
}

bool OpenVRHelper::isActive() {
	return system != nullptr;
}

void OpenVRHelper::submit(unsigned int left, unsigned int right) {
	submit(left, vr::EVREye::Eye_Left);
	submit(right, vr::EVREye::Eye_Right);
}

void OpenVRHelper::submit(unsigned int texture, EVREye eye) {
	Texture_t tex = { (void*)texture, vr::API_OpenGL, vr::ColorSpace_Gamma };

	vr::VRTextureBounds_t *bounds = (vr::VRTextureBounds_t*)0;
	auto flags = vr::EVRSubmitFlags::Submit_Default;
	VRCompositor()->Submit(eye, &tex, bounds, flags);
}

void OpenVRHelper::submitDistortionApplied(unsigned int left, unsigned int right) {
	submitDistortionApplied(left, vr::EVREye::Eye_Left);
	submitDistortionApplied(right, vr::EVREye::Eye_Right);
}

void OpenVRHelper::submitDistortionApplied(unsigned int texture, EVREye eye) {
	Texture_t tex = { (void*)texture, vr::API_OpenGL, vr::ColorSpace_Gamma };

	vr::VRTextureBounds_t *bounds = (vr::VRTextureBounds_t*)0;
	auto flags = vr::Submit_LensDistortionAlreadyApplied;
	VRCompositor()->Submit(eye, &tex, bounds, flags);
}

DistortionMap OpenVRHelper::computeDistortionMap(EVREye eye, int width, int height) {

	//EVREye eye = EVREye::Eye_Left;

	//int width = 100;
	//int height = 100;

	int numPixels = width * height;

	DistortionMap map;
	map.red = vector<UV>(numPixels);
	map.green = vector<UV>(numPixels);
	map.blue = vector<UV>(numPixels);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {

			float u = float(x) / float(width - 1);
			float v = float(y) / float(height - 1);

			vr::DistortionCoordinates_t outCoordinatates;

			system->ComputeDistortion(eye, u, v, &outCoordinatates);

			int targetIndex = width * y + x;

			map.red[targetIndex] = { outCoordinatates.rfRed[0], outCoordinatates.rfRed[1] };
			map.green[targetIndex] = { outCoordinatates.rfGreen[0], outCoordinatates.rfGreen[1] };
			map.blue[targetIndex] = { outCoordinatates.rfBlue[0], outCoordinatates.rfBlue[1] };
		}
	}

	return map;
}


void OpenVRHelper::processEvents() {
	VREvent_t event;
	while (system->PollNextEvent(&event, sizeof(event))) {
		//ProcessVREvent(event);
	}

	vector<TrackedDeviceIndex_t> triggered;

	for (vr::TrackedDeviceIndex_t unDevice = 0; unDevice < vr::k_unMaxTrackedDeviceCount; unDevice++) {
		vr::VRControllerState_t &state = controllerStates[unDevice];
		if (system->GetControllerState(unDevice, &state, sizeof(state))) {

			auto previousState = buttonMap[unDevice];
			auto currentState = state.ulButtonPressed;

			uint64_t justPressed = (previousState ^ currentState) & currentState;
			uint64_t justReleased = (previousState ^ currentState) & previousState;

			buttonMap[unDevice] = state.ulButtonPressed;

		}
	}
}


void OpenVRHelper::updatePose() {
	if (!system) {
		return;
	}

	VRCompositor()->WaitGetPoses(trackedDevicePose, k_unMaxTrackedDeviceCount, NULL, 0);

	for (int nDevice = 0; nDevice < vr::k_unMaxTrackedDeviceCount; ++nDevice) {
	
		if (trackedDevicePose[nDevice].bPoseIsValid) {
			previousDevicePose[nDevice] = devicePose[nDevice];
			devicePose[nDevice] = steamToGLM(trackedDevicePose[nDevice].mDeviceToAbsoluteTracking);
		}
	}
	
	if (trackedDevicePose[k_unTrackedDeviceIndex_Hmd].bPoseIsValid) {
		hmdPose = devicePose[k_unTrackedDeviceIndex_Hmd];
	}

}

void OpenVRHelper::postPresentHandoff() {
	VRCompositor()->PostPresentHandoff();
};

Pose OpenVRHelper::getPose(int deviceID) {

	if (deviceID < 0) {
		return{ false, glm::dmat4() };
	}

	if (trackedDevicePose[deviceID].bPoseIsValid) {
		return { true, devicePose[deviceID] };
	} else {
		return{ false, glm::dmat4() };
	}
}

Pose OpenVRHelper::getLeftControllerPose() {
	int leftHandID = system->GetTrackedDeviceIndexForControllerRole(vr::ETrackedControllerRole::TrackedControllerRole_LeftHand);

	return getPose(leftHandID);
}

Pose OpenVRHelper::getRightControllerPose() {
	int rightHandID = system->GetTrackedDeviceIndexForControllerRole(vr::ETrackedControllerRole::TrackedControllerRole_RightHand);

	return getPose(rightHandID);
}

vr::VRControllerState_t OpenVRHelper::getLeftControllerState(){
	int leftHandID = system->GetTrackedDeviceIndexForControllerRole(vr::ETrackedControllerRole::TrackedControllerRole_LeftHand);

	if(leftHandID < 0){
		return vr::VRControllerState_t();
	}

	vr::VRControllerState_t state = controllerStates[leftHandID];

	return state;
}

vr::VRControllerState_t OpenVRHelper::getRightControllerState(){
	int rightHandID = system->GetTrackedDeviceIndexForControllerRole(vr::ETrackedControllerRole::TrackedControllerRole_RightHand);

	if(rightHandID < 0){
		return vr::VRControllerState_t();
	}

	vr::VRControllerState_t state = controllerStates[rightHandID];

	return state;
}

vector<unsigned int> OpenVRHelper::getRecommmendedRenderTargetSize(){

	uint32_t width;
	uint32_t height;
	system->GetRecommendedRenderTargetSize(&width, &height);

	return { width, height };
}


//-----------------------------------------------------------------------------
// code taken from hellovr_opengl_main.cpp
//-----------------------------------------------------------------------------
string OpenVRHelper::getTrackedDeviceString(TrackedDeviceIndex_t device, TrackedDeviceProperty prop, TrackedPropertyError *peError) {
	uint32_t unRequiredBufferLen = system->GetStringTrackedDeviceProperty(device, prop, nullptr, 0, peError);
	if (unRequiredBufferLen == 0)
		return "";

	char *pchBuffer = new char[unRequiredBufferLen];
	unRequiredBufferLen = system->GetStringTrackedDeviceProperty(device, prop, pchBuffer, unRequiredBufferLen, peError);
	std::string sResult = pchBuffer;
	delete[] pchBuffer;

	return sResult;
}


//-----------------------------------------------------------------------------
// from hellovr_opengl_main.cpp CMainApplication::GetHMDMatrixProjectionEye()
// and https://github.com/ValveSoftware/openvr/wiki/IVRSystem::GetProjectionRaw
//-----------------------------------------------------------------------------
dmat4 OpenVRHelper::getProjection(EVREye eye, float nearClip, float farClip) {
	if (!system) {
		return dmat4();
	}

	float left = 0;
	float right = 0;
	float top = 0;
	float bottom = 0;
	system->GetProjectionRaw(eye, &left, &right, &top, &bottom);

	float idx = 1.0f / (right - left);
	float idy = 1.0f / (bottom - top);
	float idz = 1.0f / (farClip - nearClip);
	float sx = right + left;
	float sy = bottom + top;
	float zFar = farClip;
	float zNear = nearClip;

	// reverse z with infinite far
	//auto customProj = glm::dmat4(
	//	2.0 * idx, 0.0, 0.0, 0.0,
	//	0.0, 2.0 * idy, 0.0, 0.0,
	//	sx * idx, sy * idy, zFar*idz - 1.0, -1.0,
	//	0.0, 0.0, zFar*zNear*idz, 0.0
	//	);

	auto customProj = glm::dmat4(
		2.0 * idx, 0.0, 0.0, 0.0,
		0.0, 2.0 * idy, 0.0, 0.0,
		sx * idx, sy * idy, 0.0, -1.0,
		0.0, 0.0, zNear, 0.0
	);



	//vr::HmdMatrix44_t mat = system->GetProjectionMatrix(eye, nearClip, farClip, vr::API_OpenGL);
	//
	//auto openvrProj = glm::dmat4(
	//	mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0],
	//	mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1],
	//	mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2],
	//	mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]
	//	);

	return customProj;
}


float OpenVRHelper::getFOV() {
	float l_left = 0.0f, l_right = 0.0f, l_top = 0.0f, l_bottom = 0.0f;
	system->GetProjectionRaw(vr::EVREye::Eye_Left, &l_left, &l_right, &l_top, &l_bottom);

	// top and bottom seem reversed. Asume larger value to be the top.
	// see https://github.com/ValveSoftware/openvr/issues/110
	float realTop = std::max(l_top, l_bottom);

	return 2.0f * atan2(realTop, 1.0);
}

//-----------------------------------------------------------------------------
// from hellovr_opengl_main.cpp CMainApplication::GetHMDMatrixPoseEye()
//-----------------------------------------------------------------------------
dmat4 OpenVRHelper::getEyePose(Hmd_Eye nEye) {
	if (!system) {
		return glm::dmat4();
	}

	HmdMatrix34_t mat = system->GetEyeToHeadTransform(nEye);

	dmat4 matrix(
		mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.0,
		mat.m[0][1], mat.m[1][1], mat.m[2][1], 0.0,
		mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.0,
		mat.m[0][3], mat.m[1][3], mat.m[2][3], 1.0f
	);

	return matrix;
}


dmat4 OpenVRHelper::steamToGLM(const HmdMatrix34_t &mSteam) {

	dmat4 matrix{
		mSteam.m[0][0], mSteam.m[1][0], mSteam.m[2][0], 0.0,
		mSteam.m[0][1], mSteam.m[1][1], mSteam.m[2][1], 0.0,
		mSteam.m[0][2], mSteam.m[1][2], mSteam.m[2][2], 0.0,
		mSteam.m[0][3], mSteam.m[1][3], mSteam.m[2][3], 1.0f
	};

	return matrix;
}


