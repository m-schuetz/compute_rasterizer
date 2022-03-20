//// adapted and copied from hellovr_opengl_main.cpp
//// LICENSING this file under the same license, see libs/openvr/LICENSE
//// 
//
//#pragma once
//
//#include <iostream>
//
//#include <openvr.h>
//
//using namespace std;
//
//
//struct VrRuntime{
//
//	vr::IVRSystem *vrSystem = nullptr;
//	string strDriver;
//	string strDisplay;
//
//	vr::TrackedDevicePose_t trackedDevicePose[vr::k_unMaxTrackedDeviceCount];
//	int m_iValidPoseCount;
//	std::string m_strPoseClasses;                            // what classes we saw poses for this frame
//	char m_rDevClassChar[ vr::k_unMaxTrackedDeviceCount ];   // for each device, a character representing its class
//	glm::dmat4 m_rmat4DevicePose[ vr::k_unMaxTrackedDeviceCount ];
//	vector<vr::VRControllerState_t> controllerStates = vector<vr::VRControllerState_t>(vr::k_unMaxTrackedDeviceCount);
//	uint64_t buttonMap[vr::k_unMaxTrackedDeviceCount];
//
//	glm::dmat4 hmdPose;
//	glm::dmat4 projLeft;
//	glm::dmat4 projRight;
//	glm::dmat4 poseLeft;
//	glm::dmat4 poseRight;
//
//	float near = 0.1;
//	float far = 100'000.0;
//
//	VrRuntime(){
//
//	}
//
//	void init(){
//
//		vr::EVRInitError error = vr::VRInitError_None;
//		vrSystem = vr::VR_Init(&error, vr::VRApplication_Scene);
//
//		if (error != vr::VRInitError_None){
//			vrSystem = nullptr;
//			
//			std::cout << "failed to create VR runtime" << std::endl;
//
//			return;
//		}
//
//		if (!vr::VRCompositor()){
//
//			std::cout << "failed to initialize compositor" << std::endl;
//			
//			return;
//		}
//
//		strDriver = GetTrackedDeviceString(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_TrackingSystemName_String);
//		strDisplay = GetTrackedDeviceString(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SerialNumber_String);
//
//		std::cout << "VR driver: " << strDriver << std::endl;
//		std::cout << "VR display: " << strDisplay << std::endl;
//
//		projLeft = getProjEye(vr::Eye_Left);
//		projRight = getProjEye(vr::Eye_Right);
//		poseLeft = getEyePose(vr::Eye_Left);
//		poseRight = getEyePose(vr::Eye_Right);
//
//		
//	}
//
//	void submit(GLuint left, GLuint right){
//
//		std::cout << "submit " << left << ", " << right << std::endl;
//
//		// vr::Texture_t leftEyeTexture = {(void*)(uintptr_t)left, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
//		// vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture );
//
//		// vr::Texture_t rightEyeTexture = {(void*)(uintptr_t)right, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
//		// vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture );
//
//		{
//			vr::Texture_t tex = {(void*)(uintptr_t)left, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
//			vr::VRTextureBounds_t *bounds = (vr::VRTextureBounds_t*)0;
//			auto flags = vr::EVRSubmitFlags::Submit_Default;
//			vr::VRCompositor()->Submit(vr::Eye_Left, &tex, bounds, flags);
//		}
//
//		{
//			vr::Texture_t tex = {(void*)(uintptr_t)right, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
//			vr::VRTextureBounds_t *bounds = (vr::VRTextureBounds_t*)0;
//			auto flags = vr::EVRSubmitFlags::Submit_Default;
//			vr::VRCompositor()->Submit(vr::Eye_Right, &tex, bounds, flags);
//		}
//
//	}
//
//	void updatePose(){
//		if(!vrSystem){
//			return;
//		}
//
//		vr::VRCompositor()->WaitGetPoses(trackedDevicePose, vr::k_unMaxTrackedDeviceCount, NULL, 0 );
//
//		m_iValidPoseCount = 0;
//		m_strPoseClasses = "";
//
//		for ( int nDevice = 0; nDevice < vr::k_unMaxTrackedDeviceCount; ++nDevice ){
//			
//			if ( trackedDevicePose[nDevice].bPoseIsValid ){
//				m_iValidPoseCount++;
//				m_rmat4DevicePose[nDevice] = toDmat4(trackedDevicePose[nDevice].mDeviceToAbsoluteTracking);
//				
//				if (m_rDevClassChar[nDevice]==0){
//
//					switch (vrSystem->GetTrackedDeviceClass(nDevice))
//					{
//						case vr::TrackedDeviceClass_Controller:        m_rDevClassChar[nDevice] = 'C'; break;
//						case vr::TrackedDeviceClass_HMD:               m_rDevClassChar[nDevice] = 'H'; break;
//						case vr::TrackedDeviceClass_Invalid:           m_rDevClassChar[nDevice] = 'I'; break;
//						case vr::TrackedDeviceClass_GenericTracker:    m_rDevClassChar[nDevice] = 'G'; break;
//						case vr::TrackedDeviceClass_TrackingReference: m_rDevClassChar[nDevice] = 'T'; break;
//						default:                                       m_rDevClassChar[nDevice] = '?'; break;
//					}
//
//				}
//
//				m_strPoseClasses += m_rDevClassChar[nDevice];
//			}
//
//		}
//
//		if ( trackedDevicePose[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid ){
//			hmdPose = m_rmat4DevicePose[vr::k_unTrackedDeviceIndex_Hmd];
//			hmdPose = glm::inverse(hmdPose);
//		}
//	}
//
//	void handleInput(){
//
//		// Process SteamVR events
//		vr::VREvent_t event;
//		while( vrSystem->PollNextEvent( &event, sizeof( event ) ) ){
//			processVrEvent( event );
//		}
//
//		vector<vr::TrackedDeviceIndex_t> triggered;
//
//		for (vr::TrackedDeviceIndex_t unDevice = 0; unDevice < vr::k_unMaxTrackedDeviceCount; unDevice++) {
//			vr::VRControllerState_t &state = controllerStates[unDevice];
//			if (vrSystem->GetControllerState(unDevice, &state, sizeof(state))) {
//
//				auto previousState = buttonMap[unDevice];
//				auto currentState = state.ulButtonPressed;
//
//				uint64_t justPressed = (previousState ^ currentState) & currentState;
//				uint64_t justReleased = (previousState ^ currentState) & previousState;
//
//				buttonMap[unDevice] = state.ulButtonPressed;
//
//			}
//		}
//
//	}
//
//	void processVrEvent(const vr::VREvent_t & event){
//		
//		if(event.eventType == vr::VREvent_TrackedDeviceDeactivated){
//			std::cout << "Device detached: " << event.trackedDeviceIndex << std::endl;
//		}else if(event.eventType == vr::VREvent_TrackedDeviceUpdated){
//			std::cout << "Device updated: " << event.trackedDeviceIndex << std::endl;
//		}
//		
//	}
//
//	glm::dmat4 getProjEye(vr::Hmd_Eye eye){
//		if(!vrSystem){
//			return glm::dmat4();
//		}else{
//			auto mat = vrSystem->GetProjectionMatrix(eye, near, far);
//
//			return glm::dmat4(
//				mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0],
//				mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1], 
//				mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2], 
//				mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]
//			);
//		}
//	}
//
//	glm::dmat4 getEyePose(vr::Hmd_Eye eye){
//		if(!vrSystem){
//			return glm::dmat4();
//		}else{
//			auto pose = vrSystem->GetEyeToHeadTransform(eye);
//
//			glm::dmat4 matrixObj(
//				pose.m[0][0], pose.m[1][0], pose.m[2][0], 0.0, 
//				pose.m[0][1], pose.m[1][1], pose.m[2][1], 0.0,
//				pose.m[0][2], pose.m[1][2], pose.m[2][2], 0.0,
//				pose.m[0][3], pose.m[1][3], pose.m[2][3], 1.0f
//			);
//
//			return glm::inverse(matrixObj);
//		}
//	}
//
//	string GetTrackedDeviceString(vr::TrackedDeviceIndex_t unDevice, vr::TrackedDeviceProperty prop, vr::TrackedPropertyError *peError = nullptr ){
//		uint32_t unRequiredBufferLen = vr::VRSystem()->GetStringTrackedDeviceProperty( unDevice, prop, nullptr, 0, peError );
//		if( unRequiredBufferLen == 0 ){
//			return "";
//		}
//
//		char *pchBuffer = new char[ unRequiredBufferLen ];
//		unRequiredBufferLen = vr::VRSystem()->GetStringTrackedDeviceProperty( unDevice, prop, pchBuffer, unRequiredBufferLen, peError );
//		std::string sResult = pchBuffer;
//		delete [] pchBuffer;
//
//		return sResult;
//	}
//
//	vector<uint32_t> getRecommmendedRenderTargetSize(){
//
//		uint32_t width;
//		uint32_t height;
//		vrSystem->GetRecommendedRenderTargetSize(&width, &height);
//
//		return { width, height };
//	}
//
//	glm::dmat4 toDmat4(const vr::HmdMatrix34_t &matPose ){
//
//		glm::dmat4 matrixObj(
//			matPose.m[0][0], matPose.m[1][0], matPose.m[2][0], 0.0,
//			matPose.m[0][1], matPose.m[1][1], matPose.m[2][1], 0.0,
//			matPose.m[0][2], matPose.m[1][2], matPose.m[2][2], 0.0,
//			matPose.m[0][3], matPose.m[1][3], matPose.m[2][3], 1.0f
//		);
//
//		return matrixObj;
//	}
//
//	void postPresentHandoff() {
//		vr::VRCompositor()->PostPresentHandoff();
//	};
//
//};