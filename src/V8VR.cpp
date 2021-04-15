
#include "V8Helper.h"
#include "OpenVRHelper.h"

#include <vector>

#include <glm/gtc/type_ptr.hpp>

using std::vector;

//typedef v8::Persistent<Function, v8::CopyablePersistentTraits<v8::Function>> PersistentFunction;

void V8Helper::setupVR(){

	// Create a template for the global object and set the
	// built-in global functions.
	Local<ObjectTemplate> tpl = ObjectTemplate::New(isolate);
	tpl->SetInternalFieldCount(1);

	tpl->Set(String::NewFromUtf8(isolate, "start"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("start requires 0 arguments");
			return;
		}

		bool started = OpenVRHelper::instance()->start();

		args.GetReturnValue().Set(started);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "stop"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("stop requires 0 arguments");
			return;
		}
	
		OpenVRHelper::instance()->stop();
	}));
	
	tpl->Set(String::NewFromUtf8(isolate, "isActive"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("isActive requires 0 arguments");
			return;
		}
	
		bool isActive = OpenVRHelper::instance()->isActive();
	
		args.GetReturnValue().Set(isActive);
	}));
	
	tpl->Set(String::NewFromUtf8(isolate, "submit"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("submit requires 2 arguments");
			return;
		}
	
		unsigned int left = args[0]->Uint32Value();
		unsigned int right = args[1]->Uint32Value();
	
		OpenVRHelper::instance()->submit(left, right);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "submitDistortionApplied"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("submitDistortionApplied requires 2 arguments");
			return;
		}

		unsigned int left = args[0]->Uint32Value();
		unsigned int right = args[1]->Uint32Value();

		OpenVRHelper::instance()->submitDistortionApplied(left, right);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "computeDistortionMap"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("computeDistortionMap requires 2 arguments");
			return;
		}

		unsigned int width = args[0]->Uint32Value();
		unsigned int height = args[1]->Uint32Value();

		

		auto isolate = Isolate::GetCurrent();
		Local<ObjectTemplate> tmplMap = ObjectTemplate::New(isolate);
		auto object = tmplMap->NewInstance();

		{
			EVREye eye = EVREye::Eye_Left;
			DistortionMap map = OpenVRHelper::instance()->computeDistortionMap(eye, width, height);

			int bytesPerChannel = map.red.size() * sizeof(UV);

			Local<ArrayBuffer> v8LeftRed = v8::ArrayBuffer::New(Isolate::GetCurrent(), bytesPerChannel);
			Local<ArrayBuffer> v8LeftGreen = v8::ArrayBuffer::New(Isolate::GetCurrent(), bytesPerChannel);
			Local<ArrayBuffer> v8LeftBlue = v8::ArrayBuffer::New(Isolate::GetCurrent(), bytesPerChannel);

			auto v8DataLeftRed = v8LeftRed->GetContents().Data();
			auto v8DataLeftGreen = v8LeftGreen->GetContents().Data();
			auto v8DataLeftBlue = v8LeftBlue->GetContents().Data();

			memcpy(v8DataLeftRed, map.red.data(), bytesPerChannel);
			memcpy(v8DataLeftGreen, map.green.data(), bytesPerChannel);
			memcpy(v8DataLeftBlue, map.blue.data(), bytesPerChannel);

			object->Set(String::NewFromUtf8(isolate, "leftRed"), v8LeftRed);
			object->Set(String::NewFromUtf8(isolate, "leftGreen"), v8LeftGreen);
			object->Set(String::NewFromUtf8(isolate, "leftBlue"), v8LeftBlue);
		}

		{
			EVREye eye = EVREye::Eye_Right;
			DistortionMap map = OpenVRHelper::instance()->computeDistortionMap(eye, width, height);

			int bytesPerChannel = map.red.size() * sizeof(UV);

			Local<ArrayBuffer> v8RightRed = v8::ArrayBuffer::New(Isolate::GetCurrent(), bytesPerChannel);
			Local<ArrayBuffer> v8RightGreen = v8::ArrayBuffer::New(Isolate::GetCurrent(), bytesPerChannel);
			Local<ArrayBuffer> v8RightBlue = v8::ArrayBuffer::New(Isolate::GetCurrent(), bytesPerChannel);

			auto v8DataRightRed = v8RightRed->GetContents().Data();
			auto v8DataRightGreen = v8RightGreen->GetContents().Data();
			auto v8DataRightBlue = v8RightBlue->GetContents().Data();

			memcpy(v8DataRightRed, map.red.data(), bytesPerChannel);
			memcpy(v8DataRightGreen, map.green.data(), bytesPerChannel);
			memcpy(v8DataRightBlue, map.blue.data(), bytesPerChannel);

			object->Set(String::NewFromUtf8(isolate, "rightRed"), v8RightRed);
			object->Set(String::NewFromUtf8(isolate, "rightGreen"), v8RightGreen);
			object->Set(String::NewFromUtf8(isolate, "rightBlue"), v8RightBlue);
		}

		args.GetReturnValue().Set(object);
	}));
	
	tpl->Set(String::NewFromUtf8(isolate, "processEvents"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("processEvents requires 0 arguments");
			return;
		}
	
		OpenVRHelper::instance()->processEvents();
	}));
	
	tpl->Set(String::NewFromUtf8(isolate, "updatePose"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("updatePose requires 0 arguments");
			return;
		}
	
		OpenVRHelper::instance()->updatePose();
	}));

	tpl->Set(String::NewFromUtf8(isolate, "postPresentHandoff"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("updatePose requires 0 arguments");
			return;
		}

		OpenVRHelper::instance()->postPresentHandoff();
		}));

	tpl->Set(String::NewFromUtf8(isolate, "getRecommmendedRenderTargetSize"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("getRecommmendedRenderTargetSize requires 0 arguments");
			return;
		}

		auto size = OpenVRHelper::instance()->getRecommmendedRenderTargetSize();
		unsigned int width = size[0];
		unsigned int height = size[1];
		
		auto isolate = Isolate::GetCurrent();
		Local<Object> obj = Object::New(isolate);

		auto keyWidth = String::NewFromUtf8(isolate, "width");
		auto valueWidth = Number::New(isolate, width);
		obj->Set(keyWidth, valueWidth);

		auto keyHeight = String::NewFromUtf8(isolate, "height");
		auto valueHeight = Number::New(isolate, height);
		obj->Set(keyHeight, valueHeight);

		args.GetReturnValue().Set(obj);
	}));

	//tpl->Set(String::NewFromUtf8(isolate, "getHMDPose"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
	//	if (args.Length() != 0) {
	//		V8Helper::_instance->throwException("getHMDPose requires 0 arguments");
	//		return;
	//	}
	//
	//	auto isolate = Isolate::GetCurrent();
	//	auto ovr = OpenVRHelper::instance();
	//
	//	dmat4 hmdPose = ovr->hmdPose;
	//	
	//	int numBytes = 16 * sizeof(double);
	//	Local<ArrayBuffer> v8Buffer = v8::ArrayBuffer::New(Isolate::GetCurrent(), numBytes);
	//	Local<Float64Array> dArray = Float64Array::New(v8Buffer, 0, 16);
	//	
	//	auto v8Data = v8Buffer->GetContents().Data();
	//	
	//	memcpy(v8Data, glm::value_ptr(hmdPose), numBytes);
	//
	//	args.GetReturnValue().Set(dArray);
	//}));

	tpl->Set(String::NewFromUtf8(isolate, "getHMDPose"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("getHMDPose requires 0 arguments");
			return;
		}

		auto isolate = Isolate::GetCurrent();
		auto ovr = OpenVRHelper::instance();

		dmat4 hmdPose = ovr->hmdPose;

		int numBytes = 16 * sizeof(double);
		Local<ArrayBuffer> v8Buffer = v8::ArrayBuffer::New(Isolate::GetCurrent(), numBytes);
		Local<Float64Array> dArray = Float64Array::New(v8Buffer, 0, 16);

		auto v8Data = v8Buffer->GetContents().Data();

		memcpy(v8Data, glm::value_ptr(hmdPose), numBytes);

		args.GetReturnValue().Set(dArray);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getLeftControllerPose"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("getLeftControllerPose requires 0 arguments");
			return;
		}

		auto isolate = Isolate::GetCurrent();
		auto ovr = OpenVRHelper::instance();

		Pose pose = ovr->getLeftControllerPose();

		if (pose.valid) {
			int numBytes = 16 * sizeof(double);
			Local<ArrayBuffer> v8Buffer = v8::ArrayBuffer::New(Isolate::GetCurrent(), numBytes);
			Local<Float64Array> dArray = Float64Array::New(v8Buffer, 0, 16);

			auto v8Data = v8Buffer->GetContents().Data();

			memcpy(v8Data, glm::value_ptr(pose.transform), numBytes);

			args.GetReturnValue().Set(dArray);
		} else {
			args.GetReturnValue().Set(v8::Null(isolate));
		}
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getRightControllerPose"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("getRightControllerPose requires 0 arguments");
			return;
		}

		auto isolate = Isolate::GetCurrent();
		auto ovr = OpenVRHelper::instance();

		Pose pose = ovr->getRightControllerPose();

		if (pose.valid) {
			int numBytes = 16 * sizeof(double);
			Local<ArrayBuffer> v8Buffer = v8::ArrayBuffer::New(Isolate::GetCurrent(), numBytes);
			Local<Float64Array> dArray = Float64Array::New(v8Buffer, 0, 16);

			auto v8Data = v8Buffer->GetContents().Data();

			memcpy(v8Data, glm::value_ptr(pose.transform), numBytes);

			args.GetReturnValue().Set(dArray);
		} else {
			args.GetReturnValue().Set(v8::Null(isolate));
		}
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getLeftEyePose"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("getLeftEyePose requires 0 arguments");
			return;
		}

		auto isolate = Isolate::GetCurrent();
		auto ovr = OpenVRHelper::instance();

		dmat4 pose = ovr->getEyePose(vr::EVREye::Eye_Left);

		int numBytes = 16 * sizeof(double);
		Local<ArrayBuffer> v8Buffer = v8::ArrayBuffer::New(Isolate::GetCurrent(), numBytes);
		Local<Float64Array> dArray = Float64Array::New(v8Buffer, 0, 16);

		auto v8Data = v8Buffer->GetContents().Data();

		memcpy(v8Data, glm::value_ptr(pose), numBytes);

		args.GetReturnValue().Set(dArray);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getRightEyePose"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("getRightEyePose requires 0 arguments");
			return;
		}

		auto isolate = Isolate::GetCurrent();
		auto ovr = OpenVRHelper::instance();

		dmat4 pose = ovr->getEyePose(vr::EVREye::Eye_Right);

		int numBytes = 16 * sizeof(double);
		Local<ArrayBuffer> v8Buffer = v8::ArrayBuffer::New(Isolate::GetCurrent(), numBytes);
		Local<Float64Array> dArray = Float64Array::New(v8Buffer, 0, 16);

		auto v8Data = v8Buffer->GetContents().Data();

		memcpy(v8Data, glm::value_ptr(pose), numBytes);

		args.GetReturnValue().Set(dArray);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getLeftProjection"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("getLeftProjection requires 0 arguments");
			return;
		}

		auto isolate = Isolate::GetCurrent();
		auto ovr = OpenVRHelper::instance();

		float near = args[0]->NumberValue();
		float far = args[1]->NumberValue();

		dmat4 pose = ovr->getProjection(vr::EVREye::Eye_Left, near, far);

		int numBytes = 16 * sizeof(double);
		Local<ArrayBuffer> v8Buffer = v8::ArrayBuffer::New(Isolate::GetCurrent(), numBytes);
		Local<Float64Array> dArray = Float64Array::New(v8Buffer, 0, 16);

		auto v8Data = v8Buffer->GetContents().Data();

		memcpy(v8Data, glm::value_ptr(pose), numBytes);

		args.GetReturnValue().Set(dArray);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getRightProjection"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("getRightProjection requires 0 arguments");
			return;
		}

		auto isolate = Isolate::GetCurrent();
		auto ovr = OpenVRHelper::instance();

		float near = args[0]->NumberValue();
		float far = args[1]->NumberValue();

		dmat4 pose = ovr->getProjection(vr::EVREye::Eye_Right, near, far);

		int numBytes = 16 * sizeof(double);
		Local<ArrayBuffer> v8Buffer = v8::ArrayBuffer::New(Isolate::GetCurrent(), numBytes);
		Local<Float64Array> dArray = Float64Array::New(v8Buffer, 0, 16);

		auto v8Data = v8Buffer->GetContents().Data();

		memcpy(v8Data, glm::value_ptr(pose), numBytes);

		args.GetReturnValue().Set(dArray);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getControllerStateLeft"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("getControllerStateLeft requires 0 arguments");
			return;
		}

		auto isolate = Isolate::GetCurrent();
		auto ovr = OpenVRHelper::instance();
		
		int leftHandID = ovr->system->GetTrackedDeviceIndexForControllerRole(vr::ETrackedControllerRole::TrackedControllerRole_LeftHand);

		//vr::VRControllerState_t state;
		//ovr->system->GetControllerState(leftHandID, &state, sizeof(state));
		vr::VRControllerState_t state = ovr->controllerStates[leftHandID];
		
		auto pressed = Array::New(isolate, 64);
		auto touched = Array::New(isolate, 64);

		for (int i = 0; i < 64; i++) {

			uint64_t mask = 1ull << i;

			if (i == 32) {
				int a = 10;
			}

			bool isPressed = (state.ulButtonPressed & mask) > 0ull;
			bool isTouched = (state.ulButtonTouched & mask) > 0ull;

			auto v8IsPressed = v8::Boolean::New(isolate, isPressed);
			auto v8IsTouched = v8::Boolean::New(isolate, isTouched);

			pressed->Set(i, v8IsPressed);
			touched->Set(i, v8IsTouched);

		}

		auto nx = v8::Number::New(isolate, state.rAxis->x);
		auto ny = v8::Number::New(isolate, state.rAxis->y);

		auto axis = Array::New(isolate, 2);
		axis->Set(uint32_t(0), nx);
		axis->Set(uint32_t(1), ny);
		
		Local<Object> obj = Object::New(isolate);

		auto keyPressed = String::NewFromUtf8(isolate, "pressed");
		obj->Set(keyPressed, pressed);

		auto keyTouched = String::NewFromUtf8(isolate, "touched");
		obj->Set(keyTouched, touched);

		auto keyAxis = String::NewFromUtf8(isolate, "axis");
		obj->Set(keyAxis, axis);




		args.GetReturnValue().Set(obj);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getControllerStateRight"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("getControllerStateRight requires 0 arguments");
			return;
		}

		auto isolate = Isolate::GetCurrent();
		auto ovr = OpenVRHelper::instance();

		int rightHandID = ovr->system->GetTrackedDeviceIndexForControllerRole(vr::ETrackedControllerRole::TrackedControllerRole_RightHand);

		//vr::VRControllerState_t state;
		//ovr->system->GetControllerState(rightHandID, &state, sizeof(state));
		vr::VRControllerState_t state = ovr->controllerStates[rightHandID];

		auto pressed = Array::New(isolate, 64);
		auto touched = Array::New(isolate, 64);

		for (int i = 0; i < 64; i++) {

			uint64_t mask = 1ull << i;

			if (i == 33) {
				int debug = 10;
			}

			bool isPressed = (state.ulButtonPressed & mask) > 0ull;
			bool isTouched = (state.ulButtonTouched & mask) > 0ull;

			auto v8IsPressed = v8::Boolean::New(isolate, isPressed);
			auto v8IsTouched = v8::Boolean::New(isolate, isTouched);

			pressed->Set(i, v8IsPressed);
			touched->Set(i, v8IsTouched);

		}

		auto nx = v8::Number::New(isolate, state.rAxis->x);
		auto ny = v8::Number::New(isolate, state.rAxis->y);

		auto axis = Array::New(isolate, 2);
		axis->Set(uint32_t(0), nx);
		axis->Set(uint32_t(1), ny);

		Local<Object> obj = Object::New(isolate);

		auto keyPressed = String::NewFromUtf8(isolate, "pressed");
		obj->Set(keyPressed, pressed);

		auto keyTouched = String::NewFromUtf8(isolate, "touched");
		obj->Set(keyTouched, touched);

		auto keyAxis = String::NewFromUtf8(isolate, "axis");
		obj->Set(keyAxis, axis);


		args.GetReturnValue().Set(obj);
	}));

	Local<Object> obj = tpl->NewInstance();

	context->Global()->Set(
		String::NewFromUtf8(isolate, "vr"),
		obj);
}