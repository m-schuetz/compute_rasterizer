#pragma once

#include <string>
#include <iostream>
#include <unordered_map>
#include <map>

#include "libplatform/libplatform.h"
#include "v8.h"
#include "v8-inspector.h"

#include "GL\glew.h"
#include "GLFW\glfw3.h"

using std::string;
using std::unordered_map;
using std::map;
using std::cout;
using std::endl;

using v8::Isolate;
using v8::ArrayBuffer;
using v8::Local;
using v8::HandleScope;
using v8::ObjectTemplate;
using v8::FunctionTemplate;
using v8::V8;
using v8::Platform;
using v8::Handle;
using v8::Context;
using v8::String;
using v8::Value;
using v8::FunctionCallbackInfo;
using v8::PropertyCallbackInfo;
using v8::NewStringType;
using v8::Script;
using v8::Persistent;
using v8::Object;
using v8::External;
using v8::EscapableHandleScope;
using v8::Array;
using v8::Uint8Array;
using v8::Float32Array;
using v8::Float64Array;
using v8::TypedArray;
using v8::NamedPropertyGetterCallback;
using v8::Promise;
using v8::CopyablePersistentTraits;
using v8::Function;
using v8::Number;
using v8::TryCatch;

class ArrayBufferAllocator : public v8::ArrayBuffer::Allocator {
public:
	virtual void* Allocate(size_t length) {
		void* data = AllocateUninitialized(length);
		return data == NULL ? data : memset(data, 0, length);
	}
	virtual void* AllocateUninitialized(size_t length) { return malloc(length); }
	virtual void Free(void* data, size_t) { free(data); }
};

struct Scope {
	Isolate::Scope iscope;
	HandleScope hscope;

	Scope(Isolate *isolate) : iscope(isolate), hscope(isolate) {

	}
};



class InspectorClientImpl : public v8_inspector::V8InspectorClient {

public:
	
	InspectorClientImpl() {
		
	}

	void runMessageLoopOnPause(int contextGroupId) {
		cout << "runMessageLoopOnPause" << endl;
	}

	void quitMessageLoopOnPause() {
		cout << "quitMessageLoopOnPause" << endl;
	}

};

class ChannelImpl : public v8_inspector::V8Inspector::Channel {

public:
	ChannelImpl() {

	}

	void sendResponse(int callId, std::unique_ptr<v8_inspector::StringBuffer> message) {
		cout << "sendResponse" << endl;
	}

	void sendNotification(std::unique_ptr<v8_inspector::StringBuffer> message) {

		cout << "sendNotification" << endl;
	}

	void flushProtocolNotifications() {
		cout << "flushProtocolNotifications" << endl;
		
	}

};

class V8Helper {
public:

	Isolate * isolate = nullptr;
	Scope *scope = nullptr;

	Handle<Context> context;
	Context::Scope *context_scope = nullptr;
	GLFWwindow* window = nullptr;
	double timeSinceLastFrame = 0.0;

	map<string, string> debugValue;

	Local<ObjectTemplate> tplSceneNode;

	static V8Helper *_instance;


	V8Helper() {
		// Initialize V8.
		V8::InitializeICU();
		V8::InitializeExternalStartupData("");

		Platform* platform = v8::platform::CreateDefaultPlatform();
		V8::InitializePlatform(platform);
		V8::Initialize();

		// Create a new Isolate and make it the current one.
		//ArrayBufferAllocator allocator;
		Isolate::CreateParams create_params;
		create_params.array_buffer_allocator = v8::ArrayBuffer::Allocator::NewDefaultAllocator();
		isolate = Isolate::New(create_params);

		scope = new Scope(isolate);

		//Local<ObjectTemplate> globalTemplate = ObjectTemplate::New();
		//context = Context::New(isolate, NULL, globalTemplate);
		context = v8::Context::New(isolate);

		
		
		//auto inspectorClient = new InspectorClientImpl();
		//auto inspector = v8_inspector::V8Inspector::create(isolate, inspectorClient);
		//auto channel = new ChannelImpl();
		//
		//v8_inspector::StringView view;
		//auto session = inspector->connect(1, channel, view);
		//
		//auto message_view = v8_inspector::StringView();
		//session->dispatchProtocolMessage(message_view);
		//
		//inspector->contextCreated(
		//	v8_inspector::V8ContextInfo(context, 1, v8_inspector::StringView()));



		context_scope = new Context::Scope(context);

		V8Helper::_instance = this;

		//runScript("1 + 2");
	}

	static V8Helper *instance() {
		return V8Helper::_instance;
	}

	void registerFunction(string name, v8::FunctionCallback callback) {
		context->Global()->Set(
			String::NewFromUtf8(isolate, name.c_str()),
			FunctionTemplate::New(isolate, callback)->GetFunction()
		);
	}

	void throwException(string message) {
		isolate->ThrowException(
			String::NewFromUtf8(isolate, message.c_str(),
				v8::NewStringType::kNormal).ToLocalChecked());
	}

	void runScript(string command);
	void runScriptSilent(string command);
	Local<v8::Script> compileScript(string command);

	void setupV8();
	void setupGL();
	void setupWindow();
	void setupVR();
	void setupV8GLExtBindings(Local<ObjectTemplate>& tpl);
	//void setupV8GLExtBindings(Local<ObjectTemplate>& tpl, unordered_map<string, int> &constants);

};
