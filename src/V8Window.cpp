
#include "V8Helper.h"

#include <vector>

#include "Application.h"

#include <Windows.h>
#include <thread>
#include <chrono>

using std::vector;

typedef v8::Persistent<Function, v8::CopyablePersistentTraits<v8::Function>> PersistentFunction;

void V8Helper::setupWindow(){
	// Create a template for the global object and set the
	// built-in global functions.
	Local<ObjectTemplate> tpl = ObjectTemplate::New(isolate);
	tpl->SetInternalFieldCount(1);

	tpl->SetAccessor(String::NewFromUtf8(isolate, "width"), [](Local<String> property, const PropertyCallbackInfo<Value>& info) {

		auto window = V8Helper::instance()->window;

		int width, height;
		glfwGetWindowSize(window, &width, &height);

		auto value = GL_FALSE;
		info.GetReturnValue().Set(width);
	}, [](Local<String> property, Local<Value> value, const PropertyCallbackInfo<void>& info) {

		int width = value->NumberValue();

		auto window = V8Helper::instance()->window;

		int tmp, height;
		glfwGetWindowSize(window, &tmp, &height);
		glfwSetWindowSize(window, width, height);

	});

	tpl->SetAccessor(String::NewFromUtf8(isolate, "height"), [](Local<String> property, const PropertyCallbackInfo<Value>& info) {

		auto window = V8Helper::instance()->window;

		int width, height;
		glfwGetWindowSize(window, &width, &height);

		auto value = GL_FALSE;
		info.GetReturnValue().Set(height);
	}, [](Local<String> property, Local<Value> value, const PropertyCallbackInfo<void>& info) {

		int height = value->NumberValue();

		auto window = V8Helper::instance()->window;

		int width, tmp;
		glfwGetWindowSize(window, &width, &tmp);
		glfwSetWindowSize(window, width, height);
	});

	tpl->SetAccessor(String::NewFromUtf8(isolate, "x"), [](Local<String> property, const PropertyCallbackInfo<Value>& info) {

		auto window = V8Helper::instance()->window;

		int x, y;
		glfwGetWindowPos(window, &x, &y);

		auto value = GL_FALSE;
		info.GetReturnValue().Set(x);
	}, [](Local<String> property, Local<Value> value, const PropertyCallbackInfo<void>& info) {

		int x = value->NumberValue();

		auto window = V8Helper::instance()->window;

		int tmp, y;
		glfwGetWindowPos(window, &tmp, &y);
		glfwSetWindowPos(window, x, y);

	});

	tpl->SetAccessor(String::NewFromUtf8(isolate, "y"), [](Local<String> property, const PropertyCallbackInfo<Value>& info) {

		auto window = V8Helper::instance()->window;

		int x, y;
		glfwGetWindowPos(window, &x, &y);

		auto value = GL_FALSE;
		info.GetReturnValue().Set(y);
	}, [](Local<String> property, Local<Value> value, const PropertyCallbackInfo<void>& info) {

		int y = value->NumberValue();

		auto window = V8Helper::instance()->window;

		int x, tmp;
		glfwGetWindowPos(window, &x, &tmp);
		glfwSetWindowPos(window, x, y);
	});


	tpl->SetAccessor(String::NewFromUtf8(isolate, "monitorWidth"), [](Local<String> property, const PropertyCallbackInfo<Value>& info) {

		int numMonitors;
		GLFWmonitor** monitors = glfwGetMonitors(&numMonitors);
		const GLFWvidmode * mode = glfwGetVideoMode(monitors[0]);

		info.GetReturnValue().Set(mode->width);
	});

	tpl->SetAccessor(String::NewFromUtf8(isolate, "monitorHeight"), [](Local<String> property, const PropertyCallbackInfo<Value>& info) {

		int numMonitors;
		GLFWmonitor** monitors = glfwGetMonitors(&numMonitors);
		const GLFWvidmode * mode = glfwGetVideoMode(monitors[0]);

		info.GetReturnValue().Set(mode->height);
	});

	tpl->SetAccessor(String::NewFromUtf8(isolate, "monitors"), [](Local<String> property, const PropertyCallbackInfo<Value>& info) {

		int numMonitors;
		GLFWmonitor** monitors = glfwGetMonitors(&numMonitors);

		auto isolate = Isolate::GetCurrent();
		Local<ObjectTemplate> monitorsTempl = ObjectTemplate::New(isolate);
		auto objMonitors = monitorsTempl->NewInstance();

		auto lMonitors = Array::New(isolate, numMonitors);

		for (int i = 0; i < numMonitors; i++) {

			const GLFWvidmode * mode = glfwGetVideoMode(monitors[i]);

			Local<ObjectTemplate> monitorTempl = ObjectTemplate::New(isolate);
			auto objMonitor = monitorTempl->NewInstance();

			auto lWidth = v8::Integer::New(isolate, mode->width);
			auto lHeight = v8::Integer::New(isolate, mode->height);

			objMonitor->Set(String::NewFromUtf8(isolate, "width"), lWidth);
			objMonitor->Set(String::NewFromUtf8(isolate, "height"), lHeight);

			lMonitors->Set(i, objMonitor);
		}

		info.GetReturnValue().Set(lMonitors);
	});


	tpl->SetAccessor(String::NewFromUtf8(isolate, "timeSinceLastFrame"), [](Local<String> property, const PropertyCallbackInfo<Value>& info) {

		auto timeSinceLastFrame = V8Helper::instance()->timeSinceLastFrame;

		info.GetReturnValue().Set(timeSinceLastFrame);
	});

	V8Helper::_instance->registerFunction("insertTextAt", [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("insertTextAt requires 3 arguments");
			return;
		}

		String::Utf8Value textUTF8(args[0]);
		string text = *textUTF8;

		int x = args[1]->Int32Value();
		int y = args[2]->Int32Value();

		std::thread t([text, x, y]() {

			using namespace std::chrono_literals;

			//string clipboardText = "test string!! :)";

			{

				OpenClipboard(nullptr);

				EmptyClipboard();

				HGLOBAL hg = GlobalAlloc(GMEM_MOVEABLE, text.size() + 1);
				if (!hg) {
					CloseClipboard();
					return;
				}

				memcpy(GlobalLock(hg), text.c_str(), text.size() + 1);
				GlobalUnlock(hg);
				SetClipboardData(CF_TEXT, hg);
				CloseClipboard();
				GlobalFree(hg);



			}

			{ 

				vector<INPUT> inputs;

				{ // MOUSE MOVE
					INPUT in;
					in.type = INPUT_MOUSE;
					in.mi.dx = (65535 * x) / 2560;
					in.mi.dy = (65535 * y) / 1440;
					in.mi.dwFlags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE;
					inputs.push_back(in);
				}

				{ // MOUSE DOWN
					INPUT in;
					in.type = INPUT_MOUSE;
					in.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
					inputs.push_back(in);
				}

				{ // MOUSE UP
					INPUT in;
					in.type = INPUT_MOUSE;
					in.mi.dwFlags = MOUSEEVENTF_LEFTUP;
					inputs.push_back(in);
				}

				{ // END DOWN
					INPUT in;
					in.type = INPUT_KEYBOARD;
					in.ki.wVk = VK_END;
					in.ki.dwFlags = 0;
					inputs.push_back(in);
				}

				{ // END UP
					INPUT in;
					in.type = INPUT_KEYBOARD;
					in.ki.wVk = VK_END;
					in.ki.dwFlags = KEYEVENTF_KEYUP;
					inputs.push_back(in);
				}

				{ // RETURN DOWN
					INPUT in;
					in.type = INPUT_KEYBOARD;
					in.ki.wVk = VK_RETURN;
					in.ki.dwFlags = 0;
					inputs.push_back(in);
				}

				{ // RETURN UP
					INPUT in;
					in.type = INPUT_KEYBOARD;
					in.ki.wVk = VK_RETURN;
					in.ki.dwFlags = KEYEVENTF_KEYUP;
					inputs.push_back(in);
				}

				{ // CTRL DOWN
					INPUT in;
					in.type = INPUT_KEYBOARD;
					in.ki.wVk = VK_CONTROL;
					in.ki.dwFlags = 0;
					inputs.push_back(in);
				}

				{ // V DOWN
					INPUT in;
					in.type = INPUT_KEYBOARD;
					in.ki.wVk = 'V';
					in.ki.dwFlags = 0;
					inputs.push_back(in);
				}

				{ // V UP
					INPUT in;
					in.type = INPUT_KEYBOARD;
					in.ki.wVk = 'V';
					in.ki.dwFlags = KEYEVENTF_KEYUP;
					inputs.push_back(in);
				}

				{ // CTRL UP
					INPUT in;
					in.type = INPUT_KEYBOARD;
					in.ki.wVk = VK_CONTROL;
					in.ki.dwFlags = KEYEVENTF_KEYUP;
					inputs.push_back(in);
				}

				SendInput(inputs.size(), inputs.data(), sizeof(INPUT));
			}

			using namespace std::chrono_literals;
			std::this_thread::sleep_for(100ms);

		});
		t.detach();

	});

	V8Helper::_instance->registerFunction("addEventListener", [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("addEventListener requires 2 arguments");
			return;
		}

		String::Utf8Value eventNameUTF8(args[0]);
		string eventName = *eventNameUTF8;

		Local<Value> callbackValue = args[1];
		Local<Function> callback = Local<Function>::Cast(callbackValue);

		auto isolate = Isolate::GetCurrent();
		auto persistent = PersistentFunction(isolate, callback);

		if (eventName == "mousemove") {
			Application::instance()->addMouseMoveListener([isolate, persistent](MouseMoveEvent data) {
				Local<Function> local = Local<Function>::New(isolate, persistent);

				int argc = 1;
				Local<Value> argv[1];

				Local<Object> obj = Object::New(isolate);

				auto keyX = String::NewFromUtf8(isolate, "x");
				auto valueX = Number::New(isolate, data.x);
				obj->Set(keyX, valueX);

				auto keyY = String::NewFromUtf8(isolate, "y");
				auto valueY = Number::New(isolate, data.y);
				obj->Set(keyY, valueY);

				argv[0] = obj;

				local->Call(local, argc, argv);
			});
		} else if (eventName == "mousescroll") {
			Application::instance()->addMouseScrollListener([isolate, persistent](MouseScrollEvent data) {
				Local<Function> local = Local<Function>::New(isolate, persistent);

				int argc = 1;
				Local<Value> argv[1];

				Local<Object> obj = Object::New(isolate);

				auto keyXOffset = String::NewFromUtf8(isolate, "xoffset");
				auto valueXOffset = Number::New(isolate, data.xoffset);
				obj->Set(keyXOffset, valueXOffset);

				auto keyYOffset = String::NewFromUtf8(isolate, "yoffset");
				auto valueYOffset = Number::New(isolate, data.yoffset);
				obj->Set(keyYOffset, valueYOffset);

				argv[0] = obj;

				local->Call(local, argc, argv);
			});
		} else if (eventName == "mousedown") {
			Application::instance()->addMouseDownListener([isolate, persistent](MouseButtonEvent data) {
				Local<Function> local = Local<Function>::New(isolate, persistent);

				int argc = 1;
				Local<Value> argv[1];

				Local<Object> obj = Object::New(isolate);

				auto keyButton = String::NewFromUtf8(isolate, "button");
				auto valueButton = Number::New(isolate, data.button);
				obj->Set(keyButton, valueButton);

				auto keyMods = String::NewFromUtf8(isolate, "mods");
				auto valueMods = Number::New(isolate, data.mods);
				obj->Set(keyMods, valueMods);

				argv[0] = obj;

				local->Call(local, argc, argv);
			});
		} else if (eventName == "mouseup") {
			Application::instance()->addMouseUpListener([isolate, persistent](MouseButtonEvent data) {
				Local<Function> local = Local<Function>::New(isolate, persistent);

				int argc = 1;
				Local<Value> argv[1];

				Local<Object> obj = Object::New(isolate);

				auto keyButton = String::NewFromUtf8(isolate, "button");
				auto valueButton = Number::New(isolate, data.button);
				obj->Set(keyButton, valueButton);

				auto keyMods = String::NewFromUtf8(isolate, "mods");
				auto valueMods = Number::New(isolate, data.mods);
				obj->Set(keyMods, valueMods);

				argv[0] = obj;

				local->Call(local, argc, argv);
			});
		} else if (eventName == "keyevent") {
			Application::instance()->addKeyEventListener([isolate, persistent](KeyEvent data) {
				Local<Function> local = Local<Function>::New(isolate, persistent);

				int argc = 1;
				Local<Value> argv[1];

				Local<Object> obj = Object::New(isolate);

				auto keyKey = String::NewFromUtf8(isolate, "key");
				auto valueKey = Number::New(isolate, data.key);
				obj->Set(keyKey, valueKey);

				auto keyScancode = String::NewFromUtf8(isolate, "scancode");
				auto valueScancode = Number::New(isolate, data.scancode);
				obj->Set(keyScancode, valueScancode);

				auto keyAction = String::NewFromUtf8(isolate, "action");
				auto valueAction = Number::New(isolate, data.action);
				obj->Set(keyAction, valueAction);

				auto keyMods = String::NewFromUtf8(isolate, "mods");
				auto valueMods = Number::New(isolate, data.mods);
				obj->Set(keyMods, valueMods);

				argv[0] = obj;

				local->Call(local, argc, argv);
			});
		}
		
	});



	Local<Object> obj = tpl->NewInstance();

	context->Global()->Set(
		String::NewFromUtf8(isolate, "window"),
		obj
	);
}