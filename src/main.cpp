
#define NOMINMAX

#include <iostream>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <thread>

#include "GL\glew.h"
#include "GLFW\glfw3.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"
#include "implot/implot.h"
#include "implot/implot_internal.h"

#include "modules/progressive/ProgressiveLoader.h"
#include "modules/progressive/progressive.h"
#include "modules/progressive/ProgressiveBINLoader.h"

#include "GLTimerQueries.h"
#include "unsuck.hpp"
#include "LasLoader.h"


using std::unordered_map;
using std::vector;
using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::thread;

using namespace LASLoaderThreaded;

namespace fs = std::filesystem;

static long long start_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();

#define IMGUI_ENABLED

#include "Application.h"
#include "V8Helper.h"
#include "utils.h"
#include "Shader.h"

struct GLUpdateBuffer{
	void* mapPtr = nullptr;
	uint32_t size = 0;
	GLuint handle = 0;
	void* data = nullptr;
};

static GLUpdateBuffer updateBuffer = GLUpdateBuffer();

static void APIENTRY debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {


	//if (
	//	severity == GL_DEBUG_SEVERITY_NOTIFICATION 
	//	|| severity == GL_DEBUG_SEVERITY_LOW 
	//	|| severity == GL_DEBUG_SEVERITY_MEDIUM
	//	) {
	//	return;
	//}

	cout << "OPENGL DEBUG CALLBACK: " << message << endl;
}


void error_callback(int error, const char* description){
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){

	cout << "key: " << key << ", scancode: " << scancode << ", action: " << action << ", mods: " << mods << endl;

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}

	KeyEvent data = { key, scancode, action, mods };

	Application::instance()->dispatchKeyEvent(data);
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos){
	//cout << "xpos: " << xpos << ", ypos: " << ypos << endl;
	MouseMoveEvent data = { xpos, ypos };

	Application::instance()->dispatchMouseMoveEvent(data);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
	MouseScrollEvent data = { xoffset, yoffset };

	Application::instance()->dispatchMouseScrollEvent(data);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods){
	//if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
	//	popup_menu();

	MouseButtonEvent data = { button, action, mods };

	if (action == GLFW_PRESS) {
		Application::instance()->dispatchMouseDownEvent(data);
	} else if(action == GLFW_RELEASE) {
		Application::instance()->dispatchMouseUpEvent(data);
	}
}

uint64_t frameCount = 0;

// ScrollingBuffer from ImPlot implot_demo.cpp.
// MIT License
// url: https://github.com/epezent/implot
struct ScrollingBuffer {
	int MaxSize;
	int Offset;
	ImVector<ImVec2> Data;
	ScrollingBuffer() {
		MaxSize = 2000;
		Offset = 0;
		Data.reserve(MaxSize);
	}
	void AddPoint(float x, float y) {
		if (Data.size() < MaxSize)
			Data.push_back(ImVec2(x, y));
		else {
			Data[Offset] = ImVec2(x, y);
			Offset = (Offset + 1) % MaxSize;
		}
	}
	void Erase() {
		if (Data.size() > 0) {
			Data.shrink(0);
			Offset = 0;
		}
	}
};



int main() {

	cout << std::setprecision(3) << std::fixed;
	cout << "<main> " << "(" << now() << ")" << endl;

	//{
	//	cout << "building js package" << endl;
	//
	//	std::system("cd ../../ & rollup -c");
	//}
	//cout << "<built> " << "(" << now() << ")" << endl;

	glfwSetErrorCallback(error_callback);

	if (!glfwInit()) {
		// Initialization failed
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	//glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
	glfwWindowHint(GLFW_DECORATED, false);

	int numMonitors;
	GLFWmonitor** monitors = glfwGetMonitors(&numMonitors);

	GLFWwindow* window = nullptr;

	cout << "<20> " << "(" << now() << ")" << endl;

	cout << "<create windows>" << endl;
	if (numMonitors > 1) {
		const GLFWvidmode * modeLeft = glfwGetVideoMode(monitors[0]);
		const GLFWvidmode * modeRight = glfwGetVideoMode(monitors[1]);

		window = glfwCreateWindow(modeRight->width, modeRight->height - 300, "Simple example", nullptr, nullptr);

		if (!window) {
			glfwTerminate();
			exit(EXIT_FAILURE);
		}

		glfwSetWindowPos(window, modeLeft->width, 0);
	} else {
		const GLFWvidmode * mode = glfwGetVideoMode(monitors[0]);

		window = glfwCreateWindow(mode->width / 2, mode->height / 2, "Simple example", nullptr, nullptr);

		if (!window) {
			glfwTerminate();
			exit(EXIT_FAILURE);
		}

		glfwSetWindowPos(window, mode->width / 2, 2 * mode->height / 3);
	}

	cout << "<windows created> " << "(" << now() << ")" << endl;

	cout << "<set input callbacks>" << endl;
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "glew error: %s\n", glewGetErrorString(err));
	}

	cout << "<glewInit done> " << "(" << now() << ")" << endl;

	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
	glDebugMessageCallback(debugCallback, NULL);

#ifdef IMGUI_ENABLED
	{ // SETUP IMGUI
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImPlot::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init("#version 450");
		ImGui::StyleColorsDark();
	}
#endif

	//high_resolution_clock::time_point start = high_resolution_clock::now();
	//high_resolution_clock::time_point previous = start;

	int fpsCounter = 0;
	//high_resolution_clock::time_point lastFPSTime = start;
	double start = now();
	double tPrevious = start;
	double tPreviousFPSMeasure = start;


	V8Helper::instance()->window = window;
	V8Helper::instance()->setupV8();

	cout << "<V8 has been set up> " << "(" << now() << ")" << endl;

	{
		cout << "<run start.js>" << endl;
		string code = loadFileAsString("../../src_js/start.js");
		V8Helper::instance()->runScript(code);
	}

	cout << "<start.js was executed> " << "(" << now() << ")" << endl;

	auto updateJS = V8Helper::instance()->compileScript("update();");
	auto renderJS = V8Helper::instance()->compileScript("render();");

	{

		int numPoints = 30'000'000;
		int bytesPerPoint = 16;

		updateBuffer.size = numPoints * bytesPerPoint;

		glCreateBuffers(1, &updateBuffer.handle);

		{// map buffer method, see https://www.slideshare.net/CassEveritt/approaching-zero-driver-overhead/85
			GLbitfield mapFlags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
			GLbitfield storageFlags = mapFlags | GL_DYNAMIC_STORAGE_BIT;

			glNamedBufferStorage(updateBuffer.handle, updateBuffer.size, nullptr, storageFlags);

			updateBuffer.mapPtr = glMapNamedBufferRange(updateBuffer.handle, 0, updateBuffer.size, mapFlags);
		}

		//{ // bufferData method
		//	
		//	glNamedBufferData(updateBuffer.handle, updateBuffer.size, nullptr, GL_DYNAMIC_DRAW);

		//	updateBuffer.data = malloc(10'000'000 * 16);
		//}
	}

	V8Helper::_instance->registerFunction("loadLASProgressive", [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("loadLASProgressive requires 1 arguments");
			return;
		}

		String::Utf8Value fileUTF8(args[0]);
		string file = *fileUTF8;

		auto loader = loadLasProgressive(file);
		cout << loader << endl;

		auto isolate = Isolate::GetCurrent();
		Local<ObjectTemplate> lasTempl = ObjectTemplate::New(isolate);
		lasTempl->SetInternalFieldCount(1);

		lasTempl->Set(String::NewFromUtf8(isolate, "isDone"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
			if (args.Length() != 0) {
				V8Helper::_instance->throwException("isDone requires 0 arguments");
				return;
			}

			auto isolate = Isolate::GetCurrent();

			Local<Object> self = args.Holder();
			Local<External> wrap = Local<External>::Cast(self->GetInternalField(0));
			void* ptr = wrap->Value();
			ProgressiveLoader* loader = static_cast<ProgressiveLoader*>(ptr);

			bool isDone = loader->isDone();

			args.GetReturnValue().Set(v8::Boolean::New(isolate, isDone));
		}));

		lasTempl->Set(String::NewFromUtf8(isolate, "dispose"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
			if (args.Length() != 0) {
				V8Helper::_instance->throwException("dispose requires 0 arguments");
				return;
			}

			auto isolate = Isolate::GetCurrent();

			Local<Object> self = args.Holder();
			Local<External> wrap = Local<External>::Cast(self->GetInternalField(0));
			void* ptr = wrap->Value();
			ProgressiveLoader* loader = static_cast<ProgressiveLoader*>(ptr);

			loader->dispose();

			//loader->loader = nullptr;

			//delete loader;
		}));

		auto objLAS = lasTempl->NewInstance();
		objLAS->SetInternalField(0, External::New(isolate, loader));

		auto lNumPoints = v8::Integer::New(isolate, loader->loader->header.numPoints);

		auto lHandles = Array::New(isolate, loader->ssVertexBuffers.size());
		for (int i = 0; i < loader->ssVertexBuffers.size(); i++) {
			auto lHandle = v8::Integer::New(isolate, loader->ssVertexBuffers[i]);
			lHandles->Set(i, lHandle);
		}
		objLAS->Set(String::NewFromUtf8(isolate, "handles"), lHandles);
		objLAS->Set(String::NewFromUtf8(isolate, "numPoints"), lNumPoints);

		{
			Local<ObjectTemplate> boxTempl = ObjectTemplate::New(isolate);
			auto objBox = lasTempl->NewInstance();

			auto& header = loader->loader->header;

			auto lMin = Array::New(isolate, 3);
			lMin->Set(0, v8::Number::New(isolate, header.minX));
			lMin->Set(1, v8::Number::New(isolate, header.minY));
			lMin->Set(2, v8::Number::New(isolate, header.minZ));

			auto lMax = Array::New(isolate, 3);
			lMax->Set(0, v8::Number::New(isolate, header.maxX));
			lMax->Set(1, v8::Number::New(isolate, header.maxY));
			lMax->Set(2, v8::Number::New(isolate, header.maxZ));

			objBox->Set(String::NewFromUtf8(isolate, "min"), lMin);
			objBox->Set(String::NewFromUtf8(isolate, "max"), lMax);

			objLAS->Set(String::NewFromUtf8(isolate, "boundingBox"), objBox);
		}

	

		args.GetReturnValue().Set(objLAS);
	});

	V8Helper::_instance->registerFunction("loadLAS", [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("loadLAS requires 1 arguments");
			return;
		}


		String::Utf8Value fileUTF8(args[0]);
		string file = *fileUTF8;


		// LOAD LAS/LAZ
		auto tStart = now();
		auto las = LasLoader::loadLas(file);

		// CREATE GPU BUFFERS
		GLuint buf_XYZ = -1;
		{
			auto src = las->buffers["XYZ"];
			glCreateBuffers(1, &buf_XYZ);
			glNamedBufferData(buf_XYZ, src->size, src->data, GL_STATIC_DRAW);
		}
		
		GLuint buf_rgba = -1;
		{
			auto src = las->buffers["rgba"];
			glCreateBuffers(1, &buf_rgba);
			glNamedBufferData(buf_rgba, src->size, src->data, GL_STATIC_DRAW);
		}

		GLuint buf_XYZRGBA = -1;
		{
			auto src = las->buffers["XYZRGBA"];
			glCreateBuffers(1, &buf_XYZRGBA);
			glNamedBufferData(buf_XYZRGBA, src->size, src->data, GL_STATIC_DRAW);			
		}

		GLuint buf_XYZRGBA_uint16 = -1;
		{
			auto src = las->buffers["XYZRGBA_uint16"];
			glCreateBuffers(1, &buf_XYZRGBA_uint16);
			glNamedBufferData(buf_XYZRGBA_uint16, src->size, src->data, GL_STATIC_DRAW);
		}

		GLuint buf_XYZRGBA_uint13 = -1;
		{
			auto src = las->buffers["XYZRGBA_uint13"];
			glCreateBuffers(1, &buf_XYZRGBA_uint13);
			glNamedBufferData(buf_XYZRGBA_uint13, src->size, src->data, GL_STATIC_DRAW);
		}

		// V8 STUFF

		auto isolate = Isolate::GetCurrent();
		Local<ObjectTemplate> lasTempl = ObjectTemplate::New(isolate);
		auto objLAS = lasTempl->NewInstance();

		auto lNumPoints = v8::Integer::New(isolate, las->numPoints);

		auto lHandles = Array::New(isolate, 5);

		auto lHandle0 = v8::Integer::New(isolate, buf_XYZ);
		lHandles->Set(0, lHandle0);

		auto lHandle1 = v8::Integer::New(isolate, buf_rgba);
		lHandles->Set(1, lHandle1);

		auto lHandle2 = v8::Integer::New(isolate, buf_XYZRGBA);
		lHandles->Set(2, lHandle2);

		auto lHandle3 = v8::Integer::New(isolate, buf_XYZRGBA_uint16);
		lHandles->Set(3, lHandle3);

		auto lHandle4 = v8::Integer::New(isolate, buf_XYZRGBA_uint13);
		lHandles->Set(4, lHandle4);

		objLAS->Set(String::NewFromUtf8(isolate, "handles"), lHandles);
		objLAS->Set(String::NewFromUtf8(isolate, "numPoints"), lNumPoints);

		{
			Local<ObjectTemplate> boxTempl = ObjectTemplate::New(isolate);
			auto objBox = lasTempl->NewInstance();


			auto lMin = Array::New(isolate, 3);
			lMin->Set(0, v8::Number::New(isolate, las->minX));
			lMin->Set(1, v8::Number::New(isolate, las->minY));
			lMin->Set(2, v8::Number::New(isolate, las->minZ));

			auto lMax = Array::New(isolate, 3);
			lMax->Set(0, v8::Number::New(isolate, las->maxX));
			lMax->Set(1, v8::Number::New(isolate, las->maxY));
			lMax->Set(2, v8::Number::New(isolate, las->maxZ));

			objBox->Set(String::NewFromUtf8(isolate, "min"), lMin);
			objBox->Set(String::NewFromUtf8(isolate, "max"), lMax);

			objLAS->Set(String::NewFromUtf8(isolate, "boundingBox"), objBox);
		}

		auto duration = now() - tStart;
		cout << "duration: " << duration << "s" << endl;

		auto pObjLAS = Persistent<Object, CopyablePersistentTraits<Object>>(isolate, objLAS);

		args.GetReturnValue().Set(objLAS);
	});

	V8Helper::_instance->registerFunction("getTimings", [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("getTimings requires 0 arguments");
			return;
		}

		Isolate* isolate = Isolate::GetCurrent();

		stringstream ss;
		ss << "{" << endl;
		ss << "\t\"timings\": [" << endl;


		auto durations = GLTimerQueries::instance()->durations;
		auto stats = GLTimerQueries::instance()->stats;

		int i = 0;
		for (auto duration : durations) {

			auto stat = stats[duration.label];
			double avg = stat.sum / stat.count;
			double min = stat.min;
			double max = stat.max;
			
			ss << "\t\t\t{\"label\": \"" << duration.label << "\", \"avg\": " << avg << ", \"min\": " << min << ", \"max\": " << max << "}";
			if (i < durations.size() - 1) {
				ss << ",";
			}
			ss << endl;
			
			i++;
		}

		ss << "\t]" << endl;
		ss << "}" << endl;

		//Local<String> v8String = String::NewFromUtf8(v8::Isolate::GetCurrent(), ss.str().c_str(), v8::String::kNormalString);

		auto str = String::NewFromUtf8(isolate, ss.str().c_str());

		args.GetReturnValue().Set(str);

	});


	cout << "<entering first render loop> " << "(" << now() << ")" << endl;

	vector<float> frameTimes(1000, 0);
	float frameTimesArray[1000];
	string selectedMethod = "";

	while (!glfwWindowShouldClose(window)){

		GLTimerQueries::frameStart();

		//cout << frameCount << endl;

		// ----------------
		// TIME
		// ----------------

		//high_resolution_clock::time_point now = high_resolution_clock::now();
		//double nanosecondsSinceLastFrame = double((now - previous).count());
		//double nanosecondsSinceLastFPSMeasure = double((now - lastFPSTime).count());
		//
		//double timeSinceLastFrame = nanosecondsSinceLastFrame / 1'000'000'000;
		//double timeSinceLastFPSMeasure = nanosecondsSinceLastFPSMeasure / 1'000'000'000;

		double tCurrent = now();
		double timeSinceLastFrame = tCurrent - tPrevious;
		tPrevious = tCurrent;

		double timeSinceLastFPSMeasure = tCurrent - tPreviousFPSMeasure;

		if(timeSinceLastFPSMeasure >= 1.0){
			double fps = double(fpsCounter) / timeSinceLastFPSMeasure;
			stringstream ssFPS; 
			ssFPS << fps;
			
			V8Helper::instance()->debugValue["FPS"] = ssFPS.str();

			tPreviousFPSMeasure = tCurrent;
			fpsCounter = 0;
		}
		V8Helper::instance()->timeSinceLastFrame = timeSinceLastFrame;
		frameTimes[frameCount % frameTimes.size()] = timeSinceLastFrame;


		// ----------------

		EventQueue::instance->process();

		#ifdef IMGUI_ENABLED
			// UPDATE IMGUI
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
		#endif

		//if (timeSinceLastFrame > 0.016) {
		//	cout << "too slow! time since last frame: " << int(timeSinceLastFrame * 1000.0) << "ms" << endl;
		//}

		{
			static double toggle = now();
			static int missedFrames = 0;

			if (timeSinceLastFrame > 0.016) {
				missedFrames++;
			}

			if(now() - toggle >= 1.0){

				string msg = "";
				if (missedFrames == 0) {
					msg = std::to_string(missedFrames);
				} else {
					msg = "<b style=\"color:red\">" + std::to_string(missedFrames) + "</b>";
				}
				V8Helper::instance()->debugValue["#missed frames"] = msg;

				missedFrames = 0;
				toggle = now();
			}

		}


		// ----------------
		// RENDER WITH JAVASCRIPT
		// ----------------
		
		//Application::instance()->lockScreenCapture();
		updateJS->Run(V8Helper::instance()->context);
		renderJS->Run(V8Helper::instance()->context);
		//Application::instance()->unlockScreenCapture();

		#ifdef IMGUI_ENABLED
		{ // RENDER IMGUI PERFORMANCE WINDOW

			string fps = V8Helper::instance()->debugValue["FPS"];

			ImGui::Begin("Performance");
			ImGui::Text((rightPad("FPS:", 30) + fps).c_str());

			static float history = 2.0f;
			static ScrollingBuffer sFrames;
			static ScrollingBuffer s60fps;
			float t = now();
			sFrames.AddPoint(t, 1000.0 * timeSinceLastFrame);
			s60fps.AddPoint(t, 1000.0 / 60.0);
			//s60fps.AddPoint(t - history, 1000.0 / 60.0);
			//s60fps.AddPoint(t, 1000.0 / 60.0);
			static ImPlotAxisFlags rt_axis = ImPlotAxisFlags_NoTickLabels;
			ImPlot::SetNextPlotLimitsX(t - history, t, ImGuiCond_Always);
			ImPlot::SetNextPlotLimitsY(0, 30, ImGuiCond_Always);

			if (ImPlot::BeginPlot("Timings", nullptr, nullptr, ImVec2(-1, 200))){ // , ImVec2(-1, 150), 0, rt_axis, rt_axis | ImPlotAxisFlags_LockMin)) {
				//ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.75f);

				auto x = &sFrames.Data[0].x;
				auto y = &sFrames.Data[0].y;
				ImPlot::PlotShaded("frame time(ms)", x, y, sFrames.Data.size(), -Infinity, sFrames.Offset, 2 * sizeof(float));

				//ImPlot::PlotShaded("60 FPS", &s60fps.Data[0].x, &s60fps.Data[0].y, s60fps.Data.size(), 0.0, s60fps.Offset, 2 * sizeof(float));
				ImPlot::PlotLine("60 FPS", &s60fps.Data[0].x, &s60fps.Data[0].y, s60fps.Data.size(), s60fps.Offset, 2 * sizeof(float));


				ImPlot::EndPlot();
			}

			{
				string text = "Durations: \n";
				auto durations = GLTimerQueries::instance()->durations;
				auto stats = GLTimerQueries::instance()->stats;

				for (auto duration : durations) {

					auto stat = stats[duration.label];
					double avg = stat.sum / stat.count;
					double min = stat.min;
					double max = stat.max;

					text = text + rightPad(duration.label + ":", 30);
					text = text + "avg(" + formatNumber(avg, 3) + ") ";
					text = text + "min(" + formatNumber(min, 3) + ") ";
					text = text + "max(" + formatNumber(max, 3) + ")\n";

					//double milies = double(duration.nanos / 1000ul) / 1000.0;
					//text = text + rightPad(duration.label + ":", 30) + formatNumber(milies, 3) + "\n";
				}

				ImGui::Text(text.c_str());
			}
			

			ImGui::End();
		}

		{ // IMGUI SETTINGS WINDOW
			ImGui::Begin("Settings");

			//ImGui::Text("test");

			const char* items[] = { 
				"GL_POINTS", 
				"atomicMin", 
				"early-z", 
				"reduce", 
				"early-z & reduce",
				"dedup",
				"busy-loop",
				"just-set",
				"HQS",
				"HQS1",
				"HQS1R"
			};
			static int item_current_idx = 4;
			int numItems = IM_ARRAYSIZE(items);

			ImGui::Text("Method:");
			if (ImGui::BeginListBox("##listbox 2", ImVec2(-FLT_MIN, numItems * ImGui::GetTextLineHeightWithSpacing()))){
				for (int n = 0; n < numItems; n++){
					const bool is_selected = (item_current_idx == n);

					if (ImGui::Selectable(items[n], is_selected)) {
						item_current_idx = n;
					}

					// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
					if (is_selected) {
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndListBox();
			}
			selectedMethod = items[item_current_idx];

			if (selectedMethod == "GL_POINTS") {
				V8Helper::instance()->runScriptSilent("renderDebug = renderPointCloudBasic");
			} else if (selectedMethod == "atomicMin") {
				V8Helper::instance()->runScriptSilent("renderDebug = renderPointCloudCompute");
			} else if (selectedMethod == "early-z") {
				V8Helper::instance()->runScriptSilent("renderDebug = render_compute_earlyDepth");
			}else if (selectedMethod == "reduce") {
				V8Helper::instance()->runScriptSilent("renderDebug = render_compute_ballot");
			}else if (selectedMethod == "early-z & reduce") {
				V8Helper::instance()->runScriptSilent("renderDebug = render_compute_ballot_earlyDepth");
			}else if (selectedMethod == "dedup") {
				V8Helper::instance()->runScriptSilent("renderDebug = render_compute_ballot_earlyDepth_dedup");
			}else if (selectedMethod == "busy-loop") {
				V8Helper::instance()->runScriptSilent("renderDebug = render_compute_guenther");
			}else if (selectedMethod == "just-set") {
				V8Helper::instance()->runScriptSilent("renderDebug = renderComputeJustSet");
			}else if (selectedMethod == "HQS") {
				V8Helper::instance()->runScriptSilent("renderDebug = renderComputeHQS");
			}else if (selectedMethod == "HQS1") {
				V8Helper::instance()->runScriptSilent("renderDebug = renderComputeHQS_1x64bit");
			}else if (selectedMethod == "HQS1R") {
				V8Helper::instance()->runScriptSilent("renderDebug = renderComputeHQS_1x64bit_fast");
			}

			//ImGui::Text(("Selected: " + selectedMethod).c_str());

			ImGui::End();
		}

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		#endif

		GLTimerQueries::frameEnd();


		// ----------------
		// swap and events
		// ----------------
		glfwSwapBuffers(window);
		glfwPollEvents();

		fpsCounter++;
		frameCount++;
	}

	#ifdef IMGUI_ENABLED
		ImPlot::DestroyContext();
		ImGui::DestroyContext();
	#endif


	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);

	return 0;
}