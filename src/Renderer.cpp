
#include <filesystem>

#include "Renderer.h"

#include "drawBoundingBoxes.h"
#include "drawBoundingBoxesIndirect.h"
#include "drawBoxes.h"
#include "drawPoints.h"
#include "GLTimerQueries.h"
#include "Runtime.h"

namespace fs = std::filesystem;

auto _controls = make_shared<OrbitControls>();

static void APIENTRY debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {

	if (
		severity == GL_DEBUG_SEVERITY_NOTIFICATION 
		|| severity == GL_DEBUG_SEVERITY_LOW 
		|| severity == GL_DEBUG_SEVERITY_MEDIUM
		) {
		return;
	}

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

	Runtime::keyStates[key] = action;

	cout << key << endl;
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos){
	ImGuiIO& io = ImGui::GetIO();
	if(io.WantCaptureMouse){
		return;
	}
	
	Runtime::mousePosition = {xpos, ypos};

	_controls->onMouseMove(xpos, ypos);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
	ImGuiIO& io = ImGui::GetIO();
	if(io.WantCaptureMouse){
		return;
	}

	_controls->onMouseScroll(xoffset, yoffset);
}


void drop_callback(GLFWwindow* window, int count, const char **paths){
	for(int i = 0; i < count; i++){
		cout << paths[i] << endl;
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods){

	cout << "start button: " << button << ", action: " << action << ", mods: " << mods << endl;

	ImGuiIO& io = ImGui::GetIO();
	if(io.WantCaptureMouse){
		return;
	}

	cout << "end button: " << button << ", action: " << action << ", mods: " << mods << endl;


	if(action == 1){
		Runtime::mouseButtons = Runtime::mouseButtons | (1 << button);
	}else if(action == 0){
		uint32_t mask = ~(1 << button);
		Runtime::mouseButtons = Runtime::mouseButtons & mask;
	}

	_controls->onMouseButton(button, action, mods);
}

Renderer::Renderer(){
	this->controls = _controls;
	camera = make_shared<Camera>();

	init();

	//framebuffer = this->createFramebuffer(128, 128);

	View view1;
	view1.framebuffer = this->createFramebuffer(128, 128);
	View view2;
	view2.framebuffer = this->createFramebuffer(128, 128);
	views.push_back(view1);
	views.push_back(view2);
}

void Renderer::init(){
	glfwSetErrorCallback(error_callback);

	if (!glfwInit()) {
		// Initialization failed
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
	glfwWindowHint(GLFW_DECORATED, true);

	int numMonitors;
	GLFWmonitor** monitors = glfwGetMonitors(&numMonitors);

	cout << "<create windows>" << endl;
	//if (numMonitors > 1) {
	//	const GLFWvidmode * modeLeft = glfwGetVideoMode(monitors[0]);
	//	const GLFWvidmode * modeRight = glfwGetVideoMode(monitors[1]);

	//	window = glfwCreateWindow(modeRight->width, modeRight->height - 300, "Simple example", nullptr, nullptr);

	//	if (!window) {
	//		glfwTerminate();
	//		exit(EXIT_FAILURE);
	//	}

	//	glfwSetWindowPos(window, modeLeft->width, 0);
	//} 
	//else 
	{
		const GLFWvidmode * mode = glfwGetVideoMode(monitors[0]);

		window = glfwCreateWindow(mode->width - 100, mode->height - 100, "Simple example", nullptr, nullptr);

		if (!window) {
			glfwTerminate();
			exit(EXIT_FAILURE);
		}

		glfwSetWindowPos(window, 50, 50);
	}

	cout << "<set input callbacks>" << endl;
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback);
	//glfwSetDropCallback(window, drop_callback);

	static Renderer* ref = this;
	glfwSetDropCallback(window, [](GLFWwindow*, int count, const char **paths){

		vector<string> files;
		for(int i = 0; i < count; i++){
			string file = paths[i];
			files.push_back(file);
		}

		for(auto &listener : ref->fileDropListeners){
			listener(files);
		}
	});

	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "glew error: %s\n", glewGetErrorString(err));
	}

	cout << "<glewInit done> " << "(" << now() << ")" << endl;

	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_HIGH, 0, NULL, GL_TRUE);
	glDebugMessageCallback(debugCallback, NULL);

	{ // SETUP IMGUI
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImPlot::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init("#version 450");
		ImGui::StyleColorsDark();
	}

}

shared_ptr<Texture> Renderer::createTexture(int width, int height, GLuint colorType) {

	auto texture = Texture::create(width, height, colorType, this);

	return texture;
}

shared_ptr<Framebuffer> Renderer::createFramebuffer(int width, int height) {

	auto framebuffer = Framebuffer::create(this);
	//framebuffer->setSize(width, height);

	GLenum status = glCheckNamedFramebufferStatus(framebuffer->handle, GL_FRAMEBUFFER);

	if (status != GL_FRAMEBUFFER_COMPLETE) {
		cout << "framebuffer incomplete" << endl;
	}

	return framebuffer;
}

void Renderer::loop(function<void(void)> update, function<void(void)> render){

	int fpsCounter = 0;
	double start = now();
	double tPrevious = start;
	double tPreviousFPSMeasure = start;

	vector<float> frameTimes(1000, 0);
	//float frameTimesArray[1000];

	while (!glfwWindowShouldClose(window)){

		GLTimerQueries::frameStart();

		Debug::clearFrameStats();

		// TIMING
		double timeSinceLastFrame;
		{
			double tCurrent = now();
			timeSinceLastFrame = tCurrent - tPrevious;
			tPrevious = tCurrent;

			double timeSinceLastFPSMeasure = tCurrent - tPreviousFPSMeasure;

			if(timeSinceLastFPSMeasure >= 1.0){
				this->fps = double(fpsCounter) / timeSinceLastFPSMeasure;

				tPreviousFPSMeasure = tCurrent;
				fpsCounter = 0;
			}
			frameTimes[frameCount % frameTimes.size()] = timeSinceLastFrame;
		}
		

		// WINDOW
		int width, height;
		glfwGetWindowSize(window, &width, &height);
		camera->setSize(width, height);
		this->width = width;
		this->height = height;

		EventQueue::instance()->process();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, this->width, this->height);


		{ 
			controls->update();

			camera->world = controls->world;
			camera->position = camera->world * dvec4(0.0, 0.0, 0.0, 1.0);

			drawQueue.clear();
		}

		if(vrEnabled && !Debug::dummyVR){
			auto ovr = OpenVRHelper::instance();
			ovr->updatePose();
			ovr->processEvents();

			float near = 0.1;
			float far = 100'000.0;

			dmat4 projLeft = ovr->getProjection(vr::EVREye::Eye_Left, near, far);
			dmat4 projRight = ovr->getProjection(vr::EVREye::Eye_Right, near, far);
		}

		{ // UPDATE & RENDER
			camera->update();
			update();
			camera->update();


			glClearColor(0.0f, 0.2f, 0.3f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			glEnable(GL_DEPTH_TEST);

			render();

			{
				auto& view = this->views[0];

				glBindFramebuffer(GL_FRAMEBUFFER, view.framebuffer->handle);
			}

			if(vrEnabled){

				{ // LEFT
					auto& viewParams = views[0];

					glBindFramebuffer(GL_FRAMEBUFFER, viewParams.framebuffer->handle);

					auto camera_vr_world = make_shared<Camera>();
					camera_vr_world->view = viewParams.view;
					camera_vr_world->proj = viewParams.proj;

					_drawBoundingBoxes(camera_vr_world.get(), drawQueue.boundingBoxes);
					_drawBoxes(camera_vr_world.get(), drawQueue.boxes);
				}

				{ // RIGHT
					auto& viewParams = views[1];

					glBindFramebuffer(GL_FRAMEBUFFER, viewParams.framebuffer->handle);

					auto camera_vr_world = make_shared<Camera>();
					camera_vr_world->view = viewParams.view;
					camera_vr_world->proj = viewParams.proj;

					_drawBoundingBoxes(camera_vr_world.get(), drawQueue.boundingBoxes);
					_drawBoxes(camera_vr_world.get(), drawQueue.boxes);
				}

				
			}else{
				//glBindFramebuffer(GL_FRAMEBUFFER, 0);
				//glViewport(0, 0, this->width, this->height);

				_drawBoundingBoxes(camera.get(), drawQueue.boundingBoxes);
				_drawBoxes(camera.get(), drawQueue.boxes);
			}
			

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glViewport(0, 0, this->width, this->height);

		}



		// IMGUI
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		auto windowSize_infos = ImVec2(490, 280);
		auto windowSize_perf = ImVec2(490, 340);
		auto windowSize_datasets = ImVec2(490, 260);
		auto windowSize_debug = ImVec2(490, 200);

		auto windowSize_state = ImVec2(370, 320);
		auto windowSize_settings = ImVec2(370, 370);

		{ // RENDER IMGUI PERFORMANCE WINDOW

			stringstream ssFPS; 
			ssFPS << this->fps;
			string strFps = ssFPS.str();

			ImGui::SetNextWindowPos(ImVec2(10, 10 + windowSize_infos.y + 10));
			ImGui::SetNextWindowSize(windowSize_perf);

			ImGui::Begin("Performance");
			ImGui::Text((rightPad("FPS:", 30) + strFps).c_str());

			static float history = 2.0f;
			static ScrollingBuffer sFrames;
			static ScrollingBuffer s60fps;
			static ScrollingBuffer s120fps;
			float t = now();

			{
				auto durations = GLTimerQueries::instance()->durations;
				auto stats = GLTimerQueries::instance()->stats;

				if(stats.find("frame") != stats.end()){
					auto stat = stats["frame"];
					double avg = stat.sum / stat.count;

					sFrames.AddPoint(t, avg);
				}else{
					// add bogus entry until proper frame times become available
					sFrames.AddPoint(t, 100.0);
				}
			}

			// sFrames.AddPoint(t, 1000.0f * timeSinceLastFrame);
			s60fps.AddPoint(t, 1000.0f / 60.0f);
			s120fps.AddPoint(t, 1000.0f / 120.0f);
			static ImPlotAxisFlags rt_axis = ImPlotAxisFlags_NoTickLabels;
			ImPlot::SetNextPlotLimitsX(t - history, t, ImGuiCond_Always);
			ImPlot::SetNextPlotLimitsY(0, 30, ImGuiCond_Always);

			if (ImPlot::BeginPlot("Timings", nullptr, nullptr, ImVec2(-1, 200))){

				auto x = &sFrames.Data[0].x;
				auto y = &sFrames.Data[0].y;
				ImPlot::PlotShaded("frame time(ms)", x, y, sFrames.Data.size(), -Infinity, sFrames.Offset, 2 * sizeof(float));

				ImPlot::PlotLine("16.6ms (60 FPS)", &s60fps.Data[0].x, &s60fps.Data[0].y, s60fps.Data.size(), s60fps.Offset, 2 * sizeof(float));
				ImPlot::PlotLine(" 8.3ms (120 FPS)", &s120fps.Data[0].x, &s120fps.Data[0].y, s120fps.Data.size(), s120fps.Offset, 2 * sizeof(float));

				ImPlot::EndPlot();
			}

			{

				static vector<Duration> s_durations;
				static unordered_map<string, GLTStats> s_stats;
				static float toggle = 0.0;

				// update duration only once per second
				// updating at full speed makes it hard to read it
				if(toggle > 1.0){
					s_durations = GLTimerQueries::instance()->durations;
					s_stats = GLTimerQueries::instance()->stats;

					toggle = 0.0;
				}
				toggle = toggle + timeSinceLastFrame;


				string text = "Durations: \n";
				// auto durations = GLTimerQueries::instance()->durations;
				// auto stats = GLTimerQueries::instance()->stats;

				for (auto duration : s_durations) {

					auto stat = s_stats[duration.label];
					double avg = stat.sum / stat.count;
					double min = stat.min;
					double max = stat.max;

					text = text + rightPad(duration.label + ":", 30);
					text = text + "avg(" + formatNumber(avg, 3) + ") ";
					text = text + "min(" + formatNumber(min, 3) + ") ";
					text = text + "max(" + formatNumber(max, 3) + ")\n";
				}

				ImGui::Text(text.c_str());
			}


			ImGui::End();
		}

		{ // RENDER IMGUI STATE

			stringstream ssFPS; 
			ssFPS << this->fps;
			string strFps = ssFPS.str();

			ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - windowSize_state.x - 10, 10));
			ImGui::SetNextWindowSize(windowSize_state);

			ImGui::Begin("State");
			ImGui::Text((rightPad("FPS:", 10) + strFps).c_str());

			string strCamera = 
				formatNumber(camera->position.x, 2) + " / " +
				formatNumber(camera->position.y, 2) + " / " +
				formatNumber(camera->position.z, 2);

			ImGui::Text((rightPad("campos:", 10) + strCamera).c_str());

			string strPoints = "";
			if(Runtime::lasLoaderSparse){
				strPoints = formatNumber(Runtime::lasLoaderSparse->numPointsLoaded);
			}
			ImGui::Text((rightPad("#points:", 10) + strPoints).c_str());

			ImGui::Separator(); 

			auto& values = Debug::getInstance()->values;
			for(auto& [key, value] : values){
				ImGui::Text(key.c_str());
				ImGui::SameLine(240.0);
				ImGui::Text(value.c_str());
			}

			if(Debug::frameStats.size() > 0){
				ImGui::Separator(); 
			}

			stringstream ssStats;
			for(auto& [key, value] : Debug::frameStats){

				if(key == "divider" && value == ""){
					ImGui::Separator(); 
				}else{
					ImGui::Text(key.c_str());
					ImGui::SameLine(240.0);
					ImGui::Text(value.c_str());

					ssStats << key << "\t" << value << endl;
				}
			}
			
			if(Debug::frameStats.size() > 0){
				ImGui::Separator(); 
			}

			if(Debug::frameStats.size() > 0 && ImGui::Button("copy")) {
#ifdef _WIN32
				toClipboard(ssStats.str());
#endif
			}

			ImGui::End();
		}

		{ // RENDER IMGUI INPUT

			ImGui::SetNextWindowPos(ImVec2(
				10, 
				10 + windowSize_infos.y + 10 + windowSize_perf.y + 10 + windowSize_datasets.y + 10));
			ImGui::SetNextWindowSize(windowSize_debug);

			ImGui::Begin("Debug Stuff");
			
			//if (ImGui::Button("toggle update")){
			//	Debug::updateEnabled = !Debug::updateEnabled;
			//}

			{
				static bool checked = Debug::updateEnabled; 
				ImGui::Checkbox("update", &checked);

				Debug::updateEnabled = checked;
			}

			{
				static bool checked = Debug::showBoundingBox; 
				ImGui::Checkbox("show bounding box", &checked);

				Debug::showBoundingBox = checked;
			}

			{
				static bool checked = Debug::colorizeChunks; 
				ImGui::Checkbox("colorize chunks", &checked);

				Debug::colorizeChunks = checked;
			}

			{
				static bool checked = Debug::colorizeOverdraw; 
				ImGui::Checkbox("colorize overdraw", &checked);

				Debug::colorizeOverdraw = checked;
			}

			// {
			// 	static bool checked = Debug::boolMisc; 
			// 	ImGui::Checkbox("misc", &checked);

			// 	Debug::boolMisc = checked;
			// }

			if (ImGui::Button("copy camera")) {
				auto pos = controls->getPosition();
				auto target = controls->target;

				stringstream ss;
				ss<< std::setprecision(2) << std::fixed;
				ss << "position: " << pos.x << ", " << pos.y << ", " << pos.z << endl;
				ss << "renderer->controls->yaw = " << controls->yaw << ";" << endl;
				ss << "renderer->controls->pitch = " << controls->pitch << ";" << endl;
				ss << "renderer->controls->radius = " << controls->radius << ";" << endl;
				ss << "renderer->controls->target = {" << target.x << ", " << target.y << ", " << target.z << "};" << endl;

				string str = ss.str();
#ifdef _WIN32
				toClipboard(str);
#endif
			}

			// if (ImGui::Button("copy vr matrices")) {
			// 	Debug::requestCopyVrMatrices = true;
			// }

			if (ImGui::Button("reset view")) {
				Debug::requestResetView = true;
			}

			// if (ImGui::Button("copy tree")) {

			// 	Debug::doCopyTree = true;
			// }

			// if (ImGui::Button("toggle VR")) {

			// 	this->toggleVR();
			// }


			ImGui::End();
		}

		{ // RENDER IMGUI INFOS

			ImGui::SetNextWindowPos(ImVec2(10, 10));
			ImGui::SetNextWindowSize(windowSize_infos);

			ImGui::Begin("Infos, features and limitations");

			ImGui::Text(R"ER01(Semi-user-friendly version of the research prototype for 
the fast software-rasterization of point clouds.)ER01");
			ImGui::Text(R"ER01(
- Drag&Drop LAS files to load them. 
- Requires Windows and NVIDIA GPUs. 
  PRs for AMD are welcome.
- Loads all points of the dropped point clouds, 
  regardless of available GPU memory. 
  Make sure you have about 2GB + 16 byte per point of GPU memory.
  (e.g. about 18GB GPU memory for 1 billion points)
- Maximum of 1 billion points.
				)ER01");
			ImGui::Text("URL: https://github.com/m-schuetz/compute_rasterizer");

			ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
			ImGui::Text(R"ER01(Issues: 
- Multi-threaded loading is now enabled, but may ocasionally cause 
  bugs (missing points). 
  Workaround: "numThreads = 1;" in LasLoaderSparse.cpp
- Loading compressed LAZ files is 10x slower than LAS.
				)ER01");
			// ImGui::InputText("##text1", txt_green, sizeof(txt_green));
			ImGui::PopStyleColor();

			
			ImGui::End();
		}

		{ // IMGUI SETTINGS WINDOW

			ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - windowSize_settings.x - 10, 10 + windowSize_state.y + 10));
			ImGui::SetNextWindowSize(windowSize_settings);

			ImGui::Begin("Settings");

			//const char* items[] = { 
			//	"loop",
			//	"loop_nodes",
			//	"loop_compress_nodewise",
			//	"loop_hqs",
			//	"loop_las",
			//	"loop_las_hqs"
			//};
			static int item_current_idx = 0;
			int numItems = Runtime::methods.size();
			string currentGroup = "none";

			ImGui::Text("Method:");
			if (ImGui::BeginListBox("##listbox 2", ImVec2(-FLT_MIN, (6 + numItems) * ImGui::GetTextLineHeightWithSpacing()))){
				for (int n = 0; n < numItems; n++){
					const bool is_selected = (item_current_idx == n);
					auto method = Runtime::methods[n];

					if(method->group != currentGroup){
						ImGui::Separator(); 
						
						string text = method->group;
						float font_size = ImGui::GetFontSize() * text.size() / 2;
						ImGui::SameLine(ImGui::GetWindowSize().x / 2 - font_size + (font_size / 2));
						ImGui::Text(text.c_str());
						ImGui::Separator();
						currentGroup = method->group;
					}

					if (ImGui::Selectable(method->name.c_str(), is_selected)) {
						item_current_idx = n;
					}

					


					// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
					if (is_selected) {
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndListBox();
			}

			ImGui::Text("LOD: ");
			static float LOD = Debug::LOD;
			ImGui::SliderFloat("rotation", &LOD, 0.0, 20.0);
			Debug::LOD = LOD;

			// {
			// 	static bool checked = Debug::lodEnabled; 
			// 	ImGui::Checkbox("enable LOD", &checked);

			// 	Debug::lodEnabled = checked;
			// }

			{
				static bool checked = Debug::frustumCullingEnabled; 
				ImGui::Checkbox("enable Frustum Culling", &checked);

				Debug::frustumCullingEnabled = checked;
			}

			{
				static bool checked = Debug::updateFrustum; 
				ImGui::Checkbox("update frustum", &checked);

				Debug::updateFrustum = checked;
			}

			{
				static bool checked = Debug::enableShaderDebugValue; 
				ImGui::Checkbox("read shader debug value", &checked);

				Debug::enableShaderDebugValue = checked;
			}

			this->selectedMethod = Runtime::methods[item_current_idx]->name;
			Runtime::setSelectedMethod(Runtime::methods[item_current_idx]->name);

			auto selected = Runtime::getSelectedMethod();
			if(selected){
				ImGui::Text(selected->description.c_str());
			}

			ImGui::End();
		}

		{ // IMGUI DATA SETS

			ImGui::SetNextWindowPos(ImVec2(
				10, 
				10 + windowSize_infos.y + 10 + windowSize_perf.y + 10));
			ImGui::SetNextWindowSize(windowSize_datasets);

			ImGui::Begin("Data Sets");

			auto lasfiles = Runtime::lasLoaderSparse;

			static int item_current_idx = 0;
			int numItems = lasfiles == nullptr ? 0 : lasfiles->files.size();

			ImGui::Text("Point Clouds:");
			if (ImGui::BeginListBox("##listbox 3", ImVec2(-FLT_MIN, (11) * ImGui::GetTextLineHeightWithSpacing()))){
				for (int n = 0; n < numItems; n++){
					const bool is_selected = (item_current_idx == n);

					auto lasfile = lasfiles->files[n];
					string filename = fs::path(lasfile->path).filename().string();

					if (ImGui::Selectable(filename.c_str(), is_selected)) {
						item_current_idx = n;
					}

					lasfile->isHovered = ImGui::IsItemHovered();
					lasfile->isDoubleClicked = ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0);

					// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
					if (is_selected) {
						ImGui::SetItemDefaultFocus();
					}

					lasfile->isSelected = is_selected;
				}
				ImGui::EndListBox();
			}
			ImGui::Text("Double click point cloud to zoom.");

			ImGui::End();
		}

		if(vrEnabled){
			auto source0 = views[0].framebuffer;
			auto source1 = views[1].framebuffer;

			glBlitNamedFramebuffer(
				source0->handle, 0,
				0, 0, source0->width, source0->height,
				0, 0, width / 2, height,
				GL_COLOR_BUFFER_BIT, GL_LINEAR);

			glBlitNamedFramebuffer(
				source1->handle, 0,
				0, 0, source1->width, source1->height,
				width / 2, 0, width, height,
				GL_COLOR_BUFFER_BIT, GL_LINEAR);
		}else{
			auto source = views[0].framebuffer;
			glBlitNamedFramebuffer(
				source->handle, 0,
				0, 0, source->width, source->height,
				0, 0, 0 + source->width, 0 + source->height,
				GL_COLOR_BUFFER_BIT, GL_LINEAR);
		}

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


		if(vrEnabled && !Debug::dummyVR){
			
			GLuint left = views[0].framebuffer->colorAttachments[0]->handle;
			GLuint right = views[1].framebuffer->colorAttachments[0]->handle;

			OpenVRHelper::instance()->submit(left, right);
			OpenVRHelper::instance()->postPresentHandoff();
		}

		// FINISH FRAME
		GLTimerQueries::frameEnd();

		glfwSwapBuffers(window);
		glfwPollEvents();

		// glFlush();
		// glFinish();

		fpsCounter++;
		frameCount++;
	}

	ImPlot::DestroyContext();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);

}

void Renderer::drawBox(glm::dvec3 position, glm::dvec3 scale, glm::ivec3 color){

	Box box;
	box.min = position - scale / 2.0;
	box.max = position + scale / 2.0;;
	box.color = color;

	drawQueue.boxes.push_back(box);

	//_drawBox(camera.get(), position, scale, color);
}

void Renderer::drawBoundingBox(glm::dvec3 position, glm::dvec3 scale, glm::ivec3 color){

	Box box;
	box.min = position - scale / 2.0;
	box.max = position + scale / 2.0;;
	box.color = color;

	drawQueue.boundingBoxes.push_back(box);
}

void Renderer::drawBoundingBoxes(Camera* camera, GLBuffer buffer){
	_drawBoundingBoxesIndirect(camera, buffer);
}

void Renderer::drawPoints(void* points, int numPoints){
	_drawPoints(camera.get(), points, numPoints);
}

void Renderer::drawPoints(GLuint vao, GLuint vbo, int numPoints) {
	_drawPoints(camera.get(), vao, vbo, numPoints);
}

void Renderer::toggleVR() {
	
	if(this->vrEnabled){
		if(!Debug::dummyVR){
			OpenVRHelper::instance()->stop();
		}
		this->vrEnabled = false;
	}else{
		if(!Debug::dummyVR){
			OpenVRHelper::instance()->start();
		}
		this->vrEnabled = true;
	}

}

void Renderer::setVR(bool enable) {
	if(enable){
		if(!Debug::dummyVR){
			OpenVRHelper::instance()->start();
		}

		this->vrEnabled = enable;
	}else{
		if(!Debug::dummyVR){
			OpenVRHelper::instance()->stop();
		}

		this->vrEnabled = enable;
	}
}

// shared_ptr<GLBuffer> createBuffer(uint32_t size){

// 	GLuint handle;
// 	glCreateBuffers(1, &handle);
// 	glNamedBufferStorage(handle, size, 0, GL_DYNAMIC_STORAGE_BIT);

// 	auto buffer = make_shared<GLBuffer>();
// 	buffer->handle = handle;
// 	buffer->size = size;

// 	return buffer;
// }