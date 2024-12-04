
#pragma once

#include <functional>
#include <vector>
#include <string>

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"
#include "implot.h"
#include "implot_internal.h"

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

//#include "VrRuntime.h"
#include "unsuck.hpp"
#include "Debug.h"
#include "OrbitControls.h"
#include "Camera.h"
#include "Box.h"
#include "Framebuffer.h"
#include "Texture.h"
#include "GLBuffer.h"

#include "OpenVRHelper.h"

using namespace std;
using glm::dvec3;
using glm::dvec4;
using glm::dmat4;


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

struct DrawQueue{
	vector<Box> boxes;
	vector<Box> boundingBoxes;

	void clear(){
		boxes.clear();
		boundingBoxes.clear();
	}
};


struct View{
	dmat4 view;
	dmat4 proj;
	shared_ptr<Framebuffer> framebuffer = nullptr;
};

struct Renderer{

	GLFWwindow* window = nullptr;
	double fps = 0.0;
	int64_t frameCount = 0;
	
	shared_ptr<Camera> camera = nullptr;
	shared_ptr<OrbitControls> controls = nullptr;

	bool vrEnabled = false;
	
	vector<View> views;

	vector<function<void(vector<string>)>> fileDropListeners;

	int width = 0;
	int height = 0;
	string selectedMethod = "";

	DrawQueue drawQueue;

	Renderer();

	void init();

	shared_ptr<Texture> createTexture(int width, int height, GLuint colorType);

	shared_ptr<Framebuffer> createFramebuffer(int width, int height);

	inline GLBuffer createBuffer(int64_t size){
		GLuint handle;
		glCreateBuffers(1, &handle);
		glNamedBufferStorage(handle, size, 0, GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT);

		GLBuffer buffer;
		buffer.handle = handle;
		buffer.size = size;

		return buffer;
	}

	inline GLBuffer createSparseBuffer(int64_t size){
		GLuint handle;
		glCreateBuffers(1, &handle);
		glNamedBufferStorage(handle, size, 0, GL_DYNAMIC_STORAGE_BIT | GL_SPARSE_STORAGE_BIT_ARB );

		GLBuffer buffer;
		buffer.handle = handle;
		buffer.size = size;

		return buffer;
	}

	inline GLBuffer createUniformBuffer(int64_t size){
		GLuint handle;
		glCreateBuffers(1, &handle);
		glNamedBufferStorage(handle, size, 0, GL_DYNAMIC_STORAGE_BIT );

		GLBuffer buffer;
		buffer.handle = handle;
		buffer.size = size;

		return buffer;
	}

	inline shared_ptr<Buffer> readBuffer(GLBuffer glBuffer, uint32_t offset, uint32_t size){

		auto target = make_shared<Buffer>(size);

		glGetNamedBufferSubData(glBuffer.handle, offset, size, target->data);

		return target;
	}

	//inline int64_t getAvailableGpuMemory(){
	//	GLint available = 0;
	//	glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &available);
	//}

	void loop(function<void(void)> update, function<void(void)> render);

	void drawBox(glm::dvec3 position, glm::dvec3 scale, glm::ivec3 color);

	void drawBoundingBox(glm::dvec3 position, glm::dvec3 scale, glm::ivec3 color);
	void drawBoundingBoxes(Camera* camera, GLBuffer buffer);

	void drawPoints(void* points, int numPoints);

	void drawPoints(GLuint vao, GLuint vbo, int numPoints);

	void toggleVR();

	void setVR(bool enable);

	void onFileDrop(function<void(vector<string>)> callback){
		fileDropListeners.push_back(callback);
	}

};