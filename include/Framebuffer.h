
#pragma once

#include <vector>
#include <memory>

#include "GL\glew.h"
#include "GLFW\glfw3.h"

#include "Texture.h"

using namespace std;

struct Renderer;

struct Framebuffer {

	vector<shared_ptr<Texture>> colorAttachments;
	shared_ptr<Texture> depth;
	GLuint handle = -1;
	Renderer* renderer = nullptr;

	int width = 0;
	int height = 0;

	Framebuffer() {
		
	}

	static shared_ptr<Framebuffer> create(Renderer* renderer);

	void setSize(int width, int height);

};