#pragma once

#include "GL\glew.h"
#include "GLFW\glfw3.h"

#include <memory>

using namespace std;

struct Renderer;

struct Texture {

	Renderer* renderer = nullptr;
	GLuint handle = -1;
	GLuint colorType = -1;
	int width = 0;
	int height = 0;

	static shared_ptr<Texture> create(int width, int height, GLuint colorType, Renderer* renderer);

	void setSize(int width, int height);

};