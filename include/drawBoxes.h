
#pragma once

#include "GL\glew.h"

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "Shader.h"
#include "Renderer.h"

using glm::dvec3;

void _drawBoxes(Camera* camera, vector<Box>& boxes){

	static bool initialized = false;
	static Shader* shader = nullptr;
	static GLuint vao = 0;
	static GLuint ssBoxes = 0;
	constexpr int MAX_BOXES = 100'000;
	static Buffer buffer(MAX_BOXES * 48);

	if(!initialized){
		string vsPath = "./shaders/box.vs";
		string fsPath = "./shaders/box.fs";

		shader = new Shader(vsPath, fsPath);

		GLuint tmpBoxes = 0;
		glGenVertexArrays(1, &vao);

		glCreateBuffers(1, &ssBoxes);
		glNamedBufferData(ssBoxes, 48 * MAX_BOXES, nullptr, GL_DYNAMIC_DRAW);

		initialized = true;
	}

	glBindVertexArray(vao);

	glUseProgram(shader->program);

	glm::mat4 view = camera->view;
	glm::mat4 proj = camera->proj;
	glUniformMatrix4fv(0, 1, GL_FALSE, &view[0][0]);
	glUniformMatrix4fv(1, 1, GL_FALSE, &proj[0][0]);

	int stride = 48;
	for(int i = 0; i < boxes.size(); i++){
		auto box = boxes[i];

		dvec3 position = box.center();
		dvec3 size = box.size();

		uint32_t r = box.color.r;
		uint32_t g = box.color.g;
		uint32_t b = box.color.b;
		uint32_t color = (r << 0) | (g << 8) | (b << 16);

		buffer.set<float>(position.x, stride * i + 0);
		buffer.set<float>(position.y, stride * i + 4);
		buffer.set<float>(position.z, stride * i + 8);
		buffer.set<float>(size.x, stride * i + 16);
		buffer.set<float>(size.y, stride * i + 20);
		buffer.set<float>(size.z, stride * i + 24);
		buffer.set<uint32_t>(color, stride * i + 32);
	}

	glNamedBufferSubData(ssBoxes, 0, stride * boxes.size(), buffer.data);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssBoxes);

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	glDrawArrays(GL_TRIANGLES, 0, boxes.size() * 36);
	// glDrawArrays(GL_LINES, 0, boxes.size() * 24);
	//glDrawArrays(GL_POINTS, 0, boxes.size() * 24);
}