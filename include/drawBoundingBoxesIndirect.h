
#pragma once

#include "GL\glew.h"

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "Shader.h"
#include "Renderer.h"

using glm::dvec3;

void _drawBoundingBoxesIndirect(Camera* camera, GLBuffer buffer){

	static bool initialized = false;
	static Shader* shader = nullptr;
	static GLuint vao = 0;

	if(!initialized){
		string vsPath = "./shaders/boundingBox.vs";
		string fsPath = "./shaders/boundingBox.fs";

		shader = new Shader(vsPath, fsPath);

		glGenVertexArrays(1, &vao);

		initialized = true;
	}

	glBindVertexArray(vao);

	glUseProgram(shader->program);

	glm::mat4 view = camera->view;
	glm::mat4 proj = camera->proj;
	glUniformMatrix4fv(0, 1, GL_FALSE, &view[0][0]);
	glUniformMatrix4fv(1, 1, GL_FALSE, &proj[0][0]);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer.handle);

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	// glDrawArrays(GL_LINES, 0, boxes.size() * 24);
	// glDrawArrays(GL_POINTS, 0, boxes.size() * 24);

	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, buffer.handle);

	glDrawArraysIndirect(GL_LINES, 0);
}