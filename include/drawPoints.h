
#pragma once

#include "GL\glew.h"

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "Shader.h"
#include "Renderer.h"

void _drawPoints(Camera* camera, void* points, int numPoints){

	static bool initialized = false;
	static Shader* shader = nullptr;
	static GLuint vao = 0;
	static GLuint ssPoints = 0;
	constexpr int MAX_POINTS = 50'000'000;
	static float tStart = now();

	if(!initialized){
		string vsPath = "./shaders/points.vs";
		string fsPath = "./shaders/points.fs";

		shader = new Shader(vsPath, fsPath);

		glGenVertexArrays(1, &vao);

		glCreateBuffers(1, &ssPoints);
		glNamedBufferData(ssPoints, 32 * MAX_POINTS, nullptr, GL_DYNAMIC_DRAW);

		initialized = true;
	}

	glBindVertexArray(vao);

	glUseProgram(shader->program);

	glm::mat4 view = camera->view;
	glm::mat4 proj = camera->proj;
	dmat4 viewProj = camera->proj * camera->view;
	//glUniformMatrix4fv(0, 1, GL_FALSE, &view[0][0]);
	//glUniformMatrix4fv(1, 1, GL_FALSE, &proj[0][0]);
	glUniformMatrix4dv(0, 1, GL_FALSE, &viewProj[0][0]);

	glNamedBufferSubData(ssPoints, 0, 32 * numPoints, points);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssPoints);

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	glDrawArrays(GL_POINTS, 0, numPoints);
}

void _drawPoints(Camera* camera, GLuint vao, GLuint vbo, int numPoints) {

	static bool initialized = false;
	static Shader* shader = nullptr;

	static float tStart = now();

	if (!initialized) {
		string vsPath = "./shaders/points.vs";
		string fsPath = "./shaders/points.fs";

		shader = new Shader(vsPath, fsPath);

		initialized = true;
	}

	glBindVertexArray(vao);

	glUseProgram(shader->program);

	glm::mat4 view = camera->view;
	glm::mat4 proj = camera->proj;
	glm::dmat4 viewProj = camera->proj * camera->view;
	//glUniformMatrix4fv(0, 1, GL_FALSE, &view[0][0]);
	//glUniformMatrix4fv(1, 1, GL_FALSE, &proj[0][0]);
	glUniformMatrix4dv(0, 1, GL_FALSE, &viewProj[0][0]);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vbo);

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	glDrawArrays(GL_POINTS, 0, numPoints);
	//glDrawArrays(GL_POINTS, 0, 1000);
}