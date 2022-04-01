
#pragma once

#include <unordered_map>
#include <string>
#include <iostream>


#include "GL\glew.h"
#include "GLFW\glfw3.h"

#include "unsuck.hpp"

using std::unordered_map;
using std::string;
using std::cout;
using std::endl;

struct ShaderComponent {
	string path = "";
	GLuint shaderType = -1;
	string source = "";
	GLuint shader;

	ShaderComponent(string path, GLuint shaderType) {
		this->path = path;
		this->shaderType = shaderType;
	}
};


struct Shader {

	vector<ShaderComponent> components;

	GLuint program = -1;

	unordered_map<string, unsigned int> uniformLocations;

	Shader(string vsPath, string fsPath) {

		setComponents({
			{vsPath, GL_VERTEX_SHADER},
			{fsPath, GL_FRAGMENT_SHADER}
		});

		//compile();
	}

	Shader(vector<ShaderComponent> components) {
		setComponents(components);

		/*compile();*/
	}

	void setComponents(vector<ShaderComponent> components) {
		this->components = components;

		compile();

		for (auto& component : components) {
			monitorFile(component.path, [&]() {
				compile();
			});
		}

	}

	void compile() {

		for (auto& component : components) {
			cout << "compiling component " << component.path << endl;

			component.source = readTextFile(component.path);
			bool compiled = compileShader(component);	

			if (!compiled) {
				return;
			}
		}


		if (program == -1) {
			program = glCreateProgram();
		} else {
			glUseProgram(0);
			glDeleteProgram(program);
			program = glCreateProgram();
		}

		for (auto& component : components) {
			glAttachShader(program, component.shader);
		}

		glLinkProgram(program);

		GLint isLinked = 0;
		glGetProgramiv(program, GL_LINK_STATUS, &isLinked);
		if (isLinked == GL_FALSE) {
			GLint maxLength = 0;
			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

			//The maxLength includes the NULL character
			std::vector<GLchar> infoLog(maxLength);
			glGetProgramInfoLog(program, maxLength, &maxLength, &infoLog[0]);

			cout << "error while linking vs/fs shader " << endl;
			cout << string(infoLog.begin(), infoLog.end()).c_str() << endl;

			//The program is useless now. So delete it.
			glDeleteProgram(program);

			//Provide the infolog in whatever manner you deem best.
			//Exit with failure.
			return;
		}

		glUseProgram(program);

		{ // QUERY UNIFORMS
		  // see http://stackoverflow.com/questions/440144/in-opengl-is-there-a-way-to-get-a-list-of-all-uniforms-attribs-used-by-a-shade

			uniformLocations.clear();

			GLint count;
			glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &count);

			GLint size;
			GLenum type;

			const GLsizei bufSize = 64;
			GLchar buffer[bufSize];
			GLsizei length;

			for (int i = 0; i < count; i++) {
				glGetActiveUniform(program, (GLuint)i, bufSize, &length, &size, &type, buffer);
				string name = string(buffer, length);
				GLuint id = glGetUniformLocation(program, name.c_str());

				uniformLocations[name] = id;
			}
		}

		for (auto& component : components) {
			glDetachShader(program, component.shader);
			glDeleteShader(component.shader);
		}

	}

	bool compileShader(ShaderComponent& component) {

		int glshader = glCreateShader(component.shaderType);

		const char * vsc = component.source.c_str();
		glShaderSource(glshader, 1, &vsc, NULL);
		glCompileShader(glshader);
		auto abc = glGetError();

		GLint isCompiled = 0;
		glGetShaderiv(glshader, GL_COMPILE_STATUS, &isCompiled);
		if (isCompiled == GL_FALSE) {
			GLint maxLength = 0;
			glGetShaderiv(glshader, GL_INFO_LOG_LENGTH, &maxLength);

			std::vector<GLchar> infoLog(maxLength);
			glGetShaderInfoLog(glshader, maxLength, &maxLength, &infoLog[0]);

			glDeleteShader(glshader);

			std::string str(infoLog.begin(), infoLog.end());
			cout << "error in shader" << endl;
			cout << str << endl;

			return false;
		}

		component.shader = glshader;

		return true;
	}

};

