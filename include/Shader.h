
#pragma once

#include <unordered_map>
#include <string>
#include <iostream>


#include "GL\glew.h"
#include "GLFW\glfw3.h"

#include "utils.h"

using std::unordered_map;
using std::string;
using std::cout;
using std::endl;


class Shader {

public:

	string vsPath = "";
	string fsPath = "";
	GLuint program = -1;

	unordered_map<string, unsigned int> uniformLocations;

	Shader(string vsPath, string fsPath) {
		this->vsPath = vsPath;
		this->fsPath = fsPath;

		compile();
	}

	void compile() {
		string vsSource = loadFileAsString(vsPath);
		string fsSource = loadFileAsString(fsPath);

		cout << "compiling vs shader: " << vsPath << endl;
		//cout << vsSource << endl;

		int vs = compileShader(vsSource, GL_VERTEX_SHADER);

		cout << "compiling fs shader: " << fsPath << endl;
		//cout << fsSource << endl;

		int fs = compileShader(fsSource, GL_FRAGMENT_SHADER);

		if (vs == -1 || fs == -1) {
			return;
		}


		if (program == -1) {
			program = glCreateProgram();
		} else {
			glUseProgram(0);
			glDeleteProgram(program);
			program = glCreateProgram();
		}

		glAttachShader(program, vs);
		glAttachShader(program, fs);

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

		glDetachShader(program, vs);
		glDetachShader(program, fs);
		glDeleteShader(vs);
		glDeleteShader(fs);

	}

	GLuint compileShader(string source, GLuint shaderType) {

		int glshader = glCreateShader(shaderType);

		const char * vsc = source.c_str();
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
			//qDebug() << "error in vertex shader " << name.c_str();
			//qDebug() << str.c_str();


			return -1;
		}

		return glshader;
	}

};

