
#pragma once

#include <unordered_map>
#include <string>
#include <iostream>
#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "unsuck.hpp"
#include "nvrtc.h"
#include <cmath>
#include "cuda.h"

struct CudaProgram {

	bool compiled = false;
	std::string csPath;
	nvrtcProgram prog;
	CUmodule mod;
	CUfunction kernel = nullptr;

	CudaProgram(std::string path) {
		csPath = path;
		compile();
			monitorFile(csPath, [&]() {
				compile();
			});
	}

	void compile() {
		nvrtcProgram newProg;
		std::string source = "#include \"" + csPath + "\"";
		nvrtcCreateProgram(&newProg, source.c_str(), "source.cu", 0, NULL, NULL);
		std::vector<const char*> opts = { "--gpu-architecture=compute_60",
			"--use_fast_math",
			"--extra-device-vectorization",
			"-lineinfo",
			"-I ./modules/compute_loop_las_cuda" };
		nvrtcResult res = nvrtcCompileProgram(newProg, opts.size(), opts.data());
		if (res != NVRTC_SUCCESS)
		{
			size_t logSize;
			nvrtcGetProgramLogSize(newProg, &logSize);
			char* log = new char[logSize];
			nvrtcGetProgramLog(newProg, log);
			std::cerr << log << std::endl;
			delete[] log;
			return;
		}

		if (compiled) {
			cuModuleUnload(mod);
			nvrtcDestroyProgram(&prog);
		}

		prog = newProg;

		size_t ptxSize;
		nvrtcGetPTXSize(prog, &ptxSize);
		char* ptx = new char[ptxSize];
		nvrtcGetPTX(prog, ptx);

		cuModuleLoadDataEx(&mod, ptx, 0, 0, 0);
		cuModuleGetFunction(&kernel, mod, "kernel");
		
		delete[] ptx;
		
		compiled = true;
	}
};

