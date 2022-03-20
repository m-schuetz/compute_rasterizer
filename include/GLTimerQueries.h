

#pragma once

#include "GL\glew.h"
//#include "GLFW\glfw3.h"

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <queue>

using std::cout;
using std::endl;
using std::string;
using std::unordered_map;
using std::vector;
using std::deque;
using std::queue;
using std::shared_ptr;


struct GLTimestamp {
	string label;
	GLuint handle = -1;
	int age = 0;
	bool shouldPrint = false;
};

struct GLFrame {

	vector<GLTimestamp> timestamps;

};

struct Timestamp {
	string label;
	uint64_t nanos;
	bool shouldPrint = false;
};

struct Duration {
	string label;
	uint64_t nanos;
};

struct GLTStats {
	string label;
	double min = 10000000000.0;
	double max = 0.0;
	double sum = 0.0;
	double count = 0.0;
};

struct GLTimerQueries {

	bool enabled = true;
	//vector<GLTimestamp> timestamps;
	GLFrame currentFrame;

	// store some frames, evaluate timestamps after some frames
	queue<GLFrame> frames;

	vector<Timestamp> timings;
	vector<Duration> durations;
	unordered_map<string, GLTStats> stats_buildup;
	unordered_map<string, GLTStats> stats;

	static GLTimerQueries* instance() {

		static GLTimerQueries* _instance = new GLTimerQueries();

		return _instance;
	}

	static void frameStart();
	static void frameEnd();
	static void timestamp(string label);
	static void timestampPrint(string label);


};

