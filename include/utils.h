

#pragma once


#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <thread>
#include <mutex>
#include <vector>
#include <filesystem>
#include <chrono>
#include <functional>

using std::string;
using std::ifstream;
using std::stringstream;
using std::thread;
using std::mutex;
using std::vector;
using std::cout;
using std::endl;
using namespace std::chrono_literals;

namespace fs = std::filesystem;

class EventQueue {
public:
	static EventQueue *instance;
	vector<std::function<void()>> queue;
	mutex mtx;

	void add(std::function<void()> event) {
		mtx.lock();
		this->queue.push_back(event);
		mtx.unlock();
	}

	//void clear() {
	//
	//	mtx.lock();
	//	this->queue.clear();
	//	mtx.unlock();
	//}

	void process() {
		//cout << "start processing" << endl;

		mtx.lock();
		vector<std::function<void()>> q = queue;
		queue = vector<std::function<void()>>();
		mtx.unlock();

		for (auto &event : q) {
			//cout << "process" << endl;
			event();

			//cout << "processed" << endl;
		}

		//cout << "finished processing" << endl;

		//clear();
	}
};

void schedule(std::function<void()> event);

static string loadFileAsString(string path) {
	std::ifstream t(path);
	std::stringstream buffer;
	buffer << t.rdbuf();

	return buffer.str();
}

vector<char> loadFile(string file);

double now();


// TODO procide means to cancel monitoring
void monitorFile(string file, std::function<void()> event);