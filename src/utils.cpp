
#include "utils.h"

EventQueue *EventQueue::instance = new EventQueue();

void schedule(std::function<void()> event) {
	EventQueue::instance->add(event);
}

void monitorFile(string file, std::function<void()> event) {

	std::thread([file, event]() {

		if (!fs::exists(file)) {
			cout << "ERROR(monitorFile): file does not exist: " << file << endl;

			return;
		}

		auto lastWriteTime = fs::last_write_time(fs::path(file));

		while (true) {
			std::this_thread::sleep_for(20ms);

			auto currentWriteTime = fs::last_write_time(fs::path(file));

			if (currentWriteTime > lastWriteTime) {

				schedule(event);

				lastWriteTime = currentWriteTime;
			}

		}

	}).detach();
}

vector<char> loadFile(string file) {

	//double start = now();

	std::ifstream in(file, std::ios::binary);

	//std::streamsize size = in.tellg();
	//in.seekg(0, std::ios::beg);

	// https://stackoverflow.com/questions/18816126/c-read-the-whole-file-in-buffer
	//std::vector<char> buffer(size);
	//if (in.read(buffer.data(), size)){
		/* worked! */
	//}


	std::vector<char> buffer((std::istreambuf_iterator<char>(in)), (std::istreambuf_iterator<char>()));

	//double end = now();
	//double duration = end - start;
	//cout << "loadFile() duration: " << duration << "s" << endl;

	return buffer; 
}


//static long long start_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
//
//double now() {
//	auto now = std::chrono::high_resolution_clock::now();
//	long long nanosSinceStart = now.time_since_epoch().count() - start_time;
//
//	double secondsSinceStart = double(nanosSinceStart) / 1'000'000'000;
//
//	return secondsSinceStart;
//}