
#include "GLTimerQueries.h"

#include "unsuck.hpp"

void GLTimerQueries::timestamp(string label) {
	GLTimerQueries* glt = GLTimerQueries::instance();

	if (!glt->enabled) {
		return;
	}

	GLuint handle;
	glGenQueries(1, &handle);
	glQueryCounter(handle, GL_TIMESTAMP);

	GLTimestamp timestamp;
	timestamp.label = label;
	timestamp.handle = handle;

	glt->currentFrame.timestamps.push_back(timestamp);
}

void GLTimerQueries::frameStart() {
	GLTimerQueries* glt = GLTimerQueries::instance();

	if (!glt->enabled) {
		return;
	}

	GLTimerQueries::instance()->currentFrame = GLFrame();

	GLTimerQueries::timestamp("frame-start");
}

void GLTimerQueries::frameEnd() {
	GLTimerQueries* glt = GLTimerQueries::instance();

	if (!glt->enabled) {
		return;
	}

	GLTimerQueries::timestamp("frame-end");

	glt->frames.push(glt->currentFrame);
	glt->currentFrame = GLFrame();

	if (glt->frames.size() > 3) {
		auto frame = glt->frames.front();
		glt->frames.pop();

		auto& stats = glt->stats_buildup;

		vector<Timestamp> timings;

		for (auto& timestamp : frame.timestamps) {
			uint64_t result = 123;
			glGetQueryObjectui64v(timestamp.handle, GL_QUERY_RESULT_AVAILABLE, &result);
			bool timestampAvailable = result == GL_TRUE;

			if (timestampAvailable) {
				uint64_t nanos = 123;
				glGetQueryObjectui64v(timestamp.handle, GL_QUERY_RESULT, &nanos);

				Timestamp item;
				item.label = timestamp.label;
				item.nanos = nanos;
				timings.push_back(item);

			} else {
				cout << "could not resolve timestamp " << endl;
			}

			glDeleteQueries(1, &timestamp.handle);
		}

		auto startTime = timings[0].nanos;
		unordered_map<string, Timestamp> starts;
		vector<Duration> durations;

		for (auto& timestamp : timings) {
			timestamp.nanos = timestamp.nanos - startTime;

			if (endsWith(timestamp.label, "-start")) {
				starts[timestamp.label] = timestamp;
				//
			} else if (endsWith(timestamp.label, "-end")) {
				string baseLabel = timestamp.label.substr(0, timestamp.label.size() - 4);
				string startLabel = baseLabel + "-start";
				auto start = starts[startLabel];

				auto duration = timestamp.nanos - start.nanos;

				double millies = double(duration) / 1'000'000;

				if (stats.find(baseLabel) == stats.end()) {
					stats[baseLabel] = GLTStats();
				}

				auto& stat = stats[baseLabel];

				stat.min = std::min(stat.min, millies);
				stat.max = std::max(stat.max, millies);
				stat.sum += millies;
				stat.count++;

				Duration item;
				item.label = baseLabel;
				item.nanos = duration;

				durations.push_back(item);
			}
		}


		// Update once per <window>. TODO: compute mean, update that once per second.
		double window = 1.0;
		static double toggle = now();
		if (now() - toggle > window) {
			glt->timings = timings;
			glt->durations = durations;
			toggle = now();

			glt->stats = glt->stats_buildup;
			glt->stats_buildup = unordered_map<string, GLTStats>();
		}

	}
}