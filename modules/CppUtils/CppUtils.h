
#pragma once

#include <chrono>

class Utils {
public:

	static double now() {

		static long long tFirst = std::chrono::high_resolution_clock::now().time_since_epoch().count();

		auto now = std::chrono::high_resolution_clock::now();
		long long nanosSinceStart = now.time_since_epoch().count() - tFirst;

		double secondsSinceStart = double(nanosSinceStart) / 1'000'000'000;

		return secondsSinceStart;
	}

};

//class punct_facet : public std::numpunct<char> {
//protected:
//	char do_decimal_point() const { return '.'; };
//	char do_thousands_sep() const { return '\''; };
//	string do_grouping() const { return "\3"; }
//};
//
//template<class T>
//inline string formatNumber(T number, int decimals = 0) {
//	stringstream ss;
//
//	ss.imbue(std::locale(std::cout.getloc(), new punct_facet));
//	ss << std::fixed << std::setprecision(decimals);
//	ss << number;
//
//	return ss.str();
//}