
#pragma once

#include <string>
#include <unordered_map>
#include <map>

using namespace std;

struct Debug{

	map<string, string> values;

	inline static vector<std::pair<string, string>> frameStats;

	inline static bool updateEnabled = true;
	inline static bool updateFrustum = true;
	inline static bool showBoundingBox = false;
	inline static bool doCopyTree = false;
	inline static bool boolMisc = false;
	inline static float LOD = 1.0;
	inline static bool lodEnabled = false;
	inline static bool frustumCullingEnabled = true;
	inline static bool enableShaderDebugValue = false;
	inline static bool requestCopyVrMatrices = false;
	inline static bool dummyVR = false;
	inline static bool requestResetView = false;
	inline static bool colorizeChunks = false;
	inline static bool colorizeOverdraw = false;



	Debug(){
		
	}

	static Debug* getInstance(){
		static Debug* instance = new Debug();

		return instance;
	}

	static void set(string key, string value){
		Debug::getInstance()->values[key] = value;
	}

	static string get(string key){

		auto& values = Debug::getInstance()->values;

		if(values.find(key) != values.end()){
			return values[key];
		}else{
			return "undefined";
		}
	}

	static void pushFrameStat(string key, string value){
		frameStats.push_back({key, value});
	}

	static void clearFrameStats(){
		frameStats.clear();
	}

};