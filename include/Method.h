
#pragma once

#include <string>

using std::string;

struct Renderer;

struct Method{

	string name = "no name";
	string description = "";
	// int group = 0;
	string group = "no group";

	Method(){

	}

	virtual void update(Renderer* renderer) = 0;
	virtual void render(Renderer* renderer) = 0;

};