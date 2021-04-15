
#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <functional>

using std::unordered_map;
using std::vector;
using std::string;
using std::function;

struct MouseMoveEvent {
	double x = 0.0;
	double y = 0.0;
};

struct MouseScrollEvent {
	double xoffset = 0.0;
	double yoffset = 0.0;
};

struct MouseButtonEvent {
	int button = 0;
	int action = 0;
	int mods = 0;
};

struct KeyEvent {
	int key = 0;
	int scancode = 0;
	int action = 0;
	int mods = 0;
};

struct MouseCursor {
	int width = 0;
	int height = 0;
	vector<unsigned char> data;
	int x = 0;
	int y = 0;
	int pitch = 0;
	int type = 0;
};

struct DesktopTexture {
	unsigned int textureHandle = 0;
	bool hasChanged = false;
};


class Application{
	vector<function<void(MouseMoveEvent)>> mouseMoveListeners;
	vector<function<void(MouseScrollEvent)>> mouseScrollListeners;
	vector<function<void(MouseButtonEvent)>> mouseDownListeners;
	vector<function<void(MouseButtonEvent)>> mouseUpListeners;
	vector<function<void(KeyEvent)>> keyEventListeners;


public:
	bool reportState = true;

	static Application * instance() {
		return _instance;
	}

	void addMouseMoveListener(function<void(MouseMoveEvent)> callback);
	void dispatchMouseMoveEvent(MouseMoveEvent data);

	void addMouseScrollListener(function<void(MouseScrollEvent)> callback);
	void dispatchMouseScrollEvent(MouseScrollEvent data);

	void addMouseDownListener(function<void(MouseButtonEvent)> callback);
	void dispatchMouseDownEvent(MouseButtonEvent data);

	void addMouseUpListener(function<void(MouseButtonEvent)> callback);
	void dispatchMouseUpEvent(MouseButtonEvent data);

	void addKeyEventListener(function<void(KeyEvent)> callback);
	void dispatchKeyEvent(KeyEvent data);

	DesktopTexture acquireDesktopTexture();
	MouseCursor getCursor();
	void lockScreenCapture();
	void unlockScreenCapture();

private: 
	static Application * _instance;

	Application() {

	}
};

