
#include "Application.h"

#include "GL\glew.h"
#include "GL\wglew.h"
#include "GLFW\glfw3.h"

#include <iostream>

using std::cout;
using std::endl;

//app = new ApplicationState();

Application* Application::_instance = new Application();

void Application::addMouseMoveListener(function<void(MouseMoveEvent)> callback){
	mouseMoveListeners.push_back(callback);
}

void Application::dispatchMouseMoveEvent(MouseMoveEvent data) {
	for (auto &listener : mouseMoveListeners) {
		listener(data);
	}
}


void Application::addMouseScrollListener(function<void(MouseScrollEvent)> callback) {
	mouseScrollListeners.push_back(callback);
}

void Application::dispatchMouseScrollEvent(MouseScrollEvent data) {
	for (auto &listener : mouseScrollListeners) {
		listener(data);
	}
}


void Application::addMouseDownListener(function<void(MouseButtonEvent)> callback) {
	mouseDownListeners.push_back(callback);
}

void Application::dispatchMouseDownEvent(MouseButtonEvent data) {
	for (auto &listener : mouseDownListeners) {
		listener(data);
	}
}


void Application::addMouseUpListener(function<void(MouseButtonEvent)> callback) {
	mouseUpListeners.push_back(callback);
}

void Application::dispatchMouseUpEvent(MouseButtonEvent data) {
	for (auto &listener : mouseUpListeners) {
		listener(data);
	}
}


void Application::addKeyEventListener(function<void(KeyEvent)> callback) {
	keyEventListeners.push_back(callback);
}

void Application::dispatchKeyEvent(KeyEvent data) {
	for (auto &listener : keyEventListeners) {
		listener(data);
	}
}



#include <Windows.h>
//#include <dxgi.h>
#include <dxgi1_2.h>
#include <d3d11.h>

// see https://www.codeproject.com/Tips/1116253/Desktop-screen-capture-on-Windows-via-Windows-Desk
vector<D3D_FEATURE_LEVEL> gFeatureLevels = {
	D3D_FEATURE_LEVEL_11_0,
};

MouseCursor cursor;

bool dxInitialized = false;
D3D_FEATURE_LEVEL lFeatureLevel;
ID3D11Device *lDevice = nullptr;
ID3D11DeviceContext *lImmediateContext = nullptr;
IDXGIOutputDuplication *lDeskDupl = nullptr;
DXGI_OUTDUPL_DESC lOutputDuplDesc;
DXGI_OUTPUT_DESC lOutputDesc;
ID3D11Texture2D *lAcquiredDesktopImage = nullptr;
ID3D11Texture2D *d3dTexture = nullptr;
ID3D11Texture2D *d3dTexture2 = nullptr;
vector<BYTE> metaDataBuffer;

HANDLE gl_desktopTextureHandle = nullptr;
GLuint gl_desktopTextureName = -1;
HANDLE gl_handleD3D;
HANDLE textureHandle;

IDXGIFactory2 *dxgi_factory = nullptr;
IDXGIAdapter1 *dxgi_adapter = NULL;
IDXGIOutput *dxgi_output = NULL;
IDXGIOutput1 *dxgi_output1 = NULL;



int texWidth = 1280;
int texHeight = 720;


void checkError(HRESULT hr, string errorMessage) {
	if (hr < 0) {
		cout << errorMessage << endl;
		exit(1);
	}
}

void initDX() {

	UINT creationFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_DEBUG;
	
	//HRESULT hrCreateFactory = CreateDXGIFactory1(__uuidof(IDXGIFactory2), (void **)&dxgi_factory);
	//checkError(hrCreateFactory, "ERROR: CreateDXGIFactory1 failed");
	//
	//HRESULT hrEnumAdapters = dxgi_factory->EnumAdapters1(0, &dxgi_adapter);
	//checkError(hrEnumAdapters, "ERROR: EnumAdapters1 failed");
	//
	//HRESULT hrEnumOutput = dxgi_adapter->EnumOutputs(0, &dxgi_output);
	//checkError(hrEnumOutput, "ERROR: EnumOutputs failed");
	//
	//HRESULT hrIDXGIOutput1 = dxgi_output->QueryInterface(__uuidof(IDXGIOutput1), (void **)&dxgi_output1);
	//checkError(hrIDXGIOutput1, "ERROR: hrIDXGIOutput1 failed");




	//HRESULT hrCreateDevice = D3D11CreateDevice(
	//	dxgi_adapter,
	//	D3D_DRIVER_TYPE_UNKNOWN,
	//	nullptr,
	//	creationFlags,
	//	gFeatureLevels.data(),
	//	gFeatureLevels.size(),
	//	D3D11_SDK_VERSION,
	//	&lDevice,
	//	&lFeatureLevel,
	//	&lImmediateContext);
	HRESULT hrCreateDevice = D3D11CreateDevice(
		dxgi_adapter,
		D3D_DRIVER_TYPE_HARDWARE,
		nullptr,
		creationFlags,
		gFeatureLevels.data(),
		gFeatureLevels.size(),
		D3D11_SDK_VERSION,
		&lDevice,
		&lFeatureLevel,
		&lImmediateContext);
	checkError(hrCreateDevice, "ERROR: D3D11CreateDevice failed");


	{ // DUPLICATION
		
		// necessary???
		//Sleep(100);

		IDXGIDevice *lDxgiDevice;
		HRESULT hrQueryInterface = lDevice->QueryInterface(IID_PPV_ARGS(&lDxgiDevice));
		checkError(hrQueryInterface, "ERROR: hrQueryInterface failed");


		IDXGIAdapter *lDxgiAdapter;
		HRESULT hrlDxgiDeviceGetParent = lDxgiDevice->GetParent(
			__uuidof(IDXGIAdapter),
			reinterpret_cast<void**>(&lDxgiAdapter));
		checkError(hrlDxgiDeviceGetParent, "ERROR: lDxgiDevice->GetParent failed");

		lDxgiDevice->Release();
		lDxgiDevice = nullptr;

		UINT Output = 0;
		IDXGIOutput *lDxgiOutput;
		HRESULT hrEnumOutputs = lDxgiAdapter->EnumOutputs(
			Output,
			&lDxgiOutput);
		checkError(hrEnumOutputs, "ERROR: EnumOutputs failed");

		lDxgiAdapter->Release();
		lDxgiAdapter = nullptr;


		HRESULT hrlDxgiOutputGetDesc = lDxgiOutput->GetDesc(&lOutputDesc);
		checkError(hrlDxgiOutputGetDesc, "ERROR: lDxgiOutput->GetDesc failed");

		IDXGIOutput1 *lDxgiOutput1;
		HRESULT hrlDxgiOutputQueryInterface = lDxgiOutput->QueryInterface(IID_PPV_ARGS(&lDxgiOutput1));
		checkError(hrlDxgiOutputQueryInterface, "ERROR: lDxgiOutput->QueryInterface failed");

		lDxgiOutput->Release();
		lDxgiOutput = nullptr;



		HRESULT hrDuplicateOutput = lDxgiOutput1->DuplicateOutput(lDevice, &lDeskDupl);
		checkError(hrDuplicateOutput, "ERROR: DuplicateOutput failed");

		lDxgiOutput1->Release();
		lDxgiOutput1 = nullptr;

	}


	int bpp = 4;

	{ // DX11 TEXTURE
		D3D11_TEXTURE2D_DESC desc;
		D3D11_SUBRESOURCE_DATA tbsd;


		desc.Width = texWidth;
		desc.Height = texHeight;
		desc.MipLevels = 1;
		desc.ArraySize = 1;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
		desc.CPUAccessFlags = 0;
		desc.MiscFlags = 0;

		vector<unsigned char> buffer(texWidth * texHeight* bpp, 100);

		for (int i = 0; i < texHeight; i++) {
			for (int j = 0; j < texWidth; j++) {
				int offset = bpp * (j + i * texWidth);

				buffer[offset + 0] = 255.0 * double((i) / double(texHeight));
				buffer[offset + 1] = 255.0 * double((j) / double(texWidth));
				buffer[offset + 2] = 50;
				buffer[offset + 3] = 255.0;
			}
		}

		tbsd.pSysMem = buffer.data();
		tbsd.SysMemPitch = texWidth * bpp;
		tbsd.SysMemSlicePitch = texWidth * texHeight * bpp;

		HRESULT hrCreateTexture = lDevice->CreateTexture2D(&desc, &tbsd, &d3dTexture);
		checkError(hrCreateTexture, "ERROR: CreateTexture2D failed");

		int a = 10;
	}

	{ // DX11 TEXTURE 2
		D3D11_TEXTURE2D_DESC desc;
		D3D11_SUBRESOURCE_DATA tbsd;


		desc.Width = texWidth;
		desc.Height = texHeight;
		desc.MipLevels = 1;
		desc.ArraySize = 1;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
		desc.CPUAccessFlags = 0;
		desc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

		vector<unsigned char> buffer(texWidth * texHeight* bpp, 100);

		for (int i = 0; i < texHeight; i++) {
			for (int j = 0; j < texWidth; j++) {
				int offset = bpp * (j + i * texWidth);

				buffer[offset + 0] = 255;
				buffer[offset + 1] = 255;
				buffer[offset + 2] = 100;
				buffer[offset + 3] = 255;
			}
		}

		tbsd.pSysMem = buffer.data();
		tbsd.SysMemPitch = texWidth * bpp;
		tbsd.SysMemSlicePitch = texWidth * texHeight * bpp;

		HRESULT hrCreateTexture = lDevice->CreateTexture2D(&desc, &tbsd, &d3dTexture2);
		checkError(hrCreateTexture, "ERROR: CreateTexture2D failed");

		int a = 10;
	}

	//{
	//	D3D11_BOX box;
	//	box.left = 0;
	//	box.top = 0;
	//	box.right = texWidth;
	//	box.bottom = texHeight;
	//	box.front = 0;
	//	box.back = 1;
	//
	//	lImmediateContext->CopySubresourceRegion(d3dTexture, 0, 0, 0, 0, d3dTexture2, 0, &box);
	//	//lImmediateContext->CopyResource(d3dTexture, d3dTexture2);
	//}

	
	IDXGIResource* pDXGIResource = NULL;
	HRESULT hrQuery = d3dTexture->QueryInterface(__uuidof(IDXGIResource), (LPVOID*) &pDXGIResource);
	checkError(hrQuery, "ERROR: d3dTexture->QueryInterface failed");

	HANDLE sharedHandle;
	HRESULT hrGetSharedHandle = pDXGIResource->GetSharedHandle(&sharedHandle);
	checkError(hrGetSharedHandle, "ERROR: pDXGIResource->GetSharedHandle failed");
	
	wglDXSetResourceShareHandleNV(d3dTexture, sharedHandle);




	gl_handleD3D = wglDXOpenDeviceNV(lDevice);

	glCreateTextures(GL_TEXTURE_2D, 1, &gl_desktopTextureName);

	textureHandle = wglDXRegisterObjectNV(
		gl_handleD3D, 
		d3dTexture, 
		gl_desktopTextureName, 
		GL_TEXTURE_2D, WGL_ACCESS_READ_WRITE_NV);

	dxInitialized = true;
}

DesktopTexture Application::acquireDesktopTexture() {

	if (!dxInitialized) {
		initDX();
	}

	DesktopTexture result;
	result.textureHandle = gl_desktopTextureName;
	result.hasChanged = false;

	IDXGIResource *lDesktopResource = nullptr;
	DXGI_OUTDUPL_FRAME_INFO lFrameInfo;

	HRESULT hrAcquire = lDeskDupl->AcquireNextFrame(0, &lFrameInfo, &lDesktopResource);
	if (hrAcquire == DXGI_ERROR_WAIT_TIMEOUT) {
		//cout << "wait timeout" << endl;

		lDeskDupl->ReleaseFrame();

		return result;
	} else if (hrAcquire < 0) {
		cout << "ERROR: AcquireNextFrame failed" << endl;
		exit(1);
	} else {
		result.hasChanged = true;
	}

	if (lAcquiredDesktopImage) {
		lAcquiredDesktopImage->Release();
		lAcquiredDesktopImage = nullptr;
	}

	

	
	HRESULT hrQueryI = lDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&lAcquiredDesktopImage));
	checkError(hrQueryI, "ERROR: lDesktopResource->QueryInterface failed");

	D3D11_TEXTURE2D_DESC textureDescriptor;
	lAcquiredDesktopImage->GetDesc(&textureDescriptor);


	if (lFrameInfo.TotalMetadataBufferSize > 0) {

		if (lFrameInfo.TotalMetadataBufferSize > metaDataBuffer.size()) {
			metaDataBuffer.resize(lFrameInfo.TotalMetadataBufferSize);
		}

		UINT bufferSize = lFrameInfo.TotalMetadataBufferSize;

		HRESULT hrGetMoveRects = -lDeskDupl->GetFrameMoveRects(bufferSize, reinterpret_cast<DXGI_OUTDUPL_MOVE_RECT*>(metaDataBuffer.data()), &bufferSize);
		checkError(hrGetMoveRects, "ERROR: DeskDupl->GetFrameMoveRects failed");

		UINT moveCount = bufferSize / sizeof(DXGI_OUTDUPL_MOVE_RECT);
		BYTE* DirtyRects = metaDataBuffer.data() + bufferSize;
		bufferSize = lFrameInfo.TotalMetadataBufferSize - bufferSize;

		HRESULT hrDirtyRects = lDeskDupl->GetFrameDirtyRects(bufferSize, reinterpret_cast<RECT*>(DirtyRects), &bufferSize);
		checkError(hrDirtyRects, "ERROR: GetFrameDirtyRects failed");
	
		UINT DirtyCount = bufferSize / sizeof(RECT);
	}

	unlockScreenCapture();

	if(true){
		D3D11_BOX box;
		box.left = 0;
		box.top = 0;
		box.right = texWidth;
		box.bottom = texHeight;
		box.front = 0;
		box.back = 1;


		lImmediateContext->CopySubresourceRegion(d3dTexture, 0, 0, 0, 0, lAcquiredDesktopImage, 0, &box);

		static int toggle = 0;
		toggle++;
	}

	{

		// update position
		if (lFrameInfo.PointerPosition.Visible) {
			auto pointerPos = lFrameInfo.PointerPosition;

			cursor.x = pointerPos.Position.x;
			cursor.y = pointerPos.Position.y;
		}

		bool shouldUpdateCursorData =
			lFrameInfo.LastMouseUpdateTime.QuadPart &&
			lFrameInfo.PointerPosition.Visible &&
			lFrameInfo.PointerShapeBufferSize > 0;

		if (shouldUpdateCursorData) {

			auto pointerBufferSize = lFrameInfo.PointerShapeBufferSize;
			BYTE* pointerData = new BYTE[pointerBufferSize];
			DXGI_OUTDUPL_POINTER_SHAPE_INFO pointerInfo = {};

			HRESULT hrPShape = lDeskDupl->GetFramePointerShape(pointerBufferSize, pointerData, &pointerBufferSize, &pointerInfo);
			if (hrPShape == DXGI_ERROR_MORE_DATA) {
				delete[] pointerData;

				pointerData = new BYTE[pointerBufferSize];
				hrPShape = lDeskDupl->GetFramePointerShape(pointerBufferSize, pointerData, &pointerBufferSize, &pointerInfo);
			}

			//if (pointerInfo.Width == 32 && pointerInfo.Height == 64) {
			//	int breakpointHere = 10;
			//
			//	cout << "=====" << endl;
			//
			//	for (int i = 0; i < 256; i++) {
			//		int val = pointerData[i];
			//		cout << val << ", ";
			//	}
			//
			//
			//	cout << "=====" << endl;
			//}

			if (hrPShape == S_OK) {
				cursor.data = vector<unsigned char>(pointerData, pointerData + pointerBufferSize);

				cursor.width = pointerInfo.Width;
				cursor.height = pointerInfo.Height;
				cursor.pitch = pointerInfo.Pitch;
				cursor.type = pointerInfo.Type;
				
				if (pointerInfo.Type == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME) {

					cursor.height = pointerInfo.Height / 2;
					
					// https://docs.microsoft.com/en-us/windows-hardware/drivers/display/drawing-monochrome-pointers

					int numPixels = cursor.width * cursor.height;
					vector<BYTE> singleBitData = cursor.data;
					vector<BYTE> rgbaData = vector<BYTE>(4 * numPixels);
					
					for (int row = 0; row < cursor.height; row++) {
						for (int col = 0; col < cursor.width; col++) {
					
							int pixelIndex = row * cursor.width + col;
							int andBitIndex = pixelIndex;
							int andByteIndex = andBitIndex / 8;
					
							int xorBitIndex = pixelIndex + numPixels;
							int xorByteIndex = xorBitIndex / 8;
					
							int localBitIndex = 7 - andBitIndex % 8;

							if (row == 2 && col == 0) {
								int breakpoint = 0;
							}
					
							int andBit = (singleBitData[andByteIndex] >> localBitIndex) & 1;
							int xorBit = (singleBitData[xorByteIndex] >> localBitIndex) & 1;
					
							int r = 0;
							int g = 0;
							int b = 0;
							int a = 255;

							if (andBit == 0 && xorBit == 0) {
								// black
							} else if (andBit == 0 && xorBit == 1) {
								// white
								r = 255;
								g = 255;
								b = 255;
							} else if (andBit == 1 && xorBit == 0) {
								// transparent
								a = 0;
							} else {
								// inverted (not supported, treat as black)
								r = 255;
								g = 255;
								b = 255;
								a = 255;
							}

							rgbaData[4 * pixelIndex + 0] = a;
							rgbaData[4 * pixelIndex + 1] = r;
							rgbaData[4 * pixelIndex + 2] = g;
							rgbaData[4 * pixelIndex + 3] = b;
					
						}
					}

					cursor.data = rgbaData;

				} else if(pointerInfo.Type == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR){

				} else if (pointerInfo.Type == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR) {

				}

				
			}

			delete [] pointerData;
		}


	}

	HRESULT hrReleaseFrame = lDeskDupl->ReleaseFrame();
	checkError(hrReleaseFrame, "ERROR: failed to release frame");

	lDesktopResource->Release();
	lDesktopResource = nullptr;

	lAcquiredDesktopImage->Release();
	lAcquiredDesktopImage = nullptr;

	lockScreenCapture();

	return result;
}

MouseCursor Application::getCursor() {
	return cursor;
}

void Application::lockScreenCapture() {
	if (gl_handleD3D) {
		wglDXLockObjectsNV(gl_handleD3D, 1, &textureHandle);
	}
}

void Application::unlockScreenCapture() {
	if (gl_handleD3D) {
		wglDXUnlockObjectsNV(gl_handleD3D, 1, &textureHandle);
	}
}
