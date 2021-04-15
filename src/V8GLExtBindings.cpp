
#include "V8Helper.h"

#include <iostream>
#include <vector>

#include "GL\glew.h"
#include "GLFW\glfw3.h"

#include "Shader.h"
#include "ComputeShader.h"
#include "V8Shader.h"
#include "V8ComputeShader.h"


#define CREATE_CONSTANT_ACCESSOR( name, value) \
	tpl->SetAccessor(String::NewFromUtf8(isolate, name), [](Local<String> property, const PropertyCallbackInfo<Value>& info) { \
		info.GetReturnValue().Set(value); \
	})

//void* getArgPointerVoid(const Local<Value> &arg) {
//	void* pointer = nullptr;
//	if (arg->IsArrayBuffer()) {
//		v8::Local<v8::ArrayBuffer> buffer = (arg).As<v8::ArrayBuffer>();
//		void *bdata = buffer->GetContents().Data();
//		pointer = reinterpret_cast<void*>(bdata);
//	} else if (arg->IsArrayBufferView()) {
//		v8::Local<v8::ArrayBufferView> view = (arg).As<v8::ArrayBufferView>();
//		auto buffer = view->Buffer();
//		void *bdata = view->Buffer()->GetContents().Data();
//		pointer = reinterpret_cast<void*>(bdata);
//	} else {
//		cout << "ERROR(glTexImage3D): array must be of type ArrayBuffer" << endl;
//		exit(1);
//	}
//
//	return pointer;
//}


void V8Helper::setupV8GLExtBindings(Local<ObjectTemplate>& tpl){
	
	/* ------------------------------ GL_VERSION_1_2 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("UNSIGNED_BYTE_3_3_2", GL_UNSIGNED_BYTE_3_3_2);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_SHORT_4_4_4_4", GL_UNSIGNED_SHORT_4_4_4_4);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_SHORT_5_5_5_1", GL_UNSIGNED_SHORT_5_5_5_1);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_8_8_8_8", GL_UNSIGNED_INT_8_8_8_8);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_10_10_10_2", GL_UNSIGNED_INT_10_10_10_2);
	CREATE_CONSTANT_ACCESSOR("RESCALE_NORMAL", GL_RESCALE_NORMAL);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_BYTE_2_3_3_REV", GL_UNSIGNED_BYTE_2_3_3_REV);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_SHORT_5_6_5", GL_UNSIGNED_SHORT_5_6_5);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_SHORT_5_6_5_REV", GL_UNSIGNED_SHORT_5_6_5_REV);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_SHORT_4_4_4_4_REV", GL_UNSIGNED_SHORT_4_4_4_4_REV);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_SHORT_1_5_5_5_REV", GL_UNSIGNED_SHORT_1_5_5_5_REV);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_8_8_8_8_REV", GL_UNSIGNED_INT_8_8_8_8_REV);
	CREATE_CONSTANT_ACCESSOR("BGR", GL_BGR);
	CREATE_CONSTANT_ACCESSOR("BGRA", GL_BGRA);
	CREATE_CONSTANT_ACCESSOR("MAX_ELEMENTS_VERTICES", GL_MAX_ELEMENTS_VERTICES);
	CREATE_CONSTANT_ACCESSOR("MAX_ELEMENTS_INDICES", GL_MAX_ELEMENTS_INDICES);
	CREATE_CONSTANT_ACCESSOR("CLAMP_TO_EDGE", GL_CLAMP_TO_EDGE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_MIN_LOD", GL_TEXTURE_MIN_LOD);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_MAX_LOD", GL_TEXTURE_MAX_LOD);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BASE_LEVEL", GL_TEXTURE_BASE_LEVEL);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_MAX_LEVEL", GL_TEXTURE_MAX_LEVEL);
	CREATE_CONSTANT_ACCESSOR("LIGHT_MODEL_COLOR_CONTROL", GL_LIGHT_MODEL_COLOR_CONTROL);
	CREATE_CONSTANT_ACCESSOR("SINGLE_COLOR", GL_SINGLE_COLOR);
	CREATE_CONSTANT_ACCESSOR("SEPARATE_SPECULAR_COLOR", GL_SEPARATE_SPECULAR_COLOR);
	CREATE_CONSTANT_ACCESSOR("SMOOTH_POINT_SIZE_RANGE", GL_SMOOTH_POINT_SIZE_RANGE);
	CREATE_CONSTANT_ACCESSOR("SMOOTH_POINT_SIZE_GRANULARITY", GL_SMOOTH_POINT_SIZE_GRANULARITY);
	CREATE_CONSTANT_ACCESSOR("SMOOTH_LINE_WIDTH_RANGE", GL_SMOOTH_LINE_WIDTH_RANGE);
	CREATE_CONSTANT_ACCESSOR("SMOOTH_LINE_WIDTH_GRANULARITY", GL_SMOOTH_LINE_WIDTH_GRANULARITY);
	CREATE_CONSTANT_ACCESSOR("ALIASED_POINT_SIZE_RANGE", GL_ALIASED_POINT_SIZE_RANGE);
	CREATE_CONSTANT_ACCESSOR("ALIASED_LINE_WIDTH_RANGE", GL_ALIASED_LINE_WIDTH_RANGE);
	CREATE_CONSTANT_ACCESSOR("PACK_SKIP_IMAGES", GL_PACK_SKIP_IMAGES);
	CREATE_CONSTANT_ACCESSOR("PACK_IMAGE_HEIGHT", GL_PACK_IMAGE_HEIGHT);
	CREATE_CONSTANT_ACCESSOR("UNPACK_SKIP_IMAGES", GL_UNPACK_SKIP_IMAGES);
	CREATE_CONSTANT_ACCESSOR("UNPACK_IMAGE_HEIGHT", GL_UNPACK_IMAGE_HEIGHT);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_3D", GL_TEXTURE_3D);
	CREATE_CONSTANT_ACCESSOR("PROXY_TEXTURE_3D", GL_PROXY_TEXTURE_3D);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_DEPTH", GL_TEXTURE_DEPTH);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_WRAP_R", GL_TEXTURE_WRAP_R);
	CREATE_CONSTANT_ACCESSOR("MAX_3D_TEXTURE_SIZE", GL_MAX_3D_TEXTURE_SIZE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BINDING_3D", GL_TEXTURE_BINDING_3D);
	CREATE_CONSTANT_ACCESSOR("MAX_ELEMENTS_VERTICES", GL_MAX_ELEMENTS_VERTICES);
	CREATE_CONSTANT_ACCESSOR("MAX_ELEMENTS_INDICES", GL_MAX_ELEMENTS_INDICES);

	tpl->Set(String::NewFromUtf8(isolate, "drawRangeElements"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("drawRangeElements requires 6 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLuint start = args[1]->Uint32Value();
		GLuint end = args[2]->Uint32Value();
		GLsizei count = args[3]->Int32Value();
		GLenum type = args[4]->Uint32Value();

		void* indices = nullptr;
		if (args[5]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[5]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else if (args[5]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[5]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glDrawRangeElements): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glDrawRangeElements(mode, start, end, count, type, indices);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texImage3D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 10) {
			V8Helper::_instance->throwException("texImage3D requires 10 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint internalFormat = args[2]->Int32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();
		GLsizei depth = args[5]->Int32Value();
		GLint border = args[6]->Int32Value();
		GLenum format = args[7]->Uint32Value();
		GLenum type = args[8]->Uint32Value();

		void* pixels = nullptr;
		if (args[9]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[9]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else if (args[9]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[9]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glTexImage3D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glTexImage3D(target, level, internalFormat, width, height, depth, border, format, type, pixels);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texSubImage3D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 11) {
			V8Helper::_instance->throwException("texSubImage3D requires 11 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLint zoffset = args[4]->Int32Value();
		GLsizei width = args[5]->Int32Value();
		GLsizei height = args[6]->Int32Value();
		GLsizei depth = args[7]->Int32Value();
		GLenum format = args[8]->Uint32Value();
		GLenum type = args[9]->Uint32Value();

		void* pixels = nullptr;
		if (args[10]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[10]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else if (args[10]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[10]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glTexSubImage3D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "copyTexSubImage3D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 9) {
			V8Helper::_instance->throwException("copyTexSubImage3D requires 9 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLint zoffset = args[4]->Int32Value();
		GLint x = args[5]->Int32Value();
		GLint y = args[6]->Int32Value();
		GLsizei width = args[7]->Int32Value();
		GLsizei height = args[8]->Int32Value();

		glCopyTexSubImage3D(target, level, xoffset, yoffset, zoffset, x, y, width, height);
	}));



	// empty / skipped / ignored: GL_VERSION_1_2_1
	/* ------------------------------ GL_VERSION_1_3 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE0", GL_TEXTURE0);
	CREATE_CONSTANT_ACCESSOR("TEXTURE1", GL_TEXTURE1);
	CREATE_CONSTANT_ACCESSOR("TEXTURE2", GL_TEXTURE2);
	CREATE_CONSTANT_ACCESSOR("TEXTURE3", GL_TEXTURE3);
	CREATE_CONSTANT_ACCESSOR("TEXTURE4", GL_TEXTURE4);
	CREATE_CONSTANT_ACCESSOR("TEXTURE5", GL_TEXTURE5);
	CREATE_CONSTANT_ACCESSOR("TEXTURE6", GL_TEXTURE6);
	CREATE_CONSTANT_ACCESSOR("TEXTURE7", GL_TEXTURE7);
	CREATE_CONSTANT_ACCESSOR("TEXTURE8", GL_TEXTURE8);
	CREATE_CONSTANT_ACCESSOR("TEXTURE9", GL_TEXTURE9);
	CREATE_CONSTANT_ACCESSOR("TEXTURE10", GL_TEXTURE10);
	CREATE_CONSTANT_ACCESSOR("TEXTURE11", GL_TEXTURE11);
	CREATE_CONSTANT_ACCESSOR("TEXTURE12", GL_TEXTURE12);
	CREATE_CONSTANT_ACCESSOR("TEXTURE13", GL_TEXTURE13);
	CREATE_CONSTANT_ACCESSOR("TEXTURE14", GL_TEXTURE14);
	CREATE_CONSTANT_ACCESSOR("TEXTURE15", GL_TEXTURE15);
	CREATE_CONSTANT_ACCESSOR("TEXTURE16", GL_TEXTURE16);
	CREATE_CONSTANT_ACCESSOR("TEXTURE17", GL_TEXTURE17);
	CREATE_CONSTANT_ACCESSOR("TEXTURE18", GL_TEXTURE18);
	CREATE_CONSTANT_ACCESSOR("TEXTURE19", GL_TEXTURE19);
	CREATE_CONSTANT_ACCESSOR("TEXTURE20", GL_TEXTURE20);
	CREATE_CONSTANT_ACCESSOR("TEXTURE21", GL_TEXTURE21);
	CREATE_CONSTANT_ACCESSOR("TEXTURE22", GL_TEXTURE22);
	CREATE_CONSTANT_ACCESSOR("TEXTURE23", GL_TEXTURE23);
	CREATE_CONSTANT_ACCESSOR("TEXTURE24", GL_TEXTURE24);
	CREATE_CONSTANT_ACCESSOR("TEXTURE25", GL_TEXTURE25);
	CREATE_CONSTANT_ACCESSOR("TEXTURE26", GL_TEXTURE26);
	CREATE_CONSTANT_ACCESSOR("TEXTURE27", GL_TEXTURE27);
	CREATE_CONSTANT_ACCESSOR("TEXTURE28", GL_TEXTURE28);
	CREATE_CONSTANT_ACCESSOR("TEXTURE29", GL_TEXTURE29);
	CREATE_CONSTANT_ACCESSOR("TEXTURE30", GL_TEXTURE30);
	CREATE_CONSTANT_ACCESSOR("TEXTURE31", GL_TEXTURE31);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_TEXTURE", GL_ACTIVE_TEXTURE);
	CREATE_CONSTANT_ACCESSOR("CLIENT_ACTIVE_TEXTURE", GL_CLIENT_ACTIVE_TEXTURE);
	CREATE_CONSTANT_ACCESSOR("MAX_TEXTURE_UNITS", GL_MAX_TEXTURE_UNITS);
	CREATE_CONSTANT_ACCESSOR("NORMAL_MAP", GL_NORMAL_MAP);
	CREATE_CONSTANT_ACCESSOR("REFLECTION_MAP", GL_REFLECTION_MAP);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_CUBE_MAP", GL_TEXTURE_CUBE_MAP);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BINDING_CUBE_MAP", GL_TEXTURE_BINDING_CUBE_MAP);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_CUBE_MAP_POSITIVE_X", GL_TEXTURE_CUBE_MAP_POSITIVE_X);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_CUBE_MAP_NEGATIVE_X", GL_TEXTURE_CUBE_MAP_NEGATIVE_X);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_CUBE_MAP_POSITIVE_Y", GL_TEXTURE_CUBE_MAP_POSITIVE_Y);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_CUBE_MAP_NEGATIVE_Y", GL_TEXTURE_CUBE_MAP_NEGATIVE_Y);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_CUBE_MAP_POSITIVE_Z", GL_TEXTURE_CUBE_MAP_POSITIVE_Z);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_CUBE_MAP_NEGATIVE_Z", GL_TEXTURE_CUBE_MAP_NEGATIVE_Z);
	CREATE_CONSTANT_ACCESSOR("PROXY_TEXTURE_CUBE_MAP", GL_PROXY_TEXTURE_CUBE_MAP);
	CREATE_CONSTANT_ACCESSOR("MAX_CUBE_MAP_TEXTURE_SIZE", GL_MAX_CUBE_MAP_TEXTURE_SIZE);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_ALPHA", GL_COMPRESSED_ALPHA);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_LUMINANCE", GL_COMPRESSED_LUMINANCE);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_LUMINANCE_ALPHA", GL_COMPRESSED_LUMINANCE_ALPHA);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_INTENSITY", GL_COMPRESSED_INTENSITY);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGB", GL_COMPRESSED_RGB);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGBA", GL_COMPRESSED_RGBA);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_COMPRESSION_HINT", GL_TEXTURE_COMPRESSION_HINT);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_COMPRESSED_IMAGE_SIZE", GL_TEXTURE_COMPRESSED_IMAGE_SIZE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_COMPRESSED", GL_TEXTURE_COMPRESSED);
	CREATE_CONSTANT_ACCESSOR("NUM_COMPRESSED_TEXTURE_FORMATS", GL_NUM_COMPRESSED_TEXTURE_FORMATS);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_TEXTURE_FORMATS", GL_COMPRESSED_TEXTURE_FORMATS);
	CREATE_CONSTANT_ACCESSOR("MULTISAMPLE", GL_MULTISAMPLE);
	CREATE_CONSTANT_ACCESSOR("SAMPLE_ALPHA_TO_COVERAGE", GL_SAMPLE_ALPHA_TO_COVERAGE);
	CREATE_CONSTANT_ACCESSOR("SAMPLE_ALPHA_TO_ONE", GL_SAMPLE_ALPHA_TO_ONE);
	CREATE_CONSTANT_ACCESSOR("SAMPLE_COVERAGE", GL_SAMPLE_COVERAGE);
	CREATE_CONSTANT_ACCESSOR("SAMPLE_BUFFERS", GL_SAMPLE_BUFFERS);
	CREATE_CONSTANT_ACCESSOR("SAMPLES", GL_SAMPLES);
	CREATE_CONSTANT_ACCESSOR("SAMPLE_COVERAGE_VALUE", GL_SAMPLE_COVERAGE_VALUE);
	CREATE_CONSTANT_ACCESSOR("SAMPLE_COVERAGE_INVERT", GL_SAMPLE_COVERAGE_INVERT);
	CREATE_CONSTANT_ACCESSOR("MULTISAMPLE_BIT", GL_MULTISAMPLE_BIT);
	CREATE_CONSTANT_ACCESSOR("TRANSPOSE_MODELVIEW_MATRIX", GL_TRANSPOSE_MODELVIEW_MATRIX);
	CREATE_CONSTANT_ACCESSOR("TRANSPOSE_PROJECTION_MATRIX", GL_TRANSPOSE_PROJECTION_MATRIX);
	CREATE_CONSTANT_ACCESSOR("TRANSPOSE_TEXTURE_MATRIX", GL_TRANSPOSE_TEXTURE_MATRIX);
	CREATE_CONSTANT_ACCESSOR("TRANSPOSE_COLOR_MATRIX", GL_TRANSPOSE_COLOR_MATRIX);
	CREATE_CONSTANT_ACCESSOR("COMBINE", GL_COMBINE);
	CREATE_CONSTANT_ACCESSOR("COMBINE_RGB", GL_COMBINE_RGB);
	CREATE_CONSTANT_ACCESSOR("COMBINE_ALPHA", GL_COMBINE_ALPHA);
	CREATE_CONSTANT_ACCESSOR("SOURCE0_RGB", GL_SOURCE0_RGB);
	CREATE_CONSTANT_ACCESSOR("SOURCE1_RGB", GL_SOURCE1_RGB);
	CREATE_CONSTANT_ACCESSOR("SOURCE2_RGB", GL_SOURCE2_RGB);
	CREATE_CONSTANT_ACCESSOR("SOURCE0_ALPHA", GL_SOURCE0_ALPHA);
	CREATE_CONSTANT_ACCESSOR("SOURCE1_ALPHA", GL_SOURCE1_ALPHA);
	CREATE_CONSTANT_ACCESSOR("SOURCE2_ALPHA", GL_SOURCE2_ALPHA);
	CREATE_CONSTANT_ACCESSOR("OPERAND0_RGB", GL_OPERAND0_RGB);
	CREATE_CONSTANT_ACCESSOR("OPERAND1_RGB", GL_OPERAND1_RGB);
	CREATE_CONSTANT_ACCESSOR("OPERAND2_RGB", GL_OPERAND2_RGB);
	CREATE_CONSTANT_ACCESSOR("OPERAND0_ALPHA", GL_OPERAND0_ALPHA);
	CREATE_CONSTANT_ACCESSOR("OPERAND1_ALPHA", GL_OPERAND1_ALPHA);
	CREATE_CONSTANT_ACCESSOR("OPERAND2_ALPHA", GL_OPERAND2_ALPHA);
	CREATE_CONSTANT_ACCESSOR("RGB_SCALE", GL_RGB_SCALE);
	CREATE_CONSTANT_ACCESSOR("ADD_SIGNED", GL_ADD_SIGNED);
	CREATE_CONSTANT_ACCESSOR("INTERPOLATE", GL_INTERPOLATE);
	CREATE_CONSTANT_ACCESSOR("SUBTRACT", GL_SUBTRACT);
	CREATE_CONSTANT_ACCESSOR("CONSTANT", GL_CONSTANT);
	CREATE_CONSTANT_ACCESSOR("PRIMARY_COLOR", GL_PRIMARY_COLOR);
	CREATE_CONSTANT_ACCESSOR("PREVIOUS", GL_PREVIOUS);
	CREATE_CONSTANT_ACCESSOR("DOT3_RGB", GL_DOT3_RGB);
	CREATE_CONSTANT_ACCESSOR("DOT3_RGBA", GL_DOT3_RGBA);
	CREATE_CONSTANT_ACCESSOR("CLAMP_TO_BORDER", GL_CLAMP_TO_BORDER);

	tpl->Set(String::NewFromUtf8(isolate, "activeTexture"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("activeTexture requires 1 arguments");
			return;
		}

		GLenum texture = args[0]->Uint32Value();

		glActiveTexture(texture);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "clientActiveTexture"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("clientActiveTexture requires 1 arguments");
			return;
		}

		GLenum texture = args[0]->Uint32Value();

		glClientActiveTexture(texture);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "compressedTexImage1D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("compressedTexImage1D requires 7 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLint border = args[4]->Int32Value();
		GLsizei imageSize = args[5]->Int32Value();

		void* data = nullptr;
		if (args[6]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[6]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[6]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[6]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glCompressedTexImage1D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glCompressedTexImage1D(target, level, internalformat, width, border, imageSize, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "compressedTexImage2D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 8) {
			V8Helper::_instance->throwException("compressedTexImage2D requires 8 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();
		GLint border = args[5]->Int32Value();
		GLsizei imageSize = args[6]->Int32Value();

		void* data = nullptr;
		if (args[7]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[7]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[7]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[7]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glCompressedTexImage2D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glCompressedTexImage2D(target, level, internalformat, width, height, border, imageSize, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "compressedTexImage3D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 9) {
			V8Helper::_instance->throwException("compressedTexImage3D requires 9 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();
		GLsizei depth = args[5]->Int32Value();
		GLint border = args[6]->Int32Value();
		GLsizei imageSize = args[7]->Int32Value();

		void* data = nullptr;
		if (args[8]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[8]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[8]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[8]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glCompressedTexImage3D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glCompressedTexImage3D(target, level, internalformat, width, height, depth, border, imageSize, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "compressedTexSubImage1D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("compressedTexSubImage1D requires 7 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLsizei width = args[3]->Int32Value();
		GLenum format = args[4]->Uint32Value();
		GLsizei imageSize = args[5]->Int32Value();

		void* data = nullptr;
		if (args[6]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[6]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[6]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[6]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glCompressedTexSubImage1D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glCompressedTexSubImage1D(target, level, xoffset, width, format, imageSize, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "compressedTexSubImage2D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 9) {
			V8Helper::_instance->throwException("compressedTexSubImage2D requires 9 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLsizei width = args[4]->Int32Value();
		GLsizei height = args[5]->Int32Value();
		GLenum format = args[6]->Uint32Value();
		GLsizei imageSize = args[7]->Int32Value();

		void* data = nullptr;
		if (args[8]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[8]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[8]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[8]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glCompressedTexSubImage2D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glCompressedTexSubImage2D(target, level, xoffset, yoffset, width, height, format, imageSize, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "compressedTexSubImage3D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 11) {
			V8Helper::_instance->throwException("compressedTexSubImage3D requires 11 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLint zoffset = args[4]->Int32Value();
		GLsizei width = args[5]->Int32Value();
		GLsizei height = args[6]->Int32Value();
		GLsizei depth = args[7]->Int32Value();
		GLenum format = args[8]->Uint32Value();
		GLsizei imageSize = args[9]->Int32Value();

		void* data = nullptr;
		if (args[10]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[10]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[10]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[10]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glCompressedTexSubImage3D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glCompressedTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getCompressedTexImage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getCompressedTexImage requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint lod = args[1]->Int32Value();

		void* img = nullptr;
		if (args[2]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[2]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			img = reinterpret_cast<void*>(bdata);
		} else if (args[2]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[2]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			img = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetCompressedTexImage): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glGetCompressedTexImage(target, lod, img);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "loadTransposeMatrixd"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("loadTransposeMatrixd requires 1 arguments");
			return;
		}


		GLdouble* m = nullptr;
		if (args[0]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[0]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			m = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glLoadTransposeMatrixd): array must be of type Float64Array" << endl;
			exit(1);
		}


		glLoadTransposeMatrixd(m);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "loadTransposeMatrixf"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("loadTransposeMatrixf requires 1 arguments");
			return;
		}


		GLfloat* m = nullptr;
		if (args[0]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[0]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			m = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glLoadTransposeMatrixf): array must be of type Float32Array" << endl;
			exit(1);
		}


		glLoadTransposeMatrixf(m);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multTransposeMatrixd"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("multTransposeMatrixd requires 1 arguments");
			return;
		}


		GLdouble* m = nullptr;
		if (args[0]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[0]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			m = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glMultTransposeMatrixd): array must be of type Float64Array" << endl;
			exit(1);
		}


		glMultTransposeMatrixd(m);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multTransposeMatrixf"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("multTransposeMatrixf requires 1 arguments");
			return;
		}


		GLfloat* m = nullptr;
		if (args[0]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[0]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			m = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glMultTransposeMatrixf): array must be of type Float32Array" << endl;
			exit(1);
		}


		glMultTransposeMatrixf(m);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord1d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord1d requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLdouble s = args[1]->NumberValue();

		glMultiTexCoord1d(target, s);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord1dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord1dv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLdouble* v = nullptr;
		if (args[1]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[1]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord1dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glMultiTexCoord1dv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord1f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord1f requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLfloat s = GLfloat(args[1]->NumberValue());

		glMultiTexCoord1f(target, s);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord1fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord1fv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLfloat* v = nullptr;
		if (args[1]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[1]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord1fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glMultiTexCoord1fv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord1i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord1i requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint s = args[1]->Int32Value();

		glMultiTexCoord1i(target, s);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord1iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord1iv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLint* v = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord1iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glMultiTexCoord1iv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord1s"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord1s requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLshort s = GLshort(args[1]->Int32Value());

		glMultiTexCoord1s(target, s);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord1sv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord1sv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLshort* v = nullptr;
		if (args[1]->IsInt16Array()) {
			v8::Local<v8::Int16Array> view = (args[1]).As<v8::Int16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLshort*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord1sv): array must be of type Int16Array" << endl;
			exit(1);
		}


		glMultiTexCoord1sv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord2d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("multiTexCoord2d requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLdouble s = args[1]->NumberValue();
		GLdouble t = args[2]->NumberValue();

		glMultiTexCoord2d(target, s, t);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord2dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord2dv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLdouble* v = nullptr;
		if (args[1]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[1]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord2dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glMultiTexCoord2dv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord2f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("multiTexCoord2f requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLfloat s = GLfloat(args[1]->NumberValue());
		GLfloat t = GLfloat(args[2]->NumberValue());

		glMultiTexCoord2f(target, s, t);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord2fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord2fv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLfloat* v = nullptr;
		if (args[1]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[1]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord2fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glMultiTexCoord2fv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord2i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("multiTexCoord2i requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint s = args[1]->Int32Value();
		GLint t = args[2]->Int32Value();

		glMultiTexCoord2i(target, s, t);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord2iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord2iv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLint* v = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord2iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glMultiTexCoord2iv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord2s"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("multiTexCoord2s requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLshort s = GLshort(args[1]->Int32Value());
		GLshort t = GLshort(args[2]->Int32Value());

		glMultiTexCoord2s(target, s, t);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord2sv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord2sv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLshort* v = nullptr;
		if (args[1]->IsInt16Array()) {
			v8::Local<v8::Int16Array> view = (args[1]).As<v8::Int16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLshort*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord2sv): array must be of type Int16Array" << endl;
			exit(1);
		}


		glMultiTexCoord2sv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord3d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("multiTexCoord3d requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLdouble s = args[1]->NumberValue();
		GLdouble t = args[2]->NumberValue();
		GLdouble r = args[3]->NumberValue();

		glMultiTexCoord3d(target, s, t, r);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord3dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord3dv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLdouble* v = nullptr;
		if (args[1]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[1]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord3dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glMultiTexCoord3dv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord3f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("multiTexCoord3f requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLfloat s = GLfloat(args[1]->NumberValue());
		GLfloat t = GLfloat(args[2]->NumberValue());
		GLfloat r = GLfloat(args[3]->NumberValue());

		glMultiTexCoord3f(target, s, t, r);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord3fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord3fv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLfloat* v = nullptr;
		if (args[1]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[1]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord3fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glMultiTexCoord3fv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord3i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("multiTexCoord3i requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint s = args[1]->Int32Value();
		GLint t = args[2]->Int32Value();
		GLint r = args[3]->Int32Value();

		glMultiTexCoord3i(target, s, t, r);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord3iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord3iv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLint* v = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord3iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glMultiTexCoord3iv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord3s"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("multiTexCoord3s requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLshort s = GLshort(args[1]->Int32Value());
		GLshort t = GLshort(args[2]->Int32Value());
		GLshort r = GLshort(args[3]->Int32Value());

		glMultiTexCoord3s(target, s, t, r);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord3sv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord3sv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLshort* v = nullptr;
		if (args[1]->IsInt16Array()) {
			v8::Local<v8::Int16Array> view = (args[1]).As<v8::Int16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLshort*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord3sv): array must be of type Int16Array" << endl;
			exit(1);
		}


		glMultiTexCoord3sv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord4d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("multiTexCoord4d requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLdouble s = args[1]->NumberValue();
		GLdouble t = args[2]->NumberValue();
		GLdouble r = args[3]->NumberValue();
		GLdouble q = args[4]->NumberValue();

		glMultiTexCoord4d(target, s, t, r, q);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord4dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord4dv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLdouble* v = nullptr;
		if (args[1]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[1]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord4dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glMultiTexCoord4dv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord4f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("multiTexCoord4f requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLfloat s = GLfloat(args[1]->NumberValue());
		GLfloat t = GLfloat(args[2]->NumberValue());
		GLfloat r = GLfloat(args[3]->NumberValue());
		GLfloat q = GLfloat(args[4]->NumberValue());

		glMultiTexCoord4f(target, s, t, r, q);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord4fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord4fv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLfloat* v = nullptr;
		if (args[1]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[1]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord4fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glMultiTexCoord4fv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord4i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("multiTexCoord4i requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint s = args[1]->Int32Value();
		GLint t = args[2]->Int32Value();
		GLint r = args[3]->Int32Value();
		GLint q = args[4]->Int32Value();

		glMultiTexCoord4i(target, s, t, r, q);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord4iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord4iv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLint* v = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord4iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glMultiTexCoord4iv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord4s"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("multiTexCoord4s requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLshort s = GLshort(args[1]->Int32Value());
		GLshort t = GLshort(args[2]->Int32Value());
		GLshort r = GLshort(args[3]->Int32Value());
		GLshort q = GLshort(args[4]->Int32Value());

		glMultiTexCoord4s(target, s, t, r, q);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord4sv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("multiTexCoord4sv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLshort* v = nullptr;
		if (args[1]->IsInt16Array()) {
			v8::Local<v8::Int16Array> view = (args[1]).As<v8::Int16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLshort*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoord4sv): array must be of type Int16Array" << endl;
			exit(1);
		}


		glMultiTexCoord4sv(target, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "sampleCoverage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("sampleCoverage requires 2 arguments");
			return;
		}

		GLclampf value = GLclampf(args[0]->NumberValue());
		GLboolean invert = GLboolean(args[1]->Uint32Value());

		glSampleCoverage(value, invert);
	}));



	/* ------------------------------ GL_VERSION_1_4 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("GENERATE_MIPMAP", GL_GENERATE_MIPMAP);
	CREATE_CONSTANT_ACCESSOR("GENERATE_MIPMAP_HINT", GL_GENERATE_MIPMAP_HINT);
	CREATE_CONSTANT_ACCESSOR("DEPTH_COMPONENT16", GL_DEPTH_COMPONENT16);
	CREATE_CONSTANT_ACCESSOR("DEPTH_COMPONENT24", GL_DEPTH_COMPONENT24);
	CREATE_CONSTANT_ACCESSOR("DEPTH_COMPONENT32", GL_DEPTH_COMPONENT32);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_DEPTH_SIZE", GL_TEXTURE_DEPTH_SIZE);
	CREATE_CONSTANT_ACCESSOR("DEPTH_TEXTURE_MODE", GL_DEPTH_TEXTURE_MODE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_COMPARE_MODE", GL_TEXTURE_COMPARE_MODE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_COMPARE_FUNC", GL_TEXTURE_COMPARE_FUNC);
	CREATE_CONSTANT_ACCESSOR("COMPARE_R_TO_TEXTURE", GL_COMPARE_R_TO_TEXTURE);
	CREATE_CONSTANT_ACCESSOR("FOG_COORDINATE_SOURCE", GL_FOG_COORDINATE_SOURCE);
	CREATE_CONSTANT_ACCESSOR("FOG_COORDINATE", GL_FOG_COORDINATE);
	CREATE_CONSTANT_ACCESSOR("FRAGMENT_DEPTH", GL_FRAGMENT_DEPTH);
	CREATE_CONSTANT_ACCESSOR("CURRENT_FOG_COORDINATE", GL_CURRENT_FOG_COORDINATE);
	CREATE_CONSTANT_ACCESSOR("FOG_COORDINATE_ARRAY_TYPE", GL_FOG_COORDINATE_ARRAY_TYPE);
	CREATE_CONSTANT_ACCESSOR("FOG_COORDINATE_ARRAY_STRIDE", GL_FOG_COORDINATE_ARRAY_STRIDE);
	CREATE_CONSTANT_ACCESSOR("FOG_COORDINATE_ARRAY_POINTER", GL_FOG_COORDINATE_ARRAY_POINTER);
	CREATE_CONSTANT_ACCESSOR("FOG_COORDINATE_ARRAY", GL_FOG_COORDINATE_ARRAY);
	CREATE_CONSTANT_ACCESSOR("POINT_SIZE_MIN", GL_POINT_SIZE_MIN);
	CREATE_CONSTANT_ACCESSOR("POINT_SIZE_MAX", GL_POINT_SIZE_MAX);
	CREATE_CONSTANT_ACCESSOR("POINT_FADE_THRESHOLD_SIZE", GL_POINT_FADE_THRESHOLD_SIZE);
	CREATE_CONSTANT_ACCESSOR("POINT_DISTANCE_ATTENUATION", GL_POINT_DISTANCE_ATTENUATION);
	CREATE_CONSTANT_ACCESSOR("COLOR_SUM", GL_COLOR_SUM);
	CREATE_CONSTANT_ACCESSOR("CURRENT_SECONDARY_COLOR", GL_CURRENT_SECONDARY_COLOR);
	CREATE_CONSTANT_ACCESSOR("SECONDARY_COLOR_ARRAY_SIZE", GL_SECONDARY_COLOR_ARRAY_SIZE);
	CREATE_CONSTANT_ACCESSOR("SECONDARY_COLOR_ARRAY_TYPE", GL_SECONDARY_COLOR_ARRAY_TYPE);
	CREATE_CONSTANT_ACCESSOR("SECONDARY_COLOR_ARRAY_STRIDE", GL_SECONDARY_COLOR_ARRAY_STRIDE);
	CREATE_CONSTANT_ACCESSOR("SECONDARY_COLOR_ARRAY_POINTER", GL_SECONDARY_COLOR_ARRAY_POINTER);
	CREATE_CONSTANT_ACCESSOR("SECONDARY_COLOR_ARRAY", GL_SECONDARY_COLOR_ARRAY);
	CREATE_CONSTANT_ACCESSOR("BLEND_DST_RGB", GL_BLEND_DST_RGB);
	CREATE_CONSTANT_ACCESSOR("BLEND_SRC_RGB", GL_BLEND_SRC_RGB);
	CREATE_CONSTANT_ACCESSOR("BLEND_DST_ALPHA", GL_BLEND_DST_ALPHA);
	CREATE_CONSTANT_ACCESSOR("BLEND_SRC_ALPHA", GL_BLEND_SRC_ALPHA);
	CREATE_CONSTANT_ACCESSOR("INCR_WRAP", GL_INCR_WRAP);
	CREATE_CONSTANT_ACCESSOR("DECR_WRAP", GL_DECR_WRAP);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_FILTER_CONTROL", GL_TEXTURE_FILTER_CONTROL);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_LOD_BIAS", GL_TEXTURE_LOD_BIAS);
	CREATE_CONSTANT_ACCESSOR("MAX_TEXTURE_LOD_BIAS", GL_MAX_TEXTURE_LOD_BIAS);
	CREATE_CONSTANT_ACCESSOR("MIRRORED_REPEAT", GL_MIRRORED_REPEAT);

	tpl->Set(String::NewFromUtf8(isolate, "blendEquation"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("blendEquation requires 1 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();

		glBlendEquation(mode);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "blendColor"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("blendColor requires 4 arguments");
			return;
		}

		GLclampf red = GLclampf(args[0]->NumberValue());
		GLclampf green = GLclampf(args[1]->NumberValue());
		GLclampf blue = GLclampf(args[2]->NumberValue());
		GLclampf alpha = GLclampf(args[3]->NumberValue());

		glBlendColor(red, green, blue, alpha);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "fogCoordf"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("fogCoordf requires 1 arguments");
			return;
		}

		GLfloat coord = GLfloat(args[0]->NumberValue());

		glFogCoordf(coord);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "fogCoordfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("fogCoordfv requires 1 arguments");
			return;
		}


		GLfloat* coord = nullptr;
		if (args[0]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[0]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			coord = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glFogCoordfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glFogCoordfv(coord);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "fogCoordd"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("fogCoordd requires 1 arguments");
			return;
		}

		GLdouble coord = args[0]->NumberValue();

		glFogCoordd(coord);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "fogCoorddv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("fogCoorddv requires 1 arguments");
			return;
		}


		GLdouble* coord = nullptr;
		if (args[0]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[0]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			coord = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glFogCoorddv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glFogCoorddv(coord);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "fogCoordPointer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("fogCoordPointer requires 3 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();
		GLsizei stride = args[1]->Int32Value();

		void* pointer = nullptr;
		if (args[2]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[2]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pointer = reinterpret_cast<void*>(bdata);
		} else if (args[2]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[2]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pointer = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glFogCoordPointer): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glFogCoordPointer(type, stride, pointer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiDrawArrays"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("multiDrawArrays requires 4 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();

		GLint* first = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			first = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glMultiDrawArrays): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLsizei* count = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			count = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glMultiDrawArrays): array must be of type Int32Array" << endl;
			exit(1);
		}

		GLsizei drawcount = args[3]->Int32Value();

		glMultiDrawArrays(mode, first, count, drawcount);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "pointParameteri"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("pointParameteri requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();
		GLint param = args[1]->Int32Value();

		glPointParameteri(pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "pointParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("pointParameteriv requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();

		GLint* params = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glPointParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glPointParameteriv(pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "pointParameterf"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("pointParameterf requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();
		GLfloat param = GLfloat(args[1]->NumberValue());

		glPointParameterf(pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "pointParameterfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("pointParameterfv requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();

		GLfloat* params = nullptr;
		if (args[1]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[1]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glPointParameterfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glPointParameterfv(pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3b"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("secondaryColor3b requires 3 arguments");
			return;
		}

		GLbyte red = GLbyte(args[0]->Int32Value());
		GLbyte green = GLbyte(args[1]->Int32Value());
		GLbyte blue = GLbyte(args[2]->Int32Value());

		glSecondaryColor3b(red, green, blue);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3bv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("secondaryColor3bv requires 1 arguments");
			return;
		}


		GLbyte* v = nullptr;
		if (args[0]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[0]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLbyte*>(bdata);
		} else {
			cout << "ERROR(glSecondaryColor3bv): array must be of type Int8Array" << endl;
			exit(1);
		}


		glSecondaryColor3bv(v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("secondaryColor3d requires 3 arguments");
			return;
		}

		GLdouble red = args[0]->NumberValue();
		GLdouble green = args[1]->NumberValue();
		GLdouble blue = args[2]->NumberValue();

		glSecondaryColor3d(red, green, blue);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("secondaryColor3dv requires 1 arguments");
			return;
		}


		GLdouble* v = nullptr;
		if (args[0]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[0]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glSecondaryColor3dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glSecondaryColor3dv(v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("secondaryColor3f requires 3 arguments");
			return;
		}

		GLfloat red = GLfloat(args[0]->NumberValue());
		GLfloat green = GLfloat(args[1]->NumberValue());
		GLfloat blue = GLfloat(args[2]->NumberValue());

		glSecondaryColor3f(red, green, blue);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("secondaryColor3fv requires 1 arguments");
			return;
		}


		GLfloat* v = nullptr;
		if (args[0]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[0]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glSecondaryColor3fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glSecondaryColor3fv(v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("secondaryColor3i requires 3 arguments");
			return;
		}

		GLint red = args[0]->Int32Value();
		GLint green = args[1]->Int32Value();
		GLint blue = args[2]->Int32Value();

		glSecondaryColor3i(red, green, blue);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("secondaryColor3iv requires 1 arguments");
			return;
		}


		GLint* v = nullptr;
		if (args[0]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[0]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glSecondaryColor3iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glSecondaryColor3iv(v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3s"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("secondaryColor3s requires 3 arguments");
			return;
		}

		GLshort red = GLshort(args[0]->Int32Value());
		GLshort green = GLshort(args[1]->Int32Value());
		GLshort blue = GLshort(args[2]->Int32Value());

		glSecondaryColor3s(red, green, blue);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3sv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("secondaryColor3sv requires 1 arguments");
			return;
		}


		GLshort* v = nullptr;
		if (args[0]->IsInt16Array()) {
			v8::Local<v8::Int16Array> view = (args[0]).As<v8::Int16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLshort*>(bdata);
		} else {
			cout << "ERROR(glSecondaryColor3sv): array must be of type Int16Array" << endl;
			exit(1);
		}


		glSecondaryColor3sv(v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3ub"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("secondaryColor3ub requires 3 arguments");
			return;
		}

		GLubyte red = GLubyte(args[0]->Uint32Value());
		GLubyte green = GLubyte(args[1]->Uint32Value());
		GLubyte blue = GLubyte(args[2]->Uint32Value());

		glSecondaryColor3ub(red, green, blue);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3ubv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("secondaryColor3ubv requires 1 arguments");
			return;
		}


		GLubyte* v = nullptr;
		if (args[0]->IsUint8Array()) {
			v8::Local<v8::Uint8Array> view = (args[0]).As<v8::Uint8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLubyte*>(bdata);
		} else {
			cout << "ERROR(glSecondaryColor3ubv): array must be of type Uint8Array" << endl;
			exit(1);
		}


		glSecondaryColor3ubv(v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("secondaryColor3ui requires 3 arguments");
			return;
		}

		GLuint red = args[0]->Uint32Value();
		GLuint green = args[1]->Uint32Value();
		GLuint blue = args[2]->Uint32Value();

		glSecondaryColor3ui(red, green, blue);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("secondaryColor3uiv requires 1 arguments");
			return;
		}


		GLuint* v = nullptr;
		if (args[0]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[0]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glSecondaryColor3uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glSecondaryColor3uiv(v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3us"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("secondaryColor3us requires 3 arguments");
			return;
		}

		GLushort red = GLushort(args[0]->Uint32Value());
		GLushort green = GLushort(args[1]->Uint32Value());
		GLushort blue = GLushort(args[2]->Uint32Value());

		glSecondaryColor3us(red, green, blue);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColor3usv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("secondaryColor3usv requires 1 arguments");
			return;
		}


		GLushort* v = nullptr;
		if (args[0]->IsUint16Array()) {
			v8::Local<v8::Uint16Array> view = (args[0]).As<v8::Uint16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLushort*>(bdata);
		} else {
			cout << "ERROR(glSecondaryColor3usv): array must be of type Uint16Array" << endl;
			exit(1);
		}


		glSecondaryColor3usv(v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColorPointer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("secondaryColorPointer requires 4 arguments");
			return;
		}

		GLint size = args[0]->Int32Value();
		GLenum type = args[1]->Uint32Value();
		GLsizei stride = args[2]->Int32Value();

		void* pointer = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pointer = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pointer = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glSecondaryColorPointer): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glSecondaryColorPointer(size, type, stride, pointer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "blendFuncSeparate"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("blendFuncSeparate requires 4 arguments");
			return;
		}

		GLenum sfactorRGB = args[0]->Uint32Value();
		GLenum dfactorRGB = args[1]->Uint32Value();
		GLenum sfactorAlpha = args[2]->Uint32Value();
		GLenum dfactorAlpha = args[3]->Uint32Value();

		glBlendFuncSeparate(sfactorRGB, dfactorRGB, sfactorAlpha, dfactorAlpha);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos2d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("windowPos2d requires 2 arguments");
			return;
		}

		GLdouble x = args[0]->NumberValue();
		GLdouble y = args[1]->NumberValue();

		glWindowPos2d(x, y);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos2f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("windowPos2f requires 2 arguments");
			return;
		}

		GLfloat x = GLfloat(args[0]->NumberValue());
		GLfloat y = GLfloat(args[1]->NumberValue());

		glWindowPos2f(x, y);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos2i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("windowPos2i requires 2 arguments");
			return;
		}

		GLint x = args[0]->Int32Value();
		GLint y = args[1]->Int32Value();

		glWindowPos2i(x, y);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos2s"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("windowPos2s requires 2 arguments");
			return;
		}

		GLshort x = GLshort(args[0]->Int32Value());
		GLshort y = GLshort(args[1]->Int32Value());

		glWindowPos2s(x, y);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos2dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("windowPos2dv requires 1 arguments");
			return;
		}


		GLdouble* p = nullptr;
		if (args[0]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[0]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			p = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glWindowPos2dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glWindowPos2dv(p);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos2fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("windowPos2fv requires 1 arguments");
			return;
		}


		GLfloat* p = nullptr;
		if (args[0]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[0]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			p = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glWindowPos2fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glWindowPos2fv(p);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos2iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("windowPos2iv requires 1 arguments");
			return;
		}


		GLint* p = nullptr;
		if (args[0]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[0]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			p = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glWindowPos2iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glWindowPos2iv(p);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos2sv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("windowPos2sv requires 1 arguments");
			return;
		}


		GLshort* p = nullptr;
		if (args[0]->IsInt16Array()) {
			v8::Local<v8::Int16Array> view = (args[0]).As<v8::Int16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			p = reinterpret_cast<GLshort*>(bdata);
		} else {
			cout << "ERROR(glWindowPos2sv): array must be of type Int16Array" << endl;
			exit(1);
		}


		glWindowPos2sv(p);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos3d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("windowPos3d requires 3 arguments");
			return;
		}

		GLdouble x = args[0]->NumberValue();
		GLdouble y = args[1]->NumberValue();
		GLdouble z = args[2]->NumberValue();

		glWindowPos3d(x, y, z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos3f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("windowPos3f requires 3 arguments");
			return;
		}

		GLfloat x = GLfloat(args[0]->NumberValue());
		GLfloat y = GLfloat(args[1]->NumberValue());
		GLfloat z = GLfloat(args[2]->NumberValue());

		glWindowPos3f(x, y, z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos3i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("windowPos3i requires 3 arguments");
			return;
		}

		GLint x = args[0]->Int32Value();
		GLint y = args[1]->Int32Value();
		GLint z = args[2]->Int32Value();

		glWindowPos3i(x, y, z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos3s"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("windowPos3s requires 3 arguments");
			return;
		}

		GLshort x = GLshort(args[0]->Int32Value());
		GLshort y = GLshort(args[1]->Int32Value());
		GLshort z = GLshort(args[2]->Int32Value());

		glWindowPos3s(x, y, z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos3dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("windowPos3dv requires 1 arguments");
			return;
		}


		GLdouble* p = nullptr;
		if (args[0]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[0]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			p = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glWindowPos3dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glWindowPos3dv(p);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos3fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("windowPos3fv requires 1 arguments");
			return;
		}


		GLfloat* p = nullptr;
		if (args[0]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[0]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			p = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glWindowPos3fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glWindowPos3fv(p);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos3iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("windowPos3iv requires 1 arguments");
			return;
		}


		GLint* p = nullptr;
		if (args[0]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[0]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			p = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glWindowPos3iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glWindowPos3iv(p);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "windowPos3sv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("windowPos3sv requires 1 arguments");
			return;
		}


		GLshort* p = nullptr;
		if (args[0]->IsInt16Array()) {
			v8::Local<v8::Int16Array> view = (args[0]).As<v8::Int16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			p = reinterpret_cast<GLshort*>(bdata);
		} else {
			cout << "ERROR(glWindowPos3sv): array must be of type Int16Array" << endl;
			exit(1);
		}


		glWindowPos3sv(p);
	}));



	/* ------------------------------ GL_VERSION_1_5 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("BUFFER_SIZE", GL_BUFFER_SIZE);
	CREATE_CONSTANT_ACCESSOR("BUFFER_USAGE", GL_BUFFER_USAGE);
	CREATE_CONSTANT_ACCESSOR("QUERY_COUNTER_BITS", GL_QUERY_COUNTER_BITS);
	CREATE_CONSTANT_ACCESSOR("CURRENT_QUERY", GL_CURRENT_QUERY);
	CREATE_CONSTANT_ACCESSOR("QUERY_RESULT", GL_QUERY_RESULT);
	CREATE_CONSTANT_ACCESSOR("QUERY_RESULT_AVAILABLE", GL_QUERY_RESULT_AVAILABLE);
	CREATE_CONSTANT_ACCESSOR("ARRAY_BUFFER", GL_ARRAY_BUFFER);
	CREATE_CONSTANT_ACCESSOR("ELEMENT_ARRAY_BUFFER", GL_ELEMENT_ARRAY_BUFFER);
	CREATE_CONSTANT_ACCESSOR("ARRAY_BUFFER_BINDING", GL_ARRAY_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("ELEMENT_ARRAY_BUFFER_BINDING", GL_ELEMENT_ARRAY_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("VERTEX_ARRAY_BUFFER_BINDING", GL_VERTEX_ARRAY_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("NORMAL_ARRAY_BUFFER_BINDING", GL_NORMAL_ARRAY_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("COLOR_ARRAY_BUFFER_BINDING", GL_COLOR_ARRAY_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("INDEX_ARRAY_BUFFER_BINDING", GL_INDEX_ARRAY_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_COORD_ARRAY_BUFFER_BINDING", GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("EDGE_FLAG_ARRAY_BUFFER_BINDING", GL_EDGE_FLAG_ARRAY_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("SECONDARY_COLOR_ARRAY_BUFFER_BINDING", GL_SECONDARY_COLOR_ARRAY_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("FOG_COORDINATE_ARRAY_BUFFER_BINDING", GL_FOG_COORDINATE_ARRAY_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("WEIGHT_ARRAY_BUFFER_BINDING", GL_WEIGHT_ARRAY_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("VERTEX_ATTRIB_ARRAY_BUFFER_BINDING", GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("READ_ONLY", GL_READ_ONLY);
	CREATE_CONSTANT_ACCESSOR("WRITE_ONLY", GL_WRITE_ONLY);
	CREATE_CONSTANT_ACCESSOR("READ_WRITE", GL_READ_WRITE);
	CREATE_CONSTANT_ACCESSOR("BUFFER_ACCESS", GL_BUFFER_ACCESS);
	CREATE_CONSTANT_ACCESSOR("BUFFER_MAPPED", GL_BUFFER_MAPPED);
	CREATE_CONSTANT_ACCESSOR("BUFFER_MAP_POINTER", GL_BUFFER_MAP_POINTER);
	CREATE_CONSTANT_ACCESSOR("STREAM_DRAW", GL_STREAM_DRAW);
	CREATE_CONSTANT_ACCESSOR("STREAM_READ", GL_STREAM_READ);
	CREATE_CONSTANT_ACCESSOR("STREAM_COPY", GL_STREAM_COPY);
	CREATE_CONSTANT_ACCESSOR("STATIC_DRAW", GL_STATIC_DRAW);
	CREATE_CONSTANT_ACCESSOR("STATIC_READ", GL_STATIC_READ);
	CREATE_CONSTANT_ACCESSOR("STATIC_COPY", GL_STATIC_COPY);
	CREATE_CONSTANT_ACCESSOR("DYNAMIC_DRAW", GL_DYNAMIC_DRAW);
	CREATE_CONSTANT_ACCESSOR("DYNAMIC_READ", GL_DYNAMIC_READ);
	CREATE_CONSTANT_ACCESSOR("DYNAMIC_COPY", GL_DYNAMIC_COPY);
	CREATE_CONSTANT_ACCESSOR("SAMPLES_PASSED", GL_SAMPLES_PASSED);
	CREATE_CONSTANT_ACCESSOR("FOG_COORD_SRC", GL_FOG_COORD_SRC);
	CREATE_CONSTANT_ACCESSOR("FOG_COORD", GL_FOG_COORD);
	CREATE_CONSTANT_ACCESSOR("CURRENT_FOG_COORD", GL_CURRENT_FOG_COORD);
	CREATE_CONSTANT_ACCESSOR("FOG_COORD_ARRAY_TYPE", GL_FOG_COORD_ARRAY_TYPE);
	CREATE_CONSTANT_ACCESSOR("FOG_COORD_ARRAY_STRIDE", GL_FOG_COORD_ARRAY_STRIDE);
	CREATE_CONSTANT_ACCESSOR("FOG_COORD_ARRAY_POINTER", GL_FOG_COORD_ARRAY_POINTER);
	CREATE_CONSTANT_ACCESSOR("FOG_COORD_ARRAY", GL_FOG_COORD_ARRAY);
	CREATE_CONSTANT_ACCESSOR("FOG_COORD_ARRAY_BUFFER_BINDING", GL_FOG_COORD_ARRAY_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("SRC0_RGB", GL_SRC0_RGB);
	CREATE_CONSTANT_ACCESSOR("SRC1_RGB", GL_SRC1_RGB);
	CREATE_CONSTANT_ACCESSOR("SRC2_RGB", GL_SRC2_RGB);
	CREATE_CONSTANT_ACCESSOR("SRC0_ALPHA", GL_SRC0_ALPHA);
	CREATE_CONSTANT_ACCESSOR("SRC1_ALPHA", GL_SRC1_ALPHA);
	CREATE_CONSTANT_ACCESSOR("SRC2_ALPHA", GL_SRC2_ALPHA);

	tpl->Set(String::NewFromUtf8(isolate, "genQueries"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("genQueries requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* ids = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			ids = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGenQueries): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGenQueries(n, ids);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "deleteQueries"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("deleteQueries requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* ids = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			ids = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glDeleteQueries): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glDeleteQueries(n, ids);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "beginQuery"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("beginQuery requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint id = args[1]->Uint32Value();

		glBeginQuery(target, id);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "endQuery"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("endQuery requires 1 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		glEndQuery(target);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getQueryiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getQueryiv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetQueryiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetQueryiv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getQueryObjectiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getQueryObjectiv requires 3 arguments");
			return;
		}

		GLuint id = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetQueryObjectiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetQueryObjectiv(id, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getQueryObjectuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getQueryObjectuiv requires 3 arguments");
			return;
		}

		GLuint id = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLuint* params = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGetQueryObjectuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGetQueryObjectuiv(id, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "bindBuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("bindBuffer requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint buffer = args[1]->Uint32Value();

		glBindBuffer(target, buffer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "deleteBuffers"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("deleteBuffers requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* buffers = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			buffers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glDeleteBuffers): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glDeleteBuffers(n, buffers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "genBuffers"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("genBuffers requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* buffers = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			buffers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGenBuffers): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGenBuffers(n, buffers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "bufferData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("bufferData requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizeiptr size = GLsizeiptr(args[1]->Int32Value());

		void* data = nullptr;
		if (args[2]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[2]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[2]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[2]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			//cout << "ERROR(glBufferData): array must be of type ArrayBuffer" << endl;
			//exit(1);
		}

		GLenum usage = args[3]->Uint32Value();

		glBufferData(target, size, data, usage);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "bufferSubData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("bufferSubData requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLintptr offset = GLintptr(args[1]->Int32Value());
		GLsizeiptr size = GLsizeiptr(args[2]->Int32Value());

		void* data = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glBufferSubData): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glBufferSubData(target, offset, size, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getBufferSubData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getBufferSubData requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLintptr offset = GLintptr(args[1]->Int32Value());
		GLsizeiptr size = GLsizeiptr(args[2]->Int32Value());

		void* data = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetBufferSubData): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glGetBufferSubData(target, offset, size, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getBufferParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getBufferParameteriv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetBufferParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetBufferParameteriv(target, pname, params);
	}));




	/* ------------------------------ GL_VERSION_2_0 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("BLEND_EQUATION_RGB", GL_BLEND_EQUATION_RGB);
	CREATE_CONSTANT_ACCESSOR("VERTEX_ATTRIB_ARRAY_ENABLED", GL_VERTEX_ATTRIB_ARRAY_ENABLED);
	CREATE_CONSTANT_ACCESSOR("VERTEX_ATTRIB_ARRAY_SIZE", GL_VERTEX_ATTRIB_ARRAY_SIZE);
	CREATE_CONSTANT_ACCESSOR("VERTEX_ATTRIB_ARRAY_STRIDE", GL_VERTEX_ATTRIB_ARRAY_STRIDE);
	CREATE_CONSTANT_ACCESSOR("VERTEX_ATTRIB_ARRAY_TYPE", GL_VERTEX_ATTRIB_ARRAY_TYPE);
	CREATE_CONSTANT_ACCESSOR("CURRENT_VERTEX_ATTRIB", GL_CURRENT_VERTEX_ATTRIB);
	CREATE_CONSTANT_ACCESSOR("VERTEX_PROGRAM_POINT_SIZE", GL_VERTEX_PROGRAM_POINT_SIZE);
	CREATE_CONSTANT_ACCESSOR("VERTEX_PROGRAM_TWO_SIDE", GL_VERTEX_PROGRAM_TWO_SIDE);
	CREATE_CONSTANT_ACCESSOR("VERTEX_ATTRIB_ARRAY_POINTER", GL_VERTEX_ATTRIB_ARRAY_POINTER);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BACK_FUNC", GL_STENCIL_BACK_FUNC);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BACK_FAIL", GL_STENCIL_BACK_FAIL);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BACK_PASS_DEPTH_FAIL", GL_STENCIL_BACK_PASS_DEPTH_FAIL);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BACK_PASS_DEPTH_PASS", GL_STENCIL_BACK_PASS_DEPTH_PASS);
	CREATE_CONSTANT_ACCESSOR("MAX_DRAW_BUFFERS", GL_MAX_DRAW_BUFFERS);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER0", GL_DRAW_BUFFER0);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER1", GL_DRAW_BUFFER1);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER2", GL_DRAW_BUFFER2);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER3", GL_DRAW_BUFFER3);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER4", GL_DRAW_BUFFER4);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER5", GL_DRAW_BUFFER5);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER6", GL_DRAW_BUFFER6);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER7", GL_DRAW_BUFFER7);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER8", GL_DRAW_BUFFER8);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER9", GL_DRAW_BUFFER9);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER10", GL_DRAW_BUFFER10);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER11", GL_DRAW_BUFFER11);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER12", GL_DRAW_BUFFER12);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER13", GL_DRAW_BUFFER13);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER14", GL_DRAW_BUFFER14);
	CREATE_CONSTANT_ACCESSOR("DRAW_BUFFER15", GL_DRAW_BUFFER15);
	CREATE_CONSTANT_ACCESSOR("BLEND_EQUATION_ALPHA", GL_BLEND_EQUATION_ALPHA);
	CREATE_CONSTANT_ACCESSOR("POINT_SPRITE", GL_POINT_SPRITE);
	CREATE_CONSTANT_ACCESSOR("COORD_REPLACE", GL_COORD_REPLACE);
	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_ATTRIBS", GL_MAX_VERTEX_ATTRIBS);
	CREATE_CONSTANT_ACCESSOR("VERTEX_ATTRIB_ARRAY_NORMALIZED", GL_VERTEX_ATTRIB_ARRAY_NORMALIZED);
	CREATE_CONSTANT_ACCESSOR("MAX_TEXTURE_COORDS", GL_MAX_TEXTURE_COORDS);
	CREATE_CONSTANT_ACCESSOR("MAX_TEXTURE_IMAGE_UNITS", GL_MAX_TEXTURE_IMAGE_UNITS);
	CREATE_CONSTANT_ACCESSOR("FRAGMENT_SHADER", GL_FRAGMENT_SHADER);
	CREATE_CONSTANT_ACCESSOR("VERTEX_SHADER", GL_VERTEX_SHADER);
	CREATE_CONSTANT_ACCESSOR("MAX_FRAGMENT_UNIFORM_COMPONENTS", GL_MAX_FRAGMENT_UNIFORM_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_UNIFORM_COMPONENTS", GL_MAX_VERTEX_UNIFORM_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_VARYING_FLOATS", GL_MAX_VARYING_FLOATS);
	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_TEXTURE_IMAGE_UNITS", GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_TEXTURE_IMAGE_UNITS", GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS);
	CREATE_CONSTANT_ACCESSOR("SHADER_TYPE", GL_SHADER_TYPE);
	CREATE_CONSTANT_ACCESSOR("FLOAT_VEC2", GL_FLOAT_VEC2);
	CREATE_CONSTANT_ACCESSOR("FLOAT_VEC3", GL_FLOAT_VEC3);
	CREATE_CONSTANT_ACCESSOR("FLOAT_VEC4", GL_FLOAT_VEC4);
	CREATE_CONSTANT_ACCESSOR("INT_VEC2", GL_INT_VEC2);
	CREATE_CONSTANT_ACCESSOR("INT_VEC3", GL_INT_VEC3);
	CREATE_CONSTANT_ACCESSOR("INT_VEC4", GL_INT_VEC4);
	CREATE_CONSTANT_ACCESSOR("BOOL", GL_BOOL);
	CREATE_CONSTANT_ACCESSOR("BOOL_VEC2", GL_BOOL_VEC2);
	CREATE_CONSTANT_ACCESSOR("BOOL_VEC3", GL_BOOL_VEC3);
	CREATE_CONSTANT_ACCESSOR("BOOL_VEC4", GL_BOOL_VEC4);
	CREATE_CONSTANT_ACCESSOR("FLOAT_MAT2", GL_FLOAT_MAT2);
	CREATE_CONSTANT_ACCESSOR("FLOAT_MAT3", GL_FLOAT_MAT3);
	CREATE_CONSTANT_ACCESSOR("FLOAT_MAT4", GL_FLOAT_MAT4);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_1D", GL_SAMPLER_1D);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_2D", GL_SAMPLER_2D);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_3D", GL_SAMPLER_3D);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_CUBE", GL_SAMPLER_CUBE);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_1D_SHADOW", GL_SAMPLER_1D_SHADOW);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_2D_SHADOW", GL_SAMPLER_2D_SHADOW);
	CREATE_CONSTANT_ACCESSOR("DELETE_STATUS", GL_DELETE_STATUS);
	CREATE_CONSTANT_ACCESSOR("COMPILE_STATUS", GL_COMPILE_STATUS);
	CREATE_CONSTANT_ACCESSOR("LINK_STATUS", GL_LINK_STATUS);
	CREATE_CONSTANT_ACCESSOR("VALIDATE_STATUS", GL_VALIDATE_STATUS);
	CREATE_CONSTANT_ACCESSOR("INFO_LOG_LENGTH", GL_INFO_LOG_LENGTH);
	CREATE_CONSTANT_ACCESSOR("ATTACHED_SHADERS", GL_ATTACHED_SHADERS);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_UNIFORMS", GL_ACTIVE_UNIFORMS);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_UNIFORM_MAX_LENGTH", GL_ACTIVE_UNIFORM_MAX_LENGTH);
	CREATE_CONSTANT_ACCESSOR("SHADER_SOURCE_LENGTH", GL_SHADER_SOURCE_LENGTH);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_ATTRIBUTES", GL_ACTIVE_ATTRIBUTES);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_ATTRIBUTE_MAX_LENGTH", GL_ACTIVE_ATTRIBUTE_MAX_LENGTH);
	CREATE_CONSTANT_ACCESSOR("FRAGMENT_SHADER_DERIVATIVE_HINT", GL_FRAGMENT_SHADER_DERIVATIVE_HINT);
	CREATE_CONSTANT_ACCESSOR("SHADING_LANGUAGE_VERSION", GL_SHADING_LANGUAGE_VERSION);
	CREATE_CONSTANT_ACCESSOR("CURRENT_PROGRAM", GL_CURRENT_PROGRAM);
	CREATE_CONSTANT_ACCESSOR("POINT_SPRITE_COORD_ORIGIN", GL_POINT_SPRITE_COORD_ORIGIN);
	CREATE_CONSTANT_ACCESSOR("LOWER_LEFT", GL_LOWER_LEFT);
	CREATE_CONSTANT_ACCESSOR("UPPER_LEFT", GL_UPPER_LEFT);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BACK_REF", GL_STENCIL_BACK_REF);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BACK_VALUE_MASK", GL_STENCIL_BACK_VALUE_MASK);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BACK_WRITEMASK", GL_STENCIL_BACK_WRITEMASK);

	tpl->Set(String::NewFromUtf8(isolate, "blendEquationSeparate"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("blendEquationSeparate requires 2 arguments");
			return;
		}

		GLenum modeRGB = args[0]->Uint32Value();
		GLenum modeAlpha = args[1]->Uint32Value();

		glBlendEquationSeparate(modeRGB, modeAlpha);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "drawBuffers"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("drawBuffers requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLenum* bufs = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			bufs = reinterpret_cast<GLenum*>(bdata);
		} else {
			cout << "ERROR(glDrawBuffers): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glDrawBuffers(n, bufs);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "stencilOpSeparate"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("stencilOpSeparate requires 4 arguments");
			return;
		}

		GLenum face = args[0]->Uint32Value();
		GLenum sfail = args[1]->Uint32Value();
		GLenum dpfail = args[2]->Uint32Value();
		GLenum dppass = args[3]->Uint32Value();

		glStencilOpSeparate(face, sfail, dpfail, dppass);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "stencilFuncSeparate"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("stencilFuncSeparate requires 4 arguments");
			return;
		}

		GLenum frontfunc = args[0]->Uint32Value();
		GLenum backfunc = args[1]->Uint32Value();
		GLint ref = args[2]->Int32Value();
		GLuint mask = args[3]->Uint32Value();

		glStencilFuncSeparate(frontfunc, backfunc, ref, mask);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "stencilMaskSeparate"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("stencilMaskSeparate requires 2 arguments");
			return;
		}

		GLenum face = args[0]->Uint32Value();
		GLuint mask = args[1]->Uint32Value();

		glStencilMaskSeparate(face, mask);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "attachShader"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("attachShader requires 2 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLuint shader = args[1]->Uint32Value();

		glAttachShader(program, shader);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "bindAttribLocation"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("bindAttribLocation requires 3 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();

		GLchar* name = nullptr;
		if (args[2]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[2]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			name = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glBindAttribLocation): array must be of type Int8Array" << endl;
			exit(1);
		}


		glBindAttribLocation(program, index, name);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "compileShader"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("compileShader requires 1 arguments");
			return;
		}

		GLuint shader = args[0]->Uint32Value();

		glCompileShader(shader);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "deleteProgram"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("deleteProgram requires 1 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();

		glDeleteProgram(program);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "deleteShader"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("deleteShader requires 1 arguments");
			return;
		}

		GLuint shader = args[0]->Uint32Value();

		glDeleteShader(shader);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "detachShader"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("detachShader requires 2 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLuint shader = args[1]->Uint32Value();

		glDetachShader(program, shader);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "disableVertexAttribArray"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("disableVertexAttribArray requires 1 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		glDisableVertexAttribArray(index);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "enableVertexAttribArray"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("enableVertexAttribArray requires 1 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		glEnableVertexAttribArray(index);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getActiveAttrib"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("getActiveAttrib requires 7 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();
		GLsizei maxLength = args[2]->Int32Value();

		GLsizei* length = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetActiveAttrib): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLint* size = nullptr;
		if (args[4]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[4]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			size = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetActiveAttrib): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLenum* type = nullptr;
		if (args[5]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[5]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			type = reinterpret_cast<GLenum*>(bdata);
		} else {
			cout << "ERROR(glGetActiveAttrib): array must be of type Uint32Array" << endl;
			exit(1);
		}


		GLchar* name = nullptr;
		if (args[6]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[6]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			name = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetActiveAttrib): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetActiveAttrib(program, index, maxLength, length, size, type, name);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getActiveUniform"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("getActiveUniform requires 7 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();
		GLsizei maxLength = args[2]->Int32Value();

		GLsizei* length = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetActiveUniform): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLint* size = nullptr;
		if (args[4]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[4]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			size = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetActiveUniform): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLenum* type = nullptr;
		if (args[5]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[5]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			type = reinterpret_cast<GLenum*>(bdata);
		} else {
			cout << "ERROR(glGetActiveUniform): array must be of type Uint32Array" << endl;
			exit(1);
		}


		GLchar* name = nullptr;
		if (args[6]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[6]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			name = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetActiveUniform): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetActiveUniform(program, index, maxLength, length, size, type, name);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getAttachedShaders"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getAttachedShaders requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLsizei maxCount = args[1]->Int32Value();

		GLsizei* count = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			count = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetAttachedShaders): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLuint* shaders = nullptr;
		if (args[3]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[3]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			shaders = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGetAttachedShaders): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGetAttachedShaders(program, maxCount, count, shaders);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getProgramiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getProgramiv requires 3 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* param = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			param = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetProgramiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetProgramiv(program, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getProgramInfoLog"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getProgramInfoLog requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLsizei bufSize = args[1]->Int32Value();

		GLsizei* length = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetProgramInfoLog): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLchar* infoLog = nullptr;
		if (args[3]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[3]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			infoLog = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetProgramInfoLog): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetProgramInfoLog(program, bufSize, length, infoLog);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getShaderiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getShaderiv requires 3 arguments");
			return;
		}

		GLuint shader = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* param = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			param = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetShaderiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetShaderiv(shader, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getShaderInfoLog"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getShaderInfoLog requires 4 arguments");
			return;
		}

		GLuint shader = args[0]->Uint32Value();
		GLsizei bufSize = args[1]->Int32Value();

		GLsizei* length = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetShaderInfoLog): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLchar* infoLog = nullptr;
		if (args[3]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[3]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			infoLog = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetShaderInfoLog): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetShaderInfoLog(shader, bufSize, length, infoLog);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "getUniformfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getUniformfv requires 3 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();

		GLfloat* params = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glGetUniformfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glGetUniformfv(program, location, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getUniformiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getUniformiv requires 3 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetUniformiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetUniformiv(program, location, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getVertexAttribdv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getVertexAttribdv requires 3 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLdouble* params = nullptr;
		if (args[2]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[2]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glGetVertexAttribdv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glGetVertexAttribdv(index, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getVertexAttribfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getVertexAttribfv requires 3 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfloat* params = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glGetVertexAttribfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glGetVertexAttribfv(index, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getVertexAttribiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getVertexAttribiv requires 3 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetVertexAttribiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetVertexAttribiv(index, pname, params);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "linkProgram"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("linkProgram requires 1 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();

		glLinkProgram(program);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getShaderSource"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getShaderSource requires 4 arguments");
			return;
		}

		GLuint obj = args[0]->Uint32Value();
		GLsizei maxLength = args[1]->Int32Value();

		GLsizei* length = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetShaderSource): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLchar* source = nullptr;
		if (args[3]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[3]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			source = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetShaderSource): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetShaderSource(obj, maxLength, length, source);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "useProgram"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("useProgram requires 1 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();

		glUseProgram(program);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform1f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("uniform1f requires 2 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLfloat v0 = GLfloat(args[1]->NumberValue());

		glUniform1f(location, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform1fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform1fv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLfloat* value = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glUniform1fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glUniform1fv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform1i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("uniform1i requires 2 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLint v0 = args[1]->Int32Value();

		glUniform1i(location, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform1iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform1iv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLint* value = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glUniform1iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glUniform1iv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform2f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform2f requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLfloat v0 = GLfloat(args[1]->NumberValue());
		GLfloat v1 = GLfloat(args[2]->NumberValue());

		glUniform2f(location, v0, v1);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform2fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform2fv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLfloat* value = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glUniform2fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glUniform2fv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform2i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform2i requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLint v0 = args[1]->Int32Value();
		GLint v1 = args[2]->Int32Value();

		glUniform2i(location, v0, v1);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform2iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform2iv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLint* value = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glUniform2iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glUniform2iv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform3f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniform3f requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLfloat v0 = GLfloat(args[1]->NumberValue());
		GLfloat v1 = GLfloat(args[2]->NumberValue());
		GLfloat v2 = GLfloat(args[3]->NumberValue());

		glUniform3f(location, v0, v1, v2);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform3fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform3fv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLfloat* value = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glUniform3fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glUniform3fv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform3i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniform3i requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLint v0 = args[1]->Int32Value();
		GLint v1 = args[2]->Int32Value();
		GLint v2 = args[3]->Int32Value();

		glUniform3i(location, v0, v1, v2);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform3iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform3iv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLint* value = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glUniform3iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glUniform3iv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform4f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("uniform4f requires 5 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLfloat v0 = GLfloat(args[1]->NumberValue());
		GLfloat v1 = GLfloat(args[2]->NumberValue());
		GLfloat v2 = GLfloat(args[3]->NumberValue());
		GLfloat v3 = GLfloat(args[4]->NumberValue());

		glUniform4f(location, v0, v1, v2, v3);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform4fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform4fv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLfloat* value = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glUniform4fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glUniform4fv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform4i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("uniform4i requires 5 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLint v0 = args[1]->Int32Value();
		GLint v1 = args[2]->Int32Value();
		GLint v2 = args[3]->Int32Value();
		GLint v3 = args[4]->Int32Value();

		glUniform4i(location, v0, v1, v2, v3);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform4iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform4iv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLint* value = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glUniform4iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glUniform4iv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix2fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix2fv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix2fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glUniformMatrix2fv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix3fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix3fv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix3fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glUniformMatrix3fv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix4fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix4fv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix4fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glUniformMatrix4fv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "validateProgram"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("validateProgram requires 1 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();

		glValidateProgram(program);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib1d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib1d requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLdouble x = args[1]->NumberValue();

		glVertexAttrib1d(index, x);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib1dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib1dv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLdouble* v = nullptr;
		if (args[1]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[1]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib1dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glVertexAttrib1dv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib1f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib1f requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLfloat x = GLfloat(args[1]->NumberValue());

		glVertexAttrib1f(index, x);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib1fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib1fv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLfloat* v = nullptr;
		if (args[1]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[1]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib1fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glVertexAttrib1fv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib1s"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib1s requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLshort x = GLshort(args[1]->Int32Value());

		glVertexAttrib1s(index, x);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib1sv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib1sv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLshort* v = nullptr;
		if (args[1]->IsInt16Array()) {
			v8::Local<v8::Int16Array> view = (args[1]).As<v8::Int16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLshort*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib1sv): array must be of type Int16Array" << endl;
			exit(1);
		}


		glVertexAttrib1sv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib2d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("vertexAttrib2d requires 3 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLdouble x = args[1]->NumberValue();
		GLdouble y = args[2]->NumberValue();

		glVertexAttrib2d(index, x, y);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib2dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib2dv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLdouble* v = nullptr;
		if (args[1]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[1]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib2dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glVertexAttrib2dv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib2f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("vertexAttrib2f requires 3 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLfloat x = GLfloat(args[1]->NumberValue());
		GLfloat y = GLfloat(args[2]->NumberValue());

		glVertexAttrib2f(index, x, y);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib2fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib2fv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLfloat* v = nullptr;
		if (args[1]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[1]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib2fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glVertexAttrib2fv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib2s"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("vertexAttrib2s requires 3 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLshort x = GLshort(args[1]->Int32Value());
		GLshort y = GLshort(args[2]->Int32Value());

		glVertexAttrib2s(index, x, y);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib2sv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib2sv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLshort* v = nullptr;
		if (args[1]->IsInt16Array()) {
			v8::Local<v8::Int16Array> view = (args[1]).As<v8::Int16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLshort*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib2sv): array must be of type Int16Array" << endl;
			exit(1);
		}


		glVertexAttrib2sv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib3d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttrib3d requires 4 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLdouble x = args[1]->NumberValue();
		GLdouble y = args[2]->NumberValue();
		GLdouble z = args[3]->NumberValue();

		glVertexAttrib3d(index, x, y, z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib3dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib3dv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLdouble* v = nullptr;
		if (args[1]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[1]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib3dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glVertexAttrib3dv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib3f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttrib3f requires 4 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLfloat x = GLfloat(args[1]->NumberValue());
		GLfloat y = GLfloat(args[2]->NumberValue());
		GLfloat z = GLfloat(args[3]->NumberValue());

		glVertexAttrib3f(index, x, y, z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib3fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib3fv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLfloat* v = nullptr;
		if (args[1]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[1]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib3fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glVertexAttrib3fv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib3s"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttrib3s requires 4 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLshort x = GLshort(args[1]->Int32Value());
		GLshort y = GLshort(args[2]->Int32Value());
		GLshort z = GLshort(args[3]->Int32Value());

		glVertexAttrib3s(index, x, y, z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib3sv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib3sv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLshort* v = nullptr;
		if (args[1]->IsInt16Array()) {
			v8::Local<v8::Int16Array> view = (args[1]).As<v8::Int16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLshort*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib3sv): array must be of type Int16Array" << endl;
			exit(1);
		}


		glVertexAttrib3sv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4Nbv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib4Nbv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLbyte* v = nullptr;
		if (args[1]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[1]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLbyte*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib4Nbv): array must be of type Int8Array" << endl;
			exit(1);
		}


		glVertexAttrib4Nbv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4Niv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib4Niv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLint* v = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib4Niv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glVertexAttrib4Niv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4Nsv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib4Nsv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLshort* v = nullptr;
		if (args[1]->IsInt16Array()) {
			v8::Local<v8::Int16Array> view = (args[1]).As<v8::Int16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLshort*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib4Nsv): array must be of type Int16Array" << endl;
			exit(1);
		}


		glVertexAttrib4Nsv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4Nub"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("vertexAttrib4Nub requires 5 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLubyte x = GLubyte(args[1]->Uint32Value());
		GLubyte y = GLubyte(args[2]->Uint32Value());
		GLubyte z = GLubyte(args[3]->Uint32Value());
		GLubyte w = GLubyte(args[4]->Uint32Value());

		glVertexAttrib4Nub(index, x, y, z, w);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4Nubv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib4Nubv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLubyte* v = nullptr;
		if (args[1]->IsUint8Array()) {
			v8::Local<v8::Uint8Array> view = (args[1]).As<v8::Uint8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLubyte*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib4Nubv): array must be of type Uint8Array" << endl;
			exit(1);
		}


		glVertexAttrib4Nubv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4Nuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib4Nuiv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLuint* v = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib4Nuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glVertexAttrib4Nuiv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4Nusv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib4Nusv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLushort* v = nullptr;
		if (args[1]->IsUint16Array()) {
			v8::Local<v8::Uint16Array> view = (args[1]).As<v8::Uint16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLushort*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib4Nusv): array must be of type Uint16Array" << endl;
			exit(1);
		}


		glVertexAttrib4Nusv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4bv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib4bv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLbyte* v = nullptr;
		if (args[1]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[1]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLbyte*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib4bv): array must be of type Int8Array" << endl;
			exit(1);
		}


		glVertexAttrib4bv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("vertexAttrib4d requires 5 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLdouble x = args[1]->NumberValue();
		GLdouble y = args[2]->NumberValue();
		GLdouble z = args[3]->NumberValue();
		GLdouble w = args[4]->NumberValue();

		glVertexAttrib4d(index, x, y, z, w);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib4dv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLdouble* v = nullptr;
		if (args[1]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[1]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib4dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glVertexAttrib4dv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("vertexAttrib4f requires 5 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLfloat x = GLfloat(args[1]->NumberValue());
		GLfloat y = GLfloat(args[2]->NumberValue());
		GLfloat z = GLfloat(args[3]->NumberValue());
		GLfloat w = GLfloat(args[4]->NumberValue());

		glVertexAttrib4f(index, x, y, z, w);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib4fv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLfloat* v = nullptr;
		if (args[1]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[1]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib4fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glVertexAttrib4fv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib4iv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLint* v = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib4iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glVertexAttrib4iv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4s"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("vertexAttrib4s requires 5 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLshort x = GLshort(args[1]->Int32Value());
		GLshort y = GLshort(args[2]->Int32Value());
		GLshort z = GLshort(args[3]->Int32Value());
		GLshort w = GLshort(args[4]->Int32Value());

		glVertexAttrib4s(index, x, y, z, w);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4sv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib4sv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLshort* v = nullptr;
		if (args[1]->IsInt16Array()) {
			v8::Local<v8::Int16Array> view = (args[1]).As<v8::Int16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLshort*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib4sv): array must be of type Int16Array" << endl;
			exit(1);
		}


		glVertexAttrib4sv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4ubv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib4ubv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLubyte* v = nullptr;
		if (args[1]->IsUint8Array()) {
			v8::Local<v8::Uint8Array> view = (args[1]).As<v8::Uint8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLubyte*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib4ubv): array must be of type Uint8Array" << endl;
			exit(1);
		}


		glVertexAttrib4ubv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib4uiv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLuint* v = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib4uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glVertexAttrib4uiv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttrib4usv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttrib4usv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLushort* v = nullptr;
		if (args[1]->IsUint16Array()) {
			v8::Local<v8::Uint16Array> view = (args[1]).As<v8::Uint16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLushort*>(bdata);
		} else {
			cout << "ERROR(glVertexAttrib4usv): array must be of type Uint16Array" << endl;
			exit(1);
		}


		glVertexAttrib4usv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribPointer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("vertexAttribPointer requires 6 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLint size = args[1]->Int32Value();
		GLenum type = args[2]->Uint32Value();
		GLboolean normalized = GLboolean(args[3]->Uint32Value());
		GLsizei stride = args[4]->Int32Value();

		void* pointer = nullptr;
		if (args[5]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[5]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pointer = reinterpret_cast<void*>(bdata);
		} else if (args[5]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[5]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pointer = reinterpret_cast<void*>(bdata);
		} else if (args[5]->IsNumber()) {
			int value = args[5]->NumberValue();
			pointer = (void*)value;
		} else {
			cout << "ERROR(glVertexAttribPointer): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glVertexAttribPointer(index, size, type, normalized, stride, pointer);
	}));



	/* ------------------------------ GL_VERSION_2_1 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("CURRENT_RASTER_SECONDARY_COLOR", GL_CURRENT_RASTER_SECONDARY_COLOR);
	CREATE_CONSTANT_ACCESSOR("PIXEL_PACK_BUFFER", GL_PIXEL_PACK_BUFFER);
	CREATE_CONSTANT_ACCESSOR("PIXEL_UNPACK_BUFFER", GL_PIXEL_UNPACK_BUFFER);
	CREATE_CONSTANT_ACCESSOR("PIXEL_PACK_BUFFER_BINDING", GL_PIXEL_PACK_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("PIXEL_UNPACK_BUFFER_BINDING", GL_PIXEL_UNPACK_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("FLOAT_MAT2x3", GL_FLOAT_MAT2x3);
	CREATE_CONSTANT_ACCESSOR("FLOAT_MAT2x4", GL_FLOAT_MAT2x4);
	CREATE_CONSTANT_ACCESSOR("FLOAT_MAT3x2", GL_FLOAT_MAT3x2);
	CREATE_CONSTANT_ACCESSOR("FLOAT_MAT3x4", GL_FLOAT_MAT3x4);
	CREATE_CONSTANT_ACCESSOR("FLOAT_MAT4x2", GL_FLOAT_MAT4x2);
	CREATE_CONSTANT_ACCESSOR("FLOAT_MAT4x3", GL_FLOAT_MAT4x3);
	CREATE_CONSTANT_ACCESSOR("SRGB", GL_SRGB);
	CREATE_CONSTANT_ACCESSOR("SRGB8", GL_SRGB8);
	CREATE_CONSTANT_ACCESSOR("SRGB_ALPHA", GL_SRGB_ALPHA);
	CREATE_CONSTANT_ACCESSOR("SRGB8_ALPHA8", GL_SRGB8_ALPHA8);
	CREATE_CONSTANT_ACCESSOR("SLUMINANCE_ALPHA", GL_SLUMINANCE_ALPHA);
	CREATE_CONSTANT_ACCESSOR("SLUMINANCE8_ALPHA8", GL_SLUMINANCE8_ALPHA8);
	CREATE_CONSTANT_ACCESSOR("SLUMINANCE", GL_SLUMINANCE);
	CREATE_CONSTANT_ACCESSOR("SLUMINANCE8", GL_SLUMINANCE8);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_SRGB", GL_COMPRESSED_SRGB);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_SRGB_ALPHA", GL_COMPRESSED_SRGB_ALPHA);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_SLUMINANCE", GL_COMPRESSED_SLUMINANCE);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_SLUMINANCE_ALPHA", GL_COMPRESSED_SLUMINANCE_ALPHA);

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix2x3fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix2x3fv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix2x3fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glUniformMatrix2x3fv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix3x2fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix3x2fv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix3x2fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glUniformMatrix3x2fv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix2x4fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix2x4fv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix2x4fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glUniformMatrix2x4fv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix4x2fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix4x2fv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix4x2fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glUniformMatrix4x2fv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix3x4fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix3x4fv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix3x4fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glUniformMatrix3x4fv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix4x3fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix4x3fv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix4x3fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glUniformMatrix4x3fv(location, count, transpose, value);
	}));



	/* ------------------------------ GL_VERSION_3_0 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("COMPARE_REF_TO_TEXTURE", GL_COMPARE_REF_TO_TEXTURE);
	CREATE_CONSTANT_ACCESSOR("CLIP_DISTANCE0", GL_CLIP_DISTANCE0);
	CREATE_CONSTANT_ACCESSOR("CLIP_DISTANCE1", GL_CLIP_DISTANCE1);
	CREATE_CONSTANT_ACCESSOR("CLIP_DISTANCE2", GL_CLIP_DISTANCE2);
	CREATE_CONSTANT_ACCESSOR("CLIP_DISTANCE3", GL_CLIP_DISTANCE3);
	CREATE_CONSTANT_ACCESSOR("CLIP_DISTANCE4", GL_CLIP_DISTANCE4);
	CREATE_CONSTANT_ACCESSOR("CLIP_DISTANCE5", GL_CLIP_DISTANCE5);
	CREATE_CONSTANT_ACCESSOR("MAX_CLIP_DISTANCES", GL_MAX_CLIP_DISTANCES);
	CREATE_CONSTANT_ACCESSOR("MAJOR_VERSION", GL_MAJOR_VERSION);
	CREATE_CONSTANT_ACCESSOR("MINOR_VERSION", GL_MINOR_VERSION);
	CREATE_CONSTANT_ACCESSOR("NUM_EXTENSIONS", GL_NUM_EXTENSIONS);
	CREATE_CONSTANT_ACCESSOR("CONTEXT_FLAGS", GL_CONTEXT_FLAGS);
	CREATE_CONSTANT_ACCESSOR("DEPTH_BUFFER", GL_DEPTH_BUFFER);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BUFFER", GL_STENCIL_BUFFER);
	CREATE_CONSTANT_ACCESSOR("CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT", GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT);
	CREATE_CONSTANT_ACCESSOR("RGBA32F", GL_RGBA32F);
	CREATE_CONSTANT_ACCESSOR("RGB32F", GL_RGB32F);
	CREATE_CONSTANT_ACCESSOR("RGBA16F", GL_RGBA16F);
	CREATE_CONSTANT_ACCESSOR("RGB16F", GL_RGB16F);
	CREATE_CONSTANT_ACCESSOR("VERTEX_ATTRIB_ARRAY_INTEGER", GL_VERTEX_ATTRIB_ARRAY_INTEGER);
	CREATE_CONSTANT_ACCESSOR("MAX_ARRAY_TEXTURE_LAYERS", GL_MAX_ARRAY_TEXTURE_LAYERS);
	CREATE_CONSTANT_ACCESSOR("MIN_PROGRAM_TEXEL_OFFSET", GL_MIN_PROGRAM_TEXEL_OFFSET);
	CREATE_CONSTANT_ACCESSOR("MAX_PROGRAM_TEXEL_OFFSET", GL_MAX_PROGRAM_TEXEL_OFFSET);
	CREATE_CONSTANT_ACCESSOR("CLAMP_VERTEX_COLOR", GL_CLAMP_VERTEX_COLOR);
	CREATE_CONSTANT_ACCESSOR("CLAMP_FRAGMENT_COLOR", GL_CLAMP_FRAGMENT_COLOR);
	CREATE_CONSTANT_ACCESSOR("CLAMP_READ_COLOR", GL_CLAMP_READ_COLOR);
	CREATE_CONSTANT_ACCESSOR("FIXED_ONLY", GL_FIXED_ONLY);
	CREATE_CONSTANT_ACCESSOR("MAX_VARYING_COMPONENTS", GL_MAX_VARYING_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_RED_TYPE", GL_TEXTURE_RED_TYPE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_GREEN_TYPE", GL_TEXTURE_GREEN_TYPE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BLUE_TYPE", GL_TEXTURE_BLUE_TYPE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_ALPHA_TYPE", GL_TEXTURE_ALPHA_TYPE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_LUMINANCE_TYPE", GL_TEXTURE_LUMINANCE_TYPE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_INTENSITY_TYPE", GL_TEXTURE_INTENSITY_TYPE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_DEPTH_TYPE", GL_TEXTURE_DEPTH_TYPE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_1D_ARRAY", GL_TEXTURE_1D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("PROXY_TEXTURE_1D_ARRAY", GL_PROXY_TEXTURE_1D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_2D_ARRAY", GL_TEXTURE_2D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("PROXY_TEXTURE_2D_ARRAY", GL_PROXY_TEXTURE_2D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BINDING_1D_ARRAY", GL_TEXTURE_BINDING_1D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BINDING_2D_ARRAY", GL_TEXTURE_BINDING_2D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("R11F_G11F_B10F", GL_R11F_G11F_B10F);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_10F_11F_11F_REV", GL_UNSIGNED_INT_10F_11F_11F_REV);
	CREATE_CONSTANT_ACCESSOR("RGB9_E5", GL_RGB9_E5);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_5_9_9_9_REV", GL_UNSIGNED_INT_5_9_9_9_REV);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_SHARED_SIZE", GL_TEXTURE_SHARED_SIZE);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH", GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_BUFFER_MODE", GL_TRANSFORM_FEEDBACK_BUFFER_MODE);
	CREATE_CONSTANT_ACCESSOR("MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS", GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_VARYINGS", GL_TRANSFORM_FEEDBACK_VARYINGS);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_BUFFER_START", GL_TRANSFORM_FEEDBACK_BUFFER_START);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_BUFFER_SIZE", GL_TRANSFORM_FEEDBACK_BUFFER_SIZE);
	CREATE_CONSTANT_ACCESSOR("PRIMITIVES_GENERATED", GL_PRIMITIVES_GENERATED);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN", GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
	CREATE_CONSTANT_ACCESSOR("RASTERIZER_DISCARD", GL_RASTERIZER_DISCARD);
	CREATE_CONSTANT_ACCESSOR("MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS", GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS", GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS);
	CREATE_CONSTANT_ACCESSOR("INTERLEAVED_ATTRIBS", GL_INTERLEAVED_ATTRIBS);
	CREATE_CONSTANT_ACCESSOR("SEPARATE_ATTRIBS", GL_SEPARATE_ATTRIBS);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_BUFFER", GL_TRANSFORM_FEEDBACK_BUFFER);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_BUFFER_BINDING", GL_TRANSFORM_FEEDBACK_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("RGBA32UI", GL_RGBA32UI);
	CREATE_CONSTANT_ACCESSOR("RGB32UI", GL_RGB32UI);
	CREATE_CONSTANT_ACCESSOR("RGBA16UI", GL_RGBA16UI);
	CREATE_CONSTANT_ACCESSOR("RGB16UI", GL_RGB16UI);
	CREATE_CONSTANT_ACCESSOR("RGBA8UI", GL_RGBA8UI);
	CREATE_CONSTANT_ACCESSOR("RGB8UI", GL_RGB8UI);
	CREATE_CONSTANT_ACCESSOR("RGBA32I", GL_RGBA32I);
	CREATE_CONSTANT_ACCESSOR("RGB32I", GL_RGB32I);
	CREATE_CONSTANT_ACCESSOR("RGBA16I", GL_RGBA16I);
	CREATE_CONSTANT_ACCESSOR("RGB16I", GL_RGB16I);
	CREATE_CONSTANT_ACCESSOR("RGBA8I", GL_RGBA8I);
	CREATE_CONSTANT_ACCESSOR("RGB8I", GL_RGB8I);
	CREATE_CONSTANT_ACCESSOR("RED_INTEGER", GL_RED_INTEGER);
	CREATE_CONSTANT_ACCESSOR("GREEN_INTEGER", GL_GREEN_INTEGER);
	CREATE_CONSTANT_ACCESSOR("BLUE_INTEGER", GL_BLUE_INTEGER);
	CREATE_CONSTANT_ACCESSOR("ALPHA_INTEGER", GL_ALPHA_INTEGER);
	CREATE_CONSTANT_ACCESSOR("RGB_INTEGER", GL_RGB_INTEGER);
	CREATE_CONSTANT_ACCESSOR("RGBA_INTEGER", GL_RGBA_INTEGER);
	CREATE_CONSTANT_ACCESSOR("BGR_INTEGER", GL_BGR_INTEGER);
	CREATE_CONSTANT_ACCESSOR("BGRA_INTEGER", GL_BGRA_INTEGER);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_1D_ARRAY", GL_SAMPLER_1D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_2D_ARRAY", GL_SAMPLER_2D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_1D_ARRAY_SHADOW", GL_SAMPLER_1D_ARRAY_SHADOW);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_2D_ARRAY_SHADOW", GL_SAMPLER_2D_ARRAY_SHADOW);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_CUBE_SHADOW", GL_SAMPLER_CUBE_SHADOW);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_VEC2", GL_UNSIGNED_INT_VEC2);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_VEC3", GL_UNSIGNED_INT_VEC3);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_VEC4", GL_UNSIGNED_INT_VEC4);
	CREATE_CONSTANT_ACCESSOR("INT_SAMPLER_1D", GL_INT_SAMPLER_1D);
	CREATE_CONSTANT_ACCESSOR("INT_SAMPLER_2D", GL_INT_SAMPLER_2D);
	CREATE_CONSTANT_ACCESSOR("INT_SAMPLER_3D", GL_INT_SAMPLER_3D);
	CREATE_CONSTANT_ACCESSOR("INT_SAMPLER_CUBE", GL_INT_SAMPLER_CUBE);
	CREATE_CONSTANT_ACCESSOR("INT_SAMPLER_1D_ARRAY", GL_INT_SAMPLER_1D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("INT_SAMPLER_2D_ARRAY", GL_INT_SAMPLER_2D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_SAMPLER_1D", GL_UNSIGNED_INT_SAMPLER_1D);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_SAMPLER_2D", GL_UNSIGNED_INT_SAMPLER_2D);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_SAMPLER_3D", GL_UNSIGNED_INT_SAMPLER_3D);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_SAMPLER_CUBE", GL_UNSIGNED_INT_SAMPLER_CUBE);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_SAMPLER_1D_ARRAY", GL_UNSIGNED_INT_SAMPLER_1D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_SAMPLER_2D_ARRAY", GL_UNSIGNED_INT_SAMPLER_2D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("QUERY_WAIT", GL_QUERY_WAIT);
	CREATE_CONSTANT_ACCESSOR("QUERY_NO_WAIT", GL_QUERY_NO_WAIT);
	CREATE_CONSTANT_ACCESSOR("QUERY_BY_REGION_WAIT", GL_QUERY_BY_REGION_WAIT);
	CREATE_CONSTANT_ACCESSOR("QUERY_BY_REGION_NO_WAIT", GL_QUERY_BY_REGION_NO_WAIT);

	tpl->Set(String::NewFromUtf8(isolate, "colorMaski"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("colorMaski requires 5 arguments");
			return;
		}

		GLuint buf = args[0]->Uint32Value();
		GLboolean red = GLboolean(args[1]->Uint32Value());
		GLboolean green = GLboolean(args[2]->Uint32Value());
		GLboolean blue = GLboolean(args[3]->Uint32Value());
		GLboolean alpha = GLboolean(args[4]->Uint32Value());

		glColorMaski(buf, red, green, blue, alpha);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getBooleani_v"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getBooleani_v requires 3 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();

		GLboolean* data = nullptr;
		if (args[2]->IsUint8Array()) {
			v8::Local<v8::Uint8Array> view = (args[2]).As<v8::Uint8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<GLboolean*>(bdata);
		} else {
			cout << "ERROR(glGetBooleani_v): array must be of type Uint8Array" << endl;
			exit(1);
		}


		glGetBooleani_v(pname, index, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "enablei"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("enablei requires 2 arguments");
			return;
		}

		GLenum cap = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();

		glEnablei(cap, index);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "disablei"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("disablei requires 2 arguments");
			return;
		}

		GLenum cap = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();

		glDisablei(cap, index);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "beginTransformFeedback"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("beginTransformFeedback requires 1 arguments");
			return;
		}

		GLenum primitiveMode = args[0]->Uint32Value();

		glBeginTransformFeedback(primitiveMode);
	}));




	tpl->Set(String::NewFromUtf8(isolate, "clampColor"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("clampColor requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum clamp = args[1]->Uint32Value();

		glClampColor(target, clamp);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "beginConditionalRender"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("beginConditionalRender requires 2 arguments");
			return;
		}

		GLuint id = args[0]->Uint32Value();
		GLenum mode = args[1]->Uint32Value();

		glBeginConditionalRender(id, mode);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI1i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribI1i requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLint v0 = args[1]->Int32Value();

		glVertexAttribI1i(index, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI2i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("vertexAttribI2i requires 3 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLint v0 = args[1]->Int32Value();
		GLint v1 = args[2]->Int32Value();

		glVertexAttribI2i(index, v0, v1);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI3i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttribI3i requires 4 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLint v0 = args[1]->Int32Value();
		GLint v1 = args[2]->Int32Value();
		GLint v2 = args[3]->Int32Value();

		glVertexAttribI3i(index, v0, v1, v2);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI4i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("vertexAttribI4i requires 5 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLint v0 = args[1]->Int32Value();
		GLint v1 = args[2]->Int32Value();
		GLint v2 = args[3]->Int32Value();
		GLint v3 = args[4]->Int32Value();

		glVertexAttribI4i(index, v0, v1, v2, v3);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI1ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribI1ui requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLuint v0 = args[1]->Uint32Value();

		glVertexAttribI1ui(index, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI2ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("vertexAttribI2ui requires 3 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLuint v0 = args[1]->Uint32Value();
		GLuint v1 = args[2]->Uint32Value();

		glVertexAttribI2ui(index, v0, v1);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI3ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttribI3ui requires 4 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLuint v0 = args[1]->Uint32Value();
		GLuint v1 = args[2]->Uint32Value();
		GLuint v2 = args[3]->Uint32Value();

		glVertexAttribI3ui(index, v0, v1, v2);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI4ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("vertexAttribI4ui requires 5 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLuint v0 = args[1]->Uint32Value();
		GLuint v1 = args[2]->Uint32Value();
		GLuint v2 = args[3]->Uint32Value();
		GLuint v3 = args[4]->Uint32Value();

		glVertexAttribI4ui(index, v0, v1, v2, v3);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI1iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribI1iv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLint* v0 = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v0 = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribI1iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glVertexAttribI1iv(index, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI2iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribI2iv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLint* v0 = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v0 = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribI2iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glVertexAttribI2iv(index, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI3iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribI3iv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLint* v0 = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v0 = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribI3iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glVertexAttribI3iv(index, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI4iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribI4iv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLint* v0 = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v0 = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribI4iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glVertexAttribI4iv(index, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI1uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribI1uiv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLuint* v0 = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v0 = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribI1uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glVertexAttribI1uiv(index, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI2uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribI2uiv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLuint* v0 = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v0 = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribI2uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glVertexAttribI2uiv(index, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI3uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribI3uiv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLuint* v0 = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v0 = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribI3uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glVertexAttribI3uiv(index, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI4uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribI4uiv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLuint* v0 = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v0 = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribI4uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glVertexAttribI4uiv(index, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI4bv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribI4bv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLbyte* v0 = nullptr;
		if (args[1]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[1]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v0 = reinterpret_cast<GLbyte*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribI4bv): array must be of type Int8Array" << endl;
			exit(1);
		}


		glVertexAttribI4bv(index, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI4sv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribI4sv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLshort* v0 = nullptr;
		if (args[1]->IsInt16Array()) {
			v8::Local<v8::Int16Array> view = (args[1]).As<v8::Int16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v0 = reinterpret_cast<GLshort*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribI4sv): array must be of type Int16Array" << endl;
			exit(1);
		}


		glVertexAttribI4sv(index, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI4ubv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribI4ubv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLubyte* v0 = nullptr;
		if (args[1]->IsUint8Array()) {
			v8::Local<v8::Uint8Array> view = (args[1]).As<v8::Uint8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v0 = reinterpret_cast<GLubyte*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribI4ubv): array must be of type Uint8Array" << endl;
			exit(1);
		}


		glVertexAttribI4ubv(index, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribI4usv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribI4usv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLushort* v0 = nullptr;
		if (args[1]->IsUint16Array()) {
			v8::Local<v8::Uint16Array> view = (args[1]).As<v8::Uint16Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v0 = reinterpret_cast<GLushort*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribI4usv): array must be of type Uint16Array" << endl;
			exit(1);
		}


		glVertexAttribI4usv(index, v0);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "getVertexAttribIiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getVertexAttribIiv requires 3 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetVertexAttribIiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetVertexAttribIiv(index, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getVertexAttribIuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getVertexAttribIuiv requires 3 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLuint* params = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGetVertexAttribIuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGetVertexAttribIuiv(index, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getUniformuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getUniformuiv requires 3 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();

		GLuint* params = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGetUniformuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGetUniformuiv(program, location, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "bindFragDataLocation"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("bindFragDataLocation requires 3 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLuint colorNumber = args[1]->Uint32Value();

		GLchar* name = nullptr;
		if (args[2]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[2]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			name = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glBindFragDataLocation): array must be of type Int8Array" << endl;
			exit(1);
		}


		glBindFragDataLocation(program, colorNumber, name);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform1ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("uniform1ui requires 2 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLuint v0 = args[1]->Uint32Value();

		glUniform1ui(location, v0);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform2ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform2ui requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLuint v0 = args[1]->Uint32Value();
		GLuint v1 = args[2]->Uint32Value();

		glUniform2ui(location, v0, v1);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform3ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniform3ui requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLuint v0 = args[1]->Uint32Value();
		GLuint v1 = args[2]->Uint32Value();
		GLuint v2 = args[3]->Uint32Value();

		glUniform3ui(location, v0, v1, v2);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform4ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("uniform4ui requires 5 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLuint v0 = args[1]->Uint32Value();
		GLuint v1 = args[2]->Uint32Value();
		GLuint v2 = args[3]->Uint32Value();
		GLuint v3 = args[4]->Uint32Value();

		glUniform4ui(location, v0, v1, v2, v3);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform1uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform1uiv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLuint* value = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glUniform1uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glUniform1uiv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform2uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform2uiv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLuint* value = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glUniform2uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glUniform2uiv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform3uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform3uiv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLuint* value = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glUniform3uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glUniform3uiv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform4uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform4uiv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLuint* value = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glUniform4uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glUniform4uiv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texParameterIiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("texParameterIiv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glTexParameterIiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glTexParameterIiv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texParameterIuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("texParameterIuiv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLuint* params = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glTexParameterIuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glTexParameterIuiv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getTexParameterIiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getTexParameterIiv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetTexParameterIiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetTexParameterIiv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getTexParameterIuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getTexParameterIuiv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLuint* params = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGetTexParameterIuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGetTexParameterIuiv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "clearBufferiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("clearBufferiv requires 3 arguments");
			return;
		}

		GLenum buffer = args[0]->Uint32Value();
		GLint drawBuffer = args[1]->Int32Value();

		GLint* value = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glClearBufferiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glClearBufferiv(buffer, drawBuffer, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "clearBufferuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("clearBufferuiv requires 3 arguments");
			return;
		}

		GLenum buffer = args[0]->Uint32Value();
		GLint drawBuffer = args[1]->Int32Value();

		GLuint* value = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glClearBufferuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glClearBufferuiv(buffer, drawBuffer, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "clearBufferfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("clearBufferfv requires 3 arguments");
			return;
		}

		GLenum buffer = args[0]->Uint32Value();
		GLint drawBuffer = args[1]->Int32Value();

		GLfloat* value = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glClearBufferfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glClearBufferfv(buffer, drawBuffer, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "clearBufferfi"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("clearBufferfi requires 4 arguments");
			return;
		}

		GLenum buffer = args[0]->Uint32Value();
		GLint drawBuffer = args[1]->Int32Value();
		GLfloat depth = GLfloat(args[2]->NumberValue());
		GLint stencil = args[3]->Int32Value();

		glClearBufferfi(buffer, drawBuffer, depth, stencil);
	}));



	/* ------------------------------ GL_VERSION_3_1 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("SAMPLER_2D_RECT", GL_SAMPLER_2D_RECT);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_2D_RECT_SHADOW", GL_SAMPLER_2D_RECT_SHADOW);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_BUFFER", GL_SAMPLER_BUFFER);
	CREATE_CONSTANT_ACCESSOR("INT_SAMPLER_2D_RECT", GL_INT_SAMPLER_2D_RECT);
	CREATE_CONSTANT_ACCESSOR("INT_SAMPLER_BUFFER", GL_INT_SAMPLER_BUFFER);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_SAMPLER_2D_RECT", GL_UNSIGNED_INT_SAMPLER_2D_RECT);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_SAMPLER_BUFFER", GL_UNSIGNED_INT_SAMPLER_BUFFER);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BUFFER", GL_TEXTURE_BUFFER);
	CREATE_CONSTANT_ACCESSOR("MAX_TEXTURE_BUFFER_SIZE", GL_MAX_TEXTURE_BUFFER_SIZE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BINDING_BUFFER", GL_TEXTURE_BINDING_BUFFER);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BUFFER_DATA_STORE_BINDING", GL_TEXTURE_BUFFER_DATA_STORE_BINDING);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BUFFER_FORMAT", GL_TEXTURE_BUFFER_FORMAT);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_RECTANGLE", GL_TEXTURE_RECTANGLE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BINDING_RECTANGLE", GL_TEXTURE_BINDING_RECTANGLE);
	CREATE_CONSTANT_ACCESSOR("PROXY_TEXTURE_RECTANGLE", GL_PROXY_TEXTURE_RECTANGLE);
	CREATE_CONSTANT_ACCESSOR("MAX_RECTANGLE_TEXTURE_SIZE", GL_MAX_RECTANGLE_TEXTURE_SIZE);
	CREATE_CONSTANT_ACCESSOR("RED_SNORM", GL_RED_SNORM);
	CREATE_CONSTANT_ACCESSOR("RG_SNORM", GL_RG_SNORM);
	CREATE_CONSTANT_ACCESSOR("RGB_SNORM", GL_RGB_SNORM);
	CREATE_CONSTANT_ACCESSOR("RGBA_SNORM", GL_RGBA_SNORM);
	CREATE_CONSTANT_ACCESSOR("R8_SNORM", GL_R8_SNORM);
	CREATE_CONSTANT_ACCESSOR("RG8_SNORM", GL_RG8_SNORM);
	CREATE_CONSTANT_ACCESSOR("RGB8_SNORM", GL_RGB8_SNORM);
	CREATE_CONSTANT_ACCESSOR("RGBA8_SNORM", GL_RGBA8_SNORM);
	CREATE_CONSTANT_ACCESSOR("R16_SNORM", GL_R16_SNORM);
	CREATE_CONSTANT_ACCESSOR("RG16_SNORM", GL_RG16_SNORM);
	CREATE_CONSTANT_ACCESSOR("RGB16_SNORM", GL_RGB16_SNORM);
	CREATE_CONSTANT_ACCESSOR("RGBA16_SNORM", GL_RGBA16_SNORM);
	CREATE_CONSTANT_ACCESSOR("SIGNED_NORMALIZED", GL_SIGNED_NORMALIZED);
	CREATE_CONSTANT_ACCESSOR("PRIMITIVE_RESTART", GL_PRIMITIVE_RESTART);
	CREATE_CONSTANT_ACCESSOR("PRIMITIVE_RESTART_INDEX", GL_PRIMITIVE_RESTART_INDEX);
	CREATE_CONSTANT_ACCESSOR("BUFFER_ACCESS_FLAGS", GL_BUFFER_ACCESS_FLAGS);
	CREATE_CONSTANT_ACCESSOR("BUFFER_MAP_LENGTH", GL_BUFFER_MAP_LENGTH);
	CREATE_CONSTANT_ACCESSOR("BUFFER_MAP_OFFSET", GL_BUFFER_MAP_OFFSET);

	tpl->Set(String::NewFromUtf8(isolate, "drawArraysInstanced"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("drawArraysInstanced requires 4 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLint first = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLsizei primcount = args[3]->Int32Value();

		glDrawArraysInstanced(mode, first, count, primcount);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "drawElementsInstanced"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("drawElementsInstanced requires 5 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLsizei count = args[1]->Int32Value();
		GLenum type = args[2]->Uint32Value();

		void* indices = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glDrawElementsInstanced): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLsizei primcount = args[4]->Int32Value();

		glDrawElementsInstanced(mode, count, type, indices, primcount);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texBuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("texBuffer requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum internalFormat = args[1]->Uint32Value();
		GLuint buffer = args[2]->Uint32Value();

		glTexBuffer(target, internalFormat, buffer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "primitiveRestartIndex"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("primitiveRestartIndex requires 1 arguments");
			return;
		}

		GLuint buffer = args[0]->Uint32Value();

		glPrimitiveRestartIndex(buffer);
	}));



	/* ------------------------------ GL_VERSION_3_2 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("CONTEXT_CORE_PROFILE_BIT", GL_CONTEXT_CORE_PROFILE_BIT);
	CREATE_CONSTANT_ACCESSOR("CONTEXT_COMPATIBILITY_PROFILE_BIT", GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);
	CREATE_CONSTANT_ACCESSOR("LINES_ADJACENCY", GL_LINES_ADJACENCY);
	CREATE_CONSTANT_ACCESSOR("LINE_STRIP_ADJACENCY", GL_LINE_STRIP_ADJACENCY);
	CREATE_CONSTANT_ACCESSOR("TRIANGLES_ADJACENCY", GL_TRIANGLES_ADJACENCY);
	CREATE_CONSTANT_ACCESSOR("TRIANGLE_STRIP_ADJACENCY", GL_TRIANGLE_STRIP_ADJACENCY);
	CREATE_CONSTANT_ACCESSOR("PROGRAM_POINT_SIZE", GL_PROGRAM_POINT_SIZE);
	CREATE_CONSTANT_ACCESSOR("MAX_GEOMETRY_TEXTURE_IMAGE_UNITS", GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_LAYERED", GL_FRAMEBUFFER_ATTACHMENT_LAYERED);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS", GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS);
	CREATE_CONSTANT_ACCESSOR("GEOMETRY_SHADER", GL_GEOMETRY_SHADER);
	CREATE_CONSTANT_ACCESSOR("GEOMETRY_VERTICES_OUT", GL_GEOMETRY_VERTICES_OUT);
	CREATE_CONSTANT_ACCESSOR("GEOMETRY_INPUT_TYPE", GL_GEOMETRY_INPUT_TYPE);
	CREATE_CONSTANT_ACCESSOR("GEOMETRY_OUTPUT_TYPE", GL_GEOMETRY_OUTPUT_TYPE);
	CREATE_CONSTANT_ACCESSOR("MAX_GEOMETRY_UNIFORM_COMPONENTS", GL_MAX_GEOMETRY_UNIFORM_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_GEOMETRY_OUTPUT_VERTICES", GL_MAX_GEOMETRY_OUTPUT_VERTICES);
	CREATE_CONSTANT_ACCESSOR("MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS", GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_OUTPUT_COMPONENTS", GL_MAX_VERTEX_OUTPUT_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_GEOMETRY_INPUT_COMPONENTS", GL_MAX_GEOMETRY_INPUT_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_GEOMETRY_OUTPUT_COMPONENTS", GL_MAX_GEOMETRY_OUTPUT_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_FRAGMENT_INPUT_COMPONENTS", GL_MAX_FRAGMENT_INPUT_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("CONTEXT_PROFILE_MASK", GL_CONTEXT_PROFILE_MASK);



	tpl->Set(String::NewFromUtf8(isolate, "framebufferTexture"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("framebufferTexture requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum attachment = args[1]->Uint32Value();
		GLuint texture = args[2]->Uint32Value();
		GLint level = args[3]->Int32Value();

		glFramebufferTexture(target, attachment, texture, level);
	}));



	/* ------------------------------ GL_VERSION_3_3 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("RGB10_A2UI", GL_RGB10_A2UI);
	CREATE_CONSTANT_ACCESSOR("VERTEX_ATTRIB_ARRAY_DIVISOR", GL_VERTEX_ATTRIB_ARRAY_DIVISOR);

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribDivisor"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribDivisor requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLuint divisor = args[1]->Uint32Value();

		glVertexAttribDivisor(index, divisor);
	}));



	/* ------------------------------ GL_VERSION_4_0 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("SAMPLE_SHADING", GL_SAMPLE_SHADING);
	CREATE_CONSTANT_ACCESSOR("MIN_SAMPLE_SHADING_VALUE", GL_MIN_SAMPLE_SHADING_VALUE);
	CREATE_CONSTANT_ACCESSOR("MIN_PROGRAM_TEXTURE_GATHER_OFFSET", GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET);
	CREATE_CONSTANT_ACCESSOR("MAX_PROGRAM_TEXTURE_GATHER_OFFSET", GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET);
	CREATE_CONSTANT_ACCESSOR("MAX_PROGRAM_TEXTURE_GATHER_COMPONENTS", GL_MAX_PROGRAM_TEXTURE_GATHER_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_CUBE_MAP_ARRAY", GL_TEXTURE_CUBE_MAP_ARRAY);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BINDING_CUBE_MAP_ARRAY", GL_TEXTURE_BINDING_CUBE_MAP_ARRAY);
	CREATE_CONSTANT_ACCESSOR("PROXY_TEXTURE_CUBE_MAP_ARRAY", GL_PROXY_TEXTURE_CUBE_MAP_ARRAY);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_CUBE_MAP_ARRAY", GL_SAMPLER_CUBE_MAP_ARRAY);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_CUBE_MAP_ARRAY_SHADOW", GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW);
	CREATE_CONSTANT_ACCESSOR("INT_SAMPLER_CUBE_MAP_ARRAY", GL_INT_SAMPLER_CUBE_MAP_ARRAY);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY", GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY);

	tpl->Set(String::NewFromUtf8(isolate, "minSampleShading"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("minSampleShading requires 1 arguments");
			return;
		}

		GLclampf value = GLclampf(args[0]->NumberValue());

		glMinSampleShading(value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "blendEquationSeparatei"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("blendEquationSeparatei requires 3 arguments");
			return;
		}

		GLuint buf = args[0]->Uint32Value();
		GLenum modeRGB = args[1]->Uint32Value();
		GLenum modeAlpha = args[2]->Uint32Value();

		glBlendEquationSeparatei(buf, modeRGB, modeAlpha);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "blendEquationi"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("blendEquationi requires 2 arguments");
			return;
		}

		GLuint buf = args[0]->Uint32Value();
		GLenum mode = args[1]->Uint32Value();

		glBlendEquationi(buf, mode);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "blendFuncSeparatei"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("blendFuncSeparatei requires 5 arguments");
			return;
		}

		GLuint buf = args[0]->Uint32Value();
		GLenum srcRGB = args[1]->Uint32Value();
		GLenum dstRGB = args[2]->Uint32Value();
		GLenum srcAlpha = args[3]->Uint32Value();
		GLenum dstAlpha = args[4]->Uint32Value();

		glBlendFuncSeparatei(buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "blendFunci"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("blendFunci requires 3 arguments");
			return;
		}

		GLuint buf = args[0]->Uint32Value();
		GLenum src = args[1]->Uint32Value();
		GLenum dst = args[2]->Uint32Value();

		glBlendFunci(buf, src, dst);
	}));



	// empty / skipped / ignored: GL_VERSION_4_1
	/* ------------------------------ GL_VERSION_4_2 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("COPY_READ_BUFFER_BINDING", GL_COPY_READ_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("COPY_WRITE_BUFFER_BINDING", GL_COPY_WRITE_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_ACTIVE", GL_TRANSFORM_FEEDBACK_ACTIVE);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_PAUSED", GL_TRANSFORM_FEEDBACK_PAUSED);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGBA_BPTC_UNORM", GL_COMPRESSED_RGBA_BPTC_UNORM);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_SRGB_ALPHA_BPTC_UNORM", GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGB_BPTC_SIGNED_FLOAT", GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT", GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT);



	/* ------------------------------ GL_VERSION_4_3 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("VERTEX_ATTRIB_ARRAY_LONG", GL_VERTEX_ATTRIB_ARRAY_LONG);
	CREATE_CONSTANT_ACCESSOR("NUM_SHADING_LANGUAGE_VERSIONS", GL_NUM_SHADING_LANGUAGE_VERSIONS);



	/* ------------------------------ GL_VERSION_4_4 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_ATTRIB_STRIDE", GL_MAX_VERTEX_ATTRIB_STRIDE);
	CREATE_CONSTANT_ACCESSOR("PRIMITIVE_RESTART_FOR_PATCHES_SUPPORTED", GL_PRIMITIVE_RESTART_FOR_PATCHES_SUPPORTED);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BUFFER_BINDING", GL_TEXTURE_BUFFER_BINDING);



	/* ------------------------------ GL_VERSION_4_5 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("CONTEXT_FLAG_ROBUST_ACCESS_BIT", GL_CONTEXT_FLAG_ROBUST_ACCESS_BIT);

	tpl->Set(String::NewFromUtf8(isolate, "getnTexImage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("getnTexImage requires 6 arguments");
			return;
		}

		GLenum tex = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLenum format = args[2]->Uint32Value();
		GLenum type = args[3]->Uint32Value();
		GLsizei bufSize = args[4]->Int32Value();

		GLvoid* pixels = nullptr;
		if (args[5]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[5]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pixels = reinterpret_cast<GLvoid*>(bdata);
		} else if (args[5]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[5]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pixels = reinterpret_cast<GLvoid*>(bdata);
		} else {
			cout << "ERROR(glGetnTexImage): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glGetnTexImage(tex, level, format, type, bufSize, pixels);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getnCompressedTexImage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getnCompressedTexImage requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint lod = args[1]->Int32Value();
		GLsizei bufSize = args[2]->Int32Value();

		GLvoid* pixels = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pixels = reinterpret_cast<GLvoid*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pixels = reinterpret_cast<GLvoid*>(bdata);
		} else {
			cout << "ERROR(glGetnCompressedTexImage): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glGetnCompressedTexImage(target, lod, bufSize, pixels);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getnUniformdv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getnUniformdv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei bufSize = args[2]->Int32Value();

		GLdouble* params = nullptr;
		if (args[3]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[3]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glGetnUniformdv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glGetnUniformdv(program, location, bufSize, params);
	}));



	/* ------------------------------ GL_VERSION_4_6 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("PARAMETER_BUFFER", GL_PARAMETER_BUFFER);
	CREATE_CONSTANT_ACCESSOR("PARAMETER_BUFFER_BINDING", GL_PARAMETER_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("VERTICES_SUBMITTED", GL_VERTICES_SUBMITTED);
	CREATE_CONSTANT_ACCESSOR("PRIMITIVES_SUBMITTED", GL_PRIMITIVES_SUBMITTED);
	CREATE_CONSTANT_ACCESSOR("VERTEX_SHADER_INVOCATIONS", GL_VERTEX_SHADER_INVOCATIONS);
	CREATE_CONSTANT_ACCESSOR("TESS_CONTROL_SHADER_PATCHES", GL_TESS_CONTROL_SHADER_PATCHES);
	CREATE_CONSTANT_ACCESSOR("TESS_EVALUATION_SHADER_INVOCATIONS", GL_TESS_EVALUATION_SHADER_INVOCATIONS);
	CREATE_CONSTANT_ACCESSOR("GEOMETRY_SHADER_PRIMITIVES_EMITTED", GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED);
	CREATE_CONSTANT_ACCESSOR("FRAGMENT_SHADER_INVOCATIONS", GL_FRAGMENT_SHADER_INVOCATIONS);
	CREATE_CONSTANT_ACCESSOR("COMPUTE_SHADER_INVOCATIONS", GL_COMPUTE_SHADER_INVOCATIONS);
	CREATE_CONSTANT_ACCESSOR("CLIPPING_INPUT_PRIMITIVES", GL_CLIPPING_INPUT_PRIMITIVES);
	CREATE_CONSTANT_ACCESSOR("CLIPPING_OUTPUT_PRIMITIVES", GL_CLIPPING_OUTPUT_PRIMITIVES);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_OVERFLOW", GL_TRANSFORM_FEEDBACK_OVERFLOW);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_STREAM_OVERFLOW", GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW);
	CREATE_CONSTANT_ACCESSOR("POLYGON_OFFSET_CLAMP", GL_POLYGON_OFFSET_CLAMP);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_MAX_ANISOTROPY", GL_TEXTURE_MAX_ANISOTROPY);
	CREATE_CONSTANT_ACCESSOR("MAX_TEXTURE_MAX_ANISOTROPY", GL_MAX_TEXTURE_MAX_ANISOTROPY);
	CREATE_CONSTANT_ACCESSOR("SHADER_BINARY_FORMAT_SPIR_V", GL_SHADER_BINARY_FORMAT_SPIR_V);
	CREATE_CONSTANT_ACCESSOR("SPIR_V_BINARY", GL_SPIR_V_BINARY);
	CREATE_CONSTANT_ACCESSOR("CONTEXT_FLAG_NO_ERROR_BIT", GL_CONTEXT_FLAG_NO_ERROR_BIT);
	CREATE_CONSTANT_ACCESSOR("SPIR_V_EXTENSIONS", GL_SPIR_V_EXTENSIONS);
	CREATE_CONSTANT_ACCESSOR("NUM_SPIR_V_EXTENSIONS", GL_NUM_SPIR_V_EXTENSIONS);

	tpl->Set(String::NewFromUtf8(isolate, "multiDrawArraysIndirectCount"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("multiDrawArraysIndirectCount requires 5 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();

		GLvoid* indirect = nullptr;
		if (args[1]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[1]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			indirect = reinterpret_cast<GLvoid*>(bdata);
		} else if (args[1]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[1]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indirect = reinterpret_cast<GLvoid*>(bdata);
		} else {
			cout << "ERROR(glMultiDrawArraysIndirectCount): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLintptr drawcount = GLintptr(args[2]->Int32Value());
		GLsizei maxdrawcount = args[3]->Int32Value();
		GLsizei stride = args[4]->Int32Value();

		glMultiDrawArraysIndirectCount(mode, indirect, drawcount, maxdrawcount, stride);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiDrawElementsIndirectCount"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("multiDrawElementsIndirectCount requires 6 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();

		GLvoid* indirect = nullptr;
		if (args[2]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[2]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			indirect = reinterpret_cast<GLvoid*>(bdata);
		} else if (args[2]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[2]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indirect = reinterpret_cast<GLvoid*>(bdata);
		} else {
			cout << "ERROR(glMultiDrawElementsIndirectCount): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLintptr drawcount = GLintptr(args[3]->Int32Value());
		GLsizei maxdrawcount = args[4]->Int32Value();
		GLsizei stride = args[5]->Int32Value();

		glMultiDrawElementsIndirectCount(mode, type, indirect, drawcount, maxdrawcount, stride);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "specializeShader"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("specializeShader requires 5 arguments");
			return;
		}

		GLuint shader = args[0]->Uint32Value();

		GLchar* pEntryPoint = nullptr;
		if (args[1]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[1]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pEntryPoint = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glSpecializeShader): array must be of type Int8Array" << endl;
			exit(1);
		}

		GLuint numSpecializationConstants = args[2]->Uint32Value();

		GLuint* pConstantIndex = nullptr;
		if (args[3]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[3]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pConstantIndex = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glSpecializeShader): array must be of type Uint32Array" << endl;
			exit(1);
		}


		GLuint* pConstantValue = nullptr;
		if (args[4]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[4]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pConstantValue = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glSpecializeShader): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glSpecializeShader(shader, pEntryPoint, numSpecializationConstants, pConstantIndex, pConstantValue);
	}));



	// empty / skipped / ignored: GL_3DFX_multisample
	// empty / skipped / ignored: GL_3DFX_tbuffer
	// empty / skipped / ignored: GL_3DFX_texture_compression_FXT1
	// empty / skipped / ignored: GL_AMD_blend_minmax_factor
	// empty / skipped / ignored: GL_AMD_compressed_3DC_texture
	// empty / skipped / ignored: GL_AMD_compressed_ATC_texture
	// empty / skipped / ignored: GL_AMD_conservative_depth
	// empty / skipped / ignored: GL_AMD_debug_output
	// empty / skipped / ignored: GL_AMD_depth_clamp_separate
	// empty / skipped / ignored: GL_AMD_draw_buffers_blend
	// empty / skipped / ignored: GL_AMD_framebuffer_sample_positions
	// empty / skipped / ignored: GL_AMD_gcn_shader
	// empty / skipped / ignored: GL_AMD_gpu_shader_half_float
	// empty / skipped / ignored: GL_AMD_gpu_shader_int16
	// empty / skipped / ignored: GL_AMD_gpu_shader_int64
	/* ------------------------------ GL_AMD_interleaved_elements ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("RED", GL_RED);
	CREATE_CONSTANT_ACCESSOR("GREEN", GL_GREEN);
	CREATE_CONSTANT_ACCESSOR("BLUE", GL_BLUE);
	CREATE_CONSTANT_ACCESSOR("ALPHA", GL_ALPHA);
	CREATE_CONSTANT_ACCESSOR("RG8UI", GL_RG8UI);
	CREATE_CONSTANT_ACCESSOR("RG16UI", GL_RG16UI);
	CREATE_CONSTANT_ACCESSOR("RGBA8UI", GL_RGBA8UI);



	// empty / skipped / ignored: GL_AMD_multi_draw_indirect
	// empty / skipped / ignored: GL_AMD_name_gen_delete
	// empty / skipped / ignored: GL_AMD_occlusion_query_event
	// empty / skipped / ignored: GL_AMD_performance_monitor
	// empty / skipped / ignored: GL_AMD_pinned_memory
	// empty / skipped / ignored: GL_AMD_program_binary_Z400
	// empty / skipped / ignored: GL_AMD_query_buffer_object
	// empty / skipped / ignored: GL_AMD_sample_positions
	/* ------------------------------ GL_AMD_seamless_cubemap_per_texture ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_CUBE_MAP_SEAMLESS", GL_TEXTURE_CUBE_MAP_SEAMLESS);



	// empty / skipped / ignored: GL_AMD_shader_atomic_counter_ops
	// empty / skipped / ignored: GL_AMD_shader_ballot
	// empty / skipped / ignored: GL_AMD_shader_explicit_vertex_parameter
	// empty / skipped / ignored: GL_AMD_shader_stencil_export
	// empty / skipped / ignored: GL_AMD_shader_stencil_value_export
	// empty / skipped / ignored: GL_AMD_shader_trinary_minmax
	/* ------------------------------ GL_AMD_sparse_texture ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("MAX_SPARSE_ARRAY_TEXTURE_LAYERS", GL_MAX_SPARSE_ARRAY_TEXTURE_LAYERS);



	// empty / skipped / ignored: GL_AMD_stencil_operation_extended
	// empty / skipped / ignored: GL_AMD_texture_gather_bias_lod
	// empty / skipped / ignored: GL_AMD_texture_texture4
	// empty / skipped / ignored: GL_AMD_transform_feedback3_lines_triangles
	// empty / skipped / ignored: GL_AMD_transform_feedback4
	// empty / skipped / ignored: GL_AMD_vertex_shader_layer
	// empty / skipped / ignored: GL_AMD_vertex_shader_tessellator
	// empty / skipped / ignored: GL_AMD_vertex_shader_viewport_index
	// empty / skipped / ignored: GL_ANDROID_extension_pack_es31a
	// empty / skipped / ignored: GL_ANGLE_depth_texture
	/* ------------------------------ GL_ANGLE_framebuffer_blit ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("DRAW_FRAMEBUFFER_BINDING_ANGLE", GL_DRAW_FRAMEBUFFER_BINDING_ANGLE);
	CREATE_CONSTANT_ACCESSOR("READ_FRAMEBUFFER_ANGLE", GL_READ_FRAMEBUFFER_ANGLE);
	CREATE_CONSTANT_ACCESSOR("DRAW_FRAMEBUFFER_ANGLE", GL_DRAW_FRAMEBUFFER_ANGLE);
	CREATE_CONSTANT_ACCESSOR("READ_FRAMEBUFFER_BINDING_ANGLE", GL_READ_FRAMEBUFFER_BINDING_ANGLE);

	tpl->Set(String::NewFromUtf8(isolate, "blitFramebufferANGLE"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 10) {
			V8Helper::_instance->throwException("blitFramebufferANGLE requires 10 arguments");
			return;
		}

		GLint srcX0 = args[0]->Int32Value();
		GLint srcY0 = args[1]->Int32Value();
		GLint srcX1 = args[2]->Int32Value();
		GLint srcY1 = args[3]->Int32Value();
		GLint dstX0 = args[4]->Int32Value();
		GLint dstY0 = args[5]->Int32Value();
		GLint dstX1 = args[6]->Int32Value();
		GLint dstY1 = args[7]->Int32Value();
		GLbitfield mask = args[8]->Uint32Value();
		GLenum filter = args[9]->Uint32Value();

		glBlitFramebufferANGLE(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
	}));



	/* ------------------------------ GL_ANGLE_framebuffer_multisample ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("RENDERBUFFER_SAMPLES_ANGLE", GL_RENDERBUFFER_SAMPLES_ANGLE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_ANGLE", GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_ANGLE);
	CREATE_CONSTANT_ACCESSOR("MAX_SAMPLES_ANGLE", GL_MAX_SAMPLES_ANGLE);

	tpl->Set(String::NewFromUtf8(isolate, "renderbufferStorageMultisampleANGLE"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("renderbufferStorageMultisampleANGLE requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei samples = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();

		glRenderbufferStorageMultisampleANGLE(target, samples, internalformat, width, height);
	}));



	/* ------------------------------ GL_ANGLE_instanced_arrays ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE", GL_VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE);

	tpl->Set(String::NewFromUtf8(isolate, "drawArraysInstancedANGLE"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("drawArraysInstancedANGLE requires 4 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLint first = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLsizei primcount = args[3]->Int32Value();

		glDrawArraysInstancedANGLE(mode, first, count, primcount);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "drawElementsInstancedANGLE"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("drawElementsInstancedANGLE requires 5 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLsizei count = args[1]->Int32Value();
		GLenum type = args[2]->Uint32Value();

		void* indices = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glDrawElementsInstancedANGLE): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLsizei primcount = args[4]->Int32Value();

		glDrawElementsInstancedANGLE(mode, count, type, indices, primcount);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribDivisorANGLE"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribDivisorANGLE requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLuint divisor = args[1]->Uint32Value();

		glVertexAttribDivisorANGLE(index, divisor);
	}));



	/* ------------------------------ GL_ANGLE_pack_reverse_row_order ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("PACK_REVERSE_ROW_ORDER_ANGLE", GL_PACK_REVERSE_ROW_ORDER_ANGLE);



	/* ------------------------------ GL_ANGLE_program_binary ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("PROGRAM_BINARY_ANGLE", GL_PROGRAM_BINARY_ANGLE);



	/* ------------------------------ GL_ANGLE_texture_compression_dxt1 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGB_S3TC_DXT1_ANGLE", GL_COMPRESSED_RGB_S3TC_DXT1_ANGLE);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGBA_S3TC_DXT1_ANGLE", GL_COMPRESSED_RGBA_S3TC_DXT1_ANGLE);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGBA_S3TC_DXT3_ANGLE", GL_COMPRESSED_RGBA_S3TC_DXT3_ANGLE);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGBA_S3TC_DXT5_ANGLE", GL_COMPRESSED_RGBA_S3TC_DXT5_ANGLE);



	/* ------------------------------ GL_ANGLE_texture_compression_dxt3 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGB_S3TC_DXT1_ANGLE", GL_COMPRESSED_RGB_S3TC_DXT1_ANGLE);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGBA_S3TC_DXT1_ANGLE", GL_COMPRESSED_RGBA_S3TC_DXT1_ANGLE);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGBA_S3TC_DXT3_ANGLE", GL_COMPRESSED_RGBA_S3TC_DXT3_ANGLE);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGBA_S3TC_DXT5_ANGLE", GL_COMPRESSED_RGBA_S3TC_DXT5_ANGLE);



	/* ------------------------------ GL_ANGLE_texture_compression_dxt5 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGB_S3TC_DXT1_ANGLE", GL_COMPRESSED_RGB_S3TC_DXT1_ANGLE);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGBA_S3TC_DXT1_ANGLE", GL_COMPRESSED_RGBA_S3TC_DXT1_ANGLE);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGBA_S3TC_DXT3_ANGLE", GL_COMPRESSED_RGBA_S3TC_DXT3_ANGLE);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGBA_S3TC_DXT5_ANGLE", GL_COMPRESSED_RGBA_S3TC_DXT5_ANGLE);



	/* ------------------------------ GL_ANGLE_texture_usage ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_USAGE_ANGLE", GL_TEXTURE_USAGE_ANGLE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_ANGLE", GL_FRAMEBUFFER_ATTACHMENT_ANGLE);



	/* ------------------------------ GL_ANGLE_timer_query ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("QUERY_COUNTER_BITS_ANGLE", GL_QUERY_COUNTER_BITS_ANGLE);
	CREATE_CONSTANT_ACCESSOR("CURRENT_QUERY_ANGLE", GL_CURRENT_QUERY_ANGLE);
	CREATE_CONSTANT_ACCESSOR("QUERY_RESULT_ANGLE", GL_QUERY_RESULT_ANGLE);
	CREATE_CONSTANT_ACCESSOR("QUERY_RESULT_AVAILABLE_ANGLE", GL_QUERY_RESULT_AVAILABLE_ANGLE);
	CREATE_CONSTANT_ACCESSOR("TIME_ELAPSED_ANGLE", GL_TIME_ELAPSED_ANGLE);
	CREATE_CONSTANT_ACCESSOR("TIMESTAMP_ANGLE", GL_TIMESTAMP_ANGLE);

	tpl->Set(String::NewFromUtf8(isolate, "beginQueryANGLE"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("beginQueryANGLE requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint id = args[1]->Uint32Value();

		glBeginQueryANGLE(target, id);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "deleteQueriesANGLE"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("deleteQueriesANGLE requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* ids = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			ids = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glDeleteQueriesANGLE): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glDeleteQueriesANGLE(n, ids);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "endQueryANGLE"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("endQueryANGLE requires 1 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		glEndQueryANGLE(target);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "genQueriesANGLE"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("genQueriesANGLE requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* ids = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			ids = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGenQueriesANGLE): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGenQueriesANGLE(n, ids);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "getQueryObjectivANGLE"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getQueryObjectivANGLE requires 3 arguments");
			return;
		}

		GLuint id = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetQueryObjectivANGLE): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetQueryObjectivANGLE(id, pname, params);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "getQueryObjectuivANGLE"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getQueryObjectuivANGLE requires 3 arguments");
			return;
		}

		GLuint id = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLuint* params = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGetQueryObjectuivANGLE): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGetQueryObjectuivANGLE(id, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getQueryivANGLE"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getQueryivANGLE requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetQueryivANGLE): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetQueryivANGLE(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "queryCounterANGLE"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("queryCounterANGLE requires 2 arguments");
			return;
		}

		GLuint id = args[0]->Uint32Value();
		GLenum target = args[1]->Uint32Value();

		glQueryCounterANGLE(id, target);
	}));



	/* ------------------------------ GL_ANGLE_translated_shader_source ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TRANSLATED_SHADER_SOURCE_LENGTH_ANGLE", GL_TRANSLATED_SHADER_SOURCE_LENGTH_ANGLE);

	tpl->Set(String::NewFromUtf8(isolate, "getTranslatedShaderSourceANGLE"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getTranslatedShaderSourceANGLE requires 4 arguments");
			return;
		}

		GLuint shader = args[0]->Uint32Value();
		GLsizei bufsize = args[1]->Int32Value();

		GLsizei* length = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetTranslatedShaderSourceANGLE): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLchar* source = nullptr;
		if (args[3]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[3]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			source = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetTranslatedShaderSourceANGLE): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetTranslatedShaderSourceANGLE(shader, bufsize, length, source);
	}));



	// empty / skipped / ignored: GL_APPLE_aux_depth_stencil
	// empty / skipped / ignored: GL_APPLE_client_storage
	// empty / skipped / ignored: GL_APPLE_clip_distance
	// empty / skipped / ignored: GL_APPLE_color_buffer_packed_float
	// empty / skipped / ignored: GL_APPLE_copy_texture_levels
	// empty / skipped / ignored: GL_APPLE_element_array
	// empty / skipped / ignored: GL_APPLE_fence
	// empty / skipped / ignored: GL_APPLE_float_pixels
	// empty / skipped / ignored: GL_APPLE_flush_buffer_range
	// empty / skipped / ignored: GL_APPLE_framebuffer_multisample
	// empty / skipped / ignored: GL_APPLE_object_purgeable
	// empty / skipped / ignored: GL_APPLE_pixel_buffer
	// empty / skipped / ignored: GL_APPLE_rgb_422
	// empty / skipped / ignored: GL_APPLE_row_bytes
	// empty / skipped / ignored: GL_APPLE_specular_vector
	// empty / skipped / ignored: GL_APPLE_sync
	// empty / skipped / ignored: GL_APPLE_texture_2D_limited_npot
	// empty / skipped / ignored: GL_APPLE_texture_format_BGRA8888
	// empty / skipped / ignored: GL_APPLE_texture_max_level
	// empty / skipped / ignored: GL_APPLE_texture_packed_float
	// empty / skipped / ignored: GL_APPLE_texture_range
	// empty / skipped / ignored: GL_APPLE_transform_hint
	// empty / skipped / ignored: GL_APPLE_vertex_array_object
	// empty / skipped / ignored: GL_APPLE_vertex_array_range
	// empty / skipped / ignored: GL_APPLE_vertex_program_evaluators
	// empty / skipped / ignored: GL_APPLE_ycbcr_422
	// empty / skipped / ignored: GL_ARB_arrays_of_arrays
	/* ------------------------------ GL_ARB_base_instance ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "drawArraysInstancedBaseInstance"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("drawArraysInstancedBaseInstance requires 5 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLint first = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLsizei primcount = args[3]->Int32Value();
		GLuint baseinstance = args[4]->Uint32Value();

		glDrawArraysInstancedBaseInstance(mode, first, count, primcount, baseinstance);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "drawElementsInstancedBaseInstance"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("drawElementsInstancedBaseInstance requires 6 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLsizei count = args[1]->Int32Value();
		GLenum type = args[2]->Uint32Value();

		void* indices = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glDrawElementsInstancedBaseInstance): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLsizei primcount = args[4]->Int32Value();
		GLuint baseinstance = args[5]->Uint32Value();

		glDrawElementsInstancedBaseInstance(mode, count, type, indices, primcount, baseinstance);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "drawElementsInstancedBaseVertexBaseInstance"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("drawElementsInstancedBaseVertexBaseInstance requires 7 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLsizei count = args[1]->Int32Value();
		GLenum type = args[2]->Uint32Value();

		void* indices = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glDrawElementsInstancedBaseVertexBaseInstance): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLsizei primcount = args[4]->Int32Value();
		GLint basevertex = args[5]->Int32Value();
		GLuint baseinstance = args[6]->Uint32Value();

		glDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, primcount, basevertex, baseinstance);
	}));



	// empty / skipped / ignored: GL_ARB_bindless_texture
	/* ------------------------------ GL_ARB_blend_func_extended ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("SRC1_COLOR", GL_SRC1_COLOR);
	CREATE_CONSTANT_ACCESSOR("ONE_MINUS_SRC1_COLOR", GL_ONE_MINUS_SRC1_COLOR);
	CREATE_CONSTANT_ACCESSOR("ONE_MINUS_SRC1_ALPHA", GL_ONE_MINUS_SRC1_ALPHA);
	CREATE_CONSTANT_ACCESSOR("MAX_DUAL_SOURCE_DRAW_BUFFERS", GL_MAX_DUAL_SOURCE_DRAW_BUFFERS);




	/* ------------------------------ GL_ARB_buffer_storage ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("MAP_READ_BIT", GL_MAP_READ_BIT);
	CREATE_CONSTANT_ACCESSOR("MAP_WRITE_BIT", GL_MAP_WRITE_BIT);
	CREATE_CONSTANT_ACCESSOR("MAP_PERSISTENT_BIT", GL_MAP_PERSISTENT_BIT);
	CREATE_CONSTANT_ACCESSOR("MAP_COHERENT_BIT", GL_MAP_COHERENT_BIT);
	CREATE_CONSTANT_ACCESSOR("DYNAMIC_STORAGE_BIT", GL_DYNAMIC_STORAGE_BIT);
	CREATE_CONSTANT_ACCESSOR("CLIENT_STORAGE_BIT", GL_CLIENT_STORAGE_BIT);
	CREATE_CONSTANT_ACCESSOR("CLIENT_MAPPED_BUFFER_BARRIER_BIT", GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("BUFFER_IMMUTABLE_STORAGE", GL_BUFFER_IMMUTABLE_STORAGE);
	CREATE_CONSTANT_ACCESSOR("BUFFER_STORAGE_FLAGS", GL_BUFFER_STORAGE_FLAGS);

	tpl->Set(String::NewFromUtf8(isolate, "bufferStorage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("bufferStorage requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizeiptr size = GLsizeiptr(args[1]->Int32Value());

		void* data = nullptr;
		if (args[2]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[2]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[2]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[2]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glBufferStorage): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLbitfield flags = args[3]->Uint32Value();

		glBufferStorage(target, size, data, flags);
	}));



	/* ------------------------------ GL_ARB_clear_buffer_object ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "clearBufferData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("clearBufferData requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLenum format = args[2]->Uint32Value();
		GLenum type = args[3]->Uint32Value();

		void* data = nullptr;
		if (args[4]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[4]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[4]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[4]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glClearBufferData): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glClearBufferData(target, internalformat, format, type, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "clearBufferSubData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("clearBufferSubData requires 7 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLintptr offset = GLintptr(args[2]->Int32Value());
		GLsizeiptr size = GLsizeiptr(args[3]->Int32Value());
		GLenum format = args[4]->Uint32Value();
		GLenum type = args[5]->Uint32Value();

		void* data = nullptr;
		if (args[6]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[6]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[6]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[6]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glClearBufferSubData): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glClearBufferSubData(target, internalformat, offset, size, format, type, data);
	}));



	/* ------------------------------ GL_ARB_clear_texture ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("CLEAR_TEXTURE", GL_CLEAR_TEXTURE);

	tpl->Set(String::NewFromUtf8(isolate, "clearTexImage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("clearTexImage requires 5 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLenum format = args[2]->Uint32Value();
		GLenum type = args[3]->Uint32Value();

		void* data = nullptr;
		if (args[4]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[4]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[4]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[4]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glClearTexImage): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glClearTexImage(texture, level, format, type, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "clearTexSubImage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 11) {
			V8Helper::_instance->throwException("clearTexSubImage requires 11 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLint zoffset = args[4]->Int32Value();
		GLsizei width = args[5]->Int32Value();
		GLsizei height = args[6]->Int32Value();
		GLsizei depth = args[7]->Int32Value();
		GLenum format = args[8]->Uint32Value();
		GLenum type = args[9]->Uint32Value();

		void* data = nullptr;
		if (args[10]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[10]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[10]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[10]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glClearTexSubImage): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glClearTexSubImage(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, type, data);
	}));



	/* ------------------------------ GL_ARB_clip_control ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("LOWER_LEFT", GL_LOWER_LEFT);
	CREATE_CONSTANT_ACCESSOR("UPPER_LEFT", GL_UPPER_LEFT);
	CREATE_CONSTANT_ACCESSOR("CLIP_ORIGIN", GL_CLIP_ORIGIN);
	CREATE_CONSTANT_ACCESSOR("CLIP_DEPTH_MODE", GL_CLIP_DEPTH_MODE);
	CREATE_CONSTANT_ACCESSOR("NEGATIVE_ONE_TO_ONE", GL_NEGATIVE_ONE_TO_ONE);
	CREATE_CONSTANT_ACCESSOR("ZERO_TO_ONE", GL_ZERO_TO_ONE);

	tpl->Set(String::NewFromUtf8(isolate, "clipControl"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("clipControl requires 2 arguments");
			return;
		}

		GLenum origin = args[0]->Uint32Value();
		GLenum depth = args[1]->Uint32Value();

		glClipControl(origin, depth);
	}));



	// empty / skipped / ignored: GL_ARB_cl_event
	// empty / skipped / ignored: GL_ARB_color_buffer_float
	// empty / skipped / ignored: GL_ARB_compatibility
	/* ------------------------------ GL_ARB_compressed_texture_pixel_storage ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("UNPACK_COMPRESSED_BLOCK_WIDTH", GL_UNPACK_COMPRESSED_BLOCK_WIDTH);
	CREATE_CONSTANT_ACCESSOR("UNPACK_COMPRESSED_BLOCK_HEIGHT", GL_UNPACK_COMPRESSED_BLOCK_HEIGHT);
	CREATE_CONSTANT_ACCESSOR("UNPACK_COMPRESSED_BLOCK_DEPTH", GL_UNPACK_COMPRESSED_BLOCK_DEPTH);
	CREATE_CONSTANT_ACCESSOR("UNPACK_COMPRESSED_BLOCK_SIZE", GL_UNPACK_COMPRESSED_BLOCK_SIZE);
	CREATE_CONSTANT_ACCESSOR("PACK_COMPRESSED_BLOCK_WIDTH", GL_PACK_COMPRESSED_BLOCK_WIDTH);
	CREATE_CONSTANT_ACCESSOR("PACK_COMPRESSED_BLOCK_HEIGHT", GL_PACK_COMPRESSED_BLOCK_HEIGHT);
	CREATE_CONSTANT_ACCESSOR("PACK_COMPRESSED_BLOCK_DEPTH", GL_PACK_COMPRESSED_BLOCK_DEPTH);
	CREATE_CONSTANT_ACCESSOR("PACK_COMPRESSED_BLOCK_SIZE", GL_PACK_COMPRESSED_BLOCK_SIZE);



	/* ------------------------------ GL_ARB_compute_shader ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("COMPUTE_SHADER_BIT", GL_COMPUTE_SHADER_BIT);
	CREATE_CONSTANT_ACCESSOR("MAX_COMPUTE_SHARED_MEMORY_SIZE", GL_MAX_COMPUTE_SHARED_MEMORY_SIZE);
	CREATE_CONSTANT_ACCESSOR("MAX_COMPUTE_UNIFORM_COMPONENTS", GL_MAX_COMPUTE_UNIFORM_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS", GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMPUTE_ATOMIC_COUNTERS", GL_MAX_COMPUTE_ATOMIC_COUNTERS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS", GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("COMPUTE_WORK_GROUP_SIZE", GL_COMPUTE_WORK_GROUP_SIZE);
	CREATE_CONSTANT_ACCESSOR("MAX_COMPUTE_WORK_GROUP_INVOCATIONS", GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BLOCK_REFERENCED_BY_COMPUTE_SHADER", GL_UNIFORM_BLOCK_REFERENCED_BY_COMPUTE_SHADER);
	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BUFFER_REFERENCED_BY_COMPUTE_SHADER", GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_COMPUTE_SHADER);
	CREATE_CONSTANT_ACCESSOR("DISPATCH_INDIRECT_BUFFER", GL_DISPATCH_INDIRECT_BUFFER);
	CREATE_CONSTANT_ACCESSOR("DISPATCH_INDIRECT_BUFFER_BINDING", GL_DISPATCH_INDIRECT_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("COMPUTE_SHADER", GL_COMPUTE_SHADER);
	CREATE_CONSTANT_ACCESSOR("MAX_COMPUTE_UNIFORM_BLOCKS", GL_MAX_COMPUTE_UNIFORM_BLOCKS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMPUTE_TEXTURE_IMAGE_UNITS", GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMPUTE_IMAGE_UNIFORMS", GL_MAX_COMPUTE_IMAGE_UNIFORMS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMPUTE_WORK_GROUP_COUNT", GL_MAX_COMPUTE_WORK_GROUP_COUNT);
	CREATE_CONSTANT_ACCESSOR("MAX_COMPUTE_WORK_GROUP_SIZE", GL_MAX_COMPUTE_WORK_GROUP_SIZE);

	tpl->Set(String::NewFromUtf8(isolate, "dispatchCompute"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("dispatchCompute requires 3 arguments");
			return;
		}

		GLuint num_groups_x = args[0]->Uint32Value();
		GLuint num_groups_y = args[1]->Uint32Value();
		GLuint num_groups_z = args[2]->Uint32Value();

		glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "dispatchComputeIndirect"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("dispatchComputeIndirect requires 1 arguments");
			return;
		}

		GLintptr indirect = GLintptr(args[0]->Int32Value());

		glDispatchComputeIndirect(indirect);
	}));



	// empty / skipped / ignored: GL_ARB_compute_variable_group_size
	/* ------------------------------ GL_ARB_conditional_render_inverted ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("QUERY_WAIT_INVERTED", GL_QUERY_WAIT_INVERTED);
	CREATE_CONSTANT_ACCESSOR("QUERY_NO_WAIT_INVERTED", GL_QUERY_NO_WAIT_INVERTED);
	CREATE_CONSTANT_ACCESSOR("QUERY_BY_REGION_WAIT_INVERTED", GL_QUERY_BY_REGION_WAIT_INVERTED);
	CREATE_CONSTANT_ACCESSOR("QUERY_BY_REGION_NO_WAIT_INVERTED", GL_QUERY_BY_REGION_NO_WAIT_INVERTED);



	// empty / skipped / ignored: GL_ARB_conservative_depth
	/* ------------------------------ GL_ARB_copy_buffer ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("COPY_READ_BUFFER", GL_COPY_READ_BUFFER);
	CREATE_CONSTANT_ACCESSOR("COPY_WRITE_BUFFER", GL_COPY_WRITE_BUFFER);

	tpl->Set(String::NewFromUtf8(isolate, "copyBufferSubData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("copyBufferSubData requires 5 arguments");
			return;
		}

		GLenum readtarget = args[0]->Uint32Value();
		GLenum writetarget = args[1]->Uint32Value();
		GLintptr readoffset = GLintptr(args[2]->Int32Value());
		GLintptr writeoffset = GLintptr(args[3]->Int32Value());
		GLsizeiptr size = GLsizeiptr(args[4]->Int32Value());

		glCopyBufferSubData(readtarget, writetarget, readoffset, writeoffset, size);
	}));



	/* ------------------------------ GL_ARB_copy_image ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "copyImageSubData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 15) {
			V8Helper::_instance->throwException("copyImageSubData requires 15 arguments");
			return;
		}

		GLuint srcName = args[0]->Uint32Value();
		GLenum srcTarget = args[1]->Uint32Value();
		GLint srcLevel = args[2]->Int32Value();
		GLint srcX = args[3]->Int32Value();
		GLint srcY = args[4]->Int32Value();
		GLint srcZ = args[5]->Int32Value();
		GLuint dstName = args[6]->Uint32Value();
		GLenum dstTarget = args[7]->Uint32Value();
		GLint dstLevel = args[8]->Int32Value();
		GLint dstX = args[9]->Int32Value();
		GLint dstY = args[10]->Int32Value();
		GLint dstZ = args[11]->Int32Value();
		GLsizei srcWidth = args[12]->Int32Value();
		GLsizei srcHeight = args[13]->Int32Value();
		GLsizei srcDepth = args[14]->Int32Value();

		glCopyImageSubData(srcName, srcTarget, srcLevel, srcX, srcY, srcZ, dstName, dstTarget, dstLevel, dstX, dstY, dstZ, srcWidth, srcHeight, srcDepth);
	}));



	/* ------------------------------ GL_ARB_cull_distance ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("MAX_CULL_DISTANCES", GL_MAX_CULL_DISTANCES);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_CLIP_AND_CULL_DISTANCES", GL_MAX_COMBINED_CLIP_AND_CULL_DISTANCES);



	// empty / skipped / ignored: GL_ARB_debug_output
	/* ------------------------------ GL_ARB_depth_buffer_float ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("DEPTH_COMPONENT32F", GL_DEPTH_COMPONENT32F);
	CREATE_CONSTANT_ACCESSOR("DEPTH32F_STENCIL8", GL_DEPTH32F_STENCIL8);
	CREATE_CONSTANT_ACCESSOR("FLOAT_32_UNSIGNED_INT_24_8_REV", GL_FLOAT_32_UNSIGNED_INT_24_8_REV);



	/* ------------------------------ GL_ARB_depth_clamp ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("DEPTH_CLAMP", GL_DEPTH_CLAMP);



	// empty / skipped / ignored: GL_ARB_depth_texture
	// empty / skipped / ignored: GL_ARB_derivative_control
	/* ------------------------------ GL_ARB_direct_state_access ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_TARGET", GL_TEXTURE_TARGET);
	CREATE_CONSTANT_ACCESSOR("QUERY_TARGET", GL_QUERY_TARGET);

	tpl->Set(String::NewFromUtf8(isolate, "bindTextureUnit"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("bindTextureUnit requires 2 arguments");
			return;
		}

		GLuint unit = args[0]->Uint32Value();
		GLuint texture = args[1]->Uint32Value();

		glBindTextureUnit(unit, texture);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "blitNamedFramebuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 12) {
			V8Helper::_instance->throwException("blitNamedFramebuffer requires 12 arguments");
			return;
		}

		GLuint readFramebuffer = args[0]->Uint32Value();
		GLuint drawFramebuffer = args[1]->Uint32Value();
		GLint srcX0 = args[2]->Int32Value();
		GLint srcY0 = args[3]->Int32Value();
		GLint srcX1 = args[4]->Int32Value();
		GLint srcY1 = args[5]->Int32Value();
		GLint dstX0 = args[6]->Int32Value();
		GLint dstY0 = args[7]->Int32Value();
		GLint dstX1 = args[8]->Int32Value();
		GLint dstY1 = args[9]->Int32Value();
		GLbitfield mask = args[10]->Uint32Value();
		GLenum filter = args[11]->Uint32Value();

		glBlitNamedFramebuffer(readFramebuffer, drawFramebuffer, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "clearNamedBufferData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("clearNamedBufferData requires 5 arguments");
			return;
		}

		GLuint buffer = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLenum format = args[2]->Uint32Value();
		GLenum type = args[3]->Uint32Value();

		void* data = nullptr;
		if (args[4]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[4]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[4]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[4]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			//cout << "ERROR(glClearNamedBufferData): array must be of type ArrayBuffer" << endl;
			//exit(1);
		}


		glClearNamedBufferData(buffer, internalformat, format, type, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "clearNamedBufferSubData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("clearNamedBufferSubData requires 7 arguments");
			return;
		}

		GLuint buffer = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLintptr offset = GLintptr(args[2]->Int32Value());
		GLsizeiptr size = GLsizeiptr(args[3]->Int32Value());
		GLenum format = args[4]->Uint32Value();
		GLenum type = args[5]->Uint32Value();

		void* data = nullptr;
		if (args[6]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[6]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[6]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[6]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glClearNamedBufferSubData): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glClearNamedBufferSubData(buffer, internalformat, offset, size, format, type, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "clearNamedFramebufferfi"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("clearNamedFramebufferfi requires 5 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLenum buffer = args[1]->Uint32Value();
		GLint drawbuffer = args[2]->Int32Value();
		GLfloat depth = GLfloat(args[3]->NumberValue());
		GLint stencil = args[4]->Int32Value();

		glClearNamedFramebufferfi(framebuffer, buffer, drawbuffer, depth, stencil);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "clearNamedFramebufferfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("clearNamedFramebufferfv requires 4 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLenum buffer = args[1]->Uint32Value();
		GLint drawbuffer = args[2]->Int32Value();

		GLfloat* value = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glClearNamedFramebufferfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glClearNamedFramebufferfv(framebuffer, buffer, drawbuffer, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "clearNamedFramebufferiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("clearNamedFramebufferiv requires 4 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLenum buffer = args[1]->Uint32Value();
		GLint drawbuffer = args[2]->Int32Value();

		GLint* value = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glClearNamedFramebufferiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glClearNamedFramebufferiv(framebuffer, buffer, drawbuffer, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "clearNamedFramebufferuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("clearNamedFramebufferuiv requires 4 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLenum buffer = args[1]->Uint32Value();
		GLint drawbuffer = args[2]->Int32Value();

		GLuint* value = nullptr;
		if (args[3]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[3]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glClearNamedFramebufferuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glClearNamedFramebufferuiv(framebuffer, buffer, drawbuffer, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "compressedTextureSubImage1D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("compressedTextureSubImage1D requires 7 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLsizei width = args[3]->Int32Value();
		GLenum format = args[4]->Uint32Value();
		GLsizei imageSize = args[5]->Int32Value();

		void* data = nullptr;
		if (args[6]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[6]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[6]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[6]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glCompressedTextureSubImage1D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glCompressedTextureSubImage1D(texture, level, xoffset, width, format, imageSize, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "compressedTextureSubImage2D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 9) {
			V8Helper::_instance->throwException("compressedTextureSubImage2D requires 9 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLsizei width = args[4]->Int32Value();
		GLsizei height = args[5]->Int32Value();
		GLenum format = args[6]->Uint32Value();
		GLsizei imageSize = args[7]->Int32Value();

		void* data = nullptr;
		if (args[8]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[8]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[8]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[8]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glCompressedTextureSubImage2D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glCompressedTextureSubImage2D(texture, level, xoffset, yoffset, width, height, format, imageSize, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "compressedTextureSubImage3D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 11) {
			V8Helper::_instance->throwException("compressedTextureSubImage3D requires 11 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLint zoffset = args[4]->Int32Value();
		GLsizei width = args[5]->Int32Value();
		GLsizei height = args[6]->Int32Value();
		GLsizei depth = args[7]->Int32Value();
		GLenum format = args[8]->Uint32Value();
		GLsizei imageSize = args[9]->Int32Value();

		void* data = nullptr;
		if (args[10]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[10]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[10]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[10]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glCompressedTextureSubImage3D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glCompressedTextureSubImage3D(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "copyNamedBufferSubData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("copyNamedBufferSubData requires 5 arguments");
			return;
		}

		GLuint readBuffer = args[0]->Uint32Value();
		GLuint writeBuffer = args[1]->Uint32Value();
		GLintptr readOffset = GLintptr(args[2]->Int32Value());
		GLintptr writeOffset = GLintptr(args[3]->Int32Value());
		GLsizeiptr size = GLsizeiptr(args[4]->Int32Value());

		glCopyNamedBufferSubData(readBuffer, writeBuffer, readOffset, writeOffset, size);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "copyTextureSubImage1D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("copyTextureSubImage1D requires 6 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint x = args[3]->Int32Value();
		GLint y = args[4]->Int32Value();
		GLsizei width = args[5]->Int32Value();

		glCopyTextureSubImage1D(texture, level, xoffset, x, y, width);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "copyTextureSubImage2D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 8) {
			V8Helper::_instance->throwException("copyTextureSubImage2D requires 8 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLint x = args[4]->Int32Value();
		GLint y = args[5]->Int32Value();
		GLsizei width = args[6]->Int32Value();
		GLsizei height = args[7]->Int32Value();

		glCopyTextureSubImage2D(texture, level, xoffset, yoffset, x, y, width, height);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "copyTextureSubImage3D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 9) {
			V8Helper::_instance->throwException("copyTextureSubImage3D requires 9 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLint zoffset = args[4]->Int32Value();
		GLint x = args[5]->Int32Value();
		GLint y = args[6]->Int32Value();
		GLsizei width = args[7]->Int32Value();
		GLsizei height = args[8]->Int32Value();

		glCopyTextureSubImage3D(texture, level, xoffset, yoffset, zoffset, x, y, width, height);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "createBuffers"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("createBuffers requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* buffers = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			buffers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glCreateBuffers): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glCreateBuffers(n, buffers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "createFramebuffers"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("createFramebuffers requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* framebuffers = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			framebuffers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glCreateFramebuffers): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glCreateFramebuffers(n, framebuffers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "createProgramPipelines"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("createProgramPipelines requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* pipelines = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pipelines = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glCreateProgramPipelines): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glCreateProgramPipelines(n, pipelines);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "createQueries"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("createQueries requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei n = args[1]->Int32Value();

		GLuint* ids = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			ids = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glCreateQueries): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glCreateQueries(target, n, ids);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "createRenderbuffers"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("createRenderbuffers requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* renderbuffers = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			renderbuffers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glCreateRenderbuffers): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glCreateRenderbuffers(n, renderbuffers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "createSamplers"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("createSamplers requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* samplers = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			samplers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glCreateSamplers): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glCreateSamplers(n, samplers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "createTextures"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("createTextures requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei n = args[1]->Int32Value();

		GLuint* textures = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			textures = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glCreateTextures): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glCreateTextures(target, n, textures);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "createTransformFeedbacks"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("createTransformFeedbacks requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* ids = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			ids = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glCreateTransformFeedbacks): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glCreateTransformFeedbacks(n, ids);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "createVertexArrays"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("createVertexArrays requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* arrays = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			arrays = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glCreateVertexArrays): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glCreateVertexArrays(n, arrays);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "disableVertexArrayAttrib"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("disableVertexArrayAttrib requires 2 arguments");
			return;
		}

		GLuint vaobj = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();

		glDisableVertexArrayAttrib(vaobj, index);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "enableVertexArrayAttrib"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("enableVertexArrayAttrib requires 2 arguments");
			return;
		}

		GLuint vaobj = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();

		glEnableVertexArrayAttrib(vaobj, index);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "flushMappedNamedBufferRange"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("flushMappedNamedBufferRange requires 3 arguments");
			return;
		}

		GLuint buffer = args[0]->Uint32Value();
		GLintptr offset = GLintptr(args[1]->Int32Value());
		GLsizeiptr length = GLsizeiptr(args[2]->Int32Value());

		glFlushMappedNamedBufferRange(buffer, offset, length);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "generateTextureMipmap"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("generateTextureMipmap requires 1 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();

		glGenerateTextureMipmap(texture);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getCompressedTextureImage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getCompressedTextureImage requires 4 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLsizei bufSize = args[2]->Int32Value();

		void* pixels = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetCompressedTextureImage): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glGetCompressedTextureImage(texture, level, bufSize, pixels);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "getNamedBufferParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getNamedBufferParameteriv requires 3 arguments");
			return;
		}

		GLuint buffer = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetNamedBufferParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetNamedBufferParameteriv(buffer, pname, params);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "getNamedBufferSubData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getNamedBufferSubData requires 4 arguments");
			return;
		}

		GLuint buffer = args[0]->Uint32Value();
		GLintptr offset = GLintptr(args[1]->Int32Value());
		GLsizeiptr size = GLsizeiptr(args[2]->Int32Value());

		void* data = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetNamedBufferSubData): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glGetNamedBufferSubData(buffer, offset, size, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getNamedFramebufferAttachmentParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getNamedFramebufferAttachmentParameteriv requires 4 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLenum attachment = args[1]->Uint32Value();
		GLenum pname = args[2]->Uint32Value();

		GLint* params = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetNamedFramebufferAttachmentParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetNamedFramebufferAttachmentParameteriv(framebuffer, attachment, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getNamedFramebufferParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getNamedFramebufferParameteriv requires 3 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* param = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			param = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetNamedFramebufferParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetNamedFramebufferParameteriv(framebuffer, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getNamedRenderbufferParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getNamedRenderbufferParameteriv requires 3 arguments");
			return;
		}

		GLuint renderbuffer = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetNamedRenderbufferParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetNamedRenderbufferParameteriv(renderbuffer, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getQueryBufferObjecti64v"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getQueryBufferObjecti64v requires 4 arguments");
			return;
		}

		GLuint id = args[0]->Uint32Value();
		GLuint buffer = args[1]->Uint32Value();
		GLenum pname = args[2]->Uint32Value();
		GLintptr offset = GLintptr(args[3]->Int32Value());

		glGetQueryBufferObjecti64v(id, buffer, pname, offset);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getQueryBufferObjectiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getQueryBufferObjectiv requires 4 arguments");
			return;
		}

		GLuint id = args[0]->Uint32Value();
		GLuint buffer = args[1]->Uint32Value();
		GLenum pname = args[2]->Uint32Value();
		GLintptr offset = GLintptr(args[3]->Int32Value());

		glGetQueryBufferObjectiv(id, buffer, pname, offset);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getQueryBufferObjectui64v"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getQueryBufferObjectui64v requires 4 arguments");
			return;
		}

		GLuint id = args[0]->Uint32Value();
		GLuint buffer = args[1]->Uint32Value();
		GLenum pname = args[2]->Uint32Value();
		GLintptr offset = GLintptr(args[3]->Int32Value());

		glGetQueryBufferObjectui64v(id, buffer, pname, offset);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getQueryBufferObjectuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getQueryBufferObjectuiv requires 4 arguments");
			return;
		}

		GLuint id = args[0]->Uint32Value();
		GLuint buffer = args[1]->Uint32Value();
		GLenum pname = args[2]->Uint32Value();
		GLintptr offset = GLintptr(args[3]->Int32Value());

		glGetQueryBufferObjectuiv(id, buffer, pname, offset);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getTextureImage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("getTextureImage requires 6 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLenum format = args[2]->Uint32Value();
		GLenum type = args[3]->Uint32Value();
		GLsizei bufSize = args[4]->Int32Value();

		void* pixels = nullptr;
		if (args[5]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[5]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else if (args[5]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[5]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetTextureImage): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glGetTextureImage(texture, level, format, type, bufSize, pixels);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getTextureLevelParameterfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getTextureLevelParameterfv requires 4 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLenum pname = args[2]->Uint32Value();

		GLfloat* params = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glGetTextureLevelParameterfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glGetTextureLevelParameterfv(texture, level, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getTextureLevelParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getTextureLevelParameteriv requires 4 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLenum pname = args[2]->Uint32Value();

		GLint* params = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetTextureLevelParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetTextureLevelParameteriv(texture, level, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getTextureParameterIiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getTextureParameterIiv requires 3 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetTextureParameterIiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetTextureParameterIiv(texture, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getTextureParameterIuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getTextureParameterIuiv requires 3 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLuint* params = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGetTextureParameterIuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGetTextureParameterIuiv(texture, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getTextureParameterfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getTextureParameterfv requires 3 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfloat* params = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glGetTextureParameterfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glGetTextureParameterfv(texture, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getTextureParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getTextureParameteriv requires 3 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetTextureParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetTextureParameteriv(texture, pname, params);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "getTransformFeedbacki_v"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getTransformFeedbacki_v requires 4 arguments");
			return;
		}

		GLuint xfb = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLuint index = args[2]->Uint32Value();

		GLint* param = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			param = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetTransformFeedbacki_v): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetTransformFeedbacki_v(xfb, pname, index, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getTransformFeedbackiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getTransformFeedbackiv requires 3 arguments");
			return;
		}

		GLuint xfb = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* param = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			param = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetTransformFeedbackiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetTransformFeedbackiv(xfb, pname, param);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "getVertexArrayIndexediv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getVertexArrayIndexediv requires 4 arguments");
			return;
		}

		GLuint vaobj = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();
		GLenum pname = args[2]->Uint32Value();

		GLint* param = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			param = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetVertexArrayIndexediv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetVertexArrayIndexediv(vaobj, index, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getVertexArrayiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getVertexArrayiv requires 3 arguments");
			return;
		}

		GLuint vaobj = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* param = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			param = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetVertexArrayiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetVertexArrayiv(vaobj, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "invalidateNamedFramebufferData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("invalidateNamedFramebufferData requires 3 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLsizei numAttachments = args[1]->Int32Value();

		GLenum* attachments = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			attachments = reinterpret_cast<GLenum*>(bdata);
		} else {
			cout << "ERROR(glInvalidateNamedFramebufferData): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glInvalidateNamedFramebufferData(framebuffer, numAttachments, attachments);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "invalidateNamedFramebufferSubData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("invalidateNamedFramebufferSubData requires 7 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLsizei numAttachments = args[1]->Int32Value();

		GLenum* attachments = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			attachments = reinterpret_cast<GLenum*>(bdata);
		} else {
			cout << "ERROR(glInvalidateNamedFramebufferSubData): array must be of type Uint32Array" << endl;
			exit(1);
		}

		GLint x = args[3]->Int32Value();
		GLint y = args[4]->Int32Value();
		GLsizei width = args[5]->Int32Value();
		GLsizei height = args[6]->Int32Value();

		glInvalidateNamedFramebufferSubData(framebuffer, numAttachments, attachments, x, y, width, height);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "namedBufferData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("namedBufferData requires 4 arguments");
			return;
		}

		GLuint buffer = args[0]->Uint32Value();
		GLsizeiptr size = GLsizeiptr(args[1]->Int32Value());

		void* data = nullptr;
		if (args[2]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[2]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[2]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[2]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			//cout << "ERROR(glNamedBufferData): array must be of type ArrayBuffer" << endl;
			//exit(1);
		}

		GLenum usage = args[3]->Uint32Value();

		glNamedBufferData(buffer, size, data, usage);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "namedBufferStorage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("namedBufferStorage requires 4 arguments");
			return;
		}

		GLuint buffer = args[0]->Uint32Value();
		GLsizeiptr size = GLsizeiptr(args[1]->Int32Value());

		void* data = nullptr;
		if (args[2]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[2]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[2]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[2]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glNamedBufferStorage): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLbitfield flags = args[3]->Uint32Value();

		glNamedBufferStorage(buffer, size, data, flags);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "namedBufferSubData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("namedBufferSubData requires 4 arguments");
			return;
		}

		GLuint buffer = args[0]->Uint32Value();
		GLintptr offset = GLintptr(args[1]->Int32Value());
		GLsizeiptr size = GLsizeiptr(args[2]->Int32Value());

		void* data = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glNamedBufferSubData): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glNamedBufferSubData(buffer, offset, size, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "namedFramebufferDrawBuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("namedFramebufferDrawBuffer requires 2 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLenum mode = args[1]->Uint32Value();

		glNamedFramebufferDrawBuffer(framebuffer, mode);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "namedFramebufferDrawBuffers"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("namedFramebufferDrawBuffers requires 3 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLsizei n = args[1]->Int32Value();

		GLenum* bufs = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			bufs = reinterpret_cast<GLenum*>(bdata);
		} else {
			cout << "ERROR(glNamedFramebufferDrawBuffers): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glNamedFramebufferDrawBuffers(framebuffer, n, bufs);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "namedFramebufferParameteri"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("namedFramebufferParameteri requires 3 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLint param = args[2]->Int32Value();

		glNamedFramebufferParameteri(framebuffer, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "namedFramebufferReadBuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("namedFramebufferReadBuffer requires 2 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLenum mode = args[1]->Uint32Value();

		glNamedFramebufferReadBuffer(framebuffer, mode);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "namedFramebufferRenderbuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("namedFramebufferRenderbuffer requires 4 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLenum attachment = args[1]->Uint32Value();
		GLenum renderbuffertarget = args[2]->Uint32Value();
		GLuint renderbuffer = args[3]->Uint32Value();

		glNamedFramebufferRenderbuffer(framebuffer, attachment, renderbuffertarget, renderbuffer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "namedFramebufferTexture"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("namedFramebufferTexture requires 4 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLenum attachment = args[1]->Uint32Value();
		GLuint texture = args[2]->Uint32Value();
		GLint level = args[3]->Int32Value();

		glNamedFramebufferTexture(framebuffer, attachment, texture, level);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "namedFramebufferTextureLayer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("namedFramebufferTextureLayer requires 5 arguments");
			return;
		}

		GLuint framebuffer = args[0]->Uint32Value();
		GLenum attachment = args[1]->Uint32Value();
		GLuint texture = args[2]->Uint32Value();
		GLint level = args[3]->Int32Value();
		GLint layer = args[4]->Int32Value();

		glNamedFramebufferTextureLayer(framebuffer, attachment, texture, level, layer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "namedRenderbufferStorage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("namedRenderbufferStorage requires 4 arguments");
			return;
		}

		GLuint renderbuffer = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLsizei width = args[2]->Int32Value();
		GLsizei height = args[3]->Int32Value();

		glNamedRenderbufferStorage(renderbuffer, internalformat, width, height);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "namedRenderbufferStorageMultisample"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("namedRenderbufferStorageMultisample requires 5 arguments");
			return;
		}

		GLuint renderbuffer = args[0]->Uint32Value();
		GLsizei samples = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();

		glNamedRenderbufferStorageMultisample(renderbuffer, samples, internalformat, width, height);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureBuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("textureBuffer requires 3 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLuint buffer = args[2]->Uint32Value();

		glTextureBuffer(texture, internalformat, buffer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureBufferRange"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("textureBufferRange requires 5 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLuint buffer = args[2]->Uint32Value();
		GLintptr offset = GLintptr(args[3]->Int32Value());
		GLsizeiptr size = GLsizeiptr(args[4]->Int32Value());

		glTextureBufferRange(texture, internalformat, buffer, offset, size);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureParameterIiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("textureParameterIiv requires 3 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glTextureParameterIiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glTextureParameterIiv(texture, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureParameterIuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("textureParameterIuiv requires 3 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLuint* params = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glTextureParameterIuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glTextureParameterIuiv(texture, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureParameterf"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("textureParameterf requires 3 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLfloat param = GLfloat(args[2]->NumberValue());

		glTextureParameterf(texture, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureParameterfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("textureParameterfv requires 3 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfloat* param = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			param = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glTextureParameterfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glTextureParameterfv(texture, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureParameteri"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("textureParameteri requires 3 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLint param = args[2]->Int32Value();

		glTextureParameteri(texture, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("textureParameteriv requires 3 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* param = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			param = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glTextureParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glTextureParameteriv(texture, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureStorage1D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("textureStorage1D requires 4 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLsizei levels = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();

		glTextureStorage1D(texture, levels, internalformat, width);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureStorage2D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("textureStorage2D requires 5 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLsizei levels = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();

		glTextureStorage2D(texture, levels, internalformat, width, height);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureStorage2DMultisample"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("textureStorage2DMultisample requires 6 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLsizei samples = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();
		GLboolean fixedsamplelocations = GLboolean(args[5]->Uint32Value());

		glTextureStorage2DMultisample(texture, samples, internalformat, width, height, fixedsamplelocations);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureStorage3D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("textureStorage3D requires 6 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLsizei levels = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();
		GLsizei depth = args[5]->Int32Value();

		glTextureStorage3D(texture, levels, internalformat, width, height, depth);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureStorage3DMultisample"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("textureStorage3DMultisample requires 7 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLsizei samples = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();
		GLsizei depth = args[5]->Int32Value();
		GLboolean fixedsamplelocations = GLboolean(args[6]->Uint32Value());

		glTextureStorage3DMultisample(texture, samples, internalformat, width, height, depth, fixedsamplelocations);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureSubImage1D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("textureSubImage1D requires 7 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLsizei width = args[3]->Int32Value();
		GLenum format = args[4]->Uint32Value();
		GLenum type = args[5]->Uint32Value();

		void* pixels = nullptr;
		if (args[6]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[6]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else if (args[6]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[6]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glTextureSubImage1D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glTextureSubImage1D(texture, level, xoffset, width, format, type, pixels);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureSubImage2D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 9) {
			V8Helper::_instance->throwException("textureSubImage2D requires 9 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLsizei width = args[4]->Int32Value();
		GLsizei height = args[5]->Int32Value();
		GLenum format = args[6]->Uint32Value();
		GLenum type = args[7]->Uint32Value();

		void* pixels = nullptr;
		if (args[8]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[8]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else if (args[8]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[8]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glTextureSubImage2D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glTextureSubImage2D(texture, level, xoffset, yoffset, width, height, format, type, pixels);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "textureSubImage3D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 11) {
			V8Helper::_instance->throwException("textureSubImage3D requires 11 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLint zoffset = args[4]->Int32Value();
		GLsizei width = args[5]->Int32Value();
		GLsizei height = args[6]->Int32Value();
		GLsizei depth = args[7]->Int32Value();
		GLenum format = args[8]->Uint32Value();
		GLenum type = args[9]->Uint32Value();

		void* pixels = nullptr;
		if (args[10]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[10]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else if (args[10]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[10]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glTextureSubImage3D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glTextureSubImage3D(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "transformFeedbackBufferBase"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("transformFeedbackBufferBase requires 3 arguments");
			return;
		}

		GLuint xfb = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();
		GLuint buffer = args[2]->Uint32Value();

		glTransformFeedbackBufferBase(xfb, index, buffer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "transformFeedbackBufferRange"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("transformFeedbackBufferRange requires 5 arguments");
			return;
		}

		GLuint xfb = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();
		GLuint buffer = args[2]->Uint32Value();
		GLintptr offset = GLintptr(args[3]->Int32Value());
		GLsizeiptr size = GLsizeiptr(args[4]->Int32Value());

		glTransformFeedbackBufferRange(xfb, index, buffer, offset, size);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexArrayAttribBinding"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("vertexArrayAttribBinding requires 3 arguments");
			return;
		}

		GLuint vaobj = args[0]->Uint32Value();
		GLuint attribindex = args[1]->Uint32Value();
		GLuint bindingindex = args[2]->Uint32Value();

		glVertexArrayAttribBinding(vaobj, attribindex, bindingindex);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexArrayAttribFormat"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("vertexArrayAttribFormat requires 6 arguments");
			return;
		}

		GLuint vaobj = args[0]->Uint32Value();
		GLuint attribindex = args[1]->Uint32Value();
		GLint size = args[2]->Int32Value();
		GLenum type = args[3]->Uint32Value();
		GLboolean normalized = GLboolean(args[4]->Uint32Value());
		GLuint relativeoffset = args[5]->Uint32Value();

		glVertexArrayAttribFormat(vaobj, attribindex, size, type, normalized, relativeoffset);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexArrayAttribIFormat"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("vertexArrayAttribIFormat requires 5 arguments");
			return;
		}

		GLuint vaobj = args[0]->Uint32Value();
		GLuint attribindex = args[1]->Uint32Value();
		GLint size = args[2]->Int32Value();
		GLenum type = args[3]->Uint32Value();
		GLuint relativeoffset = args[4]->Uint32Value();

		glVertexArrayAttribIFormat(vaobj, attribindex, size, type, relativeoffset);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexArrayAttribLFormat"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("vertexArrayAttribLFormat requires 5 arguments");
			return;
		}

		GLuint vaobj = args[0]->Uint32Value();
		GLuint attribindex = args[1]->Uint32Value();
		GLint size = args[2]->Int32Value();
		GLenum type = args[3]->Uint32Value();
		GLuint relativeoffset = args[4]->Uint32Value();

		glVertexArrayAttribLFormat(vaobj, attribindex, size, type, relativeoffset);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexArrayBindingDivisor"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("vertexArrayBindingDivisor requires 3 arguments");
			return;
		}

		GLuint vaobj = args[0]->Uint32Value();
		GLuint bindingindex = args[1]->Uint32Value();
		GLuint divisor = args[2]->Uint32Value();

		glVertexArrayBindingDivisor(vaobj, bindingindex, divisor);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexArrayElementBuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexArrayElementBuffer requires 2 arguments");
			return;
		}

		GLuint vaobj = args[0]->Uint32Value();
		GLuint buffer = args[1]->Uint32Value();

		glVertexArrayElementBuffer(vaobj, buffer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexArrayVertexBuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("vertexArrayVertexBuffer requires 5 arguments");
			return;
		}

		GLuint vaobj = args[0]->Uint32Value();
		GLuint bindingindex = args[1]->Uint32Value();
		GLuint buffer = args[2]->Uint32Value();
		GLintptr offset = GLintptr(args[3]->Int32Value());
		GLsizei stride = args[4]->Int32Value();

		glVertexArrayVertexBuffer(vaobj, bindingindex, buffer, offset, stride);
	}));




	// empty / skipped / ignored: GL_ARB_draw_buffers
	// empty / skipped / ignored: GL_ARB_draw_buffers_blend
	/* ------------------------------ GL_ARB_draw_elements_base_vertex ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "drawElementsBaseVertex"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("drawElementsBaseVertex requires 5 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLsizei count = args[1]->Int32Value();
		GLenum type = args[2]->Uint32Value();

		void* indices = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glDrawElementsBaseVertex): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLint basevertex = args[4]->Int32Value();

		glDrawElementsBaseVertex(mode, count, type, indices, basevertex);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "drawElementsInstancedBaseVertex"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("drawElementsInstancedBaseVertex requires 6 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLsizei count = args[1]->Int32Value();
		GLenum type = args[2]->Uint32Value();

		void* indices = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glDrawElementsInstancedBaseVertex): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLsizei primcount = args[4]->Int32Value();
		GLint basevertex = args[5]->Int32Value();

		glDrawElementsInstancedBaseVertex(mode, count, type, indices, primcount, basevertex);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "drawRangeElementsBaseVertex"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("drawRangeElementsBaseVertex requires 7 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLuint start = args[1]->Uint32Value();
		GLuint end = args[2]->Uint32Value();
		GLsizei count = args[3]->Int32Value();
		GLenum type = args[4]->Uint32Value();

		void* indices = nullptr;
		if (args[5]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[5]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else if (args[5]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[5]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indices = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glDrawRangeElementsBaseVertex): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLint basevertex = args[6]->Int32Value();

		glDrawRangeElementsBaseVertex(mode, start, end, count, type, indices, basevertex);
	}));




	/* ------------------------------ GL_ARB_draw_indirect ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("DRAW_INDIRECT_BUFFER", GL_DRAW_INDIRECT_BUFFER);
	CREATE_CONSTANT_ACCESSOR("DRAW_INDIRECT_BUFFER_BINDING", GL_DRAW_INDIRECT_BUFFER_BINDING);

	tpl->Set(String::NewFromUtf8(isolate, "drawArraysIndirect"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("drawArraysIndirect requires 2 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();

		void* indirect = nullptr;
		if (args[1]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[1]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			indirect = reinterpret_cast<void*>(bdata);
		} else if (args[1]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[1]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indirect = reinterpret_cast<void*>(bdata);
		} else if(args[1]->IsNumber()){
			int offset = args[1]->Uint32Value();
			
			char* ptr = nullptr;
			ptr = ptr + offset;
			indirect = (void*)ptr;

			//indirect = (void*)(((char*)nullptr) + offset);
			//cout << "ERROR(glDrawArraysIndirect): array must be of type ArrayBuffer" << endl;
			//exit(1);
		}


		glDrawArraysIndirect(mode, indirect);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "drawElementsIndirect"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("drawElementsIndirect requires 3 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();

		void* indirect = nullptr;
		if (args[2]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[2]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			indirect = reinterpret_cast<void*>(bdata);
		} else if (args[2]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[2]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indirect = reinterpret_cast<void*>(bdata);
		} else {
			
		}


		glDrawElementsIndirect(mode, type, indirect);
	}));



	// empty / skipped / ignored: GL_ARB_draw_instanced
	/* ------------------------------ GL_ARB_enhanced_layouts ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("LOCATION_COMPONENT", GL_LOCATION_COMPONENT);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_BUFFER_INDEX", GL_TRANSFORM_FEEDBACK_BUFFER_INDEX);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_BUFFER_STRIDE", GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE);



	/* ------------------------------ GL_ARB_ES2_compatibility ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("FIXED", GL_FIXED);
	CREATE_CONSTANT_ACCESSOR("IMPLEMENTATION_COLOR_READ_TYPE", GL_IMPLEMENTATION_COLOR_READ_TYPE);
	CREATE_CONSTANT_ACCESSOR("IMPLEMENTATION_COLOR_READ_FORMAT", GL_IMPLEMENTATION_COLOR_READ_FORMAT);
	CREATE_CONSTANT_ACCESSOR("RGB565", GL_RGB565);
	CREATE_CONSTANT_ACCESSOR("LOW_FLOAT", GL_LOW_FLOAT);
	CREATE_CONSTANT_ACCESSOR("MEDIUM_FLOAT", GL_MEDIUM_FLOAT);
	CREATE_CONSTANT_ACCESSOR("HIGH_FLOAT", GL_HIGH_FLOAT);
	CREATE_CONSTANT_ACCESSOR("LOW_INT", GL_LOW_INT);
	CREATE_CONSTANT_ACCESSOR("MEDIUM_INT", GL_MEDIUM_INT);
	CREATE_CONSTANT_ACCESSOR("HIGH_INT", GL_HIGH_INT);
	CREATE_CONSTANT_ACCESSOR("SHADER_BINARY_FORMATS", GL_SHADER_BINARY_FORMATS);
	CREATE_CONSTANT_ACCESSOR("NUM_SHADER_BINARY_FORMATS", GL_NUM_SHADER_BINARY_FORMATS);
	CREATE_CONSTANT_ACCESSOR("SHADER_COMPILER", GL_SHADER_COMPILER);
	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_UNIFORM_VECTORS", GL_MAX_VERTEX_UNIFORM_VECTORS);
	CREATE_CONSTANT_ACCESSOR("MAX_VARYING_VECTORS", GL_MAX_VARYING_VECTORS);
	CREATE_CONSTANT_ACCESSOR("MAX_FRAGMENT_UNIFORM_VECTORS", GL_MAX_FRAGMENT_UNIFORM_VECTORS);

	tpl->Set(String::NewFromUtf8(isolate, "clearDepthf"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("clearDepthf requires 1 arguments");
			return;
		}

		GLclampf d = GLclampf(args[0]->NumberValue());

		glClearDepthf(d);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "depthRangef"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("depthRangef requires 2 arguments");
			return;
		}

		GLclampf n = GLclampf(args[0]->NumberValue());
		GLclampf f = GLclampf(args[1]->NumberValue());

		glDepthRangef(n, f);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getShaderPrecisionFormat"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getShaderPrecisionFormat requires 4 arguments");
			return;
		}

		GLenum shadertype = args[0]->Uint32Value();
		GLenum precisiontype = args[1]->Uint32Value();

		GLint* range = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			range = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetShaderPrecisionFormat): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLint* precision = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			precision = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetShaderPrecisionFormat): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetShaderPrecisionFormat(shadertype, precisiontype, range, precision);
	}));





	/* ------------------------------ GL_ARB_ES3_1_compatibility ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "memoryBarrierByRegion"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("memoryBarrierByRegion requires 1 arguments");
			return;
		}

		GLbitfield barriers = args[0]->Uint32Value();

		glMemoryBarrierByRegion(barriers);
	}));



	// empty / skipped / ignored: GL_ARB_ES3_2_compatibility
	/* ------------------------------ GL_ARB_ES3_compatibility ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_IMMUTABLE_LEVELS", GL_TEXTURE_IMMUTABLE_LEVELS);
	CREATE_CONSTANT_ACCESSOR("PRIMITIVE_RESTART_FIXED_INDEX", GL_PRIMITIVE_RESTART_FIXED_INDEX);
	CREATE_CONSTANT_ACCESSOR("ANY_SAMPLES_PASSED_CONSERVATIVE", GL_ANY_SAMPLES_PASSED_CONSERVATIVE);
	CREATE_CONSTANT_ACCESSOR("MAX_ELEMENT_INDEX", GL_MAX_ELEMENT_INDEX);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_R11_EAC", GL_COMPRESSED_R11_EAC);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_SIGNED_R11_EAC", GL_COMPRESSED_SIGNED_R11_EAC);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RG11_EAC", GL_COMPRESSED_RG11_EAC);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_SIGNED_RG11_EAC", GL_COMPRESSED_SIGNED_RG11_EAC);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGB8_ETC2", GL_COMPRESSED_RGB8_ETC2);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_SRGB8_ETC2", GL_COMPRESSED_SRGB8_ETC2);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2", GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2", GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RGBA8_ETC2_EAC", GL_COMPRESSED_RGBA8_ETC2_EAC);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_SRGB8_ALPHA8_ETC2_EAC", GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC);



	// empty / skipped / ignored: GL_ARB_explicit_attrib_location
	/* ------------------------------ GL_ARB_explicit_uniform_location ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("MAX_UNIFORM_LOCATIONS", GL_MAX_UNIFORM_LOCATIONS);



	// empty / skipped / ignored: GL_ARB_fragment_coord_conventions
	// empty / skipped / ignored: GL_ARB_fragment_layer_viewport
	// empty / skipped / ignored: GL_ARB_fragment_program
	// empty / skipped / ignored: GL_ARB_fragment_program_shadow
	// empty / skipped / ignored: GL_ARB_fragment_shader
	// empty / skipped / ignored: GL_ARB_fragment_shader_interlock
	/* ------------------------------ GL_ARB_framebuffer_no_attachments ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_DEFAULT_WIDTH", GL_FRAMEBUFFER_DEFAULT_WIDTH);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_DEFAULT_HEIGHT", GL_FRAMEBUFFER_DEFAULT_HEIGHT);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_DEFAULT_LAYERS", GL_FRAMEBUFFER_DEFAULT_LAYERS);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_DEFAULT_SAMPLES", GL_FRAMEBUFFER_DEFAULT_SAMPLES);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS", GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS);
	CREATE_CONSTANT_ACCESSOR("MAX_FRAMEBUFFER_WIDTH", GL_MAX_FRAMEBUFFER_WIDTH);
	CREATE_CONSTANT_ACCESSOR("MAX_FRAMEBUFFER_HEIGHT", GL_MAX_FRAMEBUFFER_HEIGHT);
	CREATE_CONSTANT_ACCESSOR("MAX_FRAMEBUFFER_LAYERS", GL_MAX_FRAMEBUFFER_LAYERS);
	CREATE_CONSTANT_ACCESSOR("MAX_FRAMEBUFFER_SAMPLES", GL_MAX_FRAMEBUFFER_SAMPLES);

	tpl->Set(String::NewFromUtf8(isolate, "framebufferParameteri"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("framebufferParameteri requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLint param = args[2]->Int32Value();

		glFramebufferParameteri(target, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getFramebufferParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getFramebufferParameteriv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetFramebufferParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetFramebufferParameteriv(target, pname, params);
	}));



	/* ------------------------------ GL_ARB_framebuffer_object ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("INVALID_FRAMEBUFFER_OPERATION", GL_INVALID_FRAMEBUFFER_OPERATION);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING", GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE", GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_RED_SIZE", GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_GREEN_SIZE", GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_BLUE_SIZE", GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE", GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE", GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE", GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_DEFAULT", GL_FRAMEBUFFER_DEFAULT);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_UNDEFINED", GL_FRAMEBUFFER_UNDEFINED);
	CREATE_CONSTANT_ACCESSOR("DEPTH_STENCIL_ATTACHMENT", GL_DEPTH_STENCIL_ATTACHMENT);
	CREATE_CONSTANT_ACCESSOR("INDEX", GL_INDEX);
	CREATE_CONSTANT_ACCESSOR("MAX_RENDERBUFFER_SIZE", GL_MAX_RENDERBUFFER_SIZE);
	CREATE_CONSTANT_ACCESSOR("DEPTH_STENCIL", GL_DEPTH_STENCIL);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_24_8", GL_UNSIGNED_INT_24_8);
	CREATE_CONSTANT_ACCESSOR("DEPTH24_STENCIL8", GL_DEPTH24_STENCIL8);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_STENCIL_SIZE", GL_TEXTURE_STENCIL_SIZE);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_NORMALIZED", GL_UNSIGNED_NORMALIZED);
	CREATE_CONSTANT_ACCESSOR("SRGB", GL_SRGB);
	CREATE_CONSTANT_ACCESSOR("DRAW_FRAMEBUFFER_BINDING", GL_DRAW_FRAMEBUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_BINDING", GL_FRAMEBUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("RENDERBUFFER_BINDING", GL_RENDERBUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("READ_FRAMEBUFFER", GL_READ_FRAMEBUFFER);
	CREATE_CONSTANT_ACCESSOR("DRAW_FRAMEBUFFER", GL_DRAW_FRAMEBUFFER);
	CREATE_CONSTANT_ACCESSOR("READ_FRAMEBUFFER_BINDING", GL_READ_FRAMEBUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("RENDERBUFFER_SAMPLES", GL_RENDERBUFFER_SAMPLES);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE", GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_OBJECT_NAME", GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL", GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE", GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER", GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_COMPLETE", GL_FRAMEBUFFER_COMPLETE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_INCOMPLETE_ATTACHMENT", GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT", GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER", GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_INCOMPLETE_READ_BUFFER", GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_UNSUPPORTED", GL_FRAMEBUFFER_UNSUPPORTED);
	CREATE_CONSTANT_ACCESSOR("MAX_COLOR_ATTACHMENTS", GL_MAX_COLOR_ATTACHMENTS);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT0", GL_COLOR_ATTACHMENT0);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT1", GL_COLOR_ATTACHMENT1);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT2", GL_COLOR_ATTACHMENT2);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT3", GL_COLOR_ATTACHMENT3);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT4", GL_COLOR_ATTACHMENT4);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT5", GL_COLOR_ATTACHMENT5);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT6", GL_COLOR_ATTACHMENT6);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT7", GL_COLOR_ATTACHMENT7);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT8", GL_COLOR_ATTACHMENT8);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT9", GL_COLOR_ATTACHMENT9);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT10", GL_COLOR_ATTACHMENT10);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT11", GL_COLOR_ATTACHMENT11);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT12", GL_COLOR_ATTACHMENT12);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT13", GL_COLOR_ATTACHMENT13);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT14", GL_COLOR_ATTACHMENT14);
	CREATE_CONSTANT_ACCESSOR("COLOR_ATTACHMENT15", GL_COLOR_ATTACHMENT15);
	CREATE_CONSTANT_ACCESSOR("DEPTH_ATTACHMENT", GL_DEPTH_ATTACHMENT);
	CREATE_CONSTANT_ACCESSOR("STENCIL_ATTACHMENT", GL_STENCIL_ATTACHMENT);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER", GL_FRAMEBUFFER);
	CREATE_CONSTANT_ACCESSOR("RENDERBUFFER", GL_RENDERBUFFER);
	CREATE_CONSTANT_ACCESSOR("RENDERBUFFER_WIDTH", GL_RENDERBUFFER_WIDTH);
	CREATE_CONSTANT_ACCESSOR("RENDERBUFFER_HEIGHT", GL_RENDERBUFFER_HEIGHT);
	CREATE_CONSTANT_ACCESSOR("RENDERBUFFER_INTERNAL_FORMAT", GL_RENDERBUFFER_INTERNAL_FORMAT);
	CREATE_CONSTANT_ACCESSOR("STENCIL_INDEX1", GL_STENCIL_INDEX1);
	CREATE_CONSTANT_ACCESSOR("STENCIL_INDEX4", GL_STENCIL_INDEX4);
	CREATE_CONSTANT_ACCESSOR("STENCIL_INDEX8", GL_STENCIL_INDEX8);
	CREATE_CONSTANT_ACCESSOR("STENCIL_INDEX16", GL_STENCIL_INDEX16);
	CREATE_CONSTANT_ACCESSOR("RENDERBUFFER_RED_SIZE", GL_RENDERBUFFER_RED_SIZE);
	CREATE_CONSTANT_ACCESSOR("RENDERBUFFER_GREEN_SIZE", GL_RENDERBUFFER_GREEN_SIZE);
	CREATE_CONSTANT_ACCESSOR("RENDERBUFFER_BLUE_SIZE", GL_RENDERBUFFER_BLUE_SIZE);
	CREATE_CONSTANT_ACCESSOR("RENDERBUFFER_ALPHA_SIZE", GL_RENDERBUFFER_ALPHA_SIZE);
	CREATE_CONSTANT_ACCESSOR("RENDERBUFFER_DEPTH_SIZE", GL_RENDERBUFFER_DEPTH_SIZE);
	CREATE_CONSTANT_ACCESSOR("RENDERBUFFER_STENCIL_SIZE", GL_RENDERBUFFER_STENCIL_SIZE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_INCOMPLETE_MULTISAMPLE", GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE);
	CREATE_CONSTANT_ACCESSOR("MAX_SAMPLES", GL_MAX_SAMPLES);

	tpl->Set(String::NewFromUtf8(isolate, "bindFramebuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("bindFramebuffer requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint framebuffer = args[1]->Uint32Value();

		glBindFramebuffer(target, framebuffer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "bindRenderbuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("bindRenderbuffer requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint renderbuffer = args[1]->Uint32Value();

		glBindRenderbuffer(target, renderbuffer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "blitFramebuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 10) {
			V8Helper::_instance->throwException("blitFramebuffer requires 10 arguments");
			return;
		}

		GLint srcX0 = args[0]->Int32Value();
		GLint srcY0 = args[1]->Int32Value();
		GLint srcX1 = args[2]->Int32Value();
		GLint srcY1 = args[3]->Int32Value();
		GLint dstX0 = args[4]->Int32Value();
		GLint dstY0 = args[5]->Int32Value();
		GLint dstX1 = args[6]->Int32Value();
		GLint dstY1 = args[7]->Int32Value();
		GLbitfield mask = args[8]->Uint32Value();
		GLenum filter = args[9]->Uint32Value();

		glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "deleteFramebuffers"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("deleteFramebuffers requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* framebuffers = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			framebuffers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glDeleteFramebuffers): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glDeleteFramebuffers(n, framebuffers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "deleteRenderbuffers"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("deleteRenderbuffers requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* renderbuffers = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			renderbuffers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glDeleteRenderbuffers): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glDeleteRenderbuffers(n, renderbuffers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "framebufferRenderbuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("framebufferRenderbuffer requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum attachment = args[1]->Uint32Value();
		GLenum renderbuffertarget = args[2]->Uint32Value();
		GLuint renderbuffer = args[3]->Uint32Value();

		glFramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "framebufferTexture1D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("framebufferTexture1D requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum attachment = args[1]->Uint32Value();
		GLenum textarget = args[2]->Uint32Value();
		GLuint texture = args[3]->Uint32Value();
		GLint level = args[4]->Int32Value();

		glFramebufferTexture1D(target, attachment, textarget, texture, level);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "framebufferTexture2D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("framebufferTexture2D requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum attachment = args[1]->Uint32Value();
		GLenum textarget = args[2]->Uint32Value();
		GLuint texture = args[3]->Uint32Value();
		GLint level = args[4]->Int32Value();

		glFramebufferTexture2D(target, attachment, textarget, texture, level);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "framebufferTexture3D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("framebufferTexture3D requires 6 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum attachment = args[1]->Uint32Value();
		GLenum textarget = args[2]->Uint32Value();
		GLuint texture = args[3]->Uint32Value();
		GLint level = args[4]->Int32Value();
		GLint layer = args[5]->Int32Value();

		glFramebufferTexture3D(target, attachment, textarget, texture, level, layer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "framebufferTextureLayer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("framebufferTextureLayer requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum attachment = args[1]->Uint32Value();
		GLuint texture = args[2]->Uint32Value();
		GLint level = args[3]->Int32Value();
		GLint layer = args[4]->Int32Value();

		glFramebufferTextureLayer(target, attachment, texture, level, layer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "genFramebuffers"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("genFramebuffers requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* framebuffers = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			framebuffers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGenFramebuffers): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGenFramebuffers(n, framebuffers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "genRenderbuffers"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("genRenderbuffers requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* renderbuffers = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			renderbuffers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGenRenderbuffers): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGenRenderbuffers(n, renderbuffers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "generateMipmap"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("generateMipmap requires 1 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		glGenerateMipmap(target);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getFramebufferAttachmentParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getFramebufferAttachmentParameteriv requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum attachment = args[1]->Uint32Value();
		GLenum pname = args[2]->Uint32Value();

		GLint* params = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetFramebufferAttachmentParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetFramebufferAttachmentParameteriv(target, attachment, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getRenderbufferParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getRenderbufferParameteriv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetRenderbufferParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetRenderbufferParameteriv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "renderbufferStorage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("renderbufferStorage requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLsizei width = args[2]->Int32Value();
		GLsizei height = args[3]->Int32Value();

		glRenderbufferStorage(target, internalformat, width, height);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "renderbufferStorageMultisample"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("renderbufferStorageMultisample requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei samples = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();

		glRenderbufferStorageMultisample(target, samples, internalformat, width, height);
	}));



	/* ------------------------------ GL_ARB_framebuffer_sRGB ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_SRGB", GL_FRAMEBUFFER_SRGB);



	/* ------------------------------ GL_ARB_geometry_shader4 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER", GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER);



	/* ------------------------------ GL_ARB_get_program_binary ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("PROGRAM_BINARY_RETRIEVABLE_HINT", GL_PROGRAM_BINARY_RETRIEVABLE_HINT);
	CREATE_CONSTANT_ACCESSOR("PROGRAM_BINARY_LENGTH", GL_PROGRAM_BINARY_LENGTH);
	CREATE_CONSTANT_ACCESSOR("NUM_PROGRAM_BINARY_FORMATS", GL_NUM_PROGRAM_BINARY_FORMATS);
	CREATE_CONSTANT_ACCESSOR("PROGRAM_BINARY_FORMATS", GL_PROGRAM_BINARY_FORMATS);


	tpl->Set(String::NewFromUtf8(isolate, "programBinary"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programBinary requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLenum binaryFormat = args[1]->Uint32Value();

		void* binary = nullptr;
		if (args[2]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[2]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			binary = reinterpret_cast<void*>(bdata);
		} else if (args[2]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[2]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			binary = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glProgramBinary): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLsizei length = args[3]->Int32Value();

		glProgramBinary(program, binaryFormat, binary, length);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programParameteri"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("programParameteri requires 3 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLint value = args[2]->Int32Value();

		glProgramParameteri(program, pname, value);
	}));



	/* ------------------------------ GL_ARB_get_texture_sub_image ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "getCompressedTextureSubImage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 10) {
			V8Helper::_instance->throwException("getCompressedTextureSubImage requires 10 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLint zoffset = args[4]->Int32Value();
		GLsizei width = args[5]->Int32Value();
		GLsizei height = args[6]->Int32Value();
		GLsizei depth = args[7]->Int32Value();
		GLsizei bufSize = args[8]->Int32Value();

		void* pixels = nullptr;
		if (args[9]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[9]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else if (args[9]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[9]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetCompressedTextureSubImage): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glGetCompressedTextureSubImage(texture, level, xoffset, yoffset, zoffset, width, height, depth, bufSize, pixels);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getTextureSubImage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 12) {
			V8Helper::_instance->throwException("getTextureSubImage requires 12 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLint zoffset = args[4]->Int32Value();
		GLsizei width = args[5]->Int32Value();
		GLsizei height = args[6]->Int32Value();
		GLsizei depth = args[7]->Int32Value();
		GLenum format = args[8]->Uint32Value();
		GLenum type = args[9]->Uint32Value();
		GLsizei bufSize = args[10]->Int32Value();

		void* pixels = nullptr;
		if (args[11]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[11]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else if (args[11]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[11]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pixels = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetTextureSubImage): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glGetTextureSubImage(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, type, bufSize, pixels);
	}));



	// empty / skipped / ignored: GL_ARB_gl_spirv
	/* ------------------------------ GL_ARB_gpu_shader5 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("GEOMETRY_SHADER_INVOCATIONS", GL_GEOMETRY_SHADER_INVOCATIONS);
	CREATE_CONSTANT_ACCESSOR("MAX_GEOMETRY_SHADER_INVOCATIONS", GL_MAX_GEOMETRY_SHADER_INVOCATIONS);
	CREATE_CONSTANT_ACCESSOR("MIN_FRAGMENT_INTERPOLATION_OFFSET", GL_MIN_FRAGMENT_INTERPOLATION_OFFSET);
	CREATE_CONSTANT_ACCESSOR("MAX_FRAGMENT_INTERPOLATION_OFFSET", GL_MAX_FRAGMENT_INTERPOLATION_OFFSET);
	CREATE_CONSTANT_ACCESSOR("FRAGMENT_INTERPOLATION_OFFSET_BITS", GL_FRAGMENT_INTERPOLATION_OFFSET_BITS);
	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_STREAMS", GL_MAX_VERTEX_STREAMS);



	/* ------------------------------ GL_ARB_gpu_shader_fp64 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("DOUBLE_MAT2", GL_DOUBLE_MAT2);
	CREATE_CONSTANT_ACCESSOR("DOUBLE_MAT3", GL_DOUBLE_MAT3);
	CREATE_CONSTANT_ACCESSOR("DOUBLE_MAT4", GL_DOUBLE_MAT4);
	CREATE_CONSTANT_ACCESSOR("DOUBLE_MAT2x3", GL_DOUBLE_MAT2x3);
	CREATE_CONSTANT_ACCESSOR("DOUBLE_MAT2x4", GL_DOUBLE_MAT2x4);
	CREATE_CONSTANT_ACCESSOR("DOUBLE_MAT3x2", GL_DOUBLE_MAT3x2);
	CREATE_CONSTANT_ACCESSOR("DOUBLE_MAT3x4", GL_DOUBLE_MAT3x4);
	CREATE_CONSTANT_ACCESSOR("DOUBLE_MAT4x2", GL_DOUBLE_MAT4x2);
	CREATE_CONSTANT_ACCESSOR("DOUBLE_MAT4x3", GL_DOUBLE_MAT4x3);
	CREATE_CONSTANT_ACCESSOR("DOUBLE_VEC2", GL_DOUBLE_VEC2);
	CREATE_CONSTANT_ACCESSOR("DOUBLE_VEC3", GL_DOUBLE_VEC3);
	CREATE_CONSTANT_ACCESSOR("DOUBLE_VEC4", GL_DOUBLE_VEC4);

	tpl->Set(String::NewFromUtf8(isolate, "getUniformdv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getUniformdv requires 3 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();

		GLdouble* params = nullptr;
		if (args[2]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[2]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glGetUniformdv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glGetUniformdv(program, location, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform1d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("uniform1d requires 2 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLdouble x = args[1]->NumberValue();

		glUniform1d(location, x);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform1dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform1dv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLdouble* value = nullptr;
		if (args[2]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[2]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glUniform1dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glUniform1dv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform2d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform2d requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLdouble x = args[1]->NumberValue();
		GLdouble y = args[2]->NumberValue();

		glUniform2d(location, x, y);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform2dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform2dv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLdouble* value = nullptr;
		if (args[2]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[2]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glUniform2dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glUniform2dv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform3d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniform3d requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLdouble x = args[1]->NumberValue();
		GLdouble y = args[2]->NumberValue();
		GLdouble z = args[3]->NumberValue();

		glUniform3d(location, x, y, z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform3dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform3dv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLdouble* value = nullptr;
		if (args[2]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[2]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glUniform3dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glUniform3dv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform4d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("uniform4d requires 5 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLdouble x = args[1]->NumberValue();
		GLdouble y = args[2]->NumberValue();
		GLdouble z = args[3]->NumberValue();
		GLdouble w = args[4]->NumberValue();

		glUniform4d(location, x, y, z, w);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniform4dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniform4dv requires 3 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();

		GLdouble* value = nullptr;
		if (args[2]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[2]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glUniform4dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glUniform4dv(location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix2dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix2dv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[3]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[3]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix2dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glUniformMatrix2dv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix2x3dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix2x3dv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[3]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[3]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix2x3dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glUniformMatrix2x3dv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix2x4dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix2x4dv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[3]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[3]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix2x4dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glUniformMatrix2x4dv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix3dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix3dv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[3]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[3]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix3dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glUniformMatrix3dv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix3x2dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix3x2dv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[3]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[3]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix3x2dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glUniformMatrix3x2dv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix3x4dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix3x4dv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[3]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[3]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix3x4dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glUniformMatrix3x4dv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix4dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix4dv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[3]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[3]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix4dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glUniformMatrix4dv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix4x2dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix4x2dv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[3]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[3]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix4x2dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glUniformMatrix4x2dv(location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformMatrix4x3dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("uniformMatrix4x3dv requires 4 arguments");
			return;
		}

		GLint location = args[0]->Int32Value();
		GLsizei count = args[1]->Int32Value();
		GLboolean transpose = GLboolean(args[2]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[3]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[3]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glUniformMatrix4x3dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glUniformMatrix4x3dv(location, count, transpose, value);
	}));



	// empty / skipped / ignored: GL_ARB_gpu_shader_int64
	// empty / skipped / ignored: GL_ARB_half_float_pixel
	/* ------------------------------ GL_ARB_half_float_vertex ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("HALF_FLOAT", GL_HALF_FLOAT);



	/* ------------------------------ GL_ARB_imaging ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("CONSTANT_COLOR", GL_CONSTANT_COLOR);
	CREATE_CONSTANT_ACCESSOR("ONE_MINUS_CONSTANT_COLOR", GL_ONE_MINUS_CONSTANT_COLOR);
	CREATE_CONSTANT_ACCESSOR("CONSTANT_ALPHA", GL_CONSTANT_ALPHA);
	CREATE_CONSTANT_ACCESSOR("ONE_MINUS_CONSTANT_ALPHA", GL_ONE_MINUS_CONSTANT_ALPHA);
	CREATE_CONSTANT_ACCESSOR("BLEND_COLOR", GL_BLEND_COLOR);
	CREATE_CONSTANT_ACCESSOR("FUNC_ADD", GL_FUNC_ADD);
	CREATE_CONSTANT_ACCESSOR("MIN", GL_MIN);
	CREATE_CONSTANT_ACCESSOR("MAX", GL_MAX);
	CREATE_CONSTANT_ACCESSOR("BLEND_EQUATION", GL_BLEND_EQUATION);
	CREATE_CONSTANT_ACCESSOR("FUNC_SUBTRACT", GL_FUNC_SUBTRACT);
	CREATE_CONSTANT_ACCESSOR("FUNC_REVERSE_SUBTRACT", GL_FUNC_REVERSE_SUBTRACT);
	CREATE_CONSTANT_ACCESSOR("CONVOLUTION_1D", GL_CONVOLUTION_1D);
	CREATE_CONSTANT_ACCESSOR("CONVOLUTION_2D", GL_CONVOLUTION_2D);
	CREATE_CONSTANT_ACCESSOR("SEPARABLE_2D", GL_SEPARABLE_2D);
	CREATE_CONSTANT_ACCESSOR("CONVOLUTION_BORDER_MODE", GL_CONVOLUTION_BORDER_MODE);
	CREATE_CONSTANT_ACCESSOR("CONVOLUTION_FILTER_SCALE", GL_CONVOLUTION_FILTER_SCALE);
	CREATE_CONSTANT_ACCESSOR("CONVOLUTION_FILTER_BIAS", GL_CONVOLUTION_FILTER_BIAS);
	CREATE_CONSTANT_ACCESSOR("REDUCE", GL_REDUCE);
	CREATE_CONSTANT_ACCESSOR("CONVOLUTION_FORMAT", GL_CONVOLUTION_FORMAT);
	CREATE_CONSTANT_ACCESSOR("CONVOLUTION_WIDTH", GL_CONVOLUTION_WIDTH);
	CREATE_CONSTANT_ACCESSOR("CONVOLUTION_HEIGHT", GL_CONVOLUTION_HEIGHT);
	CREATE_CONSTANT_ACCESSOR("MAX_CONVOLUTION_WIDTH", GL_MAX_CONVOLUTION_WIDTH);
	CREATE_CONSTANT_ACCESSOR("MAX_CONVOLUTION_HEIGHT", GL_MAX_CONVOLUTION_HEIGHT);
	CREATE_CONSTANT_ACCESSOR("POST_CONVOLUTION_RED_SCALE", GL_POST_CONVOLUTION_RED_SCALE);
	CREATE_CONSTANT_ACCESSOR("POST_CONVOLUTION_GREEN_SCALE", GL_POST_CONVOLUTION_GREEN_SCALE);
	CREATE_CONSTANT_ACCESSOR("POST_CONVOLUTION_BLUE_SCALE", GL_POST_CONVOLUTION_BLUE_SCALE);
	CREATE_CONSTANT_ACCESSOR("POST_CONVOLUTION_ALPHA_SCALE", GL_POST_CONVOLUTION_ALPHA_SCALE);
	CREATE_CONSTANT_ACCESSOR("POST_CONVOLUTION_RED_BIAS", GL_POST_CONVOLUTION_RED_BIAS);
	CREATE_CONSTANT_ACCESSOR("POST_CONVOLUTION_GREEN_BIAS", GL_POST_CONVOLUTION_GREEN_BIAS);
	CREATE_CONSTANT_ACCESSOR("POST_CONVOLUTION_BLUE_BIAS", GL_POST_CONVOLUTION_BLUE_BIAS);
	CREATE_CONSTANT_ACCESSOR("POST_CONVOLUTION_ALPHA_BIAS", GL_POST_CONVOLUTION_ALPHA_BIAS);
	CREATE_CONSTANT_ACCESSOR("HISTOGRAM", GL_HISTOGRAM);
	CREATE_CONSTANT_ACCESSOR("PROXY_HISTOGRAM", GL_PROXY_HISTOGRAM);
	CREATE_CONSTANT_ACCESSOR("HISTOGRAM_WIDTH", GL_HISTOGRAM_WIDTH);
	CREATE_CONSTANT_ACCESSOR("HISTOGRAM_FORMAT", GL_HISTOGRAM_FORMAT);
	CREATE_CONSTANT_ACCESSOR("HISTOGRAM_RED_SIZE", GL_HISTOGRAM_RED_SIZE);
	CREATE_CONSTANT_ACCESSOR("HISTOGRAM_GREEN_SIZE", GL_HISTOGRAM_GREEN_SIZE);
	CREATE_CONSTANT_ACCESSOR("HISTOGRAM_BLUE_SIZE", GL_HISTOGRAM_BLUE_SIZE);
	CREATE_CONSTANT_ACCESSOR("HISTOGRAM_ALPHA_SIZE", GL_HISTOGRAM_ALPHA_SIZE);
	CREATE_CONSTANT_ACCESSOR("HISTOGRAM_LUMINANCE_SIZE", GL_HISTOGRAM_LUMINANCE_SIZE);
	CREATE_CONSTANT_ACCESSOR("HISTOGRAM_SINK", GL_HISTOGRAM_SINK);
	CREATE_CONSTANT_ACCESSOR("MINMAX", GL_MINMAX);
	CREATE_CONSTANT_ACCESSOR("MINMAX_FORMAT", GL_MINMAX_FORMAT);
	CREATE_CONSTANT_ACCESSOR("MINMAX_SINK", GL_MINMAX_SINK);
	CREATE_CONSTANT_ACCESSOR("TABLE_TOO_LARGE", GL_TABLE_TOO_LARGE);
	CREATE_CONSTANT_ACCESSOR("COLOR_MATRIX", GL_COLOR_MATRIX);
	CREATE_CONSTANT_ACCESSOR("COLOR_MATRIX_STACK_DEPTH", GL_COLOR_MATRIX_STACK_DEPTH);
	CREATE_CONSTANT_ACCESSOR("MAX_COLOR_MATRIX_STACK_DEPTH", GL_MAX_COLOR_MATRIX_STACK_DEPTH);
	CREATE_CONSTANT_ACCESSOR("POST_COLOR_MATRIX_RED_SCALE", GL_POST_COLOR_MATRIX_RED_SCALE);
	CREATE_CONSTANT_ACCESSOR("POST_COLOR_MATRIX_GREEN_SCALE", GL_POST_COLOR_MATRIX_GREEN_SCALE);
	CREATE_CONSTANT_ACCESSOR("POST_COLOR_MATRIX_BLUE_SCALE", GL_POST_COLOR_MATRIX_BLUE_SCALE);
	CREATE_CONSTANT_ACCESSOR("POST_COLOR_MATRIX_ALPHA_SCALE", GL_POST_COLOR_MATRIX_ALPHA_SCALE);
	CREATE_CONSTANT_ACCESSOR("POST_COLOR_MATRIX_RED_BIAS", GL_POST_COLOR_MATRIX_RED_BIAS);
	CREATE_CONSTANT_ACCESSOR("POST_COLOR_MATRIX_GREEN_BIAS", GL_POST_COLOR_MATRIX_GREEN_BIAS);
	CREATE_CONSTANT_ACCESSOR("POST_COLOR_MATRIX_BLUE_BIAS", GL_POST_COLOR_MATRIX_BLUE_BIAS);
	CREATE_CONSTANT_ACCESSOR("POST_COLOR_MATRIX_ALPHA_BIAS", GL_POST_COLOR_MATRIX_ALPHA_BIAS);
	CREATE_CONSTANT_ACCESSOR("COLOR_TABLE", GL_COLOR_TABLE);
	CREATE_CONSTANT_ACCESSOR("POST_CONVOLUTION_COLOR_TABLE", GL_POST_CONVOLUTION_COLOR_TABLE);
	CREATE_CONSTANT_ACCESSOR("POST_COLOR_MATRIX_COLOR_TABLE", GL_POST_COLOR_MATRIX_COLOR_TABLE);
	CREATE_CONSTANT_ACCESSOR("PROXY_COLOR_TABLE", GL_PROXY_COLOR_TABLE);
	CREATE_CONSTANT_ACCESSOR("PROXY_POST_CONVOLUTION_COLOR_TABLE", GL_PROXY_POST_CONVOLUTION_COLOR_TABLE);
	CREATE_CONSTANT_ACCESSOR("PROXY_POST_COLOR_MATRIX_COLOR_TABLE", GL_PROXY_POST_COLOR_MATRIX_COLOR_TABLE);
	CREATE_CONSTANT_ACCESSOR("COLOR_TABLE_SCALE", GL_COLOR_TABLE_SCALE);
	CREATE_CONSTANT_ACCESSOR("COLOR_TABLE_BIAS", GL_COLOR_TABLE_BIAS);
	CREATE_CONSTANT_ACCESSOR("COLOR_TABLE_FORMAT", GL_COLOR_TABLE_FORMAT);
	CREATE_CONSTANT_ACCESSOR("COLOR_TABLE_WIDTH", GL_COLOR_TABLE_WIDTH);
	CREATE_CONSTANT_ACCESSOR("COLOR_TABLE_RED_SIZE", GL_COLOR_TABLE_RED_SIZE);
	CREATE_CONSTANT_ACCESSOR("COLOR_TABLE_GREEN_SIZE", GL_COLOR_TABLE_GREEN_SIZE);
	CREATE_CONSTANT_ACCESSOR("COLOR_TABLE_BLUE_SIZE", GL_COLOR_TABLE_BLUE_SIZE);
	CREATE_CONSTANT_ACCESSOR("COLOR_TABLE_ALPHA_SIZE", GL_COLOR_TABLE_ALPHA_SIZE);
	CREATE_CONSTANT_ACCESSOR("COLOR_TABLE_LUMINANCE_SIZE", GL_COLOR_TABLE_LUMINANCE_SIZE);
	CREATE_CONSTANT_ACCESSOR("COLOR_TABLE_INTENSITY_SIZE", GL_COLOR_TABLE_INTENSITY_SIZE);
	CREATE_CONSTANT_ACCESSOR("IGNORE_BORDER", GL_IGNORE_BORDER);
	CREATE_CONSTANT_ACCESSOR("CONSTANT_BORDER", GL_CONSTANT_BORDER);
	CREATE_CONSTANT_ACCESSOR("WRAP_BORDER", GL_WRAP_BORDER);
	CREATE_CONSTANT_ACCESSOR("REPLICATE_BORDER", GL_REPLICATE_BORDER);
	CREATE_CONSTANT_ACCESSOR("CONVOLUTION_BORDER_COLOR", GL_CONVOLUTION_BORDER_COLOR);

	tpl->Set(String::NewFromUtf8(isolate, "colorTable"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("colorTable requires 6 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLsizei width = args[2]->Int32Value();
		GLenum format = args[3]->Uint32Value();
		GLenum type = args[4]->Uint32Value();

		void* table = nullptr;
		if (args[5]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[5]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			table = reinterpret_cast<void*>(bdata);
		} else if (args[5]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[5]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			table = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glColorTable): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glColorTable(target, internalformat, width, format, type, table);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "colorSubTable"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("colorSubTable requires 6 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei start = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLenum format = args[3]->Uint32Value();
		GLenum type = args[4]->Uint32Value();

		void* data = nullptr;
		if (args[5]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[5]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[5]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[5]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glColorSubTable): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glColorSubTable(target, start, count, format, type, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "colorTableParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("colorTableParameteriv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glColorTableParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glColorTableParameteriv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "colorTableParameterfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("colorTableParameterfv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfloat* params = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glColorTableParameterfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glColorTableParameterfv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "copyColorSubTable"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("copyColorSubTable requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei start = args[1]->Int32Value();
		GLint x = args[2]->Int32Value();
		GLint y = args[3]->Int32Value();
		GLsizei width = args[4]->Int32Value();

		glCopyColorSubTable(target, start, x, y, width);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "copyColorTable"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("copyColorTable requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLint x = args[2]->Int32Value();
		GLint y = args[3]->Int32Value();
		GLsizei width = args[4]->Int32Value();

		glCopyColorTable(target, internalformat, x, y, width);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getColorTable"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getColorTable requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum format = args[1]->Uint32Value();
		GLenum type = args[2]->Uint32Value();

		void* table = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			table = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			table = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetColorTable): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glGetColorTable(target, format, type, table);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getColorTableParameterfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getColorTableParameterfv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfloat* params = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glGetColorTableParameterfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glGetColorTableParameterfv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getColorTableParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getColorTableParameteriv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetColorTableParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetColorTableParameteriv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "histogram"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("histogram requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei width = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLboolean sink = GLboolean(args[3]->Uint32Value());

		glHistogram(target, width, internalformat, sink);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "resetHistogram"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("resetHistogram requires 1 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		glResetHistogram(target);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getHistogram"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("getHistogram requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLboolean reset = GLboolean(args[1]->Uint32Value());
		GLenum format = args[2]->Uint32Value();
		GLenum type = args[3]->Uint32Value();

		void* values = nullptr;
		if (args[4]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[4]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			values = reinterpret_cast<void*>(bdata);
		} else if (args[4]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[4]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			values = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetHistogram): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glGetHistogram(target, reset, format, type, values);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getHistogramParameterfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getHistogramParameterfv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfloat* params = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glGetHistogramParameterfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glGetHistogramParameterfv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getHistogramParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getHistogramParameteriv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetHistogramParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetHistogramParameteriv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "minmax"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("minmax requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLboolean sink = GLboolean(args[2]->Uint32Value());

		glMinmax(target, internalformat, sink);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "resetMinmax"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("resetMinmax requires 1 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		glResetMinmax(target);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getMinmaxParameterfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getMinmaxParameterfv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfloat* params = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glGetMinmaxParameterfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glGetMinmaxParameterfv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getMinmaxParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getMinmaxParameteriv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetMinmaxParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetMinmaxParameteriv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "convolutionFilter1D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("convolutionFilter1D requires 6 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLsizei width = args[2]->Int32Value();
		GLenum format = args[3]->Uint32Value();
		GLenum type = args[4]->Uint32Value();

		void* image = nullptr;
		if (args[5]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[5]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			image = reinterpret_cast<void*>(bdata);
		} else if (args[5]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[5]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			image = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glConvolutionFilter1D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glConvolutionFilter1D(target, internalformat, width, format, type, image);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "convolutionFilter2D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("convolutionFilter2D requires 7 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLsizei width = args[2]->Int32Value();
		GLsizei height = args[3]->Int32Value();
		GLenum format = args[4]->Uint32Value();
		GLenum type = args[5]->Uint32Value();

		void* image = nullptr;
		if (args[6]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[6]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			image = reinterpret_cast<void*>(bdata);
		} else if (args[6]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[6]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			image = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glConvolutionFilter2D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glConvolutionFilter2D(target, internalformat, width, height, format, type, image);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "convolutionParameterf"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("convolutionParameterf requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLfloat params = GLfloat(args[2]->NumberValue());

		glConvolutionParameterf(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "convolutionParameterfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("convolutionParameterfv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfloat* params = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glConvolutionParameterfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glConvolutionParameterfv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "convolutionParameteri"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("convolutionParameteri requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLint params = args[2]->Int32Value();

		glConvolutionParameteri(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "convolutionParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("convolutionParameteriv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glConvolutionParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glConvolutionParameteriv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "copyConvolutionFilter1D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("copyConvolutionFilter1D requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLint x = args[2]->Int32Value();
		GLint y = args[3]->Int32Value();
		GLsizei width = args[4]->Int32Value();

		glCopyConvolutionFilter1D(target, internalformat, x, y, width);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "copyConvolutionFilter2D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("copyConvolutionFilter2D requires 6 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLint x = args[2]->Int32Value();
		GLint y = args[3]->Int32Value();
		GLsizei width = args[4]->Int32Value();
		GLsizei height = args[5]->Int32Value();

		glCopyConvolutionFilter2D(target, internalformat, x, y, width, height);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getConvolutionFilter"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getConvolutionFilter requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum format = args[1]->Uint32Value();
		GLenum type = args[2]->Uint32Value();

		void* image = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			image = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			image = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetConvolutionFilter): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glGetConvolutionFilter(target, format, type, image);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getConvolutionParameterfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getConvolutionParameterfv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfloat* params = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glGetConvolutionParameterfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glGetConvolutionParameterfv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getConvolutionParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getConvolutionParameteriv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetConvolutionParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetConvolutionParameteriv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "separableFilter2D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 8) {
			V8Helper::_instance->throwException("separableFilter2D requires 8 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLsizei width = args[2]->Int32Value();
		GLsizei height = args[3]->Int32Value();
		GLenum format = args[4]->Uint32Value();
		GLenum type = args[5]->Uint32Value();

		void* row = nullptr;
		if (args[6]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[6]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			row = reinterpret_cast<void*>(bdata);
		} else if (args[6]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[6]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			row = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glSeparableFilter2D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		void* column = nullptr;
		if (args[7]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[7]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			column = reinterpret_cast<void*>(bdata);
		} else if (args[7]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[7]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			column = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glSeparableFilter2D): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glSeparableFilter2D(target, internalformat, width, height, format, type, row, column);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getSeparableFilter"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("getSeparableFilter requires 6 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum format = args[1]->Uint32Value();
		GLenum type = args[2]->Uint32Value();

		void* row = nullptr;
		if (args[3]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[3]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			row = reinterpret_cast<void*>(bdata);
		} else if (args[3]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[3]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			row = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetSeparableFilter): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		void* column = nullptr;
		if (args[4]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[4]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			column = reinterpret_cast<void*>(bdata);
		} else if (args[4]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[4]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			column = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetSeparableFilter): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		void* span = nullptr;
		if (args[5]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[5]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			span = reinterpret_cast<void*>(bdata);
		} else if (args[5]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[5]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			span = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetSeparableFilter): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glGetSeparableFilter(target, format, type, row, column, span);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getMinmax"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("getMinmax requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLboolean reset = GLboolean(args[1]->Uint32Value());
		GLenum format = args[2]->Uint32Value();
		GLenum types = args[3]->Uint32Value();

		void* values = nullptr;
		if (args[4]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[4]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			values = reinterpret_cast<void*>(bdata);
		} else if (args[4]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[4]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			values = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetMinmax): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glGetMinmax(target, reset, format, types, values);
	}));



	// empty / skipped / ignored: GL_ARB_indirect_parameters
	// empty / skipped / ignored: GL_ARB_instanced_arrays
	/* ------------------------------ GL_ARB_internalformat_query ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("NUM_SAMPLE_COUNTS", GL_NUM_SAMPLE_COUNTS);

	tpl->Set(String::NewFromUtf8(isolate, "getInternalformativ"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("getInternalformativ requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLenum pname = args[2]->Uint32Value();
		GLsizei bufSize = args[3]->Int32Value();

		GLint* params = nullptr;
		if (args[4]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[4]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetInternalformativ): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetInternalformativ(target, internalformat, pname, bufSize, params);
	}));



	/* ------------------------------ GL_ARB_internalformat_query2 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_SUPPORTED", GL_INTERNALFORMAT_SUPPORTED);
	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_PREFERRED", GL_INTERNALFORMAT_PREFERRED);
	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_RED_SIZE", GL_INTERNALFORMAT_RED_SIZE);
	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_GREEN_SIZE", GL_INTERNALFORMAT_GREEN_SIZE);
	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_BLUE_SIZE", GL_INTERNALFORMAT_BLUE_SIZE);
	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_ALPHA_SIZE", GL_INTERNALFORMAT_ALPHA_SIZE);
	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_DEPTH_SIZE", GL_INTERNALFORMAT_DEPTH_SIZE);
	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_STENCIL_SIZE", GL_INTERNALFORMAT_STENCIL_SIZE);
	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_SHARED_SIZE", GL_INTERNALFORMAT_SHARED_SIZE);
	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_RED_TYPE", GL_INTERNALFORMAT_RED_TYPE);
	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_GREEN_TYPE", GL_INTERNALFORMAT_GREEN_TYPE);
	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_BLUE_TYPE", GL_INTERNALFORMAT_BLUE_TYPE);
	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_ALPHA_TYPE", GL_INTERNALFORMAT_ALPHA_TYPE);
	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_DEPTH_TYPE", GL_INTERNALFORMAT_DEPTH_TYPE);
	CREATE_CONSTANT_ACCESSOR("INTERNALFORMAT_STENCIL_TYPE", GL_INTERNALFORMAT_STENCIL_TYPE);
	CREATE_CONSTANT_ACCESSOR("MAX_WIDTH", GL_MAX_WIDTH);
	CREATE_CONSTANT_ACCESSOR("MAX_HEIGHT", GL_MAX_HEIGHT);
	CREATE_CONSTANT_ACCESSOR("MAX_DEPTH", GL_MAX_DEPTH);
	CREATE_CONSTANT_ACCESSOR("MAX_LAYERS", GL_MAX_LAYERS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_DIMENSIONS", GL_MAX_COMBINED_DIMENSIONS);
	CREATE_CONSTANT_ACCESSOR("COLOR_COMPONENTS", GL_COLOR_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("DEPTH_COMPONENTS", GL_DEPTH_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("STENCIL_COMPONENTS", GL_STENCIL_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("COLOR_RENDERABLE", GL_COLOR_RENDERABLE);
	CREATE_CONSTANT_ACCESSOR("DEPTH_RENDERABLE", GL_DEPTH_RENDERABLE);
	CREATE_CONSTANT_ACCESSOR("STENCIL_RENDERABLE", GL_STENCIL_RENDERABLE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_RENDERABLE", GL_FRAMEBUFFER_RENDERABLE);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_RENDERABLE_LAYERED", GL_FRAMEBUFFER_RENDERABLE_LAYERED);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_BLEND", GL_FRAMEBUFFER_BLEND);
	CREATE_CONSTANT_ACCESSOR("READ_PIXELS", GL_READ_PIXELS);
	CREATE_CONSTANT_ACCESSOR("READ_PIXELS_FORMAT", GL_READ_PIXELS_FORMAT);
	CREATE_CONSTANT_ACCESSOR("READ_PIXELS_TYPE", GL_READ_PIXELS_TYPE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_IMAGE_FORMAT", GL_TEXTURE_IMAGE_FORMAT);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_IMAGE_TYPE", GL_TEXTURE_IMAGE_TYPE);
	CREATE_CONSTANT_ACCESSOR("GET_TEXTURE_IMAGE_FORMAT", GL_GET_TEXTURE_IMAGE_FORMAT);
	CREATE_CONSTANT_ACCESSOR("GET_TEXTURE_IMAGE_TYPE", GL_GET_TEXTURE_IMAGE_TYPE);
	CREATE_CONSTANT_ACCESSOR("MIPMAP", GL_MIPMAP);
	CREATE_CONSTANT_ACCESSOR("MANUAL_GENERATE_MIPMAP", GL_MANUAL_GENERATE_MIPMAP);
	CREATE_CONSTANT_ACCESSOR("AUTO_GENERATE_MIPMAP", GL_AUTO_GENERATE_MIPMAP);
	CREATE_CONSTANT_ACCESSOR("COLOR_ENCODING", GL_COLOR_ENCODING);
	CREATE_CONSTANT_ACCESSOR("SRGB_READ", GL_SRGB_READ);
	CREATE_CONSTANT_ACCESSOR("SRGB_WRITE", GL_SRGB_WRITE);
	CREATE_CONSTANT_ACCESSOR("FILTER", GL_FILTER);
	CREATE_CONSTANT_ACCESSOR("VERTEX_TEXTURE", GL_VERTEX_TEXTURE);
	CREATE_CONSTANT_ACCESSOR("TESS_CONTROL_TEXTURE", GL_TESS_CONTROL_TEXTURE);
	CREATE_CONSTANT_ACCESSOR("TESS_EVALUATION_TEXTURE", GL_TESS_EVALUATION_TEXTURE);
	CREATE_CONSTANT_ACCESSOR("GEOMETRY_TEXTURE", GL_GEOMETRY_TEXTURE);
	CREATE_CONSTANT_ACCESSOR("FRAGMENT_TEXTURE", GL_FRAGMENT_TEXTURE);
	CREATE_CONSTANT_ACCESSOR("COMPUTE_TEXTURE", GL_COMPUTE_TEXTURE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_SHADOW", GL_TEXTURE_SHADOW);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_GATHER", GL_TEXTURE_GATHER);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_GATHER_SHADOW", GL_TEXTURE_GATHER_SHADOW);
	CREATE_CONSTANT_ACCESSOR("SHADER_IMAGE_LOAD", GL_SHADER_IMAGE_LOAD);
	CREATE_CONSTANT_ACCESSOR("SHADER_IMAGE_STORE", GL_SHADER_IMAGE_STORE);
	CREATE_CONSTANT_ACCESSOR("SHADER_IMAGE_ATOMIC", GL_SHADER_IMAGE_ATOMIC);
	CREATE_CONSTANT_ACCESSOR("IMAGE_TEXEL_SIZE", GL_IMAGE_TEXEL_SIZE);
	CREATE_CONSTANT_ACCESSOR("IMAGE_COMPATIBILITY_CLASS", GL_IMAGE_COMPATIBILITY_CLASS);
	CREATE_CONSTANT_ACCESSOR("IMAGE_PIXEL_FORMAT", GL_IMAGE_PIXEL_FORMAT);
	CREATE_CONSTANT_ACCESSOR("IMAGE_PIXEL_TYPE", GL_IMAGE_PIXEL_TYPE);
	CREATE_CONSTANT_ACCESSOR("SIMULTANEOUS_TEXTURE_AND_DEPTH_TEST", GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_TEST);
	CREATE_CONSTANT_ACCESSOR("SIMULTANEOUS_TEXTURE_AND_STENCIL_TEST", GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_TEST);
	CREATE_CONSTANT_ACCESSOR("SIMULTANEOUS_TEXTURE_AND_DEPTH_WRITE", GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_WRITE);
	CREATE_CONSTANT_ACCESSOR("SIMULTANEOUS_TEXTURE_AND_STENCIL_WRITE", GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_WRITE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_COMPRESSED_BLOCK_WIDTH", GL_TEXTURE_COMPRESSED_BLOCK_WIDTH);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_COMPRESSED_BLOCK_HEIGHT", GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_COMPRESSED_BLOCK_SIZE", GL_TEXTURE_COMPRESSED_BLOCK_SIZE);
	CREATE_CONSTANT_ACCESSOR("CLEAR_BUFFER", GL_CLEAR_BUFFER);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_VIEW", GL_TEXTURE_VIEW);
	CREATE_CONSTANT_ACCESSOR("VIEW_COMPATIBILITY_CLASS", GL_VIEW_COMPATIBILITY_CLASS);
	CREATE_CONSTANT_ACCESSOR("FULL_SUPPORT", GL_FULL_SUPPORT);
	CREATE_CONSTANT_ACCESSOR("CAVEAT_SUPPORT", GL_CAVEAT_SUPPORT);
	CREATE_CONSTANT_ACCESSOR("IMAGE_CLASS_4_X_32", GL_IMAGE_CLASS_4_X_32);
	CREATE_CONSTANT_ACCESSOR("IMAGE_CLASS_2_X_32", GL_IMAGE_CLASS_2_X_32);
	CREATE_CONSTANT_ACCESSOR("IMAGE_CLASS_1_X_32", GL_IMAGE_CLASS_1_X_32);
	CREATE_CONSTANT_ACCESSOR("IMAGE_CLASS_4_X_16", GL_IMAGE_CLASS_4_X_16);
	CREATE_CONSTANT_ACCESSOR("IMAGE_CLASS_2_X_16", GL_IMAGE_CLASS_2_X_16);
	CREATE_CONSTANT_ACCESSOR("IMAGE_CLASS_1_X_16", GL_IMAGE_CLASS_1_X_16);
	CREATE_CONSTANT_ACCESSOR("IMAGE_CLASS_4_X_8", GL_IMAGE_CLASS_4_X_8);
	CREATE_CONSTANT_ACCESSOR("IMAGE_CLASS_2_X_8", GL_IMAGE_CLASS_2_X_8);
	CREATE_CONSTANT_ACCESSOR("IMAGE_CLASS_1_X_8", GL_IMAGE_CLASS_1_X_8);
	CREATE_CONSTANT_ACCESSOR("IMAGE_CLASS_11_11_10", GL_IMAGE_CLASS_11_11_10);
	CREATE_CONSTANT_ACCESSOR("IMAGE_CLASS_10_10_10_2", GL_IMAGE_CLASS_10_10_10_2);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_128_BITS", GL_VIEW_CLASS_128_BITS);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_96_BITS", GL_VIEW_CLASS_96_BITS);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_64_BITS", GL_VIEW_CLASS_64_BITS);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_48_BITS", GL_VIEW_CLASS_48_BITS);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_32_BITS", GL_VIEW_CLASS_32_BITS);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_24_BITS", GL_VIEW_CLASS_24_BITS);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_16_BITS", GL_VIEW_CLASS_16_BITS);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_8_BITS", GL_VIEW_CLASS_8_BITS);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_S3TC_DXT1_RGB", GL_VIEW_CLASS_S3TC_DXT1_RGB);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_S3TC_DXT1_RGBA", GL_VIEW_CLASS_S3TC_DXT1_RGBA);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_S3TC_DXT3_RGBA", GL_VIEW_CLASS_S3TC_DXT3_RGBA);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_S3TC_DXT5_RGBA", GL_VIEW_CLASS_S3TC_DXT5_RGBA);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_RGTC1_RED", GL_VIEW_CLASS_RGTC1_RED);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_RGTC2_RG", GL_VIEW_CLASS_RGTC2_RG);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_BPTC_UNORM", GL_VIEW_CLASS_BPTC_UNORM);
	CREATE_CONSTANT_ACCESSOR("VIEW_CLASS_BPTC_FLOAT", GL_VIEW_CLASS_BPTC_FLOAT);




	/* ------------------------------ GL_ARB_invalidate_subdata ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "invalidateBufferData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("invalidateBufferData requires 1 arguments");
			return;
		}

		GLuint buffer = args[0]->Uint32Value();

		glInvalidateBufferData(buffer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "invalidateBufferSubData"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("invalidateBufferSubData requires 3 arguments");
			return;
		}

		GLuint buffer = args[0]->Uint32Value();
		GLintptr offset = GLintptr(args[1]->Int32Value());
		GLsizeiptr length = GLsizeiptr(args[2]->Int32Value());

		glInvalidateBufferSubData(buffer, offset, length);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "invalidateFramebuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("invalidateFramebuffer requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei numAttachments = args[1]->Int32Value();

		GLenum* attachments = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			attachments = reinterpret_cast<GLenum*>(bdata);
		} else {
			cout << "ERROR(glInvalidateFramebuffer): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glInvalidateFramebuffer(target, numAttachments, attachments);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "invalidateSubFramebuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("invalidateSubFramebuffer requires 7 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei numAttachments = args[1]->Int32Value();

		GLenum* attachments = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			attachments = reinterpret_cast<GLenum*>(bdata);
		} else {
			cout << "ERROR(glInvalidateSubFramebuffer): array must be of type Uint32Array" << endl;
			exit(1);
		}

		GLint x = args[3]->Int32Value();
		GLint y = args[4]->Int32Value();
		GLsizei width = args[5]->Int32Value();
		GLsizei height = args[6]->Int32Value();

		glInvalidateSubFramebuffer(target, numAttachments, attachments, x, y, width, height);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "invalidateTexImage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("invalidateTexImage requires 2 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();

		glInvalidateTexImage(texture, level);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "invalidateTexSubImage"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 8) {
			V8Helper::_instance->throwException("invalidateTexSubImage requires 8 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLint zoffset = args[4]->Int32Value();
		GLsizei width = args[5]->Int32Value();
		GLsizei height = args[6]->Int32Value();
		GLsizei depth = args[7]->Int32Value();

		glInvalidateTexSubImage(texture, level, xoffset, yoffset, zoffset, width, height, depth);
	}));



	/* ------------------------------ GL_ARB_map_buffer_alignment ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("MIN_MAP_BUFFER_ALIGNMENT", GL_MIN_MAP_BUFFER_ALIGNMENT);



	/* ------------------------------ GL_ARB_map_buffer_range ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("MAP_READ_BIT", GL_MAP_READ_BIT);
	CREATE_CONSTANT_ACCESSOR("MAP_WRITE_BIT", GL_MAP_WRITE_BIT);
	CREATE_CONSTANT_ACCESSOR("MAP_INVALIDATE_RANGE_BIT", GL_MAP_INVALIDATE_RANGE_BIT);
	CREATE_CONSTANT_ACCESSOR("MAP_INVALIDATE_BUFFER_BIT", GL_MAP_INVALIDATE_BUFFER_BIT);
	CREATE_CONSTANT_ACCESSOR("MAP_FLUSH_EXPLICIT_BIT", GL_MAP_FLUSH_EXPLICIT_BIT);
	CREATE_CONSTANT_ACCESSOR("MAP_UNSYNCHRONIZED_BIT", GL_MAP_UNSYNCHRONIZED_BIT);

	tpl->Set(String::NewFromUtf8(isolate, "flushMappedBufferRange"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("flushMappedBufferRange requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLintptr offset = GLintptr(args[1]->Int32Value());
		GLsizeiptr length = GLsizeiptr(args[2]->Int32Value());

		glFlushMappedBufferRange(target, offset, length);
	}));



	// empty / skipped / ignored: GL_ARB_matrix_palette
	// empty / skipped / ignored: GL_ARB_multisample
	// empty / skipped / ignored: GL_ARB_multitexture
	/* ------------------------------ GL_ARB_multi_bind ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "bindBuffersBase"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("bindBuffersBase requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint first = args[1]->Uint32Value();
		GLsizei count = args[2]->Int32Value();

		GLuint* buffers = nullptr;
		if (args[3]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[3]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			buffers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glBindBuffersBase): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glBindBuffersBase(target, first, count, buffers);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "bindImageTextures"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("bindImageTextures requires 3 arguments");
			return;
		}

		GLuint first = args[0]->Uint32Value();
		GLsizei count = args[1]->Int32Value();

		GLuint* textures = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			textures = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glBindImageTextures): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glBindImageTextures(first, count, textures);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "bindSamplers"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("bindSamplers requires 3 arguments");
			return;
		}

		GLuint first = args[0]->Uint32Value();
		GLsizei count = args[1]->Int32Value();

		GLuint* samplers = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			samplers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glBindSamplers): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glBindSamplers(first, count, samplers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "bindTextures"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("bindTextures requires 3 arguments");
			return;
		}

		GLuint first = args[0]->Uint32Value();
		GLsizei count = args[1]->Int32Value();

		GLuint* textures = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			textures = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glBindTextures): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glBindTextures(first, count, textures);
	}));




	/* ------------------------------ GL_ARB_multi_draw_indirect ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "multiDrawArraysIndirect"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("multiDrawArraysIndirect requires 4 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();

		void* indirect = nullptr;
		if (args[1]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[1]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			indirect = reinterpret_cast<void*>(bdata);
		} else if (args[1]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[1]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indirect = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glMultiDrawArraysIndirect): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLsizei primcount = args[2]->Int32Value();
		GLsizei stride = args[3]->Int32Value();

		glMultiDrawArraysIndirect(mode, indirect, primcount, stride);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiDrawElementsIndirect"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("multiDrawElementsIndirect requires 5 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();

		void* indirect = nullptr;
		if (args[2]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[2]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			indirect = reinterpret_cast<void*>(bdata);
		} else if (args[2]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[2]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indirect = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glMultiDrawElementsIndirect): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLsizei primcount = args[3]->Int32Value();
		GLsizei stride = args[4]->Int32Value();

		glMultiDrawElementsIndirect(mode, type, indirect, primcount, stride);
	}));



	// empty / skipped / ignored: GL_ARB_occlusion_query
	/* ------------------------------ GL_ARB_occlusion_query2 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("ANY_SAMPLES_PASSED", GL_ANY_SAMPLES_PASSED);



	// empty / skipped / ignored: GL_ARB_parallel_shader_compile
	/* ------------------------------ GL_ARB_pipeline_statistics_query ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("GEOMETRY_SHADER_INVOCATIONS", GL_GEOMETRY_SHADER_INVOCATIONS);



	// empty / skipped / ignored: GL_ARB_pixel_buffer_object
	// empty / skipped / ignored: GL_ARB_point_parameters
	// empty / skipped / ignored: GL_ARB_point_sprite
	/* ------------------------------ GL_ARB_polygon_offset_clamp ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("POLYGON_OFFSET_CLAMP", GL_POLYGON_OFFSET_CLAMP);

	tpl->Set(String::NewFromUtf8(isolate, "polygonOffsetClamp"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("polygonOffsetClamp requires 3 arguments");
			return;
		}

		GLfloat factor = GLfloat(args[0]->NumberValue());
		GLfloat units = GLfloat(args[1]->NumberValue());
		GLfloat clamp = GLfloat(args[2]->NumberValue());

		glPolygonOffsetClamp(factor, units, clamp);
	}));



	// empty / skipped / ignored: GL_ARB_post_depth_coverage
	/* ------------------------------ GL_ARB_program_interface_query ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("UNIFORM", GL_UNIFORM);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BLOCK", GL_UNIFORM_BLOCK);
	CREATE_CONSTANT_ACCESSOR("PROGRAM_INPUT", GL_PROGRAM_INPUT);
	CREATE_CONSTANT_ACCESSOR("PROGRAM_OUTPUT", GL_PROGRAM_OUTPUT);
	CREATE_CONSTANT_ACCESSOR("BUFFER_VARIABLE", GL_BUFFER_VARIABLE);
	CREATE_CONSTANT_ACCESSOR("SHADER_STORAGE_BLOCK", GL_SHADER_STORAGE_BLOCK);
	CREATE_CONSTANT_ACCESSOR("IS_PER_PATCH", GL_IS_PER_PATCH);
	CREATE_CONSTANT_ACCESSOR("VERTEX_SUBROUTINE", GL_VERTEX_SUBROUTINE);
	CREATE_CONSTANT_ACCESSOR("TESS_CONTROL_SUBROUTINE", GL_TESS_CONTROL_SUBROUTINE);
	CREATE_CONSTANT_ACCESSOR("TESS_EVALUATION_SUBROUTINE", GL_TESS_EVALUATION_SUBROUTINE);
	CREATE_CONSTANT_ACCESSOR("GEOMETRY_SUBROUTINE", GL_GEOMETRY_SUBROUTINE);
	CREATE_CONSTANT_ACCESSOR("FRAGMENT_SUBROUTINE", GL_FRAGMENT_SUBROUTINE);
	CREATE_CONSTANT_ACCESSOR("COMPUTE_SUBROUTINE", GL_COMPUTE_SUBROUTINE);
	CREATE_CONSTANT_ACCESSOR("VERTEX_SUBROUTINE_UNIFORM", GL_VERTEX_SUBROUTINE_UNIFORM);
	CREATE_CONSTANT_ACCESSOR("TESS_CONTROL_SUBROUTINE_UNIFORM", GL_TESS_CONTROL_SUBROUTINE_UNIFORM);
	CREATE_CONSTANT_ACCESSOR("TESS_EVALUATION_SUBROUTINE_UNIFORM", GL_TESS_EVALUATION_SUBROUTINE_UNIFORM);
	CREATE_CONSTANT_ACCESSOR("GEOMETRY_SUBROUTINE_UNIFORM", GL_GEOMETRY_SUBROUTINE_UNIFORM);
	CREATE_CONSTANT_ACCESSOR("FRAGMENT_SUBROUTINE_UNIFORM", GL_FRAGMENT_SUBROUTINE_UNIFORM);
	CREATE_CONSTANT_ACCESSOR("COMPUTE_SUBROUTINE_UNIFORM", GL_COMPUTE_SUBROUTINE_UNIFORM);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_VARYING", GL_TRANSFORM_FEEDBACK_VARYING);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_RESOURCES", GL_ACTIVE_RESOURCES);
	CREATE_CONSTANT_ACCESSOR("MAX_NAME_LENGTH", GL_MAX_NAME_LENGTH);
	CREATE_CONSTANT_ACCESSOR("MAX_NUM_ACTIVE_VARIABLES", GL_MAX_NUM_ACTIVE_VARIABLES);
	CREATE_CONSTANT_ACCESSOR("MAX_NUM_COMPATIBLE_SUBROUTINES", GL_MAX_NUM_COMPATIBLE_SUBROUTINES);
	CREATE_CONSTANT_ACCESSOR("NAME_LENGTH", GL_NAME_LENGTH);
	CREATE_CONSTANT_ACCESSOR("TYPE", GL_TYPE);
	CREATE_CONSTANT_ACCESSOR("ARRAY_SIZE", GL_ARRAY_SIZE);
	CREATE_CONSTANT_ACCESSOR("OFFSET", GL_OFFSET);
	CREATE_CONSTANT_ACCESSOR("BLOCK_INDEX", GL_BLOCK_INDEX);
	CREATE_CONSTANT_ACCESSOR("ARRAY_STRIDE", GL_ARRAY_STRIDE);
	CREATE_CONSTANT_ACCESSOR("MATRIX_STRIDE", GL_MATRIX_STRIDE);
	CREATE_CONSTANT_ACCESSOR("IS_ROW_MAJOR", GL_IS_ROW_MAJOR);
	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BUFFER_INDEX", GL_ATOMIC_COUNTER_BUFFER_INDEX);
	CREATE_CONSTANT_ACCESSOR("BUFFER_BINDING", GL_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("BUFFER_DATA_SIZE", GL_BUFFER_DATA_SIZE);
	CREATE_CONSTANT_ACCESSOR("NUM_ACTIVE_VARIABLES", GL_NUM_ACTIVE_VARIABLES);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_VARIABLES", GL_ACTIVE_VARIABLES);
	CREATE_CONSTANT_ACCESSOR("REFERENCED_BY_VERTEX_SHADER", GL_REFERENCED_BY_VERTEX_SHADER);
	CREATE_CONSTANT_ACCESSOR("REFERENCED_BY_TESS_CONTROL_SHADER", GL_REFERENCED_BY_TESS_CONTROL_SHADER);
	CREATE_CONSTANT_ACCESSOR("REFERENCED_BY_TESS_EVALUATION_SHADER", GL_REFERENCED_BY_TESS_EVALUATION_SHADER);
	CREATE_CONSTANT_ACCESSOR("REFERENCED_BY_GEOMETRY_SHADER", GL_REFERENCED_BY_GEOMETRY_SHADER);
	CREATE_CONSTANT_ACCESSOR("REFERENCED_BY_FRAGMENT_SHADER", GL_REFERENCED_BY_FRAGMENT_SHADER);
	CREATE_CONSTANT_ACCESSOR("REFERENCED_BY_COMPUTE_SHADER", GL_REFERENCED_BY_COMPUTE_SHADER);
	CREATE_CONSTANT_ACCESSOR("TOP_LEVEL_ARRAY_SIZE", GL_TOP_LEVEL_ARRAY_SIZE);
	CREATE_CONSTANT_ACCESSOR("TOP_LEVEL_ARRAY_STRIDE", GL_TOP_LEVEL_ARRAY_STRIDE);
	CREATE_CONSTANT_ACCESSOR("LOCATION", GL_LOCATION);
	CREATE_CONSTANT_ACCESSOR("LOCATION_INDEX", GL_LOCATION_INDEX);

	tpl->Set(String::NewFromUtf8(isolate, "getProgramInterfaceiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getProgramInterfaceiv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLenum programInterface = args[1]->Uint32Value();
		GLenum pname = args[2]->Uint32Value();

		GLint* params = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetProgramInterfaceiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetProgramInterfaceiv(program, programInterface, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getProgramResourceName"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("getProgramResourceName requires 6 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLenum programInterface = args[1]->Uint32Value();
		GLuint index = args[2]->Uint32Value();
		GLsizei bufSize = args[3]->Int32Value();

		GLsizei* length = nullptr;
		if (args[4]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[4]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetProgramResourceName): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLchar* name = nullptr;
		if (args[5]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[5]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			name = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetProgramResourceName): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetProgramResourceName(program, programInterface, index, bufSize, length, name);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getProgramResourceiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 8) {
			V8Helper::_instance->throwException("getProgramResourceiv requires 8 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLenum programInterface = args[1]->Uint32Value();
		GLuint index = args[2]->Uint32Value();
		GLsizei propCount = args[3]->Int32Value();

		GLenum* props = nullptr;
		if (args[4]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[4]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			props = reinterpret_cast<GLenum*>(bdata);
		} else {
			cout << "ERROR(glGetProgramResourceiv): array must be of type Uint32Array" << endl;
			exit(1);
		}

		GLsizei bufSize = args[5]->Int32Value();

		GLsizei* length = nullptr;
		if (args[6]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[6]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetProgramResourceiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLint* params = nullptr;
		if (args[7]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[7]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetProgramResourceiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetProgramResourceiv(program, programInterface, index, propCount, props, bufSize, length, params);
	}));



	/* ------------------------------ GL_ARB_provoking_vertex ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("QUADS_FOLLOW_PROVOKING_VERTEX_CONVENTION", GL_QUADS_FOLLOW_PROVOKING_VERTEX_CONVENTION);
	CREATE_CONSTANT_ACCESSOR("FIRST_VERTEX_CONVENTION", GL_FIRST_VERTEX_CONVENTION);
	CREATE_CONSTANT_ACCESSOR("LAST_VERTEX_CONVENTION", GL_LAST_VERTEX_CONVENTION);
	CREATE_CONSTANT_ACCESSOR("PROVOKING_VERTEX", GL_PROVOKING_VERTEX);

	tpl->Set(String::NewFromUtf8(isolate, "provokingVertex"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("provokingVertex requires 1 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();

		glProvokingVertex(mode);
	}));



	/* ------------------------------ GL_ARB_query_buffer_object ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("QUERY_BUFFER_BARRIER_BIT", GL_QUERY_BUFFER_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("QUERY_BUFFER", GL_QUERY_BUFFER);
	CREATE_CONSTANT_ACCESSOR("QUERY_BUFFER_BINDING", GL_QUERY_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("QUERY_RESULT_NO_WAIT", GL_QUERY_RESULT_NO_WAIT);



	// empty / skipped / ignored: GL_ARB_robustness
	// empty / skipped / ignored: GL_ARB_robustness_application_isolation
	// empty / skipped / ignored: GL_ARB_robustness_share_group_isolation
	// empty / skipped / ignored: GL_ARB_robust_buffer_access_behavior
	/* ------------------------------ GL_ARB_sampler_objects ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("SAMPLER_BINDING", GL_SAMPLER_BINDING);

	tpl->Set(String::NewFromUtf8(isolate, "bindSampler"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("bindSampler requires 2 arguments");
			return;
		}

		GLuint unit = args[0]->Uint32Value();
		GLuint sampler = args[1]->Uint32Value();

		glBindSampler(unit, sampler);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "genSamplers"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("genSamplers requires 2 arguments");
			return;
		}

		GLsizei count = args[0]->Int32Value();

		GLuint* samplers = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			samplers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGenSamplers): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGenSamplers(count, samplers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getSamplerParameterIiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getSamplerParameterIiv requires 3 arguments");
			return;
		}

		GLuint sampler = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetSamplerParameterIiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetSamplerParameterIiv(sampler, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getSamplerParameterIuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getSamplerParameterIuiv requires 3 arguments");
			return;
		}

		GLuint sampler = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLuint* params = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGetSamplerParameterIuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGetSamplerParameterIuiv(sampler, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getSamplerParameterfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getSamplerParameterfv requires 3 arguments");
			return;
		}

		GLuint sampler = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfloat* params = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glGetSamplerParameterfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glGetSamplerParameterfv(sampler, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getSamplerParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getSamplerParameteriv requires 3 arguments");
			return;
		}

		GLuint sampler = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetSamplerParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetSamplerParameteriv(sampler, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "samplerParameterIiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("samplerParameterIiv requires 3 arguments");
			return;
		}

		GLuint sampler = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glSamplerParameterIiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glSamplerParameterIiv(sampler, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "samplerParameterIuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("samplerParameterIuiv requires 3 arguments");
			return;
		}

		GLuint sampler = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLuint* params = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glSamplerParameterIuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glSamplerParameterIuiv(sampler, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "samplerParameterf"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("samplerParameterf requires 3 arguments");
			return;
		}

		GLuint sampler = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLfloat param = GLfloat(args[2]->NumberValue());

		glSamplerParameterf(sampler, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "samplerParameterfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("samplerParameterfv requires 3 arguments");
			return;
		}

		GLuint sampler = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfloat* params = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glSamplerParameterfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glSamplerParameterfv(sampler, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "samplerParameteri"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("samplerParameteri requires 3 arguments");
			return;
		}

		GLuint sampler = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLint param = args[2]->Int32Value();

		glSamplerParameteri(sampler, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "samplerParameteriv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("samplerParameteriv requires 3 arguments");
			return;
		}

		GLuint sampler = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glSamplerParameteriv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glSamplerParameteriv(sampler, pname, params);
	}));



	// empty / skipped / ignored: GL_ARB_sample_locations
	// empty / skipped / ignored: GL_ARB_sample_shading
	/* ------------------------------ GL_ARB_seamless_cubemap_per_texture ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_CUBE_MAP_SEAMLESS", GL_TEXTURE_CUBE_MAP_SEAMLESS);



	/* ------------------------------ GL_ARB_seamless_cube_map ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_CUBE_MAP_SEAMLESS", GL_TEXTURE_CUBE_MAP_SEAMLESS);



	/* ------------------------------ GL_ARB_separate_shader_objects ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("VERTEX_SHADER_BIT", GL_VERTEX_SHADER_BIT);
	CREATE_CONSTANT_ACCESSOR("FRAGMENT_SHADER_BIT", GL_FRAGMENT_SHADER_BIT);
	CREATE_CONSTANT_ACCESSOR("GEOMETRY_SHADER_BIT", GL_GEOMETRY_SHADER_BIT);
	CREATE_CONSTANT_ACCESSOR("TESS_CONTROL_SHADER_BIT", GL_TESS_CONTROL_SHADER_BIT);
	CREATE_CONSTANT_ACCESSOR("TESS_EVALUATION_SHADER_BIT", GL_TESS_EVALUATION_SHADER_BIT);
	CREATE_CONSTANT_ACCESSOR("PROGRAM_SEPARABLE", GL_PROGRAM_SEPARABLE);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_PROGRAM", GL_ACTIVE_PROGRAM);
	CREATE_CONSTANT_ACCESSOR("PROGRAM_PIPELINE_BINDING", GL_PROGRAM_PIPELINE_BINDING);
	CREATE_CONSTANT_ACCESSOR("ALL_SHADER_BITS", GL_ALL_SHADER_BITS);

	tpl->Set(String::NewFromUtf8(isolate, "activeShaderProgram"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("activeShaderProgram requires 2 arguments");
			return;
		}

		GLuint pipeline = args[0]->Uint32Value();
		GLuint program = args[1]->Uint32Value();

		glActiveShaderProgram(pipeline, program);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "bindProgramPipeline"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("bindProgramPipeline requires 1 arguments");
			return;
		}

		GLuint pipeline = args[0]->Uint32Value();

		glBindProgramPipeline(pipeline);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "deleteProgramPipelines"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("deleteProgramPipelines requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* pipelines = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pipelines = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glDeleteProgramPipelines): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glDeleteProgramPipelines(n, pipelines);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "genProgramPipelines"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("genProgramPipelines requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* pipelines = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pipelines = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGenProgramPipelines): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGenProgramPipelines(n, pipelines);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getProgramPipelineInfoLog"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getProgramPipelineInfoLog requires 4 arguments");
			return;
		}

		GLuint pipeline = args[0]->Uint32Value();
		GLsizei bufSize = args[1]->Int32Value();

		GLsizei* length = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetProgramPipelineInfoLog): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLchar* infoLog = nullptr;
		if (args[3]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[3]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			infoLog = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetProgramPipelineInfoLog): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetProgramPipelineInfoLog(pipeline, bufSize, length, infoLog);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getProgramPipelineiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getProgramPipelineiv requires 3 arguments");
			return;
		}

		GLuint pipeline = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetProgramPipelineiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetProgramPipelineiv(pipeline, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform1d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("programUniform1d requires 3 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLdouble x = args[2]->NumberValue();

		glProgramUniform1d(program, location, x);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform1dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform1dv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLdouble* value = nullptr;
		if (args[3]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[3]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform1dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glProgramUniform1dv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform1f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("programUniform1f requires 3 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLfloat x = GLfloat(args[2]->NumberValue());

		glProgramUniform1f(program, location, x);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform1fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform1fv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLfloat* value = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform1fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glProgramUniform1fv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform1i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("programUniform1i requires 3 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLint x = args[2]->Int32Value();

		glProgramUniform1i(program, location, x);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform1iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform1iv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLint* value = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform1iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glProgramUniform1iv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform1ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("programUniform1ui requires 3 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLuint x = args[2]->Uint32Value();

		glProgramUniform1ui(program, location, x);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform1uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform1uiv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLuint* value = nullptr;
		if (args[3]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[3]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform1uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glProgramUniform1uiv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform2d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform2d requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLdouble x = args[2]->NumberValue();
		GLdouble y = args[3]->NumberValue();

		glProgramUniform2d(program, location, x, y);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform2dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform2dv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLdouble* value = nullptr;
		if (args[3]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[3]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform2dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glProgramUniform2dv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform2f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform2f requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLfloat x = GLfloat(args[2]->NumberValue());
		GLfloat y = GLfloat(args[3]->NumberValue());

		glProgramUniform2f(program, location, x, y);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform2fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform2fv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLfloat* value = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform2fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glProgramUniform2fv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform2i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform2i requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLint x = args[2]->Int32Value();
		GLint y = args[3]->Int32Value();

		glProgramUniform2i(program, location, x, y);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform2iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform2iv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLint* value = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform2iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glProgramUniform2iv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform2ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform2ui requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLuint x = args[2]->Uint32Value();
		GLuint y = args[3]->Uint32Value();

		glProgramUniform2ui(program, location, x, y);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform2uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform2uiv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLuint* value = nullptr;
		if (args[3]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[3]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform2uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glProgramUniform2uiv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform3d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniform3d requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLdouble x = args[2]->NumberValue();
		GLdouble y = args[3]->NumberValue();
		GLdouble z = args[4]->NumberValue();

		glProgramUniform3d(program, location, x, y, z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform3dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform3dv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLdouble* value = nullptr;
		if (args[3]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[3]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform3dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glProgramUniform3dv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform3f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniform3f requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLfloat x = GLfloat(args[2]->NumberValue());
		GLfloat y = GLfloat(args[3]->NumberValue());
		GLfloat z = GLfloat(args[4]->NumberValue());

		glProgramUniform3f(program, location, x, y, z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform3fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform3fv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLfloat* value = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform3fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glProgramUniform3fv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform3i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniform3i requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLint x = args[2]->Int32Value();
		GLint y = args[3]->Int32Value();
		GLint z = args[4]->Int32Value();

		glProgramUniform3i(program, location, x, y, z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform3iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform3iv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLint* value = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform3iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glProgramUniform3iv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform3ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniform3ui requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLuint x = args[2]->Uint32Value();
		GLuint y = args[3]->Uint32Value();
		GLuint z = args[4]->Uint32Value();

		glProgramUniform3ui(program, location, x, y, z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform3uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform3uiv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLuint* value = nullptr;
		if (args[3]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[3]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform3uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glProgramUniform3uiv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform4d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("programUniform4d requires 6 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLdouble x = args[2]->NumberValue();
		GLdouble y = args[3]->NumberValue();
		GLdouble z = args[4]->NumberValue();
		GLdouble w = args[5]->NumberValue();

		glProgramUniform4d(program, location, x, y, z, w);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform4dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform4dv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLdouble* value = nullptr;
		if (args[3]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[3]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform4dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glProgramUniform4dv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform4f"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("programUniform4f requires 6 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLfloat x = GLfloat(args[2]->NumberValue());
		GLfloat y = GLfloat(args[3]->NumberValue());
		GLfloat z = GLfloat(args[4]->NumberValue());
		GLfloat w = GLfloat(args[5]->NumberValue());

		glProgramUniform4f(program, location, x, y, z, w);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform4fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform4fv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLfloat* value = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform4fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glProgramUniform4fv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform4i"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("programUniform4i requires 6 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLint x = args[2]->Int32Value();
		GLint y = args[3]->Int32Value();
		GLint z = args[4]->Int32Value();
		GLint w = args[5]->Int32Value();

		glProgramUniform4i(program, location, x, y, z, w);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform4iv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform4iv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLint* value = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform4iv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glProgramUniform4iv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform4ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("programUniform4ui requires 6 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLuint x = args[2]->Uint32Value();
		GLuint y = args[3]->Uint32Value();
		GLuint z = args[4]->Uint32Value();
		GLuint w = args[5]->Uint32Value();

		glProgramUniform4ui(program, location, x, y, z, w);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniform4uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("programUniform4uiv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();

		GLuint* value = nullptr;
		if (args[3]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[3]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glProgramUniform4uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glProgramUniform4uiv(program, location, count, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix2dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix2dv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[4]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[4]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix2dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix2dv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix2fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix2fv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[4]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[4]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix2fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix2fv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix2x3dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix2x3dv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[4]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[4]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix2x3dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix2x3dv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix2x3fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix2x3fv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[4]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[4]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix2x3fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix2x3fv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix2x4dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix2x4dv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[4]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[4]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix2x4dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix2x4dv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix2x4fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix2x4fv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[4]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[4]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix2x4fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix2x4fv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix3dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix3dv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[4]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[4]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix3dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix3dv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix3fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix3fv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[4]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[4]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix3fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix3fv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix3x2dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix3x2dv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[4]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[4]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix3x2dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix3x2dv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix3x2fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix3x2fv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[4]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[4]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix3x2fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix3x2fv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix3x4dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix3x4dv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[4]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[4]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix3x4dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix3x4dv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix3x4fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix3x4fv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[4]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[4]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix3x4fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix3x4fv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix4dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix4dv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[4]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[4]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix4dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix4dv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix4fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix4fv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[4]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[4]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix4fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix4fv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix4x2dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix4x2dv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[4]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[4]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix4x2dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix4x2dv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix4x2fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix4x2fv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[4]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[4]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix4x2fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix4x2fv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix4x3dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix4x3dv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLdouble* value = nullptr;
		if (args[4]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[4]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix4x3dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix4x3dv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "programUniformMatrix4x3fv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("programUniformMatrix4x3fv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei count = args[2]->Int32Value();
		GLboolean transpose = GLboolean(args[3]->Uint32Value());

		GLfloat* value = nullptr;
		if (args[4]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[4]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glProgramUniformMatrix4x3fv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glProgramUniformMatrix4x3fv(program, location, count, transpose, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "useProgramStages"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("useProgramStages requires 3 arguments");
			return;
		}

		GLuint pipeline = args[0]->Uint32Value();
		GLbitfield stages = args[1]->Uint32Value();
		GLuint program = args[2]->Uint32Value();

		glUseProgramStages(pipeline, stages, program);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "validateProgramPipeline"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("validateProgramPipeline requires 1 arguments");
			return;
		}

		GLuint pipeline = args[0]->Uint32Value();

		glValidateProgramPipeline(pipeline);
	}));



	/* ------------------------------ GL_ARB_shader_atomic_counters ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BUFFER", GL_ATOMIC_COUNTER_BUFFER);
	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BUFFER_BINDING", GL_ATOMIC_COUNTER_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BUFFER_START", GL_ATOMIC_COUNTER_BUFFER_START);
	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BUFFER_SIZE", GL_ATOMIC_COUNTER_BUFFER_SIZE);
	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BUFFER_DATA_SIZE", GL_ATOMIC_COUNTER_BUFFER_DATA_SIZE);
	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTERS", GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTERS);
	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTER_INDICES", GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTER_INDICES);
	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BUFFER_REFERENCED_BY_VERTEX_SHADER", GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_VERTEX_SHADER);
	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_CONTROL_SHADER", GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_CONTROL_SHADER);
	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_EVALUATION_SHADER", GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_EVALUATION_SHADER);
	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BUFFER_REFERENCED_BY_GEOMETRY_SHADER", GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_GEOMETRY_SHADER);
	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BUFFER_REFERENCED_BY_FRAGMENT_SHADER", GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_FRAGMENT_SHADER);
	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_ATOMIC_COUNTER_BUFFERS", GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_CONTROL_ATOMIC_COUNTER_BUFFERS", GL_MAX_TESS_CONTROL_ATOMIC_COUNTER_BUFFERS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_EVALUATION_ATOMIC_COUNTER_BUFFERS", GL_MAX_TESS_EVALUATION_ATOMIC_COUNTER_BUFFERS);
	CREATE_CONSTANT_ACCESSOR("MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS", GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS);
	CREATE_CONSTANT_ACCESSOR("MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS", GL_MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_ATOMIC_COUNTER_BUFFERS", GL_MAX_COMBINED_ATOMIC_COUNTER_BUFFERS);
	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_ATOMIC_COUNTERS", GL_MAX_VERTEX_ATOMIC_COUNTERS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_CONTROL_ATOMIC_COUNTERS", GL_MAX_TESS_CONTROL_ATOMIC_COUNTERS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_EVALUATION_ATOMIC_COUNTERS", GL_MAX_TESS_EVALUATION_ATOMIC_COUNTERS);
	CREATE_CONSTANT_ACCESSOR("MAX_GEOMETRY_ATOMIC_COUNTERS", GL_MAX_GEOMETRY_ATOMIC_COUNTERS);
	CREATE_CONSTANT_ACCESSOR("MAX_FRAGMENT_ATOMIC_COUNTERS", GL_MAX_FRAGMENT_ATOMIC_COUNTERS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_ATOMIC_COUNTERS", GL_MAX_COMBINED_ATOMIC_COUNTERS);
	CREATE_CONSTANT_ACCESSOR("MAX_ATOMIC_COUNTER_BUFFER_SIZE", GL_MAX_ATOMIC_COUNTER_BUFFER_SIZE);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_ATOMIC_COUNTER_BUFFERS", GL_ACTIVE_ATOMIC_COUNTER_BUFFERS);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_ATOMIC_COUNTER_BUFFER_INDEX", GL_UNIFORM_ATOMIC_COUNTER_BUFFER_INDEX);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_ATOMIC_COUNTER", GL_UNSIGNED_INT_ATOMIC_COUNTER);
	CREATE_CONSTANT_ACCESSOR("MAX_ATOMIC_COUNTER_BUFFER_BINDINGS", GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS);

	tpl->Set(String::NewFromUtf8(isolate, "getActiveAtomicCounterBufferiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getActiveAtomicCounterBufferiv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLuint bufferIndex = args[1]->Uint32Value();
		GLenum pname = args[2]->Uint32Value();

		GLint* params = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetActiveAtomicCounterBufferiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetActiveAtomicCounterBufferiv(program, bufferIndex, pname, params);
	}));



	// empty / skipped / ignored: GL_ARB_shader_atomic_counter_ops
	// empty / skipped / ignored: GL_ARB_shader_ballot
	// empty / skipped / ignored: GL_ARB_shader_bit_encoding
	// empty / skipped / ignored: GL_ARB_shader_clock
	// empty / skipped / ignored: GL_ARB_shader_draw_parameters
	// empty / skipped / ignored: GL_ARB_shader_group_vote
	/* ------------------------------ GL_ARB_shader_image_load_store ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("VERTEX_ATTRIB_ARRAY_BARRIER_BIT", GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("ELEMENT_ARRAY_BARRIER_BIT", GL_ELEMENT_ARRAY_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BARRIER_BIT", GL_UNIFORM_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_FETCH_BARRIER_BIT", GL_TEXTURE_FETCH_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("SHADER_IMAGE_ACCESS_BARRIER_BIT", GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("COMMAND_BARRIER_BIT", GL_COMMAND_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("PIXEL_BUFFER_BARRIER_BIT", GL_PIXEL_BUFFER_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_UPDATE_BARRIER_BIT", GL_TEXTURE_UPDATE_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("BUFFER_UPDATE_BARRIER_BIT", GL_BUFFER_UPDATE_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_BARRIER_BIT", GL_FRAMEBUFFER_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_BARRIER_BIT", GL_TRANSFORM_FEEDBACK_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("ATOMIC_COUNTER_BARRIER_BIT", GL_ATOMIC_COUNTER_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("MAX_IMAGE_UNITS", GL_MAX_IMAGE_UNITS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS", GL_MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS);
	CREATE_CONSTANT_ACCESSOR("IMAGE_BINDING_NAME", GL_IMAGE_BINDING_NAME);
	CREATE_CONSTANT_ACCESSOR("IMAGE_BINDING_LEVEL", GL_IMAGE_BINDING_LEVEL);
	CREATE_CONSTANT_ACCESSOR("IMAGE_BINDING_LAYERED", GL_IMAGE_BINDING_LAYERED);
	CREATE_CONSTANT_ACCESSOR("IMAGE_BINDING_LAYER", GL_IMAGE_BINDING_LAYER);
	CREATE_CONSTANT_ACCESSOR("IMAGE_BINDING_ACCESS", GL_IMAGE_BINDING_ACCESS);
	CREATE_CONSTANT_ACCESSOR("IMAGE_1D", GL_IMAGE_1D);
	CREATE_CONSTANT_ACCESSOR("IMAGE_2D", GL_IMAGE_2D);
	CREATE_CONSTANT_ACCESSOR("IMAGE_3D", GL_IMAGE_3D);
	CREATE_CONSTANT_ACCESSOR("IMAGE_2D_RECT", GL_IMAGE_2D_RECT);
	CREATE_CONSTANT_ACCESSOR("IMAGE_CUBE", GL_IMAGE_CUBE);
	CREATE_CONSTANT_ACCESSOR("IMAGE_BUFFER", GL_IMAGE_BUFFER);
	CREATE_CONSTANT_ACCESSOR("IMAGE_1D_ARRAY", GL_IMAGE_1D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("IMAGE_2D_ARRAY", GL_IMAGE_2D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("IMAGE_CUBE_MAP_ARRAY", GL_IMAGE_CUBE_MAP_ARRAY);
	CREATE_CONSTANT_ACCESSOR("IMAGE_2D_MULTISAMPLE", GL_IMAGE_2D_MULTISAMPLE);
	CREATE_CONSTANT_ACCESSOR("IMAGE_2D_MULTISAMPLE_ARRAY", GL_IMAGE_2D_MULTISAMPLE_ARRAY);
	CREATE_CONSTANT_ACCESSOR("INT_IMAGE_1D", GL_INT_IMAGE_1D);
	CREATE_CONSTANT_ACCESSOR("INT_IMAGE_2D", GL_INT_IMAGE_2D);
	CREATE_CONSTANT_ACCESSOR("INT_IMAGE_3D", GL_INT_IMAGE_3D);
	CREATE_CONSTANT_ACCESSOR("INT_IMAGE_2D_RECT", GL_INT_IMAGE_2D_RECT);
	CREATE_CONSTANT_ACCESSOR("INT_IMAGE_CUBE", GL_INT_IMAGE_CUBE);
	CREATE_CONSTANT_ACCESSOR("INT_IMAGE_BUFFER", GL_INT_IMAGE_BUFFER);
	CREATE_CONSTANT_ACCESSOR("INT_IMAGE_1D_ARRAY", GL_INT_IMAGE_1D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("INT_IMAGE_2D_ARRAY", GL_INT_IMAGE_2D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("INT_IMAGE_CUBE_MAP_ARRAY", GL_INT_IMAGE_CUBE_MAP_ARRAY);
	CREATE_CONSTANT_ACCESSOR("INT_IMAGE_2D_MULTISAMPLE", GL_INT_IMAGE_2D_MULTISAMPLE);
	CREATE_CONSTANT_ACCESSOR("INT_IMAGE_2D_MULTISAMPLE_ARRAY", GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_IMAGE_1D", GL_UNSIGNED_INT_IMAGE_1D);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_IMAGE_2D", GL_UNSIGNED_INT_IMAGE_2D);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_IMAGE_3D", GL_UNSIGNED_INT_IMAGE_3D);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_IMAGE_2D_RECT", GL_UNSIGNED_INT_IMAGE_2D_RECT);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_IMAGE_CUBE", GL_UNSIGNED_INT_IMAGE_CUBE);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_IMAGE_BUFFER", GL_UNSIGNED_INT_IMAGE_BUFFER);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_IMAGE_1D_ARRAY", GL_UNSIGNED_INT_IMAGE_1D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_IMAGE_2D_ARRAY", GL_UNSIGNED_INT_IMAGE_2D_ARRAY);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY", GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_IMAGE_2D_MULTISAMPLE", GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY", GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY);
	CREATE_CONSTANT_ACCESSOR("MAX_IMAGE_SAMPLES", GL_MAX_IMAGE_SAMPLES);
	CREATE_CONSTANT_ACCESSOR("IMAGE_BINDING_FORMAT", GL_IMAGE_BINDING_FORMAT);
	CREATE_CONSTANT_ACCESSOR("IMAGE_FORMAT_COMPATIBILITY_TYPE", GL_IMAGE_FORMAT_COMPATIBILITY_TYPE);
	CREATE_CONSTANT_ACCESSOR("IMAGE_FORMAT_COMPATIBILITY_BY_SIZE", GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE);
	CREATE_CONSTANT_ACCESSOR("IMAGE_FORMAT_COMPATIBILITY_BY_CLASS", GL_IMAGE_FORMAT_COMPATIBILITY_BY_CLASS);
	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_IMAGE_UNIFORMS", GL_MAX_VERTEX_IMAGE_UNIFORMS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_CONTROL_IMAGE_UNIFORMS", GL_MAX_TESS_CONTROL_IMAGE_UNIFORMS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_EVALUATION_IMAGE_UNIFORMS", GL_MAX_TESS_EVALUATION_IMAGE_UNIFORMS);
	CREATE_CONSTANT_ACCESSOR("MAX_GEOMETRY_IMAGE_UNIFORMS", GL_MAX_GEOMETRY_IMAGE_UNIFORMS);
	CREATE_CONSTANT_ACCESSOR("MAX_FRAGMENT_IMAGE_UNIFORMS", GL_MAX_FRAGMENT_IMAGE_UNIFORMS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_IMAGE_UNIFORMS", GL_MAX_COMBINED_IMAGE_UNIFORMS);
	CREATE_CONSTANT_ACCESSOR("ALL_BARRIER_BITS", GL_ALL_BARRIER_BITS);

	tpl->Set(String::NewFromUtf8(isolate, "bindImageTexture"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("bindImageTexture requires 7 arguments");
			return;
		}

		GLuint unit = args[0]->Uint32Value();
		GLuint texture = args[1]->Uint32Value();
		GLint level = args[2]->Int32Value();
		GLboolean layered = GLboolean(args[3]->Uint32Value());
		GLint layer = args[4]->Int32Value();
		GLenum access = args[5]->Uint32Value();
		GLenum format = args[6]->Uint32Value();

		glBindImageTexture(unit, texture, level, layered, layer, access, format);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "memoryBarrier"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("memoryBarrier requires 1 arguments");
			return;
		}

		GLbitfield barriers = args[0]->Uint32Value();

		glMemoryBarrier(barriers);
	}));



	// empty / skipped / ignored: GL_ARB_shader_image_size
	// empty / skipped / ignored: GL_ARB_shader_objects
	// empty / skipped / ignored: GL_ARB_shader_precision
	// empty / skipped / ignored: GL_ARB_shader_stencil_export
	/* ------------------------------ GL_ARB_shader_storage_buffer_object ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("SHADER_STORAGE_BARRIER_BIT", GL_SHADER_STORAGE_BARRIER_BIT);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_SHADER_OUTPUT_RESOURCES", GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES);
	CREATE_CONSTANT_ACCESSOR("SHADER_STORAGE_BUFFER", GL_SHADER_STORAGE_BUFFER);
	CREATE_CONSTANT_ACCESSOR("SHADER_STORAGE_BUFFER_BINDING", GL_SHADER_STORAGE_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("SHADER_STORAGE_BUFFER_START", GL_SHADER_STORAGE_BUFFER_START);
	CREATE_CONSTANT_ACCESSOR("SHADER_STORAGE_BUFFER_SIZE", GL_SHADER_STORAGE_BUFFER_SIZE);
	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_SHADER_STORAGE_BLOCKS", GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS);
	CREATE_CONSTANT_ACCESSOR("MAX_GEOMETRY_SHADER_STORAGE_BLOCKS", GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS", GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS", GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS);
	CREATE_CONSTANT_ACCESSOR("MAX_FRAGMENT_SHADER_STORAGE_BLOCKS", GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMPUTE_SHADER_STORAGE_BLOCKS", GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_SHADER_STORAGE_BLOCKS", GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS);
	CREATE_CONSTANT_ACCESSOR("MAX_SHADER_STORAGE_BUFFER_BINDINGS", GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS);
	CREATE_CONSTANT_ACCESSOR("MAX_SHADER_STORAGE_BLOCK_SIZE", GL_MAX_SHADER_STORAGE_BLOCK_SIZE);
	CREATE_CONSTANT_ACCESSOR("SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT", GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT);

	tpl->Set(String::NewFromUtf8(isolate, "shaderStorageBlockBinding"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("shaderStorageBlockBinding requires 3 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLuint storageBlockIndex = args[1]->Uint32Value();
		GLuint storageBlockBinding = args[2]->Uint32Value();

		glShaderStorageBlockBinding(program, storageBlockIndex, storageBlockBinding);
	}));



	/* ------------------------------ GL_ARB_shader_subroutine ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("ACTIVE_SUBROUTINES", GL_ACTIVE_SUBROUTINES);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_SUBROUTINE_UNIFORMS", GL_ACTIVE_SUBROUTINE_UNIFORMS);
	CREATE_CONSTANT_ACCESSOR("MAX_SUBROUTINES", GL_MAX_SUBROUTINES);
	CREATE_CONSTANT_ACCESSOR("MAX_SUBROUTINE_UNIFORM_LOCATIONS", GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS", GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_SUBROUTINE_MAX_LENGTH", GL_ACTIVE_SUBROUTINE_MAX_LENGTH);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_SUBROUTINE_UNIFORM_MAX_LENGTH", GL_ACTIVE_SUBROUTINE_UNIFORM_MAX_LENGTH);
	CREATE_CONSTANT_ACCESSOR("NUM_COMPATIBLE_SUBROUTINES", GL_NUM_COMPATIBLE_SUBROUTINES);
	CREATE_CONSTANT_ACCESSOR("COMPATIBLE_SUBROUTINES", GL_COMPATIBLE_SUBROUTINES);

	tpl->Set(String::NewFromUtf8(isolate, "getActiveSubroutineName"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("getActiveSubroutineName requires 6 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLenum shadertype = args[1]->Uint32Value();
		GLuint index = args[2]->Uint32Value();
		GLsizei bufsize = args[3]->Int32Value();

		GLsizei* length = nullptr;
		if (args[4]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[4]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetActiveSubroutineName): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLchar* name = nullptr;
		if (args[5]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[5]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			name = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetActiveSubroutineName): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetActiveSubroutineName(program, shadertype, index, bufsize, length, name);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getActiveSubroutineUniformName"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("getActiveSubroutineUniformName requires 6 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLenum shadertype = args[1]->Uint32Value();
		GLuint index = args[2]->Uint32Value();
		GLsizei bufsize = args[3]->Int32Value();

		GLsizei* length = nullptr;
		if (args[4]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[4]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetActiveSubroutineUniformName): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLchar* name = nullptr;
		if (args[5]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[5]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			name = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetActiveSubroutineUniformName): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetActiveSubroutineUniformName(program, shadertype, index, bufsize, length, name);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getActiveSubroutineUniformiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("getActiveSubroutineUniformiv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLenum shadertype = args[1]->Uint32Value();
		GLuint index = args[2]->Uint32Value();
		GLenum pname = args[3]->Uint32Value();

		GLint* values = nullptr;
		if (args[4]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[4]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			values = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetActiveSubroutineUniformiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetActiveSubroutineUniformiv(program, shadertype, index, pname, values);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getProgramStageiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getProgramStageiv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLenum shadertype = args[1]->Uint32Value();
		GLenum pname = args[2]->Uint32Value();

		GLint* values = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			values = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetProgramStageiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetProgramStageiv(program, shadertype, pname, values);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getUniformSubroutineuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getUniformSubroutineuiv requires 3 arguments");
			return;
		}

		GLenum shadertype = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();

		GLuint* params = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGetUniformSubroutineuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGetUniformSubroutineuiv(shadertype, location, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "uniformSubroutinesuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniformSubroutinesuiv requires 3 arguments");
			return;
		}

		GLenum shadertype = args[0]->Uint32Value();
		GLsizei count = args[1]->Int32Value();

		GLuint* indices = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			indices = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glUniformSubroutinesuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glUniformSubroutinesuiv(shadertype, count, indices);
	}));



	// empty / skipped / ignored: GL_ARB_shader_texture_image_samples
	// empty / skipped / ignored: GL_ARB_shader_texture_lod
	// empty / skipped / ignored: GL_ARB_shader_viewport_layer_array
	// empty / skipped / ignored: GL_ARB_shading_language_100
	// empty / skipped / ignored: GL_ARB_shading_language_420pack
	// empty / skipped / ignored: GL_ARB_shading_language_include
	// empty / skipped / ignored: GL_ARB_shading_language_packing
	// empty / skipped / ignored: GL_ARB_shadow
	// empty / skipped / ignored: GL_ARB_shadow_ambient
	// empty / skipped / ignored: GL_ARB_sparse_buffer
	// empty / skipped / ignored: GL_ARB_sparse_texture
	// empty / skipped / ignored: GL_ARB_sparse_texture2
	// empty / skipped / ignored: GL_ARB_sparse_texture_clamp
	/* ------------------------------ GL_ARB_spirv_extensions ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("SPIR_V_EXTENSIONS", GL_SPIR_V_EXTENSIONS);
	CREATE_CONSTANT_ACCESSOR("NUM_SPIR_V_EXTENSIONS", GL_NUM_SPIR_V_EXTENSIONS);



	/* ------------------------------ GL_ARB_stencil_texturing ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("DEPTH_STENCIL_TEXTURE_MODE", GL_DEPTH_STENCIL_TEXTURE_MODE);



	/* ------------------------------ GL_ARB_sync ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("SYNC_FLUSH_COMMANDS_BIT", GL_SYNC_FLUSH_COMMANDS_BIT);
	CREATE_CONSTANT_ACCESSOR("MAX_SERVER_WAIT_TIMEOUT", GL_MAX_SERVER_WAIT_TIMEOUT);
	CREATE_CONSTANT_ACCESSOR("OBJECT_TYPE", GL_OBJECT_TYPE);
	CREATE_CONSTANT_ACCESSOR("SYNC_CONDITION", GL_SYNC_CONDITION);
	CREATE_CONSTANT_ACCESSOR("SYNC_STATUS", GL_SYNC_STATUS);
	CREATE_CONSTANT_ACCESSOR("SYNC_FLAGS", GL_SYNC_FLAGS);
	CREATE_CONSTANT_ACCESSOR("SYNC_FENCE", GL_SYNC_FENCE);
	CREATE_CONSTANT_ACCESSOR("SYNC_GPU_COMMANDS_COMPLETE", GL_SYNC_GPU_COMMANDS_COMPLETE);
	CREATE_CONSTANT_ACCESSOR("UNSIGNALED", GL_UNSIGNALED);
	CREATE_CONSTANT_ACCESSOR("SIGNALED", GL_SIGNALED);
	CREATE_CONSTANT_ACCESSOR("ALREADY_SIGNALED", GL_ALREADY_SIGNALED);
	CREATE_CONSTANT_ACCESSOR("TIMEOUT_EXPIRED", GL_TIMEOUT_EXPIRED);
	CREATE_CONSTANT_ACCESSOR("CONDITION_SATISFIED", GL_CONDITION_SATISFIED);
	CREATE_CONSTANT_ACCESSOR("WAIT_FAILED", GL_WAIT_FAILED);







	/* ------------------------------ GL_ARB_tessellation_shader ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("PATCHES", GL_PATCHES);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BLOCK_REFERENCED_BY_TESS_CONTROL_SHADER", GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_CONTROL_SHADER);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BLOCK_REFERENCED_BY_TESS_EVALUATION_SHADER", GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_EVALUATION_SHADER);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_CONTROL_INPUT_COMPONENTS", GL_MAX_TESS_CONTROL_INPUT_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_EVALUATION_INPUT_COMPONENTS", GL_MAX_TESS_EVALUATION_INPUT_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_TESS_CONTROL_UNIFORM_COMPONENTS", GL_MAX_COMBINED_TESS_CONTROL_UNIFORM_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_TESS_EVALUATION_UNIFORM_COMPONENTS", GL_MAX_COMBINED_TESS_EVALUATION_UNIFORM_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("PATCH_VERTICES", GL_PATCH_VERTICES);
	CREATE_CONSTANT_ACCESSOR("PATCH_DEFAULT_INNER_LEVEL", GL_PATCH_DEFAULT_INNER_LEVEL);
	CREATE_CONSTANT_ACCESSOR("PATCH_DEFAULT_OUTER_LEVEL", GL_PATCH_DEFAULT_OUTER_LEVEL);
	CREATE_CONSTANT_ACCESSOR("TESS_CONTROL_OUTPUT_VERTICES", GL_TESS_CONTROL_OUTPUT_VERTICES);
	CREATE_CONSTANT_ACCESSOR("TESS_GEN_MODE", GL_TESS_GEN_MODE);
	CREATE_CONSTANT_ACCESSOR("TESS_GEN_SPACING", GL_TESS_GEN_SPACING);
	CREATE_CONSTANT_ACCESSOR("TESS_GEN_VERTEX_ORDER", GL_TESS_GEN_VERTEX_ORDER);
	CREATE_CONSTANT_ACCESSOR("TESS_GEN_POINT_MODE", GL_TESS_GEN_POINT_MODE);
	CREATE_CONSTANT_ACCESSOR("ISOLINES", GL_ISOLINES);
	CREATE_CONSTANT_ACCESSOR("FRACTIONAL_ODD", GL_FRACTIONAL_ODD);
	CREATE_CONSTANT_ACCESSOR("FRACTIONAL_EVEN", GL_FRACTIONAL_EVEN);
	CREATE_CONSTANT_ACCESSOR("MAX_PATCH_VERTICES", GL_MAX_PATCH_VERTICES);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_GEN_LEVEL", GL_MAX_TESS_GEN_LEVEL);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_CONTROL_UNIFORM_COMPONENTS", GL_MAX_TESS_CONTROL_UNIFORM_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_EVALUATION_UNIFORM_COMPONENTS", GL_MAX_TESS_EVALUATION_UNIFORM_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_CONTROL_TEXTURE_IMAGE_UNITS", GL_MAX_TESS_CONTROL_TEXTURE_IMAGE_UNITS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_EVALUATION_TEXTURE_IMAGE_UNITS", GL_MAX_TESS_EVALUATION_TEXTURE_IMAGE_UNITS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_CONTROL_OUTPUT_COMPONENTS", GL_MAX_TESS_CONTROL_OUTPUT_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_PATCH_COMPONENTS", GL_MAX_TESS_PATCH_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS", GL_MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_EVALUATION_OUTPUT_COMPONENTS", GL_MAX_TESS_EVALUATION_OUTPUT_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("TESS_EVALUATION_SHADER", GL_TESS_EVALUATION_SHADER);
	CREATE_CONSTANT_ACCESSOR("TESS_CONTROL_SHADER", GL_TESS_CONTROL_SHADER);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_CONTROL_UNIFORM_BLOCKS", GL_MAX_TESS_CONTROL_UNIFORM_BLOCKS);
	CREATE_CONSTANT_ACCESSOR("MAX_TESS_EVALUATION_UNIFORM_BLOCKS", GL_MAX_TESS_EVALUATION_UNIFORM_BLOCKS);

	tpl->Set(String::NewFromUtf8(isolate, "patchParameterfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("patchParameterfv requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();

		GLfloat* values = nullptr;
		if (args[1]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[1]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			values = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glPatchParameterfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glPatchParameterfv(pname, values);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "patchParameteri"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("patchParameteri requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();
		GLint value = args[1]->Int32Value();

		glPatchParameteri(pname, value);
	}));



	/* ------------------------------ GL_ARB_texture_barrier ------------------------------ */





	// empty / skipped / ignored: GL_ARB_texture_border_clamp
	// empty / skipped / ignored: GL_ARB_texture_buffer_object
	// empty / skipped / ignored: GL_ARB_texture_buffer_object_rgb32
	/* ------------------------------ GL_ARB_texture_buffer_range ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_BUFFER_OFFSET", GL_TEXTURE_BUFFER_OFFSET);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BUFFER_SIZE", GL_TEXTURE_BUFFER_SIZE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BUFFER_OFFSET_ALIGNMENT", GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT);

	tpl->Set(String::NewFromUtf8(isolate, "texBufferRange"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("texBufferRange requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum internalformat = args[1]->Uint32Value();
		GLuint buffer = args[2]->Uint32Value();
		GLintptr offset = GLintptr(args[3]->Int32Value());
		GLsizeiptr size = GLsizeiptr(args[4]->Int32Value());

		glTexBufferRange(target, internalformat, buffer, offset, size);
	}));



	// empty / skipped / ignored: GL_ARB_texture_compression
	// empty / skipped / ignored: GL_ARB_texture_compression_bptc
	/* ------------------------------ GL_ARB_texture_compression_rgtc ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RED_RGTC1", GL_COMPRESSED_RED_RGTC1);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_SIGNED_RED_RGTC1", GL_COMPRESSED_SIGNED_RED_RGTC1);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RG_RGTC2", GL_COMPRESSED_RG_RGTC2);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_SIGNED_RG_RGTC2", GL_COMPRESSED_SIGNED_RG_RGTC2);



	// empty / skipped / ignored: GL_ARB_texture_cube_map
	// empty / skipped / ignored: GL_ARB_texture_cube_map_array
	// empty / skipped / ignored: GL_ARB_texture_env_add
	// empty / skipped / ignored: GL_ARB_texture_env_combine
	// empty / skipped / ignored: GL_ARB_texture_env_crossbar
	// empty / skipped / ignored: GL_ARB_texture_env_dot3
	/* ------------------------------ GL_ARB_texture_filter_anisotropic ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_MAX_ANISOTROPY", GL_TEXTURE_MAX_ANISOTROPY);
	CREATE_CONSTANT_ACCESSOR("MAX_TEXTURE_MAX_ANISOTROPY", GL_MAX_TEXTURE_MAX_ANISOTROPY);



	// empty / skipped / ignored: GL_ARB_texture_filter_minmax
	// empty / skipped / ignored: GL_ARB_texture_float
	// empty / skipped / ignored: GL_ARB_texture_gather
	// empty / skipped / ignored: GL_ARB_texture_mirrored_repeat
	/* ------------------------------ GL_ARB_texture_mirror_clamp_to_edge ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("MIRROR_CLAMP_TO_EDGE", GL_MIRROR_CLAMP_TO_EDGE);



	/* ------------------------------ GL_ARB_texture_multisample ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("SAMPLE_POSITION", GL_SAMPLE_POSITION);
	CREATE_CONSTANT_ACCESSOR("SAMPLE_MASK", GL_SAMPLE_MASK);
	CREATE_CONSTANT_ACCESSOR("SAMPLE_MASK_VALUE", GL_SAMPLE_MASK_VALUE);
	CREATE_CONSTANT_ACCESSOR("MAX_SAMPLE_MASK_WORDS", GL_MAX_SAMPLE_MASK_WORDS);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_2D_MULTISAMPLE", GL_TEXTURE_2D_MULTISAMPLE);
	CREATE_CONSTANT_ACCESSOR("PROXY_TEXTURE_2D_MULTISAMPLE", GL_PROXY_TEXTURE_2D_MULTISAMPLE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_2D_MULTISAMPLE_ARRAY", GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
	CREATE_CONSTANT_ACCESSOR("PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY", GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BINDING_2D_MULTISAMPLE", GL_TEXTURE_BINDING_2D_MULTISAMPLE);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY", GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_SAMPLES", GL_TEXTURE_SAMPLES);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_FIXED_SAMPLE_LOCATIONS", GL_TEXTURE_FIXED_SAMPLE_LOCATIONS);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_2D_MULTISAMPLE", GL_SAMPLER_2D_MULTISAMPLE);
	CREATE_CONSTANT_ACCESSOR("INT_SAMPLER_2D_MULTISAMPLE", GL_INT_SAMPLER_2D_MULTISAMPLE);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE", GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE);
	CREATE_CONSTANT_ACCESSOR("SAMPLER_2D_MULTISAMPLE_ARRAY", GL_SAMPLER_2D_MULTISAMPLE_ARRAY);
	CREATE_CONSTANT_ACCESSOR("INT_SAMPLER_2D_MULTISAMPLE_ARRAY", GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY);
	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY", GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY);
	CREATE_CONSTANT_ACCESSOR("MAX_COLOR_TEXTURE_SAMPLES", GL_MAX_COLOR_TEXTURE_SAMPLES);
	CREATE_CONSTANT_ACCESSOR("MAX_DEPTH_TEXTURE_SAMPLES", GL_MAX_DEPTH_TEXTURE_SAMPLES);
	CREATE_CONSTANT_ACCESSOR("MAX_INTEGER_SAMPLES", GL_MAX_INTEGER_SAMPLES);

	tpl->Set(String::NewFromUtf8(isolate, "getMultisamplefv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getMultisamplefv requires 3 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();

		GLfloat* val = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			val = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glGetMultisamplefv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glGetMultisamplefv(pname, index, val);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "sampleMaski"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("sampleMaski requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLbitfield mask = args[1]->Uint32Value();

		glSampleMaski(index, mask);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texImage2DMultisample"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("texImage2DMultisample requires 6 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei samples = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();
		GLboolean fixedsamplelocations = GLboolean(args[5]->Uint32Value());

		glTexImage2DMultisample(target, samples, internalformat, width, height, fixedsamplelocations);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texImage3DMultisample"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("texImage3DMultisample requires 7 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei samples = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();
		GLsizei depth = args[5]->Int32Value();
		GLboolean fixedsamplelocations = GLboolean(args[6]->Uint32Value());

		glTexImage3DMultisample(target, samples, internalformat, width, height, depth, fixedsamplelocations);
	}));



	// empty / skipped / ignored: GL_ARB_texture_non_power_of_two
	// empty / skipped / ignored: GL_ARB_texture_query_levels
	// empty / skipped / ignored: GL_ARB_texture_query_lod
	// empty / skipped / ignored: GL_ARB_texture_rectangle
	/* ------------------------------ GL_ARB_texture_rg ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RED", GL_COMPRESSED_RED);
	CREATE_CONSTANT_ACCESSOR("COMPRESSED_RG", GL_COMPRESSED_RG);
	CREATE_CONSTANT_ACCESSOR("RG", GL_RG);
	CREATE_CONSTANT_ACCESSOR("RG_INTEGER", GL_RG_INTEGER);
	CREATE_CONSTANT_ACCESSOR("R8", GL_R8);
	CREATE_CONSTANT_ACCESSOR("R16", GL_R16);
	CREATE_CONSTANT_ACCESSOR("RG8", GL_RG8);
	CREATE_CONSTANT_ACCESSOR("RG16", GL_RG16);
	CREATE_CONSTANT_ACCESSOR("R16F", GL_R16F);
	CREATE_CONSTANT_ACCESSOR("R32F", GL_R32F);
	CREATE_CONSTANT_ACCESSOR("RG16F", GL_RG16F);
	CREATE_CONSTANT_ACCESSOR("RG32F", GL_RG32F);
	CREATE_CONSTANT_ACCESSOR("R8I", GL_R8I);
	CREATE_CONSTANT_ACCESSOR("R8UI", GL_R8UI);
	CREATE_CONSTANT_ACCESSOR("R16I", GL_R16I);
	CREATE_CONSTANT_ACCESSOR("R16UI", GL_R16UI);
	CREATE_CONSTANT_ACCESSOR("R32I", GL_R32I);
	CREATE_CONSTANT_ACCESSOR("R32UI", GL_R32UI);
	CREATE_CONSTANT_ACCESSOR("RG8I", GL_RG8I);
	CREATE_CONSTANT_ACCESSOR("RG8UI", GL_RG8UI);
	CREATE_CONSTANT_ACCESSOR("RG16I", GL_RG16I);
	CREATE_CONSTANT_ACCESSOR("RG16UI", GL_RG16UI);
	CREATE_CONSTANT_ACCESSOR("RG32I", GL_RG32I);
	CREATE_CONSTANT_ACCESSOR("RG32UI", GL_RG32UI);



	/* ------------------------------ GL_ARB_texture_rgb10_a2ui ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("RGB10_A2UI", GL_RGB10_A2UI);



	/* ------------------------------ GL_ARB_texture_stencil8 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("STENCIL_INDEX", GL_STENCIL_INDEX);
	CREATE_CONSTANT_ACCESSOR("STENCIL_INDEX8", GL_STENCIL_INDEX8);



	/* ------------------------------ GL_ARB_texture_storage ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_IMMUTABLE_FORMAT", GL_TEXTURE_IMMUTABLE_FORMAT);

	tpl->Set(String::NewFromUtf8(isolate, "texStorage1D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("texStorage1D requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei levels = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();

		glTexStorage1D(target, levels, internalformat, width);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texStorage2D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("texStorage2D requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei levels = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();

		glTexStorage2D(target, levels, internalformat, width, height);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texStorage3D"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("texStorage3D requires 6 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei levels = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();
		GLsizei depth = args[5]->Int32Value();

		glTexStorage3D(target, levels, internalformat, width, height, depth);
	}));



	/* ------------------------------ GL_ARB_texture_storage_multisample ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "texStorage2DMultisample"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("texStorage2DMultisample requires 6 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei samples = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();
		GLboolean fixedsamplelocations = GLboolean(args[5]->Uint32Value());

		glTexStorage2DMultisample(target, samples, internalformat, width, height, fixedsamplelocations);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texStorage3DMultisample"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("texStorage3DMultisample requires 7 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLsizei samples = args[1]->Int32Value();
		GLenum internalformat = args[2]->Uint32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();
		GLsizei depth = args[5]->Int32Value();
		GLboolean fixedsamplelocations = GLboolean(args[6]->Uint32Value());

		glTexStorage3DMultisample(target, samples, internalformat, width, height, depth, fixedsamplelocations);
	}));



	/* ------------------------------ GL_ARB_texture_swizzle ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_SWIZZLE_R", GL_TEXTURE_SWIZZLE_R);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_SWIZZLE_G", GL_TEXTURE_SWIZZLE_G);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_SWIZZLE_B", GL_TEXTURE_SWIZZLE_B);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_SWIZZLE_A", GL_TEXTURE_SWIZZLE_A);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_SWIZZLE_RGBA", GL_TEXTURE_SWIZZLE_RGBA);



	/* ------------------------------ GL_ARB_texture_view ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_VIEW_MIN_LEVEL", GL_TEXTURE_VIEW_MIN_LEVEL);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_VIEW_NUM_LEVELS", GL_TEXTURE_VIEW_NUM_LEVELS);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_VIEW_MIN_LAYER", GL_TEXTURE_VIEW_MIN_LAYER);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_VIEW_NUM_LAYERS", GL_TEXTURE_VIEW_NUM_LAYERS);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_IMMUTABLE_LEVELS", GL_TEXTURE_IMMUTABLE_LEVELS);

	tpl->Set(String::NewFromUtf8(isolate, "textureView"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 8) {
			V8Helper::_instance->throwException("textureView requires 8 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLenum target = args[1]->Uint32Value();
		GLuint origtexture = args[2]->Uint32Value();
		GLenum internalformat = args[3]->Uint32Value();
		GLuint minlevel = args[4]->Uint32Value();
		GLuint numlevels = args[5]->Uint32Value();
		GLuint minlayer = args[6]->Uint32Value();
		GLuint numlayers = args[7]->Uint32Value();

		glTextureView(texture, target, origtexture, internalformat, minlevel, numlevels, minlayer, numlayers);
	}));



	/* ------------------------------ GL_ARB_timer_query ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TIME_ELAPSED", GL_TIME_ELAPSED);
	CREATE_CONSTANT_ACCESSOR("TIMESTAMP", GL_TIMESTAMP);



	tpl->Set(String::NewFromUtf8(isolate, "queryCounter"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("queryCounter requires 2 arguments");
			return;
		}

		GLuint id = args[0]->Uint32Value();
		GLenum target = args[1]->Uint32Value();

		glQueryCounter(id, target);
	}));



	/* ------------------------------ GL_ARB_transform_feedback2 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK", GL_TRANSFORM_FEEDBACK);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_BUFFER_PAUSED", GL_TRANSFORM_FEEDBACK_BUFFER_PAUSED);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_BUFFER_ACTIVE", GL_TRANSFORM_FEEDBACK_BUFFER_ACTIVE);
	CREATE_CONSTANT_ACCESSOR("TRANSFORM_FEEDBACK_BINDING", GL_TRANSFORM_FEEDBACK_BINDING);

	tpl->Set(String::NewFromUtf8(isolate, "bindTransformFeedback"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("bindTransformFeedback requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint id = args[1]->Uint32Value();

		glBindTransformFeedback(target, id);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "deleteTransformFeedbacks"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("deleteTransformFeedbacks requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* ids = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			ids = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glDeleteTransformFeedbacks): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glDeleteTransformFeedbacks(n, ids);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "drawTransformFeedback"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("drawTransformFeedback requires 2 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLuint id = args[1]->Uint32Value();

		glDrawTransformFeedback(mode, id);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "genTransformFeedbacks"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("genTransformFeedbacks requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* ids = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			ids = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGenTransformFeedbacks): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGenTransformFeedbacks(n, ids);
	}));





	/* ------------------------------ GL_ARB_transform_feedback3 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("MAX_TRANSFORM_FEEDBACK_BUFFERS", GL_MAX_TRANSFORM_FEEDBACK_BUFFERS);
	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_STREAMS", GL_MAX_VERTEX_STREAMS);

	tpl->Set(String::NewFromUtf8(isolate, "beginQueryIndexed"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("beginQueryIndexed requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();
		GLuint id = args[2]->Uint32Value();

		glBeginQueryIndexed(target, index, id);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "drawTransformFeedbackStream"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("drawTransformFeedbackStream requires 3 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLuint id = args[1]->Uint32Value();
		GLuint stream = args[2]->Uint32Value();

		glDrawTransformFeedbackStream(mode, id, stream);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "endQueryIndexed"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("endQueryIndexed requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();

		glEndQueryIndexed(target, index);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getQueryIndexediv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getQueryIndexediv requires 4 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();
		GLenum pname = args[2]->Uint32Value();

		GLint* params = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetQueryIndexediv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetQueryIndexediv(target, index, pname, params);
	}));



	/* ------------------------------ GL_ARB_transform_feedback_instanced ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "drawTransformFeedbackInstanced"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("drawTransformFeedbackInstanced requires 3 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLuint id = args[1]->Uint32Value();
		GLsizei primcount = args[2]->Int32Value();

		glDrawTransformFeedbackInstanced(mode, id, primcount);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "drawTransformFeedbackStreamInstanced"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("drawTransformFeedbackStreamInstanced requires 4 arguments");
			return;
		}

		GLenum mode = args[0]->Uint32Value();
		GLuint id = args[1]->Uint32Value();
		GLuint stream = args[2]->Uint32Value();
		GLsizei primcount = args[3]->Int32Value();

		glDrawTransformFeedbackStreamInstanced(mode, id, stream, primcount);
	}));



	// empty / skipped / ignored: GL_ARB_transform_feedback_overflow_query
	// empty / skipped / ignored: GL_ARB_transpose_matrix
	/* ------------------------------ GL_ARB_uniform_buffer_object ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("INVALID_INDEX", GL_INVALID_INDEX);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BUFFER", GL_UNIFORM_BUFFER);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BUFFER_BINDING", GL_UNIFORM_BUFFER_BINDING);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BUFFER_START", GL_UNIFORM_BUFFER_START);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BUFFER_SIZE", GL_UNIFORM_BUFFER_SIZE);
	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_UNIFORM_BLOCKS", GL_MAX_VERTEX_UNIFORM_BLOCKS);
	CREATE_CONSTANT_ACCESSOR("MAX_GEOMETRY_UNIFORM_BLOCKS", GL_MAX_GEOMETRY_UNIFORM_BLOCKS);
	CREATE_CONSTANT_ACCESSOR("MAX_FRAGMENT_UNIFORM_BLOCKS", GL_MAX_FRAGMENT_UNIFORM_BLOCKS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_UNIFORM_BLOCKS", GL_MAX_COMBINED_UNIFORM_BLOCKS);
	CREATE_CONSTANT_ACCESSOR("MAX_UNIFORM_BUFFER_BINDINGS", GL_MAX_UNIFORM_BUFFER_BINDINGS);
	CREATE_CONSTANT_ACCESSOR("MAX_UNIFORM_BLOCK_SIZE", GL_MAX_UNIFORM_BLOCK_SIZE);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS", GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS", GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS", GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BUFFER_OFFSET_ALIGNMENT", GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH", GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH);
	CREATE_CONSTANT_ACCESSOR("ACTIVE_UNIFORM_BLOCKS", GL_ACTIVE_UNIFORM_BLOCKS);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_TYPE", GL_UNIFORM_TYPE);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_SIZE", GL_UNIFORM_SIZE);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_NAME_LENGTH", GL_UNIFORM_NAME_LENGTH);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BLOCK_INDEX", GL_UNIFORM_BLOCK_INDEX);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_OFFSET", GL_UNIFORM_OFFSET);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_ARRAY_STRIDE", GL_UNIFORM_ARRAY_STRIDE);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_MATRIX_STRIDE", GL_UNIFORM_MATRIX_STRIDE);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_IS_ROW_MAJOR", GL_UNIFORM_IS_ROW_MAJOR);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BLOCK_BINDING", GL_UNIFORM_BLOCK_BINDING);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BLOCK_DATA_SIZE", GL_UNIFORM_BLOCK_DATA_SIZE);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BLOCK_NAME_LENGTH", GL_UNIFORM_BLOCK_NAME_LENGTH);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BLOCK_ACTIVE_UNIFORMS", GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES", GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER", GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BLOCK_REFERENCED_BY_GEOMETRY_SHADER", GL_UNIFORM_BLOCK_REFERENCED_BY_GEOMETRY_SHADER);
	CREATE_CONSTANT_ACCESSOR("UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER", GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER);

	tpl->Set(String::NewFromUtf8(isolate, "bindBufferBase"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("bindBufferBase requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();
		GLuint buffer = args[2]->Uint32Value();

		glBindBufferBase(target, index, buffer);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "bindBufferRange"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("bindBufferRange requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();
		GLuint buffer = args[2]->Uint32Value();
		GLintptr offset = GLintptr(args[3]->Int32Value());
		GLsizeiptr size = GLsizeiptr(args[4]->Int32Value());

		glBindBufferRange(target, index, buffer, offset, size);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getActiveUniformBlockName"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("getActiveUniformBlockName requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLuint uniformBlockIndex = args[1]->Uint32Value();
		GLsizei bufSize = args[2]->Int32Value();

		GLsizei* length = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetActiveUniformBlockName): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLchar* uniformBlockName = nullptr;
		if (args[4]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[4]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			uniformBlockName = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetActiveUniformBlockName): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetActiveUniformBlockName(program, uniformBlockIndex, bufSize, length, uniformBlockName);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getActiveUniformBlockiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getActiveUniformBlockiv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLuint uniformBlockIndex = args[1]->Uint32Value();
		GLenum pname = args[2]->Uint32Value();

		GLint* params = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetActiveUniformBlockiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetActiveUniformBlockiv(program, uniformBlockIndex, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getActiveUniformName"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("getActiveUniformName requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLuint uniformIndex = args[1]->Uint32Value();
		GLsizei bufSize = args[2]->Int32Value();

		GLsizei* length = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetActiveUniformName): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLchar* uniformName = nullptr;
		if (args[4]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[4]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			uniformName = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetActiveUniformName): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetActiveUniformName(program, uniformIndex, bufSize, length, uniformName);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getActiveUniformsiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("getActiveUniformsiv requires 5 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLsizei uniformCount = args[1]->Int32Value();

		GLuint* uniformIndices = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			uniformIndices = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGetActiveUniformsiv): array must be of type Uint32Array" << endl;
			exit(1);
		}

		GLenum pname = args[3]->Uint32Value();

		GLint* params = nullptr;
		if (args[4]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[4]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetActiveUniformsiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetActiveUniformsiv(program, uniformCount, uniformIndices, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getIntegeri_v"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getIntegeri_v requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();

		GLint* data = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetIntegeri_v): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetIntegeri_v(target, index, data);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "uniformBlockBinding"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("uniformBlockBinding requires 3 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLuint uniformBlockIndex = args[1]->Uint32Value();
		GLuint uniformBlockBinding = args[2]->Uint32Value();

		glUniformBlockBinding(program, uniformBlockIndex, uniformBlockBinding);
	}));



	/* ------------------------------ GL_ARB_vertex_array_bgra ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("BGRA", GL_BGRA);



	/* ------------------------------ GL_ARB_vertex_array_object ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("VERTEX_ARRAY_BINDING", GL_VERTEX_ARRAY_BINDING);

	tpl->Set(String::NewFromUtf8(isolate, "bindVertexArray"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("bindVertexArray requires 1 arguments");
			return;
		}

		GLuint array = args[0]->Uint32Value();

		glBindVertexArray(array);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "deleteVertexArrays"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("deleteVertexArrays requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* arrays = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			arrays = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glDeleteVertexArrays): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glDeleteVertexArrays(n, arrays);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "genVertexArrays"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("genVertexArrays requires 2 arguments");
			return;
		}

		GLsizei n = args[0]->Int32Value();

		GLuint* arrays = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			arrays = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGenVertexArrays): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGenVertexArrays(n, arrays);
	}));



	/* ------------------------------ GL_ARB_vertex_attrib_64bit ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "getVertexAttribLdv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getVertexAttribLdv requires 3 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLdouble* params = nullptr;
		if (args[2]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[2]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glGetVertexAttribLdv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glGetVertexAttribLdv(index, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribL1d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribL1d requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLdouble x = args[1]->NumberValue();

		glVertexAttribL1d(index, x);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribL1dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribL1dv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLdouble* v = nullptr;
		if (args[1]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[1]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribL1dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glVertexAttribL1dv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribL2d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("vertexAttribL2d requires 3 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLdouble x = args[1]->NumberValue();
		GLdouble y = args[2]->NumberValue();

		glVertexAttribL2d(index, x, y);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribL2dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribL2dv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLdouble* v = nullptr;
		if (args[1]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[1]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribL2dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glVertexAttribL2dv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribL3d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttribL3d requires 4 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLdouble x = args[1]->NumberValue();
		GLdouble y = args[2]->NumberValue();
		GLdouble z = args[3]->NumberValue();

		glVertexAttribL3d(index, x, y, z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribL3dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribL3dv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLdouble* v = nullptr;
		if (args[1]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[1]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribL3dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glVertexAttribL3dv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribL4d"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("vertexAttribL4d requires 5 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLdouble x = args[1]->NumberValue();
		GLdouble y = args[2]->NumberValue();
		GLdouble z = args[3]->NumberValue();
		GLdouble w = args[4]->NumberValue();

		glVertexAttribL4d(index, x, y, z, w);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribL4dv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribL4dv requires 2 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();

		GLdouble* v = nullptr;
		if (args[1]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[1]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			v = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribL4dv): array must be of type Float64Array" << endl;
			exit(1);
		}


		glVertexAttribL4dv(index, v);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribLPointer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("vertexAttribLPointer requires 5 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLint size = args[1]->Int32Value();
		GLenum type = args[2]->Uint32Value();
		GLsizei stride = args[3]->Int32Value();

		void* pointer = nullptr;
		if (args[4]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[4]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			pointer = reinterpret_cast<void*>(bdata);
		} else if (args[4]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[4]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			pointer = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribLPointer): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glVertexAttribLPointer(index, size, type, stride, pointer);
	}));



	/* ------------------------------ GL_ARB_vertex_attrib_binding ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("VERTEX_ATTRIB_BINDING", GL_VERTEX_ATTRIB_BINDING);
	CREATE_CONSTANT_ACCESSOR("VERTEX_ATTRIB_RELATIVE_OFFSET", GL_VERTEX_ATTRIB_RELATIVE_OFFSET);
	CREATE_CONSTANT_ACCESSOR("VERTEX_BINDING_DIVISOR", GL_VERTEX_BINDING_DIVISOR);
	CREATE_CONSTANT_ACCESSOR("VERTEX_BINDING_OFFSET", GL_VERTEX_BINDING_OFFSET);
	CREATE_CONSTANT_ACCESSOR("VERTEX_BINDING_STRIDE", GL_VERTEX_BINDING_STRIDE);
	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_ATTRIB_RELATIVE_OFFSET", GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET);
	CREATE_CONSTANT_ACCESSOR("MAX_VERTEX_ATTRIB_BINDINGS", GL_MAX_VERTEX_ATTRIB_BINDINGS);
	CREATE_CONSTANT_ACCESSOR("VERTEX_BINDING_BUFFER", GL_VERTEX_BINDING_BUFFER);

	tpl->Set(String::NewFromUtf8(isolate, "bindVertexBuffer"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("bindVertexBuffer requires 4 arguments");
			return;
		}

		GLuint bindingindex = args[0]->Uint32Value();
		GLuint buffer = args[1]->Uint32Value();
		GLintptr offset = GLintptr(args[2]->Int32Value());
		GLsizei stride = args[3]->Int32Value();

		glBindVertexBuffer(bindingindex, buffer, offset, stride);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribBinding"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexAttribBinding requires 2 arguments");
			return;
		}

		GLuint attribindex = args[0]->Uint32Value();
		GLuint bindingindex = args[1]->Uint32Value();

		glVertexAttribBinding(attribindex, bindingindex);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribFormat"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("vertexAttribFormat requires 5 arguments");
			return;
		}

		GLuint attribindex = args[0]->Uint32Value();
		GLint size = args[1]->Int32Value();
		GLenum type = args[2]->Uint32Value();
		GLboolean normalized = GLboolean(args[3]->Uint32Value());
		GLuint relativeoffset = args[4]->Uint32Value();

		glVertexAttribFormat(attribindex, size, type, normalized, relativeoffset);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribIFormat"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttribIFormat requires 4 arguments");
			return;
		}

		GLuint attribindex = args[0]->Uint32Value();
		GLint size = args[1]->Int32Value();
		GLenum type = args[2]->Uint32Value();
		GLuint relativeoffset = args[3]->Uint32Value();

		glVertexAttribIFormat(attribindex, size, type, relativeoffset);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribLFormat"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttribLFormat requires 4 arguments");
			return;
		}

		GLuint attribindex = args[0]->Uint32Value();
		GLint size = args[1]->Int32Value();
		GLenum type = args[2]->Uint32Value();
		GLuint relativeoffset = args[3]->Uint32Value();

		glVertexAttribLFormat(attribindex, size, type, relativeoffset);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexBindingDivisor"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexBindingDivisor requires 2 arguments");
			return;
		}

		GLuint bindingindex = args[0]->Uint32Value();
		GLuint divisor = args[1]->Uint32Value();

		glVertexBindingDivisor(bindingindex, divisor);
	}));



	// empty / skipped / ignored: GL_ARB_vertex_blend
	// empty / skipped / ignored: GL_ARB_vertex_buffer_object
	// empty / skipped / ignored: GL_ARB_vertex_program
	// empty / skipped / ignored: GL_ARB_vertex_shader
	/* ------------------------------ GL_ARB_vertex_type_10f_11f_11f_rev ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_10F_11F_11F_REV", GL_UNSIGNED_INT_10F_11F_11F_REV);



	/* ------------------------------ GL_ARB_vertex_type_2_10_10_10_rev ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("UNSIGNED_INT_2_10_10_10_REV", GL_UNSIGNED_INT_2_10_10_10_REV);
	CREATE_CONSTANT_ACCESSOR("INT_2_10_10_10_REV", GL_INT_2_10_10_10_REV);

	tpl->Set(String::NewFromUtf8(isolate, "colorP3ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("colorP3ui requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();
		GLuint color = args[1]->Uint32Value();

		glColorP3ui(type, color);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "colorP3uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("colorP3uiv requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();

		GLuint* color = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			color = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glColorP3uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glColorP3uiv(type, color);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "colorP4ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("colorP4ui requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();
		GLuint color = args[1]->Uint32Value();

		glColorP4ui(type, color);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "colorP4uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("colorP4uiv requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();

		GLuint* color = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			color = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glColorP4uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glColorP4uiv(type, color);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoordP1ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("multiTexCoordP1ui requires 3 arguments");
			return;
		}

		GLenum texture = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();
		GLuint coords = args[2]->Uint32Value();

		glMultiTexCoordP1ui(texture, type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoordP1uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("multiTexCoordP1uiv requires 3 arguments");
			return;
		}

		GLenum texture = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();

		GLuint* coords = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			coords = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoordP1uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glMultiTexCoordP1uiv(texture, type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoordP2ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("multiTexCoordP2ui requires 3 arguments");
			return;
		}

		GLenum texture = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();
		GLuint coords = args[2]->Uint32Value();

		glMultiTexCoordP2ui(texture, type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoordP2uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("multiTexCoordP2uiv requires 3 arguments");
			return;
		}

		GLenum texture = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();

		GLuint* coords = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			coords = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoordP2uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glMultiTexCoordP2uiv(texture, type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoordP3ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("multiTexCoordP3ui requires 3 arguments");
			return;
		}

		GLenum texture = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();
		GLuint coords = args[2]->Uint32Value();

		glMultiTexCoordP3ui(texture, type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoordP3uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("multiTexCoordP3uiv requires 3 arguments");
			return;
		}

		GLenum texture = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();

		GLuint* coords = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			coords = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoordP3uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glMultiTexCoordP3uiv(texture, type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoordP4ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("multiTexCoordP4ui requires 3 arguments");
			return;
		}

		GLenum texture = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();
		GLuint coords = args[2]->Uint32Value();

		glMultiTexCoordP4ui(texture, type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoordP4uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("multiTexCoordP4uiv requires 3 arguments");
			return;
		}

		GLenum texture = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();

		GLuint* coords = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			coords = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glMultiTexCoordP4uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glMultiTexCoordP4uiv(texture, type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "normalP3ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("normalP3ui requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();
		GLuint coords = args[1]->Uint32Value();

		glNormalP3ui(type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "normalP3uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("normalP3uiv requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();

		GLuint* coords = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			coords = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glNormalP3uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glNormalP3uiv(type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColorP3ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("secondaryColorP3ui requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();
		GLuint color = args[1]->Uint32Value();

		glSecondaryColorP3ui(type, color);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "secondaryColorP3uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("secondaryColorP3uiv requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();

		GLuint* color = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			color = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glSecondaryColorP3uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glSecondaryColorP3uiv(type, color);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texCoordP1ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("texCoordP1ui requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();
		GLuint coords = args[1]->Uint32Value();

		glTexCoordP1ui(type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texCoordP1uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("texCoordP1uiv requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();

		GLuint* coords = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			coords = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glTexCoordP1uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glTexCoordP1uiv(type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texCoordP2ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("texCoordP2ui requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();
		GLuint coords = args[1]->Uint32Value();

		glTexCoordP2ui(type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texCoordP2uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("texCoordP2uiv requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();

		GLuint* coords = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			coords = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glTexCoordP2uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glTexCoordP2uiv(type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texCoordP3ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("texCoordP3ui requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();
		GLuint coords = args[1]->Uint32Value();

		glTexCoordP3ui(type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texCoordP3uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("texCoordP3uiv requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();

		GLuint* coords = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			coords = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glTexCoordP3uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glTexCoordP3uiv(type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texCoordP4ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("texCoordP4ui requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();
		GLuint coords = args[1]->Uint32Value();

		glTexCoordP4ui(type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texCoordP4uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("texCoordP4uiv requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();

		GLuint* coords = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			coords = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glTexCoordP4uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glTexCoordP4uiv(type, coords);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribP1ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttribP1ui requires 4 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();
		GLboolean normalized = GLboolean(args[2]->Uint32Value());
		GLuint value = args[3]->Uint32Value();

		glVertexAttribP1ui(index, type, normalized, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribP1uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttribP1uiv requires 4 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();
		GLboolean normalized = GLboolean(args[2]->Uint32Value());

		GLuint* value = nullptr;
		if (args[3]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[3]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribP1uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glVertexAttribP1uiv(index, type, normalized, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribP2ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttribP2ui requires 4 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();
		GLboolean normalized = GLboolean(args[2]->Uint32Value());
		GLuint value = args[3]->Uint32Value();

		glVertexAttribP2ui(index, type, normalized, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribP2uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttribP2uiv requires 4 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();
		GLboolean normalized = GLboolean(args[2]->Uint32Value());

		GLuint* value = nullptr;
		if (args[3]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[3]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribP2uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glVertexAttribP2uiv(index, type, normalized, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribP3ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttribP3ui requires 4 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();
		GLboolean normalized = GLboolean(args[2]->Uint32Value());
		GLuint value = args[3]->Uint32Value();

		glVertexAttribP3ui(index, type, normalized, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribP3uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttribP3uiv requires 4 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();
		GLboolean normalized = GLboolean(args[2]->Uint32Value());

		GLuint* value = nullptr;
		if (args[3]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[3]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribP3uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glVertexAttribP3uiv(index, type, normalized, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribP4ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttribP4ui requires 4 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();
		GLboolean normalized = GLboolean(args[2]->Uint32Value());
		GLuint value = args[3]->Uint32Value();

		glVertexAttribP4ui(index, type, normalized, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexAttribP4uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("vertexAttribP4uiv requires 4 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();
		GLboolean normalized = GLboolean(args[2]->Uint32Value());

		GLuint* value = nullptr;
		if (args[3]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[3]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glVertexAttribP4uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glVertexAttribP4uiv(index, type, normalized, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexP2ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexP2ui requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();
		GLuint value = args[1]->Uint32Value();

		glVertexP2ui(type, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexP2uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexP2uiv requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();

		GLuint* value = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glVertexP2uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glVertexP2uiv(type, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexP3ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexP3ui requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();
		GLuint value = args[1]->Uint32Value();

		glVertexP3ui(type, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexP3uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexP3uiv requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();

		GLuint* value = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glVertexP3uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glVertexP3uiv(type, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexP4ui"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexP4ui requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();
		GLuint value = args[1]->Uint32Value();

		glVertexP4ui(type, value);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "vertexP4uiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("vertexP4uiv requires 2 arguments");
			return;
		}

		GLenum type = args[0]->Uint32Value();

		GLuint* value = nullptr;
		if (args[1]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[1]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			value = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glVertexP4uiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glVertexP4uiv(type, value);
	}));



	/* ------------------------------ GL_ARB_viewport_array ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("DEPTH_RANGE", GL_DEPTH_RANGE);
	CREATE_CONSTANT_ACCESSOR("VIEWPORT", GL_VIEWPORT);
	CREATE_CONSTANT_ACCESSOR("SCISSOR_BOX", GL_SCISSOR_BOX);
	CREATE_CONSTANT_ACCESSOR("SCISSOR_TEST", GL_SCISSOR_TEST);
	CREATE_CONSTANT_ACCESSOR("MAX_VIEWPORTS", GL_MAX_VIEWPORTS);
	CREATE_CONSTANT_ACCESSOR("VIEWPORT_SUBPIXEL_BITS", GL_VIEWPORT_SUBPIXEL_BITS);
	CREATE_CONSTANT_ACCESSOR("VIEWPORT_BOUNDS_RANGE", GL_VIEWPORT_BOUNDS_RANGE);
	CREATE_CONSTANT_ACCESSOR("LAYER_PROVOKING_VERTEX", GL_LAYER_PROVOKING_VERTEX);
	CREATE_CONSTANT_ACCESSOR("VIEWPORT_INDEX_PROVOKING_VERTEX", GL_VIEWPORT_INDEX_PROVOKING_VERTEX);
	CREATE_CONSTANT_ACCESSOR("UNDEFINED_VERTEX", GL_UNDEFINED_VERTEX);
	CREATE_CONSTANT_ACCESSOR("FIRST_VERTEX_CONVENTION", GL_FIRST_VERTEX_CONVENTION);
	CREATE_CONSTANT_ACCESSOR("LAST_VERTEX_CONVENTION", GL_LAST_VERTEX_CONVENTION);
	CREATE_CONSTANT_ACCESSOR("PROVOKING_VERTEX", GL_PROVOKING_VERTEX);



	tpl->Set(String::NewFromUtf8(isolate, "getDoublei_v"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getDoublei_v requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();

		GLdouble* data = nullptr;
		if (args[2]->IsFloat64Array()) {
			v8::Local<v8::Float64Array> view = (args[2]).As<v8::Float64Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<GLdouble*>(bdata);
		} else {
			cout << "ERROR(glGetDoublei_v): array must be of type Float64Array" << endl;
			exit(1);
		}


		glGetDoublei_v(target, index, data);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getFloati_v"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getFloati_v requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLuint index = args[1]->Uint32Value();

		GLfloat* data = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glGetFloati_v): array must be of type Float32Array" << endl;
			exit(1);
		}


		glGetFloati_v(target, index, data);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "scissorIndexed"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("scissorIndexed requires 5 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLint left = args[1]->Int32Value();
		GLint bottom = args[2]->Int32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();

		glScissorIndexed(index, left, bottom, width, height);
	}));



	tpl->Set(String::NewFromUtf8(isolate, "viewportIndexedf"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("viewportIndexedf requires 5 arguments");
			return;
		}

		GLuint index = args[0]->Uint32Value();
		GLfloat x = GLfloat(args[1]->NumberValue());
		GLfloat y = GLfloat(args[2]->NumberValue());
		GLfloat w = GLfloat(args[3]->NumberValue());
		GLfloat h = GLfloat(args[4]->NumberValue());

		glViewportIndexedf(index, x, y, w, h);
	}));




	// empty / skipped / ignored: GL_ARB_window_pos
	/* ------------------------------ GL_ARM_mali_program_binary ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("MALI_PROGRAM_BINARY_ARM", GL_MALI_PROGRAM_BINARY_ARM);



	/* ------------------------------ GL_ARM_mali_shader_binary ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("MALI_SHADER_BINARY_ARM", GL_MALI_SHADER_BINARY_ARM);



	// empty / skipped / ignored: GL_ARM_rgba8
	/* ------------------------------ GL_ARM_shader_framebuffer_fetch ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("FETCH_PER_SAMPLE_ARM", GL_FETCH_PER_SAMPLE_ARM);
	CREATE_CONSTANT_ACCESSOR("FRAGMENT_SHADER_FRAMEBUFFER_FETCH_MRT_ARM", GL_FRAGMENT_SHADER_FRAMEBUFFER_FETCH_MRT_ARM);



	// empty / skipped / ignored: GL_ARM_shader_framebuffer_fetch_depth_stencil
	/* ------------------------------ GL_ATIX_point_sprites ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_POINT_MODE_ATIX", GL_TEXTURE_POINT_MODE_ATIX);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_POINT_ONE_COORD_ATIX", GL_TEXTURE_POINT_ONE_COORD_ATIX);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_POINT_SPRITE_ATIX", GL_TEXTURE_POINT_SPRITE_ATIX);
	CREATE_CONSTANT_ACCESSOR("POINT_SPRITE_CULL_MODE_ATIX", GL_POINT_SPRITE_CULL_MODE_ATIX);
	CREATE_CONSTANT_ACCESSOR("POINT_SPRITE_CULL_CENTER_ATIX", GL_POINT_SPRITE_CULL_CENTER_ATIX);
	CREATE_CONSTANT_ACCESSOR("POINT_SPRITE_CULL_CLIP_ATIX", GL_POINT_SPRITE_CULL_CLIP_ATIX);



	/* ------------------------------ GL_ATIX_texture_env_combine3 ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("MODULATE_ADD_ATIX", GL_MODULATE_ADD_ATIX);
	CREATE_CONSTANT_ACCESSOR("MODULATE_SIGNED_ADD_ATIX", GL_MODULATE_SIGNED_ADD_ATIX);
	CREATE_CONSTANT_ACCESSOR("MODULATE_SUBTRACT_ATIX", GL_MODULATE_SUBTRACT_ATIX);



	/* ------------------------------ GL_ATIX_texture_env_route ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("SECONDARY_COLOR_ATIX", GL_SECONDARY_COLOR_ATIX);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_OUTPUT_RGB_ATIX", GL_TEXTURE_OUTPUT_RGB_ATIX);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_OUTPUT_ALPHA_ATIX", GL_TEXTURE_OUTPUT_ALPHA_ATIX);



	/* ------------------------------ GL_ATIX_vertex_shader_output_point_size ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("OUTPUT_POINT_SIZE_ATIX", GL_OUTPUT_POINT_SIZE_ATIX);



	// empty / skipped / ignored: GL_ATI_draw_buffers
	// empty / skipped / ignored: GL_ATI_element_array
	// empty / skipped / ignored: GL_ATI_envmap_bumpmap
	// empty / skipped / ignored: GL_ATI_fragment_shader
	// empty / skipped / ignored: GL_ATI_map_object_buffer
	// empty / skipped / ignored: GL_ATI_meminfo
	// empty / skipped / ignored: GL_ATI_pn_triangles
	// empty / skipped / ignored: GL_ATI_separate_stencil
	// empty / skipped / ignored: GL_ATI_shader_texture_lod
	// empty / skipped / ignored: GL_ATI_texture_compression_3dc
	// empty / skipped / ignored: GL_ATI_texture_env_combine3
	// empty / skipped / ignored: GL_ATI_texture_float
	// empty / skipped / ignored: GL_ATI_texture_mirror_once
	// empty / skipped / ignored: GL_ATI_text_fragment_shader
	// empty / skipped / ignored: GL_ATI_vertex_array_object
	// empty / skipped / ignored: GL_ATI_vertex_attrib_array_object
	// empty / skipped / ignored: GL_ATI_vertex_streams
	// empty / skipped / ignored: GL_EGL_KHR_context_flush_control
	// empty / skipped / ignored: GL_EGL_NV_robustness_video_memory_purge
	// empty / skipped / ignored: GL_EXT_422_pixels
	// empty / skipped / ignored: GL_EXT_abgr
	// empty / skipped / ignored: GL_EXT_base_instance
	// empty / skipped / ignored: GL_EXT_bgra
	// empty / skipped / ignored: GL_EXT_bindable_uniform
	// empty / skipped / ignored: GL_EXT_blend_color
	// empty / skipped / ignored: GL_EXT_blend_equation_separate
	// empty / skipped / ignored: GL_EXT_blend_func_extended
	// empty / skipped / ignored: GL_EXT_blend_func_separate
	// empty / skipped / ignored: GL_EXT_blend_logic_op
	// empty / skipped / ignored: GL_EXT_blend_minmax
	// empty / skipped / ignored: GL_EXT_blend_subtract
	/* ------------------------------ GL_EXT_buffer_storage ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("MAP_READ_BIT", GL_MAP_READ_BIT);
	CREATE_CONSTANT_ACCESSOR("MAP_WRITE_BIT", GL_MAP_WRITE_BIT);



	// empty / skipped / ignored: GL_EXT_Cg_shader
	// empty / skipped / ignored: GL_EXT_clear_texture
	// empty / skipped / ignored: GL_EXT_clip_cull_distance
	// empty / skipped / ignored: GL_EXT_clip_volume_hint
	// empty / skipped / ignored: GL_EXT_cmyka
	// empty / skipped / ignored: GL_EXT_color_buffer_float
	// empty / skipped / ignored: GL_EXT_color_buffer_half_float
	// empty / skipped / ignored: GL_EXT_color_subtable
	// empty / skipped / ignored: GL_EXT_compiled_vertex_array
	// empty / skipped / ignored: GL_EXT_compressed_ETC1_RGB8_sub_texture
	// empty / skipped / ignored: GL_EXT_conservative_depth
	// empty / skipped / ignored: GL_EXT_convolution
	// empty / skipped / ignored: GL_EXT_coordinate_frame
	// empty / skipped / ignored: GL_EXT_copy_image
	// empty / skipped / ignored: GL_EXT_copy_texture
	// empty / skipped / ignored: GL_EXT_cull_vertex
	// empty / skipped / ignored: GL_EXT_debug_label
	// empty / skipped / ignored: GL_EXT_debug_marker
	// empty / skipped / ignored: GL_EXT_depth_bounds_test
	// empty / skipped / ignored: GL_EXT_direct_state_access
	// empty / skipped / ignored: GL_EXT_discard_framebuffer
	// empty / skipped / ignored: GL_EXT_draw_buffers
	// empty / skipped / ignored: GL_EXT_draw_buffers2
	// empty / skipped / ignored: GL_EXT_draw_buffers_indexed
	// empty / skipped / ignored: GL_EXT_draw_elements_base_vertex
	// empty / skipped / ignored: GL_EXT_draw_instanced
	// empty / skipped / ignored: GL_EXT_draw_range_elements
	// empty / skipped / ignored: GL_EXT_EGL_image_array
	// empty / skipped / ignored: GL_EXT_external_buffer
	// empty / skipped / ignored: GL_EXT_float_blend
	// empty / skipped / ignored: GL_EXT_fog_coord
	// empty / skipped / ignored: GL_EXT_fragment_lighting
	// empty / skipped / ignored: GL_EXT_frag_depth
	// empty / skipped / ignored: GL_EXT_framebuffer_blit
	// empty / skipped / ignored: GL_EXT_framebuffer_multisample
	// empty / skipped / ignored: GL_EXT_framebuffer_multisample_blit_scaled
	// empty / skipped / ignored: GL_EXT_framebuffer_object
	// empty / skipped / ignored: GL_EXT_framebuffer_sRGB
	// empty / skipped / ignored: GL_EXT_geometry_point_size
	// empty / skipped / ignored: GL_EXT_geometry_shader
	// empty / skipped / ignored: GL_EXT_geometry_shader4
	// empty / skipped / ignored: GL_EXT_gpu_program_parameters
	// empty / skipped / ignored: GL_EXT_gpu_shader4
	// empty / skipped / ignored: GL_EXT_gpu_shader5
	// empty / skipped / ignored: GL_EXT_histogram
	// empty / skipped / ignored: GL_EXT_index_array_formats
	// empty / skipped / ignored: GL_EXT_index_func
	// empty / skipped / ignored: GL_EXT_index_material
	// empty / skipped / ignored: GL_EXT_index_texture
	// empty / skipped / ignored: GL_EXT_instanced_arrays
	// empty / skipped / ignored: GL_EXT_light_texture
	// empty / skipped / ignored: GL_EXT_map_buffer_range
	// empty / skipped / ignored: GL_EXT_memory_object
	// empty / skipped / ignored: GL_EXT_memory_object_fd
	// empty / skipped / ignored: GL_EXT_memory_object_win32
	// empty / skipped / ignored: GL_EXT_misc_attribute
	// empty / skipped / ignored: GL_EXT_multiple_textures
	// empty / skipped / ignored: GL_EXT_multisample
	// empty / skipped / ignored: GL_EXT_multisampled_render_to_texture
	// empty / skipped / ignored: GL_EXT_multisampled_render_to_texture2
	// empty / skipped / ignored: GL_EXT_multisample_compatibility
	// empty / skipped / ignored: GL_EXT_multiview_draw_buffers
	// empty / skipped / ignored: GL_EXT_multi_draw_arrays
	// empty / skipped / ignored: GL_EXT_multi_draw_indirect
	// empty / skipped / ignored: GL_EXT_packed_depth_stencil
	// empty / skipped / ignored: GL_EXT_packed_float
	// empty / skipped / ignored: GL_EXT_packed_pixels
	/* ------------------------------ GL_EXT_paletted_texture ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_1D", GL_TEXTURE_1D);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_2D", GL_TEXTURE_2D);
	CREATE_CONSTANT_ACCESSOR("PROXY_TEXTURE_1D", GL_PROXY_TEXTURE_1D);
	CREATE_CONSTANT_ACCESSOR("PROXY_TEXTURE_2D", GL_PROXY_TEXTURE_2D);



	// empty / skipped / ignored: GL_EXT_pixel_buffer_object
	// empty / skipped / ignored: GL_EXT_pixel_transform
	// empty / skipped / ignored: GL_EXT_pixel_transform_color_table
	// empty / skipped / ignored: GL_EXT_point_parameters
	// empty / skipped / ignored: GL_EXT_polygon_offset
	// empty / skipped / ignored: GL_EXT_polygon_offset_clamp
	// empty / skipped / ignored: GL_EXT_post_depth_coverage
	// empty / skipped / ignored: GL_EXT_provoking_vertex
	// empty / skipped / ignored: GL_EXT_pvrtc_sRGB
	// empty / skipped / ignored: GL_EXT_raster_multisample
	// empty / skipped / ignored: GL_EXT_read_format_bgra
	/* ------------------------------ GL_EXT_render_snorm ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("BYTE", GL_BYTE);
	CREATE_CONSTANT_ACCESSOR("SHORT", GL_SHORT);
	CREATE_CONSTANT_ACCESSOR("R8_SNORM", GL_R8_SNORM);
	CREATE_CONSTANT_ACCESSOR("RG8_SNORM", GL_RG8_SNORM);
	CREATE_CONSTANT_ACCESSOR("RGBA8_SNORM", GL_RGBA8_SNORM);



	// empty / skipped / ignored: GL_EXT_rescale_normal
	// empty / skipped / ignored: GL_EXT_scene_marker
	// empty / skipped / ignored: GL_EXT_secondary_color
	// empty / skipped / ignored: GL_EXT_semaphore
	// empty / skipped / ignored: GL_EXT_semaphore_fd
	// empty / skipped / ignored: GL_EXT_semaphore_win32
	// empty / skipped / ignored: GL_EXT_separate_shader_objects
	// empty / skipped / ignored: GL_EXT_separate_specular_color
	// empty / skipped / ignored: GL_EXT_shader_framebuffer_fetch
	// empty / skipped / ignored: GL_EXT_shader_group_vote
	// empty / skipped / ignored: GL_EXT_shader_image_load_formatted
	// empty / skipped / ignored: GL_EXT_shader_image_load_store
	// empty / skipped / ignored: GL_EXT_shader_implicit_conversions
	// empty / skipped / ignored: GL_EXT_shader_integer_mix
	// empty / skipped / ignored: GL_EXT_shader_io_blocks
	// empty / skipped / ignored: GL_EXT_shader_non_constant_global_initializers
	// empty / skipped / ignored: GL_EXT_shader_pixel_local_storage
	// empty / skipped / ignored: GL_EXT_shader_pixel_local_storage2
	// empty / skipped / ignored: GL_EXT_shader_texture_lod
	// empty / skipped / ignored: GL_EXT_shadow_funcs
	// empty / skipped / ignored: GL_EXT_shadow_samplers
	// empty / skipped / ignored: GL_EXT_shared_texture_palette
	/* ------------------------------ GL_EXT_sparse_texture ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_2D", GL_TEXTURE_2D);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_3D", GL_TEXTURE_3D);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_CUBE_MAP", GL_TEXTURE_CUBE_MAP);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_2D_ARRAY", GL_TEXTURE_2D_ARRAY);



	// empty / skipped / ignored: GL_EXT_sparse_texture2
	// empty / skipped / ignored: GL_EXT_sRGB
	// empty / skipped / ignored: GL_EXT_sRGB_write_control
	// empty / skipped / ignored: GL_EXT_stencil_clear_tag
	// empty / skipped / ignored: GL_EXT_stencil_two_side
	// empty / skipped / ignored: GL_EXT_stencil_wrap
	// empty / skipped / ignored: GL_EXT_subtexture
	// empty / skipped / ignored: GL_EXT_texture
	// empty / skipped / ignored: GL_EXT_texture3D
	// empty / skipped / ignored: GL_EXT_texture_array
	// empty / skipped / ignored: GL_EXT_texture_buffer_object
	// empty / skipped / ignored: GL_EXT_texture_compression_astc_decode_mode
	// empty / skipped / ignored: GL_EXT_texture_compression_astc_decode_mode_rgb9e5
	// empty / skipped / ignored: GL_EXT_texture_compression_bptc
	// empty / skipped / ignored: GL_EXT_texture_compression_dxt1
	// empty / skipped / ignored: GL_EXT_texture_compression_latc
	// empty / skipped / ignored: GL_EXT_texture_compression_rgtc
	// empty / skipped / ignored: GL_EXT_texture_compression_s3tc
	// empty / skipped / ignored: GL_EXT_texture_cube_map
	// empty / skipped / ignored: GL_EXT_texture_cube_map_array
	/* ------------------------------ GL_EXT_texture_edge_clamp ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("CLAMP_TO_EDGE_EXT", GL_CLAMP_TO_EDGE_EXT);



	// empty / skipped / ignored: GL_EXT_texture_env
	// empty / skipped / ignored: GL_EXT_texture_env_add
	// empty / skipped / ignored: GL_EXT_texture_env_combine
	// empty / skipped / ignored: GL_EXT_texture_env_dot3
	// empty / skipped / ignored: GL_EXT_texture_filter_anisotropic
	// empty / skipped / ignored: GL_EXT_texture_filter_minmax
	// empty / skipped / ignored: GL_EXT_texture_format_BGRA8888
	// empty / skipped / ignored: GL_EXT_texture_integer
	// empty / skipped / ignored: GL_EXT_texture_lod_bias
	// empty / skipped / ignored: GL_EXT_texture_mirror_clamp
	// empty / skipped / ignored: GL_EXT_texture_norm16
	// empty / skipped / ignored: GL_EXT_texture_object
	// empty / skipped / ignored: GL_EXT_texture_perturb_normal
	// empty / skipped / ignored: GL_EXT_texture_rectangle
	// empty / skipped / ignored: GL_EXT_texture_rg
	// empty / skipped / ignored: GL_EXT_texture_shared_exponent
	/* ------------------------------ GL_EXT_texture_snorm ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("RED_SNORM", GL_RED_SNORM);
	CREATE_CONSTANT_ACCESSOR("RG_SNORM", GL_RG_SNORM);
	CREATE_CONSTANT_ACCESSOR("RGB_SNORM", GL_RGB_SNORM);
	CREATE_CONSTANT_ACCESSOR("RGBA_SNORM", GL_RGBA_SNORM);
	CREATE_CONSTANT_ACCESSOR("R8_SNORM", GL_R8_SNORM);
	CREATE_CONSTANT_ACCESSOR("RG8_SNORM", GL_RG8_SNORM);
	CREATE_CONSTANT_ACCESSOR("RGB8_SNORM", GL_RGB8_SNORM);
	CREATE_CONSTANT_ACCESSOR("RGBA8_SNORM", GL_RGBA8_SNORM);
	CREATE_CONSTANT_ACCESSOR("R16_SNORM", GL_R16_SNORM);
	CREATE_CONSTANT_ACCESSOR("RG16_SNORM", GL_RG16_SNORM);
	CREATE_CONSTANT_ACCESSOR("RGB16_SNORM", GL_RGB16_SNORM);
	CREATE_CONSTANT_ACCESSOR("RGBA16_SNORM", GL_RGBA16_SNORM);
	CREATE_CONSTANT_ACCESSOR("SIGNED_NORMALIZED", GL_SIGNED_NORMALIZED);
	CREATE_CONSTANT_ACCESSOR("ALPHA_SNORM", GL_ALPHA_SNORM);
	CREATE_CONSTANT_ACCESSOR("LUMINANCE_SNORM", GL_LUMINANCE_SNORM);
	CREATE_CONSTANT_ACCESSOR("LUMINANCE_ALPHA_SNORM", GL_LUMINANCE_ALPHA_SNORM);
	CREATE_CONSTANT_ACCESSOR("INTENSITY_SNORM", GL_INTENSITY_SNORM);
	CREATE_CONSTANT_ACCESSOR("ALPHA8_SNORM", GL_ALPHA8_SNORM);
	CREATE_CONSTANT_ACCESSOR("LUMINANCE8_SNORM", GL_LUMINANCE8_SNORM);
	CREATE_CONSTANT_ACCESSOR("LUMINANCE8_ALPHA8_SNORM", GL_LUMINANCE8_ALPHA8_SNORM);
	CREATE_CONSTANT_ACCESSOR("INTENSITY8_SNORM", GL_INTENSITY8_SNORM);
	CREATE_CONSTANT_ACCESSOR("ALPHA16_SNORM", GL_ALPHA16_SNORM);
	CREATE_CONSTANT_ACCESSOR("LUMINANCE16_SNORM", GL_LUMINANCE16_SNORM);
	CREATE_CONSTANT_ACCESSOR("LUMINANCE16_ALPHA16_SNORM", GL_LUMINANCE16_ALPHA16_SNORM);
	CREATE_CONSTANT_ACCESSOR("INTENSITY16_SNORM", GL_INTENSITY16_SNORM);



	// empty / skipped / ignored: GL_EXT_texture_sRGB
	// empty / skipped / ignored: GL_EXT_texture_sRGB_decode
	// empty / skipped / ignored: GL_EXT_texture_sRGB_R8
	// empty / skipped / ignored: GL_EXT_texture_sRGB_RG8
	// empty / skipped / ignored: GL_EXT_texture_storage
	// empty / skipped / ignored: GL_EXT_texture_swizzle
	// empty / skipped / ignored: GL_EXT_texture_type_2_10_10_10_REV
	/* ------------------------------ GL_EXT_texture_view ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_IMMUTABLE_LEVELS", GL_TEXTURE_IMMUTABLE_LEVELS);



	// empty / skipped / ignored: GL_EXT_timer_query
	// empty / skipped / ignored: GL_EXT_transform_feedback
	// empty / skipped / ignored: GL_EXT_unpack_subimage
	// empty / skipped / ignored: GL_EXT_vertex_array
	/* ------------------------------ GL_EXT_vertex_array_bgra ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("BGRA", GL_BGRA);



	// empty / skipped / ignored: GL_EXT_vertex_array_setXXX
	// empty / skipped / ignored: GL_EXT_vertex_attrib_64bit
	// empty / skipped / ignored: GL_EXT_vertex_shader
	// empty / skipped / ignored: GL_EXT_vertex_weighting
	// empty / skipped / ignored: GL_EXT_win32_keyed_mutex
	// empty / skipped / ignored: GL_EXT_window_rectangles
	// empty / skipped / ignored: GL_EXT_x11_sync_object
	// empty / skipped / ignored: GL_EXT_YUV_target
	/* ------------------------------ GL_GREMEDY_frame_terminator ------------------------------ */





	/* ------------------------------ GL_GREMEDY_string_marker ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "stringMarkerGREMEDY"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("stringMarkerGREMEDY requires 2 arguments");
			return;
		}

		GLsizei len = args[0]->Int32Value();

		void* string = nullptr;
		if (args[1]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[1]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			string = reinterpret_cast<void*>(bdata);
		} else if (args[1]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[1]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			string = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glStringMarkerGREMEDY): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glStringMarkerGREMEDY(len, string);
	}));



	// empty / skipped / ignored: GL_HP_convolution_border_modes
	/* ------------------------------ GL_HP_image_transform ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "getImageTransformParameterfvHP"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getImageTransformParameterfvHP requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfloat* params = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glGetImageTransformParameterfvHP): array must be of type Float32Array" << endl;
			exit(1);
		}


		glGetImageTransformParameterfvHP(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getImageTransformParameterivHP"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getImageTransformParameterivHP requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetImageTransformParameterivHP): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetImageTransformParameterivHP(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "imageTransformParameterfHP"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("imageTransformParameterfHP requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLfloat param = GLfloat(args[2]->NumberValue());

		glImageTransformParameterfHP(target, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "imageTransformParameterfvHP"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("imageTransformParameterfvHP requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfloat* params = nullptr;
		if (args[2]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[2]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glImageTransformParameterfvHP): array must be of type Float32Array" << endl;
			exit(1);
		}


		glImageTransformParameterfvHP(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "imageTransformParameteriHP"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("imageTransformParameteriHP requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLint param = args[2]->Int32Value();

		glImageTransformParameteriHP(target, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "imageTransformParameterivHP"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("imageTransformParameterivHP requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLint* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glImageTransformParameterivHP): array must be of type Int32Array" << endl;
			exit(1);
		}


		glImageTransformParameterivHP(target, pname, params);
	}));



	// empty / skipped / ignored: GL_HP_occlusion_test
	// empty / skipped / ignored: GL_HP_texture_lighting
	// empty / skipped / ignored: GL_IBM_cull_vertex
	// empty / skipped / ignored: GL_IBM_multimode_draw_arrays
	// empty / skipped / ignored: GL_IBM_rasterpos_clip
	// empty / skipped / ignored: GL_IBM_static_data
	// empty / skipped / ignored: GL_IBM_texture_mirrored_repeat
	// empty / skipped / ignored: GL_IBM_vertex_array_lists
	// empty / skipped / ignored: GL_INGR_color_clamp
	// empty / skipped / ignored: GL_INGR_interlace_read
	// empty / skipped / ignored: GL_INTEL_conservative_rasterization
	// empty / skipped / ignored: GL_INTEL_fragment_shader_ordering
	// empty / skipped / ignored: GL_INTEL_framebuffer_CMAA
	// empty / skipped / ignored: GL_INTEL_map_texture
	// empty / skipped / ignored: GL_INTEL_parallel_arrays
	// empty / skipped / ignored: GL_INTEL_performance_query
	// empty / skipped / ignored: GL_INTEL_texture_scissor
	// empty / skipped / ignored: GL_KHR_blend_equation_advanced
	// empty / skipped / ignored: GL_KHR_blend_equation_advanced_coherent
	// empty / skipped / ignored: GL_KHR_context_flush_control
	/* ------------------------------ GL_KHR_debug ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("CONTEXT_FLAG_DEBUG_BIT", GL_CONTEXT_FLAG_DEBUG_BIT);
	CREATE_CONSTANT_ACCESSOR("STACK_OVERFLOW", GL_STACK_OVERFLOW);
	CREATE_CONSTANT_ACCESSOR("STACK_UNDERFLOW", GL_STACK_UNDERFLOW);
	CREATE_CONSTANT_ACCESSOR("DEBUG_OUTPUT_SYNCHRONOUS", GL_DEBUG_OUTPUT_SYNCHRONOUS);
	CREATE_CONSTANT_ACCESSOR("DEBUG_NEXT_LOGGED_MESSAGE_LENGTH", GL_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH);
	CREATE_CONSTANT_ACCESSOR("DEBUG_CALLBACK_FUNCTION", GL_DEBUG_CALLBACK_FUNCTION);
	CREATE_CONSTANT_ACCESSOR("DEBUG_CALLBACK_USER_PARAM", GL_DEBUG_CALLBACK_USER_PARAM);
	CREATE_CONSTANT_ACCESSOR("DEBUG_SOURCE_API", GL_DEBUG_SOURCE_API);
	CREATE_CONSTANT_ACCESSOR("DEBUG_SOURCE_WINDOW_SYSTEM", GL_DEBUG_SOURCE_WINDOW_SYSTEM);
	CREATE_CONSTANT_ACCESSOR("DEBUG_SOURCE_SHADER_COMPILER", GL_DEBUG_SOURCE_SHADER_COMPILER);
	CREATE_CONSTANT_ACCESSOR("DEBUG_SOURCE_THIRD_PARTY", GL_DEBUG_SOURCE_THIRD_PARTY);
	CREATE_CONSTANT_ACCESSOR("DEBUG_SOURCE_APPLICATION", GL_DEBUG_SOURCE_APPLICATION);
	CREATE_CONSTANT_ACCESSOR("DEBUG_SOURCE_OTHER", GL_DEBUG_SOURCE_OTHER);
	CREATE_CONSTANT_ACCESSOR("DEBUG_TYPE_ERROR", GL_DEBUG_TYPE_ERROR);
	CREATE_CONSTANT_ACCESSOR("DEBUG_TYPE_DEPRECATED_BEHAVIOR", GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR);
	CREATE_CONSTANT_ACCESSOR("DEBUG_TYPE_UNDEFINED_BEHAVIOR", GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR);
	CREATE_CONSTANT_ACCESSOR("DEBUG_TYPE_PORTABILITY", GL_DEBUG_TYPE_PORTABILITY);
	CREATE_CONSTANT_ACCESSOR("DEBUG_TYPE_PERFORMANCE", GL_DEBUG_TYPE_PERFORMANCE);
	CREATE_CONSTANT_ACCESSOR("DEBUG_TYPE_OTHER", GL_DEBUG_TYPE_OTHER);
	CREATE_CONSTANT_ACCESSOR("DEBUG_TYPE_MARKER", GL_DEBUG_TYPE_MARKER);
	CREATE_CONSTANT_ACCESSOR("DEBUG_TYPE_PUSH_GROUP", GL_DEBUG_TYPE_PUSH_GROUP);
	CREATE_CONSTANT_ACCESSOR("DEBUG_TYPE_POP_GROUP", GL_DEBUG_TYPE_POP_GROUP);
	CREATE_CONSTANT_ACCESSOR("DEBUG_SEVERITY_NOTIFICATION", GL_DEBUG_SEVERITY_NOTIFICATION);
	CREATE_CONSTANT_ACCESSOR("MAX_DEBUG_GROUP_STACK_DEPTH", GL_MAX_DEBUG_GROUP_STACK_DEPTH);
	CREATE_CONSTANT_ACCESSOR("DEBUG_GROUP_STACK_DEPTH", GL_DEBUG_GROUP_STACK_DEPTH);
	CREATE_CONSTANT_ACCESSOR("BUFFER", GL_BUFFER);
	CREATE_CONSTANT_ACCESSOR("SHADER", GL_SHADER);
	CREATE_CONSTANT_ACCESSOR("PROGRAM", GL_PROGRAM);
	CREATE_CONSTANT_ACCESSOR("QUERY", GL_QUERY);
	CREATE_CONSTANT_ACCESSOR("PROGRAM_PIPELINE", GL_PROGRAM_PIPELINE);
	CREATE_CONSTANT_ACCESSOR("SAMPLER", GL_SAMPLER);
	CREATE_CONSTANT_ACCESSOR("DISPLAY_LIST", GL_DISPLAY_LIST);
	CREATE_CONSTANT_ACCESSOR("MAX_LABEL_LENGTH", GL_MAX_LABEL_LENGTH);
	CREATE_CONSTANT_ACCESSOR("MAX_DEBUG_MESSAGE_LENGTH", GL_MAX_DEBUG_MESSAGE_LENGTH);
	CREATE_CONSTANT_ACCESSOR("MAX_DEBUG_LOGGED_MESSAGES", GL_MAX_DEBUG_LOGGED_MESSAGES);
	CREATE_CONSTANT_ACCESSOR("DEBUG_LOGGED_MESSAGES", GL_DEBUG_LOGGED_MESSAGES);
	CREATE_CONSTANT_ACCESSOR("DEBUG_SEVERITY_HIGH", GL_DEBUG_SEVERITY_HIGH);
	CREATE_CONSTANT_ACCESSOR("DEBUG_SEVERITY_MEDIUM", GL_DEBUG_SEVERITY_MEDIUM);
	CREATE_CONSTANT_ACCESSOR("DEBUG_SEVERITY_LOW", GL_DEBUG_SEVERITY_LOW);
	CREATE_CONSTANT_ACCESSOR("DEBUG_OUTPUT", GL_DEBUG_OUTPUT);


	tpl->Set(String::NewFromUtf8(isolate, "debugMessageControl"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("debugMessageControl requires 6 arguments");
			return;
		}

		GLenum source = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();
		GLenum severity = args[2]->Uint32Value();
		GLsizei count = args[3]->Int32Value();

		GLuint* ids = nullptr;
		if (args[4]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[4]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			ids = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glDebugMessageControl): array must be of type Uint32Array" << endl;
			exit(1);
		}

		GLboolean enabled = GLboolean(args[5]->Uint32Value());

		glDebugMessageControl(source, type, severity, count, ids, enabled);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "debugMessageInsert"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("debugMessageInsert requires 6 arguments");
			return;
		}

		GLenum source = args[0]->Uint32Value();
		GLenum type = args[1]->Uint32Value();
		GLuint id = args[2]->Uint32Value();
		GLenum severity = args[3]->Uint32Value();
		GLsizei length = args[4]->Int32Value();

		GLchar* buf = nullptr;
		if (args[5]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[5]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			buf = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glDebugMessageInsert): array must be of type Int8Array" << endl;
			exit(1);
		}


		glDebugMessageInsert(source, type, id, severity, length, buf);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getObjectLabel"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("getObjectLabel requires 5 arguments");
			return;
		}

		GLenum identifier = args[0]->Uint32Value();
		GLuint name = args[1]->Uint32Value();
		GLsizei bufSize = args[2]->Int32Value();

		GLsizei* length = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetObjectLabel): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLchar* label = nullptr;
		if (args[4]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[4]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			label = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetObjectLabel): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetObjectLabel(identifier, name, bufSize, length, label);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getObjectPtrLabel"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getObjectPtrLabel requires 4 arguments");
			return;
		}


		void* ptr = nullptr;
		if (args[0]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[0]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			ptr = reinterpret_cast<void*>(bdata);
		} else if (args[0]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[0]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			ptr = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glGetObjectPtrLabel): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLsizei bufSize = args[1]->Int32Value();

		GLsizei* length = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetObjectPtrLabel): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLchar* label = nullptr;
		if (args[3]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[3]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			label = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetObjectPtrLabel): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetObjectPtrLabel(ptr, bufSize, length, label);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "objectLabel"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("objectLabel requires 4 arguments");
			return;
		}

		GLenum identifier = args[0]->Uint32Value();
		GLuint name = args[1]->Uint32Value();
		GLsizei length = args[2]->Int32Value();

		GLchar* label = nullptr;
		if (args[3]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[3]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			label = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glObjectLabel): array must be of type Int8Array" << endl;
			exit(1);
		}


		glObjectLabel(identifier, name, length, label);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "objectPtrLabel"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("objectPtrLabel requires 3 arguments");
			return;
		}


		void* ptr = nullptr;
		if (args[0]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[0]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			ptr = reinterpret_cast<void*>(bdata);
		} else if (args[0]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[0]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			ptr = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glObjectPtrLabel): array must be of type ArrayBuffer" << endl;
			exit(1);
		}

		GLsizei length = args[1]->Int32Value();

		GLchar* label = nullptr;
		if (args[2]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[2]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			label = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glObjectPtrLabel): array must be of type Int8Array" << endl;
			exit(1);
		}


		glObjectPtrLabel(ptr, length, label);
	}));





	// empty / skipped / ignored: GL_KHR_no_error
	// empty / skipped / ignored: GL_KHR_parallel_shader_compile
	/* ------------------------------ GL_KHR_robustness ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("CONTEXT_LOST", GL_CONTEXT_LOST);
	CREATE_CONSTANT_ACCESSOR("LOSE_CONTEXT_ON_RESET", GL_LOSE_CONTEXT_ON_RESET);
	CREATE_CONSTANT_ACCESSOR("GUILTY_CONTEXT_RESET", GL_GUILTY_CONTEXT_RESET);
	CREATE_CONSTANT_ACCESSOR("INNOCENT_CONTEXT_RESET", GL_INNOCENT_CONTEXT_RESET);
	CREATE_CONSTANT_ACCESSOR("UNKNOWN_CONTEXT_RESET", GL_UNKNOWN_CONTEXT_RESET);
	CREATE_CONSTANT_ACCESSOR("RESET_NOTIFICATION_STRATEGY", GL_RESET_NOTIFICATION_STRATEGY);
	CREATE_CONSTANT_ACCESSOR("NO_RESET_NOTIFICATION", GL_NO_RESET_NOTIFICATION);
	CREATE_CONSTANT_ACCESSOR("CONTEXT_ROBUST_ACCESS", GL_CONTEXT_ROBUST_ACCESS);

	tpl->Set(String::NewFromUtf8(isolate, "getnUniformfv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getnUniformfv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei bufSize = args[2]->Int32Value();

		GLfloat* params = nullptr;
		if (args[3]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[3]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glGetnUniformfv): array must be of type Float32Array" << endl;
			exit(1);
		}


		glGetnUniformfv(program, location, bufSize, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getnUniformiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getnUniformiv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei bufSize = args[2]->Int32Value();

		GLint* params = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetnUniformiv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetnUniformiv(program, location, bufSize, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getnUniformuiv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getnUniformuiv requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLint location = args[1]->Int32Value();
		GLsizei bufSize = args[2]->Int32Value();

		GLuint* params = nullptr;
		if (args[3]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[3]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGetnUniformuiv): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGetnUniformuiv(program, location, bufSize, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "readnPixels"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 8) {
			V8Helper::_instance->throwException("readnPixels requires 8 arguments");
			return;
		}

		GLint x = args[0]->Int32Value();
		GLint y = args[1]->Int32Value();
		GLsizei width = args[2]->Int32Value();
		GLsizei height = args[3]->Int32Value();
		GLenum format = args[4]->Uint32Value();
		GLenum type = args[5]->Uint32Value();
		GLsizei bufSize = args[6]->Int32Value();

		void* data = nullptr;
		if (args[7]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[7]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else if (args[7]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[7]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			data = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glReadnPixels): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glReadnPixels(x, y, width, height, format, type, bufSize, data);
	}));



	// empty / skipped / ignored: GL_KHR_robust_buffer_access_behavior
	// empty / skipped / ignored: GL_KHR_texture_compression_astc_hdr
	// empty / skipped / ignored: GL_KHR_texture_compression_astc_ldr
	// empty / skipped / ignored: GL_KHR_texture_compression_astc_sliced_3d
	/* ------------------------------ GL_KTX_buffer_region ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("KTX_FRONT_REGION", GL_KTX_FRONT_REGION);
	CREATE_CONSTANT_ACCESSOR("KTX_BACK_REGION", GL_KTX_BACK_REGION);
	CREATE_CONSTANT_ACCESSOR("KTX_Z_REGION", GL_KTX_Z_REGION);
	CREATE_CONSTANT_ACCESSOR("KTX_STENCIL_REGION", GL_KTX_STENCIL_REGION);

	tpl->Set(String::NewFromUtf8(isolate, "deleteBufferRegion"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("deleteBufferRegion requires 1 arguments");
			return;
		}

		GLenum region = args[0]->Uint32Value();

		glDeleteBufferRegion(region);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "readBufferRegion"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("readBufferRegion requires 5 arguments");
			return;
		}

		GLuint region = args[0]->Uint32Value();
		GLint x = args[1]->Int32Value();
		GLint y = args[2]->Int32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();

		glReadBufferRegion(region, x, y, width, height);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "drawBufferRegion"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 7) {
			V8Helper::_instance->throwException("drawBufferRegion requires 7 arguments");
			return;
		}

		GLuint region = args[0]->Uint32Value();
		GLint x = args[1]->Int32Value();
		GLint y = args[2]->Int32Value();
		GLsizei width = args[3]->Int32Value();
		GLsizei height = args[4]->Int32Value();
		GLint xDest = args[5]->Int32Value();
		GLint yDest = args[6]->Int32Value();

		glDrawBufferRegion(region, x, y, width, height, xDest, yDest);
	}));



	// empty / skipped / ignored: GL_MESAX_texture_stack
	// empty / skipped / ignored: GL_MESA_pack_invert
	// empty / skipped / ignored: GL_MESA_resize_buffers
	// empty / skipped / ignored: GL_MESA_shader_integer_functions
	// empty / skipped / ignored: GL_MESA_window_pos
	// empty / skipped / ignored: GL_MESA_ycbcr_texture
	// empty / skipped / ignored: GL_NVX_blend_equation_advanced_multi_draw_buffers
	// empty / skipped / ignored: GL_NVX_conditional_render
	// empty / skipped / ignored: GL_NVX_gpu_memory_info
	// empty / skipped / ignored: GL_NVX_linked_gpu_multicast
	// empty / skipped / ignored: GL_NV_3dvision_settings
	// empty / skipped / ignored: GL_NV_alpha_to_coverage_dither_control
	// empty / skipped / ignored: GL_NV_bgr
	// empty / skipped / ignored: GL_NV_bindless_multi_draw_indirect
	// empty / skipped / ignored: GL_NV_bindless_multi_draw_indirect_count
	// empty / skipped / ignored: GL_NV_bindless_texture
	// empty / skipped / ignored: GL_NV_blend_equation_advanced
	// empty / skipped / ignored: GL_NV_blend_equation_advanced_coherent
	// empty / skipped / ignored: GL_NV_blend_minmax_factor
	// empty / skipped / ignored: GL_NV_blend_square
	// empty / skipped / ignored: GL_NV_clip_space_w_scaling
	// empty / skipped / ignored: GL_NV_command_list
	// empty / skipped / ignored: GL_NV_compute_program5
	// empty / skipped / ignored: GL_NV_conditional_render
	// empty / skipped / ignored: GL_NV_conservative_raster
	// empty / skipped / ignored: GL_NV_conservative_raster_dilate
	// empty / skipped / ignored: GL_NV_conservative_raster_pre_snap_triangles
	// empty / skipped / ignored: GL_NV_copy_buffer
	// empty / skipped / ignored: GL_NV_copy_depth_to_color
	// empty / skipped / ignored: GL_NV_copy_image
	// empty / skipped / ignored: GL_NV_deep_texture3D
	// empty / skipped / ignored: GL_NV_depth_buffer_float
	// empty / skipped / ignored: GL_NV_depth_clamp
	// empty / skipped / ignored: GL_NV_depth_range_unclamped
	// empty / skipped / ignored: GL_NV_draw_buffers
	// empty / skipped / ignored: GL_NV_draw_instanced
	// empty / skipped / ignored: GL_NV_draw_texture
	// empty / skipped / ignored: GL_NV_draw_vulkan_image
	// empty / skipped / ignored: GL_NV_EGL_stream_consumer_external
	// empty / skipped / ignored: GL_NV_evaluators
	// empty / skipped / ignored: GL_NV_explicit_attrib_location
	// empty / skipped / ignored: GL_NV_explicit_multisample
	// empty / skipped / ignored: GL_NV_fbo_color_attachments
	// empty / skipped / ignored: GL_NV_fence
	// empty / skipped / ignored: GL_NV_fill_rectangle
	// empty / skipped / ignored: GL_NV_float_buffer
	// empty / skipped / ignored: GL_NV_fog_distance
	// empty / skipped / ignored: GL_NV_fragment_coverage_to_color
	// empty / skipped / ignored: GL_NV_fragment_program
	// empty / skipped / ignored: GL_NV_fragment_program2
	// empty / skipped / ignored: GL_NV_fragment_program4
	// empty / skipped / ignored: GL_NV_fragment_program_option
	// empty / skipped / ignored: GL_NV_fragment_shader_interlock
	// empty / skipped / ignored: GL_NV_framebuffer_blit
	// empty / skipped / ignored: GL_NV_framebuffer_mixed_samples
	// empty / skipped / ignored: GL_NV_framebuffer_multisample
	// empty / skipped / ignored: GL_NV_framebuffer_multisample_coverage
	// empty / skipped / ignored: GL_NV_generate_mipmap_sRGB
	// empty / skipped / ignored: GL_NV_geometry_program4
	// empty / skipped / ignored: GL_NV_geometry_shader4
	// empty / skipped / ignored: GL_NV_geometry_shader_passthrough
	// empty / skipped / ignored: GL_NV_gpu_multicast
	// empty / skipped / ignored: GL_NV_gpu_program4
	// empty / skipped / ignored: GL_NV_gpu_program5
	// empty / skipped / ignored: GL_NV_gpu_program5_mem_extended
	// empty / skipped / ignored: GL_NV_gpu_program_fp64
	// empty / skipped / ignored: GL_NV_gpu_shader5
	// empty / skipped / ignored: GL_NV_half_float
	// empty / skipped / ignored: GL_NV_image_formats
	// empty / skipped / ignored: GL_NV_instanced_arrays
	// empty / skipped / ignored: GL_NV_internalformat_sample_query
	// empty / skipped / ignored: GL_NV_light_max_exponent
	// empty / skipped / ignored: GL_NV_multisample_coverage
	// empty / skipped / ignored: GL_NV_multisample_filter_hint
	// empty / skipped / ignored: GL_NV_non_square_matrices
	// empty / skipped / ignored: GL_NV_occlusion_query
	// empty / skipped / ignored: GL_NV_packed_depth_stencil
	// empty / skipped / ignored: GL_NV_packed_float
	// empty / skipped / ignored: GL_NV_packed_float_linear
	// empty / skipped / ignored: GL_NV_pack_subimage
	// empty / skipped / ignored: GL_NV_parameter_buffer_object
	// empty / skipped / ignored: GL_NV_parameter_buffer_object2
	/* ------------------------------ GL_NV_path_rendering ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("PRIMARY_COLOR", GL_PRIMARY_COLOR);



	// empty / skipped / ignored: GL_NV_path_rendering_shared_edge
	// empty / skipped / ignored: GL_NV_pixel_buffer_object
	// empty / skipped / ignored: GL_NV_pixel_data_range
	// empty / skipped / ignored: GL_NV_platform_binary
	// empty / skipped / ignored: GL_NV_point_sprite
	// empty / skipped / ignored: GL_NV_polygon_mode
	// empty / skipped / ignored: GL_NV_present_video
	// empty / skipped / ignored: GL_NV_primitive_restart
	// empty / skipped / ignored: GL_NV_read_depth
	// empty / skipped / ignored: GL_NV_read_depth_stencil
	// empty / skipped / ignored: GL_NV_read_stencil
	// empty / skipped / ignored: GL_NV_register_combiners
	// empty / skipped / ignored: GL_NV_register_combiners2
	// empty / skipped / ignored: GL_NV_robustness_video_memory_purge
	// empty / skipped / ignored: GL_NV_sample_locations
	// empty / skipped / ignored: GL_NV_sample_mask_override_coverage
	// empty / skipped / ignored: GL_NV_shader_atomic_counters
	// empty / skipped / ignored: GL_NV_shader_atomic_float
	// empty / skipped / ignored: GL_NV_shader_atomic_float64
	// empty / skipped / ignored: GL_NV_shader_atomic_fp16_vector
	// empty / skipped / ignored: GL_NV_shader_atomic_int64
	// empty / skipped / ignored: GL_NV_shader_buffer_load
	// empty / skipped / ignored: GL_NV_shader_noperspective_interpolation
	// empty / skipped / ignored: GL_NV_shader_storage_buffer_object
	// empty / skipped / ignored: GL_NV_shader_thread_group
	// empty / skipped / ignored: GL_NV_shader_thread_shuffle
	// empty / skipped / ignored: GL_NV_shadow_samplers_array
	// empty / skipped / ignored: GL_NV_shadow_samplers_cube
	// empty / skipped / ignored: GL_NV_sRGB_formats
	// empty / skipped / ignored: GL_NV_stereo_view_rendering
	// empty / skipped / ignored: GL_NV_tessellation_program5
	// empty / skipped / ignored: GL_NV_texgen_emboss
	// empty / skipped / ignored: GL_NV_texgen_reflection
	// empty / skipped / ignored: GL_NV_texture_array
	// empty / skipped / ignored: GL_NV_texture_barrier
	// empty / skipped / ignored: GL_NV_texture_border_clamp
	// empty / skipped / ignored: GL_NV_texture_compression_latc
	// empty / skipped / ignored: GL_NV_texture_compression_s3tc
	// empty / skipped / ignored: GL_NV_texture_compression_s3tc_update
	// empty / skipped / ignored: GL_NV_texture_compression_vtc
	// empty / skipped / ignored: GL_NV_texture_env_combine4
	// empty / skipped / ignored: GL_NV_texture_expand_normal
	// empty / skipped / ignored: GL_NV_texture_multisample
	// empty / skipped / ignored: GL_NV_texture_npot_2D_mipmap
	// empty / skipped / ignored: GL_NV_texture_rectangle
	// empty / skipped / ignored: GL_NV_texture_rectangle_compressed
	// empty / skipped / ignored: GL_NV_texture_shader
	// empty / skipped / ignored: GL_NV_texture_shader2
	// empty / skipped / ignored: GL_NV_texture_shader3
	// empty / skipped / ignored: GL_NV_transform_feedback
	// empty / skipped / ignored: GL_NV_transform_feedback2
	// empty / skipped / ignored: GL_NV_uniform_buffer_unified_memory
	// empty / skipped / ignored: GL_NV_vdpau_interop
	// empty / skipped / ignored: GL_NV_vertex_array_range
	// empty / skipped / ignored: GL_NV_vertex_array_range2
	// empty / skipped / ignored: GL_NV_vertex_attrib_integer_64bit
	// empty / skipped / ignored: GL_NV_vertex_buffer_unified_memory
	// empty / skipped / ignored: GL_NV_vertex_program
	// empty / skipped / ignored: GL_NV_vertex_program1_1
	// empty / skipped / ignored: GL_NV_vertex_program2
	// empty / skipped / ignored: GL_NV_vertex_program2_option
	// empty / skipped / ignored: GL_NV_vertex_program3
	// empty / skipped / ignored: GL_NV_vertex_program4
	// empty / skipped / ignored: GL_NV_video_capture
	/* ------------------------------ GL_NV_viewport_array ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("DEPTH_RANGE", GL_DEPTH_RANGE);
	CREATE_CONSTANT_ACCESSOR("VIEWPORT", GL_VIEWPORT);
	CREATE_CONSTANT_ACCESSOR("SCISSOR_BOX", GL_SCISSOR_BOX);
	CREATE_CONSTANT_ACCESSOR("SCISSOR_TEST", GL_SCISSOR_TEST);



	// empty / skipped / ignored: GL_NV_viewport_array2
	// empty / skipped / ignored: GL_NV_viewport_swizzle
	// empty / skipped / ignored: GL_OES_byte_coordinates
	// empty / skipped / ignored: GL_OML_interlace
	// empty / skipped / ignored: GL_OML_resample
	// empty / skipped / ignored: GL_OML_subsample
	// empty / skipped / ignored: GL_OVR_multiview
	// empty / skipped / ignored: GL_OVR_multiview2
	// empty / skipped / ignored: GL_OVR_multiview_multisampled_render_to_texture
	// empty / skipped / ignored: GL_PGI_misc_hints
	// empty / skipped / ignored: GL_PGI_vertex_hints
	/* ------------------------------ GL_QCOM_alpha_test ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("ALPHA_TEST_QCOM", GL_ALPHA_TEST_QCOM);
	CREATE_CONSTANT_ACCESSOR("ALPHA_TEST_FUNC_QCOM", GL_ALPHA_TEST_FUNC_QCOM);
	CREATE_CONSTANT_ACCESSOR("ALPHA_TEST_REF_QCOM", GL_ALPHA_TEST_REF_QCOM);

	tpl->Set(String::NewFromUtf8(isolate, "alphaFuncQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("alphaFuncQCOM requires 2 arguments");
			return;
		}

		GLenum func = args[0]->Uint32Value();
		GLclampf ref = GLclampf(args[1]->NumberValue());

		glAlphaFuncQCOM(func, ref);
	}));



	/* ------------------------------ GL_QCOM_binning_control ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("DONT_CARE", GL_DONT_CARE);
	CREATE_CONSTANT_ACCESSOR("BINNING_CONTROL_HINT_QCOM", GL_BINNING_CONTROL_HINT_QCOM);
	CREATE_CONSTANT_ACCESSOR("CPU_OPTIMIZED_QCOM", GL_CPU_OPTIMIZED_QCOM);
	CREATE_CONSTANT_ACCESSOR("GPU_OPTIMIZED_QCOM", GL_GPU_OPTIMIZED_QCOM);
	CREATE_CONSTANT_ACCESSOR("RENDER_DIRECT_TO_FRAMEBUFFER_QCOM", GL_RENDER_DIRECT_TO_FRAMEBUFFER_QCOM);



	/* ------------------------------ GL_QCOM_driver_control ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "disableDriverControlQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("disableDriverControlQCOM requires 1 arguments");
			return;
		}

		GLuint driverControl = args[0]->Uint32Value();

		glDisableDriverControlQCOM(driverControl);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "enableDriverControlQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("enableDriverControlQCOM requires 1 arguments");
			return;
		}

		GLuint driverControl = args[0]->Uint32Value();

		glEnableDriverControlQCOM(driverControl);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getDriverControlStringQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("getDriverControlStringQCOM requires 4 arguments");
			return;
		}

		GLuint driverControl = args[0]->Uint32Value();
		GLsizei bufSize = args[1]->Int32Value();

		GLsizei* length = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLsizei*>(bdata);
		} else {
			cout << "ERROR(glGetDriverControlStringQCOM): array must be of type Int32Array" << endl;
			exit(1);
		}


		GLchar* driverControlString = nullptr;
		if (args[3]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[3]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			driverControlString = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glGetDriverControlStringQCOM): array must be of type Int8Array" << endl;
			exit(1);
		}


		glGetDriverControlStringQCOM(driverControl, bufSize, length, driverControlString);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getDriverControlsQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getDriverControlsQCOM requires 3 arguments");
			return;
		}


		GLint* num = nullptr;
		if (args[0]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[0]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			num = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glGetDriverControlsQCOM): array must be of type Int32Array" << endl;
			exit(1);
		}

		GLsizei size = args[1]->Int32Value();

		GLuint* driverControls = nullptr;
		if (args[2]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[2]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			driverControls = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glGetDriverControlsQCOM): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glGetDriverControlsQCOM(num, size, driverControls);
	}));



	/* ------------------------------ GL_QCOM_extended_get ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("TEXTURE_WIDTH_QCOM", GL_TEXTURE_WIDTH_QCOM);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_HEIGHT_QCOM", GL_TEXTURE_HEIGHT_QCOM);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_DEPTH_QCOM", GL_TEXTURE_DEPTH_QCOM);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_INTERNAL_FORMAT_QCOM", GL_TEXTURE_INTERNAL_FORMAT_QCOM);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_FORMAT_QCOM", GL_TEXTURE_FORMAT_QCOM);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_TYPE_QCOM", GL_TEXTURE_TYPE_QCOM);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_IMAGE_VALID_QCOM", GL_TEXTURE_IMAGE_VALID_QCOM);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_NUM_LEVELS_QCOM", GL_TEXTURE_NUM_LEVELS_QCOM);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_TARGET_QCOM", GL_TEXTURE_TARGET_QCOM);
	CREATE_CONSTANT_ACCESSOR("TEXTURE_OBJECT_VALID_QCOM", GL_TEXTURE_OBJECT_VALID_QCOM);
	CREATE_CONSTANT_ACCESSOR("STATE_RESTORE", GL_STATE_RESTORE);


	tpl->Set(String::NewFromUtf8(isolate, "extGetBuffersQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("extGetBuffersQCOM requires 3 arguments");
			return;
		}


		GLuint* buffers = nullptr;
		if (args[0]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[0]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			buffers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glExtGetBuffersQCOM): array must be of type Uint32Array" << endl;
			exit(1);
		}

		GLint maxBuffers = args[1]->Int32Value();

		GLint* numBuffers = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			numBuffers = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glExtGetBuffersQCOM): array must be of type Int32Array" << endl;
			exit(1);
		}


		glExtGetBuffersQCOM(buffers, maxBuffers, numBuffers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "extGetFramebuffersQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("extGetFramebuffersQCOM requires 3 arguments");
			return;
		}


		GLuint* framebuffers = nullptr;
		if (args[0]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[0]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			framebuffers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glExtGetFramebuffersQCOM): array must be of type Uint32Array" << endl;
			exit(1);
		}

		GLint maxFramebuffers = args[1]->Int32Value();

		GLint* numFramebuffers = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			numFramebuffers = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glExtGetFramebuffersQCOM): array must be of type Int32Array" << endl;
			exit(1);
		}


		glExtGetFramebuffersQCOM(framebuffers, maxFramebuffers, numFramebuffers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "extGetRenderbuffersQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("extGetRenderbuffersQCOM requires 3 arguments");
			return;
		}


		GLuint* renderbuffers = nullptr;
		if (args[0]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[0]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			renderbuffers = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glExtGetRenderbuffersQCOM): array must be of type Uint32Array" << endl;
			exit(1);
		}

		GLint maxRenderbuffers = args[1]->Int32Value();

		GLint* numRenderbuffers = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			numRenderbuffers = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glExtGetRenderbuffersQCOM): array must be of type Int32Array" << endl;
			exit(1);
		}


		glExtGetRenderbuffersQCOM(renderbuffers, maxRenderbuffers, numRenderbuffers);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "extGetTexLevelParameterivQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("extGetTexLevelParameterivQCOM requires 5 arguments");
			return;
		}

		GLuint texture = args[0]->Uint32Value();
		GLenum face = args[1]->Uint32Value();
		GLint level = args[2]->Int32Value();
		GLenum pname = args[3]->Uint32Value();

		GLint* params = nullptr;
		if (args[4]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[4]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glExtGetTexLevelParameterivQCOM): array must be of type Int32Array" << endl;
			exit(1);
		}


		glExtGetTexLevelParameterivQCOM(texture, face, level, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "extGetTexSubImageQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 11) {
			V8Helper::_instance->throwException("extGetTexSubImageQCOM requires 11 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLint level = args[1]->Int32Value();
		GLint xoffset = args[2]->Int32Value();
		GLint yoffset = args[3]->Int32Value();
		GLint zoffset = args[4]->Int32Value();
		GLsizei width = args[5]->Int32Value();
		GLsizei height = args[6]->Int32Value();
		GLsizei depth = args[7]->Int32Value();
		GLenum format = args[8]->Uint32Value();
		GLenum type = args[9]->Uint32Value();

		void* texels = nullptr;
		if (args[10]->IsArrayBuffer()) {
			v8::Local<v8::ArrayBuffer> buffer = (args[10]).As<v8::ArrayBuffer>();
			void *bdata = buffer->GetContents().Data();
			texels = reinterpret_cast<void*>(bdata);
		} else if (args[10]->IsArrayBufferView()) {
			v8::Local<v8::ArrayBufferView> view = (args[10]).As<v8::ArrayBufferView>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			texels = reinterpret_cast<void*>(bdata);
		} else {
			cout << "ERROR(glExtGetTexSubImageQCOM): array must be of type ArrayBuffer" << endl;
			exit(1);
		}


		glExtGetTexSubImageQCOM(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, texels);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "extGetTexturesQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("extGetTexturesQCOM requires 3 arguments");
			return;
		}


		GLuint* textures = nullptr;
		if (args[0]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[0]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			textures = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glExtGetTexturesQCOM): array must be of type Uint32Array" << endl;
			exit(1);
		}

		GLint maxTextures = args[1]->Int32Value();

		GLint* numTextures = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			numTextures = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glExtGetTexturesQCOM): array must be of type Int32Array" << endl;
			exit(1);
		}


		glExtGetTexturesQCOM(textures, maxTextures, numTextures);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "extTexObjectStateOverrideiQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("extTexObjectStateOverrideiQCOM requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLint param = args[2]->Int32Value();

		glExtTexObjectStateOverrideiQCOM(target, pname, param);
	}));



	/* ------------------------------ GL_QCOM_extended_get2 ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "extGetProgramBinarySourceQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("extGetProgramBinarySourceQCOM requires 4 arguments");
			return;
		}

		GLuint program = args[0]->Uint32Value();
		GLenum shadertype = args[1]->Uint32Value();

		GLchar* source = nullptr;
		if (args[2]->IsInt8Array()) {
			v8::Local<v8::Int8Array> view = (args[2]).As<v8::Int8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			source = reinterpret_cast<GLchar*>(bdata);
		} else {
			cout << "ERROR(glExtGetProgramBinarySourceQCOM): array must be of type Int8Array" << endl;
			exit(1);
		}


		GLint* length = nullptr;
		if (args[3]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[3]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			length = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glExtGetProgramBinarySourceQCOM): array must be of type Int32Array" << endl;
			exit(1);
		}


		glExtGetProgramBinarySourceQCOM(program, shadertype, source, length);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "extGetProgramsQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("extGetProgramsQCOM requires 3 arguments");
			return;
		}


		GLuint* programs = nullptr;
		if (args[0]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[0]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			programs = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glExtGetProgramsQCOM): array must be of type Uint32Array" << endl;
			exit(1);
		}

		GLint maxPrograms = args[1]->Int32Value();

		GLint* numPrograms = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			numPrograms = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glExtGetProgramsQCOM): array must be of type Int32Array" << endl;
			exit(1);
		}


		glExtGetProgramsQCOM(programs, maxPrograms, numPrograms);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "extGetShadersQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("extGetShadersQCOM requires 3 arguments");
			return;
		}


		GLuint* shaders = nullptr;
		if (args[0]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[0]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			shaders = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glExtGetShadersQCOM): array must be of type Uint32Array" << endl;
			exit(1);
		}

		GLint maxShaders = args[1]->Int32Value();

		GLint* numShaders = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			numShaders = reinterpret_cast<GLint*>(bdata);
		} else {
			cout << "ERROR(glExtGetShadersQCOM): array must be of type Int32Array" << endl;
			exit(1);
		}


		glExtGetShadersQCOM(shaders, maxShaders, numShaders);
	}));



	/* ------------------------------ GL_QCOM_framebuffer_foveated ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("FOVEATION_ENABLE_BIT_QCOM", GL_FOVEATION_ENABLE_BIT_QCOM);
	CREATE_CONSTANT_ACCESSOR("FOVEATION_SCALED_BIN_METHOD_BIT_QCOM", GL_FOVEATION_SCALED_BIN_METHOD_BIT_QCOM);

	tpl->Set(String::NewFromUtf8(isolate, "framebufferFoveationConfigQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("framebufferFoveationConfigQCOM requires 5 arguments");
			return;
		}

		GLuint fbo = args[0]->Uint32Value();
		GLuint numLayers = args[1]->Uint32Value();
		GLuint focalPointsPerLayer = args[2]->Uint32Value();
		GLuint requestedFeatures = args[3]->Uint32Value();

		GLuint* providedFeatures = nullptr;
		if (args[4]->IsUint32Array()) {
			v8::Local<v8::Uint32Array> view = (args[4]).As<v8::Uint32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			providedFeatures = reinterpret_cast<GLuint*>(bdata);
		} else {
			cout << "ERROR(glFramebufferFoveationConfigQCOM): array must be of type Uint32Array" << endl;
			exit(1);
		}


		glFramebufferFoveationConfigQCOM(fbo, numLayers, focalPointsPerLayer, requestedFeatures, providedFeatures);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "framebufferFoveationParametersQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 8) {
			V8Helper::_instance->throwException("framebufferFoveationParametersQCOM requires 8 arguments");
			return;
		}

		GLuint fbo = args[0]->Uint32Value();
		GLuint layer = args[1]->Uint32Value();
		GLuint focalPoint = args[2]->Uint32Value();
		GLfloat focalX = GLfloat(args[3]->NumberValue());
		GLfloat focalY = GLfloat(args[4]->NumberValue());
		GLfloat gainX = GLfloat(args[5]->NumberValue());
		GLfloat gainY = GLfloat(args[6]->NumberValue());
		GLfloat foveaArea = GLfloat(args[7]->NumberValue());

		glFramebufferFoveationParametersQCOM(fbo, layer, focalPoint, focalX, focalY, gainX, gainY, foveaArea);
	}));



	/* ------------------------------ GL_QCOM_perfmon_global_mode ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("PERFMON_GLOBAL_MODE_QCOM", GL_PERFMON_GLOBAL_MODE_QCOM);



	/* ------------------------------ GL_QCOM_shader_framebuffer_fetch_noncoherent ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("FRAMEBUFFER_FETCH_NONCOHERENT_QCOM", GL_FRAMEBUFFER_FETCH_NONCOHERENT_QCOM);




	/* ------------------------------ GL_QCOM_tiled_rendering ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("COLOR_BUFFER_BIT0_QCOM", GL_COLOR_BUFFER_BIT0_QCOM);
	CREATE_CONSTANT_ACCESSOR("COLOR_BUFFER_BIT1_QCOM", GL_COLOR_BUFFER_BIT1_QCOM);
	CREATE_CONSTANT_ACCESSOR("COLOR_BUFFER_BIT2_QCOM", GL_COLOR_BUFFER_BIT2_QCOM);
	CREATE_CONSTANT_ACCESSOR("COLOR_BUFFER_BIT3_QCOM", GL_COLOR_BUFFER_BIT3_QCOM);
	CREATE_CONSTANT_ACCESSOR("COLOR_BUFFER_BIT4_QCOM", GL_COLOR_BUFFER_BIT4_QCOM);
	CREATE_CONSTANT_ACCESSOR("COLOR_BUFFER_BIT5_QCOM", GL_COLOR_BUFFER_BIT5_QCOM);
	CREATE_CONSTANT_ACCESSOR("COLOR_BUFFER_BIT6_QCOM", GL_COLOR_BUFFER_BIT6_QCOM);
	CREATE_CONSTANT_ACCESSOR("COLOR_BUFFER_BIT7_QCOM", GL_COLOR_BUFFER_BIT7_QCOM);
	CREATE_CONSTANT_ACCESSOR("DEPTH_BUFFER_BIT0_QCOM", GL_DEPTH_BUFFER_BIT0_QCOM);
	CREATE_CONSTANT_ACCESSOR("DEPTH_BUFFER_BIT1_QCOM", GL_DEPTH_BUFFER_BIT1_QCOM);
	CREATE_CONSTANT_ACCESSOR("DEPTH_BUFFER_BIT2_QCOM", GL_DEPTH_BUFFER_BIT2_QCOM);
	CREATE_CONSTANT_ACCESSOR("DEPTH_BUFFER_BIT3_QCOM", GL_DEPTH_BUFFER_BIT3_QCOM);
	CREATE_CONSTANT_ACCESSOR("DEPTH_BUFFER_BIT4_QCOM", GL_DEPTH_BUFFER_BIT4_QCOM);
	CREATE_CONSTANT_ACCESSOR("DEPTH_BUFFER_BIT5_QCOM", GL_DEPTH_BUFFER_BIT5_QCOM);
	CREATE_CONSTANT_ACCESSOR("DEPTH_BUFFER_BIT6_QCOM", GL_DEPTH_BUFFER_BIT6_QCOM);
	CREATE_CONSTANT_ACCESSOR("DEPTH_BUFFER_BIT7_QCOM", GL_DEPTH_BUFFER_BIT7_QCOM);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BUFFER_BIT0_QCOM", GL_STENCIL_BUFFER_BIT0_QCOM);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BUFFER_BIT1_QCOM", GL_STENCIL_BUFFER_BIT1_QCOM);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BUFFER_BIT2_QCOM", GL_STENCIL_BUFFER_BIT2_QCOM);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BUFFER_BIT3_QCOM", GL_STENCIL_BUFFER_BIT3_QCOM);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BUFFER_BIT4_QCOM", GL_STENCIL_BUFFER_BIT4_QCOM);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BUFFER_BIT5_QCOM", GL_STENCIL_BUFFER_BIT5_QCOM);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BUFFER_BIT6_QCOM", GL_STENCIL_BUFFER_BIT6_QCOM);
	CREATE_CONSTANT_ACCESSOR("STENCIL_BUFFER_BIT7_QCOM", GL_STENCIL_BUFFER_BIT7_QCOM);
	CREATE_CONSTANT_ACCESSOR("MULTISAMPLE_BUFFER_BIT0_QCOM", GL_MULTISAMPLE_BUFFER_BIT0_QCOM);
	CREATE_CONSTANT_ACCESSOR("MULTISAMPLE_BUFFER_BIT1_QCOM", GL_MULTISAMPLE_BUFFER_BIT1_QCOM);
	CREATE_CONSTANT_ACCESSOR("MULTISAMPLE_BUFFER_BIT2_QCOM", GL_MULTISAMPLE_BUFFER_BIT2_QCOM);
	CREATE_CONSTANT_ACCESSOR("MULTISAMPLE_BUFFER_BIT3_QCOM", GL_MULTISAMPLE_BUFFER_BIT3_QCOM);
	CREATE_CONSTANT_ACCESSOR("MULTISAMPLE_BUFFER_BIT4_QCOM", GL_MULTISAMPLE_BUFFER_BIT4_QCOM);
	CREATE_CONSTANT_ACCESSOR("MULTISAMPLE_BUFFER_BIT5_QCOM", GL_MULTISAMPLE_BUFFER_BIT5_QCOM);
	CREATE_CONSTANT_ACCESSOR("MULTISAMPLE_BUFFER_BIT6_QCOM", GL_MULTISAMPLE_BUFFER_BIT6_QCOM);
	CREATE_CONSTANT_ACCESSOR("MULTISAMPLE_BUFFER_BIT7_QCOM", GL_MULTISAMPLE_BUFFER_BIT7_QCOM);

	tpl->Set(String::NewFromUtf8(isolate, "endTilingQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("endTilingQCOM requires 1 arguments");
			return;
		}

		GLbitfield preserveMask = args[0]->Uint32Value();

		glEndTilingQCOM(preserveMask);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "startTilingQCOM"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("startTilingQCOM requires 5 arguments");
			return;
		}

		GLuint x = args[0]->Uint32Value();
		GLuint y = args[1]->Uint32Value();
		GLuint width = args[2]->Uint32Value();
		GLuint height = args[3]->Uint32Value();
		GLbitfield preserveMask = args[4]->Uint32Value();

		glStartTilingQCOM(x, y, width, height, preserveMask);
	}));



	/* ------------------------------ GL_QCOM_writeonly_rendering ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("WRITEONLY_RENDERING_QCOM", GL_WRITEONLY_RENDERING_QCOM);



	/* ------------------------------ GL_REGAL_enable ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("ERROR_REGAL", GL_ERROR_REGAL);
	CREATE_CONSTANT_ACCESSOR("DEBUG_REGAL", GL_DEBUG_REGAL);
	CREATE_CONSTANT_ACCESSOR("LOG_REGAL", GL_LOG_REGAL);
	CREATE_CONSTANT_ACCESSOR("EMULATION_REGAL", GL_EMULATION_REGAL);
	CREATE_CONSTANT_ACCESSOR("DRIVER_REGAL", GL_DRIVER_REGAL);
	CREATE_CONSTANT_ACCESSOR("MISSING_REGAL", GL_MISSING_REGAL);
	CREATE_CONSTANT_ACCESSOR("TRACE_REGAL", GL_TRACE_REGAL);
	CREATE_CONSTANT_ACCESSOR("CACHE_REGAL", GL_CACHE_REGAL);
	CREATE_CONSTANT_ACCESSOR("CODE_REGAL", GL_CODE_REGAL);
	CREATE_CONSTANT_ACCESSOR("STATISTICS_REGAL", GL_STATISTICS_REGAL);



	// empty / skipped / ignored: GL_REGAL_error_string
	/* ------------------------------ GL_REGAL_ES1_0_compatibility ------------------------------ */





	tpl->Set(String::NewFromUtf8(isolate, "color4x"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("color4x requires 4 arguments");
			return;
		}

		GLfixed red = args[0]->Int32Value();
		GLfixed green = args[1]->Int32Value();
		GLfixed blue = args[2]->Int32Value();
		GLfixed alpha = args[3]->Int32Value();

		glColor4x(red, green, blue, alpha);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "fogx"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("fogx requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();
		GLfixed param = args[1]->Int32Value();

		glFogx(pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "fogxv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("fogxv requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();

		GLfixed* params = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glFogxv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glFogxv(pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "frustumf"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("frustumf requires 6 arguments");
			return;
		}

		GLfloat left = GLfloat(args[0]->NumberValue());
		GLfloat right = GLfloat(args[1]->NumberValue());
		GLfloat bottom = GLfloat(args[2]->NumberValue());
		GLfloat top = GLfloat(args[3]->NumberValue());
		GLfloat zNear = GLfloat(args[4]->NumberValue());
		GLfloat zFar = GLfloat(args[5]->NumberValue());

		glFrustumf(left, right, bottom, top, zNear, zFar);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "frustumx"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("frustumx requires 6 arguments");
			return;
		}

		GLfixed left = args[0]->Int32Value();
		GLfixed right = args[1]->Int32Value();
		GLfixed bottom = args[2]->Int32Value();
		GLfixed top = args[3]->Int32Value();
		GLfixed zNear = args[4]->Int32Value();
		GLfixed zFar = args[5]->Int32Value();

		glFrustumx(left, right, bottom, top, zNear, zFar);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "lightModelx"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("lightModelx requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();
		GLfixed param = args[1]->Int32Value();

		glLightModelx(pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "lightModelxv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("lightModelxv requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();

		GLfixed* params = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glLightModelxv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glLightModelxv(pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "lightx"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("lightx requires 3 arguments");
			return;
		}

		GLenum light = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLfixed param = args[2]->Int32Value();

		glLightx(light, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "lightxv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("lightxv requires 3 arguments");
			return;
		}

		GLenum light = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfixed* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glLightxv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glLightxv(light, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "lineWidthx"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("lineWidthx requires 1 arguments");
			return;
		}

		GLfixed width = args[0]->Int32Value();

		glLineWidthx(width);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "loadMatrixx"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("loadMatrixx requires 1 arguments");
			return;
		}


		GLfixed* m = nullptr;
		if (args[0]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[0]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			m = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glLoadMatrixx): array must be of type Int32Array" << endl;
			exit(1);
		}


		glLoadMatrixx(m);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "materialx"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("materialx requires 3 arguments");
			return;
		}

		GLenum face = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLfixed param = args[2]->Int32Value();

		glMaterialx(face, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "materialxv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("materialxv requires 3 arguments");
			return;
		}

		GLenum face = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfixed* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glMaterialxv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glMaterialxv(face, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multMatrixx"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("multMatrixx requires 1 arguments");
			return;
		}


		GLfixed* m = nullptr;
		if (args[0]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[0]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			m = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glMultMatrixx): array must be of type Int32Array" << endl;
			exit(1);
		}


		glMultMatrixx(m);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "multiTexCoord4x"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 5) {
			V8Helper::_instance->throwException("multiTexCoord4x requires 5 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLfixed s = args[1]->Int32Value();
		GLfixed t = args[2]->Int32Value();
		GLfixed r = args[3]->Int32Value();
		GLfixed q = args[4]->Int32Value();

		glMultiTexCoord4x(target, s, t, r, q);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "normal3x"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("normal3x requires 3 arguments");
			return;
		}

		GLfixed nx = args[0]->Int32Value();
		GLfixed ny = args[1]->Int32Value();
		GLfixed nz = args[2]->Int32Value();

		glNormal3x(nx, ny, nz);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "orthof"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("orthof requires 6 arguments");
			return;
		}

		GLfloat left = GLfloat(args[0]->NumberValue());
		GLfloat right = GLfloat(args[1]->NumberValue());
		GLfloat bottom = GLfloat(args[2]->NumberValue());
		GLfloat top = GLfloat(args[3]->NumberValue());
		GLfloat zNear = GLfloat(args[4]->NumberValue());
		GLfloat zFar = GLfloat(args[5]->NumberValue());

		glOrthof(left, right, bottom, top, zNear, zFar);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "orthox"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 6) {
			V8Helper::_instance->throwException("orthox requires 6 arguments");
			return;
		}

		GLfixed left = args[0]->Int32Value();
		GLfixed right = args[1]->Int32Value();
		GLfixed bottom = args[2]->Int32Value();
		GLfixed top = args[3]->Int32Value();
		GLfixed zNear = args[4]->Int32Value();
		GLfixed zFar = args[5]->Int32Value();

		glOrthox(left, right, bottom, top, zNear, zFar);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "pointSizex"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("pointSizex requires 1 arguments");
			return;
		}

		GLfixed size = args[0]->Int32Value();

		glPointSizex(size);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "polygonOffsetx"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("polygonOffsetx requires 2 arguments");
			return;
		}

		GLfixed factor = args[0]->Int32Value();
		GLfixed units = args[1]->Int32Value();

		glPolygonOffsetx(factor, units);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "rotatex"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 4) {
			V8Helper::_instance->throwException("rotatex requires 4 arguments");
			return;
		}

		GLfixed angle = args[0]->Int32Value();
		GLfixed x = args[1]->Int32Value();
		GLfixed y = args[2]->Int32Value();
		GLfixed z = args[3]->Int32Value();

		glRotatex(angle, x, y, z);
	}));


	tpl->Set(String::NewFromUtf8(isolate, "scalex"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("scalex requires 3 arguments");
			return;
		}

		GLfixed x = args[0]->Int32Value();
		GLfixed y = args[1]->Int32Value();
		GLfixed z = args[2]->Int32Value();

		glScalex(x, y, z);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texEnvx"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("texEnvx requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLfixed param = args[2]->Int32Value();

		glTexEnvx(target, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texEnvxv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("texEnvxv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfixed* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glTexEnvxv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glTexEnvxv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texParameterx"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("texParameterx requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();
		GLfixed param = args[2]->Int32Value();

		glTexParameterx(target, pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "translatex"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("translatex requires 3 arguments");
			return;
		}

		GLfixed x = args[0]->Int32Value();
		GLfixed y = args[1]->Int32Value();
		GLfixed z = args[2]->Int32Value();

		glTranslatex(x, y, z);
	}));



	/* ------------------------------ GL_REGAL_ES1_1_compatibility ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "clipPlanef"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("clipPlanef requires 2 arguments");
			return;
		}

		GLenum plane = args[0]->Uint32Value();

		GLfloat* equation = nullptr;
		if (args[1]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[1]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			equation = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glClipPlanef): array must be of type Float32Array" << endl;
			exit(1);
		}


		glClipPlanef(plane, equation);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "clipPlanex"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("clipPlanex requires 2 arguments");
			return;
		}

		GLenum plane = args[0]->Uint32Value();

		GLfixed* equation = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			equation = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glClipPlanex): array must be of type Int32Array" << endl;
			exit(1);
		}


		glClipPlanex(plane, equation);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getClipPlanef"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("getClipPlanef requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();

		GLfloat* eqn = nullptr;
		if (args[1]->IsFloat32Array()) {
			v8::Local<v8::Float32Array> view = (args[1]).As<v8::Float32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			eqn = reinterpret_cast<GLfloat*>(bdata);
		} else {
			cout << "ERROR(glGetClipPlanef): array must be of type Float32Array" << endl;
			exit(1);
		}


		glGetClipPlanef(pname, eqn);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getClipPlanex"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("getClipPlanex requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();

		GLfixed* eqn = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			eqn = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glGetClipPlanex): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetClipPlanex(pname, eqn);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getFixedv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("getFixedv requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();

		GLfixed* params = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glGetFixedv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetFixedv(pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getLightxv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getLightxv requires 3 arguments");
			return;
		}

		GLenum light = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfixed* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glGetLightxv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetLightxv(light, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getMaterialxv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getMaterialxv requires 3 arguments");
			return;
		}

		GLenum face = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfixed* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glGetMaterialxv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetMaterialxv(face, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getTexEnvxv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getTexEnvxv requires 3 arguments");
			return;
		}

		GLenum env = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfixed* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glGetTexEnvxv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetTexEnvxv(env, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "getTexParameterxv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("getTexParameterxv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfixed* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glGetTexParameterxv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glGetTexParameterxv(target, pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "pointParameterx"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("pointParameterx requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();
		GLfixed param = args[1]->Int32Value();

		glPointParameterx(pname, param);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "pointParameterxv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("pointParameterxv requires 2 arguments");
			return;
		}

		GLenum pname = args[0]->Uint32Value();

		GLfixed* params = nullptr;
		if (args[1]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[1]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glPointParameterxv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glPointParameterxv(pname, params);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "texParameterxv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 3) {
			V8Helper::_instance->throwException("texParameterxv requires 3 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();
		GLenum pname = args[1]->Uint32Value();

		GLfixed* params = nullptr;
		if (args[2]->IsInt32Array()) {
			v8::Local<v8::Int32Array> view = (args[2]).As<v8::Int32Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			params = reinterpret_cast<GLfixed*>(bdata);
		} else {
			cout << "ERROR(glTexParameterxv): array must be of type Int32Array" << endl;
			exit(1);
		}


		glTexParameterxv(target, pname, params);
	}));



	// empty / skipped / ignored: GL_REGAL_extension_query
	/* ------------------------------ GL_REGAL_log ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("LOG_ERROR_REGAL", GL_LOG_ERROR_REGAL);
	CREATE_CONSTANT_ACCESSOR("LOG_WARNING_REGAL", GL_LOG_WARNING_REGAL);
	CREATE_CONSTANT_ACCESSOR("LOG_INFO_REGAL", GL_LOG_INFO_REGAL);
	CREATE_CONSTANT_ACCESSOR("LOG_APP_REGAL", GL_LOG_APP_REGAL);
	CREATE_CONSTANT_ACCESSOR("LOG_DRIVER_REGAL", GL_LOG_DRIVER_REGAL);
	CREATE_CONSTANT_ACCESSOR("LOG_INTERNAL_REGAL", GL_LOG_INTERNAL_REGAL);
	CREATE_CONSTANT_ACCESSOR("LOG_DEBUG_REGAL", GL_LOG_DEBUG_REGAL);
	CREATE_CONSTANT_ACCESSOR("LOG_STATUS_REGAL", GL_LOG_STATUS_REGAL);
	CREATE_CONSTANT_ACCESSOR("LOG_HTTP_REGAL", GL_LOG_HTTP_REGAL);




	// empty / skipped / ignored: GL_REGAL_proc_address
	// empty / skipped / ignored: GL_REND_screen_coordinates
	// empty / skipped / ignored: GL_S3_s3tc
	// empty / skipped / ignored: GL_SGIS_clip_band_hint
	// empty / skipped / ignored: GL_SGIS_color_range
	// empty / skipped / ignored: GL_SGIS_detail_texture
	// empty / skipped / ignored: GL_SGIS_fog_function
	// empty / skipped / ignored: GL_SGIS_generate_mipmap
	// empty / skipped / ignored: GL_SGIS_line_texgen
	// empty / skipped / ignored: GL_SGIS_multisample
	// empty / skipped / ignored: GL_SGIS_multitexture
	// empty / skipped / ignored: GL_SGIS_pixel_texture
	// empty / skipped / ignored: GL_SGIS_point_line_texgen
	// empty / skipped / ignored: GL_SGIS_shared_multisample
	// empty / skipped / ignored: GL_SGIS_sharpen_texture
	// empty / skipped / ignored: GL_SGIS_texture4D
	// empty / skipped / ignored: GL_SGIS_texture_border_clamp
	// empty / skipped / ignored: GL_SGIS_texture_edge_clamp
	// empty / skipped / ignored: GL_SGIS_texture_filter4
	// empty / skipped / ignored: GL_SGIS_texture_lod
	// empty / skipped / ignored: GL_SGIS_texture_select
	// empty / skipped / ignored: GL_SGIX_async
	// empty / skipped / ignored: GL_SGIX_async_histogram
	// empty / skipped / ignored: GL_SGIX_async_pixel
	/* ------------------------------ GL_SGIX_bali_g_instruments ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("BALI_NUM_TRIS_CULLED_INSTRUMENT", GL_BALI_NUM_TRIS_CULLED_INSTRUMENT);
	CREATE_CONSTANT_ACCESSOR("BALI_NUM_PRIMS_CLIPPED_INSTRUMENT", GL_BALI_NUM_PRIMS_CLIPPED_INSTRUMENT);
	CREATE_CONSTANT_ACCESSOR("BALI_NUM_PRIMS_REJECT_INSTRUMENT", GL_BALI_NUM_PRIMS_REJECT_INSTRUMENT);
	CREATE_CONSTANT_ACCESSOR("BALI_NUM_PRIMS_CLIP_RESULT_INSTRUMENT", GL_BALI_NUM_PRIMS_CLIP_RESULT_INSTRUMENT);



	/* ------------------------------ GL_SGIX_bali_r_instruments ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("BALI_FRAGMENTS_GENERATED_INSTRUMENT", GL_BALI_FRAGMENTS_GENERATED_INSTRUMENT);
	CREATE_CONSTANT_ACCESSOR("BALI_DEPTH_PASS_INSTRUMENT", GL_BALI_DEPTH_PASS_INSTRUMENT);
	CREATE_CONSTANT_ACCESSOR("BALI_R_CHIP_COUNT", GL_BALI_R_CHIP_COUNT);



	// empty / skipped / ignored: GL_SGIX_bali_timer_instruments
	// empty / skipped / ignored: GL_SGIX_blend_alpha_minmax
	// empty / skipped / ignored: GL_SGIX_blend_cadd
	// empty / skipped / ignored: GL_SGIX_blend_cmultiply
	// empty / skipped / ignored: GL_SGIX_calligraphic_fragment
	// empty / skipped / ignored: GL_SGIX_clipmap
	/* ------------------------------ GL_SGIX_color_matrix_accuracy ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("COLOR_MATRIX_HINT", GL_COLOR_MATRIX_HINT);



	// empty / skipped / ignored: GL_SGIX_color_table_index_mode
	// empty / skipped / ignored: GL_SGIX_complex_polar
	// empty / skipped / ignored: GL_SGIX_convolution_accuracy
	// empty / skipped / ignored: GL_SGIX_cube_map
	// empty / skipped / ignored: GL_SGIX_cylinder_texgen
	/* ------------------------------ GL_SGIX_datapipe ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("GEOMETRY_BIT", GL_GEOMETRY_BIT);
	CREATE_CONSTANT_ACCESSOR("IMAGE_BIT", GL_IMAGE_BIT);

	tpl->Set(String::NewFromUtf8(isolate, "addressSpace"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("addressSpace requires 2 arguments");
			return;
		}

		GLenum space = args[0]->Uint32Value();
		GLbitfield mask = args[1]->Uint32Value();

		glAddressSpace(space, mask);
	}));



	// empty / skipped / ignored: GL_SGIX_decimation
	// empty / skipped / ignored: GL_SGIX_depth_pass_instrument
	// empty / skipped / ignored: GL_SGIX_depth_texture
	// empty / skipped / ignored: GL_SGIX_dvc
	// empty / skipped / ignored: GL_SGIX_flush_raster
	// empty / skipped / ignored: GL_SGIX_fog_blend
	// empty / skipped / ignored: GL_SGIX_fog_factor_to_alpha
	// empty / skipped / ignored: GL_SGIX_fog_layers
	// empty / skipped / ignored: GL_SGIX_fog_offset
	// empty / skipped / ignored: GL_SGIX_fog_patchy
	// empty / skipped / ignored: GL_SGIX_fog_scale
	// empty / skipped / ignored: GL_SGIX_fog_texture
	// empty / skipped / ignored: GL_SGIX_fragments_instrument
	// empty / skipped / ignored: GL_SGIX_fragment_lighting_space
	// empty / skipped / ignored: GL_SGIX_fragment_specular_lighting
	// empty / skipped / ignored: GL_SGIX_framezoom
	// empty / skipped / ignored: GL_SGIX_icc_texture
	/* ------------------------------ GL_SGIX_igloo_interface ------------------------------ */

	CREATE_CONSTANT_ACCESSOR("LIGHT31", GL_LIGHT31);



	// empty / skipped / ignored: GL_SGIX_image_compression
	// empty / skipped / ignored: GL_SGIX_impact_pixel_texture
	// empty / skipped / ignored: GL_SGIX_instrument_error
	// empty / skipped / ignored: GL_SGIX_interlace
	// empty / skipped / ignored: GL_SGIX_ir_instrument1
	// empty / skipped / ignored: GL_SGIX_line_quality_hint
	// empty / skipped / ignored: GL_SGIX_list_priority
	/* ------------------------------ GL_SGIX_mpeg1 ------------------------------ */


	tpl->Set(String::NewFromUtf8(isolate, "getMPEGQuantTableubv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("getMPEGQuantTableubv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLubyte* values = nullptr;
		if (args[1]->IsUint8Array()) {
			v8::Local<v8::Uint8Array> view = (args[1]).As<v8::Uint8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			values = reinterpret_cast<GLubyte*>(bdata);
		} else {
			cout << "ERROR(glGetMPEGQuantTableubv): array must be of type Uint8Array" << endl;
			exit(1);
		}


		glGetMPEGQuantTableubv(target, values);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "mPEGQuantTableubv"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 2) {
			V8Helper::_instance->throwException("mPEGQuantTableubv requires 2 arguments");
			return;
		}

		GLenum target = args[0]->Uint32Value();

		GLubyte* values = nullptr;
		if (args[1]->IsUint8Array()) {
			v8::Local<v8::Uint8Array> view = (args[1]).As<v8::Uint8Array>();
			auto buffer = view->Buffer();
			void *bdata = view->Buffer()->GetContents().Data();
			values = reinterpret_cast<GLubyte*>(bdata);
		} else {
			cout << "ERROR(glMPEGQuantTableubv): array must be of type Uint8Array" << endl;
			exit(1);
		}


		glMPEGQuantTableubv(target, values);
	}));



	// empty / skipped / ignored: GL_SGIX_mpeg2
	// empty / skipped / ignored: GL_SGIX_nonlinear_lighting_pervertex
	// empty / skipped / ignored: GL_SGIX_nurbs_eval
	// empty / skipped / ignored: GL_SGIX_occlusion_instrument
	// empty / skipped / ignored: GL_SGIX_packed_6bytes
	// empty / skipped / ignored: GL_SGIX_pixel_texture
	// empty / skipped / ignored: GL_SGIX_pixel_texture_bits
	// empty / skipped / ignored: GL_SGIX_pixel_texture_lod
	// empty / skipped / ignored: GL_SGIX_pixel_tiles
	// empty / skipped / ignored: GL_SGIX_polynomial_ffd
	// empty / skipped / ignored: GL_SGIX_quad_mesh
	// empty / skipped / ignored: GL_SGIX_reference_plane
	// empty / skipped / ignored: GL_SGIX_resample
	// empty / skipped / ignored: GL_SGIX_scalebias_hint
	// empty / skipped / ignored: GL_SGIX_shadow
	// empty / skipped / ignored: GL_SGIX_shadow_ambient
	// empty / skipped / ignored: GL_SGIX_slim
	// empty / skipped / ignored: GL_SGIX_spotlight_cutoff
	// empty / skipped / ignored: GL_SGIX_sprite
	// empty / skipped / ignored: GL_SGIX_subdiv_patch
	// empty / skipped / ignored: GL_SGIX_subsample
	// empty / skipped / ignored: GL_SGIX_tag_sample_buffer
	// empty / skipped / ignored: GL_SGIX_texture_add_env
	// empty / skipped / ignored: GL_SGIX_texture_coordinate_clamp
	// empty / skipped / ignored: GL_SGIX_texture_lod_bias
	// empty / skipped / ignored: GL_SGIX_texture_mipmap_anisotropic
	// empty / skipped / ignored: GL_SGIX_texture_multi_buffer
	// empty / skipped / ignored: GL_SGIX_texture_phase
	// empty / skipped / ignored: GL_SGIX_texture_range
	// empty / skipped / ignored: GL_SGIX_texture_scale_bias
	// empty / skipped / ignored: GL_SGIX_texture_supersample
	// empty / skipped / ignored: GL_SGIX_vector_ops
	// empty / skipped / ignored: GL_SGIX_vertex_array_object
	// empty / skipped / ignored: GL_SGIX_vertex_preclip
	// empty / skipped / ignored: GL_SGIX_vertex_preclip_hint
	// empty / skipped / ignored: GL_SGIX_ycrcb
	// empty / skipped / ignored: GL_SGIX_ycrcba
	// empty / skipped / ignored: GL_SGIX_ycrcb_subsample
	// empty / skipped / ignored: GL_SGI_color_matrix
	// empty / skipped / ignored: GL_SGI_color_table
	// empty / skipped / ignored: GL_SGI_complex
	// empty / skipped / ignored: GL_SGI_complex_type
	// empty / skipped / ignored: GL_SGI_fft
	// empty / skipped / ignored: GL_SGI_texture_color_table
	// empty / skipped / ignored: GL_SUNX_constant_data
	// empty / skipped / ignored: GL_SUN_convolution_border_modes
	// empty / skipped / ignored: GL_SUN_global_alpha
	// empty / skipped / ignored: GL_SUN_mesh_array
	// empty / skipped / ignored: GL_SUN_read_video_pixels
	// empty / skipped / ignored: GL_SUN_slice_accum
	// empty / skipped / ignored: GL_SUN_triangle_list
	// empty / skipped / ignored: GL_SUN_vertex
	// empty / skipped / ignored: GL_WIN_phong_shading
	// empty / skipped / ignored: GL_WIN_scene_markerXXX
	// empty / skipped / ignored: GL_WIN_specular_fog
	// empty / skipped / ignored: GL_WIN_swap_hint




}