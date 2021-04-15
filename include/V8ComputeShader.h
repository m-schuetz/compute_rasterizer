
#pragma once

#include "libplatform/libplatform.h"
#include "v8.h"
#include "V8Helper.h"
#include "ComputeShader.h"


Local<ObjectTemplate> createV8ComputeShaderTemplate(v8::Isolate *isolate);

Local<Object> v8Object(ComputeShader *shader);