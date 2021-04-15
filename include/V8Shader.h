
#pragma once

#include "libplatform/libplatform.h"
#include "v8.h"
#include "V8Helper.h"
#include "Shader.h"




Local<ObjectTemplate> createV8ShaderTemplate(v8::Isolate *isolate);

Local<Object> v8Object(Shader *shader);