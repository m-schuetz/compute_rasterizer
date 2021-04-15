
#include "V8ComputeShader.h"

typedef v8::Persistent<Object, v8::CopyablePersistentTraits<v8::Object>> PersistentObject;
unordered_map<ComputeShader*, PersistentObject> computeShaderUniformHandles;

Local<ObjectTemplate> createV8ComputeShaderTemplate(v8::Isolate *isolate) {

	Local<ObjectTemplate> tpl = ObjectTemplate::New(isolate);
	tpl->SetInternalFieldCount(1);

	tpl->SetAccessor(String::NewFromUtf8(isolate, "program"), [](Local<String> property, const PropertyCallbackInfo<Value>& info) {

		Local<Object> self = info.Holder();
		Local<External> wrap = Local<External>::Cast(self->GetInternalField(0));
		void* ptr = wrap->Value();
		ComputeShader *shader = static_cast<ComputeShader*>(ptr);

		auto value = shader->program;
		info.GetReturnValue().Set(value);

	});

	tpl->SetAccessor(String::NewFromUtf8(isolate, "uniforms"), [](Local<String> property, const PropertyCallbackInfo<Value>& info) {

		v8::Isolate *isolate = Isolate::GetCurrent();

		EscapableHandleScope handle_scope(isolate);

		Local<Object> self = info.Holder();
		Local<External> wrap = Local<External>::Cast(self->GetInternalField(0));
		void* ptr = wrap->Value();
		ComputeShader *shader = static_cast<ComputeShader*>(ptr);

		if (computeShaderUniformHandles.find(shader) != computeShaderUniformHandles.end()) {
			// use existing

			//Local<Object> obj = shaderUniformHandles[shader];
			//Local<Object> obj = Local<Object>::New(isolate, shaderUniformHandles[shader]);
			PersistentObject pobj = computeShaderUniformHandles[shader];
			Local<Object> obj = Local<Object>::New(isolate, pobj);

			info.GetReturnValue().Set(obj);

		} else {
			// create new

			Local<ObjectTemplate> tpl = ObjectTemplate::New(isolate);
			tpl->SetInternalFieldCount(1);

			tpl->SetNamedPropertyHandler([](Local<String> property, const PropertyCallbackInfo<Value>& info) {

				HandleScope handle_scope(Isolate::GetCurrent());

				Local<Object> self = info.Holder();
				Local<External> wrap = Local<External>::Cast(self->GetInternalField(0));
				void* ptr = wrap->Value();
				ComputeShader *shader = static_cast<ComputeShader*>(ptr);

				String::Utf8Value str(property);
				string uniformName = *str;

				int uniformLocation = shader->uniformLocations[uniformName];

				//cout << "property: " << uniformName << " -> " << uniformLocation << endl;

				info.GetReturnValue().Set(uniformLocation);
			});

			Local<Object> obj = tpl->NewInstance();
			obj->SetInternalField(0, External::New(isolate, shader));

			computeShaderUniformHandles[shader] = PersistentObject(isolate, obj);

			//auto value = shader->program;
			info.GetReturnValue().Set(obj);
		}


	});

	return tpl;
}

Local<Object> v8Object(ComputeShader *shader) {
	auto isolate = V8Helper::instance()->isolate;
	//auto tpl = getVector3Template(isolate);
	auto tpl = createV8ComputeShaderTemplate(isolate);
	Local<Object> obj = tpl->NewInstance();
	obj->SetInternalField(0, External::New(isolate, shader));

	return obj;
}