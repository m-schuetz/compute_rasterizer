

#include "V8Shader.h"


typedef v8::Persistent<Object, v8::CopyablePersistentTraits<v8::Object>> PersistentObject;
unordered_map<Shader*, PersistentObject> shaderUniformHandles;

Local<ObjectTemplate> createV8ShaderTemplate(v8::Isolate *isolate) {

	Local<ObjectTemplate> tpl = ObjectTemplate::New(isolate);
	tpl->SetInternalFieldCount(1);

	tpl->SetAccessor(String::NewFromUtf8(isolate, "program"), [](Local<String> property, const PropertyCallbackInfo<Value>& info) {

		Local<Object> self = info.Holder();
		Local<External> wrap = Local<External>::Cast(self->GetInternalField(0));
		void* ptr = wrap->Value();
		Shader *shader = static_cast<Shader*>(ptr);

		auto value = shader->program;
		info.GetReturnValue().Set(value);

	});

	tpl->SetAccessor(String::NewFromUtf8(isolate, "uniforms"), [](Local<String> property, const PropertyCallbackInfo<Value>& info) {

		v8::Isolate *isolate = Isolate::GetCurrent();

		EscapableHandleScope handle_scope(isolate);

		Local<Object> self = info.Holder();
		Local<External> wrap = Local<External>::Cast(self->GetInternalField(0));
		void* ptr = wrap->Value();
		Shader *shader = static_cast<Shader*>(ptr);

		if (shaderUniformHandles.find(shader) != shaderUniformHandles.end()) {
			// use existing

			//Local<Object> obj = shaderUniformHandles[shader];
			//Local<Object> obj = Local<Object>::New(isolate, shaderUniformHandles[shader]);
			PersistentObject pobj = shaderUniformHandles[shader];
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
				Shader *shader = static_cast<Shader*>(ptr);

				String::Utf8Value str(property);
				string uniformName = *str;

				int uniformLocation = shader->uniformLocations[uniformName];

				//cout << "property: " << uniformName << " -> " << uniformLocation << endl;

				info.GetReturnValue().Set(uniformLocation);
			});

			Local<Object> obj = tpl->NewInstance();
			obj->SetInternalField(0, External::New(isolate, shader));

			shaderUniformHandles[shader] = PersistentObject(isolate, obj);

			//auto value = shader->program;
			info.GetReturnValue().Set(obj);
		}


	});

	return tpl;
}

Local<Object> v8Object(Shader *shader) {
	auto isolate = V8Helper::instance()->isolate;
	//auto tpl = getVector3Template(isolate);
	auto tpl = createV8ShaderTemplate(isolate);
	Local<Object> obj = tpl->NewInstance();
	obj->SetInternalField(0, External::New(isolate, shader));

	return obj;
}