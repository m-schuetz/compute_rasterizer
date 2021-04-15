
#include "V8File.h"
#include "utils.h"

#include <thread>

using std::thread;

typedef Persistent<Promise::Resolver, CopyablePersistentTraits<Promise::Resolver>> PersistentResolver;
static unordered_map<long long, PersistentResolver> resolvers;
static long long resolverID = 0;

Local<ObjectTemplate> createV8FileTemplate(v8::Isolate *isolate) {

	Local<ObjectTemplate> tpl = ObjectTemplate::New(isolate);
	tpl->SetInternalFieldCount(1);

	tpl->Set(String::NewFromUtf8(isolate, "readBytes"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("readBytes requires 1 arguments");
			return;
		}

		auto isolate = Isolate::GetCurrent();

		Local<Promise::Resolver> resolver = v8::Promise::Resolver::New(isolate);

		long long currentID = resolverID++;
		resolvers[currentID] = PersistentResolver(isolate, resolver);

		int numBytes = args[0]->Int32Value();

		Local<Object> self = args.Holder();
		Local<External> wrap = Local<External>::Cast(self->GetInternalField(0));
		void* ptr = wrap->Value();
		File *file = static_cast<File*>(ptr);

		thread t([numBytes, file, isolate, currentID]() {

			vector<char> buffer = file->readBytes(numBytes);

			schedule([currentID, isolate, buffer/*, startThread*/]() {

				Local<ArrayBuffer> v8Buffer = v8::ArrayBuffer::New(Isolate::GetCurrent(), buffer.size());

				auto v8Data = v8Buffer->GetContents().Data();

				// TODO: load file content directly to v8 buffer to avoid 2x allocation and a copy
				memcpy(v8Data, buffer.data(), buffer.size());

				auto persistantResolver = resolvers[currentID];
				Local<Promise::Resolver> resolver = Local<Promise::Resolver>::New(isolate, persistantResolver);

				resolver->Resolve(v8Buffer);

				resolvers.erase(currentID);
			});

		});
		t.detach();

		args.GetReturnValue().Set(resolver->GetPromise());
		
	}));

	tpl->Set(String::NewFromUtf8(isolate, "setReadLocation"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 1) {
			V8Helper::_instance->throwException("setReadLocation requires 1 arguments");
			return;
		}

		Local<Object> self = args.Holder();
		Local<External> wrap = Local<External>::Cast(self->GetInternalField(0));
		void* ptr = wrap->Value();
		File *file = static_cast<File*>(ptr);

		int location = args[0]->IntegerValue();

		file->setReadLocation(location);
	}));

	tpl->Set(String::NewFromUtf8(isolate, "close"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("close requires 0 arguments");
			return;
		}

		Local<Object> self = args.Holder();
		Local<External> wrap = Local<External>::Cast(self->GetInternalField(0));
		void* ptr = wrap->Value();
		File *file = static_cast<File*>(ptr);

		file->close();
	}));

	tpl->Set(String::NewFromUtf8(isolate, "fileSize"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
		if (args.Length() != 0) {
			V8Helper::_instance->throwException("fileSize requires 0 arguments");
			return;
		}

		Local<Object> self = args.Holder();
		Local<External> wrap = Local<External>::Cast(self->GetInternalField(0));
		void* ptr = wrap->Value();
		File *file = static_cast<File*>(ptr);

		int location = args[0]->IntegerValue();

		double size = double(file->fileSize());

		args.GetReturnValue().Set(size);
	}));

	return tpl;
}

Local<Object> v8Object(File *file) {
	auto isolate = V8Helper::instance()->isolate;
	auto tpl = createV8FileTemplate(isolate);
	Local<Object> obj = tpl->NewInstance();
	obj->SetInternalField(0, External::New(isolate, file));

	return obj;
}