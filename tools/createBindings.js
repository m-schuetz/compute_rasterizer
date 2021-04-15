

const fs = require("fs");

let vendors = ["OES", "EXT", "KHR", "ARB", "3DFX", "APPLE", 
		"INTEL", "AMD", "ATI", "NV", "PGI", "SGIS", "OML",
		"REND", "S3TC", "SGIX", "NVX", "SGI", "SUN", "OVR",
		"WIN", "INGR", "MESAX", "MESA", "IBM", "SUNX"];

// NOTE: js does not have uint64, therefore uint64 are regularely treated as uint32
let castings = {
	"GLuint": function(commandName, argument, type, varname, indent){
		//let code = `${indent}String::Utf8Value ${varname}UTF8(${argument});\n`;
		//code += `${indent}GLuint ${varname} = std::stoul(*${varname}UTF8);\n`;

		let code = `${indent}${type} ${varname} = ${argument}->Uint32Value();`;

		return code;
	},
	"GLint": function(commandName, argument, type, varname, indent){
		//let code = `${indent}String::Utf8Value ${varname}UTF8(${argument});\n`;
		//code += `${indent}GLint ${varname} = std::stoi(*${varname}UTF8);\n`;

		let code = `${indent}${type} ${varname} = ${argument}->Int32Value();`;

		return code;
	},
	"GLuint64": function(commandName, argument, type, varname, indent){
		//let code = `${indent}String::Utf8Value ${varname}UTF8(${argument});\n`;
		//code += `${indent}GLuint64 ${varname} = std::stoull(*${varname}UTF8);\n`;

		let code = `${indent}${type} ${varname} = ${type}(${argument}->Uint32Value());`;

		return code;
	},
	"GLint64": function(commandName, argument, type, varname, indent){
		//let code = `${indent}String::Utf8Value ${varname}UTF8(${argument});\n`;
		//code += `${indent}GLint64 ${varname} = std::stoll(*${varname}UTF8);\n`;

		let code = `${indent}${type} ${varname} = ${type}(${argument}->Int32Value());`;

		return code;
	},
	"GLushort": function(commandName, argument, type, varname, indent){
		//let code = `${indent}String::Utf8Value ${varname}UTF8(${argument});\n`;
		//code += `${indent}GLushort ${varname} = std::stoul(*${varname}UTF8);\n`;

		let code = `${indent}${type} ${varname} = ${type}(${argument}->Uint32Value());`;

		return code;
	},
	"GLshort": function(commandName, argument, type, varname, indent){
		//let code = `${indent}String::Utf8Value ${varname}UTF8(${argument});\n`;
		//code += `${indent}GLshort ${varname} = std::stoi(*${varname}UTF8);\n`;

		let code = `${indent}${type} ${varname} = ${type}(${argument}->Int32Value());`;

		return code;
	},
	"GLbyte": function(commandName, argument, type, varname, indent){
		//let code = `${indent}String::Utf8Value ${varname}UTF8(${argument});\n`;
		//code += `${indent}GLbyte ${varname} = std::stoi(*${varname}UTF8);\n`;

		let code = `${indent}${type} ${varname} = ${type}(${argument}->Int32Value());`;

		return code;
	},
	"GLubyte": function(commandName, argument, type, varname, indent){
		//let code = `${indent}String::Utf8Value ${varname}UTF8(${argument});\n`;
		//code += `${indent}GLubyte ${varname} = std::stoul(*${varname}UTF8);\n`;

		let code = `${indent}${type} ${varname} = ${type}(${argument}->Uint32Value());`;

		return code;
	},
	"GLfloat": function(commandName, argument, type, varname, indent){
		//let code = `${indent}String::Utf8Value ${varname}UTF8(${argument});\n`;
		//code += `${indent}GLfloat ${varname} = std::stof(*${varname}UTF8);\n`;

		let code = `${indent}${type} ${varname} = ${type}(${argument}->NumberValue());`;

		return code;
	},
	"GLdouble": function(commandName, argument, type, varname, indent){
		//let code = `${indent}String::Utf8Value ${varname}UTF8(${argument});\n`;
		//code += `${indent}GLdouble ${varname} = std::stod(*${varname}UTF8);\n`;

		let code = `${indent}${type} ${varname} = ${argument}->NumberValue();`;

		return code;
	},
	"GLfloat*": function(commandName, argument, type, varname, indent){

		let code = `
	${type} ${varname} = nullptr;
	if (${argument}->IsFloat32Array()) {
		v8::Local<v8::Float32Array> view = (${argument}).As<v8::Float32Array>();
		auto buffer = view->Buffer();
		void *bdata = view->Buffer()->GetContents().Data();
		${varname} = reinterpret_cast<${type}>(bdata);
	} else {
		cout << "ERROR(${commandName}): array must be of type Float32Array" << endl;
		exit(1);
	}
`
		return code;

	},
	"GLdouble*": function(commandName, argument, type, varname, indent){

		let code = `
	${type} ${varname} = nullptr;
	if (${argument}->IsFloat64Array()) {
		v8::Local<v8::Float64Array> view = (${argument}).As<v8::Float64Array>();
		auto buffer = view->Buffer();
		void *bdata = view->Buffer()->GetContents().Data();
		${varname} = reinterpret_cast<${type}>(bdata);
	} else {
		cout << "ERROR(${commandName}): array must be of type Float64Array" << endl;
		exit(1);
	}
`
		return code;

	},
	"void*": function(commandName, argument, type, varname, indent){

		let code = `
	${type} ${varname} = nullptr;
	if (${argument}->IsArrayBuffer()) {
		v8::Local<v8::ArrayBuffer> buffer = (${argument}).As<v8::ArrayBuffer>();
		void *bdata = buffer->GetContents().Data();
		${varname} = reinterpret_cast<${type}>(bdata);
	} else if (${argument}->IsArrayBufferView()) {
		v8::Local<v8::ArrayBufferView> view = (${argument}).As<v8::ArrayBufferView>();
		auto buffer = view->Buffer();
		void *bdata = view->Buffer()->GetContents().Data();
		${varname} = reinterpret_cast<${type}>(bdata);
	}else {
		cout << "ERROR(${commandName}): array must be of type ArrayBuffer" << endl;
		exit(1);
	}
`
		return code;

	},
	"GLubyte*": function(commandName, argument, type, varname, indent){

		let code = `
	${type} ${varname} = nullptr;
	if (${argument}->IsUint8Array()) {
		v8::Local<v8::Uint8Array> view = (${argument}).As<v8::Uint8Array>();
		auto buffer = view->Buffer();
		void *bdata = view->Buffer()->GetContents().Data();
		${varname} = reinterpret_cast<${type}>(bdata);
	} else {
		cout << "ERROR(${commandName}): array must be of type Uint8Array" << endl;
		exit(1);
	}
`
		return code;

	},
	"GLbyte*": function(commandName, argument, type, varname, indent){

		let code = `
	${type} ${varname} = nullptr;
	if (${argument}->IsInt8Array()) {
		v8::Local<v8::Int8Array> view = (${argument}).As<v8::Int8Array>();
		auto buffer = view->Buffer();
		void *bdata = view->Buffer()->GetContents().Data();
		${varname} = reinterpret_cast<${type}>(bdata);
	} else {
		cout << "ERROR(${commandName}): array must be of type Int8Array" << endl;
		exit(1);
	}
`
		return code;

	},
	"GLshort*": function(commandName, argument, type, varname, indent){

		let code = `
	${type} ${varname} = nullptr;
	if (${argument}->IsInt16Array()) {
		v8::Local<v8::Int16Array> view = (${argument}).As<v8::Int16Array>();
		auto buffer = view->Buffer();
		void *bdata = view->Buffer()->GetContents().Data();
		${varname} = reinterpret_cast<${type}>(bdata);
	} else {
		cout << "ERROR(${commandName}): array must be of type Int16Array" << endl;
		exit(1);
	}
`
		return code;

	},
	"GLushort*": function(commandName, argument, type, varname, indent){

		let code = `
	${type} ${varname} = nullptr;
	if (${argument}->IsUint16Array()) {
		v8::Local<v8::Uint16Array> view = (${argument}).As<v8::Uint16Array>();
		auto buffer = view->Buffer();
		void *bdata = view->Buffer()->GetContents().Data();
		${varname} = reinterpret_cast<${type}>(bdata);
	} else {
		cout << "ERROR(${commandName}): array must be of type Uint16Array" << endl;
		exit(1);
	}
`
		return code;

	},
	"GLint*": function(commandName, argument, type, varname, indent){

		let code = `
	${type} ${varname} = nullptr;
	if (${argument}->IsInt32Array()) {
		v8::Local<v8::Int32Array> view = (${argument}).As<v8::Int32Array>();
		auto buffer = view->Buffer();
		void *bdata = view->Buffer()->GetContents().Data();
		${varname} = reinterpret_cast<${type}>(bdata);
	} else {
		cout << "ERROR(${commandName}): array must be of type Int32Array" << endl;
		exit(1);
	}
`
		return code;

	},
	"GLuint*": function(commandName, argument, type, varname, indent){

		let code = `
	${type} ${varname} = nullptr;
	if (${argument}->IsUint32Array()) {
		v8::Local<v8::Uint32Array> view = (${argument}).As<v8::Uint32Array>();
		auto buffer = view->Buffer();
		void *bdata = view->Buffer()->GetContents().Data();
		${varname} = reinterpret_cast<${type}>(bdata);
	} else {
		cout << "ERROR(${commandName}): array must be of type Uint32Array" << endl;
		exit(1);
	}
`
		return code;

	}
}

castings["GLchar"] = castings["GLbyte"];
castings["GLchar*"] = castings["GLbyte*"];

castings["GLfixed"] = castings["GLint"];
castings["GLfixed*"] = castings["GLint*"];
castings["GLsizei"] = castings["GLint"];
castings["GLsizei*"] = castings["GLint*"];
castings["GLbitfield"] = castings["GLuint"];

castings["GLenum"] = castings["GLuint"];
castings["GLenum*"] = castings["GLuint*"];

castings["GLclampf"] = castings["GLfloat"];
castings["GLclampf*"] = castings["GLfloat*"];

castings["GLboolean"] = castings["GLubyte"];
castings["GLboolean*"] = castings["GLubyte*"];

castings["GLsizeiptr"] = castings["GLint64"];
castings["GLintptr"] = castings["GLint64"];

castings["GLuint64EXT"] = castings["GLuint64"];
castings["GLuint64EXT*"] = castings["GLuint64*"];

castings["GLvoid*"] = castings["void*"];




function isConstant(line){
	let isConstant = line.match(/GL_.*/) !== null;

	// unsigned long long not handled at this time
	isConstant = isConstant && !line.endsWith("ull");

	let hasVendor = false;
	for(let vendor of vendors){
		let regex1 = new RegExp(`GL_\\w*_(${vendor}) .*`, "g");
		let regex2 = new RegExp(`GL_${vendor}_.*`, "g");
		let fromVendor = (line.match(regex1) !== null) || (line.match(regex2) !== null);

		hasVendor = hasVendor || fromVendor;
	}

	return isConstant && !hasVendor;
}

function isCommand(line){
	//let isCommand = line.match(/.*/) !== null;
	let isCommand = line.match(/(void) gl\w* \(.*\)/) !== null;

	if(line.includes("glAsyncMarkerSGIX")){
		debugger;
	}
	
	let hasVendor = false;
	for(let vendor of vendors){
		let regex = RegExp(`(void) gl\\w*${vendor} \\(.*\\)`, "g");

		let fromVendor = line.match(regex) !== null;

		hasVendor = hasVendor || fromVendor;
	}


	return isCommand && !hasVendor;
}

function parse(file){
	let content = fs.readFileSync(file, "utf8");
	let lines = content.split("\n");

	let extname = lines[0].trim();
	let exturl = lines[1].trim();
	let extstring = lines[2].trim();
	let reuse = lines[3].trim();

	let constants = [];
	let commands = [];

	for(let i = 4; i < lines.length; i++){
		let line = lines[i].trim();

		if(line.includes("glAsyncMarkerSGIX")){
			debugger;
		}

		if(isConstant(line)){
			constants.push(line);
		}else if(isCommand(line)){

			line = line.replace(/ \*/g, "* ");

			commands.push(line);
		}

	}


	return {
		extname: extname,
		exturl: exturl,
		constants: constants, 
		commands: commands};
}

function createConstantBinding(constant){
	//console.log(constant);
	let matches = constant.match(/GL_(\w*)\w*(.*)/);

	let constantName = matches[1];
	let constantValue = `GL_${constantName}`;

	let binding = `CREATE_CONSTANT_ACCESSOR("${constantName}", ${constantValue});`;

	//let binding = `constants["${constantName}"] = ${constantValue};`;

	return binding;
}

function createCommandBinding(command){

	let match = command.match(/^(.+) ([a-z][a-z0-9_]*) \((.+)\)$/i);
	let returnType = match[1];
	let commandName = match[2];
	let argumentsString = match[3];
	let argumentsStrings = argumentsString.split(",").map(a => a.trim());
	let numArguments = argumentsStrings.length;

	let bindingName = commandName.charAt(2).toLowerCase() + commandName.slice(3);

	let parseArgs = "";
	let execStatement = `${commandName}(`;

	for(let i = 0; i < argumentsStrings.length; i++){
		let arg = argumentsStrings[i];

		let bracketMatch = arg.match(/(.*) (\w*)(\[.*\])/);
		if(bracketMatch){
			console.log(arg);
			arg = `${bracketMatch[1]}* ${bracketMatch[2]}`;
			console.log(arg);
		}

		let tokens = arg.split(" " );

		if(tokens[0] === "const"){
			tokens = tokens.slice(1);
		}

		if(tokens.length !== 2){
			console.log(`not handled(more than 2 tokens): ${command}`);
			return "";
		}else if(tokens[1].includes("*")){
			console.log(`not handled(TODO pointer): ${command}`);
			return "";
		}else if(castings[tokens[0]] === undefined){
			console.log(`not handled(missing casting): ${command}`);
			return "";
		}

		let varType = tokens[0];
		let varName = tokens[1];

		let indent = i == 0 ? "" : "\t";
		//let newline = i === argumentsStrings.length - 1 ? "" : "\n";


		//stringArgs += `${indent}String::Utf8Value ${varName}UTF8(args[${i}]);${newline}`;
		//parseArgs += `${indent}${varType} ${varName} = std::stoi(*${varName}UTF8);${newline}`;

		parseArgs += castings[varType](commandName, `args[${i}]`, varType, varName, "\t") + "\n";

		let comma = i === argumentsStrings.length - 1 ? "" : ", ";
		execStatement += `${varName}${comma}`;
	}
	execStatement += ");";


	let returnStatement = "// args.GetReturnValue().Set(fb);";

	let binding = 
	`tpl->Set(String::NewFromUtf8(isolate, "${bindingName}"), FunctionTemplate::New(isolate, [](const FunctionCallbackInfo<Value>& args) {
	if (args.Length() != ${numArguments}) {
		V8Helper::_instance->throwException("${bindingName} requires ${numArguments} arguments");
		return;
	}

${parseArgs}
	${execStatement}
}));
`

	return binding;
}

let constants = [];
let commands = [];

let extDir = "C:/dev/workspaces/glew/master/auto/extensions/gl";
let coreDir = "C:/dev/workspaces/glew/master/auto/core/gl";

let coreFiles = fs.readdirSync(coreDir);
let extFiles = fs.readdirSync(extDir);

let files = [];
files.push(...coreFiles
	.filter(f => f.startsWith("GL_VERSION"))
	.map(f => `${coreDir}/${f}`)
);
files.push(...extFiles
	.filter(f => f.startsWith("GL_"))
	.map(f => `${extDir}/${f}`)
);

//for(let file of files){
//	let result = parse(file);
//
//	constants.push(...result.constants);
//	commands.push(...result.commands);
//
//}

async function createBindings(){
	
	let lines = [];

	for(let file of files){
		let result = parse(file);

		let constantBindings = result.constants.map(createConstantBinding);
		let commandBindings =  result.commands.map(createCommandBinding);

		if(constantBindings.length === 0 && commandBindings.length === 0){
			lines.push(`// empty / skipped / ignored: ${result.extname}`);
		}else{
			lines.push(`/* ------------------------------ ${result.extname} ------------------------------ */`);
			lines.push(``);
			lines.push(...constantBindings);
			lines.push(``);
			lines.push(...commandBindings);
			lines.push(``);
			lines.push(``);
		}

	}

	let content = lines.join("\n");

	let fileHandle = await fs.promises.open("./bindings.txt", "w");
	await fileHandle.writeFile(content);
	await fileHandle.close();

}

createBindings();


//console.log(constantBindings);
//console.log(commands);

