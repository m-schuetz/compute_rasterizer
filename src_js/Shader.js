

if(typeof Shader === "undefined"){

	Shader = class Shader{
	
		constructor(components){

			log("==== creating new shader ====");

			this.components = components;

			this.program = null;
			this.uniforms = {};
			this.uniformBlocks = {};
			this.compiled = false;

			this.compile();
		}

	}
}

Shader.prototype.compileShader = function(source, shaderType){
	//log("compileShader");

	let glshader = gl.createShader(shaderType);

	gl.shaderSource(glshader, source);
	gl.compileShader(glshader);

	let isCompiled = new Int32Array(1);
	gl.getShaderiv(glshader, gl.COMPILE_STATUS, isCompiled);

	if(isCompiled[0] === gl.FALSE){
		log("Shader.prototype.compileShader failed");

		let maxLength = new Int32Array(1);
		gl.getShaderiv(glshader, gl.INFO_LOG_LENGTH, maxLength);
		maxLength = maxLength[0];

		log(`maxLength: ${maxLength}`);

		let infoLogU8 = new Int8Array(maxLength);
		let actualLength = new Int32Array(1);
		gl.getShaderInfoLog(glshader, maxLength, actualLength, infoLogU8);
		actualLength = actualLength[0];
		log(`actualLength: ${actualLength}`);

		gl.deleteShader(glshader);

		let str = "";
		for(let i = 0; i < actualLength; i++){
			str += String.fromCharCode(infoLogU8[i]);
		}

		log(`error in shader:`);
		log(str);

		return null;
	}

	return glshader;

};

Shader.prototype.compile = function(){

	this.compiled = false;

	let compiledComponents = [];
	for(let component of this.components){

		log(`compiling component: ${component.path}`);
		
		let source = readTextFile(component.path);
		let compiled = this.compileShader(source, component.type);

		if(compiled === null){
			log(`shader compilation failed: ${component.path}`);

			return;
		}

		compiledComponents.push(compiled);
	}

	//this.vsSource = readTextFile(this.vsPath);
	//this.fsSource = readTextFile(this.fsPath);

	//log(`compiling vs shader: ${this.vsPath}`);
	//let vs = this.compileShader(this.vsSource, gl.VERTEX_SHADER);
	//
	//log(`compiling fs shader: ${this.fsPath}`);
	//let fs = this.compileShader(this.fsSource, gl.FRAGMENT_SHADER);

	//if(vs === null || fs === null){
	//	log("Shader.compile failed");

	//	return;
	//}

	if(this.program === null){
		this.program = gl.createProgram();
	}else{
		gl.useProgram(0);
		gl.deleteProgram(this.program);
		this.program = gl.createProgram();
	}

	//log(this.program);

	for(let compiledComponent of compiledComponents){
		gl.attachShader(this.program, compiledComponent);
	}

	//gl.attachShader(this.program, vs);
	//gl.attachShader(this.program, fs);

	gl.linkProgram(this.program);

	let isLinked = new Int32Array(1);
	gl.getProgramiv(this.program, gl.LINK_STATUS, isLinked);
	if(isLinked[0] === gl.FALSE){
		log("TODO: PROGRAM LINK ERROR HANDLING!!!");

		let msg = gl.getProgramInfoLogString(this.program);

		log(msg);
		
		// // We don't need the program anymore.
		// glDeleteProgram(program);
		// // Don't leak shaders either.
		// glDeleteShader(vertexShader);
		// glDeleteShader(fragmentShader);


	}

	for(let compiledComponent of compiledComponents){
		gl.detachShader(this.program, compiledComponent);
		gl.deleteShader(compiledComponent);
	}

	//gl.detachShader(this.program, vs);
	//gl.detachShader(this.program, fs);
	//gl.deleteShader(vs);
	//gl.deleteShader(fs);

	this.queryUniforms();
	this.queryUniformBlocks();

	this.compiled = true;

}

Shader.prototype.queryUniforms = function(){

	gl.useProgram(this.program);

	this.uniforms = {};

	let count = new Int32Array(1);
	gl.getProgramiv(this.program, gl.ACTIVE_UNIFORMS, count);

	let bufSize = 64;
	let buffer = new Int8Array(bufSize);
	let bufferU8 = new Uint8Array(buffer.buffer);
	let length = new Int32Array(1);
	let size = new Int32Array(1);
	let type = new Uint32Array(1);

	for(let i = 0; i < count[0]; i++){


		gl.getActiveUniform(this.program, i, bufSize, length, size, type, buffer);

		let name = "";
		for(let j = 0; j < length[0]; j++){
			name += String.fromCharCode(bufferU8[j]);
		}

		let id = gl.getUniformLocation(this.program, name);

		this.uniforms[name] = id;

		//log(`${name} -> ${id}`);

	}
	
};

Shader.prototype.queryUniformBlocks = function(){

	this.uniformBlocks = {};

	let count = new Int32Array(1);
	gl.getProgramiv(this.program, gl.ACTIVE_UNIFORM_BLOCKS, count);

	let bufSize = 64;
	let buffer = new Int8Array(bufSize);
	let bufferU8 = new Uint8Array(buffer.buffer);
	let length = new Int32Array(1);

	for(let i = 0; i < count[0]; i++){
		gl.getActiveUniformBlockName(this.program, i, bufSize, length, buffer);

		let name = "";
		for(let j = 0; j < length[0]; j++){
			name += String.fromCharCode(bufferU8[j]);
		}

		//log(name);

		this.uniformBlocks[name] = {
			uniforms: {},
		};

	}

	for(let blockName of Object.keys(this.uniformBlocks)){
		let block = this.uniformBlocks[blockName];

		let blockIndex = gl.getUniformBlockIndex(this.program, blockName);

		let uniformCount = new Int32Array(1);
		gl.getActiveUniformBlockiv(this.program, blockIndex, gl.UNIFORM_BLOCK_ACTIVE_UNIFORMS, uniformCount);
		uniformCount = uniformCount[0];

		let uniformIndices = new Int32Array(uniformCount);
		gl.getActiveUniformBlockiv(this.program, blockIndex, gl.UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, uniformIndices);

		let blockSize = new Int32Array(1);
		gl.getActiveUniformBlockiv(this.program, blockIndex, gl.UNIFORM_BLOCK_DATA_SIZE, blockSize);
		blockSize = blockSize[0];

		let blockBinding = new Int32Array(1);
		gl.getActiveUniformBlockiv(this.program, blockIndex, gl.UNIFORM_BLOCK_BINDING, blockBinding);
		blockBinding = blockBinding[0];

		for(let uniformIndex of uniformIndices){
			gl.getActiveUniformName(this.program, uniformIndex, bufSize, length, buffer);

			let name = "";
			for(let j = 0; j < length[0]; j++){
				name += String.fromCharCode(bufferU8[j]);
			}

			name = name.replace(`${blockName}.`, "");

			let type = new Int32Array(1);
			let offset = new Int32Array(1);
			gl.getActiveUniformsiv(this.program, 1, new Uint32Array([uniformIndex]), gl.UNIFORM_TYPE, type );
			gl.getActiveUniformsiv(this.program, 1, new Uint32Array([uniformIndex]), gl.UNIFORM_OFFSET, offset );

			block.uniforms[name] = {
				name: name,
				index: uniformIndex,
				type: type[0],
				offset: offset[0],
			};

			//log(`${uniformIndex} - ${name}: ${block.uniforms[name].offset}`);
		}


		block.index = blockIndex;
		block.byteLength = blockSize;

		block.bufferData = new ArrayBuffer(block.byteLength);
		block.bufferID = gl.createBuffer();
		block.blockBinding = blockBinding;
		block.view = new DataView(block.bufferData);
		block.viewF32 = new Float32Array(block.bufferData);

		block.setFloat32 = (name, value) => {
			block.view.setFloat32(block.uniforms[name].offset, value, true);
		};

		block.setInt32 = (name, value) => {
			block.view.setInt32(block.uniforms[name].offset, value, true);
		};

		block.setFloat32Array = (name, value) => {
			block.viewF32.set(value, block.uniforms[name].offset / 4, true);
		};

		block.submit = () => {
			gl.namedBufferSubData(block.bufferID, 0, block.byteLength, block.bufferData);
		};

		block.bind = () => {
			gl.bindBufferBase(gl.UNIFORM_BUFFER, block.blockBinding, block.bufferID);
		};

		gl.namedBufferData(block.bufferID, block.byteLength, 0, gl.DYNAMIC_DRAW);

	}

};


Shader.prototype.watch = function(callback){

	if(typeof this.beingWatched !== "undefined"){
		log("shader is already being watched");

		return;
	}

	let onChange = () => {
		this.compile();

		if(callback){
			callback();
		}
	};

	// TODO monitor file
	//monitorFile(this.vsPath, onChange);
	//monitorFile(this.fsPath, onChange);

	for(let component of this.components){
		monitorFile(component.path, onChange);
	}

	this.beingWatched = true;
};


"Shader.js"