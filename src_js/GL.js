


class GLBufferAttribute{

	constructor(name, location, count, type, normalize, bytes, offset, more = {}){
		this.name = name;
		this.location = location;
		this.count = count;
		this.type = type;
		this.normalize = normalize;
		this.bytes = bytes;
		this.offset = offset;
		
		this.targetType = more.targetType;
	}

}

class GLBuffer{

	constructor(){
		this.vao = gl.createVertexArray();
		this.vbo = gl.createBuffer();
		this.vbos = new Map();
		this.count = 0;

		this.attributes = [];
	}

	setEmptyInterleaved(attributes, size){
		this.buffer = null;
		this.attributes = attributes;

		gl.bindVertexArray(this.vao);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
		gl.bufferData(gl.ARRAY_BUFFER, size, 0, gl.DYNAMIC_DRAW);

		let stride = attributes.reduce( (a, v) => a + v.bytes, 0);

		for(let attribute of attributes){
			gl.enableVertexAttribArray(attribute.location);

			if(attribute.targetType === "int"){
				gl.vertexAttribIPointer(
					attribute.location, 
					attribute.count, 
					attribute.type, 
					stride, 
					attribute.offset);
			}else{
				gl.vertexAttribPointer(
					attribute.location, 
					attribute.count, 
					attribute.type, 
					attribute.normalize, 
					stride, 
					attribute.offset);
			}

		}

		gl.bindVertexArray(0);

		this.count = 0;
	}

	setInterleaved(buffer, attributes, count){
		this.buffer = buffer;
		this.attributes = attributes;

		gl.bindVertexArray(this.vao);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
		gl.bufferData(gl.ARRAY_BUFFER, buffer.byteLength, buffer, gl.DYNAMIC_DRAW);

		let stride = attributes.reduce( (a, v) => a + v.bytes, 0);

		for(let attribute of attributes){
			gl.enableVertexAttribArray(attribute.location);

			if(attribute.targetType === "int"){
				gl.vertexAttribIPointer(
					attribute.location, 
					attribute.count, 
					attribute.type, 
					stride, 
					attribute.offset);
			}else{
				gl.vertexAttribPointer(
					attribute.location, 
					attribute.count, 
					attribute.type, 
					attribute.normalize, 
					stride, 
					attribute.offset);
			}

		}

		gl.bindVertexArray(0);

		this.count = count;
	}

	setConsecutive(buffers, attributes, count){
		this.buffers = buffers;
		this.attributes = attributes;

		gl.bindVertexArray(this.vao);

		for(let i = 0; i < attributes.length; i++){
			let attribute = attributes[i];
			let buffer = buffers[i];

			let vbo = gl.createBuffer();
			gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
			gl.bufferData(gl.ARRAY_BUFFER, buffer.byteLength, buffer, gl.DYNAMIC_DRAW);

			gl.enableVertexAttribArray(attribute.location);
			gl.vertexAttribPointer(
				attribute.location, 
				attribute.count, 
				attribute.type, 
				attribute.normalize, 
				0, 
				0);

			this.vbos.set(attribute.name, vbo);

		}

		gl.bindVertexArray(0);
		this.count = count;
	}

	set(buffer, attributes, count){
		this.setInterleaved(buffer, attributes, count);
	}

	computeBoundingBox(){

		let box = new Box3();

		let aPosition = this.attributes.find(a => a.name === "position");
		let offset = aPosition.offset;

		let count = this.count;
		let buffer = null;
		let stride = 0;

		if(this.buffer){
			buffer = new Float32Array(this.buffer);
			stride = this.attributes.reduce( (a, v) => a + v.bytes, 0);
		}else if(this.buffers){
			let bufferIndex = this.attributes.indexOf(aPosition);
			buffer = new Float32Array(this.buffers[bufferIndex]);
			stride = aPosition.bytes;
		}

		stride = stride / 4;

		for(let i = 0; i < count; i++){

			let x = buffer[i * stride + offset + 0];
			let y = buffer[i * stride + offset + 1];
			let z = buffer[i * stride + offset + 2];

			box.expandByXYZ(x, y, z);
		}

		return box;
	}

}

class GLTexture{

	constructor(width, height, data){
		this.type = gl.TEXTURE_2D;
		this.width = width;
		this.height = height;
		this.data = data;

		this.handle = gl.createTexture();

		gl.bindTexture(gl.TEXTURE_2D, this.handle);

		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_ANISOTROPY, 16.0);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);

		let level = 0;
		let border = 0;
		gl.texImage2D(
			gl.TEXTURE_2D, level, gl.RGBA, 
			this.width, this.height, border, 
			gl.RGBA, gl.UNSIGNED_BYTE, this.data);

		gl.generateMipmap(gl.TEXTURE_2D);

		gl.bindTexture(gl.TEXTURE_2D, 0);
	}

	dispose(){
		gl.deleteTexture(this.handle);
	}

};

class GLMaterial{

	constructor(){
		this.glDrawMode = gl.POINTS;
		this.shader = null;
		this.texture = null;
		this.depthTest = true;
		this.depthWrite = true;
	}

};

class SSBO{

	constructor(attributes){
		this.attributes = attributes;
		this.offsets = this.computeOffsets(this.attributes);
		this.bytes = this.attributes.reduce( (p, c) => p + c.bytes, 0);

		this.ssbo = gl.createBuffer();
		gl.namedBufferData(this.ssbo, this.bytes, 0, gl.DYNAMIC_DRAW);

		this.buffer = new ArrayBuffer(this.bytes);
		this.view = new DataView(this.buffer);
		this.viewF32 = new Float32Array(this.buffer);
	}

	computeOffsets(attributes){
		
		let offsets = {};

		let offset = 0;
		for(let attribute of attributes){
			offsets[attribute.name] = offset;

			offset += attribute.bytes;
		}

		return offsets;
	}

	setFloat32(name, value){
		this.view.setFloat32(this.offsets[name], value, true);
	}

	setFloat32Array(name, value){
		this.viewF32.set(value, this.offsets[name] / 4, true);
	}


};


