

if(typeof BrushNode === "undefined"){

	BrushNode = class BrushNode extends SceneNode{

		constructor(name){

			super(name);

			this.buffer = this.createBuffer();
			this.material = this.createMaterial();

			this.components.push(this.buffer, this.material);

		}

	};

}

BrushNode.prototype.createBuffer = function(){
	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("pivot", 1, 3, gl.FLOAT, gl.FALSE, 12, 12),
		new GLBufferAttribute("color", 2, 4, gl.FLOAT, gl.TRUE, 16, 24),
		new GLBufferAttribute("size", 3, 1, gl.FLOAT, gl.FALSE, 4, 40),
		new GLBufferAttribute("time", 4, 1, gl.FLOAT, gl.FALSE, 4, 44),
		new GLBufferAttribute("random", 5, 1, gl.FLOAT, gl.FALSE, 4, 48),
	];

	let stride = attributes.reduce( (a, v) => a + v.bytes, 0);

	let buffer = new GLBuffer();
	buffer.attributes = attributes;
	buffer.count = 0;
	buffer.stride = stride;

	let capacityPoints = 1000 * 1000;
	let capacityBytes = capacityPoints * stride;

	gl.bindVertexArray(buffer.vao);
	gl.bindBuffer(gl.ARRAY_BUFFER, buffer.vbo);
	gl.bufferData(gl.ARRAY_BUFFER, capacityBytes, 0, gl.DYNAMIC_DRAW);

	for(let attribute of attributes){
		gl.enableVertexAttribArray(attribute.location);
		gl.vertexAttribPointer(
			attribute.location, 
			attribute.count, 
			attribute.type, 
			attribute.normalize, 
			stride, 
			attribute.offset);	
	}

	gl.bindVertexArray(0);

	return buffer;
}

BrushNode.prototype.createMaterial = function(){

	let vsPath = "../../resources/shaders/brush.vs";
	let fsPath = "../../resources/shaders/brush.fs";

	let shader = new Shader([
		{type: gl.VERTEX_SHADER, path: vsPath},
		{type: gl.FRAGMENT_SHADER, path: fsPath},
	]);
	shader.watch();

	let material = new GLMaterial();

	material.shader = shader;
	material.texture = gradientTexture;
	material.glDrawMode = gl.POINTS
	
	return material;
};

BrushNode.prototype.addData = function(data){

	let buffer = this.buffer;

	let localCount = data.byteLength / buffer.stride;

	gl.bindVertexArray(buffer.vao);


	gl.namedBufferSubData(buffer.vbo, buffer.count * buffer.stride, data.byteLength, data);

	gl.bindVertexArray(0);

	buffer.count += localCount;

	//log(buffer.count * buffer.stride);

};

BrushNode.prototype.update = function(){
	//this.boundingBoxWorld.copy(this.boundingBox).applyMatrix4(this.world);
	//let node = this.parent;
	//while(node){
	//	node.boundingBoxWorld.min.min(this.boundingBoxWorld.min);
	//	node.boundingBoxWorld.max.max(this.boundingBoxWorld.max);
	//	node = node.parent;
	//}
	//for(let child of this.children){
	//	child.update();
	//}
};








