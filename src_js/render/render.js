

var frameCount = 0;


var drawImage = function(texture, x, y){

};


var renderBox = function(box, view, proj, target){

	if(typeof renderLinesShader === "undefined"){
		let vsPath = "../../resources/shaders/lines.vs";
		let fsPath = "../../resources/shaders/lines.fs";
		let shader = new Shader([
			{type: gl.VERTEX_SHADER, path: vsPath},
			{type: gl.FRAGMENT_SHADER, path: fsPath},
		]);
		shader.watch();

		renderLinesShader = shader;
	}
	let shader = renderLinesShader;
	let shader_data = shader.uniformBlocks.shader_data;

	gl.useProgram(shader.program);

	// 12 lines, 2 vertices each line, for each node/box
	let numVertices = 12 * 2;
	let vertices = new Float32Array(numVertices * 4);
	let u32 = new Uint32Array(vertices);
	let color = 0xFF00FFFF;

	// uint32 bits to float bits, since we're feeding a float buffer
	color = new Float32Array(new Uint32Array([color]).buffer)[0];

	{
		let data = [
			// BOTTOM
			box.min.x, box.min.y, box.min.z, color,
			box.max.x, box.min.y, box.min.z, color,
			
			box.max.x, box.min.y, box.min.z, color,
			box.max.x, box.min.y, box.max.z, color,
			
			box.max.x, box.min.y, box.max.z, color,
			box.min.x, box.min.y, box.max.z, color,

			box.min.x, box.min.y, box.max.z, color,
			box.min.x, box.min.y, box.min.z, color,

			// TOP
			box.min.x, box.max.y, box.min.z, color,
			box.max.x, box.max.y, box.min.z, color,
			
			box.max.x, box.max.y, box.min.z, color,
			box.max.x, box.max.y, box.max.z, color,
			
			box.max.x, box.max.y, box.max.z, color,
			box.min.x, box.max.y, box.max.z, color,

			box.min.x, box.max.y, box.max.z, color,
			box.min.x, box.max.y, box.min.z, color,

			// CONNECTIONS
			box.min.x, box.min.y, box.min.z, color,
			box.min.x, box.max.y, box.min.z, color,
			
			box.max.x, box.min.y, box.min.z, color,
			box.max.x, box.max.y, box.min.z, color,
			
			box.max.x, box.min.y, box.max.z, color,
			box.max.x, box.max.y, box.max.z, color,

			box.min.x, box.min.y, box.max.z, color,
			box.min.x, box.max.y, box.max.z, color];

		vertices.set(data, 0);

	}

	let buffer = vertices.buffer;

	let vao = gl.createVertexArray();
	let vbo = gl.createBuffer();
	gl.bindVertexArray(vao);
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
	gl.bufferData(gl.ARRAY_BUFFER, buffer.byteLength, buffer, gl.DYNAMIC_DRAW);
	

	let transform = new Matrix4();
	transform.copy(Matrix4.IDENTITY);
	transform.multiply(proj).multiply(view);

	{
		shader_data.setFloat32Array("transform", transform.elements);

		//gl.bindBufferBase(gl.UNIFORM_BUFFER, 4, shader_data.bufferID);
		shader_data.bind();
		shader_data.submit();
	}

	//let mat32 = new Float32Array(16);
	//mat32.set(transform.elements);

	//gl.uniformMatrix4fv(/*shader.uniforms.uTransform*/ 1, 1, gl.FALSE, mat32);

	gl.enableVertexAttribArray(0);
	gl.enableVertexAttribArray(1);
	gl.vertexAttribPointer(0, 3, gl.FLOAT, gl.FALSE, 16, 0);
	gl.vertexAttribPointer(1, 3, gl.UNSIGNED_BYTE, gl.TRUE, 16, 12);

	gl.drawArrays(gl.LINES, 0, vertices.length / 4);

	gl.bindBuffer(gl.ARRAY_BUFFER, 0);
	gl.deleteBuffers(1, new Uint32Array([vbo]));
	gl.disableVertexAttribArray(0);
	gl.disableVertexAttribArray(1);
	gl.bindVertexArray(0);

};


var renderLines = function(lines, view, proj){

	if(typeof renderLinesShader === "undefined"){
		let vsPath = "../../resources/shaders/lines.vs";
		let fsPath = "../../resources/shaders/lines.fs";
		let shader = new Shader([
			{type: gl.VERTEX_SHADER, path: vsPath},
			{type: gl.FRAGMENT_SHADER, path: fsPath},
		]);
		shader.watch();

		renderLinesShader = shader;
	}
	let shader = renderLinesShader;
	let shader_data = shader.uniformBlocks.shader_data;

	gl.useProgram(shader.program);
	
	
	let numVertices = lines.length / 2;
	let vertices = new Float32Array(numVertices * 4);
	let u32 = new Uint32Array(vertices);
	let color = 0xFF00FFFF;

	for(let i = 0; i < numVertices; i++){

		vertices[4 * i + 0] = lines[4 * i + 0];
		vertices[4 * i + 1] = lines[4 * i + 1];
		vertices[4 * i + 2] = lines[4 * i + 2];

		let color = lines[4 * i + 3];
		color = new Float32Array(new Uint32Array([color]).buffer)[0];

		vertices[4 * i + 3] = color;

	}

	// uint32 bits to float bits, since we're feeding a float buffer
	//color = new Float32Array(new Uint32Array([color]).buffer)[0];

	let buffer = vertices.buffer;

	let vao = gl.createVertexArray();
	let vbo = gl.createBuffer();
	gl.bindVertexArray(vao);
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
	gl.bufferData(gl.ARRAY_BUFFER, buffer.byteLength, buffer, gl.DYNAMIC_DRAW);

	let transform = new Matrix4();
	transform.copy(Matrix4.IDENTITY);
	transform.multiply(proj).multiply(view);

	{
		shader_data.setFloat32Array("transform", transform.elements);

		//gl.bindBufferBase(gl.UNIFORM_BUFFER, 4, shader_data.bufferID);
		shader_data.bind();
		shader_data.submit();
	}


	gl.enableVertexAttribArray(0);
	gl.enableVertexAttribArray(1);
	gl.vertexAttribPointer(0, 3, gl.FLOAT, gl.FALSE, 16, 0);
	gl.vertexAttribPointer(1, 4, gl.UNSIGNED_BYTE, gl.TRUE, 16, 12);

	gl.drawArrays(gl.LINES, 0, numVertices);

	gl.bindBuffer(gl.ARRAY_BUFFER, 0);
	gl.deleteBuffers(1, new Uint32Array([vbo]));
	gl.disableVertexAttribArray(0);
	gl.disableVertexAttribArray(1);
	gl.bindVertexArray(0);

};


var renderSphere = function(position, scale, view, proj){

	if(typeof renderSphereShader === "undefined"){
		let vsPath = "../../resources/shaders/lines.vs";
		let fsPath = "../../resources/shaders/lines.fs";
		let shader = new Shader([
			{type: gl.VERTEX_SHADER, path: vsPath},
			{type: gl.FRAGMENT_SHADER, path: fsPath},
		]);
		shader.watch();

		renderSphereShader = shader;
	}
	let shader = renderSphereShader;
	let shader_data = shader.uniformBlocks.shader_data;

	gl.useProgram(shader.program);
	
	
	let color = 0xFF00FFFF;

	let points = [];
	for(let i = 0; i < 1000; i++){
		let x = (Math.random() - 0.5);
		let y = (Math.random() - 0.5);
		let z = (Math.random() - 0.5);
		let l = Math.sqrt(x * x + y * y + z * z);

		x = scale * (x / l) + position.x;
		y = scale * (y / l) + position.y;
		z = scale * (z / l) + position.z;

		points.push(x, y, z, color);
	}
	let numVertices = points.length / 4;

	let vertices = new Float32Array(points);

	let buffer = vertices.buffer;

	let vao = gl.createVertexArray();
	let vbo = gl.createBuffer();
	gl.bindVertexArray(vao);
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
	gl.bufferData(gl.ARRAY_BUFFER, buffer.byteLength, buffer, gl.DYNAMIC_DRAW);

	let transform = new Matrix4();
	transform.copy(Matrix4.IDENTITY);
	transform.multiply(proj).multiply(view);

	{
		shader_data.setFloat32Array("transform", transform.elements);

		//gl.bindBufferBase(gl.UNIFORM_BUFFER, 4, shader_data.bufferID);
		shader_data.bind();
		shader_data.submit();
	}

	gl.enableVertexAttribArray(0);
	gl.enableVertexAttribArray(1);
	gl.vertexAttribPointer(0, 3, gl.FLOAT, gl.FALSE, 16, 0);
	gl.vertexAttribPointer(1, 4, gl.UNSIGNED_BYTE, gl.TRUE, 16, 12);

	gl.drawArrays(gl.POINTS, 0, numVertices);

	gl.bindBuffer(gl.ARRAY_BUFFER, 0);
	gl.deleteBuffers(1, new Uint32Array([vbo]));
	gl.disableVertexAttribArray(0);
	gl.disableVertexAttribArray(1);
	gl.bindVertexArray(0);

};

var renderDefault = function(node, view, proj, target){

	//log("lala");
	//return;

	let buffers = node.getComponents(GLBuffer);
	let material = node.getComponent(GLMaterial, {or: GLMaterial.DEFAULT});
	let shader = material.shader;
	let shader_data = shader.uniformBlocks.shader_data;

	//log(shader.program);

	let transform = new Matrix4();

	let world = node.world;

	transform.copy(Matrix4.IDENTITY);
	transform.multiply(proj).multiply(view).multiply(world);

	if(shader_data){
		shader_data.setFloat32Array("transform", transform.elements);
		shader_data.setFloat32Array("world", world.elements);
		shader_data.setFloat32Array("view", view.elements);
		shader_data.setFloat32Array("proj", proj.elements);
		//shader_data.setFloat32Array("screenSize", new Float32Array([cam.size.width, cam.size.height]));
		shader_data.setFloat32("time", now());


		//gl.bindBufferBase(gl.UNIFORM_BUFFER, 4, shader_data.bufferID);
		shader_data.bind();
		shader_data.submit();

	}

	if(material.texture !== null && shader.uniforms.uTexture !== undefined){

		let texture = material.texture;

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(texture.type, texture.handle);
		gl.uniform1i(shader.uniforms.uTexture, 0);
	}

	if(material.depthTest){
		gl.enable(gl.DEPTH_TEST);
	}else{
		gl.disable(gl.DEPTH_TEST);
	}

	if(material.depthWrite){
		gl.depthMask(true);
	}else{
		gl.depthMask(false);
	}

	for(let buffer of buffers){
		gl.bindVertexArray(buffer.vao);

		// FIXME shouldn't need to bind/unbind vbo from vao. trying bc. of compute shader
		gl.bindBuffer(gl.ARRAY_BUFFER, buffer.vbo);

		
		
		if(buffer.indirect){

			gl.bindBuffer(gl.DRAW_INDIRECT_BUFFER, buffer.indirect.ssbo);

			gl.drawArraysIndirect(material.glDrawMode, 0);

			gl.bindBuffer(gl.DRAW_INDIRECT_BUFFER, 0);

		}else{

			if(node.name === "skybox"){
				// gl.drawArrays(gl.TRIANGLES, 0, buffer.count);
			}else {	
				gl.drawArrays(material.glDrawMode, 0, buffer.count);
			}

		}

		// FIXME shouldn't need to bind/unbind vbo from vao. trying bc. of compute shader
		gl.bindBuffer(gl.ARRAY_BUFFER, 0);
	}

	gl.disable(gl.BLEND);

	//if(node.boundingBoxWorld){
	//	renderBox(node.boundingBoxWorld, view, proj, target);
	//}
}

var renderBuffers = function(view, proj, target){


	let start = now();

	
	let stack = [scene.root];

	let renderNode = (node) => {

		let state = getRenderState();
		
		let material = node.getComponent(GLMaterial, {or: GLMaterial.DEFAULT});
		let shader = material.shader;
		let shader_data = shader.uniformBlocks.shader_data;

		gl.useProgram(shader.program);
		// log(node.name)

		if(node instanceof PointCloudOctree){
			renderPointCloudOctree(node, view, proj, target);
		}
		else if(node instanceof PointCloudProgressive){


			if(typeof renderBenchmark !== "undefined"){
				renderBenchmark(node, view, proj, target);
			}else if(typeof renderDebug !== "undefined"){
				renderDebug(node, view, proj, target);
			}else{
				
				// recent
				// renderPointCloudCompute(node, view, proj, target);
				// render_compute_earlyDepth(node, view, proj, target);
				// render_compute_ballot(node, view, proj, target);
				// render_compute_ballot_earlyDepth(node, view, proj, target);
				// renderComputeHQS(node, view, proj, target); 
				// renderComputeHQS_fill(node, view, proj, target); 
				// renderComputeHQS_fill_2(node, view, proj, target); 
				// renderComputeJustSet(node, view, proj, target); 
				// renderComputeHQS_1x64bit(node, view, proj, target);
				// renderPointCloudBasic(node, view, proj, target);
				// render_compute_uint16(node, view, proj, target);
				// render_compute_uint13(node, view, proj, target);

				// older or might not work in this context
				// renderComputeLL(node, view, proj, target);
				// renderPointCloudProgressive(node, view, proj, target);
				// renderDefault(node, view, proj, target);
			}
		}
		else if(node instanceof PointCloudBasic){
			//renderPointCloudCompute(node, view, proj, target);
			//renderPointCloudProgressive(node, view, proj, target);

			//log(node.transform.elements)

			//log(node.name);
			renderDefault(node, view, proj, target);
		}
		else{

			if(typeof node.render !== "undefined"){
				node.render(view, proj, target);
			}else{
				renderDefault(node, view, proj, target);
			}
		}

		gl.bindTexture(gl.TEXTURE_2D, 0);
	};

	while(stack.length > 0){
		let node = stack.pop();

		if(!node.visible){
			continue;
		}

		// log(node.name)

		renderNode(node);

		//stack.push(...node.children);
		for(let i = node.children.length - 1; i >= 0; i--){
			stack.push(node.children[i]);
		}
	}

	for(let command of scene.drawQueue){
		if(command.name === "drawLines"){
			renderLines(command.lines, view, proj);
		}else if(command.name === "drawSphere"){
			renderSphere(command.position, command.scale, view, proj);
		}else if(command.name === "drawBox"){
			renderBox(command.box, view, proj);
		}else if(command.name === "drawNode"){
			renderNode(command.node);
		}
	}

	gl.depthMask(true);

	let duration = now() - start;
	let durationMS = (duration * 1000).toFixed(3);
	setDebugValue("duration.cp.renderBuffers", `${durationMS}ms`);
	//setDebugValue("#nodes", `${numNodes}`);
	//setDebugValue("#points", `${numPoints}`);

}



if( typeof getRenderState === "undefined"){

	getRenderState = () => {

		if( typeof renderState === "undefined"){

			let csDrawImage;
			{ // distribution shader
				let path = "../../resources/shaders/drawImage.cs";
				
				let shader = new Shader([{type: gl.COMPUTE_SHADER, path: path}]);
				shader.watch();

				csDrawImage = shader;
			}

			let cursorTexture;
			{
				let data = new Uint8Array(4 * 64 * 64);
				let texture = new GLTexture(64, 64, data);

				gl.bindTexture(gl.TEXTURE_2D, texture.handle);

				gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP);
				gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP);

				gl.bindTexture(gl.TEXTURE_2D, 0);

				cursorTexture = texture;
			}

			renderState = {csDrawImage, cursorTexture};
		}
	
		return renderState;
	};
}

var render = function(){

	// GLTimestamp("render-start");
	let start = now();

	let state = getRenderState();

	if(vr.isActive()){

	}else{
		camera.updateMatrixWorld();
	}

	frameCount++;

	camera.aspect = window.width / window.height;
	camera.updateProjectionMatrix();

	gl.disable(gl.VERTEX_PROGRAM_POINT_SIZE);

	for(let listener of listeners.render){
		listener();
	}

	if(vr.isActive()){
		renderVR();
	}else{
		renderRegular();
	}	

	scene.drawQueue = [];

	let duration = now() - start;
	let durationMS = (duration * 1000).toFixed(3);
	setDebugValue("duration.cp.render", `${durationMS}ms`);

	


}


"render.js"