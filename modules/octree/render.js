

drawBoxes = function(nodes){

	// 12 lines, 2 vertices each line, for each node/box
	let numVertices = nodes.length * 12 * 2;
	let vertices = new Float32Array(numVertices * 4);
	let u32 = new Uint32Array(vertices);
	let color = 0xFF00FFFF;

	// uint32 bits to float bits, since we're feeding a float buffer
	color = new Float32Array(new Uint32Array([color]).buffer)[0];

	for(let i = 0; i < nodes.length; i++){
		let node = nodes[i];
		let box = node.boundingBox;

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

		vertices.set(data, i * 12 * 2 * 4);

	}

	let buffer = vertices.buffer;

	let vao = gl.createVertexArray();
	let vbo = gl.createBuffer();
	gl.bindVertexArray(vao);
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
	gl.bufferData(gl.ARRAY_BUFFER, buffer.byteLength, buffer, gl.DYNAMIC_DRAW);

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

//if(typeof renderPointCloudOctreeState === "undefined")
{
	let maxNodes = 4000;
	let bytesPerNode = 16 * 4 + 16 * 4 + 16 * 4 + 16 + 16;

	let octreeData = new ArrayBuffer(12);
	let nodeData = new ArrayBuffer(maxNodes * bytesPerNode);
	let hierarchyData = new ArrayBuffer(maxNodes * 4);

	let ssOctreeData = gl.createBuffer();
	let ssNodeData = gl.createBuffer();
	let ssVisibleHierarchyData = gl.createBuffer();

	gl.namedBufferData(ssOctreeData, octreeData.byteLength, octreeData, gl.DYNAMIC_DRAW);
	gl.namedBufferData(ssNodeData, nodeData.byteLength, nodeData, gl.DYNAMIC_DRAW);
	gl.namedBufferData(ssVisibleHierarchyData, maxNodes * 4, hierarchyData, gl.DYNAMIC_DRAW);

	renderPointCloudOctreeState = {
		maxNodes: maxNodes,
		bytesPerNode: bytesPerNode,

		octreeData: octreeData,
		nodeData: nodeData,
		hierarchyData: hierarchyData,

		ssOctreeData: ssOctreeData,
		ssNodeData: ssNodeData,
		ssVisibleHierarchyData: ssVisibleHierarchyData,
	};
}

renderPointCloudOctreeBasic = function(pointcloud, view, proj){

	GLTimerQueries.mark("render-octree-start");

	let material = pointcloud.getComponent(GLMaterial, {or: GLMaterial.DEFAULT});
	let shader = material.shader;
	let shader_data = shader.uniformBlocks.shader_data;

	gl.useProgram(shader.program);

	let worldViewProj = new Float32Array(16);
	let worldView = new Float32Array(16);
	let world = pointcloud.transform;

	{
		let transform = new Matrix4();
		transform.copy(Matrix4.IDENTITY);
		transform.multiply(proj).multiply(view).multiply(world);
		worldViewProj.set(transform.elements);
	}

	{
		let transform = new Matrix4();
		transform.copy(Matrix4.IDENTITY);
		transform.multiply(view).multiply(world);
		worldView.set(transform.elements);
	}

	let visibleHierarchyData = pointcloud.computeVisibleHierarchyData(pointcloud.visibleNodes);

	let state = renderPointCloudOctreeState;
	for(let i = 0; i < pointcloud.visibleNodes.length; i++){
		let node = pointcloud.visibleNodes[i];
		let ssNodeDataOffset = i * state.bytesPerNode;

		let ndView = new Float32Array(state.nodeData);


		// worldViewProj
		ndView.set(worldViewProj, ssNodeDataOffset / 4);

		// worldView
		ndView.set(worldView, ssNodeDataOffset / 4 + 16);

		// world
		ndView.set(world.elements, ssNodeDataOffset / 4 + 32);
		//log(world.elements);

		// offset
		let bb = node.boundingBox;
		ndView[ssNodeDataOffset / 4 + 48 + 0] = bb.min.x;
		ndView[ssNodeDataOffset / 4 + 48 + 1] = bb.min.y;
		ndView[ssNodeDataOffset / 4 + 48 + 2] = bb.min.z;
		ndView[ssNodeDataOffset / 4 + 48 + 3] = 0;

		// numPoints
		new Uint32Array(state.nodeData)[ssNodeDataOffset / 4 + 48 + 4] = node.buffer.count;

		// level
		new Uint32Array(state.nodeData)[ssNodeDataOffset / 4 + 48 + 5] = node.name.length - 1;

		// vnStart
		new Uint32Array(state.nodeData)[ssNodeDataOffset / 4 + 48 + 6] = visibleHierarchyData.offsets.get(node);
	}

	let boxSize = pointcloud.root.boundingBox.getSize().x;

	//let octreeData = new ArrayBuffer(8);
	let octreeDataView = new DataView(state.octreeData);
	octreeDataView.setFloat32(0, boxSize, true);
	octreeDataView.setInt32(4, 1, true);
	octreeDataView.setInt32(8, 0, true);
	octreeDataView.setInt32(8, 0, true);
	
	octreeDataView.setInt32(8, 1, true);
	gl.namedBufferSubData(state.ssOctreeData, 0, state.octreeData.byteLength, state.octreeData);
	gl.namedBufferSubData(state.ssNodeData, 0, pointcloud.visibleNodes.length * state.bytesPerNode, state.nodeData);
	gl.namedBufferSubData(state.ssVisibleHierarchyData, 0, visibleHierarchyData.data.byteLength, visibleHierarchyData.data);

	gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, state.ssNodeData);
	gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 2, state.ssVisibleHierarchyData);
	gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 3, state.ssOctreeData);

	{

		let world = pointcloud.transform;

		let transform = new Matrix4();

		//log(world.elements);
		transform.copy(Matrix4.IDENTITY);
		transform.multiply(proj).multiply(view).multiply(world);

		let mat32 = new Float32Array(16);

		mat32.set(proj.elements);
		gl.uniformMatrix4fv(/*shader.uniforms.uProj*/ 4, 1, gl.FALSE, mat32);

		
		
		gl.uniform2f(/*shader.uniforms.uScreenSize*/ 6, camera.size.width, camera.size.height);
	}

	{
		let v1 = new Vector3(0, 0, 0).applyMatrix4(world);
		let v2 = new Vector3(1, 1, 1).normalize().applyMatrix4(world);

		let scale = v1.distanceTo(v2);

		gl.uniform1f(/*shader.uniforms.uScale*/ 33, scale);
	}

	gl.uniform1f(/*shader.uniforms.uSpacing*/ 20, pointcloud.spacing);
	//gl.uniform1f(/*shader.uniforms.uSpacing*/ 20, 2);

	gl.uniform1f(/*shader.uniforms.uMinMilimeters*/ 60, pointcloud.minMilimeters);
	gl.uniform1f(/*shader.uniforms.uMinMilimeters*/ 60, 0.02);
	gl.uniform1f(/*shader.uniforms.uPointSize*/ 61, pointcloud.pointSize);

	if(USER_STUDY_BLENDING){
		gl.uniform1f(/*shader.uniforms.uColorMultiplier*/ 71, 0.06);
	}else{
		gl.uniform1f(/*shader.uniforms.uColorMultiplier*/ 71, 1.0);
	}

	if(USER_STUDY_OCTREE_MODE === "FIXED"){
		gl.uniform1i(/*shader.uniforms.uSizeMode*/ 75, 0);
		gl.uniform1i(/*shader.uniforms.uSizeMode*/ 77, USER_STUDY_OCTREE_POINT_SIZE);
		//gl.uniform1i(/*shader.uniforms.uSizeMode*/ 77, 4);
	}else{
		gl.uniform1i(/*shader.uniforms.uSizeMode*/ 75, 1);
		//gl.uniform1i(/*shader.uniforms.uSizeMode*/ 77, 2);
	}

	gl.uniform1f(/*shader.uniforms.uUSTW*/ 74, USER_STUDY_TW);

	gl.activeTexture(gl.TEXTURE0);
	gl.bindTexture(gradientTexture.type, gradientTexture.handle);
	gl.uniform1i(shader.uniforms.uGradient, 0);
	

	let pointsRendered = 0;
	let nodesRendered = 0;
	let i = 0; 
	for(let node of pointcloud.visibleNodes){

		let buffer = node.buffer;

		gl.uniform1i(shader.uniforms.uNodeIndex, i);

		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 4, node.buffer.vbos.get("whatever"));

		gl.bindVertexArray(buffer.vao);

		gl.drawArrays(material.glDrawMode, 0, buffer.count);

		pointsRendered += buffer.count;
		nodesRendered++;

		node.transform = pointcloud.transform;
		node.glBuffers = [node.buffer];
		//renderPointCloudCompute(node, view, proj, fbo);

		i++;
	}
	
	GLTimerQueries.mark("render-octree-end");

	setDebugValue("#nodes rendered", addCommas(nodesRendered));
	setDebugValue("#points rendered", addCommas(pointsRendered));

	gl.useProgram(0);


};



renderPointCloudOctreeCompute = function(pointcloud, view, proj, target){


	if(typeof octreeComputeState === "undefined"){

		let pathRender =  `${rootDir}/modules/compute/render.cs`;
		let pathResolve = `${rootDir}/modules/compute/resolve.cs`;

		let csRender = new Shader([{type: gl.COMPUTE_SHADER, path: pathRender}]);
		let csResolve = new Shader([{type: gl.COMPUTE_SHADER, path: pathResolve}]);

		csRender.watch();
		csResolve.watch();

		let [width, height] = [3000, 2000];
		let numPixels = width * height; // TODO support resizing
		let framebuffer = new ArrayBuffer(numPixels * 8);

		let ssboFramebuffer = gl.createBuffer();
		gl.namedBufferData(ssboFramebuffer, framebuffer.byteLength, framebuffer, gl.DYNAMIC_DRAW);

		let fbo = new Framebuffer();

		octreeComputeState = {
			csRender: csRender,
			csResolve: csResolve,
			numPixels: numPixels,
			ssboFramebuffer: ssboFramebuffer,
			fbo: fbo,
		};
	}

	let csRender = octreeComputeState.csRender;
	let csResolve = octreeComputeState.csResolve;
	let ssboFramebuffer = octreeComputeState.ssboFramebuffer;
	let fbo = octreeComputeState.fbo;

	fbo.setSize(target.width, target.height);

	let mat32 = new Float32Array(16);
	let transform = new Matrix4();
	let world = pointcloud.transform;
	transform.copy(Matrix4.IDENTITY);
	transform.multiply(proj).multiply(view).multiply(world);
	mat32.set(transform.elements);

	{ // RENDER PASS

		gl.bindFramebuffer(gl.FRAMEBUFFER, 0);
		gl.useProgram(csRender.program);

		gl.uniformMatrix4fv(csRender.uniforms.uTransform, 1, gl.FALSE, mat32);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, ssboFramebuffer);

		{
			gl.activeTexture(gl.TEXTURE0);
			gl.bindTexture(gradientTexture.type, gradientTexture.handle);
			if(csRender.uniforms.uGradient){
				gl.uniform1i(csRender.uniforms.uGradient, 0);
			}
		}

		let pointsRendered = 0;
		let nodesRendered = 0;
		let i = 0; 
		for(let node of pointcloud.visibleNodes){

			let buffer = node.buffer;

			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 0, buffer.vbo);

			let {width, height} = fbo;
			gl.uniform2i(csRender.uniforms.uImageSize, width, height);

			let numPoints = buffer.count;
			let groups = parseInt(numPoints / 128);

			gl.dispatchCompute(groups, 1, 1);
		}
	}

	{ // RESOLVE
		gl.useProgram(csResolve.program);

		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, ssboFramebuffer);
		gl.bindImageTexture(0, fbo.textures[0], 0, gl.FALSE, 0, gl.READ_WRITE, gl.RGBA8UI);

		{
			gl.activeTexture(gl.TEXTURE1);
			gl.bindTexture(gradientTexture.type, gradientTexture.handle);

			if(csResolve.uniforms.uGradient){
				log("abc");
				gl.uniform1i(csResolve.uniforms.uGradient, 1);
			}
		}

		let groups = [
			parseInt(1 + fbo.width / 16),
			parseInt(1 + fbo.height / 16),
			1
		];

		gl.dispatchCompute(...groups);
	}


	gl.useProgram(0);

	gl.blitNamedFramebuffer(fbo.handle, target.handle, 
		0, 0, fbo.width, fbo.height, 
		0, 0, target.width, target.height, 
		gl.COLOR_BUFFER_BIT, gl.LINEAR);


};

//renderPointCloudOctree = renderPointCloudOctreeBasic;
renderPointCloudOctree = renderPointCloudOctreeCompute;

"render_pointcloud_octree.js"