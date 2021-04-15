
getRenderBasicState = function(target){


	if(typeof renderBasicMap === "undefined"){
		renderBasicMap = new Map();
	}

	if(!renderBasicMap.has(target)){

		let shader;
		{ // normal point cloud material 
			let vsPath = "../../resources/shaders/pointcloud_basic.vs";
			let fsPath = "../../resources/shaders/pointcloud.fs";
			
			shader = new Shader([
				{type: gl.VERTEX_SHADER, path: vsPath},
				{type: gl.FRAGMENT_SHADER, path: fsPath},
			]);
			shader.watch();
		}


		let state = {
			shader: shader,
		};

		renderBasicMap.set(target, state);
	}

	return renderBasicMap.get(target);
};

renderPointCloudBasic = function(pointcloud, view, proj, target){

	GLTimestamp("GL_POINTS-start");

	let state = getRenderBasicState(target);
	let shader = state.shader;
	let shader_data = shader.uniformBlocks.shader_data;

	let transform = new Matrix4();
	let world = pointcloud.transform;
	transform.copy(Matrix4.IDENTITY);
	transform.multiply(proj).multiply(view).multiply(world);

	gl.enable(gl.DEPTH_TEST);
	gl.depthMask(true);

	shader_data.setFloat32Array("transform", transform.elements);

	shader_data.bind();
	shader_data.submit();
	
	{ // single/few large calls
		gl.useProgram(shader.program);

		// let pointsLeft = pointcloud.numPoints;
		// let pointsLeft = 1000 * 1000;
		// let batchSize = 134 * 1000 * 1000;

		// GL_POINTS
		for(let buffer of pointcloud.glBuffers){
			gl.bindVertexArray(buffer.vao);
			let numPoints = buffer.count;

			gl.drawArrays(gl.POINTS, 0, numPoints);
		}

		gl.bindVertexArray(0);
	}


	// { // render smaller chunks
	// 	gl.useProgram(shader.program);

	// 	let pointsLeft = pointcloud.numPoints;
	// 	let batchSize = 134 * 1000 * 1000;

	// 	let numPointList = [
	// 		123, 3124, 6521, 403, 23012, 341, 7631, 2010, 30531, 230, 4310
	// 	];
	// 	let i = 0;

	// 	for(let buffer of pointcloud.glBuffers){
			
	// 		gl.bindVertexArray(buffer.vao);

	// 		let numPointsInBuffer = Math.max(Math.min(pointsLeft, batchSize), 0);
	// 		let numPointsInBufferLeft = numPointsInBuffer;
	// 		let numPointsInBufferRendered = 0;

	// 		while(numPointsInBufferLeft > 0){
	// 			// let numPoints = Math.max(Math.min(pointsLeft, batchSize), 0);
	// 			let numPoints = numPointList[i % numPointList.length];
	// 			numPoints = Math.min(numPoints, numPointsInBufferLeft);

	// 			gl.drawArrays(gl.POINTS, numPointsInBufferRendered, numPoints);

	// 			numPointsInBufferLeft -= numPoints;
	// 			numPointsInBufferRendered += numPoints;
	// 			i++;
	// 		}

	// 		pointsLeft = pointsLeft - batchSize;
	// 	}

	// 	gl.bindVertexArray(0);
	// }
	
	gl.useProgram(0);

	GLTimestamp("GL_POINTS-end");

	state.round++;


};

"render_pointcloud_basic.js"