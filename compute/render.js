
function renderComputeBasic(node, view, proj, target){

	GLTimerQueries.mark("render-compute-start");

	// TODO support resizing
	let width = 3000;
	let height = 2000;

	if(typeof computeState === "undefined"){

		let pathRender = `${rootDir}/modules/compute/render.cs`;
		let pathResolve = `${rootDir}/modules/compute/resolve.cs`;

		let csRender = new Shader([{type: gl.COMPUTE_SHADER, path: pathRender}]);
		let csResolve = new Shader([{type: gl.COMPUTE_SHADER, path: pathResolve}]);

		csRender.watch();
		csResolve.watch();

		let numPixels = width * height; // TODO support resizing
		let framebuffer = new ArrayBuffer(numPixels * 8);

		let ssboFramebuffer = gl.createBuffer();
		gl.namedBufferData(ssboFramebuffer, framebuffer.byteLength, framebuffer, gl.DYNAMIC_DRAW);

		let fbo = new Framebuffer();

		computeState = {
			csRender: csRender,
			csResolve: csResolve,
			numPixels: numPixels,
			ssboFramebuffer: ssboFramebuffer,
			fbo: fbo,
		};
	}

	let csRender = computeState.csRender;
	let csResolve = computeState.csResolve;
	let ssboFramebuffer = computeState.ssboFramebuffer;
	let fbo = computeState.fbo;

	//fbo.setSize(target.width * 2, target.height * 2);
	fbo.setSize(target.width, target.height);

	//let buffer = node.getComponent(GLBuffer);
	//let buffer = node.glBuffers[0];
	//let buffer = node.buffer;
	//let numPoints = Math.min(node.numPoints, 134 * 1000 * 1000);

	//log(node.numPoints)

	let mat32 = new Float32Array(16);
	let transform = new Matrix4();
	let world = node.transform;
	transform.copy(Matrix4.IDENTITY);
	transform.multiply(proj).multiply(view).multiply(world);
	mat32.set(transform.elements);


	if(true){ // RENDER

		GLTimerQueries.mark("render-compute-renderpass-start");

		gl.bindFramebuffer(gl.FRAMEBUFFER, 0);

		gl.useProgram(csRender.program);

		//log(transform32);
		gl.uniformMatrix4fv(csRender.uniforms.uTransform, 1, gl.FALSE, mat32);

		const e = transform.elements;
		//gl.uniform4d(csRender.uniforms.uDepthLine, e[12], e[13], e[14], e[15]);
		if(csRender.uniforms.uDepthLine){
			gl.uniform4d(csRender.uniforms.uDepthLine, e[3], e[7], e[11], e[15]);
		}
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, ssboFramebuffer);
		//gl.bindImageTexture(0, target.textures[0], 0, gl.FALSE, 0, gl.READ_WRITE, gl.RGBA8UI);

		{
			gl.activeTexture(gl.TEXTURE0);
			gl.bindTexture(gradientTexture.type, gradientTexture.handle);
			if(csRender.uniforms.uGradient){
				gl.uniform1i(csRender.uniforms.uGradient, 0);
			}
		}

		let pointsLeft = node.numPoints;
		let batchSize = 134 * 1000 * 1000;

		//gl.disable(gl.DEPTH_TEST);

		for(let buffer of node.glBuffers){
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 0, buffer.vbo);

			let {width, height} = fbo;
			gl.uniform2i(csRender.uniforms.uImageSize, width, height);

			let numPoints = Math.max(Math.min(pointsLeft, batchSize), 0);
			let groups = parseInt(numPoints / 128);
			//groups = 300;

			//groups = frameCount % groups;
			//log(numPoints);

			gl.dispatchCompute(groups, 1, 1);

			pointsLeft = pointsLeft - batchSize;
		}

		gl.useProgram(0);
		GLTimerQueries.mark("render-compute-renderpass-end");
		GLTimerQueries.measure("render.compute.regular.render", "render-compute-renderpass-start", "render-compute-renderpass-end");
	}

	{ // RESOLVE
		GLTimerQueries.mark("render-compute-resolvepass-start");
		gl.useProgram(csResolve.program);

		//gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 0, buffer.vbo);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, ssboFramebuffer);
		gl.bindImageTexture(0, fbo.textures[0], 0, gl.FALSE, 0, gl.READ_WRITE, gl.RGBA8UI);

		{
			gl.activeTexture(gl.TEXTURE1);
			gl.bindTexture(gradientTexture.type, gradientTexture.handle);

			//log(csResolve.uniforms.uGradient);

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

		gl.useProgram(0);
		GLTimerQueries.mark("render-compute-resolvepass-end");
		GLTimerQueries.measure("render.compute.regular.resolve", "render-compute-resolvepass-start", "render-compute-resolvepass-end");
	}
	
	gl.blitNamedFramebuffer(fbo.handle, target.handle, 
		0, 0, fbo.width, fbo.height, 
		0, 0, target.width, target.height, 
		gl.COLOR_BUFFER_BIT, gl.LINEAR);

	GLTimerQueries.mark("render-compute-end");
	GLTimerQueries.measure("render.compute.regular", "render-compute-start", "render-compute-end");

}




renderPointCloudCompute = renderComputeBasic;

"render compute"