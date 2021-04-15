
function renderComputeJustSet(node, view, proj, target){

	GLTimestamp("compute-set-start");

	// TODO support resizing
	let width = 3000;
	let height = 2000;

	if(typeof computeJustSetState === "undefined"){

		let pathRender = `${rootDir}/modules/compute_just_set/render.cs`;
		let pathResolve = `${rootDir}/modules/compute_just_set/resolve.cs`;

		let csRender = new Shader([{type: gl.COMPUTE_SHADER, path: pathRender}]);
		let csResolve = new Shader([{type: gl.COMPUTE_SHADER, path: pathResolve}]);

		csRender.watch();
		csResolve.watch();

		let numPixels = width * height; // TODO support resizing
		let framebuffer = new ArrayBuffer(numPixels * 8);

		let ssboFramebuffer = gl.createBuffer();
		gl.namedBufferData(ssboFramebuffer, framebuffer.byteLength, framebuffer, gl.DYNAMIC_DRAW);

		let fbo = new Framebuffer();

		computeJustSetState = {
			csRender: csRender,
			csResolve: csResolve,
			numPixels: numPixels,
			ssboFramebuffer: ssboFramebuffer,
			fbo: fbo,
		};
	}

	let csRender = computeJustSetState.csRender;
	let csResolve = computeJustSetState.csResolve;
	let ssboFramebuffer = computeJustSetState.ssboFramebuffer;
	let fbo = computeJustSetState.fbo;

	fbo.setSize(target.width, target.height);

	let mat32 = new Float32Array(16);
	let transform = new Matrix4();
	let world = node.transform;
	transform.copy(Matrix4.IDENTITY);
	transform.multiply(proj).multiply(view).multiply(world);
	mat32.set(transform.elements);


	{ // RENDER

		GLTimestamp("compute-set-render-start");

		gl.bindFramebuffer(gl.FRAMEBUFFER, 0);

		gl.useProgram(csRender.program);

		gl.uniformMatrix4fv(csRender.uniforms.uTransform, 1, gl.FALSE, mat32);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, ssboFramebuffer);
		let {width, height} = fbo;
		gl.uniform2i(csRender.uniforms.uImageSize, width, height);
		
		for(let buffer of node.glBuffers){
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 0, buffer.vbo);

			let numPoints = buffer.count;
			let groups = Math.ceil(numPoints / 128);

			gl.dispatchCompute(groups, 1, 1);
		}

		gl.useProgram(0);
		GLTimestamp("compute-set-render-end");
	}

	gl.memoryBarrier(gl.ALL_BARRIER_BITS);

	{ // RESOLVE
		GLTimestamp("compute-set-resolve-start");
		gl.useProgram(csResolve.program);

		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, ssboFramebuffer);
		gl.bindImageTexture(0, fbo.textures[0], 0, gl.FALSE, 0, gl.READ_WRITE, gl.RGBA8UI);

		let groups = [
			parseInt(1 + fbo.width / 16),
			parseInt(1 + fbo.height / 16),
			1
		];

		gl.dispatchCompute(...groups);

		gl.useProgram(0);
		GLTimestamp("compute-set-resolve-end");
	}

	gl.memoryBarrier(gl.ALL_BARRIER_BITS);
	
	gl.blitNamedFramebuffer(fbo.handle, target.handle, 
		0, 0, fbo.width, fbo.height, 
		0, 0, target.width, target.height, 
		gl.COLOR_BUFFER_BIT, gl.LINEAR);

	GLTimestamp("compute-set-end");

}




renderPointCloudCompute = renderComputeBasic;

"render compute"