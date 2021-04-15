function renderComputeLL(node, view, proj, target){

	GLTimerQueries.mark("render-compute-ll-start");

	// TODO support resizing
	let width = 3000;
	let height = 2000;

	if(typeof computeLLState === "undefined"){

		let pathRender = `${rootDir}/modules/compute_ll/render.cs`;
		let pathResolve = `${rootDir}/modules/compute_ll/resolve.cs`;

		let csRender = new Shader([{type: gl.COMPUTE_SHADER, path: pathRender}]);
		let csResolve = new Shader([{type: gl.COMPUTE_SHADER, path: pathResolve}]);

		csRender.watch();
		csResolve.watch();

		let numPixels = width * height; // TODO support resizing
		let numNodes = 100 * 1000 * 1000;
		let nodeSize = 12;

		let framebuffer = new ArrayBuffer(numPixels * 8);

		let ssboFramebuffer = gl.createBuffer();
		let ssDepth = gl.createBuffer();
		let ssHeader = gl.createBuffer();
		let ssNodes = gl.createBuffer();
		let ssMeta = gl.createBuffer();

		gl.namedBufferData(ssboFramebuffer, framebuffer.byteLength, framebuffer, gl.DYNAMIC_DRAW);
		gl.namedBufferData(ssDepth, numPixels * 4, 0, gl.DYNAMIC_DRAW);
		gl.namedBufferData(ssHeader, numPixels * 4, 0, gl.DYNAMIC_DRAW);
		gl.namedBufferData(ssNodes, numNodes * nodeSize, 0, gl.DYNAMIC_DRAW);
		gl.namedBufferData(ssMeta, 4, 0, gl.DYNAMIC_DRAW);


		let fbo = new Framebuffer();

		computeLLState = {
			csRender: csRender,
			csResolve: csResolve,
			numPixels: numPixels,
			ssboFramebuffer: ssboFramebuffer,
			ssDepth: ssDepth,
			ssHeader: ssHeader,
			ssNodes: ssNodes,
			ssMeta: ssMeta,
			fbo: fbo,
		};
	}

	let {csRender, csResolve, ssboFramebuffer, fbo} = computeLLState;
	let {ssDepth, ssHeader, ssNodes, ssMeta} = computeLLState;

	//fbo.setSize(target.width * 2, target.height * 2);
	fbo.setSize(target.width, target.height);

	let mat32 = new Float32Array(16);
	let transform = new Matrix4();
	let world = node.transform;
	transform.copy(Matrix4.IDENTITY);
	transform.multiply(proj).multiply(view).multiply(world);
	mat32.set(transform.elements);

	{ // RENDER / BUILD LL

		GLTimerQueries.mark("render-compute-ll-renderpass-start");

		gl.bindFramebuffer(gl.FRAMEBUFFER, 0);

		gl.useProgram(csRender.program);

		//log(transform32);
		gl.uniformMatrix4fv(csRender.uniforms.uTransform, 1, gl.FALSE, mat32);

		//gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, ssboFramebuffer);
		//gl.bindImageTexture(0, target.textures[0], 0, gl.FALSE, 0, gl.READ_WRITE, gl.RGBA8UI);

		gl.uniform2i(csRender.uniforms.uImageSize, target.width, target.height);

		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 2, ssDepth);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 4, ssHeader);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 5, ssNodes);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 6, ssMeta);

		{ // gradient texture
			gl.activeTexture(gl.TEXTURE0);
			gl.bindTexture(gradientTexture.type, gradientTexture.handle);
			if(csRender.uniforms.uGradient){
				gl.uniform1i(csRender.uniforms.uGradient, 0);
			}
		}

		let pointsLeft = node.numPoints;
		let batchSize = 134 * 1000 * 1000;

		for(let buffer of node.glBuffers){
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 0, buffer.vbo);

			let numPoints = Math.max(Math.min(pointsLeft, batchSize), 0);
			let groups = parseInt(numPoints / 128);
			//groups = 300;
			gl.dispatchCompute(groups, 1, 1);
			gl.memoryBarrier(gl.ALL_BARRIER_BITS);

			pointsLeft = pointsLeft - batchSize;
		}

		gl.useProgram(0);
		GLTimerQueries.mark("render-compute-ll-renderpass-end");
		GLTimerQueries.measure("render.compute.ll.render", "render-compute-ll-renderpass-start", "render-compute-ll-renderpass-end");
	}

	{ // RESOLVE
		GLTimerQueries.mark("render-compute-ll-resolvepass-start");
		gl.useProgram(csResolve.program);

		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 2, ssDepth);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 4, ssHeader);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 5, ssNodes);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 6, ssMeta);

		gl.bindImageTexture(0, fbo.textures[0], 0, gl.FALSE, 0, gl.READ_WRITE, gl.RGBA8UI);

		{ // gradient texture
			gl.activeTexture(gl.TEXTURE1);
			gl.bindTexture(gradientTexture.type, gradientTexture.handle);

			if(csResolve.uniforms.uGradient){
				//log("abc");
				gl.uniform1i(csResolve.uniforms.uGradient, 1);
			}
		}

		for(let buffer of node.glBuffers){
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 0, buffer.vbo);

			//let numPoints = Math.max(Math.min(pointsLeft, batchSize), 0);
			let numPoints = node.numPoints;
			let groups = parseInt(numPoints / 128);
			//groups = 300;
			//gl.dispatchCompute(groups, 1, 1);

			//pointsLeft = pointsLeft - batchSize;
		}


		let groups = [
			parseInt(1 + fbo.width / 16),
			parseInt(1 + fbo.height / 16),
			1
		];

		gl.memoryBarrier(gl.ALL_BARRIER_BITS);
		gl.dispatchCompute(...groups);
		gl.memoryBarrier(gl.ALL_BARRIER_BITS);

		gl.useProgram(0);
		GLTimerQueries.mark("render-compute-ll-resolvepass-end");
		GLTimerQueries.measure("render.compute.ll.resolve", "render-compute-ll-resolvepass-start", "render-compute-ll-resolvepass-end");
	}
	
	gl.blitNamedFramebuffer(fbo.handle, target.handle, 
		0, 0, fbo.width, fbo.height, 
		0, 0, target.width, target.height, 
		gl.COLOR_BUFFER_BIT, gl.LINEAR);

	GLTimerQueries.mark("render-compute-ll-end");
	GLTimerQueries.measure("render.compute.ll", "render-compute-ll-start", "render-compute-ll-end");

};

"render compute ll";