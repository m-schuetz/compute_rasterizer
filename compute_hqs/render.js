
function renderComputeHQS(node, view, proj, target){
	GLTimerQueries.mark("render-compute-start");

	// TODO support resizing
	let width = 3000;
	let height = 2000;

	if(typeof computeHQSState === "undefined"){

		let pathRenderDepth = `${rootDir}/modules/compute_hqs/render_depth.cs`;
		let pathRenderAttribute = `${rootDir}/modules/compute_hqs/render_attribute.cs`;
		let pathResolve = `${rootDir}/modules/compute_hqs/resolve.cs`;

		let csRenderDepth = new Shader([{type: gl.COMPUTE_SHADER, path: pathRenderDepth}]);
		let csRenderAttribute = new Shader([{type: gl.COMPUTE_SHADER, path: pathRenderAttribute}]);
		let csResolve = new Shader([{type: gl.COMPUTE_SHADER, path: pathResolve}]);

		csRenderDepth.watch();
		csRenderAttribute.watch();
		csResolve.watch();

		let numPixels = width * height; // TODO support resizing
		//let framebuffer = new ArrayBuffer(numPixels * 8);

		let ssboDepthbuffer = gl.createBuffer();
		let ssboFramebuffer = gl.createBuffer();
		let ssRG = gl.createBuffer();
		let ssBA = gl.createBuffer();
		gl.namedBufferData(ssboDepthbuffer, numPixels * 8, 0, gl.DYNAMIC_DRAW);
		gl.namedBufferData(ssboFramebuffer, numPixels * 8, 0, gl.DYNAMIC_DRAW);
		gl.namedBufferData(ssRG, numPixels * 8, 0, gl.DYNAMIC_DRAW);
		gl.namedBufferData(ssBA, numPixels * 8, 0, gl.DYNAMIC_DRAW);

		let fbo = new Framebuffer();

		computeHQSState = {
			csRenderDepth: csRenderDepth,
			csRenderAttribute: csRenderAttribute,
			csResolve: csResolve,
			numPixels: numPixels,
			ssboFramebuffer: ssboFramebuffer,
			ssboDepthbuffer: ssboDepthbuffer,
			ssRG: ssRG,
			ssBA: ssBA,
			fbo: fbo,
		};
	}

	let {csRenderDepth, csRenderAttribute, csResolve} = computeHQSState;
	let {ssboDepthbuffer, ssboFramebuffer, ssRG, ssBA} = computeHQSState;
	let {fbo} = computeHQSState;

	GLTimerQueries.mark("render-compute-hqs-start");

	fbo.setSize(target.width, target.height);

	let mat32 = new Float32Array(16);
	let transform = new Matrix4();
	let world = node.transform;
	transform.copy(Matrix4.IDENTITY);
	transform.multiply(proj).multiply(view).multiply(world);
	mat32.set(transform.elements);

	{ // RENDER DEPTH

		GLTimerQueries.mark("render-compute-hqs-depth-start");

		gl.bindFramebuffer(gl.FRAMEBUFFER, 0);

		gl.useProgram(csRenderDepth.program);

		gl.uniformMatrix4fv(csRenderDepth.uniforms.uTransform, 1, gl.FALSE, mat32);

		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, ssboDepthbuffer);

		let pointsLeft = node.numPoints;
		let batchSize = 134 * 1000 * 1000;


		for(let buffer of node.glBuffers){
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 0, buffer.vbo);

			let {width, height} = fbo;
			gl.uniform2i(csRenderDepth.uniforms.uImageSize, width, height);

			let numPoints = Math.max(Math.min(pointsLeft, batchSize), 0);
			let groups = parseInt(numPoints / 128);

			//log(numPoints);
			//groups = 300;
			gl.dispatchCompute(groups, 1, 1);

			pointsLeft = pointsLeft - batchSize;
		}

		gl.useProgram(0);
		GLTimerQueries.mark("render-compute-hqs-depth-end");
		GLTimerQueries.measure("render.compute.hqs.depth", "render-compute-hqs-depth-start", "render-compute-hqs-depth-end");
	}

	{ // RENDER ATTRIBUTE

		GLTimerQueries.mark("render-compute-hqs-attribute-start");

		gl.bindFramebuffer(gl.FRAMEBUFFER, 0);

		gl.useProgram(csRenderAttribute.program);

		gl.uniformMatrix4fv(csRenderAttribute.uniforms.uTransform, 1, gl.FALSE, mat32);
		
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, ssboFramebuffer);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 2, ssboDepthbuffer);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 3, ssRG);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 4, ssBA);

		{
			gl.activeTexture(gl.TEXTURE0);
			gl.bindTexture(gradientTexture.type, gradientTexture.handle);
			if(csRenderAttribute.uniforms.uGradient){
				gl.uniform1i(csRenderAttribute.uniforms.uGradient, 0);
			}
		}

		let pointsLeft = node.numPoints;
		let batchSize = 134 * 1000 * 1000;

		for(let buffer of node.glBuffers){
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 0, buffer.vbo);

			let {width, height} = fbo;
			gl.uniform2i(csRenderAttribute.uniforms.uImageSize, width, height);

			let numPoints = Math.max(Math.min(pointsLeft, batchSize), 0);
			let groups = parseInt(numPoints / 128);
			//groups = 300;
			gl.dispatchCompute(groups, 1, 1);

			pointsLeft = pointsLeft - batchSize;
		}

		gl.useProgram(0);
		GLTimerQueries.mark("render-compute-hqs-attribute-end");
		GLTimerQueries.measure("render.compute.hqs.attribute", "render-compute-hqs-attribute-start", "render-compute-hqs-attribute-end");
	}

	{ // RESOLVE
		GLTimerQueries.mark("render-compute-hqs-resolve-start");
		gl.useProgram(csResolve.program);

		//gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 0, buffer.vbo);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, ssboFramebuffer);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 2, ssboDepthbuffer);
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

		gl.useProgram(0);
		GLTimerQueries.mark("render-compute-hqs-resolve-end");
		GLTimerQueries.measure("render.compute.hqs.resolve", "render-compute-hqs-resolve-start", "render-compute-hqs-resolve-end");
	}
	
	gl.blitNamedFramebuffer(fbo.handle, target.handle, 
		0, 0, fbo.width, fbo.height, 
		0, 0, target.width, target.height, 
		gl.COLOR_BUFFER_BIT, gl.LINEAR);

	GLTimerQueries.mark("render-compute-hqs-end");
	GLTimerQueries.measure("render.compute.hqs", "render-compute-hqs-start", "render-compute-hqs-end");

};


"render compute hqs";