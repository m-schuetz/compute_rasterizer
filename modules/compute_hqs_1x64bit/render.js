
function renderComputeHQS_1x64bit(node, view, proj, target){
	GLTimestamp("compute-hqs1x64bit-start");

	// TODO support resizing
	let width = 3000;
	let height = 2000;

	if(typeof computeHQS_1x64State === "undefined"){

		let pathRenderDepth = `${rootDir}/modules/compute_hqs_1x64bit/render_depth.cs`;
		let pathRenderAttribute = `${rootDir}/modules/compute_hqs_1x64bit/render_attribute.cs`;
		let pathResolve = `${rootDir}/modules/compute_hqs_1x64bit/resolve.cs`;

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

		computeHQS_1x64State = {
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

	let {csRenderDepth, csRenderAttribute, csResolve} = computeHQS_1x64State;
	let {ssboDepthbuffer, ssboFramebuffer, ssRG, ssBA} = computeHQS_1x64State;
	let {fbo} = computeHQS_1x64State;

	fbo.setSize(target.width, target.height);

	let mat32 = new Float32Array(16);
	let transform = new Matrix4();
	let world = node.transform;
	transform.copy(Matrix4.IDENTITY);
	transform.multiply(proj).multiply(view).multiply(world);
	mat32.set(transform.elements);

	{ // RENDER DEPTH

		GLTimestamp("compute-hqs1x64bit-depth-start");

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
		GLTimestamp("compute-hqs1x64bit-depth-end");
	}

	gl.memoryBarrier(gl.ALL_BARRIER_BITS);

	{ // RENDER ATTRIBUTE

		GLTimestamp("compute-hqs1x64bit-attribute-start");

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
		GLTimestamp("compute-hqs1x64bit-attribute-end");
	}

	gl.memoryBarrier(gl.ALL_BARRIER_BITS);

	{ // RESOLVE
		GLTimestamp("compute-hqs1x64bit-resolve-start");
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
		GLTimestamp("compute-hqs1x64bit-resolve-end");
	}

	gl.memoryBarrier(gl.ALL_BARRIER_BITS);
	
	gl.blitNamedFramebuffer(fbo.handle, target.handle, 
		0, 0, fbo.width, fbo.height, 
		0, 0, target.width, target.height, 
		gl.COLOR_BUFFER_BIT, gl.LINEAR);

	// {
	// 	let [x, y] = [775, 495];
	// 	let [width, height] = [150, 100];
	// 	let multiplier = 8;
	// 	gl.blitNamedFramebuffer(fbo.handle, target.handle, 
	// 		x, y, x + width, y + height,
	// 		10, 10, width * multiplier, height * multiplier, 
	// 		gl.COLOR_BUFFER_BIT, gl.NEAREST);
	// }

	GLTimestamp("compute-hqs1x64bit-end");

};


"render compute hqs";