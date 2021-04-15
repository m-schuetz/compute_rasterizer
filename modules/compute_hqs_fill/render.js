
function renderComputeHQS_fill(node, view, proj, target){

	// TODO support resizing
	let width = 3000;
	let height = 2000;

	if(typeof computeHQSFillState === "undefined"){

		let pathRenderDepth = `${rootDir}/modules/compute_hqs_fill/render_depth.cs`;
		let pathRenderAttribute = `${rootDir}/modules/compute_hqs_fill/render_attribute.cs`;
		let pathResolve = `${rootDir}/modules/compute_hqs_fill/resolve.cs`;

		let csRenderDepth = new Shader([{type: gl.COMPUTE_SHADER, path: pathRenderDepth}]);
		let csRenderAttribute = new Shader([{type: gl.COMPUTE_SHADER, path: pathRenderAttribute}]);
		let csResolve = new Shader([{type: gl.COMPUTE_SHADER, path: pathResolve}]);

		csRenderDepth.watch();
		csRenderAttribute.watch();
		csResolve.watch();

		let numPixels = width * height; // TODO support resizing

		let ssboDepthbuffer = gl.createBuffer();
		let ssRGBA = gl.createBuffer();
		gl.namedBufferData(ssboDepthbuffer, numPixels * 8, 0, gl.DYNAMIC_COPY);
		gl.namedBufferData(ssRGBA, numPixels * 16, 0, gl.DYNAMIC_COPY);

		let fbo = new Framebuffer();
		// let fboDepth = new Framebuffer();
		let fboDilated = new Framebuffer();

		let texDepth = gl.createTexture();
		{
			gl.bindTexture(gl.TEXTURE_2D, texDepth);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

			gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32UI, width, height, 0, gl.RED_INTEGER, gl.UNSIGNED_INT, 0);
		}

		let vsPath = `${rootDir}/modules/compute_hqs_fill/fill.vs`;
		let fsPath = `${rootDir}/modules/compute_hqs_fill/fill.fs`;

		let fillShader = new Shader([
			{type: gl.VERTEX_SHADER, path: vsPath},
			{type: gl.FRAGMENT_SHADER, path: fsPath},
		]);
		fillShader.watch();

		computeHQSFillState = {
			csRenderDepth: csRenderDepth,
			csRenderAttribute: csRenderAttribute,
			csResolve: csResolve,
			numPixels: numPixels,
			ssboDepthbuffer: ssboDepthbuffer,
			ssRGBA,
			fbo, texDepth, fboDilated, fillShader
		};
	}

	let {csRenderDepth, csRenderAttribute, csResolve} = computeHQSFillState;
	let {ssboDepthbuffer, ssRGBA} = computeHQSFillState;
	let {fbo, fboDilated, texDepth} = computeHQSFillState;

	GLTimestamp("compute-hqs_fill-start");

	fbo.setSize(target.width, target.height);
	// fboDepth.setSize(target.width, target.height);
	fboDilated.setSize(target.width, target.height);

	let mat32 = new Float32Array(16);
	let transform = new Matrix4();
	let world = node.transform;
	transform.copy(Matrix4.IDENTITY);
	transform.multiply(proj).multiply(view).multiply(world);
	mat32.set(transform.elements);

	{ // RENDER DEPTH

		GLTimestamp("compute-hqs_fill-depth-start");

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
		GLTimestamp("compute-hqs_fill-depth-end");
	}

	{ // RENDER ATTRIBUTE

		GLTimestamp("compute-hqs_fill-attribute-start");

		gl.bindFramebuffer(gl.FRAMEBUFFER, 0);

		gl.useProgram(csRenderAttribute.program);

		gl.uniformMatrix4fv(csRenderAttribute.uniforms.uTransform, 1, gl.FALSE, mat32);
		
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 2, ssboDepthbuffer);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 3, ssRGBA);

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
		GLTimestamp("compute-hqs_fill-attribute-end");
	}

	{ // RESOLVE
		GLTimestamp("compute-hqs_fill-resolve-start");
		gl.useProgram(csResolve.program);

		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 2, ssboDepthbuffer);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 3, ssRGBA);

		gl.bindImageTexture(0, fbo.textures[0], 0, gl.FALSE, 0, gl.READ_WRITE, gl.RGBA8UI);
		gl.bindImageTexture(1, texDepth, 0, gl.FALSE, 0, gl.READ_WRITE, gl.R32UI);

		{
			gl.activeTexture(gl.TEXTURE1);
			gl.bindTexture(gradientTexture.type, gradientTexture.handle);

			if(csResolve.uniforms.uGradient){
				log("abc");
				gl.uniform1i(csResolve.uniforms.uGradient, 1);
			}
		}


		// let groups = [
		// 	parseInt(1 + fbo.width / 16),
		// 	parseInt(1 + fbo.height / 16),
		// 	1
		// ];

		let groups = Math.floor((fbo.width * fbo.height) / 32);
		gl.dispatchCompute(groups, 1, 1);

		gl.useProgram(0);
		GLTimestamp("compute-hqs_fill-resolve-end");
	}

	{ // DILATE
		GLTimestamp("compute-hqs_fill-fill-start");

		let shader = computeHQSFillState.fillShader;
		let shader_data = shader.uniformBlocks.shader_data;

		gl.bindFramebuffer(gl.FRAMEBUFFER, fboDilated.handle);

		gl.clearColor(1.0, 0.0, 0.0, 1.0);
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

		gl.useProgram(shader.program);

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, fbo.textures[0]);
		gl.uniform1i(shader.uniforms.uColor, 0);

		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, texDepth);
		gl.uniform1i(shader.uniforms.uDepth, 1);

		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 2, ssboDepthbuffer);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 3, ssRGBA);

		gl.disable(gl.DEPTH_TEST);
		gl.depthMask(false);
		gl.disable(gl.CULL_FACE);

		// log(camera.width);

		// shader_data.setFloat32Array("screenSize", new Float32Array([1000, 800]));
		// shader_data.bind();
		// shader_data.submit();

		let isquad = $("image_space_quad");
		let buffer = isquad.getComponent(GLBuffer);

		gl.bindVertexArray(buffer.vao);
		gl.drawArrays(gl.TRIANGLES, 0, buffer.count);
		gl.bindVertexArray(0);

		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(true);

		gl.bindFramebuffer(gl.FRAMEBUFFER, 0);

		GLTimestamp("compute-hqs_fill-fill-end");
	}

	{ // clear buffers
		let cDepth = new Uint32Array([0xFFFFFFFF]);
		let cRGB = new Uint32Array([0]);
		gl.clearNamedBufferData(ssboDepthbuffer, gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT, cDepth);
		gl.clearNamedBufferData(ssRGBA, gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT, cRGB);
	}
	
	// gl.blitNamedFramebuffer(fbo.handle, target.handle, 
	// 	0, 0, fbo.width, fbo.height, 
	// 	0, 0, target.width, target.height, 
	// 	gl.COLOR_BUFFER_BIT, gl.LINEAR);

	gl.blitNamedFramebuffer(fboDilated.handle, target.handle, 
		0, 0, fboDilated.width, fboDilated.height, 
		0, 0, target.width, target.height, 
		gl.COLOR_BUFFER_BIT, gl.LINEAR);

	// { // closeup
	// 	let sx = 965;
	// 	let sy = 420;
	// 	let w = 70;
	// 	gl.blitNamedFramebuffer(fboDilated.handle, target.handle, 
	// 		sx, sy, sx + w, sy + w, 
	// 		10, 10, 600, 600, 
	// 		gl.COLOR_BUFFER_BIT, gl.NEAREST);
	// }

	// { // closeup
	// 	let sx = 715;
	// 	let sy = 420;
	// 	let w = 70;
	// 	gl.blitNamedFramebuffer(fboDilated.handle, target.handle, 
	// 		sx, sy, sx + w, sy + w, 
	// 		10, 610, 600, 1210, 
	// 		gl.COLOR_BUFFER_BIT, gl.NEAREST);
	// }

	GLTimestamp("compute-hqs_fill-end");

};


"render compute hqs";