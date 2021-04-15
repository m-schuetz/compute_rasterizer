


if( typeof getRenderVRState === "undefined"){

	getRenderVRState = () => {

		if( typeof renderVRState === "undefined"){

			let fboLeft = new Framebuffer();
			let fboRight = new Framebuffer();
			let fboResolveLeft = new Framebuffer();
			let fboResolveRight = new Framebuffer();

			fboLeft.setNumColorAttachments(2);
			fboRight.setNumColorAttachments(2);

			let fboEDL = new Framebuffer();

			let vsPath = "../../resources/shaders/edl.vs";
			let fsPath = "../../resources/shaders/edl.fs";
			let fsPathMSAA = "../../resources/shaders/edlMSAA.fs";

			let edlShaderMSAA = new Shader([
				{type: gl.VERTEX_SHADER, path: vsPath},
				{type: gl.FRAGMENT_SHADER, path: fsPathMSAA},
			]);
			edlShaderMSAA.watch();

			let edlShader = new Shader([
				{type: gl.VERTEX_SHADER, path: vsPath},
				{type: gl.FRAGMENT_SHADER, path: fsPath},
			]);
			edlShader.watch();

			renderVRState = {
				fboEDL: fboEDL,
				edlShader: edlShader,
				edlShaderMSAA: edlShaderMSAA,
				fboLeft: fboLeft,
				fboRight: fboRight,
				fboResolveLeft: fboResolveLeft,
				fboResolveRight: fboResolveRight,
			};
		}

		return renderVRState;
	}

}

var renderVR = function(){

	let start = now();

	let state = getRenderVRState();

	let {fboLeft, fboRight} = state;
	let {fboResolveLeft, fboResolveRight} = state;

	let size = vr.getRecommmendedRenderTargetSize();

	fboLeft.setSize(size.width, size.height);
	fboRight.setSize(size.width, size.height);
	fboResolveLeft.setSize(size.width, size.height);
	fboResolveRight.setSize(size.width, size.height);

	let samples = MSAA_SAMPLES;
	fboLeft.setSamples(samples);
	fboRight.setSamples(samples);


	//let st1 = now();
	vr.updatePose();
	vr.processEvents();
	//let td = (1000 *  (now() - st1)).toFixed(3);
	//log(td);


	GLTimestamp("render-vr-start");
	let startWithoutWait = now();

	let hmdPose = new Matrix4().set(vr.getHMDPose());
	let leftPose = new Matrix4().set(vr.getLeftEyePose());
	let rightPose = new Matrix4().set(vr.getRightEyePose());

	gl.clipControl(gl.LOWER_LEFT, gl.ZERO_TO_ONE);
	gl.clearDepth(0);
	gl.enable(gl.DEPTH_TEST);
	gl.depthFunc(gl.GREATER);

	gl.disable(gl.BLEND);
	//gl.blendFunc(gl.ONE, gl.ONE);

	let [near, far] = [0.1, 1000];
	let leftProj = new Matrix4().set(vr.getLeftProjection(near, far));
	let rightProj = new Matrix4().set(vr.getRightProjection(near, far));

	{ // LEFT
		gl.bindFramebuffer(gl.FRAMEBUFFER, fboLeft.handle);
		gl.viewport(0, 0, fboLeft.width, fboLeft.height);
		//gl.clearColor(1, 1, 1, 1);
		gl.clearColor(0, 0, 0, 0);
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

		let view = new Matrix4().multiplyMatrices(hmdPose, leftPose).getInverse();
		let proj = leftProj;

		renderBuffers(view, proj, fboLeft);

	}

	if(EDL_ENABLED){

		GLTimestamp("edl-left-start");
		let fbo = fboLeft;

		let isquad = $("image_space_quad");
		let buffer = isquad.getComponent(GLBuffer);

		let samples = fbo.samples;

		let shader = (samples === 1) ? state.edlShader : state.edlShaderMSAA;
		let shader_data = shader.uniformBlocks.shader_data;

		gl.bindFramebuffer(gl.FRAMEBUFFER, fboResolveLeft.handle);

		gl.useProgram(shader.program);

		//let textureType = gl.TEXTURE_2D_MULTISAMPLE;
		let textureType = (samples === 1) ? gl.TEXTURE_2D : gl.TEXTURE_2D_MULTISAMPLE;
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(textureType, fbo.textures[0]);
		gl.uniform1i(shader.uniforms.uColor, 0);

		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(textureType, fbo.depth);
		gl.uniform1i(shader.uniforms.uDepth, 1);

		gl.disable(gl.DEPTH_TEST);
		gl.depthMask(false);
		gl.disable(gl.CULL_FACE);

		shader_data.setFloat32Array("screenSize", new Float32Array([fboResolveLeft.width, fboResolveLeft.height]));
		shader_data.setFloat32("time", now());
		shader_data.setFloat32("near", near);
		shader_data.setFloat32("far", far);
		shader_data.setFloat32("edlStrength", 0.1);
		shader_data.setFloat32("msaaSampleCount", fbo.samples);

		shader_data.bind();
		shader_data.submit();

		gl.bindVertexArray(buffer.vao);
		gl.drawArrays(gl.TRIANGLES, 0, buffer.count);
		gl.bindVertexArray(0);

		GLTimestamp("edl-left-end");

		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(true);

	}else{

		gl.blitNamedFramebuffer(fboLeft.handle, fboResolveLeft.handle, 
			0, 0, fboLeft.width, fboLeft.height, 
			0, 0, fboResolveLeft.width, fboResolveLeft.height, 
			gl.COLOR_BUFFER_BIT, gl.LINEAR);

	}

	{ // RIGHT
		gl.bindFramebuffer(gl.FRAMEBUFFER, fboRight.handle);
		gl.viewport(0, 0, fboRight.width, fboRight.height);
		gl.clear(gl.COLOR_BUFFER_BIT  | gl.DEPTH_BUFFER_BIT);


		let view = new Matrix4().multiplyMatrices(hmdPose, rightPose).getInverse();
		let proj = rightProj;
		
		renderBuffers(view, proj, fboRight);
	}

	
	if(EDL_ENABLED){

		GLTimestamp("edl-right-start");
		let fbo = fboRight;

		let isquad = $("image_space_quad");
		let buffer = isquad.getComponent(GLBuffer);

		let samples = fbo.samples;

		let shader = (samples === 1) ? state.edlShader : state.edlShaderMSAA;
		let shader_data = shader.uniformBlocks.shader_data;

		gl.bindFramebuffer(gl.FRAMEBUFFER, fboResolveRight.handle);

		gl.useProgram(shader.program);

		let textureType = (samples === 1) ? gl.TEXTURE_2D : gl.TEXTURE_2D_MULTISAMPLE;
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(textureType, fbo.textures[0]);
		gl.uniform1i(shader.uniforms.uColor, 0);

		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(textureType, fbo.depth);
		gl.uniform1i(shader.uniforms.uDepth, 1);

		gl.disable(gl.DEPTH_TEST);
		gl.depthMask(false);
		gl.disable(gl.CULL_FACE);

		shader_data.setFloat32Array("screenSize", new Float32Array([fboResolveRight.width, fboResolveRight.height]));
		shader_data.setFloat32("time", now());
		shader_data.setFloat32("near", near);
		shader_data.setFloat32("far", far);
		shader_data.setFloat32("edlStrength", 0.1);
		shader_data.setFloat32("msaaSampleCount", fbo.samples);

		shader_data.bind();
		shader_data.submit();

		gl.bindVertexArray(buffer.vao);
		gl.drawArrays(gl.TRIANGLES, 0, buffer.count);
		gl.bindVertexArray(0);

		GLTimestamp("edl-right-end");

		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(true);

	}else{

		gl.blitNamedFramebuffer(fboRight.handle, fboResolveRight.handle, 
			0, 0, fboRight.width, fboRight.height, 
			0, 0, fboResolveRight.width, fboResolveRight.height, 
			gl.COLOR_BUFFER_BIT, gl.LINEAR);

	}

	vr.submit(fboResolveLeft.textures[0], fboResolveRight.textures[0]);

	vr.postPresentHandoff();

	
	gl.blitNamedFramebuffer(fboResolveLeft.handle, 0, 
		0, 0, fboResolveLeft.width, fboResolveLeft.height, 
		0, 0, window.width + 10, window.height, 
		gl.COLOR_BUFFER_BIT, gl.LINEAR);


	GLTimestamp("render-vr-end");

	// let st1 = now();
	// vr.updatePose();
	// let td = (1000 *  (now() - st1)).toFixed(3);
	// log(td);
	// vr.processEvents();

	//gl.blitNamedFramebuffer(fboResolveRight.handle, 0, 
	//	0, 0, fboResolveRight.width, fboResolveRight.height, 
	//	window.width / 2, 0, window.width, window.height, 
	//	gl.COLOR_BUFFER_BIT, gl.LINEAR);


	{
		let durationFull = now() - start;
		let durationFullMS = (durationFull * 1000).toFixed(3);
		setDebugValue("duration.renderVR", `${durationFullMS}ms`);

		let durationBare = now() - startWithoutWait;
		let durationBareMS = (durationBare * 1000).toFixed(3);
		setDebugValue("duration.renderVR (w/o wait sync)", `${durationBareMS}ms`);
	}


}


"render_vr.js"