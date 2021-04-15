
fillToggle = 0;

dynamicFillBudgetEnabled = true;

getRenderProgressiveState = function(target){


	if(typeof renderProgressiveMap === "undefined"){
		renderProgressiveMap = new Map();
	}

	if(!renderProgressiveMap.has(target)){
		// const tStart = now();

		let ssIndirectCommand = gl.createBuffer();
		let ssFillFixedCommands = gl.createBuffer();
		let ssFillRemainingCommands = gl.createBuffer();
		let icBytes = 5 * 4;
		gl.namedBufferData(ssIndirectCommand, icBytes, new ArrayBuffer(icBytes), gl.DYNAMIC_DRAW);
		gl.namedBufferData(ssFillFixedCommands, 10 * icBytes, new ArrayBuffer(10 * icBytes), gl.DYNAMIC_DRAW);
		gl.namedBufferData(ssFillRemainingCommands, 10 * icBytes, new ArrayBuffer(10 * icBytes), gl.DYNAMIC_DRAW);

		let ssFillFixed = gl.createBuffer();
		gl.namedBufferData(ssFillFixed, 64, new ArrayBuffer(64), gl.DYNAMIC_DRAW);

		let ssTimestamps = gl.createBuffer();
		gl.namedBufferData(ssTimestamps, 24, new ArrayBuffer(24), gl.DYNAMIC_DRAW);

		let reprojectBuffer = new GLBuffer();

		let vboCapacity = 30 * 1000 * 1000;
		let vboBytes = vboCapacity * 16;

		let buffer = new ArrayBuffer(vboBytes);
		let attributes = [
			new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
			new GLBufferAttribute("value", 1, 4, gl.INT, gl.FALSE, 4, 12, {targetType: "int"}),
			new GLBufferAttribute("index", 2, 4, gl.INT, gl.FALSE, 4, 16, {targetType: "int"}),
		];
		
		reprojectBuffer.setEmptyInterleaved(attributes, vboBytes);

		let fboPrev = new Framebuffer();

		let csFill = null;
		{ // create time estimation shader
			let path = `${rootDir}/modules/progressive/compute_fill.cs`;
			let shader = new Shader([{type: gl.COMPUTE_SHADER, path: path}]);
			shader.watch();
			csFill = shader;
		}

		let shReproject = null;
		{ // reprojection shader
			let vsPath = `${rootDir}/modules/progressive/reproject.vs`;
			let fsPath = `${rootDir}/modules/progressive/reproject.fs`;

			let shader = new Shader([
				{type: gl.VERTEX_SHADER, path: vsPath},
				{type: gl.FRAGMENT_SHADER, path: fsPath},
			]);
			shader.watch();

			shReproject = shader;
		}

		let shFill = null;
		{ // add shader
			let vsPath = `${rootDir}/modules/progressive/fill.vs`;
			let fsPath = `${rootDir}/modules/progressive/fill.fs`;

			let shader = new Shader([
				{type: gl.VERTEX_SHADER, path: vsPath},
				{type: gl.FRAGMENT_SHADER, path: fsPath},
			]);
			shader.watch();

			shFill = shader;
		}

		let csCreateVBO = null;
		{ // create VBO shader
			//let path = `${rootDir}/modules/progressive/create_vbo.cs`;
			let path = `${rootDir}/modules/progressive/create_vbo_simple_duplicate_prevention.cs`;
			let shader = new Shader([{type: gl.COMPUTE_SHADER, path: path}]);
			shader.watch();
			csCreateVBO = shader;
		}



		let state = {
			ssIndirectCommand: ssIndirectCommand,
			ssTimestamps: ssTimestamps,
			ssFillFixed: ssFillFixed,
			ssFillFixedCommands: ssFillFixedCommands,
			ssFillRemainingCommands: ssFillRemainingCommands,

			csFill: csFill,

			shReproject: shReproject,
			shFill: shFill,
			csCreateVBO: csCreateVBO,

			reprojectBuffer: reprojectBuffer,
			round: 0,
			fillOffset: 0,
			fboPrev: fboPrev,
			pointclouds: new Map(),
		};

		renderProgressiveMap.set(target, state);

		// const tEnd = now();
		// const duration = (tEnd - tStart).toFixed(3);
		// log(duration);
	}

	return renderProgressiveMap.get(target);
};

renderPointCloudProgressive = (function(){

	const reproject = function(target, pointcloud, view, proj){

		let state = getRenderProgressiveState(target);
		let {shReproject, shFill, csCreateVBO} = state;
		let {ssIndirectCommand, ssTimestamps, reprojectBuffer, fboPrev} = state;

		const transform_m32 = new Float32Array(16);
		const transform = new Matrix4();
		const world = pointcloud.transform;
		transform.multiply(proj).multiply(view).multiply(world);
		transform_m32.set(transform.elements);

		GLTimerQueries.mark("render-progressive-reproject-start");
		gl.useProgram(shReproject.program);

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gradientTexture.type, gradientTexture.handle);
		if(shReproject.uniforms.uGradient){
			gl.uniform1i(shReproject.uniforms.uGradient, 0);
		}

		gl.uniformMatrix4fv(shReproject.uniforms.uWorldViewProj, 1, gl.FALSE, transform_m32);

		gl.bindVertexArray(reprojectBuffer.vao);
		gl.bindBuffer(gl.DRAW_INDIRECT_BUFFER, ssIndirectCommand);

		gl.drawArraysIndirect(gl.POINTS, 0);

		gl.bindVertexArray(0);

		GLTimerQueries.mark("render-progressive-reproject-end");
		GLTimerQueries.measure("render.progressive.p1_reproject", "render-progressive-reproject-start", "render-progressive-reproject-end");
	}

	const fillFixed = function(target, pointcloud, view, proj){ 

		let state = getRenderProgressiveState(target);
		let {shReproject, shFill, csCreateVBO} = state;
		let {ssIndirectCommand, ssTimestamps, reprojectBuffer, fboPrev} = state;

		const transform_m32 = new Float32Array(16);
		const transform = new Matrix4();
		const world = pointcloud.transform;
		transform.multiply(proj).multiply(view).multiply(world);
		transform_m32.set(transform.elements);

		GLTimerQueries.mark("render-progressive-add-start");
		gl.useProgram(shFill.program);

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gradientTexture.type, gradientTexture.handle);
		if(shFill.uniforms.uGradient){
			gl.uniform1i(shFill.uniforms.uGradient, 0);
		}

		gl.uniformMatrix4fv(shFill.uniforms.uWorldViewProj, 1, gl.FALSE, transform_m32);

		let buffers = pointcloud.glBuffers;

		if(buffers.length === 1){
			const buffer = buffers[0];

			gl.bindVertexArray(buffer.vao);

			let remainingBudget = 1000 * 1000;
			let leftToEndOfBuffer = (pointcloud.numPoints - state.fillOffset);
			
			{ // draw towards end of buffer
				let count = Math.min(remainingBudget, leftToEndOfBuffer);
				gl.uniform1i(shFill.uniforms.uOffset, 0);
				gl.drawArrays(gl.POINTS, state.fillOffset, count);
				//log(`${state.fillOffset}, ${count}`);
				state.fillOffset = (state.fillOffset + count) % pointcloud.numPoints;
				remainingBudget = remainingBudget - count;

			}

			{ //log(`${state.fillOffset}, ${count}`); draw potential remaining budget from beginning of buffer
				let count = remainingBudget;
				gl.uniform1i(shFill.uniforms.uOffset, 0);
				gl.drawArrays(gl.POINTS, state.fillOffset, count);
				//log(`${state.fillOffset}, ${count}`);
				state.fillOffset += count;

			}

		}else{

			let remainingBudget = 4000 * 1000;
			let maxChunkSize = 134 * 1000 * 1000;
			let buffers = pointcloud.glBuffers;
			let cumChunkOffsets = [0];
			let cumChunkSizes = [buffers[0].count];
			for(let i = 1; i < buffers.length; i++){
				cumChunkOffsets.push(cumChunkOffsets[i - 1] + buffers[i - 1].count);
				cumChunkSizes.push(cumChunkSizes[i - 1] + buffers[i].count);
			}

			while(remainingBudget > 0){

				let offset = state.fillOffset;
				let chunkIndex = parseInt(offset / maxChunkSize) % pointcloud.numPoints;
				let count = Math.min(remainingBudget, cumChunkSizes[chunkIndex] - offset);
				let buffer = buffers[chunkIndex];

				//log(offset + ", " + maxChunkSize + ", " + count + ", " + pointcloud.numPoints);
				//log(cumChunkOffsets[chunkIndex]);

				gl.bindVertexArray(buffer.vao);

				gl.uniform1i(shFill.uniforms.uOffset, cumChunkOffsets[chunkIndex]);
				gl.drawArrays(gl.POINTS, state.fillOffset - cumChunkOffsets[chunkIndex], count);

				remainingBudget -= count;
				state.fillOffset = (state.fillOffset + count) % pointcloud.numPoints;
			}

		}

		gl.bindVertexArray(0);

		GLTimerQueries.mark("render-progressive-add-end");
		GLTimerQueries.measure("render.progressive.p2_fill.render_fixed", "render-progressive-add-start", "render-progressive-add-end");
	}

	const fillDynamic = function(target, pointcloud, view, proj){

		let state = getRenderProgressiveState(target);
		let {shReproject, shFill, csCreateVBO} = state;
		let {ssIndirectCommand, ssTimestamps, reprojectBuffer, fboPrev} = state;

		const transform_m32 = new Float32Array(16);
		const transform = new Matrix4();
		const world = pointcloud.transform;
		transform.multiply(proj).multiply(view).multiply(world);
		transform_m32.set(transform.elements);

		GLTimerQueries.mark("render-progressive-fill-start");

		if(true){ // FILL FIXED
			GLTimerQueries.mark("render-progressive-add-start");
			gl.useProgram(shFill.program);

			{ // start fill timestamp to ssTimestamps
				let qtStart = gl.createQuery();
				gl.queryCounter(qtStart, gl.TIMESTAMP);
				gl.bindBuffer(gl.QUERRY_BUFFER, ssTimestamps);
				gl.getQueryObjectui64Indirect(qtStart, gl.QUERY_RESULT, 8);
				gl.bindBuffer(gl.QUERRY_BUFFER, 0);
				gl.deleteQuery(qtStart);
			}

			gl.activeTexture(gl.TEXTURE0);
			gl.bindTexture(gradientTexture.type, gradientTexture.handle);
			if(shFill.uniforms.uGradient){
				gl.uniform1i(shFill.uniforms.uGradient, 0);
			}

			gl.uniformMatrix4fv(shFill.uniforms.uWorldViewProj, 1, gl.FALSE, transform_m32);

			let buffers = pointcloud.glBuffers;

			if(buffers.length === 1){
				// if remaining budget larger than remaining points to end of buffer:
				// - indirectCommand[0] specifies #points to render to end of buffer
				// - indirectCommand[1] specifies #points to render form start of buffer
				const buffer = buffers[0];

				gl.uniform1i(shFill.uniforms.uOffset, 0);

				gl.bindVertexArray(buffer.vao);
				gl.bindBuffer(gl.DRAW_INDIRECT_BUFFER, state.ssFillFixedCommands);

				gl.drawArraysIndirect(gl.POINTS, 0 * 4 * 4);
				gl.drawArraysIndirect(gl.POINTS, 1 * 4 * 4);
			}else{
				// if point cloud in more than one buffer (>134M points):
				// indirectCommand[i]: range of points of buffer[i] to render
				for(let i = 0; i < buffers.length; i++){
					let buffer = buffers[i];

					gl.uniform1i(shFill.uniforms.uOffset, i * 134 * 1000 * 1000);

					gl.bindVertexArray(buffer.vao);
					gl.bindBuffer(gl.DRAW_INDIRECT_BUFFER, state.ssFillFixedCommands);

					gl.drawArraysIndirect(gl.POINTS, i * 4 * 4);
				}
			}

			gl.bindVertexArray(0);

			{ // end fill timestamp to ssTimestamps
				let qtStart = gl.createQuery();
				gl.queryCounter(qtStart, gl.TIMESTAMP);
				gl.bindBuffer(gl.QUERRY_BUFFER, ssTimestamps);
				gl.getQueryObjectui64Indirect(qtStart, gl.QUERY_RESULT, 16);
				gl.bindBuffer(gl.QUERRY_BUFFER, 0);
				gl.deleteQuery(qtStart);
			}

			GLTimerQueries.mark("render-progressive-add-end");
			GLTimerQueries.measure("render.progressive.p2_fill.render_fixed", "render-progressive-add-start", "render-progressive-add-end");
		}

		// COMPUTE FILL
		{
			// const tStart = now();
			GLTimerQueries.mark("render-progressive-fill-compute-remaining-start");
			const {csFill, ssFillFixed, ssFillFixedCommands, ssFillRemainingCommands} = state;
			
			if(!state.pointclouds.has(pointcloud)){

				let numBatches = pointcloud.glBuffers.length;
				let buffer = new ArrayBuffer((4 + numBatches) * 4);
				let view = new DataView(buffer);

				//log(`NUM POINTS!!! ${pointcloud.numPoints}`);

				view.setInt32(0, pointcloud.numPoints, true);
				view.setInt32(4, 0, true);
				view.setInt32(8, 1 * 1000 * 1000, true);
				view.setInt32(12, pointcloud.glBuffers.length, true);
				for(let i = 0; i < numBatches; i++){
					let buffer = pointcloud.glBuffers[i];
					view.setInt32(16 + i * 4, buffer.count, true);
					//log(buffer.count);
				}
				gl.namedBufferSubData(ssFillFixed, 0, buffer.byteLength, buffer);

				state.pointclouds.set(pointcloud, {});
			}

			gl.useProgram(csFill.program);

			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 0, ssFillFixed);
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, ssFillFixedCommands);
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 2, ssFillRemainingCommands);
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 3, state.ssTimestamps);

			gl.memoryBarrier(gl.ALL_BARRIER_BITS);
			gl.dispatchCompute(1, 1, 1);
			gl.memoryBarrier(gl.ALL_BARRIER_BITS);

			gl.useProgram(0);

			GLTimerQueries.mark("render-progressive-fill-compute-remaining-end");
			GLTimerQueries.measure("render.progressive.p2_fill.compute_remaining", "render-progressive-fill-compute-remaining-start", "render-progressive-fill-compute-remaining-end");

			// const tEnd = now();
			// const duration = ((tEnd - tStart) * 1000).toFixed(3);
			// log(duration);
		}

		// FILL REMAINING
		{
			GLTimerQueries.mark("render-progressive-add-remaining-start");
			gl.useProgram(shFill.program);

			gl.activeTexture(gl.TEXTURE0);
			gl.bindTexture(gradientTexture.type, gradientTexture.handle);
			if(shFill.uniforms.uGradient){
				gl.uniform1i(shFill.uniforms.uGradient, 0);
			}

			gl.uniformMatrix4fv(shFill.uniforms.uWorldViewProj, 1, gl.FALSE, transform_m32);

			let buffers = pointcloud.glBuffers;

			for(let i = 0; i < buffers.length; i++){
				let buffer = buffers[i];

				gl.uniform1i(shFill.uniforms.uOffset, i * 134 * 1000 * 1000);

				gl.bindVertexArray(buffer.vao);
				gl.bindBuffer(gl.DRAW_INDIRECT_BUFFER, state.ssFillRemainingCommands);

				gl.drawArraysIndirect(gl.POINTS, i * 4 * 4);
			}
			

			gl.bindVertexArray(0);

			GLTimerQueries.mark("render-progressive-add-remaining-end");
			GLTimerQueries.measure("render.progressive.p2_fill.render_remaining", "render-progressive-add-remaining-start", "render-progressive-add-remaining-end");
		}

		GLTimerQueries.mark("render-progressive-fill-end");
		GLTimerQueries.measure("render.progressive.p2_fill", "render-progressive-fill-start", "render-progressive-fill-end");
	}


	const createVBO = function(target, pointcloud, view, proj){ 
		
		let state = getRenderProgressiveState(target);
		let {shReproject, shFill, csCreateVBO} = state;
		let {ssIndirectCommand, ssTimestamps, reprojectBuffer, fboPrev} = state;

		//const target = fboPrev;
		GLTimerQueries.mark("render-progressive-ibo-start");

		gl.useProgram(csCreateVBO.program);

		let indirectData = new Uint32Array([0, 1, 0, 0, 0]);
		gl.namedBufferSubData(ssIndirectCommand, 0, indirectData.byteLength, indirectData);

		gl.bindImageTexture(0, target.textures[1], 0, gl.FALSE, 0, gl.READ_WRITE, gl.RGBA8);

		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, ssIndirectCommand);

		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 2, reprojectBuffer.vbo);

		pointcloud.glBuffers.forEach( (buffer, i) => {
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 3 + i, buffer.vbo);
		});


		let localSize = {
			x: 16,
			y: 16,
		};

		let groups = [
			parseInt(1 + target.width / localSize.x),
			parseInt(1 + target.height / localSize.y),
			1
		];

		if(target.samples === 2){
			groups[0] *= 2;
		}else if(target.samples === 4){
			groups[0] *= 2;
			groups[1] *= 2;
		}else if(target.samples === 8){
			groups[0] *= 4;
			groups[1] *= 2;
		}else if(target.samples === 16){
			groups[0] *= 4;
			groups[1] *= 4;
		}

		groups[0] /= 2;
		groups[1] /= 2;

		//log(groups);

		gl.memoryBarrier(gl.ALL_BARRIER_BITS);
		gl.dispatchCompute(...groups);
		gl.memoryBarrier(gl.ALL_BARRIER_BITS);

		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, 0);
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 2, 0);

		pointcloud.glBuffers.forEach( (buffer, i) => {
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 3 + i, 0);
		});
		//gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 3, 0);
		//gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 4, 0);

		GLTimerQueries.mark("render-progressive-ibo-end");
		GLTimerQueries.measure("render.progressive.p3_vbo", "render-progressive-ibo-start", "render-progressive-ibo-end");
		//GLTimerQueries.measure("render.progressive.ibo", "render-progressive-ibo-start", "render-progressive-ibo-end", (duration) => {
		//	let ms = (duration * 1000).toFixed(3);
		//	setDebugValue("gl.render.progressive.ibo", `${ms}ms`);
		//});

	};
	
	
	return function(pointcloud, view, proj, target){

		GLTimerQueries.mark("render-progressive-start");

		let state = getRenderProgressiveState(target);
		let {shReproject, shFill, csCreateVBO} = state;
		let {ssIndirectCommand, ssTimestamps, reprojectBuffer, fboPrev} = state;

		if(!pointcloud){
			return;
		}

		{ // write start timestamp to ssTimestamps
			let qtStart = gl.createQuery();
			gl.queryCounter(qtStart, gl.TIMESTAMP);
			gl.bindBuffer(gl.QUERRY_BUFFER, ssTimestamps);
			gl.getQueryObjectui64Indirect(qtStart, gl.QUERY_RESULT, 0);
			gl.bindBuffer(gl.QUERRY_BUFFER, 0);
			gl.deleteQuery(qtStart);
		}

		{ // second color attachment for indices
			let buffers = new Uint32Array([
				gl.COLOR_ATTACHMENT0, 
				gl.COLOR_ATTACHMENT1,
			]);
			gl.drawBuffers(buffers.length, buffers);
		}
		
		reproject(target, pointcloud, view, proj);
		//fillFixed(target, pointcloud, view, proj);
		fillDynamic(target, pointcloud, view, proj);
		createVBO(target, pointcloud, view, proj);

		if(true){
			gl.memoryBarrier(gl.ALL_BARRIER_BITS);
			// taken from https://stackoverflow.com/questions/2901102/how-to-print-a-number-with-commas-as-thousands-separators-in-javascript
			const numberWithCommas = (x) => {
				return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
			}

			let resultBuffer = new ArrayBuffer(10 * 5 * 4);
			gl.getNamedBufferSubData(state.ssFillFixedCommands, 0, resultBuffer.byteLength, resultBuffer);
			let view = new DataView(resultBuffer);

			let estimate = view.getUint32(5 * 16, true);

			if(typeof estimates === "undefined"){
				estimates = [];
			}
			estimates.push({
				estimate: estimate,
				timestamp: now(),
			});

			//log(estimates.length);
			estimates = estimates.filter(e => e.timestamp > now() - 1);
			//log(estimates.length);

			const values = estimates.map(e => e.estimate);
			const sum = values.reduce( (a, i) => a + i, 0);
			const max = Math.max(...values);
			const min = Math.min(...values);
			const median = estimates.length > 0 ? values.sort()[Math.ceil(estimates.length / 2)] : Infinity;
			const mean = sum / estimates.length;

			const sMin = addCommas(parseInt(min));
			const sMax = addCommas(parseInt(max));
			const sMean = addCommas(parseInt(mean));
			const sMedian = addCommas(parseInt(median));
			const msg = `{"mean": ${sMean}, "min": ${sMin}, "max": ${sMax}, "median": ${sMedian}}`;
			//log();
			setDebugValue("progressive dyn budget", msg);

			//log(numberWithCommas(estimate));
		}

		{
			const format = "${reproject}\t${fillFixed}\t${fillBudget}\t${fillRemaining}\t${fill}\t${vbo}\t${progressive}";
			const html = `</pre>
			<script>
			function copyProgressive(){

				const progressive = JSON.parse(getEntry("gl.render.progressive")).mean;
				const reproject = JSON.parse(getEntry("gl.render.progressive.p1_reproject")).mean;
				const fill = JSON.parse(getEntry("gl.render.progressive.p2_fill")).mean;
				const fillFixed = JSON.parse(getEntry("gl.render.progressive.p2_fill.render_fixed")).mean;
				const fillRemaining = JSON.parse(getEntry("gl.render.progressive.p2_fill.render_remaining")).mean;
				const fillBudget = JSON.parse(getEntry("progressive dyn budget")).mean;
				const vbo = JSON.parse(getEntry("gl.render.progressive.p3_vbo")).mean;
				const msg = \`${format}\`;

				clipboardCopy(msg);
			}
			</script>
			<input type="button" value="copy benchmark to clipboard" onclick='copyProgressive()'></input>
			<pre>`;

			setDebugValue("z.bench.progressive", html);


		}
		
		gl.useProgram(0);

		GLTimerQueries.mark("render-progressive-end");
		GLTimerQueries.measure("render.progressive", "render-progressive-start", "render-progressive-end");
	}

})();

"render_progressive.js"