
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
			let path = `${rootDir}/modules/progressive/create_vbo.cs`;
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

			fillQueries: null,
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

	const fill = function(target, pointcloud, view, proj){ 

		let state = getRenderProgressiveState(target);
		let {shReproject, shFill, csCreateVBO} = state;
		let {ssIndirectCommand, ssTimestamps, reprojectBuffer, fboPrev} = state;

		const transform_m32 = new Float32Array(16);
		const transform = new Matrix4();
		const world = pointcloud.transform;
		transform.multiply(proj).multiply(view).multiply(world);
		transform_m32.set(transform.elements);


		if(state.fillQueries === null){
			state.fillQueries =  {
				count: 1000000,
				start: gl.createQuery(),
				end: gl.createQuery(),
			};

			log(1394823042);
			log(state.fillQueries.start);

			gl.queryCounter(state.fillQueries.start, gl.TIMESTAMP);
			gl.queryCounter(state.fillQueries.end, gl.TIMESTAMP);
		}

		let fillQueries = state.fillQueries;
		let spawnNewQueries = false;
		{ // compute previous fill time, estimate budget

			let startAvailable = gl.getQueryObjectui64(fillQueries.start, gl.QUERY_RESULT_AVAILABLE) === gl.TRUE;
			let endAvailable = gl.getQueryObjectui64(fillQueries.end, gl.QUERY_RESULT_AVAILABLE) === gl.TRUE;


			if(startAvailable && endAvailable){

				let tStart = gl.getQueryObjectui64(fillQueries.start, gl.QUERY_RESULT);
				let tEnd = gl.getQueryObjectui64(fillQueries.end, gl.QUERY_RESULT);

				let nanos = tEnd - tStart;
				let millies = nanos / 1000000;

				let fillRate = fillQueries.count / millies;
				let timeBudget = 10;
				let newCount = fillRate * timeBudget;

				setDebugValue("progressive fill count", addCommas(parseInt(newCount)));

				if(newCount > 200 * 1000 * 1000){
					// just a safeguard in case something goes wrong with the estimate.
					// let's not render more than 100M points in a single frame.
				}else{
					fillQueries.count = newCount;
				}

				spawnNewQueries = true;
			}

			if(spawnNewQueries){
				gl.queryCounter(fillQueries.start, gl.TIMESTAMP);
			}

			
		}


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

			//let remainingBudget = 1000 * 1000;
			//let remainingBudget = fillQueries.count;
			let remainingBudget = fillQueries.count;
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

			//let remainingBudget = 4000 * 1000;
			let remainingBudget = fillQueries.count;
			//log(remainingBudget);
			remainingBudget = Math.min(200 * 1000 * 1000, remainingBudget);
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

		if(spawnNewQueries){
			gl.queryCounter(fillQueries.end, gl.TIMESTAMP);
		}

		GLTimerQueries.mark("render-progressive-add-end");
		GLTimerQueries.measure("render.progressive.p2_fill.render_fixed", "render-progressive-add-start", "render-progressive-add-end");
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
		fill(target, pointcloud, view, proj);
		createVBO(target, pointcloud, view, proj);

		if(false){
			gl.memoryBarrier(gl.ALL_BARRIER_BITS);
			// taken from https://stackoverflow.com/questions/2901102/how-to-print-a-number-with-commas-as-thousands-separators-in-javascript
			const numberWithCommas = (x) => {
				return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
			}

			let resultBuffer = new ArrayBuffer(10 * 5 * 4);
			gl.getNamedBufferSubData(state.ssFillCommands, 0, resultBuffer.byteLength, resultBuffer);
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

			const sMin = (parseInt(min));
			const sMax = (parseInt(max));
			const sMean = (parseInt(mean));
			const sMedian = (parseInt(median));
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