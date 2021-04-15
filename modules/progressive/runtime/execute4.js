

if(typeof e4called === "undefined"){
	e4called = true;
	
	//let las = loadBINProgressive("D:/dev/pointclouds/riegl/niederweiden_400m.bin");
	//let las = loadLASProgressive("D:/dev/pointclouds/tuwien_baugeschichte/Museum Affandi_las export/batch_1.las");
	let las = loadLASProgressive("D:/dev/pointclouds/NVIDIA/laserscans/merged.las");
	//let las = loadLASProgressive("D:/dev/pointclouds/hofbibliothek/HB_64.las");
	//let las = loadLASProgressive("D:/dev/pointclouds/tuwien_baugeschichte/Candi Sari_las export/candi_sari.las");
	//let las = loadBINProgressive("D:/dev/pointclouds/test.bin");


	let pc = new PointCloudProgressive("testcloud", "blabla");

	let handles = las.handles;

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("value", 1, 4, gl.INT, gl.FALSE, 4, 12, {targetType: "int"}),
	];

	let bytesPerPoint = attributes.reduce( (p, c) => p + c.bytes, 0);

	let maxPointsPerBuffer = 134 * 1000 * 1000;
	let numPointsLeft = las.numPoints;
	let glBuffers = handles.map( (handle) => {

		let numPointsInBuffer = numPointsLeft > maxPointsPerBuffer ? maxPointsPerBuffer : numPointsLeft;
		numPointsLeft -= maxPointsPerBuffer;

		let glbuffer = new GLBuffer();

		glbuffer.attributes = attributes;

		gl.bindVertexArray(glbuffer.vao);
		glbuffer.vbo = handle;
		gl.bindBuffer(gl.ARRAY_BUFFER, glbuffer.vbo);

		for(let attribute of attributes){

			let {location, count, type, normalize, offset} = attribute;

			gl.enableVertexAttribArray(location);

			if(attribute.targetType === "int"){
				gl.vertexAttribIPointer(location, count, type, bytesPerPoint, offset);
			}else{
				gl.vertexAttribPointer(location, count, type, normalize, bytesPerPoint, offset);
			}
		}

		gl.bindVertexArray(0);

		glbuffer.count =  numPointsInBuffer;

		return glbuffer;
	});

	pc.glBuffers = glBuffers;

	let s = 0.3;
	pc.transform.elements.set([
		s, 0, 0, 0, 
		0, 0, s, 0, 
		0, s, 0, 0, 
		-10, 1.4, -11, 1, 
	]);

	scene.root.add(pc);

	listeners.update.push(() => {

		if(pc.numPoints !== las.numPoints){
			//log(las.numPoints);
		}
		pc.numPoints = las.numPoints;
		
	});

}

// Niederweiden
// view.set(
// 	[11.859564250720245, 7.22595953573232, -23.071864520081228],
// 	[-2.9628644577436702, -2.4220795771249986, -28.215191026963357],
// );

// view.set(
// 	[-13.517644436226734, 7.727134067740751, -20.095312208539994],
// 	[-2.577821320940421, -2.019933609584122, -28.1996901090152],
// );

// // affandi
// view.set(
// 	[125.22401257568242, 44.40866270039364, 176.94497074343428],
// 	[113.43495574025215, 32.450565105907245, 165.6089264439775],
// );

// affandi
// view.set(
// 	[114.956609221704, 19.922502874545994, 105.35076237787068],
// 	[117.98850240507511, 19.101296779083743, 102.86065914999514],
// );

// // arbegen
// view.set(
// 	[12.713368935730871, 8.706469023429722, 0.884167816734989],
// 	[5.398363445056037, 4.205587572729591, 0.6353886246271072],
// );

view.set(
	[-9.619991956336769, 1.769517431458914, -10.659795935797932],
	[-9.884130540436988, 1.5546140882305126, -10.88264324899804],
);

camera.fov = 100;

// log($("testcloud"));

// if($("testcloud")){
// 	let pc = $("testcloud");
// 	let s = 0.03;
// 	pc.transform.elements.set([
// 		s, 0, 0, 0, 
// 		0, 0, s, 0, 
// 		0, s, 0, 0, 
// 		-0.5, 0.7, 1, 1, 
// 	]);

// 	log("lala");

// }