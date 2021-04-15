

if(typeof e4called === "undefined"){
	e4called = true;
	
	let las = loadLAS("D:/dev/pointclouds/weiss/pos8_lifeboats.las");
	// let las = loadLAS("D:/dev/pointclouds/riegl/retz_sort_morton.las");
	// let las = loadLAS("D:/dev/pointclouds/archpro/heidentor.las");
	// let las = loadLAS("D:/dev/pointclouds/lion.las");

	let pc = new PointCloudProgressive("testcloud", "blabla");
	pc.boundingBox.min.set(...las.boundingBox.min);
	pc.boundingBox.max.set(...las.boundingBox.max);

	let handle = las.handles[4];
	pc.vbo = handle;
	pc.numPoints = las.numPoints;

	let s = 0.3;
	pc.transform.elements.set([
		s, 0, 0, 0, 
		0, 0, -s, 0, 
		0, s, 0, 0, 
		-10, 1.4, -11, 1, 
	]);

	scene.root.add(pc);
}

// log($("testcloud").boundingBox)

window.width = 1920;
window.height = 1080;

window.x = 2560;
window.y = 0;

view.set(
	[-10.857, 3.839, -14.378],
	[-7.709, 3.015, -13.759],
);

camera.fov = 100;
camera.near = 0.1;

renderBenchmark = render_compute_uint13;