

if(typeof e4called === "undefined"){
	e4called = true;
	
	let las = loadLAS("D:/dev/pointclouds/weiss/pos8_lifeboats.las");
	// let las = loadLAS("D:/dev/pointclouds/riegl/retz_sort_morton.las");
	// let las = loadLAS("D:/dev/pointclouds/archpro/heidentor.las");
	// let las = loadLAS("D:/dev/pointclouds/lion.las");

	let pc = new PointCloudProgressive("testcloud", "blabla");
	pc.boundingBox.min.set(...las.boundingBox.min);
	pc.boundingBox.max.set(...las.boundingBox.max);

	let handle = las.handles[3];
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


window.width = 1600;
window.height = 1080;

window.x = 2560;
window.y = 0;

// lion
// view.set(
// 	[-8.60297567428115, 2.8137507646084092, -10.617733553260202], 
// 	[-9.740633921915837, 2.090449992726934, -11.652550151644672]
// );
// camera.fov = 60;

// lifeboat
view.set(
	[-10.857, 3.839, -14.378],
	[-7.709, 3.015, -13.759],
);
camera.fov = 100;



camera.near = 0.1;

MSAA_SAMPLES = 1;

// log(`
// view.set(
// 	[${view.position}],
// 	[${view.getPivot()}],
// );
// `);

// log(view.position)
// log(view.getPivot())