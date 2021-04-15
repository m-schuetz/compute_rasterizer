
var Intersections = class Intersections{

	constructor(){

	}

};

Intersections.boxLinesIntersection = function(box, lines){

	let numLines = lines.length / (4 * 2);

	for(let i = 0; i < numLines; i++){

		let startXYZ = lines.slice(8 * i, 8 * i + 3);
		let endXYZ = lines.slice(8 * i + 4, 8 * i + 7);

		let start = new Vector3(...startXYZ);
		let end = new Vector3(...endXYZ);
		let dir = new Vector3().subVectors(end, start).normalize();
		
		let ray = new Ray(start, dir);

		let I = ray.intersectBox(box);

		if(I && start.distanceTo(I) <= start.distanceTo(end)){
			return I;
		}

	}

	return null;
};


"Intersections.js"