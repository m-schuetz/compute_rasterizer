// Inspired by and adapted from three.js (Plane.js)
// 
// three.js: https://github.com/mrdoob/three.js
// Plane.js https://github.com/mrdoob/three.js/blob/dev/src/math/Plane.js
// 
// license: MIT (https://github.com/mrdoob/three.js/blob/5498e9ec318aa6f03b2877d36914616cd498b4f6/LICENSE)
//

// http://www.songho.ca/math/plane/plane.html
//
// a * x + b * y + c * z + d = 0
//
// abc: normal
// xyz: point on plane
// d: distance from origin to plane.
//    d is negative if plane is in direction of positive normal
//    e.g. p = (2, 0, 0), n = (1, 0, 0), d = -(n dot p) = -2
// d = -(normal dot pointOnPlane)
//
//

class Plane{

	constructor(normal, distance){

		if(arguments.length === 0){
			this.normal = new Vector3(1, 0, 0);
			this.distance = 0;
		}else{
			this.normal = normal;
			this.distance = distance;
		}

	}

	setFromNormalAndCoplanarPoint(normal, point){
		this.normal = normal;
		this.distance = -normal.dot(point);

		return this;
	}

	applyMatrix4(matrix){

		let p = this.normal.clone().multiplyScalar(-this.distance);

		p.applyMatrix4(matrix);

		let n = this.normal;
		let n4 = new Vector4(n.x, n.y, n.z, 0).applyMatrix4(matrix);
		let normal = new Vector3(n4.x, n4.y, n4.z);

		this.setFromNormalAndCoplanarPoint(normal, p);
	}

	toString(){
		let n = this.normal.toArray().join(", ");
		let d = this.distance;

		return `normal: (${n}), distance: ${d}`;
	}

}