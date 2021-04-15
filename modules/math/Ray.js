// Inspired by and adapted from three.js (Ray.js)
// 
// three.js: https://github.com/mrdoob/three.js
// Ray.js https://github.com/mrdoob/three.js/blob/dev/src/math/Ray.js
// 
// license: MIT (https://github.com/mrdoob/three.js/blob/5498e9ec318aa6f03b2877d36914616cd498b4f6/LICENSE)
//

class Ray{

	constructor(origin, direction){
		this.origin = origin;
		this.direction = direction;
	}

	at(t){

		let v = new Vector3();

		v.copy(this.direction);
		v.multiplyScalar(t);
		v = v.add(this.origin);

		return v;
	}

	intersectBox(box){
		let tmin, tmax, tymin, tymax, tzmin, tzmax;

		let invdirx = 1 / this.direction.x,
			invdiry = 1 / this.direction.y,
			invdirz = 1 / this.direction.z;

		let origin = this.origin;

		if ( invdirx >= 0 ) {

			tmin = ( box.min.x - origin.x ) * invdirx;
			tmax = ( box.max.x - origin.x ) * invdirx;

		} else {

			tmin = ( box.max.x - origin.x ) * invdirx;
			tmax = ( box.min.x - origin.x ) * invdirx;

		}

		if ( invdiry >= 0 ) {

			tymin = ( box.min.y - origin.y ) * invdiry;
			tymax = ( box.max.y - origin.y ) * invdiry;

		} else {

			tymin = ( box.max.y - origin.y ) * invdiry;
			tymax = ( box.min.y - origin.y ) * invdiry;

		}

		if( ( tmin > tymax ) || ( tymin > tmax ) ){
			 return null;
		}

		// These lines also handle the case where tmin or tmax is NaN
		// (result of 0 * Infinity). x !== x returns true if x is NaN

		if ( tymin > tmin || tmin !== tmin ) tmin = tymin;

		if ( tymax < tmax || tmax !== tmax ) tmax = tymax;

		if ( invdirz >= 0 ) {

			tzmin = ( box.min.z - origin.z ) * invdirz;
			tzmax = ( box.max.z - origin.z ) * invdirz;

		} else {

			tzmin = ( box.max.z - origin.z ) * invdirz;
			tzmax = ( box.min.z - origin.z ) * invdirz;

		}

		if ( ( tmin > tzmax ) || ( tzmin > tmax ) ) return null;

		if ( tzmin > tmin || tmin !== tmin ) tmin = tzmin;

		if ( tzmax < tmax || tmax !== tmax ) tmax = tzmax;

		//return point closest to the ray (positive side)

		if ( tmax < 0 ) return null;

		let t = tmin >= 0 ? tmin : tmax;

		return this.at(t);
	}

};