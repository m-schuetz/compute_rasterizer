
class Vector3{

	constructor(x, y, z){

		if(arguments.length === 0){
			this.x = 0;
			this.y = 0;
			this.z = 0;
		}else if(arguments.length === 1){
			this.x = x;
			this.y = x;
			this.z = x;
		}else{
			this.x = x;
			this.y = y;
			this.z = z;
		}
	}

	set(x, y, z){
		this.x = x;
		this.y = y;
		this.z = z;
	}

	length(){
		return Math.sqrt(this.x ** 2 + this.y ** 2 + this.z ** 2);
	}

	distanceTo(vec){
		let dx = vec.x - this.x;
		let dy = vec.y - this.y;
		let dz = vec.z - this.z;

		let distance = Math.sqrt(dx ** 2 + dy ** 2 + dz ** 2);

		return distance;
	}

	dot(v){
		return this.x * v.x + this.y * v.y + this.z * v.z;
	}

	cross(b){
		let a = this;

		let ax = a.x, ay = a.y, az = a.z;
		let bx = b.x, by = b.y, bz = b.z;


		let x = ay * bz - az * by;
		let y = az * bx - ax * bz;
		let z = ax * by - ay * bx;

		let result = new Vector3(x, y, z);

		return result;
	}

	sub(v){
		let result = new Vector3(
			this.x - v.x,
			this.y - v.y,
			this.z - v.z
		);

		return result;
	}

	add(v){
		let result = new Vector3(
			this.x + v.x,
			this.y + v.y,
			this.z + v.z
		);

		return result;
	}

	addVectors(a, b){
		this.x = a.x + b.x;
		this.y = a.y + b.y;
		this.z = a.z + b.z;

		return this;
	}

	subVectors(a, b){
		this.x = a.x - b.x;
		this.y = a.y - b.y;
		this.z = a.z - b.z;

		return this;
	}

	multiplyScalar(scalar){
		this.x *= scalar;
		this.y *= scalar;
		this.z *= scalar;

		return this;
	}

	normalize(){
		let length = Math.sqrt(this.x ** 2 + this.y ** 2 + this.z ** 2);

		this.x = this.x / length;
		this.y = this.y / length;
		this.z = this.z / length;

		return this;
	}

	applyMatrix4(m, w = 1){

		let {x, y, z} = this;
		let e = m.elements;

		this.x = e[ 0 ] * x + e[ 4 ] * y + e[ 8 ] * z + e[ 12 ] * w;
		this.y = e[ 1 ] * x + e[ 5 ] * y + e[ 9 ] * z + e[ 13 ] * w;
		this.z = e[ 2 ] * x + e[ 6 ] * y + e[ 10 ] * z + e[ 14 ] * w;
		let rw = e[ 3 ] * x + e[ 7 ] * y + e[ 11 ] * z + e[ 15 ] * w;

		this.x = this.x / rw;
		this.y = this.y / rw;
		this.z = this.z / rw;
		
		return this;
	}

	min(v){
		this.x = Math.min(this.x, v.x);
		this.y = Math.min(this.y, v.y);
		this.z = Math.min(this.z, v.z);
	}

	max(v){
		this.x = Math.max(this.x, v.x);
		this.y = Math.max(this.y, v.y);
		this.z = Math.max(this.z, v.z);
	}

	clone(){
		return new Vector3(this.x, this.y, this.z);
	}

	copy(vec){
		this.x = vec.x;
		this.y = vec.y;
		this.z = vec.z;

		return this;
	}

	toArray(){
		return [this.x, this.y, this.z];
	}

	toString(){
		return `${this.x}, ${this.y}, ${this.z}`;
	}

};
