
class Box3{

	constructor(min, max){

		if(arguments.length === 0){
			this.min = new Vector3(Infinity, Infinity, Infinity);
			this.max = new Vector3(-Infinity, -Infinity, -Infinity);
		}else if(arguments.length === 2){
			this.min = min;
			this.max = max;
		}else{
			throw new Error("unexpected number of arguments");
		}
	}

	getSize(){
		let size = new Vector3(
			this.max.x - this.min.x,
			this.max.y - this.min.y,
			this.max.z - this.min.z,
		);

		return size;
	}

	getCenter(){
		return new Vector3().addVectors(this.min, this.max).multiplyScalar(0.5);
	}

	expandByXYZ(x, y, z){
		this.min.x = Math.min(this.min.x, x);
		this.min.y = Math.min(this.min.y, y);
		this.min.z = Math.min(this.min.z, z);

		this.max.x = Math.max(this.max.x, x);
		this.max.y = Math.max(this.max.y, y);
		this.max.z = Math.max(this.max.z, z);
	}

	applyMatrix4(matrix){

		// TODO lot's of optimization opportunities if this turns out ot be a bottleneck

		let min = this.min;
		let max = this.max;

		let points = [
			new Vector3(min.x, min.y, min.z),
			new Vector3(min.x, min.y, max.z),
			new Vector3(min.x, max.y, min.z),
			new Vector3(min.x, max.y, max.z),

			new Vector3(max.x, min.y, min.z),
			new Vector3(max.x, min.y, max.z),
			new Vector3(max.x, max.y, min.z),
			new Vector3(max.x, max.y, max.z),
		];

		this.min.set(Infinity, Infinity, Infinity);
		this.max.set(-Infinity, -Infinity, -Infinity);

		for(let point of points){
			point.applyMatrix4(matrix);

			this.min.min(point);
			this.max.max(point);
		}

	}

	copy(box){
		this.min.copy(box.min);
		this.max.copy(box.max);

		return this;
	}

	clone(){
		return new Box3(this.min.clone(), this.max.clone());
	}

	toString(){
		let str = `min: ${this.min}\n`;
		str += `max: ${this.max}`;

		return str;
	}



};