
import {Vector3} from "./Vector3.mjs"

export class Box3{

	constructor(min, max){
		this.min = min ?? new Vector3(+Infinity, +Infinity, +Infinity);
		this.max = max ?? new Vector3(-Infinity, -Infinity, -Infinity);
	}

	clone(){
		return new Box3(
			this.min.clone(),
			this.max.clone()
		);
	}

	copy(box){
		this.min.copy(box.min);
		this.max.copy(box.max);
	}

	size(){
		return this.max.clone().sub(this.min);
	}

	center(){
		return this.min.clone().add(this.max).multiplyScalar(0.5);
	}

	cube(){
		let cubeSize = Math.max(...this.size().toArray());
		let min = this.min.clone();
		let max = this.min.clone().addScalar(cubeSize);
		let cube = new Box3(min, max);

		return cube;
	}

	expandByXYZ(x, y, z){
		this.min.x = Math.min(this.min.x, x);
		this.min.y = Math.min(this.min.y, y);
		this.min.z = Math.min(this.min.z, z);

		this.max.x = Math.max(this.max.x, x);
		this.max.y = Math.max(this.max.y, y);
		this.max.z = Math.max(this.max.z, z);
	}

	expandByPoint(point){
		this.min.x = Math.min(this.min.x, point.x);
		this.min.y = Math.min(this.min.y, point.y);
		this.min.z = Math.min(this.min.z, point.z);

		this.max.x = Math.max(this.max.x, point.x);
		this.max.y = Math.max(this.max.y, point.y);
		this.max.z = Math.max(this.max.z, point.z);
	}

	expandByBox(box){
		this.expandByPoint(box.min);
		this.expandByPoint(box.max);
	}

	applyMatrix4(matrix){

		let {min, max} = this;

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

		let newBox = new Box3();

		for(let point of points){
			let projected = point.applyMatrix4(matrix);
			newBox.expandByPoint(projected);
		}

		this.min.copy(newBox.min);
		this.max.copy(newBox.max);

		return this;
	}

	isFinite(){
		return this.min.isFinite() && this.max.isFinite();
	}

};