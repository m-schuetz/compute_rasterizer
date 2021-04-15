
class View{

	constructor(){
		this.yaw = 0;
		this.pitch = 0;
		this.radius = 10;
		this.position = new Vector3(0, 0, 0);
	}

	getDirection(){
		let dir = new Vector3(0, 0, -1);

		let rotYaw = new Matrix4().makeRotationY(this.yaw);
		let rotPitch = new Matrix4().makeRotationX(this.pitch);

		dir.applyMatrix4(rotPitch);
		dir.applyMatrix4(rotYaw);

		return dir;
	}

	getPivot(){
		let {position, radius} = this;
		let direction = this.getDirection();
		let pivot = new Vector3().addVectors(position, direction.multiplyScalar(radius));

		return pivot;
	}

	getSide(){
		let side = new Vector3(1, 0, 0);
		
		let rotYaw = new Matrix4().makeRotationY(this.yaw);

		side.applyMatrix4(rotYaw);

		return side;
	}

	getUp(){
		let up = new Vector3(0, 1, 0);

		let rotYaw = new Matrix4().makeRotationY(this.yaw);
		let rotPitch = new Matrix4().makeRotationX(this.pitch);

		up.applyMatrix4(rotPitch);
		up.applyMatrix4(rotYaw);

		return up;
	}

	pan(x, y){
		let side = this.getSide();
		let up = this.getUp();

		let pan = side.multiplyScalar(x).add(up.multiplyScalar(y));
		let panned = this.position.add(pan);
		this.position.copy(panned);
	}

	lookAt(){
		let V;
		if(arguments.length === 1){
			V = new Vector3().subVectors(arguments[0], this.position);
		}else if(arguments.length === 3){
			V = new Vector3().subVectors(new Vector3(...arguments), this.position);
		}

		let radius = V.length();
		let dir = V.normalize();

		this.radius = radius;
		this.setDirection(dir);
	}

	setDirection(dir){
		if(dir.x === 0 && dir.z === 0){
			this.pitch = Math.PI / 2 * Math.sign(dir.y);
		}else{
			let yaw = Math.atan2(-dir.z, dir.x) - Math.PI / 2;
			let pitch = Math.atan2(dir.y, Math.sqrt(dir.x * dir.x + dir.z * dir.z));

			this.yaw = yaw;
			this.pitch = pitch;
		}
	}

	set(position, target){
		this.position.set(...position);
		this.lookAt(...target);
	}

};