

class SceneNode{

	constructor(name){

		this.name = name;

		this.components = [];

		this.boundingBox = new Box3();
		this.boundingBoxWorld = new Box3();

		this.parent = null;
		this.children = [];

		this.aabb = new Box3();

		this.position = new Vector3(0, 0, 0);
		this.scale = new Vector3(1, 1, 1);
		this.rotation = new Matrix4();

		this.transform = new Matrix4();
		this.world = new Matrix4();

		this.visible = true;
	}

	update(){

		this.boundingBoxWorld.min.set(Infinity, Infinity, Infinity);
		this.boundingBoxWorld.max.set(-Infinity, -Infinity, -Infinity);

		for(let child of this.children){
			child.update();
		}

	}

	add(node){
		this.children.push(node);
		node.parent = this;
	}

	remove(node){
		if(!node){
			return;
		}

		this.children = this.children.filter(child => child !== node);
		node.parent = null;
	}

	removeNamed(name){
		this.children = this.children.filter(child => child.name !== name);
	}

	find(name){
		let result = this.children.find(child => child.name === name);

		return result;
	}

	traverse(callback, level = 0){

		let carryOn = callback(this, level);

		if(carryOn){
			
			for(let child of this.children){
				child.traverse(callback, level + 1);
			}

		}

	}

	getComponents(type){
		return this.components.filter(c => c instanceof type);
	}

	getComponent(type, defaultValue){
		let component = this.components.find(c => c instanceof type);

		if(!component){
			if(defaultValue && defaultValue.or){
				return defaultValue.or;
			}else{
				return null;
			}
		}

		return component;
	}

	getDirection(){
		let dir4 = new Vector4(0, 0, -1, 0);

		dir4.applyMatrix4(this.rotation);

		let dir = new Vector3(dir4.x, dir4.y, dir4.z);

		return dir;
	}

	getDirectionWorld(){
		let dir4 = new Vector4(0, 0, -1, 0);

		dir4.applyMatrix4(this.world);

		let dir = new Vector3(dir4.x, dir4.y, dir4.z);

		return dir;
	}

	lookAt(){
		if(arguments.length === 1){
			this.lookAtPoint(...arguments);
		}else if(arguments.length === 3){
			this.lookAtXYZ(...arguments);
		}
	}

	lookAtPoint(point){
		this.lookAtXYZ(point.x, point.y, point.z);
	}

	lookAtXYZ(x, y, z){

		let pos = this.position;
		let target = new Vector3(x, y, z);

		let dir = pos.sub(target).normalize();
		let straightUp = new Vector3(0, 1, 0);
		let right = straightUp.cross(dir).normalize();
		let up = dir.cross(right);

		let rotate = new Matrix4().set([
			right.x, up.x, dir.x, 0,
			right.y, up.y, dir.y, 0, 
			right.z, up.z, dir.z, 0,
			0, 0, 0, 1	
		]);

		rotate = rotate.getInverse();

		this.rotation.copy(rotate);
	}

	updateTransform(){
		let x = this.position.x;
		let y = this.position.y;
		let z = this.position.z;

		let sx = this.scale.x;
		let sy = this.scale.y;
		let sz = this.scale.z;

		let translate = new Matrix4().makeTranslation(x, y, z);
		let rotate = this.rotation;
		let scale = new Matrix4().makeScale(sx, sy, sz);

		this.transform.multiplyMatrices(translate, rotate).multiply(scale);
	}

	updateMatrixWorld(){

		this.updateTransform();

		if(this.parent === null){
			this.world.copy(this.transform);
		}else{
			this.world.multiplyMatrices(this.parent.world, this.transform);
		}

		for(let child of this.children){
			child.updateMatrixWorld();
		}

	}

}





