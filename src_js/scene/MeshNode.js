
class MeshNode extends SceneNode{

	constructor(name, buffer, material){

		super(name);

		this.buffer = buffer;
		this.material = material;

		this.components.push(buffer, material);

		this.boundingBox = buffer.computeBoundingBox();
		this.boundingBoxWorld = this.boundingBox.clone();

	}

	update(){

		this.boundingBoxWorld.copy(this.boundingBox).applyMatrix4(this.world);

		let node = this.parent;
		while(node){

			node.boundingBoxWorld.min.min(this.boundingBoxWorld.min);
			node.boundingBoxWorld.max.max(this.boundingBoxWorld.max);

			node = node.parent;
		}

		for(let child of this.children){
			child.update();
		}

	}

	intersect(ray){
		
		let box = this.boundingBoxWorld;
		let I = ray.intersectBox(box);

		return I;

	}
	


}





