
class Scene{

	constructor(){

		this.root = new SceneNode("root");
		this.drawQueue = [];

	}

	raycast(origin, direction){
		
		let stack = [];

		while(stack.length > 0){

			let node = stack.pop();

			let intersection = node.raycast(origin, direction);

			
			

		}

	}

	drawLines(id, lines){

		let command = {
			name: "drawLines",
			id: id,
			lines: lines
		};

		this.drawQueue.push(command);
	}

	drawSphere(id, args){

		let position = (args.position !== undefined) ? args.position : new Vector3();
		let scale = (args.scale !== undefined) ? args.scale : 1;

		let command = {
			name: "drawSphere",
			position: position,
			scale: scale,
		};

		this.drawQueue.push(command);
	}

	drawBox(id, box){

		let command = {
			name: "drawBox",
			box, box,
		};

		this.drawQueue.push(command);
	}

	drawNode(id, args){

		let command = {
			name: "drawNode",
			node: args.node,
			position: args.position,
			scale: args.scale,
		};

		this.drawQueue.push(command);

	}

}