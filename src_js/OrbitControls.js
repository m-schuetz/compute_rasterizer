
class OrbitControls{

	constructor(view){

		this.view = view;
		this.prev = null;

		this.fade = 10;
		this.rotationSpeed = 5;

		this.delta = {
			yaw: 0,
			pitch: 0,
			side: 0,
			up: 0,
			forward: 0,
			scroll: 0,
		};
		this.speed = 1.0;

		this.rotationEnabled = false;
		this.panEnabled = false;

		this.init();
	}

	init(){
		addEventListener("mousemove", this.onMouseMove.bind(this));
		addEventListener("mousedown", this.onMouseDown.bind(this));
		addEventListener("mouseup", this.onMouseUp.bind(this));
		addEventListener("mousescroll", this.onMouseScroll.bind(this));
	}

	onMouseScroll(e){
		this.delta.scroll += e.yoffset;
	}

	onMouseMove(e){
		//log("orbit control move");

		if(this.prev === null){
			this.prev = e;
			return;
		}

		if(this.rotationEnabled){
			this.delta.yaw += e.x - this.prev.x;
			this.delta.pitch += e.y - this.prev.y;
		}

		if(this.panEnabled){
			this.delta.side += e.x - this.prev.x;
			this.delta.up += e.y - this.prev.y;
		}
		
		this.prev = e;
	}

	onMouseDown(e){
		//log("down");
		
		if(e.button === MouseButton.LEFT){
			this.rotationEnabled = true;
		}else if(e.button === MouseButton.RIGHT){
			this.panEnabled = true;
		}
	}

	onMouseUp(e){
		//log("up");

		if(e.button === MouseButton.LEFT){
			this.rotationEnabled = false;
		}else if(e.button === MouseButton.RIGHT){
			this.panEnabled = false;
		}
	}

	update(time){

		let view = this.view;

		//let progression = Math.min(1, this.fade * time);
		//let attenuation = Math.max(0, 1 - this.fade * time);
		let progression = 0.1;

		{ // apply rotation

			// first direction
			let pivot = view.getPivot();
			view.yaw -= progression * this.delta.yaw * 0.05;
			view.pitch -= progression * this.delta.pitch * 0.05;

			// then adjust position
			let toPosition = view.getDirection().multiplyScalar(-view.radius);
			let position = pivot.add(toPosition);

			view.position.copy(position);
		}

		{ // apply pan
			let panDistance = progression * view.radius * 3 * 0.01;

			let px = -this.delta.side * panDistance;
			let py = this.delta.up * panDistance;

			view.pan(px, py);
		}

		//view.pan.bind(view)(1, 2);
		//view.pan(1, 2);
		//view.position.x += 0.01;

		{ // apply scroll
			let radius = view.radius;
			let steps = Math.abs(this.delta.scroll);
			let sign = Math.sign(this.delta.scroll);
			let stepSize = 0.1;

			let factor = 1;
			if(sign > 0){
				factor = Math.pow(1 / (1 + stepSize), steps);
			}else if (sign < 0){
				factor = Math.pow((1 + stepSize), steps);
			}

			let pivot = view.getPivot();
			view.radius = view.radius * factor;
			let toPosition = view.getDirection().multiplyScalar(-view.radius);
			let position = pivot.add(toPosition);

			view.position.copy(position);
		}

		// decellerate over time
		this.delta.yaw = 0;
		this.delta.pitch = 0;
		this.delta.scroll = 0;
		this.delta.side = 0;
		this.delta.up = 0;
	}

};