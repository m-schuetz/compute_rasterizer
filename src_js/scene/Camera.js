
class Camera extends SceneNode{
	constructor(){
		super("camera");

		this.fov = 60;
		this.near = 0.1;
		this.far = 10000;
		this.aspect = 1.0;
		this.targetDistance = 10.0;
		this.size = {width: 0, height: 0};

		this.projectionMatrix = new Matrix4();
		this.updateProjectionMatrix();
	}

	updateProjectionMatrix(){

		let fovRad = (Math.PI * this.fov) / 180;
		let top = Math.tan(0.5 * fovRad);
		let height = 2 * top;
		let width = this.aspect * height;
		let left = -0.5 * width;

		let bottom = -top;
		let right = 0.5 * width;

		this.projectionMatrix.makePerspectiveZeroToOneInfiniteFar(left, right, top, bottom, this.near);
		//this.projectionMatrix.makePerspective(left, left + width, top, top - height, this.near, this.far);
	}

	getFrustum(){

		let fovRad = (Math.PI * this.fov) / 180;
		let top = this.near * Math.tan(0.5 * fovRad);
		let bottom = -top;
		let height = 2 * top;
		let width = this.aspect * height;
		let left = -0.5 * width;
		let right = 0.5 * width;

		let near = this.near;
		let far = this.far;

		let pNear = new Plane(new Vector3(0, 0, -1), -near);
		let pFar = new Plane(new Vector3(0, 0, 1), far);
		
		let cLeft = new Vector3(left, 0, -near);
		let nLeft = new Vector3(near, 0, left).normalize();
		let pLeft = new Plane().setFromNormalAndCoplanarPoint(nLeft, cLeft);

		let cRight = new Vector3(right, 0, -near);
		let nRight = new Vector3(-near, 0, -right).normalize();
		let pRight = new Plane().setFromNormalAndCoplanarPoint(nRight, cRight);

		let cTop = new Vector3(0, top, -near);
		let nTop = new Vector3(0, -near, -top).normalize();
		let pTop = new Plane().setFromNormalAndCoplanarPoint(nTop, cTop);

		let cBottom = new Vector3(0, bottom, -near);
		let nBottom = new Vector3(0, near, bottom).normalize();
		let pBottom = new Plane().setFromNormalAndCoplanarPoint(nBottom, cBottom);

		let frustum = [
			pNear,
			pFar,
			pLeft, 
			pRight,
			pTop,
			pBottom,
		];

		return frustum;

	}

	
};
