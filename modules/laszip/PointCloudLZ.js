
class PointCloudLZ extends SceneNode{

	constructor(name){
		super(name);

		let vsPath = "../../resources/shaders/pointcloud_basic.vs";
		let fsPath = "../../resources/shaders/pointcloud.fs";
		
		let shader = new Shader([
			{type: gl.VERTEX_SHADER, path: vsPath},
			{type: gl.FRAGMENT_SHADER, path: fsPath},
		]);
		shader.watch();

		let material = new GLMaterial();
		material.shader = shader;
		material.glDrawMode = gl.POINTS;
		this.components.push(material);
	}

	setData(numPoints, position, color){

		let vao = gl.createVertexArray();

		gl.bindVertexArray(vao);

		let vboPosition = gl.createBuffer();
		let vboColor = gl.createBuffer();

		gl.namedBufferData(vboPosition, position.byteLength, position, gl.DYNAMIC_DRAW); 
		gl.namedBufferData(vboColor, color.byteLength, color, gl.DYNAMIC_DRAW); 

		gl.bindBuffer(gl.ARRAY_BUFFER, vboPosition);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
		gl.enableVertexAttribArray(0);

		gl.bindBuffer(gl.ARRAY_BUFFER, vboColor);
		gl.vertexAttribPointer(1, 4, gl.UNSIGNED_BYTE, true, 4, 0);
		gl.enableVertexAttribArray(1);

		gl.bindVertexArray(null);

		this.numPoints = numPoints;
		this.vao = vao;
	}

	setVboData(numPoints, vboPosition, vboColor){

		let vao = gl.createVertexArray();

		gl.bindVertexArray(vao);

		gl.bindBuffer(gl.ARRAY_BUFFER, vboPosition);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
		gl.enableVertexAttribArray(0);

		gl.bindBuffer(gl.ARRAY_BUFFER, vboColor);
		gl.vertexAttribPointer(1, 4, gl.UNSIGNED_BYTE, true, 4, 0);
		gl.enableVertexAttribArray(1);

		gl.bindVertexArray(null);

		this.numPoints = numPoints;
		this.vao = vao;
	}

	render(view, proj, target){
		let material = this.getComponent(GLMaterial, {or: GLMaterial.DEFAULT});
		let shader = material.shader;
		let shader_data = shader.uniformBlocks.shader_data;

		let transform = new Matrix4();

		let world = this.world;

		transform.copy(Matrix4.IDENTITY);
		transform.multiply(proj).multiply(view).multiply(world);

		if(shader_data){
			shader_data.setFloat32Array("transform", transform.elements);
			shader_data.setFloat32Array("world", world.elements);
			shader_data.setFloat32Array("view", view.elements);
			shader_data.setFloat32Array("proj", proj.elements);

			shader_data.setFloat32("time", now());

			shader_data.bind();
			shader_data.submit();
		}

		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(true);

		gl.bindVertexArray(this.vao);

		gl.drawArrays(gl.POINTS, 0, this.numPoints);

		gl.bindVertexArray(0);
	}

};