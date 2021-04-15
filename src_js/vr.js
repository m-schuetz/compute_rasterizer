

// see https://github.com/ValveSoftware/openvr/blob/master/headers/openvr.h

const OVRButtonID = {
	System: 0,
	ApplicationMenu: 1,
	Grip: 2,
	DPad_Left: 3,
	DPad_Up: 4,
	DPad_Right: 5,
	DPad_Down: 6,
	A: 7,

	ProximitySensor : 31,

	Axis0: 32,
	Axis1: 33,
	Axis2: 34,
	Axis3: 35,
	Axis4: 36,

	SteamVR_Touchpad: 32,
	SteamVR_Trigger: 33,

	Dashboard_Back: 2,

	Max: 64,
};





class VRRecorder{

	constructor(){
		this.frames = [];
		this.recording = false;
		this.frameLimits = null;
		this.orig = null;
	}

	start(){
		this.frames = [];
		this.recording = true;
	}

	stop(){
		this.recording = false;
	}

	recordFrame(){
		if(this.recording === false){
			return;
		}

		let [near, far] = [0.1, 1000];

		let hmdPose = vr.getHMDPose();
		let leftPose = vr.getLeftEyePose();
		let rightPose = vr.getRightEyePose();
		let leftProj = vr.getLeftProjection(near, far);
		let rightProj = vr.getRightProjection(near, far);

		let frame = {
			hmdPose: hmdPose,
			leftPose: leftPose,
			rightPose: rightPose,
			leftProj: leftProj,
			rightProj: rightProj,
		};

		this.frames.push(frame);
	}

	setFrameLimits(frameLimits){
		this.frameLimits = frameLimits;
	}

	hijack(){

		if(this.orig !== null){
			console.log("already hijacked");
			return;
		}

		this.orig = {
			getHMDPose: vr.getHMDPose,
			getLeftEyePose: vr.getLeftEyePose,
			getRightEyePose: vr.getRightEyePose,
			getLeftProjection: vr.getLeftProjection,
			getRightProjection: vr.getRightProjection,
		};


		vr.getHMDPose = this.getHMDPose.bind(this);
		vr.getLeftEyePose = this.getLeftEyePose.bind(this);
		vr.getRightEyePose = this.getRightEyePose.bind(this);
		vr.getLeftProjection = this.getLeftProjection.bind(this);
		vr.getRightProjection = this.getRightProjection.bind(this);
	}

	release(){
		vr.getHMDPose = this.orig.getHMDPose;
		vr.getLeftEyePose = this.orig.getLeftEyePose;
		vr.getRightEyePose = this.orig.getRightEyePose;
		vr.getLeftProjection = this.orig.getLeftProjection;
		vr.getRightProjection = this.orig.getRightProjection;

		this.orig = null;
	}

	getFrameIndex(){

		let limits;
		
		if(this.frameLimits === null){
		 	limits = [0, this.frames.length];
		}else{
			limits = this.frameLimits;
		}

		let index = (frameCount - limits[0]) % (limits[1] - limits[0]) + limits[0];

		return index;
	}

	getHMDPose(){
		let i = this.getFrameIndex();

		return this.frames[i].hmdPose;
	}

	getLeftEyePose(){
		let i = this.getFrameIndex();

		return this.frames[i].leftPose;
	}

	getRightEyePose(){
		let i = this.getFrameIndex();

		return this.frames[i].rightPose;
	}

	getLeftProjection(){
		let i = this.getFrameIndex();

		return this.frames[i].leftProj;
	}

	getRightProjection(){
		let i = this.getFrameIndex();

		return this.frames[i].rightProj;
	}

};









