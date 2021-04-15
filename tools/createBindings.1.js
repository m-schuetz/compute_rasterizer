

const fs = require("fs");


let files = [
	"C:/dev/workspaces/glew/master/auto/registry/gl/api/GL/glext.h",
	"C:/dev/workspaces/glew/master/auto/registry/gl/api/GL/glcorearb.h",
];

let vendors = ["OES", "EXT", "KHR", "ARB", "3DFX", "APPLE", 
		"INTEL", "AMD", "ATI", "NV", "PGI", "SGIS", "OML",
		"REND", "S3TC", "SGIX", "NVX", "SGI", "SUN", "OVR",
		"WIN", "INGR", "MESAX", "MESA", "IBM", "SUNX"];

let constantBlacklist = [
	"GL_GLEXT_VERSION", "GL_CLIP_DISTANCE6", "GL_CLIP_DISTANCE7"
];


function isCommand(line){
	let isCommand = line.match(/GLAPI \w* APIENTRY gl.*\(.*\);/) !== null;
	let isNotARB = line.match(/GLAPI \w* APIENTRY gl\w*(ARB) .*\(.*\);/) === null;
	let isNotAPPLE = line.match(/GLAPI \w* APIENTRY gl\w*(APPLE) .*\(.*\);/) === null;
	let isNotINTEL = line.match(/GLAPI \w* APIENTRY gl\w*(INTEL) .*\(.*\);/) === null;
	let isNotOES = line.match(/GLAPI \w* APIENTRY gl\w*(OES) .*\(.*\);/) === null;
	let isNotEXT = line.match(/GLAPI \w* APIENTRY gl\w*(EXT) .*\(.*\);/) === null;

	let hasVendor = false;
	for(let vendor of vendors){
		let regex = RegExp(`GLAPI \\w* APIENTRY gl\\w*(${vendor}) .*\\(.*\\);`, "g");

		let fromVendor = line.match(regex);

		hasVendor = hasVendor || fromVendor;
	}


	return isCommand && !hasVendor;
}

function isConstant(line){
	let isConstant = line.match(/#define GL_.*/) !== null;

	let hasVendor = false;
	for(let vendor of vendors){
		let regex1 = new RegExp(`#define GL_\\w*(${vendor}) .*`, "g");
		let regex2 = new RegExp(`#define GL_${vendor}_.*`, "g");
		let fromVendor = (line.match(regex1) !== null) || (line.match(regex2) !== null);

		hasVendor = hasVendor || fromVendor;
	}

	return isConstant && !hasVendor;
}

async function readCommandAndConstants(files){
	let commands = [];
	let constants = [];

	for(let file of files){
		let fileHandle = await fs.promises.open(file, "r");
		let content = await fileHandle.readFile({encoding: "utf8"});
		await fileHandle.close();

		let lines = content.split("\n");

		for(let line of lines){
			if(isCommand(line)){
			//if(line.match(/GLAPI \w* APIENTRY gl.*\(.*\);/) !== null){
				commands.push(line);
			}else if(isConstant(line)){
				constants.push(line);
			}
		}
	}

	return {
		commands: commands,
		constants: constants
	};
}

function createConstantBinding(constant){
	let matches = constant.match(/#define GL_(\w*)\s*(.*)*/);

	let constantName = matches[1];
	let constantValue = `GL_${constantName}`;

	let binding = `CREATE_CONSTANT_ACCESSOR("${constantName}", ${constantValue});`;

	return binding;
}

async function createBindings(){
	let {commands, constants} = await readCommandAndConstants(files);



	{
		let fileHandle = await fs.promises.open("./commands.txt", "w");

		let content = commands.join("\n");

		await fileHandle.writeFile(content);
		await fileHandle.close();
	}

	{
		let fileHandle = await fs.promises.open("./constants.txt", "w");

		let bindings = constants.map(createConstantBinding);

		let content = bindings.join("\n");

		await fileHandle.writeFile(content);
		await fileHandle.close();
	}

	console.log(`commands: ${commands.length}`);
	console.log(`constants: ${constants.length}`);
}

createBindings();



