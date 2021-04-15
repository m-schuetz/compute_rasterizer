
import {promises as fsp} from "fs";

let path = "./data/viewpoints_3090.json";

async function run(){

	let content = await fsp.readFile(path, {encoding: "utf-8"});
	let json = JSON.parse(content);

	// let originals = json.scenarios.filter(s => s.sorting === "original");

	{ // GL_POINTS / original
		let originals = json.scenarios.filter(s => s.sorting === "original");
		
		let timings = originals.map(s => s.timings.filter(t => t.method === "GL_POINTS")).flat().map(d => d.durations.find(d => d.label === "frame").avg)
		let ratio = Math.max(...timings) / Math.min(...timings);

		console.log("GL_POINTS / original");
		console.log(timings, ratio);
	}

	{ // reduce+early-z / original
		let originals = json.scenarios.filter(s => s.sorting === "original");
		
		let timings = originals.map(s => s.timings.filter(t => t.method === "compute(early-depth, ballot)")).flat().map(d => d.durations.find(d => d.label === "frame").avg)
		let ratio = Math.max(...timings) / Math.min(...timings);

		console.log("reduce+early-z / original");
		console.log(timings, ratio);
	}

	{ // GL_POINTS / shuffled
		let originals = json.scenarios.filter(s => s.sorting === "shuffled");
		
		let timings = originals.map(s => s.timings.filter(t => t.method === "GL_POINTS")).flat().map(d => d.durations.find(d => d.label === "frame").avg)
		let ratio = Math.max(...timings) / Math.min(...timings);

		console.log("GL_POINTS / shuffled");
		console.log(timings, ratio);
	}

	{ // reduce+early-z / shuffled
		let originals = json.scenarios.filter(s => s.sorting === "shuffled");
		
		let timings = originals.map(s => s.timings.filter(t => t.method === "compute(early-depth, ballot)")).flat().map(d => d.durations.find(d => d.label === "frame").avg)
		let ratio = Math.max(...timings) / Math.min(...timings);

		console.log("reduce+early-z / shuffled");
		console.log(timings, ratio);
	}




}

run();
