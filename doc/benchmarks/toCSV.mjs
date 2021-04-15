
import {promises as fsp} from "fs";

let path = "D:/temp/charts/benchmarks_retz.json";
// let path = "./benchmark_retz_zoomed_out.json";

let methodLabelMapping = {
	"GL_POINTS": "GL_POINTS",
	"compute": "atomicMin",
	"compute(early-depth)": "early-z",
	"compute(ballot)": "ballot",
	"compute(early-depth, ballot)": "(early-z, ballot)",
	"compute(just-set)": "just-set",
	"compute(hqs)": "HQS",
	"compute(hqs, 1x64bit)": "1 atomic",
	"compute(hqs, 1x64bit, float)": "1 atomic float",
	"compute(hqs, 1x64bit, fast)": "1 atomic, fast",
};

async function run(){

	let content = await fsp.readFile(path, {encoding: "utf-8"});
	let json = JSON.parse(content);

	let model = json.scenarios[0].model;
	let methods = json.scenarios[0].timings.map(t => t.method);
	let sortings = json.scenarios.map(s => s.sorting);

	let getValue = (method, sorting) => {
		let scenario = json.scenarios.find(s => s.sorting === sorting);
		let timing = scenario.timings.find(t => t.method === method);
		let frame = timing.durations.find(d => d.label === "frame");

		return frame.avg;
	};

	let firstLine = [model, ...sortings].join("\t");
	let lines = [firstLine];
	for(let method of methods){

		let line = methodLabelMapping[method] + "\t";

		for(let sorting of sortings){
			line = line + getValue(method, sorting) + "\t";
		}

		lines.push(line);
	}

	// let str = methods.join("\n");
	// let str = sortings.join("\n");
	let str = lines.join("\n");

	// let str = JSON.stringify(json, null, "\t");

	await fsp.writeFile("./data.csv", str);

}

run();
