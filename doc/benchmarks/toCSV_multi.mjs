
import {promises as fsp} from "fs";

let paths = [
	"D:/temp/charts/benchmarks_endeavor_32.json",
	"D:/temp/charts/benchmarks_endeavor_64.json",
	"D:/temp/charts/benchmarks_endeavor_128.json",
	"D:/temp/charts/benchmarks_endeavor_256.json",
	"D:/temp/charts/benchmarks_endeavor_512.json",
	"D:/temp/charts/benchmarks_endeavor_1024.json",
	"D:/temp/charts/benchmarks_endeavor_2048.json",
	"D:/temp/charts/benchmarks_endeavor_4096.json",
	"D:/temp/charts/benchmarks_endeavor_8192.json",
	"D:/temp/charts/benchmarks_endeavor_16384.json",
];

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

async function loadJSONs(){

	let jsons = {
		scenarios: [],
	};

	let i = 0;
	for(let path of paths){
		let content = await fsp.readFile(path, {encoding: "utf-8"});
		let json = JSON.parse(content);

		json.scenarios[0].sorting = `morton ${i}`;

		jsons.scenarios.push(...json.scenarios);

		i++;
	}
	
	return jsons;
}

async function run(){

	// let content = await fsp.readFile(path, {encoding: "utf-8"});
	// let json = JSON.parse(content);

	let json = await loadJSONs();

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
