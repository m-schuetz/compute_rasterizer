
import {promises as fsp} from "fs";

// let path = "./benchmarks.json";
// let path = "./benchmarks_1060.json";
// let path = "./data/models_3090.json";
let path = "./data/models_1060.json";
let multi = [
	// "./data/endeavor_original_3090.json",
	// "./data/endeavor_morton_3090.json",
	// "./data/endeavor_shuffled_3090.json",
	// "./data/endeavor_shuffled_morton_3090.json",
];

let orderLabels = {
	"original": "original",
	"morton": "Morton",
	"shuffled": "shuffled",
	"morton-shuffled": "shuffled Morton",
};

let methods = [
	'GL_POINTS',
	'atomicMin',
	'reduce',
	'early-z',
	'reduce,early-z',
	'dedup',
	'guenther',
	'just-set',
	'HQS',
	'HQS1x,protected',
];

async function loadMulti(){

	let jsons = {
		scenarios: [],
	};

	let i = 0;
	for(let path of multi){
		let content = await fsp.readFile(path, {encoding: "utf-8"});
		let json = JSON.parse(content);

		jsons.scenarios.push(...json.scenarios);

		i++;
	}
	
	return jsons;
}

async function run(){

	let content = await fsp.readFile(path, {encoding: "utf-8"});
	let json = JSON.parse(content);

	let jsons = await loadMulti();
	json.scenarios.push(...jsons.scenarios);

	let models = Array.from(new Set(json.scenarios.map(s => s.model)));
	let sortings = Array.from(new Set(json.scenarios.map(s => s.sorting)));
	console.log(models);
	console.log(sortings);

	let lines = [];

	lines.push("\\toprule");

	for(let model of models){

		for(let sorting of sortings){
			let scenario = json.scenarios.find(s => s.model === model && s.sorting === sorting);
			
			let sortingLabel = orderLabels[sorting];
			let cells;
			if(sorting === sortings[0]){
				cells = [`\\multirow{4}{*}{${model}}`.padEnd(25), sortingLabel.padStart(15)];
			}else{
				cells = [` `.padEnd(25), sortingLabel.padStart(15)];
			}

			let fastest = Math.min(...scenario.timings.filter(t => t.method !== "just-set").map(t => t.durations).flat().filter(d => d.label === "frame").map(d => d.avg));
			let slowest = Math.max(...scenario.timings.filter(t => t.method !== "just-set").map(t => t.durations).flat().filter(d => d.label === "frame").map(d => d.avg));
			// console.log(fastest);
			for(let method of methods){
				let durations = scenario.timings.find(t => t.method === method).durations;
				let duration = durations.find(d => d.label === "frame").avg;
				let cell = `${duration.toFixed(2).padStart(6)}`;

				if(method === "just-set"){

				}else if(duration <= fastest * 1.05){
					cell = `\\cellcolor{cFastest}${cell}`;
				}else if(duration <= fastest * 1.1){
					cell = `\\cellcolor{cAlsoFast}${cell}`;
				}else if(duration >= slowest * 0.95){
					cell = `\\cellcolor{cSlowest}${cell}`;
				}

				cells.push(cell);
			}

			let line = cells.join(" & ") + " \\\\";
			lines.push(line);
		}

		if(model !== models[models.length - 1]){
			lines.push("\\midrule");
		}else{
			lines.push("\\bottomrule");
		}
		


	}

	let str = lines.join("\n");
	console.log(str);


	// let str = lines.join("\n");

	// console.log(str);
}

let template = `
\begin{table*}[]
\begin{tabular}{lllrrrrrrr}
                          &                 &            & \multicolumn{4}{c}{\textbf{Basic}}                                                                                             & \multicolumn{2}{c}{\textbf{High-Quality}}                    & \multicolumn{1}{c}{\textbf{Misc}} \\
                          &                 & GL\_POINTS & \multicolumn{1}{c}{atomicMin} & \multicolumn{1}{c}{ballot} & \multicolumn{1}{c}{early-z} & \multicolumn{1}{c}{ballot, early-z} & \multicolumn{1}{c}{HQS} & \multicolumn{1}{c}{HQS 1x, robust} & \multicolumn{1}{c}{just set}      \\
\toprule
\multirow{4}{*}{Lion}     & original        &            & 1                             & 1                          & 1                           & 1                                   & 1                       & 1                                  & 1                                 \\
                          & morton          &            & 1                             & 1                          & 1                           & 1                                   & 1                       & 1                                  & 1                                 \\
                          & shuffled        &            & 1                             & 1                          & 1                           & 1                                   & 1                       & 1                                  & 1                                 \\
                          & shuffled-Morton &            & 1                             & 1                          & 1                           & 1                                   & 1                       & 1                                  & 1                                 \\
\midrule
\multirow{4}{*}{Lifeboat} & original        &            & 1                             & 1                          & 1                           & 1                                   & 1                       & 1                                  & 1                                 \\
                          & morton          &            & 1                             & 1                          & 1                           & 1                                   & 1                       & 1                                  & 1                                 \\
                          & shuffled        &            & 1                             & 1                          & 1                           & 1                                   & 1                       & 1                                  & 1                                 \\
                          & shuffled-Morton &            & 1                             & 1                          & 1                           & 1                                   & 1                       & 1                                  & 1                                \\
\bottomrule
\end{tabular}
\caption{TODO}
\label{tab:my-table}
\end{table*}
`;

run();
