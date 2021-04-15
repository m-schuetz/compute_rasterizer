
let methodLabelMapping = {
	"GL_POINTS": "GL_POINTS",
	"compute": "atomicMin",
	"compute(early-depth)": "+ early-z",
	"compute(ballot)": "+ reduce",
	"compute(early-depth, ballot)": "+ reduce & early-z",
	"compute(early-depth, ballotm dedup)": "dedup",
	"compute(just-set)": "just-set",
	"compute(hqs)": "HQS",
	"compute(hqs, 1x64bit)": "1 atomic",
	"compute(hqs, 1x64bit, float)": "1 atomic float",
	"compute(hqs, 1x64bit, protected)": "HQS 1x, robust",
	"guenther": "busy-loop",
	"GL_POINTS": "GL_POINTS",
	"atomicMin": "atomicMin",
	"reduce": "reduce",
	"early-z": "early-z",
	"reduce,early-z": "reduce,early-z",
	"dedup": "dedup",
	"just-set": "just-set",
	"HQS": "HQS",
	"HQS1x,protected": "HQS1R",
};

let orderLabelMapping = {
	"morton-shuffled": "shuffled-morton",
};

function barchart(data, args = {}){
	let groups = d3.map(data, function(d){return(d.group)}).keys()
	let subgroups = data.columns.slice(1)

	let margin = {top: 48, right: 40, bottom: 42, left: 60};
	let width = 1240 - margin.left - margin.right;
	let height = 600 - margin.top - margin.bottom;
	let chartWidth = 890;

	let maxTime = Math.max(...data.map(g => Object.values(g).slice(1).map(str => Number(str))).flat());
	let roundedMax = Math.ceil(maxTime / 0.5) * 0.5;
	let yRange = args.yRange ?? [0, roundedMax];
	let colorScheme = [
		'#1f78b4',
		'#fdbf6f',
		'#33a02c',
		'#a6cee3',
		'#e31a1c',
		'#b2df8a',
		'#fb9a99',
		'#ff7f00'];

	if(subgroups.length === 8){
		colorScheme = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00'];
	}else if(subgroups.length === 9){
		colorScheme = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6'];
		// move violet a little to the left to color the just-set method
		colorScheme = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#cab2d6','#fdbf6f','#ff7f00'];
	}else if(subgroups.length === 10){
		colorScheme = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a'];
	}else{
		colorScheme = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00'];
	}

	console.log(data)
	window.data = data;


	let container = document.createElement("div");
	document.body.append(container);
	let d3Container = d3.select(container);
	var svg = d3Container
	.append("svg")
		.attr("width", width + margin.left + margin.right)
		.attr("height", height + margin.top + margin.bottom)
	.append("g")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

	// Add X axis
	var x = d3.scaleBand()
		.domain(groups)
		.range([0, chartWidth])
		.padding([0.2])
	svg.append("g")
		.attr("transform", `translate(0, ${height})`)
		.call(d3.axisBottom(x).tickSize(2));

	// Add Y axis
	var y = d3.scaleLinear()
		.domain(yRange)
		.range([ height, 0 ]);
	svg.append("g")
		.call(d3.axisLeft(y));

	let INNER_WIDTH = chartWidth;
	const yAxisGrid = d3.axisLeft(y).tickSize(-INNER_WIDTH).tickFormat('').ticks(10);
	svg.append('g')
		.attr('class', 'y axis-grid')
		.call(yAxisGrid);
		

	// Y Axis Label
	svg.append("text")
		.attr("class", "y label")
		.attr("text-anchor", "end")
		.attr("y", -42)
		.attr("x", 5)
		.attr("dy", ".75em")
		.text("(ms)");

	// Another scale for subgroup position?
	var xSubgroup = d3.scaleBand()
		.domain(subgroups)
		.range([0, x.bandwidth()])
		.padding([0.05])

	// color palette = one color per subgroup
	var color = d3.scaleOrdinal()
		.domain(subgroups)
		.range(colorScheme);

	// Show the bars
	svg.append("g")
		.selectAll("g")
		// Enter in data = loop group per group
		.data(data)
		.enter()
		.append("g")
		.attr("transform", function(d) { return "translate(" + x(d.group) + ",0)"; })
		.selectAll("rect")
		.data(function(d) { return subgroups.map(function(key) { return {key: key, value: d[key]}; }); })
		.enter().append("rect")
		.attr("x", function(d) { return xSubgroup(d.key); })
		.attr("y", function(d) { return y(d.value); })
		.attr("width", xSubgroup.bandwidth())
		.attr("height", function(d) { return height - y(d.value); })
		.attr("fill", function(d) { return color(d.key); });

	{

		let makeLabel = (x, y, label) => {
			svg.append("text")
				.attr("class", "y label")
				.attr("y", y)
				.attr("x", x)
				.attr("dy", ".75em")
				.text(label);
		};

			let xOffset = 44;
		for(let item of data){
			let entries = Object.entries(item);

			for(let entry of entries){
				if(entry[1] > yRange[1]){
					console.log(entry[0]);

					let val = Number(entry[1]);
					let str = val.toFixed(1);
					makeLabel(xOffset, -42, `• ${str} ms`);
					xOffset += 212;
				}
			}
		}


	}
	

	{ // LEGEND
		var legspacing = 40;

		let VALUES = subgroups;
		let LABELS = subgroups;

		var legend = svg.selectAll(".legend")
			.data(VALUES)
			.enter()
			.append("g")

		legend.append("rect")
			.attr("fill", color)
			.attr("width", 20)
			.attr("height", 20)
			.attr("y", function (d, i) {
				return i * legspacing - 60 + 65;
			})
			.attr("x", chartWidth + 20);

		legend.append("text")
			.attr("class", "label")
			.attr("y", function (d, i) {
				return i * legspacing - 43 + 65;
			})
			.attr("x", chartWidth + 50)
			.attr("text-anchor", "start")
			.text(function (d, i) {
				return LABELS[i];
			});

		let svgNode = svg.node();

		{ // 60fps marker
			let path = document.createElementNS('http://www.w3.org/2000/svg', "path");
			let y = 375 + 40;
			path.setAttributeNS(null, "d", `M 910 ${y} L 930 ${y}`);
			path.setAttributeNS(null, "stroke", "rgb(255, 20, 20)");
			path.setAttributeNS(null, "stroke-width", "4px");
			path.setAttributeNS(null, "stroke-dasharray", "0.3em 0.15em");

			svgNode.append(path);
		}

		{ // 60fps text
			let text = document.createElementNS('http://www.w3.org/2000/svg', "text");
			text.setAttributeNS(null, "class", "label");
			text.textContent = "16.6ms (60fps)";
			text.setAttributeNS(null, "x", 940);
			text.setAttributeNS(null, "y", 382 + 40);
			
			svgNode.append(text);
		}



	}

	// 60 fps line
	if(yRange[1] > 16){
		const yAxisGrid = d3
			.axisLeft(y)
			.tickSize(-INNER_WIDTH)
			.tickFormat('')
			.tickValues([16.66]);

		svg.append('g')
			.attr('class', 'mark_realtime')
			.call(yAxisGrid);

		
	}

}

function getCsv(json, model){

	console.log(new Set(json.scenarios.map(s => s.sorting)));
	let groups = [ "original", "morton", "shuffled", "morton-shuffled"];
	// let subgroups = [
	// 	"GL_POINTS", 
	// 	"guenther",
	// 	"compute", 
	// 	"compute(ballot)", 
	// 	"compute(early-depth)", 
	// 	"compute(early-depth, ballot)", 
	// 	"compute(just-set)", 
	// 	"compute(hqs)", 
	// 	"compute(hqs, 1x64bit, protected)",
	// ];
	let subgroups = [
		"GL_POINTS", 
		"guenther",
		"atomicMin", 
		"reduce", 
		"early-z", 
		"reduce,early-z", 
		"dedup", 
		"just-set", 
		"HQS", 
		"HQS1x,protected",
	];
	let subgroupLabels = subgroups.map(method => methodLabelMapping[method]);

	let csv = `group,${subgroupLabels.map(s => `"${s}"`).join(",")}\n`;

	let data = json.scenarios.filter(s => s.model === model);
	for(let group of groups){
		let timings = subgroups.map(s => data.find(scenario => scenario.sorting === group).timings.find(t => t.method === s));
		let durations = timings.map(t => t.durations.find(d => d.label === "frame")).map(v => v.avg);

		let groupName = orderLabelMapping[group] ?? group;
		csv += `${groupName},${durations.join(",")}\n`;
	}

	console.log(csv)

	return csv;
}

export async function display(benchmark, args = {}){
	let models = new Set(benchmark.scenarios.map(s => s.model));

	for(let model of models){

		let title = document.createElement("h1");
		title.innerText = model;
		document.body.append(title);

		let csv = getCsv(benchmark, model);
		let data = d3.csvParse(csv);

		barchart(data, args);
	}
}