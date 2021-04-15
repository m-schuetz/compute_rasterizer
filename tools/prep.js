

const fs = require("fs");


let directory = "D:/dev/pointclouds/tuwien_baugeschichte/Museum Affandi_las export";


let files = fs.readdirSync(directory);

files = files.filter(file => file.endsWith("laz") || file.endsWith("las") && !file.startsWith("batch"));
files.sort((a, b) => {
    if(a.length != b.length){
        return a.length - b.length;
    }else{
        return a.localeCompare(b);
    }
}); 

files = files.map(file => {
    let stat = fs.lstatSync(`${directory}/${file}`);

    return {name: file, size: stat.size};
});

console.log(files);

let maxBatchSize = 450000000;
let batchSize = 0;
let batches = [];
let batch = [];

for(let file of files){

    let fd = fs.openSync(`${directory}/${file.name}`);
    let buffer = Buffer.alloc(227);
    fs.readSync(fd, buffer, 0, 227);

    let numPoints = buffer.readUInt32LE(107);
    file.numPoints = numPoints;

    if(batchSize + numPoints > maxBatchSize){
        batches.push(batch);
        batch = [];
        batchSize = 0;
    }else{
        batch.push(file);
        batchSize += numPoints;
    }
}


let i = 0;
for(let batch of batches){

    let batchSize = batch.reduce( (a, file) => a + file.numPoints, 0);

    let lf = batch.map(file => `"${file.name}"`).join(" ");
    let command = `lasmerge.exe -i ${lf} -o batch_${i}.las`;

    console.log("");
    console.log(`# BATCH ${i}, #points: ${batchSize.toLocaleString()}`);
    console.log(command);

    i++;
}

//console.log(files.join("\n"));


