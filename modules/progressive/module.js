
watchJS(`${rootDir}/modules/progressive/render_progressive.js`);
//watchJS(`${rootDir}/modules/progressive/render_progressive_single_compute.js`);
//watchJS(`${rootDir}/modules/progressive/render_progressive_from_previous.js`);
watchJS(`${rootDir}/modules/progressive/PointCloudProgressive.js`);

monitorJS(`${rootDir}/modules/progressive/runtime/execute3.js`);
monitorJS(`${rootDir}/modules/progressive/runtime/execute4.js`);
monitorJS(`${rootDir}/modules/progressive/runtime/execute5.js`);
monitorJS(`${rootDir}/modules/progressive/runtime/execute6.js`);
monitorJS(`${rootDir}/modules/progressive/runtime/niederweiden.js`);
monitorJS(`${rootDir}/modules/progressive/runtime/retz.js`);
monitorJS(`${rootDir}/modules/progressive/runtime/wien.js`);

monitorJS(`${rootDir}/modules/progressive/eval/convergence_retz.js`);
monitorJS(`${rootDir}/modules/progressive/eval/arbegen.js`);

monitorJS(`${rootDir}/modules/progressive/eval/benchmark_heidentor_progressive.js`);
monitorJS(`${rootDir}/modules/progressive/eval/benchmark_retz_progressive.js`);
monitorJS(`${rootDir}/modules/progressive/eval/benchmark_wien.js`);
monitorJS(`${rootDir}/modules/progressive/eval/benchmark_candi_sari.js`);
monitorJS(`${rootDir}/modules/progressive/eval/benchmark_matterhorn.js`);
monitorJS(`${rootDir}/modules/progressive/eval/benchmark_morro_bay.js`);
monitorJS(`${rootDir}/modules/progressive/eval/benchmark_morro_bay_1billion.js`);
monitorJS(`${rootDir}/modules/progressive/eval/screenshots_wien.js`);
monitorJS(`${rootDir}/modules/progressive/eval/vr_wien.js`);
monitorJS(`${rootDir}/modules/progressive/eval/vr_candi_sari.js`);

monitorJS(`${rootDir}/modules/progressive/progressive_pointcloud.js`);

"module progressive";