
# About

This repository contains the source code for our papers about compute rasterization of point clouds. The project is currently crude and difficult to use, but we plan to add the option to drag&drop your own point clouds after vacation. We will also add a test data set by then.

* ["Software Rasterization of 2 Billion Points in Real-Time"](https://www.cg.tuwien.ac.at/research/publications/2022/SCHUETZ-2022-PCC/) <br>
Current branch

* ["Rendering Point Clouds with Compute Shaders and Vertex Order Optimization"](https://www.cg.tuwien.ac.at/research/publications/2021/SCHUETZ-2021-PCC/)<br>
In branch [compute_rasterizer_2021](https://github.com/m-schuetz/compute_rasterizer/tree/compute_rasterizer_2021)

<img src="docs/teaser.jpg">

[paper](https://www.cg.tuwien.ac.at/research/publications/2022/SCHUETZ-2022-PCC/) - <a href="https://www.youtube.com/watch?v=9h-ElMfVIOY">video</a>

# Getting Started

* Clone the repository
* Modify ./src/main.cpp so that it loads your own data set.
    * Add a new setting
	* Change ```Setting setting = settings["..."];``` to your own setting.
* Compile build/ComputeRasterizer.sln with Visual Studio 2022.
* Run (ctrl + f5)

Currently, only point clouds in LAS format are supported.

<table>
	<tr>
		<th>Method</th>
		<th>Location</th>
	</tr>
	<tr>
		<td>basic</td>
		<td><a href="./modules/compute_loop_las">./modules/compute_loop_las</a></td>
	</tr>
	<tr>
		<td>prefetch</td>
		<td><a href="./modules/compute_loop_las2">./modules/compute_loop_las2</a></td>
	</tr>
	<tr>
		<td>hqs</td>
		<td><a href="./modules/compute_loop_las_hqs">./modules/compute_loop_las_hqs</a></td>
	</tr>
	<tr>
		<td>LOD</td>
		<td><a href="./modules/compute_loop_nodes">./modules/compute_loop_nodes</a></td>
	</tr>
	<tr>
		<td>LOD hqs</td>
		<td><a href="./modules/compute_loop_nodes_hqs">./modules/compute_loop_nodes_hqs</a></td>
	</tr>
</table>