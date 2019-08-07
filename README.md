# About

This repository contains two variations on how to render point clouds with compute shaders that are up to ten times faster than gl.drawArrays(GL_POINT, ...). In the folders _compute_ and compute_hqs_ you will find a regular non-anti-aliased version and a high-quality splatting version.

Please note that benchmarking results were obtained with the original order of points in our test files and for pixel sizes of 1 pixel. The order of points has a significant impact on performance and shuffling points severely reduces the performance gains of our compute shader approach. It can even make the high-quality compute shader version slower than the regular gl.drawArrays(GL_POINT, ...). In-depth benchmarking is subject to future work. 

## compute

Up to 2-10 times faster than GL_POINT.

* render.cs: Encodes depth and colors into a 64 bit integer, and stores the closest fragment into an SSBO with atomicMin. 
* resolve.cs: Transfers color values from the SSBO to an actual OpenGL texture.

## compute_hqs

Up to 2-3 times faster than GL_POINT.

A compute shader implementation of _High-Quality Surface Splatting on Today’s GPUs_[1]. Instead of rendering only the closest point, this approach computes the average of all points within a certain depth-range, which leads to pretty good anti-aliasing within a pixel. Currently doesn't do anti-aliasing within pixels, though.

* render_depth.cs: Creates a depth-buffer using the basic compute approach.
* render_attribute.cs: Computes the sum of colors of all points that are at most 1% behind the closest point in a pixel. Also counts how many points contribute to the sum.
* resolve.cs: Computes the average color of contributing points via: _sum(colors) / length(colors)_. Writes the result into an OpenGL texture.


[1] Mario Botsch, Alexander Hornung, Matthias Zwicker, Leif Kobbelt, "High-Quality Surface Splatting on Today’s GPUs", Eurographics Symposium on Point-Based Graphics (2005)