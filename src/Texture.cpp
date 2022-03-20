#pragma once

#include "Texture.h"

#include "Renderer.h"

shared_ptr<Texture> Texture::create(int width, int height, GLuint colorType, Renderer* renderer){

	GLuint handle;
	glCreateTextures(GL_TEXTURE_2D, 1, &handle);

	auto texture = make_shared<Texture>();
	texture->renderer = renderer;
	texture->handle = handle;
	texture->colorType = colorType;

	texture->setSize(width, height);

	//glTextureSubImage2D(handle, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

	return texture;
}

void Texture::setSize(int width, int height) {

	bool needsResize = this->width != width || this->height != height;

	if (needsResize) {

		glDeleteTextures(1, &this->handle);
		glCreateTextures(GL_TEXTURE_2D, 1, &this->handle);

		glTextureParameteri(this->handle, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(this->handle, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTextureParameteri(this->handle, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTextureParameteri(this->handle, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTextureStorage2D(this->handle, 1, this->colorType, width, height);

		this->width = width;
		this->height = height;
	}

}