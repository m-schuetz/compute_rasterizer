
#include "LasLoader.h"

#include "laszip/laszip_api.h"

namespace LasLoader {

	shared_ptr<LAS> loadLas(string path) {

		auto las = make_shared<LAS>();
		las->path = path;

		laszip_POINTER laszip_reader;
		laszip_header* header;
		laszip_point* point;

		{
			laszip_BOOL is_compressed = iEndsWith(path, ".laz") ? 1 : 0;
			laszip_BOOL request_reader = 1;

			laszip_create(&laszip_reader);
			laszip_request_compatibility_mode(laszip_reader, request_reader);
			laszip_open_reader(laszip_reader, path.c_str(), &is_compressed);
			laszip_get_header_pointer(laszip_reader, &header);
			laszip_get_point_pointer(laszip_reader, &point);
			//laszip_seek_point(laszip_reader, task->firstPoint);
		}

		int64_t numPoints = std::max(uint64_t(header->number_of_point_records), header->extended_number_of_point_records);
		int64_t bpp = header->point_data_record_length;

		auto buf_XYZ = make_shared<Buffer>(12 * numPoints);
		auto buf_RGBA = make_shared<Buffer>(4 * numPoints);
		auto buf_XYZRGBA = make_shared<Buffer>(16 * numPoints);
		auto buf_XYZRGBA_uint16 = make_shared<Buffer>(12 * numPoints);
		auto buf_XYZRGBA_uint13 = make_shared<Buffer>(8 * numPoints);

		double minX = header->min_x;
		double minY = header->min_y;
		double minZ = header->min_z;

		las->minX = header->min_x;
		las->minY = header->min_y;
		las->minZ = header->min_z;
		las->maxX = header->max_x;
		las->maxY = header->max_y;
		las->maxZ = header->max_z;

		double coordinates[3];
		for (int i = 0; i < numPoints; i++) {
			laszip_read_point(laszip_reader);
			laszip_get_coordinates(laszip_reader, coordinates);

			double x = coordinates[0] - minX;
			double y = coordinates[1] - minY;
			double z = coordinates[2] - minZ;

			uint8_t r = point->rgb[0] > 255 ? point->rgb[0] / 256 : point->rgb[0];
			uint8_t g = point->rgb[1] > 255 ? point->rgb[1] / 256 : point->rgb[1];
			uint8_t b = point->rgb[2] > 255 ? point->rgb[2] / 256 : point->rgb[2];

			buf_XYZ->data_f32[3 * i + 0] = x;
			buf_XYZ->data_f32[3 * i + 1] = y;
			buf_XYZ->data_f32[3 * i + 2] = z;

			buf_RGBA->data_u8[4 * i + 0] = r;
			buf_RGBA->data_u8[4 * i + 1] = g;
			buf_RGBA->data_u8[4 * i + 2] = b;
			buf_RGBA->data_u8[4 * i + 3] = 255;

			buf_XYZRGBA->data_f32[4 * i + 0] = x;
			buf_XYZRGBA->data_f32[4 * i + 1] = y;
			buf_XYZRGBA->data_f32[4 * i + 2] = z;
			buf_XYZRGBA->data_u8[16 * i + 12] = r;
			buf_XYZRGBA->data_u8[16 * i + 13] = g;
			buf_XYZRGBA->data_u8[16 * i + 14] = b;
			buf_XYZRGBA->data_u8[16 * i + 15] = 255;

			{
				uint16_t X = x * 1000.0;
				uint16_t Y = y * 1000.0;
				uint16_t Z = z * 1000.0;

				buf_XYZRGBA_uint16->data_u16[6 * i + 0] = X;
				buf_XYZRGBA_uint16->data_u16[6 * i + 1] = Y;
				buf_XYZRGBA_uint16->data_u16[6 * i + 2] = Z;
				buf_XYZRGBA_uint16->data_u8[12 * i + 6] = r;
				buf_XYZRGBA_uint16->data_u8[12 * i + 7] = g;
				buf_XYZRGBA_uint16->data_u8[12 * i + 8] = b;
				buf_XYZRGBA_uint16->data_u8[12 * i + 9] = 253;
				buf_XYZRGBA_uint16->data_u8[12 * i + 10] = 254;
				buf_XYZRGBA_uint16->data_u8[12 * i + 11] = 255;
			}

			{
				uint64_t X = x * 300.0;
				uint64_t Y = y * 300.0;
				uint64_t Z = z * 300.0;

				uint64_t vertex = 0;

				vertex = vertex | ((X & 0b1'1111'1111'1111ul) << 0ul);
				vertex = vertex | ((Y & 0b1'1111'1111'1111ul) << 13ul);
				vertex = vertex | ((Z & 0b1'1111'1111'1111ul) << 26ul);
				vertex = vertex | (uint64_t(r) << 40ul);
				vertex = vertex | (uint64_t(g) << 48ul);
				vertex = vertex | (uint64_t(b) << 56ul);

				buf_XYZRGBA_uint13->data_u64[i] = vertex;

				if(i < 5){
					cout << vertex;
				}
			}
		}

		laszip_close_reader(laszip_reader);
		laszip_destroy(laszip_reader);

		las->numPoints = numPoints;
		las->buffers["XYZ"] = buf_XYZ;
		las->buffers["rgba"] = buf_RGBA;
		las->buffers["XYZRGBA"] = buf_XYZRGBA;
		las->buffers["XYZRGBA_uint16"] = buf_XYZRGBA_uint16;
		las->buffers["XYZRGBA_uint13"] = buf_XYZRGBA_uint13;

		return las;
	}


}