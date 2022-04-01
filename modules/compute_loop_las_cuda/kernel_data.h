struct XYZBatch{
	int state;
	float min_x;
	float min_y;
	float min_z;
	float max_x;
	float max_y;
	float max_z;
	int numPoints;

	int padding1;
	int padding2;
	int padding3;
	int padding4;
	int padding5;
	int padding6;
	int padding7;
	int padding8;
};

struct Mat {
	float4 rows[4];
};

struct BoundingBox{
	float4 position;   		//   16   0
	float4 size;       		//   16  16
	unsigned int color;     //    4  32
							// size: 48
};

struct BoundingData {
	unsigned int count;
	unsigned int instanceCount;
	unsigned int first;
	unsigned int baseInstance;
	// 16
	unsigned int pad0;
	unsigned int pad1;
	unsigned int pad2;
	unsigned int pad3;
	// 32
	unsigned int pad4;
	unsigned int pad5;
	unsigned int pad6;
	unsigned int pad7;
	// 48
	BoundingBox* ssBoxes;
};

struct ChangingRenderData{
	Mat uTransform;
	Mat uWorldView;
	Mat uProj;
	
	float3 uCamPos;
	int2 uImageSize;
	int uPointsPerThread;
	
	float3 uBoxMin;
	float3 uBoxMax;
	int uNumPoints;
	long long int uOffsetPointToData;
	int uPointFormat;
	long long int uBytesPerPoint;
	float3 uScale;
	
	int uEnableFrustumCulling;
};