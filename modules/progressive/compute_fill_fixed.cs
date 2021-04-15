#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_EXT_gpu_shader4 : enable

layout(local_size_x = 1, local_size_y = 1) in;

struct IndirectCommand{
	uint count;
	uint primCount;
	uint firstIndex;
	uint baseInstance;
};

layout(std430, binding = 0) buffer ssComputeFillFixed{
	int uNumPoints;
	int uOffset;
	int uFillSize;
	int uNumBatches;
	int uBatchSizes[];
};

layout(std430, binding = 1) buffer ssIndirectFillCommands{
	IndirectCommand commands[];
};

int[10] getCummulativeBatchSizes(){
	int cum = 0;
	int cummulative[10];
	for(int i = 0; i < uNumBatches; i++){
		cum = cum + uBatchSizes[i];
		cummulative[i] = cum;
	}

	return cummulative;
}

void main() {
	
	// reset commands
	for(int i = 0; i < uNumBatches; i++){
		IndirectCommand c;
		c.count = 0;
		c.primCount = 1;
		c.firstIndex = 0;

		commands[i] = c;
	}

	int[10] cum = getCummulativeBatchSizes();

	int startBatch = 0;
	for(int i = 0; i < uNumBatches; i++){
		if(uOffset <= cum[i]){
			startBatch = i;
			break;
		}
	}

	int currentBatch = startBatch;
	int remainingPoints = 1 * 1000 * 1000;

	if(uNumBatches == 1){
		int pointsToEnd = uBatchSizes[0] - uOffset;
		int budgetToEnd = min(pointsToEnd, remainingPoints);
		int stillRemaining = remainingPoints - budgetToEnd;

		{
			IndirectCommand ic;
			ic.count = budgetToEnd;
			ic.primCount = 1;
			ic.firstIndex = uOffset % 134000000;

			commands[0] = ic;
		}

		{
			IndirectCommand ic;
			ic.count = stillRemaining;
			ic.primCount = 1;
			ic.firstIndex = uOffset % 134000000;

			commands[1] = ic;
		}

		//uOffset = (uOffset + remainingPoints) % uNumPoints;
		uOffset = (uOffset + remainingPoints) % uNumPoints;
	}else{

		while(remainingPoints > 0){

			int currentPoints = cum[currentBatch] - uOffset;
			currentPoints = min(currentPoints, remainingPoints);

			IndirectCommand ic;
			ic.count = currentPoints;
			ic.primCount = 1;
			ic.firstIndex = uOffset % 134000000;

			commands[currentBatch] = ic;

			remainingPoints = remainingPoints - currentPoints;
			uOffset = (uOffset + currentPoints) % uNumPoints;
			currentBatch = (currentBatch + 1)  % uNumBatches;


		}
	}
	

	//commands[4].count = int(estimatedRemainingBudget);
}






