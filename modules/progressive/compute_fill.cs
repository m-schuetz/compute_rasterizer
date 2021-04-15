#version 450

#extension GL_ARB_gpu_shader_int64 : enable

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

layout(std430, binding = 1) buffer ssFillFixedCommands{
	IndirectCommand commandsFixed[];
};

layout(std430, binding = 2) buffer ssFillRemainingCommands{
	IndirectCommand commandsRemaining[];
};

layout(std430, binding = 3) buffer ssTimestamps{
	uint64_t tStart;
	uint64_t tStartAdd;
	uint64_t tEndAdd;
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

#define BUDGET_MILLIES 5.0

#define NUM_FIXED 1000000.0


// void main(){

// 	uint64_t nanosConsumed = tEndAdd - tStart;
// 	double milliesConsumed = double(nanosConsumed) / double(1000000.0);
// 	double milliesRemaining = max(BUDGET_MILLIES - milliesConsumed, 0);
// 	double milliesConsumedByFillFixed = double(tEndAdd - tStartAdd) / double(1000000.0);

// 	double pointsPerMillies = NUM_FIXED / milliesConsumedByFillFixed;

// 	double estimatedRemainingBudget = pointsPerMillies * milliesRemaining;
// 	//estimatedRemainingBudget = 1.0 * estimatedRemainingBudget;



// 	// reset commands
// 	for(int i = 0; i < uNumBatches; i++){
// 		IndirectCommand c;
// 		c.count = 0;
// 		c.primCount = 1;
// 		c.firstIndex = 0;
// 		//c.baseInstance = 0;

// 		commandsFixed[i] = c;
// 		commandsRemaining[i] = c;
// 	}

// }

void main() {

	uint64_t nanosConsumed = tEndAdd - tStart;
	double milliesConsumed = double(nanosConsumed) / double(1000000.0);
	double milliesRemaining = max(BUDGET_MILLIES - milliesConsumed, 0);
	double milliesConsumedByFillFixed = double(tEndAdd - tStartAdd) / double(1000000.0);

	double pointsPerMillies = NUM_FIXED / milliesConsumedByFillFixed;

	double estimatedRemainingBudget = pointsPerMillies * milliesRemaining;
	estimatedRemainingBudget = min(estimatedRemainingBudget, uNumPoints - NUM_FIXED);

	// reset commands
	for(int i = 0; i < uNumBatches; i++){
		IndirectCommand c;
		c.count = 0;
		c.primCount = 1;
		c.firstIndex = 0;

		commandsFixed[i] = c;
		commandsRemaining[i] = c;
	}


	int[10] cum = getCummulativeBatchSizes();

	{
		int startBatch = uNumBatches - 1;
		for(int i = 0; i < uNumBatches; i++){
			if(uOffset < cum[i]){
				startBatch = i;
				break;
			}
		}

		int currentBatch = startBatch;
		int remainingPoints = int(estimatedRemainingBudget);

		while(remainingPoints > 0){

			int currentPoints = cum[currentBatch] - uOffset;
			currentPoints = min(currentPoints, remainingPoints);

			IndirectCommand ic;
			ic.count = currentPoints;
			ic.primCount = 1;
			ic.firstIndex = uOffset % 134000000;

			commandsRemaining[currentBatch] = ic;

			remainingPoints = remainingPoints - currentPoints;
			uOffset = (uOffset + currentPoints) % uNumPoints;
			currentBatch = (currentBatch + 1)  % uNumBatches;
		}

		commandsFixed[5].count = int(estimatedRemainingBudget);
		//commandsFixed[5].count = int(estimatedRemainingBudget);
	}

	{
		int startBatch = uNumBatches - 1;
		for(int i = 0; i < uNumBatches; i++){
			if(uOffset < cum[i]){
				startBatch = i;
				break;
			}
		}

		int currentBatch = startBatch;
		int remainingPoints = int(NUM_FIXED);

		while(remainingPoints > 0){

			int currentPoints = cum[currentBatch] - uOffset;
			currentPoints = min(currentPoints, remainingPoints);

			IndirectCommand ic;
			ic.count = currentPoints;
			ic.primCount = 1;
			ic.firstIndex = uOffset % 134000000;

			commandsFixed[currentBatch] = ic;

			remainingPoints = remainingPoints - currentPoints;
			uOffset = (uOffset + currentPoints) % uNumPoints;
			currentBatch = (currentBatch + 1)  % uNumBatches;
		}

		//commandsFixed[5].count = commandsFixed[0].count;
	}
	
}






