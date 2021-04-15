
const PointAttribute = {
	POSITION_CARTESIAN: {bytes: 12},
	COLOR_PACKED: {bytes: 4},
	RGBA: {bytes: 4},
	INTENSITY: {bytes: 2},
	CLASSIFICATION: {bytes: 1},
	RETURN_NUMBER: {bytes: 1},
	NUMBER_OF_RETURNS: {bytes: 1},
	SOURCE_ID: {bytes: 2},
	GPS_TIME: {bytes: 8},
	NORMAL_SPHEREMAPPED: {bytes: 2},
};


class PointAttributes{

	constructor(elements){
		this.elements = elements;
		this.bytes = elements.reduce( (sum, attribute) => (sum + attribute.bytes), 0);

	}

	contains(value){
		for(let element of this.elements){
			if(element === value){
				return true;
			}
		}

		return false;
	}

	offsetOf(value){
		let offset = 0;

		for(let element of this.elements){
			if(element === value){
				return offset;
			}

			offset += element.bytes;
		}

		return null;
	}

};