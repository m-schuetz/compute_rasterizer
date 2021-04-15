


const distances = {
	earth_moon: 384_400_000,
	sun_mars: 230_000_000_000,
};

// taken from: https://stackoverflow.com/questions/2901102/how-to-print-a-number-with-commas-as-thousands-separators-in-javascript
function addCommas(x) {
	return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

const possibleValues32 = 2 ** 32; // 4 billion
const possibleValues40 = 2 ** 40; // 1 trillion

console.log(addCommas(possibleValues32)); 
console.log(addCommas(possibleValues40)); 

prec32mm = 10_000 * (1_000_000 / possibleValues32);
prec40mm = 10_000 * (1_000_000 / possibleValues40);

console.log(prec32mm);
console.log(prec40mm);











