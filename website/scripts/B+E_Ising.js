const canvasBE = document.getElementById('canvasBE');
const ctxBE = canvasBE.getContext('2d');
const widthBE = canvasBE.width;
const heightBE = canvasBE.height/2;
const gridSizeBE = 20;
const numSpinsBE = Math.floor(widthBE / gridSizeBE);
const temperatureSliderBE = document.getElementById('temperatureBE');
const temperatureValueBE = document.getElementById('BE_temperature-value');
let temperatureBE = temperatureSliderBE.value;

let isSimulationPausedBE = false;

function createSequence(numSpins) {
  const spins = new Array(numSpins);
  for (let i = 0; i < numSpins; i++) {
    spins[i] = 1;
  }
  return spins;
}

let sequenceE = createSequence(numSpinsBE);
let sequenceB = createSequence(numSpinsBE);

const JbSlider = document.getElementById("JbBE");
const JbValue = document.getElementById("BE_Jb-value");
const JeSlider = document.getElementById("JeBE");
const JeValue = document.getElementById("BE_Je-value");
const JcSlider = document.getElementById("JcBE");
const JcValue = document.getElementById("BE_Jc-value");

let Jb = parseFloat(JbSlider.value);
JbValue.textContent = Jb.toFixed(1)
let Je = parseFloat(JeSlider.value);
JeValue.textContent = Je.toFixed(1)
let Jc = parseFloat(JcSlider.value);
JcValue.textContent = Jc.toFixed(1)



function coupledHamiltonian(se1,sb1,se2,sb2){
  const Ee = Je* (se1^se2);
  const Eb = Jb* (sb1^sb2);
  const Ec = Jc* (se2^sb2);

  return Ee+Eb+Ec;
}

function sampleNextSpins(se1,sb1){
  let interval = [];
  let Z=0;
  
  for (let se2=0; se2<2; se2++){
    for (let sb2=0; sb2<2; sb2++){
      const E=coupledHamiltonian(se1,sb1,se2,sb2);
      const p=Math.exp(-E/temperatureBE);
      interval.push(p+Z);
      Z+=p;
    }
  }
  const rand=Math.random()*Z;

  for (let i=0; i<4; i++)
    if (rand<=interval[i])
      return [Math.floor(i/2),i%2]
  
}


function updateBE() {
  if (!document.hidden && !isSimulationPausedBE) {
    ctxBE.clearRect(0, 0, widthBE, heightBE);
    
    const last_spin_B=sequenceB[numSpinsBE-1];
    const last_spin_E=sequenceE[numSpinsBE-1];
    for (let i=1; i<numSpinsBE; i++){
        sequenceB[i-1]=sequenceB[i];
        sequenceE[i-1]=sequenceE[i];
    }

    const new_spins=sampleNextSpins(last_spin_E,last_spin_B);
    sequenceE[numSpinsBE-1] = new_spins[0];
    sequenceB[numSpinsBE-1] = new_spins[1];


    for (let i=0; i<numSpinsBE; i++){
        ctxBE.fillStyle = sequenceE[i] === 0 ? '#000' : '#fff';
        ctxBE.fillRect(i * gridSizeBE, 0, gridSizeBE, heightBE);

        ctxBE.fillStyle = sequenceB[i] === 0 ? '#000' : '#fff';
        ctxBE.fillRect(i * gridSizeBE, heightBE, gridSizeBE, heightBE);
    }   

  }
  setTimeout(updateBE, 100);

}

temperatureSliderBE.addEventListener('input', function () {
  temperatureBE = temperatureSliderBE.value;
  temperatureValueBE.textContent = parseFloat(temperatureBE).toFixed(1)
});

JbSlider.addEventListener('input',function () {
  Jb = parseFloat(JbSlider.value);
  JbValue.textContent = Jb.toFixed(1)
});

JeSlider.addEventListener('input',function () {
  Je = parseFloat(JeSlider.value);
  JeValue.textContent = Je.toFixed(1)
});

JcSlider.addEventListener('input',function () {
  Jc = parseFloat(JcSlider.value);
  JcValue.textContent = Jc.toFixed(1)
});


// Pause simulation when canvas is not visible
function handleVisibilityChangeBE(entries) {
  const isVisible = entries[0].isIntersecting;
  isSimulationPausedBE = !isVisible;
}

// Create an intersection observer
const observerBE = new IntersectionObserver(handleVisibilityChangeBE, { threshold: 0 });

// Observe the canvas element
observerBE.observe(canvasBE);

updateBE();
