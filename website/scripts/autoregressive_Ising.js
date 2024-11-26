const canvasAR = document.getElementById('canvasAR');
const ctxAR = canvasAR.getContext('2d');
const widthAR = canvasAR.width;
const heightAR = canvasAR.height;
const gridSizeAR = 20;
const numSpinsAR = Math.floor(widthAR / gridSizeAR);
const temperatureSliderAR = document.getElementById('temperatureAR');
const temperatureValueAR = document.getElementById('AR_temperature-value');
const fpsSliderAR = document.getElementById("AR_fps-slider");
const fpsValueAR = document.getElementById("AR_fps-value");
let temperatureAR = temperatureSliderAR.value;
let isSimulationPausedAR = false;


function createSequence(numSpins) {
  const spins = new Array(numSpins);
  for (let i = 0; i < numSpins; i++) {
    spins[i] = 1;
  }
  return spins;
}

let sequenceAR = createSequence(numSpinsAR)



function updateAR() {
  if (!document.hidden && !isSimulationPausedAR) {
    ctxAR.clearRect(0, 0, widthAR, heightAR);
    
    p =  1/(1+Math.exp(-1/temperatureAR));
    const last_spin=sequenceAR[sequenceAR.length-1];
    for (let i=1; i<sequenceAR.length; i++)
        sequenceAR[i-1]=sequenceAR[i];

    sequenceAR[sequenceAR.length-1] = Math.random() > p ^ last_spin;

    for (let i=0; i<numSpinsAR; i++){
        ctxAR.fillStyle = sequenceAR[i] === 0 ? '#000' : '#fff';
        ctxAR.fillRect(i * gridSizeAR, 0, gridSizeAR, heightAR);
    }   

  }
}

temperatureSliderAR.addEventListener('input', function () {
  temperatureAR = temperatureSliderAR.value;
  temperatureValueAR.textContent = parseFloat(temperatureAR).toFixed(1)
});

// Pause simulation when canvas is not visible
function handleVisibilityChangeAR(entries) {
  const isVisible = entries[0].isIntersecting;
  isSimulationPausedAR = !isVisible;
}

// Create an intersection observer
const observerAR = new IntersectionObserver(handleVisibilityChangeAR, { threshold: 0 });

// Observe the canvas element
observerAR.observe(canvasAR);

initFpsSlider(
  fpsSliderAR,
  fpsValueAR,
  document.getElementById('AR_fps-slider-tickmarks'),
  updateAR
);