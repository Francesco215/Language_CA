const canvas1D = document.getElementById("canvas1D");
const ctx1D = canvas1D.getContext("2d");
const width = canvas1D.width;
const height = canvas1D.height;
const gridSize = 20;
const numSpinsVertical = Math.floor(height/gridSize);
const numSpinsHorizontal = Math.floor(width / gridSize);
const numSpins = numSpinsHorizontal*numSpinsVertical - (numSpinsHorizontal-1)*Math.floor(numSpinsVertical/2);
const spins = createSpins(numSpins);
const temperatureSlider = document.getElementById("temperature");
const temperatureValue1D = document.getElementById('1D_temperature-value');
let temperature = temperatureSlider.value;
let isSimulationPaused1D = false;

temperatureValue1D.textContent = parseFloat(temperature).toFixed(1)
let speedValue1D = initSpeedSlider('1D',  3000, 30);

function createSpins(numSpins) {
  const spins = new Array(numSpins);
  for (let i = 0; i < numSpins; i++) {
    spins[i] = Math.random() > 0.5 ? 1 : -1;
  }
  return spins;
}

function simulate1D() {
  for (let j = 0; j < speedValue1D.value; j++) {
    let i = Math.floor(Math.random() * numSpins);

    const spin = spins[i];
    ctx1D.fillStyle = spin === 1 ? "#000" : "#fff";
    ctx1D.fillRect(i * gridSize, 0, gridSize, height);

    const leftNeighbor = spins[(i - 1 + numSpins) % numSpins];
    const rightNeighbor = spins[(i + 1) % numSpins];
    const sum = leftNeighbor + rightNeighbor;
    const deltaE = 2 * spin * sum;
    if (deltaE <= 0 || Math.random() < Math.exp(-deltaE / temperature)) {
      spins[i] = -spin;
    }
  }
}

function simulationLoop1D() {
  if (!document.hidden && !isSimulationPaused1D) {
    simulate1D();
  }
  requestAnimationFrame(simulationLoop1D);
}

function renderLoop1D() {
  if (!document.hidden && !isSimulationPaused1D) {
    ctx1D.clearRect(0, 0, width, height);
    for (let i = 0; i < numSpins; i++) {
      ctx1D.fillStyle = spins[i] === 1 ? "#000" : "#ccc";
      const position = Serpentine(i, numSpinsHorizontal)
      ctx1D.fillRect(position[0] * gridSize, position[1]*gridSize, gridSize, gridSize);
    }
  }
  requestAnimationFrame(renderLoop1D);
}

temperatureSlider.addEventListener("input", function () {
  temperature = temperatureSlider.value;
  temperatureValue1D.textContent = parseFloat(temperature).toFixed(1)
});

// Pause simulation when canvas is not visible
function handleVisibilityChange1D(entries) {
  const isVisible = entries[0].isIntersecting;
  isSimulationPaused1D = !isVisible;
}

// Create an intersection observer
const observer = new IntersectionObserver(handleVisibilityChange1D, {
  threshold: 0,
});

// Observe the canvas element
observer.observe(canvas1D);

simulationLoop1D();
renderLoop1D();

function Serpentine(i, nSH) {
  //nSH represents the variable numSpinsHorizontal
  if (i<nSH) return [i, 0]
  if (i==nSH) return [nSH-1, 1]
  if (i<=2*nSH) return [2*nSH-i, 2]
  if (i==2*nSH+1) return [0,3]
  
  
  let recursive=Serpentine(i%(2*(nSH+1)),nSH);
  recursive[1]+=4*Math.floor(i/(2*nSH+2));
  
  return recursive
}
