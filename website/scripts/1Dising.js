const canvas1D = document.getElementById('canvas1D');
const ctx1D = canvas1D.getContext('2d');
const width = canvas1D.width;
const height = canvas1D.height;
const gridSize = 20;
const numSpins = Math.floor(width / gridSize);
const spins = createSpins(numSpins);
const temperatureSlider = document.getElementById('temperature');
let temperature = temperatureSlider.value;
let isSimulationPaused = false;

function createSpins(numSpins) {
  const spins = new Array(numSpins);
  for (let i = 0; i < numSpins; i++) {
    spins[i] = Math.random() > 0.5 ? 1 : -1;
  }
  return spins;
}

function update() {
  if (!document.hidden && !isSimulationPaused) {
    ctx1D.clearRect(0, 0, width, height);
    for (let i=0; i<numSpins; i++){
        ctx1D.fillStyle = spins[i] === 1 ? '#000' : '#fff';
        ctx1D.fillRect(i * gridSize, 0, gridSize, height);
    }
  
    let i=Math.floor(Math.random() * numSpins);
    
    const spin = spins[i];
    ctx1D.fillStyle = spin === 1 ? '#000' : '#fff';
    ctx1D.fillRect(i * gridSize, 0, gridSize, height);
    

    const leftNeighbor = spins[(i - 1 + numSpins) % numSpins];
    const rightNeighbor = spins[(i + 1) % numSpins];
    const sum = leftNeighbor + rightNeighbor;
    const deltaE = 2 * spin * sum;
    if (deltaE <= 0 || Math.random() < Math.exp(-deltaE / temperature)) {
    spins[i] = -spin;
    }

  }

  requestAnimationFrame(update);
}

temperatureSlider.addEventListener('input', function () {
  temperature = temperatureSlider.value;
});

// Pause simulation when canvas is not visible
function handleVisibilityChange(entries) {
  const isVisible = entries[0].isIntersecting;
  isSimulationPaused = !isVisible;
}

// Create an intersection observer
const observer = new IntersectionObserver(handleVisibilityChange, { threshold: 0 });

// Observe the canvas element
observer.observe(canvas1D);

update();