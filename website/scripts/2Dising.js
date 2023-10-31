const canvas2D = document.getElementById('canvas2D');
const ctx2D = canvas2D.getContext('2d');
const width2D = canvas2D.width;
const height2D = canvas2D.height;
const gridSize2D = 20;
const numCols2D = Math.floor(width2D / gridSize2D);
const numRows2D = Math.floor(height2D / gridSize2D);
const lattice2D = createLattice2D(numCols2D, numRows2D);
const temperatureSlider2D = document.getElementById('temperature2D');
const temperatureValue2D = document.getElementById('temperature2D-value');
let temperature2D = temperatureSlider2D.value;
let isSimulationPaused2D = false;
const simulation_speed2D_slider = document.getElementById("2D_fps-slider");
const simulation_speed2D_value = document.getElementById("2D_fps-value");
canvas2D.gridSize=gridSize2D;

function createLattice2D(cols, rows) {
  console.log(cols, rows)
  const lattice2D = new Array(cols);
  for (let col = 0; col < cols; col++) {
    lattice2D[col] = new Array(rows);
    for (let row = 0; row < rows; row++) {
      lattice2D[col][row] = Math.random() > 0.5 ? 1 : -1;
    }
  }
  return lattice2D;
}

let simulation_speed2D=30;
function simulate2D() {
  for (let i = 0; i < simulation_speed2D; i++) {
    const col = Math.floor(Math.random() * numCols2D);
    const row = Math.floor(Math.random() * numRows2D);
    const spin = lattice2D[col][row];
    const neighbors = [
      lattice2D[(col - 1 + numCols2D) % numCols2D][row],
      lattice2D[(col + 1) % numCols2D][row],
      lattice2D[col][(row - 1 + numRows2D) % numRows2D],
      lattice2D[col][(row + 1) % numRows2D],
    ];
    const sum = neighbors.reduce((acc, neighbor) => acc + neighbor, 0);
    const deltaE = 2 * spin * sum*Math.log(1+Math.sqrt(2))/2;
    if (deltaE <= 0 || Math.random() < Math.exp(-deltaE / temperature2D)) {
      lattice2D[col][row] = -spin;
    }
  }
}

function simulationLoop() {
  if (!document.hidden && !isSimulationPaused2D) {
    simulate2D(); // Update the simulation
    setTimeout(simulationLoop, 0);
  }else{
    requestAnimationFrame(simulationLoop); // Adjust the delay as needed
  }
}

function renderLoop() {
  if (!document.hidden && !isSimulationPaused2D){
    ctx2D.clearRect(0, 0, width2D, height2D);

    for (let col = 0; col < numCols2D; col++) {
      for (let row = 0; row < numRows2D; row++) {
        const spin = lattice2D[col][row];
        ctx2D.fillStyle = spin === 1 ? '#000' : '#fff';
        ctx2D.fillRect(col * gridSize2D, row * gridSize2D, gridSize2D, gridSize2D);
      }
    }
    drawHoverSquare2D(hoverCol, hoverRow);

  }
  requestAnimationFrame(renderLoop);
}

temperatureSlider2D.addEventListener('input', function () {
  temperature2D = temperatureSlider2D.value;
  temperatureValue2D.textContent=parseFloat(temperature2D).toFixed(1); 
});

simulation_speed2D_slider.addEventListener('input',function(){
  simulation_speed2D = simulation_speed2D_slider.value;
  simulation_speed2D_value.textContent=parseFloat(simulation_speed2D).toFixed(1);
})

// Pause simulation when canvas is not visible
function handleVisibilityChange2D(entries) {
  const isVisible = entries[0].isIntersecting;
  isSimulationPaused2D = !isVisible;
}

// Create an intersection observer2D
const observer2D = new IntersectionObserver(handleVisibilityChange2D, { threshold: 0 });

// Observe the canvas element
observer2D.observe(canvas2D);

canvas2D.addEventListener('mousemove', handleMouseMove2D);
canvas2D.addEventListener('mouseleave', hideSquare);

let hoverCol = -1;
let hoverRow = -1;

function handleMouseMove2D(event) {
  const rect = this.getBoundingClientRect();
  const mouseX = event.clientX - rect.left;
  const mouseY = event.clientY - rect.top;


  hoverCol = Math.floor(mouseX / this.gridSize);
  hoverRow = Math.floor(mouseY / this.gridSize);
}

function hideSquare() {
  hoverCol = -1;
  hoverRow = -1;
}

function drawHoverSquare2D(col, row) {
  if (col !== -1 && row !== -1) {
    const size = 3*gridSize2D; // Adjust the size of the square as needed
    const startX = col * gridSize2D + (gridSize2D - size) / 2;
    const startY = row * gridSize2D + (gridSize2D - size) / 2;
    
    ctx2D.strokeStyle = '#FE6100';
    ctx2D.lineWidth = 2;
    ctx2D.strokeRect(startX, startY, size, size);
  }
}

// Start the simulation loop
simulationLoop();

// Start the rendering loop
renderLoop();