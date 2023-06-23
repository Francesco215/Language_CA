const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const width = canvas.width;
const height = canvas.height;
const gridSize = 20;
const numCols = Math.floor(width / gridSize);
const numRows = Math.floor(height / gridSize);
const lattice = createLattice(numCols, numRows);
const temperatureSlider = document.getElementById('temperature');
let temperature = temperatureSlider.value;

function createLattice(cols, rows) {
  const lattice = new Array(cols);
  for (let col = 0; col < cols; col++) {
    lattice[col] = new Array(rows);
    for (let row = 0; row < rows; row++) {
      lattice[col][row] = Math.random() > 0.5 ? 1 : -1;
    }
  }
  return lattice;
}

function update() {
  ctx.clearRect(0, 0, width, height);

  for (let col = 0; col < numCols; col++) {
    for (let row = 0; row < numRows; row++) {
      const spin = lattice[col][row];
      ctx.fillStyle = spin === 1 ? '#000' : '#fff';
      ctx.fillRect(col * gridSize, row * gridSize, gridSize, gridSize);
    }
  }

  for (let i = 0; i < 10; i++) {
    const col = Math.floor(Math.random() * numCols);
    const row = Math.floor(Math.random() * numRows);
    const spin = lattice[col][row];
    const neighbors = [
      lattice[(col - 1 + numCols) % numCols][row],
      lattice[(col + 1) % numCols][row],
      lattice[col][(row - 1 + numRows) % numRows],
      lattice[col][(row + 1) % numRows],
    ];
    const sum = neighbors.reduce((acc, neighbor) => acc + neighbor, 0);
    const deltaE = 2 * spin * sum;
    if (deltaE <= 0 || Math.random() < Math.exp(-deltaE / temperature)) {
      lattice[col][row] = -spin;
    }
  }

  requestAnimationFrame(update);
}

temperatureSlider.addEventListener('input', function () {
  temperature = temperatureSlider.value;
});

update();
