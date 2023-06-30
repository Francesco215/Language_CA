const canvas2D = document.getElementById('canvas2D');
const ctx2D = canvas2D.getContext('2d');
const width2D = canvas2D.width;
const height2D = canvas2D.height;
const gridSize2D = 20;
const numCols2D = Math.floor(width2D / gridSize2D);
const numRows2D = Math.floor(height2D / gridSize2D);
const lattice2D = createLattice2D(numCols2D, numRows2D);
const temperatureSlider2D = document.getElementById('temperature2D');
let temperature2D = temperatureSlider2D.value;
let isSimulationPaused2D = false;

async function processBMPImage(path) {
    const image = new Image();
    image.src = path;

    await new Promise(resolve => {
        image.onload = resolve;
    });

    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    context.drawImage(image, 0, 0);

    const width = image.width;
    const height = image.height;

    const imageData = context.getImageData(0, 0, width, height);
    const pixels = imageData.data;
    const result = [];

    for (let i = 0; i < pixels.length; i += 4) {
        const r = pixels[i];
        const g = pixels[i + 1];
        const b = pixels[i + 2];

        // Assuming black and white image, check if pixel is black (0) or white (255)
        if (r === 0 && g === 0 && b === 0) {
            result.push(-1);
        } else {
            result.push(1);
        }
    }

    return result;
}

const patterns=


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

function update2D() {
    if (!document.hidden && !isSimulationPaused2D) {
        ctx2D.clearRect(0, 0, width2D, height2D);

        for (let col = 0; col < numCols2D; col++) {
            for (let row = 0; row < numRows2D; row++) {
                const spin = lattice2D[col][row];
                ctx2D.fillStyle = spin === 1 ? '#000' : '#fff';
                ctx2D.fillRect(col * gridSize2D, row * gridSize2D, gridSize2D, gridSize2D);
            }
        }

        for (let i = 0; i < 10; i++) {
            const col = Math.floor(Math.random() * numCols2D);
            const row = Math.floor(Math.random() * numRows2D);
            const spin = lattice2D[col][row];

            

            if (deltaE <= 0 || Math.random() < Math.exp(-deltaE / temperature2D)) {
                lattice2D[col][row] = -spin;
            }
        }
    }
    requestAnimationFrame(update2D);
}

temperatureSlider2D.addEventListener('input', function () {
    temperature2D = temperatureSlider2D.value;
});

// Pause simulation when canvas is not visible
function handleVisibilityChange2D(entries) {
    const isVisible = entries[0].isIntersecting;
    isSimulationPaused2D = !isVisible;
}

// Create an intersection observer2D
const observer2D = new IntersectionObserver(handleVisibilityChange2D, { threshold: 0 });

// Observe the canvas element
observer2D.observe(canvas2D);


update2D();
