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

async function readJson(url) {
    try {
        const response = await fetch(url);
        const jsonData = await response.json();
        return jsonData;
    } catch (error) {
        console.error(error);
        return null;
    }
}

patterns_path='website/images/patterns_json/'
const patterns_file=readJson(patterns_path +'patterns.json')

async function create_interaction(side_lenght){

    const interaction = new Array(side_lenght);
    
    for (let i = 0; i < pattern_names.length; i++) {
        const pattern = await readJson(patterns_path + pattern_names[i]);
        
        for (let col = 0; col < side_lenght; col++) {
            if (i===0) interaction[col] = new Array(side_lenght);
            
            for (let row = 0; row < side_lenght; row++) {
                if (i === 0) interaction[col][row]=0;
                interaction[col][row]+=pattern[row][col];
            }
        }
    }
    return interaction
}


function createLattice2D(cols, rows) {
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
