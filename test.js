const canvas2D = document.getElementById('canvas');
const ctxHop = canvas2D.getContext('2d');
const width2D = canvas2D.width;
const height2D = canvas2D.height;

const side_lenght = 32;

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


const folder_patterns = 'website/images/patterns_json/'
const path_names = folder_patterns + 'patterns.json'


//define patterns in an async way
let patterns;
async function make_patterns() {
    const pattern_names = await readJson(path_names)
    patterns = new Array(pattern_names.length)
    for (let i = 0; i < pattern_names.length; i++) {
        patterns[i] = await readJson(folder_patterns + pattern_names[i]);
        for (let col = 0; col < side_lenght; col++)
            for (let row = 0; row < side_lenght; row++)
                patterns[i][col][row] = 2 * patterns[i][col][row] - 1;
    }
    return patterns;
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

let latticeHop = createLattice2D(side_lenght, side_lenght);

//define overlaps in an async way
let overlaps;
async function make_overlaps() {
    const patterns = await make_patterns();
    overlaps = new Array(patterns.length);
    for (let i = 0; i < patterns.length; i++) {
        overlaps[i] = 0;
        for (let col = 0; col < side_lenght; col++)
            for (let row = 0; row < side_lenght; row++)
                overlaps[i] += patterns[i][col][row] * latticeHop[col][row];
    }
    return overlaps;
}
make_overlaps();

function updateHop() {
    if (!document.hidden && !isSimulationPausedHop && overlaps!=undefined) {
        ctxHop.clearRect(0, 0, widthHop, heighthop);

        for (let col = 0; col < side_lenght; col++) {
            for (let row = 0; row < side_lenght; row++) {
                const spin = latticeHop[col][row];
                ctxHop.fillStyle = spin === 1 ? '#000' : '#fff';
                ctxHop.fillRect(col * gridSizeHop, row * gridSizeHop, gridSizeHop, gridSizeHop);
            }
        }

        for (let i = 0; i < 10; i++) {
            const col = Math.floor(Math.random() * side_lenght);
            const row = Math.floor(Math.random() * side_lenght);
            const spin = latticeHop[col][row];

            let deltaE = 0;
            for (let j=0; j<patterns.length; j++){
                const new_energy = -J_Hop * overlaps[j]**2;
                const old_energy = -J_Hop * (overlaps[j]-2*spin)**2;   //check this!
                deltaE+=new_energy-old_energy
            }
            if (deltaE <= 0 || Math.random() < Math.exp(-deltaE / temperature2D)) {
                latticeHop[col][row] = -spin;
                for (let j = 0; j < patterns.length; j++)
                    latticeHop[row][col]=-spin;
            }
        }
    }
    requestAnimationFrame(updateHop);
}