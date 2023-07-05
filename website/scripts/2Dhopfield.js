const canvasHop = document.getElementById('canvas');
const ctxHop = canvasHop.getContext('2d');
const widthHop = canvasHop.width;
const heightHop = canvasHop.height;

const side_lenght = 64;
const gridSizeHop = canvasHop.width / side_lenght;

//window size represents the number of elements between the center of the grid
//and the last element on the right of the grid
let window_size = 10;


const J_Hop = 1 / window_size ** 2 //is this right?
const temperatureSliderHop = document.getElementById('temperatureHop-slider');
const temperatureValueHop = document.getElementById('temperatureHop-value');
const overlapValuePrint = document.getElementById('overlap');


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

function n_elements_in_window(window_size){
    return (2*window_size-1)**2-1
}

function neighborhood_of(col,row,window_size,side_lenght){
    let neigborhood = new Array(n_elements_in_window(window_size));
    let index=0;

    for (let i=-window_size+1; i<window_size; i++)
        for (let j=-window_size+1; j<window_size; j++)
            if (i!=0 || j!=0){
                neigborhood[index]=[(col+i+side_lenght)%side_lenght,(row+j+side_lenght)%side_lenght];
                index++;
            }
    return neigborhood
}

function updateHop() {
    if (!document.hidden && !isSimulationPausedHop && overlaps != undefined) {
        ctxHop.clearRect(0, 0, widthHop, heightHop);

        for (let col = 0; col < side_lenght; col++) {
            for (let row = 0; row < side_lenght; row++) {
                const spin = latticeHop[col][row];
                ctxHop.fillStyle = spin === 1 ? '#000' : '#fff';
                ctxHop.fillRect(col * gridSizeHop, row * gridSizeHop, gridSizeHop, gridSizeHop);
            }
        }

        for (let i = 0; i < 100; i++) {
            const col = Math.floor(Math.random() * side_lenght);
            const row = Math.floor(Math.random() * side_lenght);
            const spin = latticeHop[col][row];

            let deltaE = 0;
            const neighborhood=neighborhood_of(col,row,window_size,side_lenght);

            for (j = 0; j < n_elements_in_window(window_size); j++){
                const colj=neighborhood[j][0];
                const rowj=neighborhood[j][1];

                let interaction=0;
                for (p=0; p<patterns.length;p++)
                    interaction += patterns[p][col][row] * patterns[p][colj][rowj];

                deltaE+=interaction*latticeHop[colj][rowj];
            }
            deltaE *= 2 * spin* J_Hop;
            
            const temperature = parseFloat(temperatureSliderHop.value);
            temperatureValueHop.textContent = temperature.toFixed(1);

            if (deltaE <= 0 || Math.random() < Math.exp(-deltaE / temperature)) {
                latticeHop[col][row] = -spin;
                for (let j = 0; j < patterns.length; j++)
                    overlaps[j] -= 2 * spin * patterns[j][col][row];
            }
            overlapValuePrint.textContent = overlaps[0].toFixed(1);
        }
    }
    requestAnimationFrame(updateHop);
}


function handleVisibilityChangeHop(entries) {
    const isVisible = entries[0].isIntersecting;
    isSimulationPausedHop = !isVisible;
    if (isVisible) updateHop();
}

// Create an intersection observer2D
const observerHop = new IntersectionObserver(handleVisibilityChangeHop, { threshold: 0 });

// Observe the canvas element
observerHop.observe(canvasHop);