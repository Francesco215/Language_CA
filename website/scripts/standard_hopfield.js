const canvasHop = document.getElementById('canvasHop');
const ctxHop = canvasHop.getContext('2d');
const widthHop = canvasHop.width;
const heightHop = canvasHop.height;

const side_lenght = 64;
const gridSizeHop = canvasHop.width/side_lenght;

const J_Hop=1/side_lenght**2
const temperatureSliderHop = document.getElementById('temperatureHop-slider');
const temperatureValueHop = document.getElementById('temperatureHop-value');
let speedValueHop = initSpeedSlider('Hop',  1000, 100);



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
let patterns_ready=false;
async function make_patterns() {
    const pattern_names = await readJson(path_names)
    patterns = new Array(pattern_names.length)
    for (let i = 0; i < pattern_names.length; i++) {
        patterns[i] = await readJson(folder_patterns + pattern_names[i]);
        for (let col = 0; col < side_lenght; col++)
            for (let row = 0; row < side_lenght; row++)
                patterns[i][col][row] = 2 * patterns[i][col][row] - 1;
    }
    patterns_ready=true;
    return patterns;
}


let latticeHop = createLattice2D(side_lenght, side_lenght);

//define overlaps in an async way
let overlaps;
let overlaps_ready=false;
async function make_overlaps(latticeHop) {
    const patterns = await make_patterns();
    overlaps = new Array(patterns.length);
    for (let i = 0; i < patterns.length; i++) {
        overlaps[i] = 0;
        for (let col = 0; col < side_lenght; col++)
            for (let row = 0; row < side_lenght; row++)
                overlaps[i] += patterns[i][col][row] * latticeHop[col][row];
    }
    overlaps_ready=true;
    return overlaps;
}
make_overlaps(latticeHop);

function updateHop() {
    if (!document.hidden && !isSimulationPausedHop && overlaps_ready) {
        ctxHop.clearRect(0, 0, widthHop, heightHop);

        for (let col = 0; col < side_lenght; col++) {
            for (let row = 0; row < side_lenght; row++) {
                const spin = latticeHop[col][row];
                ctxHop.fillStyle = spin === 1 ? '#000' : '#fff';
                ctxHop.fillRect(col * gridSizeHop, row * gridSizeHop, gridSizeHop, gridSizeHop);
            }
        }

        for (let i = 0; i < speedValueHop.value; i++) {
            const col = Math.floor(Math.random() * side_lenght);
            const row = Math.floor(Math.random() * side_lenght);
            const spin = latticeHop[col][row];

            let deltaE = 0;
            for (let j=0; j<patterns.length; j++){
                const old_energy = -J_Hop * overlaps[j]**2;
                const new_energy = -J_Hop * (overlaps[j]-2*spin*patterns[j][col][row])**2;   //check this!
                deltaE+=new_energy-old_energy
            }

            const temperature = parseFloat(temperatureSliderHop.value);
            temperatureValueHop.textContent = temperature.toFixed(1);

            if (deltaE <= 0 || Math.random() < Math.exp(-deltaE / temperature)) {
                latticeHop[col][row] = -spin;
                for (let j = 0; j < patterns.length; j++)
                    overlaps[j] -= 2 * spin * patterns[j][col][row];
            }
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