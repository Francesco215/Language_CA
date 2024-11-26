const canvasHop2D = document.getElementById('canvasHop2D');
const ctxHop2D = canvasHop2D.getContext('2d');
const widthHop2D = canvasHop2D.width;
const heightHop2D = canvasHop2D.height;

const side_lenghtHop2D = 64;
const gridSizeHop2D = canvasHop2D.width / side_lenghtHop2D;

canvasHop2D.gridSize=gridSizeHop2D;

//window size represents the number of elements between the center of the grid
//and the last element on the right of the grid
let window_size = 5;


let J_Hop2D = 1 / window_size ** 2 //is this right?
const temperatureSliderHop2D = document.getElementById('temperatureHop2D-slider');
const temperatureValueHop2D = document.getElementById('temperatureHop2D-value');
let speedValueHop2D = initSpeedSlider('Hop2D',  3000, 70);
const overlapValuePrint = document.getElementById('overlap');

let temperatureHop2D=temperatureSliderHop2D.value;

const windowSliderHop2D = document.getElementById('windowHop2D-slider');
const windowValueHop2D = document.getElementById('windowHop2D-value');


let latticeHop2D = createLattice2D(side_lenghtHop2D, side_lenghtHop2D);



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

function simulateHop2D(){
    window_size= parseFloat(windowSliderHop2D.value);
    windowValueHop2D.textContent = window_size;
    J_Hop2D = 1 / window_size ** 2;
    for (let i = 0; i < speedValueHop2D.value; i++) {
        const col = Math.floor(Math.random() * side_lenghtHop2D);
        const row = Math.floor(Math.random() * side_lenghtHop2D);
        const spin = latticeHop2D[col][row];

        let deltaE = 0;
        const neighborhood=neighborhood_of(col,row,window_size,side_lenghtHop2D);


        for (j = 0; j < n_elements_in_window(window_size); j++){
            const colj=neighborhood[j][0];
            const rowj=neighborhood[j][1];

            let interaction=0;
            for (p=0; p<patterns.length;p++)
                interaction += patterns[p][col][row] * patterns[p][colj][rowj];
            
            deltaE+=interaction*latticeHop2D[colj][rowj];
        }
        deltaE *= 2 * spin* J_Hop2D;

        if (deltaE <= 0 || Math.random() < Math.exp(-deltaE / temperatureHop2D)) {
            latticeHop2D[col][row] = -spin;
        }
    }
}

temperatureSliderHop2D.addEventListener('input', function () {
    temperatureHop2D = temperatureSliderHop2D.value;
    temperatureValueHop2D.textContent=parseFloat(temperatureHop2D).toFixed(1); 
  });

function updateHop2D() {
    if (!document.hidden && !isSimulationPausedHop2D && patterns_ready) {
        simulateHop2D();
    }
    requestAnimationFrame(updateHop2D);
}

function renderLoopHop2D(){
    if (!document.hidden && !isSimulationPausedHop2D && patterns_ready) {
        ctxHop2D.clearRect(0, 0, widthHop2D, heightHop2D);

        for (let col = 0; col < side_lenghtHop2D; col++) {
            for (let row = 0; row < side_lenghtHop2D; row++) {
                const spin = latticeHop2D[col][row];
                ctxHop2D.fillStyle = spin === 1 ? '#000' : '#fff';
                ctxHop2D.fillRect(col * gridSizeHop2D, row * gridSizeHop2D, gridSizeHop2D, gridSizeHop2D);
            }
        }
        drawHoverSquareHop2D(hoverCol,hoverRow);
    }
    requestAnimationFrame(renderLoopHop2D);
}


function handleVisibilityChangeHop2D(entries) {
    const isVisible = entries[0].isIntersecting;
    isSimulationPausedHop2D = !isVisible;
    if (isVisible) {
        requestAnimationFrame(updateHop2D);
        requestAnimationFrame(renderLoopHop2D);
    }
}

// Create an intersection observer2D
const observerHop2D = new IntersectionObserver(handleVisibilityChangeHop2D, { threshold: 0 });

// Observe the canvas element
observerHop2D.observe(canvasHop2D);

canvasHop2D.addEventListener('mousemove', handleMouseMove2D);
canvasHop2D.addEventListener('mouseleave', hideSquare);


function drawHoverSquareHop2D(col, row) {
    if (col !== -1 && row !== -1) {
      const size = window_size*gridSizeHop2D; // Adjust the size of the square as needed
      const startX = col * gridSizeHop2D + (gridSizeHop2D - size) / 2;
      const startY = row * gridSizeHop2D + (gridSizeHop2D - size) / 2;
      
      ctxHop2D.strokeStyle = '#FE6100';
      ctxHop2D.lineWidth = 2;
      ctxHop2D.strokeRect(startX, startY, size, size);
    }
  }

