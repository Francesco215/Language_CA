
// Get the canvas element and its 2D rendering context
const canvasFC = document.getElementById('FC_ising-canvas');
const contextFC = canvasFC.getContext('2d');

// Get the temperature slider and its value element
const temperatureSliderFC = document.getElementById('FC_temperature-slider');
const temperatureValueFC = document.getElementById('FC_temperature-value');

// Define Ising model parameters
const N_FC = 100;   // Number of spins
const J_FC = 1/N_FC;     // Interaction strength
const radiusFC = 10; //radius of the circles
const RadiusFC = canvasFC.width*.45

const centerX = canvasFC.width / 2;
const centerY = canvasFC.height / 2;
const angle = (2 * Math.PI) / N_FC;

// Define the state of each spin (0 or 1)
let spinsFC = [];
let spin_sumFC=0;

// Initialize the spins randomly
for (let i = 0; i < N_FC; i++) {
    spinsFC[i] = Math.random() > 0.5 ? 1 : -1;
    spin_sumFC+=spinsFC[i]
}

//initialize uuu
let hoveredSpinFC = -1;
let isSimulationPausedFC = false;


// Function to draw the Ising model simulation
function drawIsingModelFC() {
    // Clear the canvas
    contextFC.clearRect(0, 0, canvasFC.width, canvasFC.height);

    // Handle mouse movement
    canvasFC.onmousemove = function(event) {
        const rect = canvasFC.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;

        hoveredSpinFC = getHoveredSpinFC(mouseX, mouseY);
    };

    // Draw lines connecting the circles if hovered
    if (hoveredSpinFC !== -1)
        for (let i = 0; i < N_FC; i++) {
            const x = centerX + Math.cos(i * angle) * (RadiusFC);
            const y = centerY + Math.sin(i * angle) * (RadiusFC);

            if (i !== hoveredSpinFC) {
                const hoveredX = centerX + Math.cos(hoveredSpinFC * angle) * (RadiusFC);
                const hoveredY = centerY + Math.sin(hoveredSpinFC * angle) * (RadiusFC);

                contextFC.beginPath();
                contextFC.moveTo(x, y);
                contextFC.lineTo(hoveredX, hoveredY);
                contextFC.strokeStyle = 'gray';
                contextFC.stroke();
            }
        }

    //draw the circles
    for (let i = 0; i < N_FC; i++) {
        const x = centerX + Math.cos(i * angle) * (RadiusFC);
        const y = centerY + Math.sin(i * angle) * (RadiusFC);

        contextFC.beginPath();
        contextFC.arc(x, y, radiusFC, 0, 2 * Math.PI);
        contextFC.fillStyle = spinsFC[i] === 1 ? 'black' : 'white';
        contextFC.fill();
        contextFC.stroke();

        
    }
}

// Helper function to get the index of the hovered spin
function getHoveredSpinFC(mouseX, mouseY) {
    for (let i = 0; i < N_FC; i++) {
        const x = centerX + Math.cos(i * angle) * (RadiusFC);
        const y = centerY + Math.sin(i * angle) * (RadiusFC);

        const dx = x - mouseX;
        const dy = y - mouseY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance <= radiusFC) {
            return i;
        }
    }

    return -1;
}

// Call the draw function to initially render the simulation
drawIsingModelFC();


// Function to update the simulation
function updateSimulationFC() {
    if (!document.hidden && !isSimulationPausedFC){
        // Get the temperature value from the slider
        const temperature = parseFloat(temperatureSliderFC.value);
        temperatureValueFC.textContent = temperature.toFixed(1);

        // Implement the update logic based on the Ising model dynamics
        const randomSpinIndex = Math.floor(Math.random() * N_FC);
        const randomSpin = spinsFC[randomSpinIndex];

        const old_energy=-J_FC*spin_sumFC**2;
        const new_energy=-J_FC*(spin_sumFC-2*randomSpin)**2;  
        // Calculate the energy change upon flipping the random spin
        const energyChange = new_energy-old_energy;

        // Implement the Metropolis algorithm to accept or reject the flip
        if (energyChange <= 0) {
            // Accept the flip
            spinsFC[randomSpinIndex] = -randomSpin;
            spin_sumFC=spin_sumFC-2*randomSpin;

        } else {
            // Accept the flip with a probability depending on the temperature
            const probability = Math.exp(-energyChange / temperature);
            if (Math.random() < probability) {
                spinsFC[randomSpinIndex] = -randomSpin;
                spin_sumFC=spin_sumFC-2*randomSpin;
            }
        }
    }
    // Redraw the simulation
    drawIsingModelFC();
}

function handleVisibilityChangeFC(entries) {
  const isVisible = entries[0].isIntersecting;
  isSimulationPausedFC = !isVisible;
}

// Create an intersection observer2D
const observerFC = new IntersectionObserver(handleVisibilityChangeFC, { threshold: 0 });

// Observe the canvas element
observerFC.observe(canvasFC);

// Call the update function periodically to animate the simulation
setInterval(updateSimulationFC, 100);
