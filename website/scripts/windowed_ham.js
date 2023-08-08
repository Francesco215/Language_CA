const numVisible = 9; // Number of visible squares
const data = Array.from({ length: numVisible + 2 }, (_, i) => i);

const square_side = 50;
const margin = square_side/4;
const area_side = square_side + 2 * margin;
const global_left_padding=10;

const window_width = 5;

const svg = d3.select("#graph")
    .attr("width", numVisible * area_side) // Adjust width based on the number of visible squares
    .attr("height", area_side);

const squares = svg.selectAll(".square")
    .data(data)
    .enter()
    .append("g")
    .attr("class", "square")
    .attr("transform", (d, i) => `translate(${i * area_side + global_left_padding}, 10) scale(1, 1)`)
    .on("mouseover", handleMouseOver)
    .on("mouseout", handleMouseOut);

squares.append("rect")
    .attr("width", square_side)
    .attr("height", square_side)
    .attr("rx", square_side/6) // Horizontal radius for rounded corners
    .attr("ry", square_side/6) // Vertical radius for rounded corners
    .style("fill", "#3498db") // Set the background color to blue
    .style("border-radius",5);

squares.append("foreignObject")
    .attr("x", square_side / 2 - 10)
    .attr("y", 7)
    .attr("width", 40)
    .attr("height", 40)
    .html(d => `<div class="katex">$x_{${d+1}}$</div>`);

function handleMouseOver(_, i) {
    svg.selectAll(".highlight").remove();

    svg.append("rect")
    .attr("class", "highlight")
    .attr("x", (i - (window_width- 1)/2) * area_side - margin / 2 + global_left_padding)
    .attr("y", 5)
    .attr("width", square_side * window_width + margin * (2 * window_width - 1))
    .attr("height", square_side + 10);
}

function handleMouseOut() {
    svg.selectAll(".highlight").remove();
}