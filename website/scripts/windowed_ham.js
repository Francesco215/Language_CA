const numVisible = 14; // Number of visible squares
const data = Array.from({ length: numVisible + 2 }, (_, i) => i);

const square_side = 50;
const margin = square_side/4;
const area_side = square_side + 2 * margin;
const globat_top_padding=20;
const global_left_padding=10;

const window_width = 5;

const svg = d3.select("#graph")
    .attr("width", numVisible * area_side) // Adjust width based on the number of visible squares
    .attr("height", area_side + globat_top_padding);

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
    .attr("y",globat_top_padding)
    .attr("rx", square_side/6) // Horizontal radius for rounded corners
    .attr("ry", square_side/6) // Vertical radius for rounded corners
    .style("fill", "#648FFF"); // Set the background color to blue

squares.append("foreignObject")
    .attr("x", d=> square_side / 2 - 10 -5*(d<numVisible/2-1))
    .attr("y", 7+globat_top_padding)
    .attr("width", 40)
    .attr("height", 40)
    .html(d => `<div class="katex">$\\textcolor{white}{s_{${d-numVisible/2+1}}}$</div>`);

function handleMouseOver(_, i) {
    svg.selectAll(".highlight-group").remove();

    const highlightGroup = svg.append("g")
        .attr("class", "highlight-group")
        .attr("transform", `translate(${(i - (window_width- 1)/2) * area_side - margin / 2 + global_left_padding}, 5)`)

    highlightGroup.append("rect")
        .attr("class", "highlight")
        .attr("y",globat_top_padding)
        .attr("width", square_side * window_width + margin * (2 * window_width - 1))
        .attr("height", square_side + 10);

    highlightGroup.append("foreignObject")
        .attr("class","highlight-text")
        .attr("y",-10)
        .attr("x",((window_width- 1)/2) * area_side - margin / 2 + global_left_padding + 13)
        .attr("width", 70)
        .attr("height", 30)
        .html(d => `<div class="katex">$\\textcolor{FE6100}{H_{${i-numVisible/2+1}}}$</div>`);

    renderMathInElement(highlightGroup.node(), {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false},
          {left: '\\(', right: '\\)', display: false},
          {left: '\\[', right: '\\]', display: true}
        ],
        throwOnError: false
      });
}

function handleMouseOut() {
    svg.selectAll(".highlight-group").remove();
}