function showContent(target) {
  showDivs(target);
  showSpoilers(target);
  gtag('event', 'click', {
    'event_category': 'article_version_switch',
    'event_label': target
  });
}

function showDivs(target) {
  // Get all the div elements
  const divs = document.querySelectorAll("div[target_audience]");

  // Loop through each div
  divs.forEach((div) => {
    if (div.getAttribute("target_audience").includes(target)) {
      // Display the selected div
      div.style.display = "block";
    } else {
      // Hide the other divs
      div.style.display = "none";
    }
  });
}


function showSpoilers(target){
    const details = document.querySelectorAll("details[target_audience]");

    details.forEach((det) =>{
        if (det.getAttribute("target_audience").includes(target)) {
            det.setAttribute("open","");
          } else {
            det.removeAttribute("open");
          }
    });
}