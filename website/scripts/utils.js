let monitorConfig;

function mean(ls) {
  return ls.reduce((accumulator, currentValue) => accumulator + currentValue, 0) / ls.length;
}

function getMonitorConfig() {
  return new Promise(resolve => {
    if (monitorConfig !== undefined) {
      resolve(monitorConfig);
    }
    let deltas = [];
    let lastTimestamp = undefined;

    function estimateMaxFps(timestamp) {
      if (lastTimestamp === undefined) {
        lastTimestamp = timestamp;
      } else {
        const actualDelta = timestamp - lastTimestamp;
        deltas.push(actualDelta)
        lastTimestamp = timestamp;
      }
      if (deltas.length < 30) {
        requestAnimationFrame(estimateMaxFps);
      } else {
        let meanDelta = mean(deltas);
        let deltasWithoutOutliers = deltas.filter(function (v) {
          return (v / meanDelta) < 1.5;
        });
        let meanDelta2 = mean(deltasWithoutOutliers);
        monitorConfig = {
          maxFps: Math.round(1000 / meanDelta2),
          deltaBetweenFrames: meanDelta2
        }
        resolve(monitorConfig);
      }
    }

    requestAnimationFrame(estimateMaxFps);
  })
}

function animateWrapper(config) {
  function animate(timestamp) {
    config.skipedFrames += 1;
    if (config.skipedFrames >= config.skipFrames()) {
      // console.log(`skiped: ${config.skipedFrames}`)
      config.update();
      config.skipedFrames = 0;
    }
    requestAnimationFrame(animate);
  }

  return animate;
}

function initFpsSlider(slider, value, datalist, update) {
  getMonitorConfig().then(monitorConfig => {
    let fps = monitorConfig.maxFps;
    let tickmarks = [0, 1];
    while (fps > 1) {
      tickmarks.push(fps);
      fps = Math.round(fps / 2)
    }
    tickmarks.forEach(function (fps) {
      let option = document.createElement('option');
      option.value = fps;
      datalist.appendChild(option);
    })
    slider.max = monitorConfig.maxFps;
    slider.value = monitorConfig.maxFps;

    slider.addEventListener('input', function () {
      let currentValue = parseInt(slider.value);
      if (!tickmarks.includes(currentValue)) {
        let closestValue = tickmarks.reduce(function (prev, curr) {
          return (Math.abs(curr - currentValue) < Math.abs(prev - currentValue) ? curr : prev);
        });
        slider.value = closestValue;
      }
    });

    requestAnimationFrame(animateWrapper({
      skipedFrames: 0,
      skipFrames: function () {
        const fps = parseInt(slider.value);
        value.textContent = fps;
        let skipFrames = Math.round((1000 / fps) / monitorConfig.deltaBetweenFrames)
        return skipFrames;
      },
      update: update,
    }));
  });
}
