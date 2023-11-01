function initSpeedSlider(prefix, max, initValue) {
  const speedSliderElem = document.getElementById(`${prefix}_speed-slider`);
  const speedValueElem = document.getElementById(`${prefix}_speed-value`);
  speedValueElem.style.marginRight = "10px";
  speedValueElem.textContent = initValue;

  speedSliderElem.max = max;
  speedSliderElem.value = initValue;
  speedSliderElem.step = 1;

  let speedValue = {
    value: parseInt(speedSliderElem.value)
  };

  speedSliderElem.addEventListener('input', function () {
    speedValue.value = parseInt(speedSliderElem.value);
    speedValueElem.textContent = speedValue.value;
  });

  return speedValue;
}
