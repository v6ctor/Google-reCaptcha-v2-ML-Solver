const thresholdEl = document.getElementById("threshold");
const threshValEl = document.getElementById("threshVal");
const enabledEl   = document.getElementById("enabled");

chrome.storage.sync.get(["threshold", "enabled"], ({ threshold = 60, enabled = true }) => {
  thresholdEl.value = threshold;
  threshValEl.textContent = `${threshold}%`;
  enabledEl.checked = enabled;
});

thresholdEl.addEventListener("input", () => {
  const v = thresholdEl.value;
  threshValEl.textContent = `${v}%`;
  chrome.storage.sync.set({ threshold: parseInt(v) });
});

enabledEl.addEventListener("change", () => {
  chrome.storage.sync.set({ enabled: enabledEl.checked });
});
