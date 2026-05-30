const SERVER = "http://127.0.0.1:5000";

chrome.runtime.onInstalled.addListener(() => {
  console.log("reCAPTCHA Solver installed.");
});

// Relay solve requests from content scripts — service workers bypass mixed-content restrictions
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type !== "solve") return;

  fetch(`${SERVER}/solve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pageurl: msg.pageurl }),
  })
    .then(r => r.json())
    .then(data => sendResponse(data))
    .catch(e => sendResponse({ error: e.message }));

  return true; // keep channel open for async response
});
