/**
 * reCAPTCHA Solver — content script
 *
 * Reads the sitekey from the reCAPTCHA anchor iframe URL, POSTs the current
 * page URL to the local solver server, and injects the returned token.
 */

let solving = false;

function getSitekey() {
  for (const iframe of document.querySelectorAll('iframe[src*="recaptcha"]')) {
    const m = iframe.src.match(/[?&]k=([^&]+)/);
    if (m) return m[1];
  }
  return null;
}

function injectToken(token) {
  // Fill every g-recaptcha-response textarea (there may be more than one)
  for (const el of document.querySelectorAll(
    '#g-recaptcha-response, textarea[name="g-recaptcha-response"]'
  )) {
    el.value = token;
    el.innerHTML = token;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }

  // Fire the widget's data-callback if defined
  for (const widget of document.querySelectorAll("[data-callback]")) {
    const fn = widget.dataset.callback;
    if (typeof window[fn] === "function") {
      try { window[fn](token); } catch (_) {}
    }
  }

  // Also try the grecaptcha internal callback registry
  try {
    const clients = window.___grecaptcha_cfg?.clients;
    if (clients) {
      for (const client of Object.values(clients)) {
        const cb = client?.l?.callback ?? client?.callback;
        if (typeof cb === "function") cb(token);
      }
    }
  } catch (_) {}
}

async function trySolve() {
  if (solving) return;
  const sitekey = getSitekey();
  if (!sitekey) return;

  solving = true;
  console.log(`[reCAPTCHA Solver] Detected sitekey ${sitekey} — calling server...`);

  try {
    const result = await new Promise(resolve =>
      chrome.runtime.sendMessage({ type: "solve", pageurl: window.location.href }, resolve)
    );
    if (result?.token) {
      console.log("[reCAPTCHA Solver] Token received — injecting.");
      injectToken(result.token);
    } else {
      console.warn("[reCAPTCHA Solver] No token:", result?.error ?? "unknown error");
    }
  } catch (e) {
    console.error("[reCAPTCHA Solver] Message failed:", e.message);
  } finally {
    solving = false;
  }
}

// Watch for the reCAPTCHA anchor iframe appearing
const observer = new MutationObserver(() => {
  if (document.querySelector('iframe[src*="api2/anchor"]')) {
    observer.disconnect();
    setTimeout(trySolve, 1500);   // small delay so reCAPTCHA finishes initializing
  }
});
observer.observe(document.body, { childList: true, subtree: true });

// Also fire immediately if the iframe is already in the DOM
if (document.querySelector('iframe[src*="api2/anchor"]')) {
  setTimeout(trySolve, 1500);
}
