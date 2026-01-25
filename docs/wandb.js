// wandb.js
// Lazy-load W&B iframe only when <details> is opened.

(function () {
  function initOne(details) {
    const src = details.getAttribute("data-wandb-src");
    if (!src) return;

    let container = details.querySelector(".wandb-container");
    if (!container) {
      container = document.createElement("div");
      container.className = "wandb-container";
      container.style.marginTop = "15px";
      details.appendChild(container);
    }

    function inject() {
      if (!details.open) return;
      if (container.dataset.loaded === "true") return;

      const iframe = document.createElement("iframe");
      iframe.src = src;
      iframe.style.border = "none";
      iframe.style.width = "100%";
      iframe.style.height = details.getAttribute("data-wandb-height") || "1024px";
      iframe.setAttribute("loading", "lazy");

      container.appendChild(iframe);
      container.dataset.loaded = "true";
    }

    details.addEventListener("toggle", inject);
    inject(); // in case it's already open
  }

  function initAll() {
    document.querySelectorAll("details[data-wandb-src]").forEach(initOne);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();
