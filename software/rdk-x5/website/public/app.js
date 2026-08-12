(() => {
  "use strict";

  const modes = ["vivid", "defense", "minimal"];
  const labels = { vivid: "完整", defense: "答辩", minimal: "极简" };
  const html = document.documentElement;
  const modeToggle = document.getElementById("modeToggle");
  const modeLabel = document.getElementById("modeLabel");
  const menuToggle = document.getElementById("menuToggle");
  const navDeck = document.getElementById("navDeck");

  function applyMode(mode) {
    const safe = modes.includes(mode) ? mode : "vivid";
    html.dataset.visualMode = safe;
    localStorage.setItem("rootscope-visual-mode", safe);
    modeLabel.textContent = labels[safe];
    modeToggle.setAttribute("aria-label", `当前为${labels[safe]}模式，点击切换`);
  }

  applyMode(html.dataset.visualMode);
  modeToggle.addEventListener("click", () => {
    const next = modes[(modes.indexOf(html.dataset.visualMode) + 1) % modes.length];
    applyMode(next);
  });

  menuToggle.addEventListener("click", () => {
    const hidden = navDeck.classList.toggle("mobile-hidden");
    menuToggle.setAttribute("aria-expanded", String(!hidden));
  });

  navDeck.addEventListener("click", (event) => {
    if (event.target.closest("a") && matchMedia("(max-width: 760px)").matches) {
      navDeck.classList.add("mobile-hidden");
      menuToggle.setAttribute("aria-expanded", "false");
    }
  });

  const sectionIds = [...document.querySelectorAll("main section[id]")].map((section) => section.id);
  const navLinks = [...document.querySelectorAll(".nav-deck a")];
  const observer = new IntersectionObserver((entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    navLinks.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
    });
  }, { rootMargin: "-30% 0px -60% 0px", threshold: [0, .2, .5] });
  sectionIds.forEach((id) => observer.observe(document.getElementById(id)));

  if ("serviceWorker" in navigator) {
    addEventListener("load", async () => {
      try {
        const registrations = await navigator.serviceWorker.getRegistrations();
        for (const registration of registrations) {
          const url = registration.active?.scriptURL || "";
          if (!url.endsWith("/sw.js")) await registration.unregister();
        }
        if ("caches" in window) {
          const keys = await caches.keys();
          await Promise.all(keys.filter((key) => key.startsWith("cmdcenter-shell-")).map((key) => caches.delete(key)));
        }
        await navigator.serviceWorker.register("/sw.js", { scope: "/" });
      } catch (_) {
        // The public page remains fully functional without offline caching.
      }
    });
  }
})();
