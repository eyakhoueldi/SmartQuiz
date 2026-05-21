(function () {
  "use strict";

  const doc = window.parent.document;
  const win = window.parent;
  const BTN_ID = "sq-sidebar-btn";

  const ICON_OPEN = `<svg width="18" height="14" viewBox="0 0 18 14" fill="none">
    <rect width="18" height="2" rx="1" fill="white"/>
    <rect y="6" width="12" height="2" rx="1" fill="white"/>
    <rect y="12" width="18" height="2" rx="1" fill="white"/>
  </svg>`;

  const ICON_CLOSE = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none">
    <rect x="0" y="6" width="14" height="2" rx="1" fill="white" transform="rotate(45 0 6)"/>
    <rect x="0" y="6" width="14" height="2" rx="1" fill="white" transform="rotate(-45 0 8)"/>
  </svg>`;

  /* ── Is the sidebar currently open/visible? ── */
  function sidebarVisible() {
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    if (!sb) return false;
    const cs = win.getComputedStyle(sb);
    if (cs.display === "none" || cs.visibility === "hidden") return false;
    const r = sb.getBoundingClientRect();
    // Open = has real width and is on screen
    return r.width > 80 && r.left > -10;
  }

  /* ── Find and click Streamlit's own sidebar toggle ── */
  function clickStreamlitToggle() {
    // When sidebar is OPEN, Streamlit renders a close button INSIDE the sidebar
    // When sidebar is CLOSED, it renders an open button in the collapsed control
    
    // 1. Look for the close button inside the sidebar (sidebar is open)
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    if (sb) {
      const closeBtn = sb.querySelector('button[aria-label*="close" i], button[aria-label*="collapse" i], button[aria-label*="hide" i]');
      if (closeBtn) { closeBtn.click(); return true; }

      // Streamlit sometimes puts the toggle as the first button in the sidebar header
      const allSbBtns = sb.querySelectorAll("button");
      if (allSbBtns.length > 0) {
        // The collapse button is usually the last button in the sidebar top area
        for (const b of allSbBtns) {
          const lbl = (b.getAttribute("aria-label") || "").toLowerCase();
          if (lbl.includes("sidebar") || lbl.includes("collapse") || lbl.includes("close") || lbl.includes("hide")) {
            b.click(); return true;
          }
        }
      }
    }

    // 2. Collapsed control (sidebar is closed)
    const ctrl = doc.querySelector('[data-testid="stSidebarCollapsedControl"]');
    if (ctrl) {
      const btn = ctrl.querySelector("button") || ctrl;
      btn.click(); return true;
    }

    // 3. Scan all page buttons for sidebar-related aria-labels
    const allBtns = Array.from(doc.querySelectorAll("button"));
    for (const btn of allBtns) {
      const lbl = (btn.getAttribute("aria-label") || "").toLowerCase();
      if (lbl.includes("sidebar") || lbl.includes("navigation")) {
        btn.click(); return true;
      }
    }

    return false;
  }

  /* ── Force-close by manipulating the sidebar DOM directly ── */
  function forceClose() {
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    if (!sb) return;
    sb.style.setProperty("transform", "translateX(-110%)", "important");
    sb.style.setProperty("visibility", "hidden", "important");
    setTimeout(() => {
      sb.style.removeProperty("transform");
      sb.style.removeProperty("visibility");
    }, 5000); // auto-restore after 5s in case Streamlit re-renders
  }

  /* ── Force-open ── */
  function forceOpen() {
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    if (!sb) return;
    sb.style.setProperty("transform", "none", "important");
    sb.style.setProperty("display", "flex", "important");
    sb.style.setProperty("visibility", "visible", "important");
    sb.style.setProperty("width", "21rem", "important");
    sb.style.setProperty("min-width", "21rem", "important");
  }

  /* ── Toggle ── */
  function toggle() {
    const isOpen = sidebarVisible();
    const clicked = clickStreamlitToggle();
    if (!clicked) {
      // Fallback to direct DOM manipulation
      isOpen ? forceClose() : forceOpen();
    }
  }

  /* ── Create the persistent toggle button ── */
  function createBtn() {
    if (doc.getElementById(BTN_ID)) return;

    const btn = doc.createElement("button");
    btn.id = BTN_ID;
    btn.innerHTML = ICON_OPEN;

    Object.assign(btn.style, {
      position:        "fixed",
      top:             "12px",
      left:            "12px",
      zIndex:          "2147483647",
      width:           "42px",
      height:          "42px",
      borderRadius:    "10px",
      background:      "linear-gradient(135deg,#7b6cff,#b96fff)",
      border:          "none",
      display:         "flex",
      alignItems:      "center",
      justifyContent:  "center",
      cursor:          "pointer",
      boxShadow:       "0 4px 18px rgba(123,108,255,.48)",
      transition:      "transform .18s, box-shadow .18s, opacity .18s",
      opacity:         "1",
    });

    btn.addEventListener("mouseenter", () => {
      btn.style.transform = "scale(1.08)";
      btn.style.boxShadow = "0 6px 26px rgba(123,108,255,.7)";
    });
    btn.addEventListener("mouseleave", () => {
      btn.style.transform = "scale(1)";
      btn.style.boxShadow = "0 4px 18px rgba(123,108,255,.48)";
    });
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      toggle();
    });

    doc.body.appendChild(btn);
  }

  /* ── Tick: update icon + position based on sidebar state ── */
  let lastState = null;
  function tick() {
    let btn = doc.getElementById(BTN_ID);
    if (!btn) { createBtn(); return; }

    const open = sidebarVisible();

    // Update icon only when state changes (avoids flicker)
    if (open !== lastState) {
      btn.innerHTML = open ? ICON_CLOSE : ICON_OPEN;
      btn.title = open ? "Close sidebar" : "Open sidebar";

      // Shift button right when sidebar is open so it doesn't overlap the sidebar edge
      btn.style.left = open ? "calc(21rem - 54px)" : "12px";

      lastState = open;
    }
  }

  setTimeout(createBtn, 500);
  setInterval(tick, 300);

})();