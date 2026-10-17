// Typeset pandoc's math spans with the self-hosted KaTeX (filters/katex.lua).
// Pandoc writes each formula as raw TeX in <span class="math inline|display">.
(function () {
  "use strict";

  // Display equations wider than the column scroll inside themselves
  // (styles/prl.scss). Only those become named, keyboard-focusable regions
  // (PLAN.md §8), and they stop being regions when they fit again.
  // A region's name is "Equation (3)" from its number, or "Unnumbered equation 2" for
  // the second display without a number. Every unnumbered display on the page is
  // counted, wide or not, so the name does not change with the window width, and no
  // two regions share a name.
  // KaTeX pins an equation number (a tag) to the right edge of the display; when the
  // formula is too wide that number would sit on top of it. A wide display gets the
  // class prl-wide, which puts the number after the formula (styles/prl.scss).
  function neededWidth(d) {
    var html = d.querySelector(".katex-html");
    if (!html) return d.scrollWidth;
    var total = 0;
    for (var c = 0; c < html.children.length; c++) {
      total += html.children[c].getBoundingClientRect().width;
    }
    return total + (html.querySelector(":scope > .katex-tag, :scope > .tag") ? 24 : 0);
  }

  function markOverflowing() {
    var displays = document.querySelectorAll(".katex-display");
    var unnumbered = 0;
    for (var i = 0; i < displays.length; i++) {
      var d = displays[i];
      var tag = d.querySelector(".katex-tag, .tag"); // KaTeX 0.18 renamed .tag
      if (!tag) unnumbered += 1;
      var wide = neededWidth(d) > d.clientWidth + 1 || d.scrollWidth > d.clientWidth + 1;
      d.classList.toggle("prl-wide", wide);
      if (wide && d.scrollWidth > d.clientWidth + 1) {
        d.setAttribute("tabindex", "0");
        d.setAttribute("role", "region");
        d.setAttribute(
          "aria-label",
          tag ? "Equation " + tag.textContent.trim() : "Unnumbered equation " + unnumbered
        );
      } else if (d.getAttribute("role") === "region") {
        d.removeAttribute("tabindex");
        d.removeAttribute("role");
        d.removeAttribute("aria-label");
      }
    }
  }

  function typeset() {
    if (!window.katex) {
      console.error("KaTeX did not load (filters/katex.lua).");
      return;
    }
    var spans = document.querySelectorAll("span.math");
    for (var i = 0; i < spans.length; i++) {
      var el = spans[i];
      var tex = el.firstChild;
      if (!tex || tex.nodeType !== Node.TEXT_NODE) continue; // already typeset
      window.katex.render(tex.data, el, {
        displayMode: el.classList.contains("display"),
        throwOnError: false,
      });
    }
    markOverflowing();
    // Widths change once KaTeX's fonts arrive and whenever the window resizes.
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(markOverflowing);
    window.addEventListener("load", markOverflowing);
    var pending;
    window.addEventListener("resize", function () {
      clearTimeout(pending);
      pending = setTimeout(markOverflowing, 150);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", typeset);
  } else {
    typeset();
  }
})();
