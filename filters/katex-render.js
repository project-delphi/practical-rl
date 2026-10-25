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

  // Short inline formulas stay on one line. KaTeX allows a line break after every
  // relation and binary operator, so "γ = 0.99" can end one line with "γ =" and start
  // the next with "0.99". A formula narrower than SHORT_PX, and than SHORT_SHARE of its
  // text column, gets the class prl-short, which turns those breaks off
  // (styles/prl.scss). Wider formulas keep them, so they still wrap on a phone. Table
  // cells are left alone, because a phone's columns can be narrower than a short formula.
  // A short formula's neighbors join it in a span.prl-glue that does not wrap: the text
  // right after it up to the next space (a period, a comma, "-greedy", "'s"), and up to
  // three characters right before it (an opening parenthesis), so neither is left alone
  // at the edge of a line. Widths are measured rather than guessed from the TeX, because
  // the same TeX renders at different sizes. A formula inside a closed fold has no width
  // yet; it is measured when the fold opens.
  var SHORT_PX = 150;
  var SHORT_SHARE = 0.65;
  var GLUE_AFTER = /^[^\s]{1,16}(?=\s|$)|^[.,;:!?)\]’”'"]+/;
  var GLUE_BEFORE = /(?:^|\s)([^\s]{1,3})$/;

  function textColumn(el) {
    var box = el.parentElement.closest("p, li, dd, dt, figcaption, blockquote, summary, div");
    if (!box) return Infinity;
    var cs = getComputedStyle(box);
    return box.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  }

  function glue(el) {
    if (el.parentNode.classList.contains("prl-glue")) return;
    var after = el.nextSibling;
    var before = el.previousSibling;
    var tail = after && after.nodeType === Node.TEXT_NODE ? GLUE_AFTER.exec(after.data) : null;
    var head = before && before.nodeType === Node.TEXT_NODE ? GLUE_BEFORE.exec(before.data) : null;
    if (!tail && !head) return;
    var span = document.createElement("span");
    span.className = "prl-glue";
    el.parentNode.insertBefore(span, el);
    if (head) {
      before.data = before.data.slice(0, before.data.length - head[1].length);
      span.appendChild(document.createTextNode(head[1]));
    }
    span.appendChild(el);
    if (tail) {
      after.data = after.data.slice(tail[0].length);
      span.appendChild(document.createTextNode(tail[0]));
    }
  }

  function keepShortTogether(root) {
    var spans = root.querySelectorAll("span.math.inline");
    var found = [];
    // Measure everything first, then change the page, so the browser lays it out once.
    for (var i = 0; i < spans.length; i++) {
      var el = spans[i];
      var html = el.querySelector(".katex-html");
      if (!html || el.closest("td, th")) continue;
      var w = 0;
      for (var c = 0; c < html.children.length; c++) {
        w += html.children[c].getBoundingClientRect().width;
      }
      if (w === 0) continue; // not displayed (a closed fold): measured when it opens
      found.push([el, w < SHORT_PX && w < SHORT_SHARE * textColumn(el)]);
    }
    for (var j = 0; j < found.length; j++) {
      found[j][0].classList.toggle("prl-short", found[j][1]);
      if (found[j][1]) glue(found[j][0]);
    }
  }

  function layout() {
    markOverflowing();
    keepShortTogether(document);
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
    layout();
    // Widths change once KaTeX's fonts arrive and whenever the window resizes.
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(layout);
    window.addEventListener("load", layout);
    var pending;
    window.addEventListener("resize", function () {
      clearTimeout(pending);
      pending = setTimeout(layout, 150);
    });
    // A fold that opens shows formulas that had no width while it was closed.
    document.addEventListener(
      "toggle",
      function (e) {
        if (e.target.open) keepShortTogether(e.target);
      },
      true
    );
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", typeset);
  } else {
    typeset();
  }
})();
