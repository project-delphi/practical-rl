--[[
katex.lua: self-hosted KaTeX for HTML pages (PLAN.md §8).

With `html-math-method: katex`, pandoc links KaTeX from a CDN (`katex@latest`
on cdn.jsdelivr.net). Quarto's object form (`{method: katex, url: ...}`) takes a
URL that is pasted into every page as is, so a relative URL breaks for pages in
subdirectories and an absolute one breaks `quarto preview` (served at /).

Instead, on pages that contain math, this filter
  1. registers the vendored copy in filters/vendor/katex/ as an HTML dependency:
     Quarto copies it (and the fonts its CSS references) to site_libs/ and writes
     page-relative links, and
  2. sets the `math` template variable to a comment, which replaces pandoc's CDN
     <script>/<link> block (pandoc fills `math` only when the document leaves it
     unset). `$if(math)$` stays true, so Quarto's window.Quarto.typesetMath exists.
katex-render.js typesets every span.math, as pandoc's inline script would.

The version must match filters/vendor/katex/ (see fonts/LICENSES.md).
]]

local KATEX_VERSION = "0.18.9"

local function is_html_page()
  return quarto.doc.is_format("html") and not quarto.doc.is_format("revealjs")
end

local function katex_selected()
  local opts = PANDOC_WRITER_OPTIONS
  if opts == nil or opts.html_math_method == nil then
    return true -- older pandoc: trust _quarto.yml
  end
  local method = opts.html_math_method
  if type(method) == "table" then
    method = method.method
  end
  return method == "katex"
end

function Pandoc(doc)
  if not (is_html_page() and katex_selected()) then
    return nil
  end
  local found = false
  doc:walk({
    Math = function(_)
      found = true
    end,
  })
  if not found then
    return nil
  end
  quarto.doc.add_html_dependency({
    name = "katex",
    version = KATEX_VERSION,
    scripts = { "vendor/katex/katex.min.js", "katex-render.js" },
    stylesheets = { "vendor/katex/katex.min.css" },
  })
  doc.meta.math = pandoc.RawInline(
    "html",
    "<!-- KaTeX " .. KATEX_VERSION .. ", self-hosted by filters/katex.lua -->"
  )
  return doc
end
