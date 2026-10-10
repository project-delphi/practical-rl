--[[
figures.lua: one figure, two files, the one that matches the theme (PLAN.md §8).

Write a figure once, with the light file:

  ![What to notice: ...](/images/01-cliff-gridworld.svg){#fig-cliff fig-alt="..."}

If images/01-cliff-gridworld.dark.svg exists next to it, the image (and its lightbox
link) is emitted twice, each in a span that Quarto shows only in one theme:

  <span class="light-content"><a class="lightbox" ...><img src=".../X.svg" alt="..."></a></span>
  <span class="dark-content"><a class="lightbox" ...><img src=".../X.dark.svg" alt="..."></a></span>

Quarto 1.10's own CSS hides .dark-content under body.quarto-light and .light-content
under body.quarto-dark (display: none, so the hidden copy is out of the accessibility
tree and the tab order); the page starts as quarto-light, so without JavaScript the
light file shows. Print always shows the light file (styles/prl.scss).

The filter runs at post-render (see _quarto.yml), after Quarto has numbered the figure,
written its caption and alt text and added the lightbox link, so the figure keeps one
number, one caption and one alt text: only the image inside it is doubled. The dark
link gets its own lightbox gallery, so the lightbox never steps from one copy to the
other. A local image without a twin gets the class `paper`, the hook for the light-card
fallback in dark mode (img.paper in styles/prl.scss); tests/test_site_figures.py
requires a twin for every images/*.svg unless it is listed there as theme-neutral.
HTML pages only; slides are untouched.
]]

local function is_html_page()
  return quarto.doc.is_format("html") and not quarto.doc.is_format("revealjs")
end

-- "images/x.svg" -> "images/x.dark.svg" (any extension; a query or fragment is kept).
-- nil for remote and data URLs and for files that are already the dark twin.
local function dark_src(src)
  if src:match("^%a[%w+.-]*:") or src:match("^//") then
    return nil
  end
  local path, rest = src:match("^([^?#]*)(.*)$")
  local base, ext = path:match("^(.+)%.(%w+)$")
  if base == nil or base:match("%.dark$") then
    return nil
  end
  return base .. ".dark." .. ext .. rest, base .. ".dark." .. ext
end

local function on_disk(path)
  local input_dir = pandoc.path.directory(quarto.doc.input_file)
  if path:sub(1, 1) == "/" then -- site-absolute: relative to the project
    local root = (quarto.project and quarto.project.directory) or input_dir
    return pandoc.path.join({ root, path:sub(2) })
  end
  return pandoc.path.join({ input_dir, path })
end

local function exists(path)
  local f = io.open(path, "r")
  if f == nil then
    return false
  end
  f:close()
  return true
end

-- the dark src if `img` has a twin on disk, else nil
local function twin(img)
  local src, file = dark_src(img.src)
  if src ~= nil and exists(on_disk(file)) then
    return src
  end
  return nil
end

local function themed(inline, class)
  return pandoc.Span({ inline }, pandoc.Attr("", { class }))
end

local function dark_image(img, src)
  local d = img:clone()
  d.src = src
  d.identifier = "" -- an id stays on the light copy only
  return d
end

local function pair_image(img)
  local src = twin(img)
  if src == nil then
    if dark_src(img.src) ~= nil and not img.classes:includes("paper") then
      img.classes:insert("paper") -- a local image with no dark twin
      return img
    end
    return nil
  end
  return pandoc.Inlines({
    themed(img, "light-content"),
    themed(dark_image(img, src), "dark-content"),
  })
end

-- Quarto's lightbox wraps a figure image in <a class="lightbox" href=same-file>.
local function pair_lightbox(link)
  if not link.classes:includes("lightbox") or #link.content ~= 1 then
    return nil
  end
  local img = link.content[1]
  if img.t ~= "Image" then
    return nil
  end
  local src = twin(img)
  if src == nil then
    return nil -- the Image function below still sees it
  end
  local dark = link:clone()
  dark.content = pandoc.Inlines({ dark_image(img, src) })
  if dark.target == img.src then
    dark.target = src
  end
  dark.identifier = ""
  if dark.attributes["gallery"] then
    dark.attributes["gallery"] = dark.attributes["gallery"] .. "-dark"
  end
  return pandoc.Inlines({ themed(link, "light-content"), themed(dark, "dark-content") })
end

return {
  {
    traverse = "topdown",
    Span = function(span)
      -- already paired (a second run, or written by hand)
      if span.classes:includes("light-content") or span.classes:includes("dark-content") then
        return nil, false
      end
    end,
    Link = function(link)
      if not is_html_page() then
        return nil
      end
      local out = pair_lightbox(link)
      if out ~= nil then
        return out, false
      end
    end,
    Image = function(img)
      if not is_html_page() then
        return nil
      end
      -- false: topdown traversal would otherwise walk into the returned spans and
      -- pair the same image again, forever
      return pair_image(img), false
    end,
  },
}
