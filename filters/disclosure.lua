--[[
disclosure.lua: collapsed callouts become native <details>/<summary> (PLAN.md §8).

  ::: {.callout-tip collapse="true" title="Answer to 3.2"}
  ...
  :::

renders as

  <details class="disclosure disclosure-tip">
    <summary>Answer to 3.2</summary>
    <div class="disclosure-body">...</div>
  </details>

A native <details> opens and closes with Enter and Space, announces its state,
needs no JavaScript, and prints open (styles/prl.scss, styles/prl-head.html).
`collapse="false"` gives an open <details>. A callout without a title gets its
type as the summary ("Tip", "Note", ...), or "Details". The callout's id and extra
classes are kept. Callouts that are cross-reference targets (#nte-, #tip-, ...)
are left to Quarto, so their numbering is unchanged. Other formats are untouched.
]]

local DEFAULT_TITLE = {
  note = "Note",
  tip = "Tip",
  warning = "Warning",
  important = "Important",
  caution = "Caution",
}

local CROSSREF_PREFIX = { nte = true, tip = true, wrn = true, imp = true, cau = true }

local function is_html_page()
  return quarto.doc.is_format("html") and not quarto.doc.is_format("revealjs")
end

local function escape_attr(s)
  return (s:gsub("&", "&amp;"):gsub('"', "&quot;"):gsub("<", "&lt;"):gsub(">", "&gt;"))
end

local function is_empty(title)
  return title == nil or pandoc.utils.stringify(title):match("^%s*$") ~= nil
end

-- Quarto hands the title over as a string, Inlines, an Inline, or Blocks
-- (a title from a heading inside the callout), depending on how it was written.
local function as_inlines(title)
  local kind = pandoc.utils.type(title)
  if kind == "string" then
    return pandoc.Inlines({ pandoc.Str(title) })
  elseif kind == "Inlines" then
    return title
  elseif kind == "Inline" then
    return pandoc.Inlines({ title })
  elseif kind == "Blocks" then
    return pandoc.utils.blocks_to_inlines(title)
  elseif kind == "Block" then
    return pandoc.utils.blocks_to_inlines({ title })
  end
  return pandoc.Inlines({ pandoc.Str(pandoc.utils.stringify(title)) })
end

function Callout(el)
  if not is_html_page() then
    return nil
  end
  local collapse = el.collapse
  if collapse == nil then
    return nil
  end
  collapse = tostring(collapse)
  if collapse ~= "true" and collapse ~= "false" then
    return nil
  end

  local id = (el.attr and el.attr.identifier) or ""
  local prefix = id:match("^(%a+)%-")
  if prefix and CROSSREF_PREFIX[prefix] then
    return nil
  end

  local kind = el.type or "note"
  local title = el.title
  if is_empty(title) then
    title = pandoc.Inlines({ pandoc.Str(DEFAULT_TITLE[kind] or "Details") })
  else
    title = as_inlines(title)
  end

  local classes = { "disclosure", "disclosure-" .. kind }
  if el.attr then
    for _, c in ipairs(el.attr.classes) do
      table.insert(classes, c)
    end
  end
  local open_tag = '<details class="' .. escape_attr(table.concat(classes, " ")) .. '"'
  if id ~= "" then
    open_tag = open_tag .. ' id="' .. escape_attr(id) .. '"'
  end
  if collapse == "false" then
    open_tag = open_tag .. " open"
  end
  open_tag = open_tag .. ">"

  local summary = pandoc.Inlines({ pandoc.RawInline("html", "<summary>") })
  summary:extend(title)
  summary:insert(pandoc.RawInline("html", "</summary>"))

  return pandoc.Div({
    pandoc.RawBlock("html", open_tag),
    pandoc.Plain(summary),
    pandoc.Div(el.content, pandoc.Attr("", { "disclosure-body" })),
    pandoc.RawBlock("html", "</details>"),
  }, pandoc.Attr("", { "disclosure-wrap" }))
end
