--[[
scroll-regions.lua: tables never make the page scroll sideways (PLAN.md §8).

Every table that is not already inside an authored `::: {.table-scroll}` region is
wrapped in `<div class="table-wrap">`. Both kinds of wrapper scroll sideways
inside themselves when the table is wider than the column (styles/prl.scss).

A wrapper that actually overflows must be reachable and named for keyboard and
screen-reader users (WCAG 2.1.1, 4.1.2). Whether it overflows depends on the
screen, so styles/prl-head.html adds tabindex="0", role="region" and an
aria-label at run time, only while it overflows. The label comes from here:
`data-region-label` is the div's own aria-label if it has one, else the table's
caption, else the nearest heading above it plus " table".
HTML pages only.
]]

local function is_html_page()
  return quarto.doc.is_format("html") and not quarto.doc.is_format("revealjs")
end

local last_heading = nil

local function caption_text(el)
  local text = nil
  el:walk({
    Table = function(t)
      if text == nil and t.caption and t.caption.long and #t.caption.long > 0 then
        text = pandoc.utils.stringify(t.caption.long)
      end
    end,
  })
  if text ~= nil and text:match("%S") then
    return text
  end
  return nil
end

local function label_for(el)
  return caption_text(el) or (last_heading and (last_heading .. " table")) or "Table"
end

return {
  {
    traverse = "topdown",
    Header = function(h)
      last_heading = pandoc.utils.stringify(h.content)
    end,
    Div = function(div)
      if not is_html_page() or not div.classes:includes("table-scroll") then
        return nil
      end
      local attrs = div.attributes
      -- aria-label is not allowed on a plain div; it is applied with role="region".
      local label = attrs["aria-label"] or label_for(div)
      attrs["aria-label"] = nil
      attrs["data-region-label"] = attrs["data-region-label"] or label
      return div, false -- its tables are already in a region
    end,
    Table = function(tbl)
      if not is_html_page() then
        return nil
      end
      local wrap = pandoc.Div({ tbl }, pandoc.Attr("", { "table-wrap" }))
      wrap.attributes["data-region-label"] = label_for(wrap)
      return wrap, false
    end,
  },
}
