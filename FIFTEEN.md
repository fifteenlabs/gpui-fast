# Fifteen's branch of gpui-fast

The `fifteen` branch is gpui-fast with the patches the Fifteen app needs on
top. They came from `fifteenlabs/zed`'s `fifteen` branch, which carried them on
upstream GPUI before the app moved to gpui-fast.

## What the branch carries

- **A vector renderer**: clip paths (`Window::with_clip_path`), isolated
  groups with opacity, blend modes and blur, drop shadow and backdrop filters
  (`Window::with_isolated_group`), gradient and image brushes for paths
  (`paint_path_with_gradient`, `paint_path_with_image`), and per-corner radii on
  `ContentMask`. Metal draws all of it; wgpu and DirectX draw groups
  un-isolated and say so once.
- **Window behaviour**: a window zoom that scales every length, opting out of
  the window shadow, adopting programmatic resizes, popup windows without
  chrome, BGRA pixel-buffer surfaces.
- **Smaller API additions**: animated PNGs in `img`, promised files on drop,
  redraw attribution and live handle counts, `draw_in_progress`, text that hugs
  its rows and reserves room for a trailing element, a content type on text
  input, `/` as a word character when wrapping, and the wgpu font and backend
  options the Android and web builds use.

The changed upstream files are listed at the top of
`script/upstream-allowlist`, so `script/check-upstream` still holds gpui-fast's
own hooks to its rules.

## How the patches meet gpui-fast

- Rounded `overflow: hidden` keeps gpui-fast's analytic rounded clip, except in
  a scene that also pushes a clip path or an isolated group. There its rounded
  clips are pushed as clip paths instead
  (`fast::rounded_clip_scene::lower`), because a rounded clip group's child
  scene is drawn outside the frame's own passes, where clip masks and group
  targets live.
- A scroll layer whose content pushes a clip path or a group is drawn into the
  frame rather than into tiles, as one whose content has paths is.
- Moving content (scroll layers, list rows) moves clip paths, group bounds and
  brush transforms with it; tiles hash clip and group markers.
- Paths take the content mask snapped to device pixels, as quads do, so a scroll
  layer clips them to the snapped viewport too.
- Metal frames always take the frame encoder that handles clip masks and group
  targets, and that encoder takes gpui-fast's grouping of path batches into
  fewer render passes (`fast::paths`). A group may span an isolated group's
  start or end, since paths are rasterized into the window-sized intermediate
  whatever target they are composited onto.

## Taking a new gpui-fast

Merge gpui-fast's `main` into `fifteen`, then run
`cargo test -p gpui --features test-support`, `script/check-upstream`, and on a
Mac `cargo test -p gpui_apple fast::` and `cargo test -p gpui_render_tests`.
