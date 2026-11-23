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
  chrome.
- **Smaller API additions**: animated PNGs in `img`, promised files on drop,
  redraw attribution and live handle counts, `draw_in_progress`, text that hugs
  its rows and reserves room for a trailing element, a content type on text
  input, no break after `/` when wrapping (a filter on upstream's UAX #14
  break opportunities), a transparent headless renderer
  (`current_headless_renderer_with_transparency`), and the wgpu font and
  backend options the Android and web builds use.
- **Animations in a view of their own**: `AnimationElement::in_own_view`
  (`fast::animation_view`) draws an animation of a `Clone` element in a
  view of its own, so its frames build only the animated element and the
  view around it is spliced rather than built again; and a splice may
  rebuild a nested view in a list's row, outside the layout tree of the view
  around it, when it asks for the layout it had.
- **Values inherited through the app**: `App::set_inherited` and
  `App::inherited` (`fast::inherited`) let an element hand a value to
  everything drawn inside it that reads only the app, such as which theme a
  part of the window is drawn in. A nested view built again on its own in a
  splice and a deferred draw take the values they were drawn with, as they take
  the text style, and a view drawn from last frame is reused only where it
  inherits what it did then.
- **Retained-renderer reactivity**: a retained view is drawn again when an
  entity it read is released, when the window's focus, size or appearance
  changes while it draws, when scroll geometry it read changes or a scroll
  handle is marked changed, and when a focus containment question it asked
  has a different answer in the frame drawn; an image cache is told which of
  its images a window still draws, and a dropped image's tiles live while
  something draws them. Upstream's cached-view fixes, which gpui-fast carries
  into `fast::retained` (reads in paint and in deferred draws, #65952;
  inherited opacity, #65951), cover none of these.

The changed upstream files are listed at the top of
`script/upstream-allowlist`, so `script/check-upstream` still holds gpui-fast's
own hooks to its rules.

## What the branch no longer carries

- **BGRA pixel-buffer surfaces**, dropped for gpui-fast's dynamic textures
  (`Window::create_dynamic_texture`, `update_dynamic_texture`,
  `paint_dynamic_texture`, `remove_dynamic_texture`). A surface is
  upstream's again: a biplanar full-range YCbCr frame. Dynamic textures copy a
  frame's bytes rather than sampling an IOSurface in place.

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
- Paths are upstream's tiled analytic-coverage paths (`PaintedPath`). A
  path's clip id rides in its `PathPaint`, and its brush in a per-path side
  buffer the Metal path tile shader reads (`PathBrushRecord`); a rounded
  content mask is evaluated per fragment, so such a path, quad or sprite never
  takes the shaders' solid fast path or the opaque depth pass.
- A layer's primitives reach the per-kind arrays only when the outermost layer
  closes; a group boundary inside a layer cuts the open layers there
  (`Scene::split_open_layers`), so a group's primitives stay one contiguous
  run.
- Native views are a ninth per-kind run (`Segment::native_views`).
- `SubpixelSprite` is laid out word for word as `MonochromeSprite`, whose
  `animation` slot it pads, because DirectX draws subpixel sprites with the
  monochrome sprite shaders.
- The Metal renderer is upstream's objc2-metal one, and the patches use its
  idioms: pipelines, textures and encoders are created fallibly, and a
  frame's CoreVideo textures live until its completion handler runs.
- A dynamic texture's content version rides in its sprite's
  `AtlasTile::padding`, which no shader reads and the scroll-layer tile hash
  covers, because `PolychromeSprite::pad` is the clip id here.
- A Metal frame that pushes no clip path and no group is drawn exactly as
  upstream draws it, depth-buffered opaque pass included. One that does is
  drawn by `vector/frame.rs`, pass by pass, back to front with no depth culling,
  because a depth buffer does not survive the frame being cut into group
  passes. Its clip masks (`vector/clips.rs`) and group targets and filters
  (`vector/groups.rs`) live beside the renderer rather than inside it, and every
  shader takes a `RenderTarget` (origin and size) where upstream takes the
  viewport size.

## Taking a new gpui-fast

Rebase `fifteen` onto gpui-fast's `main`, then run
`cargo test -p gpui --features test-support`, `script/check-upstream`, and on a
Mac `cargo test -p gpui_apple fast::layers` and `cargo test -p gpui_render_tests`.
