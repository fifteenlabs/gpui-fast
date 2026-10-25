//! Images dropped from the sprite atlas while something still draws them.
//!
//! An image cache that evicts an image drops it with [`Window::drop_image`],
//! which gives its tiles back to the atlas. Upstream, a view that still shows
//! the image asks the cache for it again as it renders the next frame. A
//! retained view is not rendered: it is drawn again from last frame, and a
//! scroll layer rasterizes its tiles from the content it recorded, so their
//! sprites still point at the tiles given back, and the next image the atlas
//! allocates over one shows in its place.
//!
//! So a tile that anything drawn still points at stays in the atlas until
//! nothing does: the scene being painted, the one last handed to the renderer,
//! the content of every scroll layer the window keeps and the rows its list
//! layers hold.

use crate::{
    AtlasKey, AtlasTextureId, AtlasTile, PolychromeSprite, RenderImage, RenderImageParams, Scene,
    Window, scene::PaintOperation, scene::Primitive,
};
use collections::FxHashSet;

/// The image tiles given back while something still drew them, to give back
/// once nothing does.
#[derive(Default)]
pub(crate) struct DeferredImageDrops {
    keys: Vec<AtlasKey>,
}

/// Drops every frame of `image` from the window's sprite atlas, keeping the
/// tiles something still draws until nothing does.
pub(crate) fn drop_image(window: &mut Window, image: &RenderImage) {
    let tiles: Vec<(AtlasKey, AtlasTile)> = (0..image.frame_count())
        .filter_map(|frame_index| {
            let key = AtlasKey::from(RenderImageParams {
                image_id: image.id,
                frame_index,
            });
            let tile = tile_of(window, &key)?;
            Some((key, tile))
        })
        .collect();
    if tiles.is_empty() {
        return;
    }
    let drawn = drawn_tiles(window);
    for (key, tile) in tiles {
        if drawn.contains(&id_of(&tile)) {
            window.deferred_image_drops.keys.push(key);
        } else {
            window.sprite_atlas.remove(&key);
        }
    }
}

/// Whether anything the window draws, or may draw again without painting it,
/// shows a frame of `image`.
pub(crate) fn image_drawn(window: &Window, image: &RenderImage) -> bool {
    let tiles: Vec<AtlasTile> = (0..image.frame_count())
        .filter_map(|frame_index| {
            tile_of(
                window,
                &AtlasKey::from(RenderImageParams {
                    image_id: image.id,
                    frame_index,
                }),
            )
        })
        .collect();
    if tiles.is_empty() {
        return false;
    }
    let drawn = drawn_tiles(window);
    tiles.iter().any(|tile| drawn.contains(&id_of(tile)))
}

/// Gives back the tiles kept for dropped images that nothing draws any more,
/// as a frame finishes.
pub(crate) fn finish_frame(window: &mut Window) {
    if window.deferred_image_drops.keys.is_empty() {
        return;
    }
    let drawn = drawn_tiles(window);
    let keys = std::mem::take(&mut window.deferred_image_drops.keys);
    for key in keys {
        match tile_of(window, &key) {
            Some(tile) if drawn.contains(&id_of(&tile)) => {
                window.deferred_image_drops.keys.push(key);
            }
            Some(_) => window.sprite_atlas.remove(&key),
            None => {}
        }
    }
}

/// The tile the atlas holds for `key`, without adding one.
fn tile_of(window: &Window, key: &AtlasKey) -> Option<AtlasTile> {
    window
        .sprite_atlas
        .get_or_insert_with(key.clone(), &mut || Ok(None))
        .ok()
        .flatten()
}

fn id_of(tile: &AtlasTile) -> (AtlasTextureId, u32) {
    (tile.texture_id, tile.tile_id.0)
}

/// Every sprite tile something the window keeps may draw.
fn drawn_tiles(window: &Window) -> FxHashSet<(AtlasTextureId, u32)> {
    let mut drawn = FxHashSet::default();
    add_scene(&window.next_frame.scene, &mut drawn);
    add_scene(&window.rendered_frame.scene, &mut drawn);
    crate::fast::layers::for_each_kept_scene(window, &mut |scene| add_scene(scene, &mut drawn));
    crate::fast::layers::for_each_kept_primitive(window, &mut |primitive| {
        if let Primitive::PolychromeSprite(sprite) = primitive {
            drawn.insert(id_of(&sprite.tile));
        }
    });
    drawn
}

fn add_scene(scene: &Scene, drawn: &mut FxHashSet<(AtlasTextureId, u32)>) {
    let mut add_sprite = |sprite: &PolychromeSprite| {
        drawn.insert(id_of(&sprite.tile));
    };
    scene.polychrome_sprites.iter().for_each(&mut add_sprite);
    for operation in &scene.paint_operations {
        if let PaintOperation::Primitive(Primitive::PolychromeSprite(sprite)) = operation {
            add_sprite(sprite);
        }
    }
    for frame in &scene.layers.frames {
        for nested in frame.content.part_scenes() {
            add_scene(nested, drawn);
        }
    }
}
