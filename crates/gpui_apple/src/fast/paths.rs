//! Path batches rasterized in groups, with fewer render command encoders.
//!
//! The frame encoder gives every path batch two render command encoders of its
//! own: it ends the pass it is in, rasterizes the batch's paths into the
//! cleared intermediate texture in a new one, and begins a new pass on its
//! target to composite them. A frame with six path batches (six sparklines)
//! thus encodes thirteen render passes. Every encoder costs the CPU a render
//! pass descriptor, an encoder and its viewport, and on Apple GPUs, which
//! render in tiles, every restarted pass stores its whole target to memory and
//! loads it back.
//!
//! Here, as in `gpui_wgpu`'s `fast::frame`, consecutive path batches that don't
//! come near each other share one rasterization pass, and the first group's
//! pass runs before the frame's first pass begins, so a frame whose path
//! batches don't overlap is encoded in two passes. Each batch's pixels in the
//! intermediate texture are then the same as if it had been rasterized alone
//! into the cleared texture, so the image is the same.
//!
//! Paths are rasterized into the window-sized intermediate whatever target
//! they are composited onto, so a group may span the start or end of an
//! isolated group: only the composite out of the intermediate lands on the
//! group's target.

use std::ops::Range;

use anyhow::Result;
use gpui::{Bounds, DevicePixels, PrimitiveBatch, ScaledPixels, Scene, SceneStep, Size};

use crate::metal_renderer::{InstanceBindings, InstanceBufferWriter, MetalRenderer};

/// How far apart, in device pixels, path batches must be to share the
/// intermediate texture. Rasterization is clipped to each path's bounds, and
/// the intermediate texture is sampled with linear filtering at texel
/// centers, so a pixel is plenty; two leave room for rounding.
const PATH_BATCH_MARGIN: f32 = 2.;

/// Past this many path batches in a group, checking that a batch overlaps
/// none of them costs more than the render passes it would save.
const MAX_GROUP_PATH_BATCHES: usize = 64;

/// A non-empty `PrimitiveBatch::Paths`, in the order the frame encoder meets
/// them.
pub(crate) struct PathBatch {
    pub(crate) paths: Range<usize>,
    /// The bounds everything the batch draws stays within.
    bounds: Bounds<ScaledPixels>,
    /// For the first batch of a group, the batches of the whole group,
    /// rasterized together before the batch is composited.
    pub(crate) group: Option<Range<usize>>,
}

/// Lists every non-empty path batch of the scene, walking its steps as the
/// frame encoder does, and groups them.
pub(crate) fn plan(scene: &Scene) -> Vec<PathBatch> {
    let mut path_batches: Vec<PathBatch> = Vec::new();
    let mut group_start = 0;
    let runs = scene.steps().filter_map(|step| match step {
        SceneStep::Run(run) => Some(run),
        SceneStep::PushGroup(_) | SceneStep::PopGroup => None,
    });
    for run in runs {
        for batch in scene.run_batches(run) {
            let PrimitiveBatch::Paths(range) = batch else {
                continue;
            };
            let paths = &scene.paths[range.clone()];
            let Some(first_path) = paths.first() else {
                continue;
            };
            let bounds = paths
                .iter()
                .skip(1)
                .fold(first_path.clipped_bounds(), |union, path| {
                    union.union(&path.clipped_bounds())
                });

            let near = bounds.dilate(ScaledPixels(PATH_BATCH_MARGIN));
            let group = &path_batches[group_start..];
            if group.is_empty()
                || group.len() >= MAX_GROUP_PATH_BATCHES
                || group.iter().any(|batch| near.intersects(&batch.bounds))
            {
                close_group(&mut path_batches, group_start);
                group_start = path_batches.len();
            }
            path_batches.push(PathBatch {
                paths: range,
                bounds,
                group: None,
            });
        }
    }
    close_group(&mut path_batches, group_start);
    path_batches
}

/// Gives the first batch of the group starting at `start` the range of all of
/// it.
fn close_group(path_batches: &mut [PathBatch], start: usize) {
    let end = path_batches.len();
    if let Some(first) = path_batches.get_mut(start) {
        first.group = Some(start..end);
    }
}

/// Rasterizes every batch of a group into the cleared intermediate texture, in
/// one pass.
pub(crate) fn rasterize_group(
    renderer: &MetalRenderer,
    scene: &Scene,
    group: &[PathBatch],
    instance_bindings: &InstanceBindings,
    writer: &mut InstanceBufferWriter,
    viewport_size: Size<DevicePixels>,
    command_buffer: &metal::CommandBufferRef,
) -> Result<()> {
    let command_encoder = renderer.new_path_rasterization_encoder(command_buffer)?;
    let rasterized = group.iter().try_for_each(|batch| {
        let paths = &scene.paths[batch.paths.clone()];
        let clipped = paths[0].clip.is_clipped();
        renderer.rasterize_paths(
            paths,
            clipped,
            instance_bindings,
            writer,
            viewport_size,
            command_encoder,
        )
    });
    command_encoder.end_encoding();
    rasterized
}

#[cfg(test)]
mod tests {
    use super::plan;
    use gpui::{Bounds, ContentMask, Path, Quad, ScaledPixels, Scene, point, px, size};
    use std::ops::Range;

    fn everything() -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            size(ScaledPixels(200.), ScaledPixels(200.)),
        )
    }

    fn square(x: f32) -> Path<ScaledPixels> {
        let mut path = Path::new(point(px(x), px(0.)));
        path.line_to(point(px(x + 10.), px(0.)));
        path.line_to(point(px(x + 10.), px(10.)));
        path.line_to(point(px(x), px(10.)));
        let mut path = path.scale(1.);
        path.content_mask = ContentMask::new(everything());
        path
    }

    /// A scene of a path at each of `xs`, with a quad over the whole scene
    /// after each, so that every path is a batch of its own.
    fn scene_of(xs: &[f32]) -> Scene {
        let mut scene = Scene::default();
        for &x in xs {
            scene.insert_primitive(square(x));
            scene.insert_primitive(Quad {
                bounds: everything(),
                content_mask: ContentMask::new(everything()),
                ..Default::default()
            });
        }
        scene.finish();
        scene
    }

    fn groups(scene: &Scene) -> Vec<Range<usize>> {
        plan(scene)
            .into_iter()
            .filter_map(|batch| batch.group)
            .collect()
    }

    #[test]
    fn disjoint_path_batches_share_a_group() {
        let scene = scene_of(&[0., 50., 100.]);
        assert_eq!(plan(&scene).len(), 3);
        assert_eq!(groups(&scene), vec![0..3]);
    }

    #[test]
    fn a_path_batch_near_one_in_its_group_starts_a_new_group() {
        let scene = scene_of(&[0., 50., 11., 100.]);
        assert_eq!(groups(&scene), vec![0..2, 2..4]);
    }
}
