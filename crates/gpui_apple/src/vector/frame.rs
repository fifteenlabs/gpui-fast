//! Metal frame encoding for scenes that clip to a path or isolate a group.
//!
//! Such a scene is drawn as its steps rather than as one run of batches: a
//! group boundary ends one render pass and begins another, on a target of the
//! group's own, and a pass that resumes drawing the target underneath loads
//! what is already there. A depth buffer does not survive being cut into
//! passes, so these frames draw back to front with nothing culled, the way
//! upstream drew every frame before it had a depth buffer.
use super::{
    CFRetained, CVMetalTexture, ClipInputIndex, DevicePixels, GlobalInputIndex, InstanceBindings,
    InstanceBufferWriter, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLLoadAction, MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLStoreAction, MTLTexture,
    MTLViewport, MetalRenderer, PrimitiveBatch, ProtocolObject, Result, Retained, Scene, Size,
    bind_render_target, mem, ptr,
};
use super::vector_groups::OpenGroup;
use anyhow::Context as _;
use gpui::{ClipPlan, Point, SceneStep, point};
use objc2::Message as _;

/// Where the attachment a render pass is drawing into sits in the window.
///
/// Every primitive reaches the GPU in scene coordinates - device pixels from
/// the window's top-left - and every fragment stage reasons in them, because
/// that is the space a content mask, a clip mask, a gradient and a signed
/// distance field are all expressed in. A group is drawn into a target of its
/// own that covers only part of the window, so the vertex stage subtracts this
/// origin to place a primitive, and the fragment stage adds it back to
/// `[[position]]`, which is the framebuffer's space and not the scene's. A
/// fragment reads its scene position off the pixel rather than as a varying
/// so that a pixel's value is exact, and drawing into a layer's tile gives the
/// same pixels as drawing into the window.
///
/// `size` is the viewport, not the texture: a group target is pooled and may be
/// larger than the group in it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct RenderTarget {
    /// The size of the region being drawn into, in device pixels.
    pub size: Size<DevicePixels>,
    /// Where that region's top-left corner is in the window, in device pixels.
    pub origin: Point<DevicePixels>,
}

impl RenderTarget {
    /// The whole of an attachment that sits at the window's origin, which is
    /// what every pass outside a group draws into.
    pub(crate) fn whole(size: Size<DevicePixels>) -> Self {
        RenderTarget {
            size,
            origin: point(DevicePixels(0), DevicePixels(0)),
        }
    }
}

/// The attachment a render pass is drawing into, and where it sits in the
/// window.
#[derive(Clone)]
pub(super) struct ActiveTarget {
    pub(super) texture: Retained<ProtocolObject<dyn MTLTexture>>,
    pub(super) render_target: RenderTarget,
    /// Whether this is the frame's own target rather than a group's.
    ///
    /// A backdrop filter has to copy out of whatever it is composited over, and
    /// the frame's target is the one texture that may refuse: a window's
    /// drawable is framebuffer-only until something asks for otherwise. A
    /// group's target is a texture this file made and can always be read.
    pub(super) is_window: bool,
}

const TRANSPARENT: MTLClearColor = MTLClearColor {
    red: 0.,
    green: 0.,
    blue: 0.,
    alpha: 0.,
};

impl MetalRenderer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn vector_frame(
        &mut self,
        scene: &Scene,
        clip_plan: &ClipPlan,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        texture: &ProtocolObject<dyn MTLTexture>,
        viewport_size: Size<DevicePixels>,
        surface_textures: &mut Vec<CFRetained<CVMetalTexture>>,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue
            .commandBuffer()
            .context("creating command buffer")?;
        crate::fast::dynamic_textures::encode_uploads(&command_buffer);
        let alpha = if self.opaque { 1. } else { 0. };

        // Every clip mask this frame samples is rasterized before the first
        // primitive is drawn, so a batch never has to interrupt a pass to go
        // and build one.
        self.render_clip_masks(clip_plan, writer, &command_buffer)?;
        self.group_targets.begin_frame();
        self.filter_scratch.begin_frame();
        self.fast_layers.note_composited(scene);

        // Where each path's tiles start among the frame's tiles, so a batch of
        // paths in any run finds its own.
        let mut path_tiles = Vec::with_capacity(scene.paths.len() + 1);
        path_tiles.push(0);
        for path in &scene.paths {
            path_tiles.push(path_tiles.last().copied().unwrap_or(0) + path.tile_count());
        }

        let mut target = ActiveTarget {
            texture: texture.retain(),
            render_target: RenderTarget::whole(viewport_size),
            is_window: true,
        };
        let mut command_encoder = self.begin_vector_pass(
            &command_buffer,
            &target,
            instance_bindings,
            Some(MTLClearColor {
                red: 0.,
                green: 0.,
                blue: 0.,
                alpha,
            }),
        )?;
        // `None` for a group that had to be painted without a target of its
        // own, which still has to be popped in step with the scene.
        let mut open_groups: Vec<Option<OpenGroup>> = Vec::new();

        for step in scene.steps() {
            let run = match step {
                SceneStep::PushGroup(spec) => {
                    let opened =
                        match self.open_group(spec, &target, viewport_size, open_groups.len()) {
                            Ok(opened) => opened,
                            Err(error) => {
                                command_encoder.endEncoding();
                                return Err(error);
                            }
                        };
                    match opened {
                        Some(group) => {
                            command_encoder.endEncoding();
                            target = group.target.clone();
                            command_encoder = self.begin_vector_pass(
                                &command_buffer,
                                &target,
                                instance_bindings,
                                Some(TRANSPARENT),
                            )?;
                            open_groups.push(Some(group));
                        }
                        None => open_groups.push(None),
                    }
                    continue;
                }
                SceneStep::PopGroup => {
                    let Some(group) = open_groups.pop() else {
                        command_encoder.endEncoding();
                        anyhow::bail!("a scene popped a group it never pushed");
                    };
                    let Some(group) = group else {
                        continue;
                    };
                    command_encoder.endEncoding();
                    // The filter passes and the backdrop copy run between the
                    // group's own pass and the composite, on encoders of their
                    // own: each writes a texture the next one reads, which one
                    // pass cannot do.
                    let prepared = self.prepare_group(&group, &command_buffer, viewport_size)?;
                    target = group.parent.clone();
                    command_encoder =
                        self.begin_vector_pass(&command_buffer, &target, instance_bindings, None)?;
                    let composited = self.composite_group(&group, &prepared, &command_encoder);
                    self.release_prepared(prepared);
                    if let Err(error) = composited {
                        command_encoder.endEncoding();
                        return Err(error);
                    }
                    continue;
                }
                SceneStep::Run(run) => run,
            };

            for batch in scene.run_batches(run) {
                let drawn = self.draw_vector_batch(
                    scene,
                    batch,
                    &path_tiles,
                    instance_bindings,
                    &command_encoder,
                    surface_textures,
                );
                if let Err(error) = drawn {
                    command_encoder.endEncoding();
                    return Err(error);
                }
            }
        }

        command_encoder.endEncoding();
        self.group_targets.end_frame();
        self.filter_scratch.end_frame();

        Ok(command_buffer)
    }

    fn draw_vector_batch(
        &mut self,
        scene: &Scene,
        batch: PrimitiveBatch,
        path_tiles: &[usize],
        instance_bindings: &InstanceBindings,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        surface_textures: &mut Vec<CFRetained<CVMetalTexture>>,
    ) -> Result<()> {
        match batch {
            PrimitiveBatch::Shadows(range) => {
                let clipped = scene.shadows[range.start].clip.is_clipped();
                self.draw_shadows(range, clipped, instance_bindings, command_encoder)
            }
            PrimitiveBatch::Quads(range) => {
                let clipped = scene.quads[range.start].clip.is_clipped();
                self.draw_quads(range, clipped, instance_bindings, command_encoder)
            }
            PrimitiveBatch::Paths(range) => {
                let clipped = scene.paths[range.start].clip().is_clipped();
                self.draw_path_tiles(
                    path_tiles[range.start]..path_tiles[range.end],
                    clipped,
                    instance_bindings,
                    command_encoder,
                )
            }
            PrimitiveBatch::Underlines(range) => {
                let clipped = scene.underlines[range.start].clip.is_clipped();
                self.draw_underlines(range, clipped, instance_bindings, command_encoder)
            }
            PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                let clipped = scene.monochrome_sprites[range.start].clip.is_clipped();
                self.draw_monochrome_sprites(
                    texture_id,
                    range,
                    clipped,
                    instance_bindings,
                    command_encoder,
                )
            }
            PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                let clipped = scene.polychrome_sprites[range.start].clip.is_clipped();
                self.draw_polychrome_sprites(
                    texture_id,
                    range,
                    clipped,
                    instance_bindings,
                    command_encoder,
                )
            }
            PrimitiveBatch::Surfaces(range) => {
                return self.draw_surfaces(
                    &scene.surfaces[range.clone()],
                    range.start,
                    instance_bindings,
                    command_encoder,
                    surface_textures,
                );
            }
            PrimitiveBatch::NativeViews(range) => {
                self.draw_native_views(range, instance_bindings, command_encoder)
            }
            PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
        }
        Ok(())
    }

    /// Begins a pass drawing into `target`: cleared to `clear`, or carrying on
    /// from what is already on it when `clear` is `None`.
    pub(super) fn begin_vector_pass(
        &mut self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        target: &ActiveTarget,
        instance_bindings: &InstanceBindings,
        clear: Option<MTLClearColor>,
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>> {
        let depth_texture = self.depth_texture_matching(&target.texture)?;
        let render_pass_descriptor = MTLRenderPassDescriptor::new();
        // SAFETY: Index zero is within the render pass's color attachment array.
        let color_attachment = unsafe {
            render_pass_descriptor
                .colorAttachments()
                .objectAtIndexedSubscript(0)
        };
        color_attachment.setTexture(Some(&target.texture));
        color_attachment.setStoreAction(MTLStoreAction::Store);
        match clear {
            Some(clear_color) => {
                color_attachment.setLoadAction(MTLLoadAction::Clear);
                color_attachment.setClearColor(clear_color);
            }
            None => color_attachment.setLoadAction(MTLLoadAction::Load),
        }
        // Every pipeline is built for a depth attachment, so every pass has
        // one, but nothing in these frames is ordered by it.
        let depth_attachment = render_pass_descriptor.depthAttachment();
        depth_attachment.setTexture(Some(&depth_texture));
        depth_attachment.setLoadAction(MTLLoadAction::Clear);
        depth_attachment.setClearDepth(0.0);
        depth_attachment.setStoreAction(MTLStoreAction::DontCare);

        let command_encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
            .context("creating render command encoder")?;
        // A group's target is pooled and so may be larger than the group; the
        // viewport is the part of it the group actually occupies, in its
        // top-left corner, and `render_target.origin` is where that sits in
        // the window.
        command_encoder.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: i32::from(target.render_target.size.width) as f64,
            height: i32::from(target.render_target.size.height) as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        command_encoder.setDepthStencilState(Some(&self.unordered_depth_state));
        crate::fast::rounded_clip::default_mask(&command_encoder);
        bind_render_target(&command_encoder, &target.render_target);
        // No quad is drawn by an opaque pass in these frames, so none is culled
        // from the quad pass for having been.
        let depth_scale = 0f32;
        // SAFETY: Metal copies the value, which matches every vertex shader's
        // `GlobalInputIndex_DepthScale` argument.
        unsafe {
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&depth_scale).cast(),
                mem::size_of_val(&depth_scale),
                GlobalInputIndex::DepthScale as usize,
            );
        }
        if let Some(clip) = &self.clip {
            // SAFETY: The clip masks were written for this scene at an aligned
            // offset, and both slots are the ones the clipped variants declare.
            // Metal retains the buffer and the atlas.
            unsafe {
                command_encoder.setFragmentBuffer_offset_atIndex(
                    Some(&instance_bindings.clip_masks.buffer),
                    instance_bindings.clip_masks.offset,
                    ClipInputIndex::Masks as usize,
                );
                command_encoder.setFragmentTexture_atIndex(
                    Some(&clip.atlas),
                    ClipInputIndex::Atlas as usize,
                );
            }
        }
        Ok(command_encoder)
    }
}
