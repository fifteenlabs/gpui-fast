//! Arbitrary-path clips: the coverage atlas a frame rasterizes its clip paths
//! into by stencil-and-cover, and the clipped pipeline variants that sample it.
use super::{
    DevicePixels, InstanceBufferWriter, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder,
    MTLCompareFunction, MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice, MTLLibrary,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLStorageMode,
    MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureUsage, MTLViewport, MetalRenderer,
    PRIMITIVE_PIPELINES, ProtocolObject, Result, Retained, Scene, Size, build_clipped_pipelines,
    mem, ptr, shader_function, size,
};
use anyhow::Context as _;
use gpui::{CLIP_TEXTURE_QUANTUM, ClipCover, ClipPlan, Extent, FillRule};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLColorWriteMask, MTLOrigin, MTLScissorRect, MTLSize,
    MTLStencilDescriptor, MTLStencilOperation, MTLTextureType,
};

/// Clip masks are rasterized with 4x MSAA, which every device supports.
/// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const CLIP_SAMPLE_COUNT: u32 = 4;

/// The stencil bit `FillRule::EvenOdd` inverts. The nonzero rule uses the whole
/// byte as a counter instead.
const EVEN_ODD_STENCIL_MASK: u32 = 1;


#[repr(C)]
enum ClipMaskInputIndex {
    Vertices = 0,
    /// The size of the attachment a mask pass draws into, which is the working
    /// pair rather than the atlas: a level is rasterized at its own origin and
    /// blitted into the atlas afterwards.
    TargetSize = 1,
    Cover = 2,
    ClipAtlas = 3,
}


impl MetalRenderer {
    /// Works out where in the coverage atlas each of this scene's clip paths
    /// goes, sizing the clip textures and building the clip pipelines if the
    /// scene is the first to ask for either.
    pub(super) fn prepare_clip_masks(
        &mut self,
        scene: &Scene,
        viewport_size: Size<DevicePixels>,
    ) -> Result<ClipPlan> {
        if scene.clips.is_empty() {
            // A frame that clips nothing still counts against the budget, so a
            // window that stops clipping - a dialog closing, an email view
            // going away - hands its textures back instead of keeping them for
            // the life of the process.
            if self.clip.is_some() {
                let atlas = self.clip_atlas_budget.floor();
                self.clip_atlas_budget
                    .observe(atlas, Extent::ZERO.quantized_to(CLIP_TEXTURE_QUANTUM));
                let work = self.clip_work_budget.floor();
                self.clip_work_budget
                    .observe(work, Extent::ZERO.quantized_to(CLIP_TEXTURE_QUANTUM));
                self.ensure_clip_resources(atlas, work)?;
            }
            return Ok(ClipPlan::unclipped());
        }
        let plan = ClipPlan::new(scene, viewport_size, self.clip_atlas_budget.floor());
        self.clip_atlas_budget
            .observe(plan.atlas, plan.minimum_atlas);
        let work = self.clip_work_budget.floor().max(plan.work);
        self.clip_work_budget.observe(work, plan.work);
        self.ensure_clip_resources(plan.atlas, work)?;
        Ok(plan)
    }

    fn ensure_clip_resources(&mut self, atlas: Extent, work: Extent) -> Result<()> {
        let device = self.device.clone();
        match &mut self.clip {
            None => {
                self.clip = Some(ClipResources::new(
                    &device,
                    &self.library,
                    self.is_apple_gpu,
                    atlas,
                    work,
                )?);
            }
            Some(clip) => clip.resize(&device, self.is_apple_gpu, atlas, work)?,
        }
        Ok(())
    }

    /// Rasterizes every clip mask this frame samples, one render pass per
    /// nesting level, shallowest first.
    ///
    /// A level is drawn into a working pair and then blitted into the atlas a
    /// tile at a time, because a multisample resolve covers the whole
    /// attachment: resolving straight into the atlas would wipe the tiles the
    /// level below just wrote, which is exactly what the level above is about
    /// to read.
    ///
    /// The working pair is therefore what a resolve costs, and it is sized to
    /// the largest level rather than to the atlas - a level whose tiles are one
    /// 40x40 clip resolves 40x40, not the whole atlas. Levels are packed into
    /// their own band of shelves for that reason, and a level with no tiles at
    /// all is skipped rather than clearing and resolving for nothing.
    pub(super) fn render_clip_masks(
        &self,
        plan: &ClipPlan,
        writer: &mut InstanceBufferWriter,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<()> {
        if plan.levels.iter().all(|level| level.clips.is_empty()) {
            return Ok(());
        }
        let clip = self
            .clip
            .as_ref()
            .context("a scene with clip paths reached the renderer without clip resources")?;
        let vertex_bindings = writer.write(&plan.vertices)?;
        let target_size = size(
            DevicePixels(clip.work_size.width),
            DevicePixels(clip.work_size.height),
        );

        for level in &plan.levels {
            if level.clips.is_empty() {
                continue;
            }
            let descriptor = MTLRenderPassDescriptor::new();
            // SAFETY: Index zero is within the render pass's color attachment array.
            let color_attachment =
                unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
            color_attachment.setTexture(Some(&clip.multisample));
            color_attachment.setResolveTexture(Some(&clip.scratch));
            color_attachment.setLoadAction(MTLLoadAction::Clear);
            color_attachment.setClearColor(MTLClearColor {
                red: 0.,
                green: 0.,
                blue: 0.,
                alpha: 0.,
            });
            color_attachment.setStoreAction(MTLStoreAction::MultisampleResolve);
            let stencil_attachment = descriptor.stencilAttachment();
            stencil_attachment.setTexture(Some(&clip.stencil));
            stencil_attachment.setLoadAction(MTLLoadAction::Clear);
            stencil_attachment.setClearStencil(0);
            stencil_attachment.setStoreAction(MTLStoreAction::DontCare);

            let encoder = command_buffer
                .renderCommandEncoderWithDescriptor(&descriptor)
                .context("creating clip mask render command encoder")?;
            encoder.setViewport(MTLViewport {
                originX: 0.,
                originY: 0.,
                width: clip.work_size.width as f64,
                height: clip.work_size.height as f64,
                znear: 0.,
                zfar: 1.,
            });
            encoder.setStencilReferenceValue(0);

            for &index in &level.clips {
                let Some(planned) = plan.clips[index].as_ref() else {
                    continue;
                };
                let even_odd = planned.fill_rule == FillRule::EvenOdd;
                encoder.setScissorRect(MTLScissorRect {
                    x: planned.work.x as usize,
                    y: planned.work.y as usize,
                    width: planned.work.width as usize,
                    height: planned.work.height as usize,
                });

                if !planned.vertices.is_empty() {
                    encoder.setRenderPipelineState(&clip.stencil_pipeline);
                    encoder.setDepthStencilState(Some(if even_odd {
                        &clip.even_odd_stencil_state
                    } else {
                        &clip.nonzero_stencil_state
                    }));
                    // SAFETY: The clip's vertices lie inside the buffer written
                    // above, the slots match `clip_stencil_vertex`, and the
                    // target size is copied from a live value.
                    unsafe {
                        encoder.setVertexBuffer_offset_atIndex(
                            Some(&vertex_bindings.buffer),
                            vertex_bindings.offset,
                            ClipMaskInputIndex::Vertices as usize,
                        );
                        encoder.setVertexBytes_length_atIndex(
                            ptr::NonNull::from(&target_size).cast(),
                            mem::size_of_val(&target_size),
                            ClipMaskInputIndex::TargetSize as usize,
                        );
                        encoder.drawPrimitives_vertexStart_vertexCount(
                            MTLPrimitiveType::Triangle,
                            planned.vertices.start,
                            planned.vertices.len(),
                        );
                    }
                }

                encoder.setRenderPipelineState(&clip.cover_pipeline);
                encoder.setDepthStencilState(Some(if even_odd {
                    &clip.even_odd_cover_state
                } else {
                    &clip.nonzero_cover_state
                }));
                // SAFETY: The slots match `clip_cover_vertex` and
                // `clip_cover_fragment`; bytes are copied from live values and
                // Metal retains the atlas.
                unsafe {
                    encoder.setVertexBytes_length_atIndex(
                        ptr::NonNull::from(&target_size).cast(),
                        mem::size_of_val(&target_size),
                        ClipMaskInputIndex::TargetSize as usize,
                    );
                    encoder.setVertexBytes_length_atIndex(
                        ptr::NonNull::from(&planned.cover).cast(),
                        mem::size_of::<ClipCover>(),
                        ClipMaskInputIndex::Cover as usize,
                    );
                    encoder.setFragmentBytes_length_atIndex(
                        ptr::NonNull::from(&planned.cover).cast(),
                        mem::size_of::<ClipCover>(),
                        ClipMaskInputIndex::Cover as usize,
                    );
                    encoder.setFragmentTexture_atIndex(
                        Some(&clip.atlas),
                        ClipMaskInputIndex::ClipAtlas as usize,
                    );
                    encoder.drawPrimitives_vertexStart_vertexCount(
                        MTLPrimitiveType::TriangleStrip,
                        0,
                        4,
                    );
                }
            }
            encoder.endEncoding();

            let blit = command_buffer
                .blitCommandEncoder()
                .context("creating clip mask blit command encoder")?;
            for &index in &level.clips {
                let Some(planned) = plan.clips[index].as_ref() else {
                    continue;
                };
                // SAFETY: Both textures are alive and retained by the command
                // buffer; the tile lies inside the working texture and inside
                // the atlas, which the plan sized to hold it.
                unsafe {
                    blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &clip.scratch,
                        0,
                        0,
                        MTLOrigin {
                            x: planned.work.x as usize,
                            y: planned.work.y as usize,
                            z: 0,
                        },
                        MTLSize {
                            width: planned.work.width as usize,
                            height: planned.work.height as usize,
                            depth: 1,
                        },
                        &clip.atlas,
                        0,
                        0,
                        MTLOrigin {
                            x: planned.atlas.x as usize,
                            y: planned.atlas.y as usize,
                            z: 0,
                        },
                    );
                }
            }
            blit.endEncoding();
        }
        Ok(())
    }
}

/// The pipelines, stencil states and textures an arbitrary-path clip needs.
///
/// Built the first time a scene carries a clip path and kept afterwards. Doing
/// it lazily is what lets the clipped variant of every pipeline exist at all:
/// building them up front would double this renderer's pipeline compilation at
/// startup for a feature nothing in gpui's own UI uses.
///
/// The bill for that is nine pipeline states compiled on the render thread
/// during the first frame that clips anything - two mask pipelines and a
/// clipped variant of each of the seven drawing ones - which on a cold shader
/// cache is a hitch the user sees once. Moving it off the critical path means
/// building them on another thread and handing them over, which is a change to
/// how this renderer owns its state rather than a change to this function.
pub(crate) struct ClipResources {
    stencil_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    cover_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    nonzero_stencil_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    even_odd_stencil_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    nonzero_cover_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    even_odd_cover_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    pub(crate) primitive_pipelines:
        [Retained<ProtocolObject<dyn MTLRenderPipelineState>>; PRIMITIVE_PIPELINES.len()],
    /// The coverage the clipped fragment shaders sample, one tile per clip.
    /// Never cleared: a tile is either written whole by this frame or never
    /// read, because a clip that has no tile says so in its `ClipMask`.
    pub(crate) atlas: Retained<ProtocolObject<dyn MTLTexture>>,
    /// Where a level's tiles are resolved before being blitted into the atlas.
    scratch: Retained<ProtocolObject<dyn MTLTexture>>,
    multisample: Retained<ProtocolObject<dyn MTLTexture>>,
    stencil: Retained<ProtocolObject<dyn MTLTexture>>,
    atlas_size: Extent,
    /// The size of the three working textures above, which is the size of the
    /// largest nesting level rather than of the atlas.
    work_size: Extent,
}

impl ClipResources {
    fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        is_apple_gpu: bool,
        atlas_size: Extent,
        work_size: Extent,
    ) -> Result<Self> {
        let atlas = Self::coverage(device, atlas_size)?;
        let (scratch, multisample, stencil) = Self::working(device, is_apple_gpu, work_size)?;
        Ok(Self {
            stencil_pipeline: build_clip_mask_pipeline_state(
                device,
                library,
                "clip_stencil",
                "clip_stencil_vertex",
                "clip_stencil_fragment",
                false,
            )?,
            cover_pipeline: build_clip_mask_pipeline_state(
                device,
                library,
                "clip_cover",
                "clip_cover_vertex",
                "clip_cover_fragment",
                true,
            )?,
            nonzero_stencil_state: build_stencil_state(device, FillRule::NonZero)?,
            even_odd_stencil_state: build_stencil_state(device, FillRule::EvenOdd)?,
            nonzero_cover_state: build_cover_state(device, FillRule::NonZero)?,
            even_odd_cover_state: build_cover_state(device, FillRule::EvenOdd)?,
            primitive_pipelines: build_clipped_pipelines(device, library)?,
            atlas,
            scratch,
            multisample,
            stencil,
            atlas_size,
            work_size,
        })
    }

    /// Reallocates whichever of the two groups of textures has changed size.
    /// The atlas and the working pair move independently: a frame can need a
    /// bigger atlas without needing a bigger level, and usually does.
    fn resize(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        is_apple_gpu: bool,
        atlas_size: Extent,
        work_size: Extent,
    ) -> Result<()> {
        if self.atlas_size != atlas_size {
            self.atlas = Self::coverage(device, atlas_size)?;
            self.atlas_size = atlas_size;
        }
        if self.work_size != work_size {
            let (scratch, multisample, stencil) = Self::working(device, is_apple_gpu, work_size)?;
            self.scratch = scratch;
            self.multisample = multisample;
            self.stencil = stencil;
            self.work_size = work_size;
        }
        Ok(())
    }

    fn coverage(
        device: &ProtocolObject<dyn MTLDevice>,
        extent: Extent,
    ) -> Result<Retained<ProtocolObject<dyn MTLTexture>>> {
        let descriptor = MTLTextureDescriptor::new();
        // SAFETY: A quantized extent is positive and this descriptor creates a 2D texture.
        unsafe {
            descriptor.setWidth(extent.width as usize);
            descriptor.setHeight(extent.height as usize);
        }
        descriptor.setPixelFormat(MTLPixelFormat::R8Unorm);
        descriptor.setStorageMode(MTLStorageMode::Private);
        descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        device
            .newTextureWithDescriptor(&descriptor)
            .context("allocating a clip coverage texture")
    }

    /// The resolve target and the multisample pair one nesting level is drawn
    /// through, before its tiles are blitted into the atlas.
    #[allow(clippy::type_complexity)]
    fn working(
        device: &ProtocolObject<dyn MTLDevice>,
        is_apple_gpu: bool,
        extent: Extent,
    ) -> Result<(
        Retained<ProtocolObject<dyn MTLTexture>>,
        Retained<ProtocolObject<dyn MTLTexture>>,
        Retained<ProtocolObject<dyn MTLTexture>>,
    )> {
        // The multisample pair is written and consumed inside one render pass,
        // so on Apple silicon it never needs backing store at all.
        let transient = if is_apple_gpu {
            MTLStorageMode::Memoryless
        } else {
            MTLStorageMode::Private
        };
        let multisample = |format| {
            let descriptor = MTLTextureDescriptor::new();
            // SAFETY: A quantized extent is positive, and the sample count is
            // the one the clip mask pipelines rasterize with.
            unsafe {
                descriptor.setWidth(extent.width as usize);
                descriptor.setHeight(extent.height as usize);
                descriptor.setSampleCount(CLIP_SAMPLE_COUNT as usize);
            }
            descriptor.setPixelFormat(format);
            descriptor.setTextureType(MTLTextureType::Type2DMultisample);
            descriptor.setStorageMode(transient);
            descriptor.setUsage(MTLTextureUsage::RenderTarget);
            device
                .newTextureWithDescriptor(&descriptor)
                .context("allocating a multisample clip texture")
        };
        Ok((
            Self::coverage(device, extent)?,
            multisample(MTLPixelFormat::R8Unorm)?,
            multisample(MTLPixelFormat::Stencil8)?,
        ))
    }
}

/// The two halves of stencil-and-cover. Neither blends: the stencil half writes
/// no colour at all, and the cover half replaces the tile's coverage outright.
fn build_clip_mask_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    writes_coverage: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vertex_fn = shader_function(library, vertex_fn_name, None)?;
    let fragment_fn = shader_function(library, fragment_fn_name, None)?;

    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(label)));
    descriptor.setVertexFunction(Some(vertex_fn.as_ref()));
    descriptor.setFragmentFunction(Some(fragment_fn.as_ref()));
    descriptor.setRasterSampleCount(CLIP_SAMPLE_COUNT as usize);
    descriptor.setStencilAttachmentPixelFormat(MTLPixelFormat::Stencil8);
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(MTLPixelFormat::R8Unorm);
    color_attachment.setBlendingEnabled(false);
    color_attachment.setWriteMask(if writes_coverage {
        MTLColorWriteMask::All
    } else {
        MTLColorWriteMask::empty()
    });

    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

/// Counts winding into the stencil buffer.
///
/// The nonzero rule counts up on one facing and down on the other, so a
/// self-overlapping contour and the interior edges of a triangle fan cancel
/// exactly. The even-odd rule flips a single bit instead, which is the same
/// count taken modulo two.
fn build_stencil_state(
    device: &ProtocolObject<dyn MTLDevice>,
    fill_rule: FillRule,
) -> Result<Retained<ProtocolObject<dyn MTLDepthStencilState>>> {
    let descriptor = MTLDepthStencilDescriptor::new();
    let face = |operation| {
        let face = MTLStencilDescriptor::new();
        face.setStencilCompareFunction(MTLCompareFunction::Always);
        face.setStencilFailureOperation(MTLStencilOperation::Keep);
        face.setDepthFailureOperation(MTLStencilOperation::Keep);
        face.setDepthStencilPassOperation(operation);
        face.setReadMask(0);
        face.setWriteMask(match fill_rule {
            FillRule::EvenOdd => EVEN_ODD_STENCIL_MASK,
            FillRule::NonZero => u32::MAX,
        });
        face
    };
    match fill_rule {
        FillRule::EvenOdd => {
            descriptor.setFrontFaceStencil(Some(&face(MTLStencilOperation::Invert)));
            descriptor.setBackFaceStencil(Some(&face(MTLStencilOperation::Invert)));
        }
        FillRule::NonZero => {
            descriptor.setFrontFaceStencil(Some(&face(MTLStencilOperation::IncrementWrap)));
            descriptor.setBackFaceStencil(Some(&face(MTLStencilOperation::DecrementWrap)));
        }
    }
    device
        .newDepthStencilStateWithDescriptor(&descriptor)
        .context("creating a clip stencil state")
}

/// Draws the tile wherever the winding says the path covers it, and zeroes the
/// stencil behind itself so the next clip in the same pass starts clean.
fn build_cover_state(
    device: &ProtocolObject<dyn MTLDevice>,
    fill_rule: FillRule,
) -> Result<Retained<ProtocolObject<dyn MTLDepthStencilState>>> {
    let mask = match fill_rule {
        FillRule::EvenOdd => EVEN_ODD_STENCIL_MASK,
        FillRule::NonZero => u32::MAX,
    };
    let descriptor = MTLDepthStencilDescriptor::new();
    let face = || {
        let face = MTLStencilDescriptor::new();
        // The reference value is left at zero, so this is "the winding is not
        // zero" under either rule.
        face.setStencilCompareFunction(MTLCompareFunction::NotEqual);
        face.setStencilFailureOperation(MTLStencilOperation::Keep);
        face.setDepthFailureOperation(MTLStencilOperation::Keep);
        face.setDepthStencilPassOperation(MTLStencilOperation::Zero);
        face.setReadMask(mask);
        face.setWriteMask(mask);
        face
    };
    descriptor.setFrontFaceStencil(Some(&face()));
    descriptor.setBackFaceStencil(Some(&face()));
    device
        .newDepthStencilStateWithDescriptor(&descriptor)
        .context("creating a clip cover state")
}

