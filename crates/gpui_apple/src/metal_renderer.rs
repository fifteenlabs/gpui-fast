use crate::metal_atlas::MetalAtlas;
use anyhow::{Context as _, Result};
use block2::RcBlock;
use gpui::{
    AtlasTextureId, AtlasTextureKind, AtlasTile, Background, Bounds, BrushExtend,
    CLIP_TEXTURE_QUANTUM, ClipCover, ClipId, ClipPlan, ClipTextureBudget, ComposeMode, ContentMask,
    DevicePixels, DeviceRect, Extent, FillRule, FilterOp, GroupFilter, GroupSpec, Hsla, MixMode,
    PaintSurface, Path, PathBrush, Point, PrimitiveBatch, ScaledPixels, Scene, SceneFilter,
    SceneStep, Size, TextureBudget, TileId, TransformationMatrix, point, size,
};
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
use image::RgbaImage;
use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_core_foundation::{CFRetained, CGSize};
use objc2_core_graphics::{CGColorSpace, kCGColorSpaceDisplayP3};
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBufferGetHeight,
    CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
    CVPixelBufferGetWidthOfPlane, kCVPixelFormatType_32BGRA,
    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVReturnSuccess,
};
use objc2_foundation::{NSRange, NSString};
#[cfg(target_os = "macos")]
use objc2_metal::MTLCopyAllDevices;
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
use objc2_metal::MTLRegion;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBlitCommandEncoder, MTLBuffer, MTLClearColor,
    MTLColorWriteMask, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompareFunction,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDepthStencilDescriptor, MTLDepthStencilState,
    MTLDevice, MTLDrawable, MTLFunctionConstantValues, MTLGPUFamily, MTLLibrary, MTLLoadAction,
    MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResourceOptions, MTLScissorRect,
    MTLSize, MTLStencilDescriptor, MTLStencilOperation, MTLStorageMode, MTLStoreAction, MTLTexture,
    MTLTextureDescriptor, MTLTextureType, MTLTextureUsage, MTLViewport,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use parking_lot::Mutex;

use std::{
    cell::Cell, collections::HashMap, mem, mem::MaybeUninit, ops::Range, ptr, slice, sync::Arc,
};

// Exported to metal
pub(crate) type PointF = gpui::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;
/// Metal requires the offset a buffer is bound at to be 256-byte aligned.
const INSTANCE_BUFFER_ALIGNMENT: usize = 256;
const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;

/// The stencil bit `FillRule::EvenOdd` inverts. The nonzero rule uses the whole
/// byte as a counter instead.
const EVEN_ODD_STENCIL_MASK: u32 = 1;

/// How deep isolated groups may nest before the innermost ones are painted
/// straight onto what is underneath them.
///
/// Each level costs a render target of its own, and a document nests nothing
/// like this far: `opacity` inside `mask-image` inside an inset shadow is
/// three.
const MAX_GROUP_DEPTH: usize = 8;
/// What a group target's sides are rounded up to, for the same reason
/// [`gpui::CLIP_TEXTURE_QUANTUM`] exists: a group whose bounds move a pixel a
/// frame is then the same allocation rather than a new texture every frame.
const GROUP_TEXTURE_QUANTUM: i32 = 256;
/// How many consecutive frames have to fit inside smaller group targets before
/// the renderer gives the memory back. Growing is immediate; shrinking waits,
/// exactly as the clip textures' own shrink delay does.
const GROUP_TEXTURE_SHRINK_FRAMES: u32 = 240;
/// The most memory the group targets may hold between them.
///
/// A group is sized to its own bounds, so this is only reached by groups that
/// each cover most of the window at several nesting levels at once - on a
/// retina display one full-screen target is already around 28 MB. Past it a
/// group is painted without isolation and says so, which is a wrong picture,
/// but a loud one and a bounded amount of memory.
const GROUP_TARGET_BUDGET_BYTES: usize = 128 * 1024 * 1024;
/// The most texels one blur pass reads on each side of the texel it writes.
///
/// [`SceneFilter::Blur`] carries a standard deviation and the tail worth
/// drawing runs to three of them, so every blur up to a sigma of a third of
/// this is integrated at every texel it covers and nothing is approximated.
/// Past that the pass keeps the same support and widens its step instead - a
/// coarser Riemann sum of the same integral, which is the shape the shadow
/// fragment shader's own five-step sum over y already has - and reports that it
/// did.
const MAX_BLUR_TAPS: i32 = 96;
/// How many working textures a filter chain may hold at once.
///
/// The most that has to be live together is a drop shadow inside a group that
/// also carries a backdrop filter: the group's filtered result, the untouched
/// copy of the backdrop, and the two a blur ping-pongs between.
const MAX_FILTER_SCRATCH: usize = 5;
/// The most memory the filter working textures may hold between them. They are
/// sized to the groups that use them, and only a group carrying a blur, a drop
/// shadow or a backdrop filter takes one at all.
const FILTER_SCRATCH_BUDGET_BYTES: usize = 128 * 1024 * 1024;

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

pub fn new_renderer(
    context: self::Context,
    transparent: bool,
    use_display_p3: bool,
) -> Result<Renderer> {
    MetalRenderer::new(context, transparent, use_display_p3)
}

pub fn new_overlay_renderer(context: self::Context, base: &Renderer) -> Result<Renderer> {
    crate::fast::composition::new_overlay_renderer(context, base)
}

pub struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<InstanceBuffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    size: usize,
}

// SAFETY: Each buffer has one CPU writer and returns to the mutex-protected pool only after GPU completion.
unsafe impl Send for InstanceBuffer {}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        unified_memory: bool,
    ) -> Result<InstanceBuffer> {
        if let Some(buffer) = self.buffers.pop() {
            return Ok(buffer);
        }
        let options = if unified_memory {
            MTLResourceOptions::StorageModeShared
                // Buffers are write only which can benefit from the combined cache
                // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                | MTLResourceOptions::CPUCacheModeWriteCombined
        } else {
            MTLResourceOptions::StorageModeManaged
        };
        let buffer = device
            .newBufferWithLength_options(self.buffer_size, options)
            .with_context(|| format!("allocating {}-byte instance buffer", self.buffer_size))?;
        Ok(InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        })
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer)
        }
    }
}

enum DisplayLinkMode {
    Detached,
    Attached {
        drawable: Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>,
    },
}

pub struct MetalRenderer {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Kept so the clipped pipeline variants can be specialized out of it the
    /// first time a scene actually carries a clip path. Building them eagerly
    /// would double this renderer's pipeline count at startup for a feature
    /// nothing in gpui's own UI uses.
    pub(crate) library: Retained<ProtocolObject<dyn MTLLibrary>>,
    clip: Option<ClipResources>,
    clip_atlas_budget: ClipTextureBudget,
    clip_work_budget: ClipTextureBudget,
    group_targets: GroupTargets,
    /// The working textures a group's filter chain runs through. Empty until a
    /// scene carries a blur, a drop shadow or a backdrop filter.
    filter_scratch: FilterScratch,
    /// Built the first time a scene actually composites a group, and only for
    /// the operators it uses: there are seven of them times a clipped and an
    /// unclipped variant times a backdrop and a plain one, and gpui's own UI
    /// asks for none.
    group_composite_pipelines: HashMap<
        (GroupComposeMode, bool, bool),
        Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    >,
    /// Built the first time a scene carries a filter that needs a pass of its
    /// own.
    filter_pipelines: HashMap<FilterPipeline, Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    /// Whether what this renderer draws the frame into can be copied out of
    /// again, which is what a top-level backdrop filter needs.
    ///
    /// An offscreen target always can. A window's drawable cannot until the
    /// layer is told to stop being framebuffer-only, which costs the display
    /// some of its fast paths and is therefore not paid until a scene turns up
    /// carrying a backdrop filter.
    drawable_is_readable: bool,
    /// Whether the layer has already been asked to stop being framebuffer-only.
    /// The drawables it had already vended were made under the old setting, so
    /// `drawable_is_readable` only follows on the frame after this is set.
    layer_reads_requested: bool,
    pub(crate) layer: Option<Retained<CAMetalLayer>>,
    is_apple_gpu: bool,
    pub(crate) is_unified_memory: bool,
    presents_with_transaction: bool,
    display_link_mode: DisplayLinkMode,
    /// For headless rendering, tracks whether output should be opaque
    pub(crate) opaque: bool,
    pub(crate) command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub(crate) primitive_pipelines:
        [Retained<ProtocolObject<dyn MTLRenderPipelineState>>; PRIMITIVE_PIPELINES.len()],
    /// Composites the rasterized path intermediate, and has no clipped twin on
    /// purpose: the clip is applied while the path is rasterized, by
    /// [`PrimitivePipeline::PathsRasterization`], so by the time the sprite is
    /// composited the coverage already carries it.
    path_sprites_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub(crate) unit_vertices: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Bound wherever a path rasterization pass has no image brush to sample: a
    /// fragment shader that declares a `texture2d` argument needs something
    /// bound even in the frames that never read it.
    default_brush_texture: Retained<ProtocolObject<dyn MTLTexture>>,
    pub(crate) instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    pub(crate) sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: CFRetained<CVMetalTextureCache>,
    pub(crate) path_intermediate_texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    pub(crate) path_intermediate_msaa_texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    path_sample_count: u32,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    headless_render_target: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    pub(crate) fast_layers: crate::fast::layers::TileCache,
}

#[repr(C)]
pub struct PathRasterizationVertex {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
    pub color: Background,
    pub bounds: Bounds<ScaledPixels>,
    /// Which path this vertex belongs to, and so which entry of the content
    /// mask buffer bound alongside these vertices applies to it.
    ///
    /// The mask lives in its own per-path buffer rather than here because a
    /// `ContentMask` is 32 bytes and a path can have thousands of vertices.
    /// `bounds` above is *not* the mask's rectangle - it is the path's bounds
    /// intersected with it, which is what the hardware clip planes want - so a
    /// rounded mask cannot be reconstructed from what is already here.
    pub path_id: u32,
}

/// One entry of the per-path brush buffer, indexed by
/// [`PathRasterizationVertex::path_id`] exactly as the content mask buffer
/// beside it is.
///
/// It sits here rather than in `Background` because a background is embedded by
/// value in every quad and in every one of these vertices; a path that has no
/// brush spends one of these, and a path that has one spends no bytes per
/// vertex at all.
#[repr(C)]
pub struct PathBrushRecord {
    pub brush: PathBrush,
    /// Whether `brush` means anything. Most paths in a batch that has any brush
    /// in it have none of their own, and this is how they say so.
    pub enabled: u32,
}

/// A stretch of one path batch's vertices that can be drawn with one brush
/// atlas texture bound.
///
/// `texture` is `None` while no path in the run has asked for one, which is the
/// whole batch in every frame that brushes nothing.
struct BrushRun {
    texture: Option<AtlasTextureId>,
    vertices: Range<u64>,
}

impl BrushRun {
    /// Whether a path wanting `texture` can be drawn in this run: one that
    /// wants no texture always can, and one that wants a texture can as long as
    /// the run has not committed to a different one.
    fn accepts(&self, texture: Option<AtlasTextureId>) -> bool {
        match (self.texture, texture) {
            (_, None) | (None, Some(_)) => true,
            (Some(bound), Some(wanted)) => bound == wanted,
        }
    }
}

impl PathBrushRecord {
    fn enabled(brush: PathBrush) -> Self {
        Self { brush, enabled: 1 }
    }

    /// The entry an unbrushed path gets. Its tile names texture zero, which the
    /// shader never reads because `enabled` is zero.
    fn disabled() -> Self {
        Self {
            brush: PathBrush {
                tile: AtlasTile {
                    texture_id: AtlasTextureId {
                        index: 0,
                        kind: AtlasTextureKind::Polychrome,
                    },
                    tile_id: TileId(0),
                    padding: 0,
                    bounds: Bounds::default(),
                },
                screen_to_brush: TransformationMatrix::unit(),
                x_extend: BrushExtend::Pad,
                y_extend: BrushExtend::Pad,
                opacity: 0.,
            },
            enabled: 0,
        }
    }
}

impl MetalRenderer {
    /// Creates a new MetalRenderer with a CAMetalLayer for window-based rendering.
    pub fn new(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        transparent: bool,
        use_display_p3: bool,
    ) -> Result<Self> {
        let device = Self::create_device()?;
        let layer = CAMetalLayer::new();
        Self::configure_layer(&layer, &device, transparent, use_display_p3);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a renderer for a CAMetalLayer owned by a platform view, such as
    /// the backing layer UIKit creates for a view whose `layerClass` is
    /// `CAMetalLayer`. The renderer retains the layer.
    pub fn from_layer(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        layer: &CAMetalLayer,
        transparent: bool,
    ) -> Result<Self> {
        let device = Self::create_device()?;
        let layer = layer.retain();
        Self::configure_layer(&layer, &device, transparent, false);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    pub(crate) fn configure_layer(
        layer: &CAMetalLayer,
        device: &ProtocolObject<dyn MTLDevice>,
        transparent: bool,
        use_display_p3: bool,
    ) {
        layer.setDevice(Some(device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        if use_display_p3 {
            // SAFETY: The constant is extern, from Core Graphics
            let display_p3_name = unsafe { kCGColorSpaceDisplayP3 };
            if let Some(display_p3_color_space) = CGColorSpace::with_name(Some(display_p3_name)) {
                layer.setColorspace(Some(&display_p3_color_space));
            } else {
                log::error!("unable to create the Display P3 color space");
            }
        }
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.setOpaque(!transparent);
        layer.setMaximumDrawableCount(3);
        // Allow texture reading for visual tests and UI automation screenshots
        // (captures without ScreenCaptureKit). Only debug builds pay the
        // presentation cost, even when `test-support` is compiled in.
        #[cfg(all(feature = "test-support", debug_assertions))]
        layer.setFramebufferOnly(false);
        layer.setAllowsNextDrawableTimeout(false);
        layer.setNeedsDisplayOnBoundsChange(true);
        // UIKit sizes a view's backing layer itself; only AppKit-hosted
        // layers need to track their superlayer's bounds.
        #[cfg(target_os = "macos")]
        layer.setAutoresizingMask(
            objc2_quartz_core::CAAutoresizingMask::LayerWidthSizable
                | objc2_quartz_core::CAAutoresizingMask::LayerHeightSizable,
        );
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    ///
    /// `transparent` decides what the offscreen target is cleared to. An opaque
    /// renderer clears to opaque black, which is what a screenshot of a normal
    /// window looks like; a transparent one clears to all zeroes, so the alpha
    /// that reads back is the coverage the scene actually produced rather than
    /// a constant 255.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn new_headless(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        transparent: bool,
    ) -> Result<Self> {
        let device = Self::create_device()?;
        Self::new_internal(device, None, !transparent, instance_buffer_pool)
    }

    #[cfg(target_os = "macos")]
    fn create_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        if let Some(device) = MTLCopyAllDevices()
            .iter()
            .min_by_key(|device| (device.isRemovable(), !device.isLowPower()))
        {
            Ok(device)
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            Self::system_default_device()
        }
    }

    #[cfg(target_os = "ios")]
    fn create_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
        Self::system_default_device()
    }

    fn system_default_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
        MTLCreateSystemDefaultDevice().context("accessing a compatible Metal device")
    }

    pub(crate) fn new_internal(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        layer: Option<Retained<CAMetalLayer>>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Result<Self> {
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(SHADERS_SOURCE_FILE),
                Some(&objc2_metal::MTLCompileOptions::new()),
            )
            .context("building Metal library")?;
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .newLibraryWithData_error(&dispatch2::DispatchData::from_static_bytes(
                SHADERS_METALLIB,
            ))
            .context("loading Metal library")?;

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        // Shared memory can be used only if CPU and GPU share the same memory space.
        // https://developer.apple.com/documentation/metal/setting-resource-storage-modes
        // iOS does not support managed resources. Its simulator may report a
        // non-unified host GPU even though resources must still use shared storage.
        let is_unified_memory = cfg!(target_os = "ios") || device.hasUnifiedMemory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supportsFamily(MTLGPUFamily::Apple1);

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        // SAFETY: Metal copies the live array using its exact byte length.
        let unit_vertices = unsafe {
            device.newBufferWithBytes_length_options(
                ptr::NonNull::from(&unit_vertices).cast(),
                mem::size_of_val(&unit_vertices),
                if is_unified_memory {
                    MTLResourceOptions::StorageModeShared
                        | MTLResourceOptions::CPUCacheModeWriteCombined
                } else {
                    MTLResourceOptions::StorageModeManaged
                },
            )
        }
        .context("allocating unit vertex buffer")?;

        let primitive_pipelines = build_primitive_pipelines(&device, &library, false)?;
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        )?;

        let default_brush_texture = build_default_brush_texture(&device)?;
        let command_queue = crate::fast::dynamic_textures::command_queue(&device)?;
        let supports_shared_storage = cfg!(target_os = "ios") || is_apple_gpu;
        #[expect(
            clippy::arc_with_non_send_sync,
            reason = "PlatformWindow::sprite_atlas requires Arc, not Send or Sync"
        )]
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), supports_shared_storage));
        crate::fast::dynamic_textures::init(device.clone(), supports_shared_storage);
        let mut cache = ptr::null_mut();
        // SAFETY: The output pointer is writable and both optional attribute dictionaries are absent.
        let status = unsafe {
            CVMetalTextureCache::create(None, None, &device, None, ptr::NonNull::from(&mut cache))
        };
        anyhow::ensure!(
            status == kCVReturnSuccess,
            "creating CoreVideo Metal texture cache: status {status}"
        );
        let cache = ptr::NonNull::new(cache).context("retrieving CoreVideo Metal texture cache")?;
        // SAFETY: The successful create call transfers ownership of this non-null cache.
        let core_video_texture_cache = unsafe { CFRetained::from_raw(cache) };

        Ok(Self {
            device,
            library,
            clip: None,
            clip_atlas_budget: ClipTextureBudget::default(),
            clip_work_budget: ClipTextureBudget::default(),
            group_targets: GroupTargets::default(),
            filter_scratch: FilterScratch::default(),
            group_composite_pipelines: HashMap::new(),
            filter_pipelines: HashMap::new(),
            // A headless renderer draws into a texture of its own making, which
            // is readable. A layer's drawables are framebuffer-only until
            // something asks otherwise - except in a debug build with test
            // support, where `configure_layer` already allows reads so that a
            // screenshot can be taken without ScreenCaptureKit.
            drawable_is_readable: layer.is_none()
                || cfg!(all(feature = "test-support", debug_assertions)),
            layer_reads_requested: false,
            layer,
            presents_with_transaction: false,
            display_link_mode: DisplayLinkMode::Detached,
            is_apple_gpu,
            is_unified_memory,
            opaque,
            command_queue,
            primitive_pipelines,
            path_sprites_pipeline_state,
            unit_vertices,
            default_brush_texture,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
            headless_render_target: None,
            fast_layers: crate::fast::layers::TileCache::default(),
        })
    }

    pub fn layer(&self) -> Option<&CAMetalLayer> {
        self.layer.as_deref()
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|layer| Retained::as_ptr(layer).cast_mut())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    /// Draw the next frame into `drawable` instead of asking the layer for one.
    ///
    /// A `CAMetalDisplayLink` hands out each frame's drawable early enough for
    /// the frame to reach the display on time; `nextDrawable` would instead
    /// wait for the compositor to release one. Pass `None` once the frame is
    /// done, so a drawable the frame did not use goes back to the layer.
    ///
    /// Returns the drawable this replaces: after a frame, `Some` means the
    /// frame neither drew nor presented.
    ///
    /// This is a setter rather than a guard scoped to one frame because the
    /// window releases its lock on the renderer while GPUI draws.
    pub fn set_frame_drawable(
        &mut self,
        drawable: Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>,
    ) -> Option<Retained<ProtocolObject<dyn CAMetalDrawable>>> {
        match &mut self.display_link_mode {
            DisplayLinkMode::Attached { drawable: current } => mem::replace(current, drawable),
            DisplayLinkMode::Detached => {
                // Only a live link hands out drawables, and a link is detached
                // only after it has been invalidated.
                debug_assert!(
                    drawable.is_none(),
                    "received a display link drawable while detached"
                );
                None
            }
        }
    }

    /// Record whether a `CAMetalDisplayLink` drives the layer. While one
    /// does, frames are drawn only into the drawables it hands out.
    pub fn set_display_link_attached(&mut self, attached: bool) {
        self.display_link_mode = if attached {
            DisplayLinkMode::Attached { drawable: None }
        } else {
            DisplayLinkMode::Detached
        };
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.setPresentsWithTransaction(presents_with_transaction);
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            layer.setDrawableSize(CGSize::new(size.width.0 as f64, size.height.0 as f64));
        }
        if let Err(error) = self.update_path_intermediate_textures(size) {
            log::error!("failed to resize path textures: {error:#}");
        }
        crate::fast::layers::TileCache::clear(&mut self.fast_layers);
    }

    pub(crate) fn update_path_intermediate_textures(
        &mut self,
        size: Size<DevicePixels>,
    ) -> Result<()> {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            return Ok(());
        }

        if self
            .path_intermediate_texture
            .as_ref()
            .is_some_and(|texture| {
                texture.width() == size.width.0 as usize
                    && texture.height() == size.height.0 as usize
            })
        {
            return Ok(());
        }

        self.path_intermediate_texture = None;
        self.path_intermediate_msaa_texture = None;
        let texture_descriptor = MTLTextureDescriptor::new();
        // SAFETY: The dimensions are positive and this descriptor creates a 2D texture.
        unsafe {
            texture_descriptor.setWidth(size.width.0 as usize);
            texture_descriptor.setHeight(size.height.0 as usize);
        }
        texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.setStorageMode(MTLStorageMode::Private);
        texture_descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let intermediate_texture = self
            .device
            .newTextureWithDescriptor(&texture_descriptor)
            .context("allocating path intermediate texture")?;

        let msaa_texture = if self.path_sample_count > 1 {
            // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
            // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
            let storage_mode = if self.is_apple_gpu {
                MTLStorageMode::Memoryless
            } else {
                MTLStorageMode::Private
            };

            let msaa_descriptor = texture_descriptor;
            msaa_descriptor.setTextureType(MTLTextureType::Type2DMultisample);
            msaa_descriptor.setStorageMode(storage_mode);
            // SAFETY: The multisample texture uses the same sample count as its render pipeline.
            unsafe {
                msaa_descriptor.setSampleCount(self.path_sample_count as _);
            }
            Some(
                self.device
                    .newTextureWithDescriptor(&msaa_descriptor)
                    .context("allocating multisample path texture")?,
            )
        } else {
            None
        };
        self.path_intermediate_texture = Some(intermediate_texture);
        self.path_intermediate_msaa_texture = msaa_texture;
        Ok(())
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.setOpaque(!transparent);
        }
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    pub fn draw(&mut self, scene: &Scene) {
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!(
                    "draw() called on headless renderer - use render_scene_to_image() instead"
                );
                return;
            }
        };
        // A backdrop filter copies out of whatever the group is composited
        // over, and at the top level that is the drawable itself. A
        // framebuffer-only layer forbids the copy, so the first scene that asks
        // for one buys the window out of it - the flag is not set from the
        // start because it costs the display its direct-to-display path, and
        // nothing in gpui's own UI has ever needed a backdrop filter. The
        // drawables already in the layer's pool were made under the old flag,
        // so this frame still composites over an unfiltered backdrop and says
        // so; the next one does not.
        if !self.drawable_is_readable {
            if self.layer_reads_requested {
                self.drawable_is_readable = true;
            } else if scene_has_a_backdrop_filter(scene) {
                layer.setFramebufferOnly(false);
                self.layer_reads_requested = true;
            }
        }

        let drawable = match &mut self.display_link_mode {
            DisplayLinkMode::Attached { drawable } => {
                let Some(drawable) = drawable.take() else {
                    log::error!("skipped a frame drawn outside its display link's update");
                    return;
                };
                drawable
            }
            DisplayLinkMode::Detached => {
                let Some(drawable) = layer.nextDrawable() else {
                    log::error!(
                        "failed to retrieve next drawable, drawable size: {:?}",
                        layer.drawableSize()
                    );
                    return;
                };
                drawable
            }
        };
        // A display link's drawable was sized when the link handed it out,
        // which can be before a resize the layer has since taken.
        let texture = drawable.texture();
        let viewport_size: Size<DevicePixels> = size(
            (texture.width() as i32).into(),
            (texture.height() as i32).into(),
        );

        let command_buffer = match self.render_frame(scene, &drawable.texture(), viewport_size) {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                log::error!("failed to render: {error:#}");
                return;
            }
        };

        if self.presents_with_transaction {
            command_buffer.commit();
            command_buffer.waitUntilScheduled();
            drawable.present();
        } else {
            command_buffer.presentDrawable(ProtocolObject::from_ref(&*drawable));
            command_buffer.commit();
        }
    }

    pub(crate) fn render_frame(
        &mut self,
        scene: &Scene,
        texture: &ProtocolObject<dyn MTLTexture>,
        viewport_size: Size<DevicePixels>,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if self.path_intermediate_texture.is_none() {
            self.update_path_intermediate_textures(viewport_size)?;
        }
        crate::fast::layers::raster::rasterize_tiles(self, scene);
        let clip_plan = self.prepare_clip_masks(scene, viewport_size)?;
        let mut writer = InstanceBufferWriter::new(
            &self.device,
            &self.instance_buffer_pool,
            self.is_unified_memory,
        )?;
        let instance_bindings = write_instances(scene, &clip_plan, &mut writer).with_context(|| {
            format!(
                "writing instance data: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                scene.paths.len(),
                scene.shadows.len(),
                scene.quads.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
            )
        })?;
        let mut surface_textures = Vec::new();
        let command_buffer = self.draw_primitives_to_texture(
            scene,
            &clip_plan,
            &instance_bindings,
            &mut writer,
            texture,
            viewport_size,
            &mut surface_textures,
        )?;

        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let frame_resources = Cell::new(Some((writer.finish(), surface_textures)));
        let block = RcBlock::new(
            move |_: ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                if let Some((instance_buffer, _surface_textures)) = frame_resources.take() {
                    instance_buffer_pool.lock().release(instance_buffer);
                }
            },
        );
        // SAFETY: Metal copies the live block and invokes it once after this frame completes.
        // Its captures own the frame resources and use the pool mutex when returning the buffer.
        unsafe {
            command_buffer.addCompletedHandler(RcBlock::as_ptr(&block));
        }

        Ok(command_buffer)
    }

    /// Renders the scene to a texture and returns the pixel data as an RGBA image.
    /// This does not present the frame to screen - useful for visual testing
    /// where we want to capture what would be rendered without displaying it.
    ///
    /// Note: This requires a layer-backed renderer. For headless rendering,
    /// use `render_scene_to_image()` instead.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        let layer = self
            .layer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("render_to_image requires a layer-backed renderer"))?;
        let viewport_size = layer.drawableSize();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        // CAMetalDisplayLink owns the drawable, so render into a texture we own.
        self.render_offscreen(scene, viewport_size)
    }

    /// Renders a scene to an image without requiring a window or CAMetalLayer.
    ///
    /// This is the primary method for headless rendering. It creates an offscreen
    /// texture, renders the scene to it, and returns the pixel data as an RGBA image.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_image: {:?}", size);
        }

        // Headless callers do not have a Cocoa event-loop pool to release
        // autoreleased command buffers and render-pass descriptors.
        objc2::rc::autoreleasepool(|_| {
            self.update_path_intermediate_textures(size)?;
            self.render_offscreen(scene, size)
        })
    }

    /// Renders a scene into a new offscreen texture of `size` and reads it
    /// back, without presenting anything.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    fn render_offscreen(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_offscreen: {:?}", size);
        }

        let texture_descriptor = MTLTextureDescriptor::new();
        // SAFETY: The dimensions are positive and this descriptor creates a 2D texture.
        unsafe {
            texture_descriptor.setWidth(size.width.0 as usize);
            texture_descriptor.setHeight(size.height.0 as usize);
        }
        texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        // Like the atlas, only Apple GPUs can create shared textures on macOS;
        // Intel Macs cannot, even with unified memory. iOS has no managed storage.
        let uses_shared_storage = cfg!(target_os = "ios") || self.is_apple_gpu;
        texture_descriptor.setStorageMode(if uses_shared_storage {
            MTLStorageMode::Shared
        } else {
            MTLStorageMode::Managed
        });
        let target_texture = self
            .device
            .newTextureWithDescriptor(&texture_descriptor)
            .context("allocating offscreen render target")?;

        let command_buffer = self.render_frame(scene, &target_texture, size)?;

        // Managed textures require an explicit blit synchronize before the CPU
        // can read back the rendered data. Without this, get_bytes returns
        // stale zeros.
        if !uses_shared_storage {
            let blit = command_buffer
                .blitCommandEncoder()
                .context("creating blit command encoder")?;
            blit.synchronizeResource(ProtocolObject::from_ref(&*target_texture));
            blit.endEncoding();
        }

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        read_texture_to_image(&target_texture)
    }

    /// Renders a scene to a reused offscreen texture without reading pixels
    /// back or blocking on GPU completion.
    ///
    /// This mirrors the CPU cost of presenting a frame to a window (scene
    /// encoding, instance buffer writes, command submission) and is used by
    /// headless benchmark rendering, where the produced pixels are never
    /// inspected.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene: {:?}", size);
        }

        objc2::rc::autoreleasepool(|_| {
            self.update_path_intermediate_textures(size)?;

            let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
                texture.width() != size.width.0 as usize
                    || texture.height() != size.height.0 as usize
            });
            if needs_new_target {
                let texture_descriptor = MTLTextureDescriptor::new();
                unsafe {
                    texture_descriptor.setWidth(size.width.0 as usize);
                    texture_descriptor.setHeight(size.height.0 as usize);
                }
                texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
                texture_descriptor
                    .setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
                texture_descriptor.setStorageMode(MTLStorageMode::Private);
                self.headless_render_target = Some(
                    self.device
                        .newTextureWithDescriptor(&texture_descriptor)
                        .context("allocating headless render target")?,
                );
            }
            let target_texture = self
                .headless_render_target
                .clone()
                .context("retrieving headless render target")?;

            let command_buffer = self.render_frame(scene, &target_texture, size)?;

            // Commit without waiting, mirroring presentation to a real window where
            // the CPU doesn't block on the GPU.
            command_buffer.commit();
            Ok(())
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_primitives_to_texture(
        &mut self,
        scene: &Scene,
        clip_plan: &ClipPlan,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        texture: &ProtocolObject<dyn MTLTexture>,
        viewport_size: Size<DevicePixels>,
        surface_textures: &mut Vec<CFRetained<CVMetalTexture>>,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if crate::fast::rounded_clip::has_groups(scene) {
            return crate::fast::rounded_clip::draw(
                self,
                scene,
                instance_bindings,
                writer,
                texture,
                viewport_size,
                surface_textures,
            );
        }
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue
            .commandBuffer()
            .context("creating command buffer")?;
        crate::fast::dynamic_textures::encode_uploads(&command_buffer);
        let alpha = if self.opaque { 1. } else { 0. };

        // Every clip mask this frame samples is rasterized before the first
        // primitive is drawn, so a batch never has to interrupt the main pass
        // to go and build one.
        self.render_clip_masks(clip_plan, writer, &command_buffer)?;
        self.group_targets.begin_frame();
        self.filter_scratch.begin_frame();

        // The first group of path batches is rasterized before the first pass
        // begins, which saves ending that pass for it.
        let path_batches = crate::fast::paths::plan(scene);
        if let Some(group) = path_batches.first().and_then(|batch| batch.group.clone()) {
            crate::fast::paths::rasterize_group(
                self,
                scene,
                &path_batches[group],
                instance_bindings,
                writer,
                viewport_size,
                &command_buffer,
            )?;
        }
        let mut next_path_batch = 0;

        // The scene is walked as steps rather than as batches so that a group
        // boundary can end one render pass and begin another. A scene with no
        // groups in it is one step of one run, which is the same batches in the
        // same order into the same encoder as before.
        let mut target = ActiveTarget {
            texture: texture.retain(),
            render_target: RenderTarget::whole(viewport_size),
            is_window: true,
        };
        let mut command_encoder = new_command_encoder_for_texture(
            &command_buffer,
            &target.texture,
            target.render_target,
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
                    match self.open_group(spec, &target, viewport_size, open_groups.len())? {
                        Some(group) => {
                            command_encoder.endEncoding();
                            target = group.target.clone();
                            command_encoder = new_command_encoder_for_texture(
                                &command_buffer,
                                &target.texture,
                                target.render_target,
                                Some(MTLClearColor {
                                    red: 0.,
                                    green: 0.,
                                    blue: 0.,
                                    alpha: 0.,
                                }),
                            )?;
                            open_groups.push(Some(group));
                        }
                        None => open_groups.push(None),
                    }
                    continue;
                }
                SceneStep::PopGroup => {
                    let group = open_groups
                        .pop()
                        .context("a scene popped a group it never pushed")?;
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
                    command_encoder = new_command_encoder_for_texture(
                        &command_buffer,
                        &target.texture,
                        target.render_target,
                        None,
                    )?;
                    let composited = self.composite_group(
                        &group,
                        &prepared,
                        instance_bindings,
                        &command_encoder,
                    );
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
                match batch {
                    PrimitiveBatch::Shadows(range) => {
                        let clipped = scene.shadows[range.start].clip.is_clipped();
                        self.draw_shadows(
                            range,
                            clipped,
                            instance_bindings,
                            &target.render_target,
                            &command_encoder,
                        )
                    }
                    PrimitiveBatch::Quads(range) => {
                        let clipped = scene.quads[range.start].clip.is_clipped();
                        self.draw_quads(
                            range,
                            clipped,
                            instance_bindings,
                            &target.render_target,
                            &command_encoder,
                        )
                    }
                    PrimitiveBatch::Paths(range) => {
                        if range.is_empty() {
                            continue;
                        }
                        let index = next_path_batch;
                        next_path_batch += 1;
                        let path_batch = &path_batches[index];
                        debug_assert_eq!(path_batch.paths, range);

                        // Paths are rasterized into a window-sized intermediate
                        // whatever target they are on their way to, so that
                        // pass keeps the viewport it always had; only the copy
                        // out of it lands on the group.
                        if index > 0
                            && let Some(group) = path_batch.group.clone()
                        {
                            command_encoder.endEncoding();
                            crate::fast::paths::rasterize_group(
                                self,
                                scene,
                                &path_batches[group],
                                instance_bindings,
                                writer,
                                viewport_size,
                                &command_buffer,
                            )?;
                            command_encoder = new_command_encoder_for_texture(
                                &command_buffer,
                                &target.texture,
                                target.render_target,
                                None,
                            )?;
                        }

                        if let Err(error) = self.draw_paths_from_intermediate(
                            &scene.paths[range],
                            writer,
                            &target.render_target,
                            viewport_size,
                            &command_encoder,
                        ) {
                            command_encoder.endEncoding();
                            return Err(error);
                        }
                    }
                    PrimitiveBatch::Underlines(range) => {
                        let clipped = scene.underlines[range.start].clip.is_clipped();
                        self.draw_underlines(
                            range,
                            clipped,
                            instance_bindings,
                            &target.render_target,
                            &command_encoder,
                        )
                    }
                    PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                        let clipped = scene.monochrome_sprites[range.start].clip.is_clipped();
                        self.draw_monochrome_sprites(
                            texture_id,
                            range,
                            clipped,
                            instance_bindings,
                            &target.render_target,
                            &command_encoder,
                        )
                    }
                    PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                        let clipped = scene.polychrome_sprites[range.start].clip.is_clipped();
                        self.draw_polychrome_sprites(
                            texture_id,
                            range,
                            clipped,
                            instance_bindings,
                            &target.render_target,
                            &command_encoder,
                        )
                    }
                    PrimitiveBatch::Surfaces(range) => {
                        if let Err(error) = self.draw_surfaces(
                            &scene.surfaces[range.clone()],
                            range.start,
                            instance_bindings,
                            &target.render_target,
                            &command_encoder,
                            surface_textures,
                        ) {
                            command_encoder.endEncoding();
                            return Err(error);
                        }
                    }
                    PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
                }
            }
        }

        command_encoder.endEncoding();
        self.group_targets.end_frame();
        self.filter_scratch.end_frame();

        Ok(command_buffer)
    }

    /// Give a group about to be pushed a render target of its own, or `None`
    /// when it has to be painted straight onto what is underneath it.
    ///
    /// Falling back is a wrong picture - two overlapping children at 0.5 come
    /// out at 0.75 where they overlap, a `DestIn` group paints instead of
    /// masking - so every reason to fall back says so once per frame rather
    /// than quietly drawing something else.
    fn open_group(
        &mut self,
        spec: &GroupSpec,
        parent: &ActiveTarget,
        viewport_size: Size<DevicePixels>,
        depth: usize,
    ) -> Result<Option<OpenGroup>> {
        let viewport = DeviceRect::viewport(viewport_size);
        // A group whose bounds have fallen off the window paints nothing either
        // way, so this is not worth a line in the log - and by the time a spec
        // reaches a renderer its bounds cover everything painted inside it
        // (`Scene::grow_group_to_its_content`), so an empty rectangle here is a
        // group that painted nothing rather than one whose caller left
        // `GroupOptions::bounds` at its default.
        let rect = DeviceRect::covering(&spec.bounds).intersect(&viewport);
        if rect.is_empty() {
            return Ok(None);
        }
        if depth >= MAX_GROUP_DEPTH {
            self.group_targets.report(format_args!(
                "isolated groups nested more than {MAX_GROUP_DEPTH} deep; \
                 the innermost are painted without isolation"
            ));
            return Ok(None);
        }
        let Some(texture) = self.group_targets.acquire(
            &self.device,
            depth,
            rect,
            GROUP_TARGET_BUDGET_BYTES,
            "the group is painted without isolation",
        )?
        else {
            return Ok(None);
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        Ok(Some(OpenGroup {
            spec: spec.clone(),
            bounds: rect.bounds(),
            texture_size,
            target: ActiveTarget {
                texture,
                render_target: RenderTarget {
                    size: size(DevicePixels(rect.width), DevicePixels(rect.height)),
                    origin: point(DevicePixels(rect.x), DevicePixels(rect.y)),
                },
                is_window: false,
            },
            parent: parent.clone(),
        }))
    }

    /// Run everything a group's `filter` and `backdrop_filter` ask for, between
    /// the pass that painted the group and the one that composites it.
    ///
    /// Each step is a whole-image quad from one texture into another, and the
    /// last texture written is what the composite samples. A step that cannot
    /// have a working texture stops the chain where it is rather than losing
    /// the steps already run, and the pool has already said why.
    fn prepare_group(
        &mut self,
        group: &OpenGroup,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        viewport_size: Size<DevicePixels>,
    ) -> Result<PreparedGroup> {
        let mut prepared = PreparedGroup {
            source: FilterImage {
                texture: group.target.texture.clone(),
                texture_size: group.texture_size,
                rect: DeviceRect::covering(&group.bounds),
                slot: None,
            },
            filter: GroupFilter::NONE,
            backdrop: None,
        };

        if let Some(filter) = &group.spec.filter {
            let mut ops = FilterOp::flatten(filter);
            // A trailing run of colour matrices costs no pass at all: the
            // composite's own fragment stage applies one on its way down, which
            // is what it did when a matrix was the only filter there was.
            if let Some(FilterOp::Matrices(matrices)) = ops.last() {
                prepared.filter = *matrices;
                ops.pop();
            }
            prepared.source = self.run_filter_ops(
                &ops,
                prepared.source,
                FilterEdge::Transparent,
                command_buffer,
            )?;
        }

        if let Some(backdrop) = &group.spec.backdrop_filter {
            prepared.backdrop =
                self.prepare_backdrop(group, backdrop, command_buffer, viewport_size)?;
        }

        Ok(prepared)
    }

    /// Copy what is already on the target underneath the group, filter the
    /// copy, and hand back both: CSS `backdrop-filter`.
    ///
    /// Both halves are needed, not just the filtered one. The group's coverage
    /// - its opacity, and the clip path in force - fades the *filter* in as
    /// well as the group, so a fragment the group only half covers shows half
    /// the filtered backdrop and half the backdrop it started from, and the
    /// composite cannot read the destination it is writing to in order to
    /// find the other half.
    fn prepare_backdrop(
        &mut self,
        group: &OpenGroup,
        filter: &SceneFilter,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        viewport_size: Size<DevicePixels>,
    ) -> Result<Option<PreparedBackdrop>> {
        if GroupComposeMode::of(group.spec.blend.compose) != Some(GroupComposeMode::SrcOver) {
            self.group_targets.report(format_args!(
                "a backdrop filter under the {:?} compose operator is not implemented; the \
                 group is composited over an unfiltered backdrop",
                group.spec.blend.compose
            ));
            return Ok(None);
        }
        if !self.drawable_is_readable && group.parent.is_window {
            // The layer's drawables were framebuffer-only when this one was
            // handed out, so nothing may copy out of it. `draw` has already
            // asked for readable ones; this group gets its backdrop from the
            // next frame on.
            self.group_targets.report(format_args!(
                "the window's drawable cannot be read back yet; the group is composited over \
                 an unfiltered backdrop for one frame"
            ));
            return Ok(None);
        }

        // The backdrop reaches past the group: a blur inside the group's own
        // rectangle still draws on what is beside it, and cropping first would
        // ring the group with a band the filter had nothing to work from. What
        // there is to copy stops at the target underneath, which is the
        // backdrop root - a group is composited onto its parent's target, so a
        // nested group's backdrop is what is inside its parent and no further,
        // which is exactly the boundary the filter effects specification draws.
        let parent = DeviceRect {
            x: i32::from(group.parent.render_target.origin.x),
            y: i32::from(group.parent.render_target.origin.y),
            width: i32::from(group.parent.render_target.size.width),
            height: i32::from(group.parent.render_target.size.height),
        };
        let viewport = DeviceRect::viewport(viewport_size);
        let wanted = DeviceRect::covering(&filter.painted_bounds(group.bounds));
        let rect = wanted.intersect(&parent).intersect(&viewport);
        if rect.is_empty() {
            return Ok(None);
        }

        let Some(original) = self.filter_scratch.acquire(&self.device, rect)? else {
            return Ok(None);
        };
        let blit = command_buffer
            .blitCommandEncoder()
            .context("creating backdrop blit command encoder")?;
        // SAFETY: Both textures are alive for the call and the command buffer
        // retains them; the copied region lies inside the parent target and
        // inside the scratch texture, which is at least `rect` large.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &group.parent.texture,
                0,
                0,
                MTLOrigin {
                    x: (rect.x - parent.x) as usize,
                    y: (rect.y - parent.y) as usize,
                    z: 0,
                },
                MTLSize {
                    width: rect.width as usize,
                    height: rect.height as usize,
                    depth: 1,
                },
                &original.texture,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
            );
        }
        blit.endEncoding();

        // The chain must not release the copy: the composite reads it too. A
        // clone without a slot is the same texture that nothing will hand back.
        let mut borrowed = original.clone();
        borrowed.slot = None;
        let ops = FilterOp::flatten(filter);
        let filtered = self.run_filter_ops(&ops, borrowed, FilterEdge::Clamp, command_buffer)?;
        Ok(Some(PreparedBackdrop { original, filtered }))
    }

    /// Run a flattened filter, one image at a time, and return the last one.
    fn run_filter_ops(
        &mut self,
        ops: &[FilterOp],
        input: FilterImage,
        edge: FilterEdge,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<FilterImage> {
        let mut current = input;
        for op in ops {
            let next = match op {
                FilterOp::Matrices(matrices) => {
                    self.run_matrix_pass(&current, matrices, edge, command_buffer)?
                }
                FilterOp::Blur { sigma_x, sigma_y } => {
                    self.run_blur(&current, *sigma_x, *sigma_y, edge, command_buffer)?
                }
                FilterOp::DropShadow {
                    offset,
                    sigma,
                    color,
                } => {
                    self.run_drop_shadow(&current, *offset, *sigma, *color, edge, command_buffer)?
                }
            };
            let Some(next) = next else {
                break;
            };
            self.filter_scratch.release(&current);
            current = next;
        }
        Ok(current)
    }

    fn run_matrix_pass(
        &mut self,
        source: &FilterImage,
        matrices: &GroupFilter,
        edge: FilterEdge,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<Option<FilterImage>> {
        let Some(destination) = self.filter_scratch.acquire(&self.device, source.rect)? else {
            return Ok(None);
        };
        let pass = FilterPass::new(source.size(), edge);
        let pipeline = self.filter_pipeline(FilterPipeline::Matrices)?;
        self.draw_filter_pass(
            &pipeline,
            &pass,
            Some(matrices),
            &source.texture,
            None,
            &destination,
            command_buffer,
        )?;
        Ok(Some(destination))
    }

    /// A separable gaussian: one pass along x, one along y.
    ///
    /// Two passes rather than one square kernel because the gaussian is
    /// separable and the saving is the whole cost model. A single pass over a
    /// `(2n+1)` square reads `(2n+1)^2` texels; two one-dimensional passes read
    /// `2(2n+1)`. At the sigma an email asks for - four device pixels, so
    /// twelve texels of tail each way - that is 625 reads against 50.
    fn run_blur(
        &mut self,
        source: &FilterImage,
        sigma_x: f32,
        sigma_y: f32,
        edge: FilterEdge,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<Option<FilterImage>> {
        let mut current: Option<FilterImage> = None;
        for (sigma, direction) in [(sigma_x, point(1., 0.)), (sigma_y, point(0., 1.))] {
            if sigma <= 0. {
                continue;
            }
            let destination = match self.filter_scratch.acquire(&self.device, source.rect)? {
                Some(destination) => destination,
                // Nothing has been released yet, so whatever the first pass
                // produced is still a complete image - blurred along one axis
                // rather than two, which is wrong, and already reported.
                None => return Ok(current),
            };
            let from = current.as_ref().unwrap_or(source);
            let pass = self.blur_pass(from, sigma, direction, edge);
            let pipeline = self.filter_pipeline(FilterPipeline::Blur)?;
            self.draw_filter_pass(
                &pipeline,
                &pass,
                None,
                &from.texture,
                None,
                &destination,
                command_buffer,
            )?;
            if let Some(previous) = current.take() {
                self.filter_scratch.release(&previous);
            }
            current = Some(destination);
        }
        Ok(current)
    }

    /// The group's own alpha, blurred, offset and tinted, drawn behind the
    /// group itself: CSS `drop-shadow()`.
    ///
    /// Unlike [`crate::MetalRenderer::draw_shadows`], which is handed a
    /// rectangle and a corner radius and integrates the gaussian over that
    /// shape analytically, this has no shape to integrate: the alpha it blurs
    /// is whatever the group happened to paint - text, an image with holes in
    /// it, a path - and the only way to blur that is to blur the pixels. The
    /// two agree on what a standard deviation means and on how far the tail is
    /// worth drawing, so a `drop-shadow` and a `box-shadow` of the same radius
    /// come out the same weight. Where the two differ is in how each arrives at
    /// it: the quad shadow's answer is analytic and this one is a sum over
    /// texels, so at a standard deviation of a device pixel or two they part by
    /// a few hundredths, and at zero the quad shadow antialiases an edge it
    /// knows the shape of where this one takes the source's alpha as it stands.
    fn run_drop_shadow(
        &mut self,
        source: &FilterImage,
        offset: PointF,
        sigma: f32,
        color: Hsla,
        edge: FilterEdge,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<Option<FilterImage>> {
        // The shadow's own blur is always over transparent black, whatever the
        // chain is running over: it is the source's alpha spreading into empty
        // space, and clamping it to the edge would smear the outermost row of
        // the group across the whole margin.
        let blurred = if sigma > 0. {
            let Some(blurred) = self.run_blur(
                source,
                sigma,
                sigma,
                FilterEdge::Transparent,
                command_buffer,
            )?
            else {
                return Ok(None);
            };
            Some(blurred)
        } else {
            None
        };
        let destination = self.filter_scratch.acquire(&self.device, source.rect)?;
        let Some(destination) = destination else {
            if let Some(blurred) = &blurred {
                self.filter_scratch.release(blurred);
            }
            return Ok(None);
        };

        let mut pass = FilterPass::new(source.size(), edge);
        pass.offset = offset;
        pass.color = color;
        let pipeline = self.filter_pipeline(FilterPipeline::DropShadow)?;
        let shadow_texture = blurred.as_ref().unwrap_or(source).texture.clone();
        let drawn = self.draw_filter_pass(
            &pipeline,
            &pass,
            None,
            &source.texture,
            Some(&shadow_texture),
            &destination,
            command_buffer,
        );
        if let Some(blurred) = &blurred {
            self.filter_scratch.release(blurred);
        }
        drawn?;
        Ok(Some(destination))
    }

    /// How many texels a blur reads, and how far apart.
    ///
    /// [`gpui::GAUSSIAN_BLUR_EXTENT`] standard deviations is the tail the scene
    /// already grew the group's target by, so reading exactly that far is
    /// reading everything there is and no more. Past [`MAX_BLUR_TAPS`] the
    /// support stays where it is and the step widens, which is a coarser
    /// Riemann sum of the same integral rather than a shorter one - the shape
    /// the shadow fragment shader's five-step sum over y already has - and it
    /// says so, because a coarser sum is an approximation and this file counts
    /// those.
    fn blur_pass(
        &mut self,
        source: &FilterImage,
        sigma: f32,
        direction: PointF,
        edge: FilterEdge,
    ) -> FilterPass {
        // Both of these round up, on values that are positive by construction.
        let divide_rounding_up = |value: i32, by: i32| (value + by - 1) / by;
        let tail = (sigma * gpui::GAUSSIAN_BLUR_EXTENT).ceil().max(1.);
        let tail = tail.min(i32::MAX as f32) as i32;
        let stride = divide_rounding_up(tail, MAX_BLUR_TAPS).max(1);
        if stride > 1 {
            self.group_targets.report(format_args!(
                "a gaussian of {sigma} device pixels reaches {tail} texels, past the \
                 {MAX_BLUR_TAPS} one pass reads; it is integrated every {stride} texels instead"
            ));
        }
        let mut pass = FilterPass::new(source.size(), edge);
        pass.direction = direction;
        pass.sigma = sigma;
        pass.taps = divide_rounding_up(tail, stride);
        pass.stride = stride;
        pass
    }

    /// Draw one filter pass: a quad over the whole of `destination`, reading
    /// `source` and writing what the pipeline's fragment stage makes of it.
    #[allow(clippy::too_many_arguments, reason = "one call site per pass kind")]
    fn draw_filter_pass(
        &self,
        pipeline: &ProtocolObject<dyn MTLRenderPipelineState>,
        pass: &FilterPass,
        matrices: Option<&GroupFilter>,
        source: &ProtocolObject<dyn MTLTexture>,
        blurred: Option<&ProtocolObject<dyn MTLTexture>>,
        destination: &FilterImage,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<()> {
        let encoder = new_command_encoder_for_texture(
            command_buffer,
            &destination.texture,
            destination.render_target(),
            Some(MTLClearColor {
                red: 0.,
                green: 0.,
                blue: 0.,
                alpha: 0.,
            }),
        )?;
        encoder.setRenderPipelineState(pipeline);
        // SAFETY: The slots match `filter_pass_vertex` and the fragment function
        // the pipeline was built with; bytes are copied from live values, and
        // Metal retains the bound buffer and textures.
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                FilterInputIndex::Vertices as usize,
            );
            encoder.setFragmentBytes_length_atIndex(
                ptr::NonNull::from(pass).cast(),
                mem::size_of::<FilterPass>(),
                FilterInputIndex::Pass as usize,
            );
            // Bound only where the pipeline declares them. A `GroupFilter` is 656
            // bytes, and a separable gaussian is two passes, so filling the slot
            // unconditionally uploaded 1312 bytes of zeroes per blurred group per
            // frame into a buffer the blur fragment shader does not name.
            if let Some(matrices) = matrices {
                encoder.setFragmentBytes_length_atIndex(
                    ptr::NonNull::from(matrices).cast(),
                    mem::size_of::<GroupFilter>(),
                    FilterInputIndex::Filter as usize,
                );
            }
            encoder.setFragmentTexture_atIndex(Some(source), FilterInputIndex::Source as usize);
            if let Some(blurred) = blurred {
                encoder
                    .setFragmentTexture_atIndex(Some(blurred), FilterInputIndex::Blurred as usize);
            }
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 6);
        }
        encoder.endEncoding();
        Ok(())
    }

    /// The pipeline one kind of filter pass draws with, built the first time a
    /// scene asks for it.
    fn filter_pipeline(
        &mut self,
        kind: FilterPipeline,
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        if let Some(pipeline) = self.filter_pipelines.get(&kind) {
            return Ok(pipeline.clone());
        }
        let pipeline = build_filter_pipeline_state(
            &self.device,
            &self.library,
            kind,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        self.filter_pipelines.insert(kind, pipeline.clone());
        Ok(pipeline)
    }

    /// Give every working texture a prepared group holds back to the pool.
    fn release_prepared(&mut self, prepared: PreparedGroup) {
        self.filter_scratch.release(&prepared.source);
        if let Some(backdrop) = &prepared.backdrop {
            self.filter_scratch.release(&backdrop.original);
            self.filter_scratch.release(&backdrop.filtered);
        }
    }

    /// Draw a finished group's target onto the target underneath it: one quad
    /// over the group's bounds, blended with the operator the group asked for.
    fn composite_group(
        &mut self,
        group: &OpenGroup,
        prepared: &PreparedGroup,
        instance_bindings: &InstanceBindings,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) -> Result<()> {
        if group.spec.blend.mix != MixMode::Normal {
            self.group_targets.report(format_args!(
                "the {:?} blend mode is not implemented; the group is composited normally",
                group.spec.blend.mix
            ));
        }
        let compose = match GroupComposeMode::of(group.spec.blend.compose) {
            Some(compose) => compose,
            None => {
                self.group_targets.report(format_args!(
                    "the {:?} compose operator is not implemented; the group is \
                     composited source-over",
                    group.spec.blend.compose
                ));
                GroupComposeMode::SrcOver
            }
        };
        let clipped = group.spec.clip.is_clipped() && self.clip.is_some();
        let backdrop = prepared.backdrop.as_ref();
        let pipeline = self.group_composite_pipeline(compose, clipped, backdrop.is_some())?;
        command_encoder.setRenderPipelineState(&pipeline);

        let backdrop_rect = backdrop
            .map(|backdrop| backdrop.original.rect)
            .unwrap_or(DeviceRect::EMPTY);
        let composite = GroupComposite {
            bounds: group.bounds,
            texture_size: prepared.source.texture_size,
            clip: group.spec.clip,
            // `f32::clamp` hands a NaN straight back, and a NaN coverage in the
            // fragment stage is a NaN in every channel it multiplies. A group
            // whose opacity is not a number is composited at full strength,
            // which is what the scene already does with one: `opacity < 1.` is
            // false for a NaN, so `Scene::fold_opacity` leaves the primitives
            // alone as well.
            opacity: if group.spec.opacity.is_nan() {
                1.
            } else {
                group.spec.opacity.clamp(0., 1.)
            },
            backdrop_origin: point(backdrop_rect.x as f32, backdrop_rect.y as f32),
            backdrop_size: size(
                DevicePixels(backdrop_rect.width),
                DevicePixels(backdrop_rect.height),
            ),
        };
        // SAFETY: The slots match `group_composite_vertex` and
        // `group_composite_fragment`; bytes are copied from live values, and
        // Metal retains the bound buffer and textures.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                GroupCompositeInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&composite).cast(),
                mem::size_of::<GroupComposite>(),
                GroupCompositeInputIndex::Composite as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&group.parent.render_target).cast(),
                mem::size_of::<RenderTarget>(),
                GroupCompositeInputIndex::RenderTarget as usize,
            );
            command_encoder.setFragmentBytes_length_atIndex(
                ptr::NonNull::from(&composite).cast(),
                mem::size_of::<GroupComposite>(),
                GroupCompositeInputIndex::Composite as usize,
            );
            command_encoder.setFragmentBytes_length_atIndex(
                ptr::NonNull::from(&prepared.filter).cast(),
                mem::size_of::<GroupFilter>(),
                GroupCompositeInputIndex::Filter as usize,
            );
            command_encoder.setFragmentTexture_atIndex(
                Some(&prepared.source.texture),
                GroupCompositeInputIndex::GroupTexture as usize,
            );
            if let Some(backdrop) = backdrop {
                command_encoder.setFragmentTexture_atIndex(
                    Some(&backdrop.original.texture),
                    GroupCompositeInputIndex::BackdropTexture as usize,
                );
                command_encoder.setFragmentTexture_atIndex(
                    Some(&backdrop.filtered.texture),
                    GroupCompositeInputIndex::FilteredBackdropTexture as usize,
                );
            }
        }
        self.bind_clip_mask(
            self.clip_variant(clipped),
            command_encoder,
            instance_bindings,
            GroupCompositeInputIndex::ClipMasks as usize,
            GroupCompositeInputIndex::ClipAtlas as usize,
        );
        // SAFETY: Six unit vertices draw the one composite quad.
        unsafe {
            command_encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                0,
                6,
            );
        }
        Ok(())
    }

    /// The composite pipeline for one operator, built the first time a scene
    /// asks for it.
    fn group_composite_pipeline(
        &mut self,
        compose: GroupComposeMode,
        clipped: bool,
        backdrop: bool,
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        if let Some(pipeline) = self
            .group_composite_pipelines
            .get(&(compose, clipped, backdrop))
        {
            return Ok(pipeline.clone());
        }
        let pipeline = build_group_composite_pipeline_state(
            &self.device,
            &self.library,
            compose,
            clipped,
            backdrop,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        self.group_composite_pipelines
            .insert((compose, clipped, backdrop), pipeline.clone());
        Ok(pipeline)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        clipped: bool,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<bool> {
        if paths.is_empty() {
            return Ok(false);
        }
        let command_encoder = self.new_path_rasterization_encoder(command_buffer)?;
        let rasterized = self.rasterize_paths(
            paths,
            clipped,
            instance_bindings,
            writer,
            viewport_size,
            &command_encoder,
        );
        command_encoder.endEncoding();
        rasterized.map(|()| true)
    }

    /// Begin a pass that clears the path intermediate texture and rasterizes
    /// into it.
    pub(crate) fn new_path_rasterization_encoder(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>> {
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        let render_pass_descriptor = MTLRenderPassDescriptor::new();
        // SAFETY: Index zero is within the render pass's color attachment array.
        let color_attachment = unsafe {
            render_pass_descriptor
                .colorAttachments()
                .objectAtIndexedSubscript(0)
        };
        color_attachment.setLoadAction(MTLLoadAction::Clear);
        color_attachment.setClearColor(MTLClearColor {
            red: 0.,
            green: 0.,
            blue: 0.,
            alpha: 0.,
        });

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.setTexture(Some(msaa_texture));
            color_attachment.setResolveTexture(Some(intermediate_texture));
            color_attachment.setStoreAction(MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.setTexture(Some(intermediate_texture));
            color_attachment.setStoreAction(MTLStoreAction::Store);
        }

        command_buffer
            .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
            .context("creating path render command encoder")
    }

    /// Rasterize one path batch into the intermediate texture, in a pass begun
    /// by `new_path_rasterization_encoder`.
    pub(crate) fn rasterize_paths(
        &self,
        paths: &[Path<ScaledPixels>],
        clipped: bool,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) -> Result<()> {
        let mut vertices = Vec::new();
        let mut content_masks = Vec::with_capacity(paths.len());
        // The clip id rides in the same per-path side buffer the content mask
        // already uses, rather than in the vertex: a path is the highest
        // vertex-count primitive gpui has, so this is one word per path
        // instead of one per vertex.
        let mut clip_ids = Vec::with_capacity(if clipped { paths.len() } else { 0 });
        // An image brush rides in a third one. One texture can be bound per
        // draw, so the batch is also cut into runs of consecutive paths that
        // agree on which texture that is. The runs stay contiguous and in
        // ascending path order: gathering every path that shares a texture into
        // one run would reorder overlapping paths, and source-over is not
        // commutative.
        let brushed = paths.iter().any(|path| path.brush.is_some());
        let mut brushes = Vec::with_capacity(if brushed { paths.len() } else { 0 });
        let mut runs: Vec<BrushRun> = Vec::new();
        for (path_id, path) in paths.iter().enumerate() {
            content_masks.push(path.content_mask);
            if clipped {
                clip_ids.push(path.clip.0);
            }
            if brushed {
                brushes.push(match path.brush {
                    Some(brush) => PathBrushRecord::enabled(brush),
                    None => PathBrushRecord::disabled(),
                });
            }
            let first_vertex = vertices.len() as u64;
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
                path_id: path_id as u32,
            }));
            let texture = path.brush.map(|brush| brush.tile.texture_id);
            match runs.last_mut() {
                Some(run) if run.accepts(texture) => {
                    run.texture = run.texture.or(texture);
                    run.vertices.end = vertices.len() as u64;
                }
                _ => runs.push(BrushRun {
                    texture,
                    vertices: first_vertex..vertices.len() as u64,
                }),
            }
        }
        let vertex_instance_bindings = writer.write(&vertices)?;
        let content_mask_bindings = writer.write(&content_masks)?;
        let clip_id_bindings = if clipped {
            Some(writer.write(&clip_ids)?)
        } else {
            None
        };
        // A fragment shader that declares a buffer needs one bound whether or
        // not it reads it, so a batch with no brush in it still binds a single
        // disabled record - and says, in the count, that no path may look at
        // it.
        let brush_count = brushes.len() as u32;
        if brushes.is_empty() {
            brushes.push(PathBrushRecord::disabled());
        }
        let brush_bindings = writer.write(&brushes)?;

        let clip = self.clip_variant(clipped);
        command_encoder
            .setRenderPipelineState(self.pipeline_for(PrimitivePipeline::PathsRasterization, clip));
        // SAFETY: Every binding was written above for exactly these paths at an
        // aligned offset, into the slots `path_rasterization_vertex` and
        // `path_rasterization_fragment` declare. Metal retains the buffers and
        // textures, bytes are copied from live values, and writer.finish
        // publishes CPU writes before submission.
        unsafe {
            if let (Some(clip), Some(clip_id_bindings)) = (clip, clip_id_bindings.as_ref()) {
                command_encoder.setFragmentBuffer_offset_atIndex(
                    Some(&clip_id_bindings.buffer),
                    clip_id_bindings.offset,
                    PathRasterizationInputIndex::ClipIds as usize,
                );
                command_encoder.setFragmentBuffer_offset_atIndex(
                    Some(&instance_bindings.clip_masks.buffer),
                    instance_bindings.clip_masks.offset,
                    PathRasterizationInputIndex::ClipMasks as usize,
                );
                command_encoder.setFragmentTexture_atIndex(
                    Some(&clip.atlas),
                    PathRasterizationInputIndex::ClipAtlas as usize,
                );
            }
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&vertex_instance_bindings.buffer),
                vertex_instance_bindings.offset,
                PathRasterizationInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                PathRasterizationInputIndex::ViewportSize as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&vertex_instance_bindings.buffer),
                vertex_instance_bindings.offset,
                PathRasterizationInputIndex::Vertices as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&content_mask_bindings.buffer),
                content_mask_bindings.offset,
                PathRasterizationInputIndex::ContentMasks as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&brush_bindings.buffer),
                brush_bindings.offset,
                PathRasterizationInputIndex::Brushes as usize,
            );
            command_encoder.setFragmentBytes_length_atIndex(
                ptr::NonNull::from(&brush_count).cast(),
                mem::size_of_val(&brush_count),
                PathRasterizationInputIndex::BrushCount as usize,
            );
        }

        // `vertex_id` in the vertex shader counts from the first vertex of the
        // draw, not from zero, so each run indexes the same buffer the whole
        // batch was written to and no offset arithmetic is needed here.
        let mut bound_texture = None;
        for run in &runs {
            if run.vertices.is_empty() {
                continue;
            }
            if bound_texture != Some(run.texture) {
                let texture = run
                    .texture
                    .and_then(|texture_id| self.sprite_atlas.metal_texture(texture_id));
                // SAFETY: The brush atlas slot takes any 2D texture; Metal
                // retains it until the pass completes.
                unsafe {
                    command_encoder.setFragmentTexture_atIndex(
                        Some(texture.as_deref().unwrap_or(&self.default_brush_texture)),
                        PathRasterizationInputIndex::BrushAtlas as usize,
                    );
                }
                bound_texture = Some(run.texture);
            }
            // SAFETY: The run's vertices lie inside the vertex buffer bound above.
            unsafe {
                command_encoder.drawPrimitives_vertexStart_vertexCount(
                    MTLPrimitiveType::Triangle,
                    run.vertices.start as usize,
                    (run.vertices.end - run.vertices.start) as usize,
                );
            }
        }
        Ok(())
    }

    pub(crate) fn draw_shadows(
        &self,
        shadows: Range<usize>,
        clipped: bool,
        instance_bindings: &InstanceBindings,
        render_target: &RenderTarget,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if shadows.is_empty() {
            return;
        }

        let clip = self.clip_variant(clipped);
        command_encoder.setRenderPipelineState(self.pipeline_for(PrimitivePipeline::Shadows, clip));
        self.bind_clip_mask(
            clip,
            command_encoder,
            instance_bindings,
            ShadowInputIndex::ClipMasks as usize,
            ShadowInputIndex::ClipAtlas as usize,
        );
        // SAFETY: Six unit vertices and the scene's shadow range match the uploaded shader layouts and slots.
        // Metal retains the buffers, writer.finish publishes their writes, and the render target bytes are copied.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                ShadowInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.shadows.buffer),
                instance_bindings.shadows.offset,
                ShadowInputIndex::Shadows as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.shadows.buffer),
                instance_bindings.shadows.offset,
                ShadowInputIndex::Shadows as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(render_target).cast(),
                mem::size_of_val(render_target),
                ShadowInputIndex::RenderTarget as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                shadows.len(),
                shadows.start,
            );
        }
    }

    pub(crate) fn draw_quads(
        &self,
        quads: Range<usize>,
        clipped: bool,
        instance_bindings: &InstanceBindings,
        render_target: &RenderTarget,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if quads.is_empty() {
            return;
        }

        let clip = self.clip_variant(clipped);
        command_encoder.setRenderPipelineState(self.pipeline_for(PrimitivePipeline::Quads, clip));
        self.bind_clip_mask(
            clip,
            command_encoder,
            instance_bindings,
            QuadInputIndex::ClipMasks as usize,
            QuadInputIndex::ClipAtlas as usize,
        );
        // SAFETY: Six unit vertices and the scene's quad range match the uploaded shader layouts and slots.
        // Metal retains the buffers, writer.finish publishes their writes, and the render target bytes are copied.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                QuadInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.quads.buffer),
                instance_bindings.quads.offset,
                QuadInputIndex::Quads as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.quads.buffer),
                instance_bindings.quads.offset,
                QuadInputIndex::Quads as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(render_target).cast(),
                mem::size_of_val(render_target),
                QuadInputIndex::RenderTarget as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                quads.len(),
                quads.start,
            );
        }
    }

    pub(crate) fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        render_target: &RenderTarget,
        intermediate_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) -> Result<()> {
        let (Some(first_path), Some(last_path)) = (paths.first(), paths.last()) else {
            return Ok(());
        };
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        command_encoder.setRenderPipelineState(&self.path_sprites_pipeline_state);
        // SAFETY: The shader slots match the unit vertices, the copied render
        // target and intermediate size, and the retained path texture. The
        // preceding render pass writes this tracked texture before it is sampled.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                SpriteInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(render_target).cast(),
                mem::size_of_val(render_target),
                SpriteInputIndex::RenderTarget as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&intermediate_size).cast(),
                mem::size_of_val(&intermediate_size),
                SpriteInputIndex::AtlasTextureSize as usize,
            );
            command_encoder.setFragmentTexture_atIndex(
                Some(intermediate_texture),
                SpriteInputIndex::AtlasTexture as usize,
            );
        }

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if last_path.order == first_path.order {
            sprites = paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect();
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            sprites = vec![PathSprite { bounds }];
        }

        let sprite_instance_bindings = writer.write(&sprites)?;
        // SAFETY: The draw uses six unit vertices and exactly the sprites written at this aligned binding.
        // Metal retains the buffer, and writer.finish publishes CPU writes before submission.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&sprite_instance_bindings.buffer),
                sprite_instance_bindings.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::Triangle,
                0,
                6,
                sprites.len(),
            );
        }
        Ok(())
    }

    pub(crate) fn draw_underlines(
        &self,
        underlines: Range<usize>,
        clipped: bool,
        instance_bindings: &InstanceBindings,
        render_target: &RenderTarget,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if underlines.is_empty() {
            return;
        }

        let clip = self.clip_variant(clipped);
        command_encoder
            .setRenderPipelineState(self.pipeline_for(PrimitivePipeline::Underlines, clip));
        self.bind_clip_mask(
            clip,
            command_encoder,
            instance_bindings,
            UnderlineInputIndex::ClipMasks as usize,
            UnderlineInputIndex::ClipAtlas as usize,
        );
        // SAFETY: Six unit vertices and the scene's underline range match the uploaded shader layouts and slots.
        // Metal retains the buffers, writer.finish publishes their writes, and the render target bytes are copied.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                UnderlineInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.underlines.buffer),
                instance_bindings.underlines.offset,
                UnderlineInputIndex::Underlines as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.underlines.buffer),
                instance_bindings.underlines.offset,
                UnderlineInputIndex::Underlines as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(render_target).cast(),
                mem::size_of_val(render_target),
                UnderlineInputIndex::RenderTarget as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                underlines.len(),
                underlines.start,
            );
        }
    }

    pub(crate) fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        clipped: bool,
        instance_bindings: &InstanceBindings,
        render_target: &RenderTarget,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if sprites.is_empty() {
            return;
        }

        let texture = crate::fast::dynamic_textures::metal_texture(&self.device, texture_id)
            .or_else(|| self.sprite_atlas.metal_texture(texture_id));
        let Some(texture) = texture else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        let clip = self.clip_variant(clipped);
        command_encoder
            .setRenderPipelineState(self.pipeline_for(PrimitivePipeline::MonochromeSprites, clip));
        self.bind_clip_mask(
            clip,
            command_encoder,
            instance_bindings,
            SpriteInputIndex::ClipMasks as usize,
            SpriteInputIndex::ClipAtlas as usize,
        );
        // SAFETY: The scene's sprite range, six unit vertices, and copied sizes match the shader layouts and slots.
        // Metal retains the buffers and texture, and writer.finish publishes the instance writes.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                SpriteInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.monochrome_sprites.buffer),
                instance_bindings.monochrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(render_target).cast(),
                mem::size_of_val(render_target),
                SpriteInputIndex::RenderTarget as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&texture_size).cast(),
                mem::size_of_val(&texture_size),
                SpriteInputIndex::AtlasTextureSize as usize,
            );
            // `monochrome_sprite_fragment` reads nothing out of the instance
            // buffer - the vertex stage hands it everything it needs, so that a
            // glyph does not pay for a struct load per fragment - and so the
            // buffer is bound to the vertex stage alone.
            command_encoder.setFragmentTexture_atIndex(
                Some(&texture),
                SpriteInputIndex::AtlasTexture as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                sprites.len(),
                sprites.start,
            );
        }
    }

    pub(crate) fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        clipped: bool,
        instance_bindings: &InstanceBindings,
        render_target: &RenderTarget,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if sprites.is_empty() {
            return;
        }
        if crate::fast::layers::composite::draw_tiles(
            self,
            texture_id,
            &sprites,
            instance_bindings,
            render_target,
            command_encoder,
        ) {
            return;
        }

        let texture = crate::fast::dynamic_textures::metal_texture(&self.device, texture_id)
            .or_else(|| self.sprite_atlas.metal_texture(texture_id));
        let Some(texture) = texture else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        let clip = self.clip_variant(clipped);
        command_encoder
            .setRenderPipelineState(self.pipeline_for(PrimitivePipeline::PolychromeSprites, clip));
        self.bind_clip_mask(
            clip,
            command_encoder,
            instance_bindings,
            SpriteInputIndex::ClipMasks as usize,
            SpriteInputIndex::ClipAtlas as usize,
        );
        // SAFETY: The scene's sprite range, six unit vertices, and copied sizes match the shader layouts and slots.
        // Metal retains the buffers and texture, and writer.finish publishes the instance writes.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                SpriteInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.polychrome_sprites.buffer),
                instance_bindings.polychrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(render_target).cast(),
                mem::size_of_val(render_target),
                SpriteInputIndex::RenderTarget as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&texture_size).cast(),
                mem::size_of_val(&texture_size),
                SpriteInputIndex::AtlasTextureSize as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.polychrome_sprites.buffer),
                instance_bindings.polychrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setFragmentTexture_atIndex(
                Some(&texture),
                SpriteInputIndex::AtlasTexture as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                sprites.len(),
                sprites.start,
            );
        }
    }

    pub(crate) fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        first_surface: usize,
        instance_bindings: &InstanceBindings,
        render_target: &RenderTarget,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        surface_textures: &mut Vec<CFRetained<CVMetalTexture>>,
    ) -> Result<()> {
        let Some(first_surface_record) = surfaces.first() else {
            return Ok(());
        };
        // A batch is homogeneous in whether it is clipped, so the whole run
        // agrees with its first element.
        let clip = self.clip_variant(first_surface_record.clip.is_clipped());
        self.bind_clip_mask(
            clip,
            command_encoder,
            instance_bindings,
            SurfaceInputIndex::ClipMasks as usize,
            SurfaceInputIndex::ClipAtlas as usize,
        );

        // SAFETY: The unit vertices, SurfaceBounds instances, and copied render target match the shader layouts and slots.
        // Metal retains the buffers, and writer.finish publishes CPU writes before submission.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                SurfaceInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.surfaces.buffer),
                instance_bindings.surfaces.offset,
                SurfaceInputIndex::Surfaces as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(render_target).cast(),
                mem::size_of_val(render_target),
                SurfaceInputIndex::RenderTarget as usize,
            );
        }

        for (index, surface) in surfaces.iter().enumerate() {
            let image_buffer = &surface.image_buffer;
            let texture_size = size(
                DevicePixels::from(CVPixelBufferGetWidth(image_buffer) as i32),
                DevicePixels::from(CVPixelBufferGetHeight(image_buffer) as i32),
            );

            let plane_texture =
                |plane, format, width, height| -> Result<CFRetained<CVMetalTexture>> {
                    let mut texture = ptr::null_mut();
                    // SAFETY: The plane and its dimensions come from this buffer's own format, and the output pointer is writable.
                    let status = unsafe {
                        CVMetalTextureCache::create_texture_from_image(
                            None,
                            &self.core_video_texture_cache,
                            image_buffer,
                            None,
                            format,
                            width,
                            height,
                            plane,
                            ptr::NonNull::from(&mut texture),
                        )
                    };
                    anyhow::ensure!(
                        status == kCVReturnSuccess,
                        "creating CoreVideo Metal texture for plane {plane}: status {status}"
                    );
                    let texture =
                        ptr::NonNull::new(texture).context("retrieving CoreVideo Metal texture")?;
                    // SAFETY: The successful create call transfers ownership of this non-null texture.
                    Ok(unsafe { CFRetained::from_raw(texture) })
                };

            match CVPixelBufferGetPixelFormatType(image_buffer) {
                format if format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange => {
                    command_encoder.setRenderPipelineState(
                        self.pipeline_for(PrimitivePipeline::Surfaces, clip),
                    );
                    let y_texture = plane_texture(
                        0,
                        MTLPixelFormat::R8Unorm,
                        CVPixelBufferGetWidthOfPlane(image_buffer, 0),
                        CVPixelBufferGetHeightOfPlane(image_buffer, 0),
                    )?;
                    let cb_cr_texture = plane_texture(
                        1,
                        MTLPixelFormat::RG8Unorm,
                        CVPixelBufferGetWidthOfPlane(image_buffer, 1),
                        CVPixelBufferGetHeightOfPlane(image_buffer, 1),
                    )?;
                    let y_metal_texture = CVMetalTextureGetTexture(&y_texture)
                        .context("retrieving luma Metal texture")?;
                    let cb_cr_metal_texture = CVMetalTextureGetTexture(&cb_cr_texture)
                        .context("retrieving chroma Metal texture")?;
                    // SAFETY: Both planes match their shader texture slots. The
                    // frame completion handler retains the CoreVideo wrappers
                    // until GPU reads finish.
                    unsafe {
                        command_encoder.setFragmentTexture_atIndex(
                            Some(&y_metal_texture),
                            SurfaceInputIndex::YTexture as usize,
                        );
                        command_encoder.setFragmentTexture_atIndex(
                            Some(&cb_cr_metal_texture),
                            SurfaceInputIndex::CbCrTexture as usize,
                        );
                    }
                    surface_textures.extend([y_texture, cb_cr_texture]);
                }
                format if format == kCVPixelFormatType_32BGRA => {
                    command_encoder.setRenderPipelineState(
                        self.pipeline_for(PrimitivePipeline::BgraSurfaces, clip),
                    );
                    // Unlike the video path above (whose AVFoundation buffers
                    // are always cache-compatible), BGRA buffers come from
                    // app code — skip the draw instead of failing the frame if
                    // the texture cache rejects one.
                    let bgra_texture = match plane_texture(
                        0,
                        MTLPixelFormat::BGRA8Unorm,
                        CVPixelBufferGetWidth(image_buffer),
                        CVPixelBufferGetHeight(image_buffer),
                    ) {
                        Ok(texture) => texture,
                        Err(error) => {
                            log::error!(
                                "failed to create Metal texture for BGRA surface: {error:#}; \
                                 skipping draw"
                            );
                            continue;
                        }
                    };
                    let Some(bgra_metal_texture) = CVMetalTextureGetTexture(&bgra_texture) else {
                        log::error!("CoreVideo returned no Metal texture for a BGRA surface");
                        continue;
                    };
                    // SAFETY: The texture matches the shader's first texture
                    // slot. The frame completion handler retains the CoreVideo
                    // wrapper until GPU reads finish.
                    unsafe {
                        command_encoder.setFragmentTexture_atIndex(
                            Some(&bgra_metal_texture),
                            SurfaceInputIndex::YTexture as usize,
                        );
                    }
                    surface_textures.push(bgra_texture);
                }
                format => {
                    log::error!("unsupported surface pixel format: {format:#x}; skipping draw");
                    continue;
                }
            }

            // SAFETY: The size is copied from a live value, and six unit
            // vertices draw one instance within the uploaded scene's surface
            // range.
            unsafe {
                command_encoder.setVertexBytes_length_atIndex(
                    ptr::NonNull::from(&texture_size).cast(),
                    mem::size_of_val(&texture_size),
                    SurfaceInputIndex::TextureSize as usize,
                );
                command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                    MTLPrimitiveType::Triangle,
                    0,
                    6,
                    1,
                    first_surface + index,
                );
            }
        }
        Ok(())
    }

    /// The clipped pipeline variants and the coverage atlas, when this batch
    /// is one that needs them.
    fn clip_variant(&self, clipped: bool) -> Option<&ClipResources> {
        if clipped { self.clip.as_ref() } else { None }
    }

    /// The pipeline that draws `pipeline`, in the specialization this batch
    /// needs.
    ///
    /// Indexing an array rather than reaching for a named field per kind is
    /// what makes a missing clipped pipeline a compile error: reading one out
    /// of `Option<&ClipResources>` by field name would fall back to the
    /// unclipped pipeline, which draws the batch with its clip path silently
    /// ignored.
    fn pipeline_for<'a>(
        &'a self,
        pipeline: PrimitivePipeline,
        clip: Option<&'a ClipResources>,
    ) -> &'a ProtocolObject<dyn MTLRenderPipelineState> {
        let pipelines = clip.map_or(&self.primitive_pipelines, |clip| &clip.primitive_pipelines);
        &pipelines[pipeline as usize]
    }

    /// Binds the coverage atlas and the per-clip records the clipped variant of
    /// a fragment shader reads. A no-op for an unclipped batch, whose pipeline
    /// declares neither.
    fn bind_clip_mask(
        &self,
        clip: Option<&ClipResources>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        instance_bindings: &InstanceBindings,
        masks_index: usize,
        atlas_index: usize,
    ) {
        let Some(clip) = clip else {
            return;
        };
        // SAFETY: The clip masks were written for this scene at an aligned
        // offset, and both slots are the ones the clipped variant declares.
        // Metal retains the buffer and the atlas.
        unsafe {
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.clip_masks.buffer),
                instance_bindings.clip_masks.offset,
                masks_index,
            );
            command_encoder.setFragmentTexture_atIndex(Some(&clip.atlas), atlas_index);
        }
    }

    /// Works out where in the coverage atlas each of this scene's clip paths
    /// goes, sizing the clip textures and building the clip pipelines if the
    /// scene is the first to ask for either.
    fn prepare_clip_masks(
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
    fn render_clip_masks(
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
                // Metal retains the unit vertices and the atlas.
                unsafe {
                    encoder.setVertexBuffer_offset_atIndex(
                        Some(&self.unit_vertices),
                        0,
                        ClipMaskInputIndex::Vertices as usize,
                    );
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
                        MTLPrimitiveType::Triangle,
                        0,
                        6,
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

pub(crate) fn new_command_encoder_for_texture(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    texture: &ProtocolObject<dyn MTLTexture>,
    render_target: RenderTarget,
    clear_color: Option<MTLClearColor>,
) -> Result<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>> {
    let render_pass_descriptor = MTLRenderPassDescriptor::new();
    // SAFETY: Index zero is within the render pass's color attachment array.
    let color_attachment = unsafe {
        render_pass_descriptor
            .colorAttachments()
            .objectAtIndexedSubscript(0)
    };
    color_attachment.setTexture(Some(texture));
    color_attachment.setStoreAction(MTLStoreAction::Store);
    if let Some(clear_color) = clear_color {
        color_attachment.setLoadAction(MTLLoadAction::Clear);
        color_attachment.setClearColor(clear_color);
    } else {
        color_attachment.setLoadAction(MTLLoadAction::Load);
    }

    let command_encoder = command_buffer
        .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
        .context("creating render command encoder")?;
    // A group's target is pooled and so may be larger than the group; the
    // viewport is the part of it the group actually occupies, in its top-left
    // corner, and `render_target.origin` is where that sits in the window.
    command_encoder.setViewport(MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(render_target.size.width) as f64,
        height: i32::from(render_target.size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    crate::fast::rounded_clip::default_mask(&command_encoder);
    Ok(command_encoder)
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
fn read_texture_to_image(texture: &ProtocolObject<dyn MTLTexture>) -> Result<RgbaImage> {
    let width = texture.width() as u32;
    let height = texture.height() as u32;
    let bytes_per_row = width as usize * 4;
    let mut pixels = vec![0u8; height as usize * bytes_per_row];

    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: width as usize,
            height: height as usize,
            depth: 1,
        },
    };
    unsafe {
        texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
            ptr::NonNull::from(pixels.as_mut_slice()).cast(),
            bytes_per_row,
            region,
            0,
        );
    }

    // Convert BGRA to RGBA (swap B and R channels)
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    RgbaImage::from_raw(width, height, pixels).context("creating RGBA image from pixel data")
}

/// A drawing pipeline that exists in both an unclipped and a clip-path
/// specialization, and how it is built.
///
/// The variant is its own index into [`MetalRenderer::primitive_pipelines`] and
/// [`ClipResources::primitive_pipelines`], which is what makes
/// [`MetalRenderer::pipeline_for`] total: a clipped batch cannot fall back to
/// the unclipped pipeline because its clipped twin was never built, which is a
/// picture with the clip path silently missing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrimitivePipeline {
    PathsRasterization = 0,
    Shadows,
    Quads,
    Underlines,
    MonochromeSprites,
    PolychromeSprites,
    Surfaces,
    BgraSurfaces,
}

/// Which blend and attachment shape a pipeline is built with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipelineShape {
    /// gpui's usual straight-alpha source-over.
    StraightAlpha,
    /// Source factor `One`, for a fragment that is already premultiplied.
    Premultiplied,
    /// Multisampled, drawn into the path intermediate.
    Multisampled,
}

/// Every [`PrimitivePipeline`], in the order the enum indexes them: the label
/// Metal's debugger shows, the vertex function, the fragment function, and how
/// it blends. A clipped pipeline is the same three names with `_clipped`
/// appended to the label and the `CLIPPED` function constant set.
const PRIMITIVE_PIPELINES: [(&str, &str, &str, PipelineShape); 8] = [
    (
        "paths_rasterization",
        "path_rasterization_vertex",
        "path_rasterization_fragment",
        PipelineShape::Multisampled,
    ),
    (
        "shadows",
        "shadow_vertex",
        "shadow_fragment",
        PipelineShape::StraightAlpha,
    ),
    (
        "quads",
        "quad_vertex",
        "quad_fragment",
        PipelineShape::StraightAlpha,
    ),
    (
        "underlines",
        "underline_vertex",
        "underline_fragment",
        PipelineShape::StraightAlpha,
    ),
    (
        "monochrome_sprites",
        "monochrome_sprite_vertex",
        "monochrome_sprite_fragment",
        PipelineShape::StraightAlpha,
    ),
    (
        "polychrome_sprites",
        "polychrome_sprite_vertex",
        "polychrome_sprite_fragment",
        PipelineShape::StraightAlpha,
    ),
    (
        "surfaces",
        "surface_vertex",
        "surface_fragment",
        PipelineShape::StraightAlpha,
    ),
    // BGRA surfaces carry premultiplied alpha (they come from GPU renderers
    // like vello, and CoreGraphics can only produce premultiplied BGRA), so
    // they blend with source factor One rather than gpui's usual straight-alpha
    // SourceAlpha factor.
    (
        "bgra_surfaces",
        "surface_vertex",
        "surface_bgra_fragment",
        PipelineShape::Premultiplied,
    ),
];

/// Builds every [`PrimitivePipeline`], either unclipped or clipped.
/// Builds every [`PrimitivePipeline`], either unclipped or clipped.
fn build_primitive_pipelines(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    clipped: bool,
) -> Result<[Retained<ProtocolObject<dyn MTLRenderPipelineState>>; PRIMITIVE_PIPELINES.len()]> {
    let pipelines = PRIMITIVE_PIPELINES
        .iter()
        .map(|&(label, vertex_fn, fragment_fn, shape)| {
            let label = if clipped {
                format!("{label}_clipped")
            } else {
                label.to_string()
            };
            match shape {
                PipelineShape::Multisampled => build_path_rasterization_pipeline_state(
                    device,
                    library,
                    &label,
                    vertex_fn,
                    fragment_fn,
                    MTLPixelFormat::BGRA8Unorm,
                    PATH_SAMPLE_COUNT,
                    clipped,
                ),
                PipelineShape::Premultiplied => build_path_sprite_pipeline_state(
                    device,
                    library,
                    &label,
                    vertex_fn,
                    fragment_fn,
                    MTLPixelFormat::BGRA8Unorm,
                    clipped,
                ),
                PipelineShape::StraightAlpha => build_pipeline_state(
                    device,
                    library,
                    &label,
                    vertex_fn,
                    fragment_fn,
                    MTLPixelFormat::BGRA8Unorm,
                    clipped,
                ),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    pipelines
        .try_into()
        .map_err(|_| anyhow::anyhow!("built a primitive pipeline count other than its kinds'"))
}

/// Looks a function up in the shader library, specialized with `constants`
/// when the function declares any.
fn shader_function(
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
    constants: Option<&MTLFunctionConstantValues>,
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLFunction>>> {
    let ns_name = NSString::from_str(name);
    match constants {
        Some(constants) => library
            .newFunctionWithName_constantValues_error(&ns_name, constants)
            .with_context(|| format!("locating function {name}")),
        None => library
            .newFunctionWithName(&ns_name)
            .with_context(|| format!("locating function {name}")),
    }
}

fn build_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: MTLPixelFormat,
    clipped: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vertex_fn = shader_function(library, vertex_fn_name, Some(&clip_constants(clipped)))?;
    let fragment_fn = shader_function(library, fragment_fn_name, Some(&clip_constants(clipped)))?;

    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(label)));
    descriptor.setVertexFunction(Some(vertex_fn.as_ref()));
    descriptor.setFragmentFunction(Some(fragment_fn.as_ref()));
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(pixel_format);
    color_attachment.setBlendingEnabled(true);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);

    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

fn build_path_sprite_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: MTLPixelFormat,
    clipped: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vertex_fn = shader_function(library, vertex_fn_name, Some(&clip_constants(clipped)))?;
    let fragment_fn = shader_function(library, fragment_fn_name, Some(&clip_constants(clipped)))?;

    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(label)));
    descriptor.setVertexFunction(Some(vertex_fn.as_ref()));
    descriptor.setFragmentFunction(Some(fragment_fn.as_ref()));
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(pixel_format);
    color_attachment.setBlendingEnabled(true);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::One);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);

    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

/// The pipeline that draws one finished group onto what is underneath it.
///
/// A group target holds premultiplied RGBA, so this is
/// [`build_path_sprite_pipeline_state`]'s blend shape rather than
/// [`build_pipeline_state`]'s: the six straight-alpha pipelines emit `S.rgb`
/// with the coverage in `S.a` and cannot express a premultiplied source at all.
/// Which Porter-Duff operator it is, is these factors together with what
/// `group_composite_fragment` emits, and the two are written to agree.
fn build_group_composite_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    compose: GroupComposeMode,
    clipped: bool,
    backdrop: bool,
    pixel_format: MTLPixelFormat,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let constants = group_composite_constants(compose, clipped, backdrop);
    let vertex_fn = shader_function(library, "group_composite_vertex", Some(&constants))?;
    let fragment_fn = shader_function(library, "group_composite_fragment", Some(&constants))?;

    let label = format!("group_composite_{compose:?}");
    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(&label)));
    descriptor.setVertexFunction(Some(vertex_fn.as_ref()));
    descriptor.setFragmentFunction(Some(fragment_fn.as_ref()));
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(pixel_format);
    color_attachment.setBlendingEnabled(true);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    // A backdrop filter replaces what is underneath the group rather than
    // blending with it, so the fragment stage does the whole composite - it was
    // handed a copy of the destination to do it with - and the blend state gets
    // out of the way. Where the group's coverage is zero that arithmetic
    // returns the copy unchanged, which is the same pixel the destination
    // already held.
    let (source, destination) = if backdrop {
        (MTLBlendFactor::One, MTLBlendFactor::Zero)
    } else {
        compose.blend_factors()
    };
    color_attachment.setSourceRGBBlendFactor(source);
    color_attachment.setSourceAlphaBlendFactor(source);
    color_attachment.setDestinationRGBBlendFactor(destination);
    color_attachment.setDestinationAlphaBlendFactor(destination);

    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

/// The pipeline one kind of filter pass draws with.
///
/// No blending at all: a pass writes the whole of its destination and what it
/// writes is the image, not something to mix with what the texture held from
/// the group before it.
fn build_filter_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    kind: FilterPipeline,
    pixel_format: MTLPixelFormat,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vertex_fn = shader_function(library, "filter_pass_vertex", None)?;
    let fragment_fn = shader_function(library, kind.fragment_function(), None)?;

    let label = format!("filter_{kind:?}");
    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(&label)));
    descriptor.setVertexFunction(Some(vertex_fn.as_ref()));
    descriptor.setFragmentFunction(Some(fragment_fn.as_ref()));
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(pixel_format);
    color_attachment.setBlendingEnabled(false);

    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

/// A one-texel texture to bind where a path rasterization pass has no image
/// brush. Nothing samples it - the per-path record says the path has no brush -
/// but the argument it fills is declared unconditionally, so it has to exist.
fn build_default_brush_texture(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>> {
    let descriptor = MTLTextureDescriptor::new();
    // SAFETY: The dimensions are positive and this descriptor creates a 2D texture.
    unsafe {
        descriptor.setWidth(1);
        descriptor.setHeight(1);
    }
    descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    descriptor.setUsage(MTLTextureUsage::ShaderRead);
    descriptor.setStorageMode(MTLStorageMode::Private);
    device
        .newTextureWithDescriptor(&descriptor)
        .context("allocating the default brush texture")
}

#[allow(clippy::too_many_arguments)]
fn build_path_rasterization_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: MTLPixelFormat,
    path_sample_count: u32,
    clipped: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vertex_fn = shader_function(library, vertex_fn_name, Some(&clip_constants(clipped)))?;
    let fragment_fn = shader_function(library, fragment_fn_name, Some(&clip_constants(clipped)))?;

    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(label)));
    descriptor.setVertexFunction(Some(vertex_fn.as_ref()));
    descriptor.setFragmentFunction(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.setRasterSampleCount(path_sample_count as _);
        descriptor.setAlphaToCoverageEnabled(false);
    }
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(pixel_format);
    color_attachment.setBlendingEnabled(true);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::One);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);

    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

#[derive(Clone)]
pub(crate) struct InstanceBinding {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) offset: usize,
}

pub(crate) struct InstanceBindings {
    pub(crate) quads: InstanceBinding,
    pub(crate) shadows: InstanceBinding,
    pub(crate) underlines: InstanceBinding,
    pub(crate) monochrome_sprites: InstanceBinding,
    pub(crate) polychrome_sprites: InstanceBinding,
    pub(crate) surfaces: InstanceBinding,
    /// One [`gpui::ClipMask`] per registered clip, plus the leading `ClipId::NONE`
    /// entry, so a fragment stage can index this by the id it already carries.
    pub(crate) clip_masks: InstanceBinding,
}

pub(crate) fn write_instances(
    scene: &Scene,
    clip_plan: &ClipPlan,
    writer: &mut InstanceBufferWriter,
) -> Result<InstanceBindings> {
    Ok(InstanceBindings {
        quads: writer.write(&scene.quads)?,
        shadows: writer.write(&scene.shadows)?,
        underlines: writer.write(&scene.underlines)?,
        monochrome_sprites: writer.write(&scene.monochrome_sprites)?,
        polychrome_sprites: writer.write(&scene.polychrome_sprites)?,
        surfaces: writer.write_iter(scene.surfaces.iter().map(|surface| SurfaceBounds {
            bounds: surface.bounds,
            content_mask: surface.content_mask,
            clip: surface.clip,
            pad: 0,
        }))?,
        clip_masks: writer.write(&clip_plan.masks)?,
    })
}

pub(crate) struct InstanceBufferWriter {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pool: Arc<Mutex<InstanceBufferPool>>,
    unified_memory: bool,
    filled: Vec<(InstanceBuffer, usize)>,
    current: InstanceBuffer,
    offset: usize,
}

impl InstanceBufferWriter {
    pub(crate) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &Arc<Mutex<InstanceBufferPool>>,
        unified_memory: bool,
    ) -> Result<Self> {
        let current = pool.lock().acquire(device, unified_memory)?;
        Ok(Self {
            device: device.retain(),
            pool: pool.clone(),
            unified_memory,
            filled: Vec::new(),
            current,
            offset: 0,
        })
    }

    fn allocate<T>(&mut self, count: usize) -> Result<(InstanceBinding, &mut [MaybeUninit<T>])> {
        let size = mem::size_of::<T>() * count;
        let mut offset = self.offset.next_multiple_of(INSTANCE_BUFFER_ALIGNMENT);
        if offset + size > self.current.size {
            self.grow(size)?;
            offset = 0;
        }
        self.offset = offset + size;

        let binding = InstanceBinding {
            buffer: self.current.metal_buffer.clone(),
            offset,
        };
        // SAFETY: The aligned reservation lies within this frame's exclusively owned buffer and does not overlap prior writes.
        let values = unsafe {
            let start = self
                .current
                .metal_buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset);
            slice::from_raw_parts_mut(start.cast::<MaybeUninit<T>>(), count)
        };
        Ok((binding, values))
    }

    pub(crate) fn write<T>(&mut self, values: &[T]) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        // SAFETY: The disjoint source and destination hold the same number of T values, and the destination stays MaybeUninit.
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                destination.as_mut_ptr().cast::<T>(),
                values.len(),
            );
        }
        Ok(binding)
    }

    pub(crate) fn write_iter<T>(
        &mut self,
        values: impl ExactSizeIterator<Item = T>,
    ) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        for (slot, value) in destination.iter_mut().zip(values) {
            slot.write(value);
        }
        Ok(binding)
    }

    fn grow(&mut self, required: usize) -> Result<()> {
        let mut pool = self.pool.lock();
        let buffer_size = (pool.buffer_size * 2)
            .max(required.next_power_of_two())
            .min(MAX_INSTANCE_BUFFER_SIZE);
        anyhow::ensure!(
            buffer_size >= required,
            "instance buffer needs {required} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}"
        );
        anyhow::ensure!(
            buffer_size > self.current.size,
            "frame instance data exceeds the {MAX_INSTANCE_BUFFER_SIZE}-byte maximum"
        );
        if buffer_size != pool.buffer_size {
            log::info!("increased instance buffer size to {buffer_size}");
            pool.reset(buffer_size);
        }
        let buffer = pool.acquire(&self.device, self.unified_memory)?;
        drop(pool);

        let filled = mem::replace(&mut self.current, buffer);
        self.filled.push((filled, self.offset));
        self.offset = 0;
        Ok(())
    }

    pub(crate) fn finish(self) -> InstanceBuffer {
        let Self {
            unified_memory,
            filled,
            current,
            offset,
            ..
        } = self;

        if !unified_memory {
            for (buffer, written) in &filled {
                if *written == 0 {
                    continue;
                }
                buffer.metal_buffer.didModifyRange(NSRange {
                    location: 0,
                    length: *written,
                });
            }
            if offset > 0 {
                current.metal_buffer.didModifyRange(NSRange {
                    location: 0,
                    length: offset,
                });
            }
        }

        // Metal retains encoded resources until the command buffer completes.
        // Only the final, largest buffer is worth keeping in the pool.
        drop(filled);
        current
    }
}

// Metal indexes buffers and textures in separate spaces, so a `ClipMasks`
// buffer and a `ClipAtlas` texture would collide as Rust enum discriminants
// without saying anything to the shader. They are given distinct values purely
// to stay a legal Rust enum.
#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    RenderTarget = 2,
    ClipMasks = 3,
    ClipAtlas = 4,
}

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    RenderTarget = 2,
    ClipMasks = 3,
    ClipAtlas = 4,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    RenderTarget = 2,
    ClipMasks = 3,
    ClipAtlas = 4,
}

#[repr(C)]
pub(crate) enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    RenderTarget = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
    ClipMasks = 5,
    ClipAtlas = 6,
}

#[repr(C)]
enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    RenderTarget = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
    ClipMasks = 6,
    ClipAtlas = 7,
}

#[repr(C)]
pub(crate) enum PathRasterizationInputIndex {
    Vertices = 0,
    ViewportSize = 1,
    ContentMasks = 2,
    ClipIds = 3,
    ClipMasks = 4,
    ClipAtlas = 5,
    Brushes = 6,
    BrushCount = 7,
    BrushAtlas = 8,
}

#[repr(C)]
enum GroupCompositeInputIndex {
    Vertices = 0,
    Composite = 1,
    RenderTarget = 2,
    Filter = 3,
    GroupTexture = 4,
    ClipMasks = 5,
    ClipAtlas = 6,
    /// The copy of what was underneath the group, and the filtered copy beside
    /// it. Bound, and declared in the shader, only for the `HAS_BACKDROP`
    /// variant.
    BackdropTexture = 7,
    FilteredBackdropTexture = 8,
}

/// What one filter pass reads. Buffers and textures index separately in Metal,
/// so `Vertices` and `Source` naming the same slot is not a collision.
#[repr(C)]
enum FilterInputIndex {
    Vertices = 0,
    Pass = 1,
    /// The colour matrices, declared by `filter_matrix_fragment` alone.
    Filter = 2,
    Source = 3,
    /// The blurred alpha a drop shadow draws behind its source, declared by
    /// `filter_drop_shadow_fragment` alone.
    Blurred = 4,
}

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

/// Where the attachment a render pass is drawing into sits in the window.
///
/// Every primitive reaches the GPU in scene coordinates - device pixels from
/// the window's top-left - and every fragment stage reasons in them, because
/// that is the space a content mask, a clip mask, a gradient and a signed
/// distance field are all expressed in. A group is drawn into a target of its
/// own that covers only part of the window, so the vertex stage subtracts this
/// origin to place a primitive, and hands the untouched scene position across
/// to the fragment stage as a varying rather than letting it read
/// `[[position]]`, which is the framebuffer's space and not the scene's.
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
    /// The whole window, which is what every pass drew into before groups
    /// existed and what every pass outside a group still draws into.
    pub(crate) fn whole(size: Size<DevicePixels>) -> Self {
        RenderTarget {
            size,
            origin: point(DevicePixels(0), DevicePixels(0)),
        }
    }
}

/// Which Porter-Duff operator a group is composited with: the operators this
/// renderer can express with a single blend state and a premultiplied source.
///
/// `SrcIn`, `SrcOut` and `DestAtop` are the three that cannot: each needs the
/// source scaled by the destination's alpha *and* the destination scaled by
/// something other than the emitted alpha, which takes dual-source blending.
/// A group asking for one of those is composited source-over and says so.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum GroupComposeMode {
    SrcOver = 0,
    DestOut = 1,
    DestIn = 2,
    Clear = 3,
    Xor = 4,
    SrcAtop = 5,
    Plus = 6,
}

impl GroupComposeMode {
    /// The operator a scene's [`ComposeMode`] maps to, or `None` for one this
    /// renderer cannot express yet.
    fn of(compose: ComposeMode) -> Option<Self> {
        match compose {
            ComposeMode::SrcOver => Some(GroupComposeMode::SrcOver),
            ComposeMode::DestOut => Some(GroupComposeMode::DestOut),
            ComposeMode::DestIn => Some(GroupComposeMode::DestIn),
            ComposeMode::Clear => Some(GroupComposeMode::Clear),
            ComposeMode::Xor => Some(GroupComposeMode::Xor),
            ComposeMode::SrcAtop => Some(GroupComposeMode::SrcAtop),
            ComposeMode::Plus | ComposeMode::PlusLighter => Some(GroupComposeMode::Plus),
            ComposeMode::Copy
            | ComposeMode::Dest
            | ComposeMode::DestOver
            | ComposeMode::SrcIn
            | ComposeMode::SrcOut
            | ComposeMode::DestAtop => None,
        }
    }

    /// The source and destination blend factors this operator composites with.
    /// Both channels take the same pair: the source is premultiplied, so its
    /// colour and its alpha combine the same way.
    fn blend_factors(self) -> (MTLBlendFactor, MTLBlendFactor) {
        use MTLBlendFactor as Factor;
        match self {
            GroupComposeMode::SrcOver => (Factor::One, Factor::OneMinusSourceAlpha),
            GroupComposeMode::DestOut => (Factor::Zero, Factor::OneMinusSourceAlpha),
            GroupComposeMode::DestIn => (Factor::Zero, Factor::SourceAlpha),
            GroupComposeMode::Clear => (Factor::Zero, Factor::OneMinusSourceAlpha),
            GroupComposeMode::Xor => (
                Factor::OneMinusDestinationAlpha,
                Factor::OneMinusSourceAlpha,
            ),
            GroupComposeMode::SrcAtop => (Factor::DestinationAlpha, Factor::OneMinusSourceAlpha),
            GroupComposeMode::Plus => (Factor::One, Factor::One),
        }
    }
}

/// The one instance the composite draw is: where a group's target goes, how
/// large it is, and what has to be cut out of it on the way down.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct GroupComposite {
    /// The part of the window the group covers, in device pixels, and so the
    /// quad the composite draws. Whole pixels, so it is texel-aligned with the
    /// target it is sampling.
    pub bounds: Bounds<ScaledPixels>,
    /// The size of the pooled texture, which is at least `bounds`.
    pub texture_size: Size<DevicePixels>,
    /// The clip path in force where the group was pushed. Its coverage is
    /// folded into what the fragment stage emits rather than applied
    /// afterwards, because a blend factor cannot vary per fragment.
    pub clip: ClipId,
    /// The alpha the whole group is composited at.
    pub opacity: f32,
    /// Where the copy of the backdrop's texel (0, 0) sits in the window. Read
    /// only when the composite was specialized with `HAS_BACKDROP`.
    pub backdrop_origin: PointF,
    /// How much of the backdrop textures holds the copy.
    pub backdrop_size: Size<DevicePixels>,
}

/// What outside the source a filter pass reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum FilterEdge {
    /// Nothing: outside the image there is transparent black.
    ///
    /// This is what a `filter` on a group is defined over. The group's own
    /// rendering sits on a transparent canvas, so a blur near its edge has
    /// genuinely nothing to pick up there and has to fade to nothing rather
    /// than smear the last row of ink outwards forever.
    Transparent = 0,
    /// The nearest texel inside the image.
    ///
    /// This is what a `backdrop-filter` is defined over: the backdrop image is
    /// cut out of something larger, so the pixels past its edge are not empty,
    /// they are merely not in the copy. Fading to transparent there would ring
    /// the filtered region with a dark halo that is in nothing behind it, which
    /// is why the filter effects specification calls for edge duplication.
    Clamp = 1,
}

/// The one instance a filter pass draws: a whole-image quad over the
/// destination, with everything the fragment stage needs to find its source.
///
/// Every pass writes the same rectangle it reads - `source_size` texels from
/// the destination's own texel (0, 0) - so nothing in a chain has to track a
/// moving origin. A blur's growth is spent before the renderer ever sees the
/// group: [`gpui::Scene::grow_group_to_its_content`] has already widened the
/// target by the gaussian's tail, so the tail lands inside the rectangle
/// rather than needing one of its own.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct FilterPass {
    /// How much of the source texture holds the image, from its texel (0, 0).
    pub source_size: Size<DevicePixels>,
    /// Which way this blur runs: (1, 0) or (0, 1). A separable gaussian is two
    /// of these, and the two together cost `2 * (2n + 1)` texture reads a texel
    /// where one square pass would cost `(2n + 1)^2` - at the sigma an email
    /// asks for, tens of reads against thousands.
    pub direction: PointF,
    /// Where a drop shadow reads its blurred alpha from, relative to the texel
    /// being written: the shadow's offset, which is subtracted.
    pub offset: PointF,
    /// A drop shadow's colour.
    pub color: Hsla,
    /// The gaussian's standard deviation, in device pixels.
    pub sigma: f32,
    /// How many texels the pass reads on each side of the one it writes.
    pub taps: i32,
    /// How far apart those reads are. One, unless the tail is longer than
    /// [`MAX_BLUR_TAPS`] reads could cover.
    pub stride: i32,
    /// A [`FilterEdge`].
    pub edge: u32,
}

impl FilterPass {
    fn new(source_size: Size<DevicePixels>, edge: FilterEdge) -> Self {
        FilterPass {
            source_size,
            direction: point(0., 0.),
            offset: point(0., 0.),
            color: Hsla::default(),
            sigma: 0.,
            taps: 0,
            stride: 1,
            edge: edge as u32,
        }
    }
}

/// The attachment a render pass is drawing into, and where it sits in the
/// window.
#[derive(Clone)]
struct ActiveTarget {
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
    render_target: RenderTarget,
    /// Whether this is the frame's own target rather than a group's.
    ///
    /// A backdrop filter has to copy out of whatever it is composited over, and
    /// the frame's target is the one texture that may refuse: a window's
    /// drawable is framebuffer-only until something asks for otherwise. A
    /// group's target is a texture this file made and can always be read.
    is_window: bool,
}

/// Everything compositing one group needs beyond the group itself.
struct PreparedGroup {
    /// The image the composite samples for the group: its own target, or
    /// whatever the last filter pass wrote.
    source: FilterImage,
    /// The trailing run of colour matrices, which the composite applies itself
    /// rather than spending a pass on.
    filter: GroupFilter,
    backdrop: Option<PreparedBackdrop>,
}

/// The two halves of a backdrop filter: what was underneath the group, and what
/// the filter made of it.
struct PreparedBackdrop {
    original: FilterImage,
    filtered: FilterImage,
}

/// The three shapes a filter pass takes, and so the three pipelines it needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum FilterPipeline {
    Blur,
    Matrices,
    DropShadow,
}

impl FilterPipeline {
    fn fragment_function(self) -> &'static str {
        match self {
            FilterPipeline::Blur => "filter_blur_fragment",
            FilterPipeline::Matrices => "filter_matrix_fragment",
            FilterPipeline::DropShadow => "filter_drop_shadow_fragment",
        }
    }
}

/// A group whose target is being drawn into, and everything compositing it
/// back down needs.
struct OpenGroup {
    spec: GroupSpec,
    /// The whole device pixels the group covers: the composite's quad, and the
    /// region of the target that holds it.
    bounds: Bounds<ScaledPixels>,
    /// The pooled texture's own size, which the composite needs to turn a
    /// position in `bounds` into a texture coordinate.
    texture_size: Size<DevicePixels>,
    target: ActiveTarget,
    /// What the group is composited back onto.
    parent: ActiveTarget,
}

/// One image a filter chain reads or writes: a texture, how much of it holds
/// the image, and where that sits in the window.
#[derive(Clone)]
struct FilterImage {
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
    /// The pooled texture's own size, which is at least `rect`'s.
    texture_size: Size<DevicePixels>,
    /// The part of the window the image covers. Its top-left corner is the
    /// texture's texel (0, 0), so every pass can be a whole-image quad drawn
    /// through the viewport a [`RenderTarget`] already sets.
    rect: DeviceRect,
    /// The scratch slot to hand back once nothing reads this any more, or
    /// `None` for a texture the scratch pool does not own - a group's own
    /// target, which is the first image of every chain.
    slot: Option<usize>,
}

impl FilterImage {
    fn size(&self) -> Size<DevicePixels> {
        size(
            DevicePixels(self.rect.width),
            DevicePixels(self.rect.height),
        )
    }

    fn render_target(&self) -> RenderTarget {
        RenderTarget {
            size: self.size(),
            origin: point(DevicePixels(self.rect.x), DevicePixels(self.rect.y)),
        }
    }
}

/// The working textures a filter chain runs through.
///
/// Not one per nesting depth, the way [`GroupTargets`] is: a chain runs
/// entirely inside one `PopGroup`, with no other group's chain in flight, so
/// every depth shares the same few slots and gives them straight back. Nothing
/// is allocated at all until a scene carries a blur, a drop shadow or a
/// backdrop filter.
#[derive(Default)]
struct FilterScratch {
    targets: GroupTargets,
    in_use: [bool; MAX_FILTER_SCRATCH],
}

impl FilterScratch {
    fn begin_frame(&mut self) {
        self.targets.begin_frame();
        self.in_use = [false; MAX_FILTER_SCRATCH];
    }

    fn end_frame(&mut self) {
        self.targets.end_frame();
    }

    /// A working texture covering `rect`, or `None` when every slot is taken or
    /// giving one out would break the budget.
    fn acquire(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        rect: DeviceRect,
    ) -> Result<Option<FilterImage>> {
        let Some(slot) = self.in_use.iter().position(|used| !used) else {
            return Ok(None);
        };
        let Some(texture) = self.targets.acquire(
            device,
            slot,
            rect,
            FILTER_SCRATCH_BUDGET_BYTES,
            "the filter it was for is dropped and the group composited unfiltered",
        )?
        else {
            return Ok(None);
        };
        self.in_use[slot] = true;
        Ok(Some(FilterImage {
            texture_size: size(
                DevicePixels(texture.width() as i32),
                DevicePixels(texture.height() as i32),
            ),
            texture,
            rect,
            slot: Some(slot),
        }))
    }

    /// Give an image's slot back, if it holds one. A group's own target holds
    /// none and is left alone.
    fn release(&mut self, image: &FilterImage) {
        if let Some(slot) = image.slot {
            self.in_use[slot] = false;
        }
    }
}

/// The render targets isolated groups are drawn into: one per nesting depth,
/// reused by every group at that depth.
///
/// A target is sized to its group's own bounds rather than to the window,
/// because a full-viewport clear per group would cost around 24 MB of bandwidth
/// each on a retina display and groups nest. Sides are quantized so a group
/// whose bounds move a pixel a frame is the same allocation, growing is
/// immediate and shrinking waits, all exactly as the clip textures do.
///
/// Siblings share a depth's texture safely: a group is composited at its
/// `PopGroup`, before the next group at that depth clears the texture again.
#[derive(Default)]
struct GroupTargets {
    levels: Vec<GroupTargetLevel>,
    /// Whether this frame has already said it could not isolate a group. One
    /// line per frame, not one per group: whatever makes one group fall back
    /// usually makes every group in the document fall back.
    reported: bool,
}

#[derive(Default)]
struct GroupTargetLevel {
    texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    /// The size of `texture`, or zero when there is none.
    allocated: Extent,
    /// What this frame started from, once the shrink countdown has had its say.
    floor: Extent,
    /// The largest this depth has been asked for this frame.
    needed: Extent,
    budget: TextureBudget<GROUP_TEXTURE_SHRINK_FRAMES>,
}

impl GroupTargets {
    fn begin_frame(&mut self) {
        self.reported = false;
        for level in &mut self.levels {
            let floor = level.budget.floor();
            if floor.width < level.allocated.width || floor.height < level.allocated.height {
                level.texture = None;
                level.allocated = Extent::ZERO;
            }
            level.floor = floor;
            level.needed = Extent::ZERO;
        }
    }

    fn end_frame(&mut self) {
        for level in &mut self.levels {
            level.budget.observe(level.allocated, level.needed);
        }
    }

    /// A target at least `rect` large for a group at `depth`, or `None` when
    /// giving it one would put the group targets over
    /// [`GROUP_TARGET_BUDGET_BYTES`].
    fn acquire(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        depth: usize,
        rect: DeviceRect,
        budget: usize,
        on_refusal: &str,
    ) -> Result<Option<Retained<ProtocolObject<dyn MTLTexture>>>> {
        while self.levels.len() <= depth {
            self.levels.push(GroupTargetLevel::default());
        }
        let required = rect.extent().quantized_to(GROUP_TEXTURE_QUANTUM);
        let wanted = self.levels[depth]
            .floor
            .max(required)
            .max(self.levels[depth].allocated);
        let elsewhere: usize = self
            .levels
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != depth)
            .map(|(_, level)| level.allocated.bytes())
            .sum();

        let level = &mut self.levels[depth];
        level.needed = level.needed.max(required);
        if level.allocated != wanted || level.texture.is_none() {
            if elsewhere + wanted.bytes() > budget {
                self.report(format_args!(
                    "a {}x{} render target would take these ones past their \
                     {budget}-byte budget; {on_refusal}",
                    wanted.width, wanted.height
                ));
                return Ok(None);
            }
            let descriptor = MTLTextureDescriptor::new();
            // SAFETY: A quantized extent is positive and this descriptor creates a 2D texture.
            unsafe {
                descriptor.setWidth(wanted.width as usize);
                descriptor.setHeight(wanted.height as usize);
            }
            descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            descriptor.setStorageMode(MTLStorageMode::Private);
            descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
            level.texture = Some(
                device
                    .newTextureWithDescriptor(&descriptor)
                    .context("allocating a group render target")?,
            );
            level.allocated = wanted;
        }
        Ok(level.texture.clone())
    }

    /// Say once a frame that a group could not be isolated, and why.
    fn report(&mut self, reason: std::fmt::Arguments<'_>) {
        if self.reported {
            return;
        }
        self.reported = true;
        log::error!("{reason}");
    }
}

/// Whether any group in a scene asks for a backdrop filter, which is what
/// decides whether the window's drawables have to become readable.
fn scene_has_a_backdrop_filter(scene: &Scene) -> bool {
    scene
        .groups
        .iter()
        .any(|group| group.backdrop_filter.is_some())
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    /// A surface is drawn one instance at a time out of its own record, so
    /// unlike every other primitive its clip id rides here rather than in the
    /// scene struct the shaders read.
    pub clip: ClipId,
    pub pad: u32,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
pub struct MetalHeadlessRenderer {
    renderer: MetalRenderer,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl MetalHeadlessRenderer {
    pub fn new() -> Result<Self> {
        Self::with_transparency(false)
    }

    /// A headless renderer whose target is cleared to transparent rather than
    /// opaque black when `transparent` is set, so a test can read the scene's
    /// own alpha back.
    pub fn with_transparency(transparent: bool) -> Result<Self> {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool, transparent)?;
        Ok(Self { renderer })
    }
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl gpui::PlatformHeadlessRenderer for MetalHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.renderer.render_scene_to_image(scene, size)
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.renderer.render_scene(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_core_foundation::{CFDictionary, CFType};
    use objc2_core_video::{
        CVPixelBufferCreate, CVPixelBufferGetBaseAddressOfPlane,
        CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
        CVPixelBufferUnlockBaseAddress, kCVPixelBufferIOSurfacePropertiesKey,
    };
    use objc2_metal::MTLCommandBufferStatus;

    #[test]
    fn path_intermediate_textures_are_reused_until_resize() -> Result<()> {
        let mut renderer = MetalRenderer::new_headless(Default::default(), false)?;
        let initial_size = size(DevicePixels(64), DevicePixels(64));
        renderer.update_path_intermediate_textures(initial_size)?;
        let initial_texture = renderer
            .path_intermediate_texture
            .clone()
            .context("missing path intermediate texture")?;
        let initial_msaa_texture = renderer.path_intermediate_msaa_texture.clone();

        renderer.update_path_intermediate_textures(initial_size)?;
        assert!(ptr::eq(
            &*initial_texture,
            renderer
                .path_intermediate_texture
                .as_deref()
                .context("missing reused texture")?,
        ));
        assert_eq!(
            initial_msaa_texture.as_ref().map(Retained::as_ptr),
            renderer
                .path_intermediate_msaa_texture
                .as_ref()
                .map(Retained::as_ptr),
        );

        renderer.update_path_intermediate_textures(size(DevicePixels(128), DevicePixels(64)))?;
        let resized_texture = renderer
            .path_intermediate_texture
            .as_deref()
            .context("missing resized texture")?;
        assert_eq!(
            (resized_texture.width(), resized_texture.height()),
            (128, 64)
        );
        assert!(!ptr::eq(&*initial_texture, resized_texture));
        if let Some(texture) = &renderer.path_intermediate_msaa_texture {
            assert_eq!((texture.width(), texture.height()), (128, 64));
        }

        for (width, height) in [(0, 64), (64, 0), (-1, 64), (64, -1)] {
            renderer.update_path_intermediate_textures(size(
                DevicePixels(width),
                DevicePixels(height),
            ))?;
            assert!(renderer.path_intermediate_texture.is_none());
            assert!(renderer.path_intermediate_msaa_texture.is_none());
            renderer.update_path_intermediate_textures(initial_size)?;
            assert!(renderer.path_intermediate_texture.is_some());
        }
        Ok(())
    }

    #[test]
    fn render_to_image_rejects_zero_sized_layer() -> Result<()> {
        let mut renderer = MetalRenderer::new(Default::default(), false, false)?;
        for (width, height) in [(0, 0), (0, 64), (64, 0)] {
            renderer.update_drawable_size(size(DevicePixels(width), DevicePixels(height)));
            assert!(renderer.render_to_image(&Scene::default()).is_err());
        }
        Ok(())
    }

    #[test]
    fn display_link_mode_consumes_and_clears_drawables() -> Result<()> {
        let mut renderer = MetalRenderer::new(Default::default(), false, false)?;
        renderer.update_drawable_size(size(DevicePixels(64), DevicePixels(64)));
        let drawable = renderer
            .layer()
            .and_then(|layer| layer.nextDrawable())
            .context("window renderer must supply a drawable")?;

        renderer.set_display_link_attached(true);
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Attached { drawable: None }
        ));
        assert!(
            renderer
                .set_frame_drawable(Some(drawable.clone()))
                .is_none()
        );
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Attached { drawable: Some(_) }
        ));
        renderer.draw(&Scene::default());
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Attached { drawable: None }
        ));
        // A frame that drew consumed its drawable; one that didn't hands it back.
        assert!(
            renderer
                .set_frame_drawable(Some(drawable.clone()))
                .is_none()
        );
        assert!(renderer.set_frame_drawable(None).is_some());

        renderer.set_frame_drawable(Some(drawable));
        renderer.set_display_link_attached(false);
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Detached
        ));
        Ok(())
    }

    #[test]
    fn render_core_video_surface() {
        let surface_properties_key = unsafe { kCVPixelBufferIOSurfacePropertiesKey };
        let attributes = CFDictionary::from_slices(
            &[surface_properties_key],
            &[&*CFDictionary::<CFType, CFType>::empty()],
        );
        let mut image_buffer = ptr::null_mut();
        assert_eq!(
            unsafe {
                CVPixelBufferCreate(
                    None,
                    16,
                    16,
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                    Some(attributes.as_opaque()),
                    ptr::NonNull::from(&mut image_buffer),
                )
            },
            kCVReturnSuccess
        );
        let image_buffer = ptr::NonNull::new(image_buffer).unwrap();
        let image_buffer = unsafe { CFRetained::from_raw(image_buffer) };
        unsafe {
            assert_eq!(
                CVPixelBufferLockBaseAddress(&image_buffer, CVPixelBufferLockFlags::empty()),
                kCVReturnSuccess
            );
            for plane in 0..2 {
                let bytes = CVPixelBufferGetBaseAddressOfPlane(&image_buffer, plane).cast::<u8>();
                let length = CVPixelBufferGetBytesPerRowOfPlane(&image_buffer, plane)
                    * CVPixelBufferGetHeightOfPlane(&image_buffer, plane);
                slice::from_raw_parts_mut(bytes, length).fill(128);
            }
            assert_eq!(
                CVPixelBufferUnlockBaseAddress(&image_buffer, CVPixelBufferLockFlags::empty()),
                kCVReturnSuccess
            );
        }

        let bounds = Bounds {
            origin: Point::default(),
            size: size(ScaledPixels(16.0), ScaledPixels(16.0)),
        };
        let mut scene = Scene::default();
        scene.insert_primitive(PaintSurface {
            order: 0,
            clip: ClipId::NONE,
            bounds,
            content_mask: ContentMask::new(bounds),
            image_buffer: image_buffer.clone(),
        });
        drop(image_buffer);
        scene.finish();

        let mut renderer =
            MetalRenderer::new_headless(Arc::new(Mutex::new(InstanceBufferPool::default())), false)
                .unwrap();
        let image = renderer
            .render_scene_to_image(&scene, size(DevicePixels(16), DevicePixels(16)))
            .unwrap();
        for image::Rgba([red, green, blue, alpha]) in image.pixels() {
            assert!(red.abs_diff(128) <= 2);
            assert!(green.abs_diff(128) <= 2);
            assert!(blue.abs_diff(128) <= 2);
            assert_eq!(*alpha, 255);
        }

        let descriptor = MTLTextureDescriptor::new();
        unsafe {
            descriptor.setWidth(16);
            descriptor.setHeight(16);
        }
        descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        descriptor.setUsage(MTLTextureUsage::RenderTarget);
        let uses_shared_storage = cfg!(target_os = "ios") || renderer.is_apple_gpu;
        descriptor.setStorageMode(if uses_shared_storage {
            MTLStorageMode::Shared
        } else {
            MTLStorageMode::Managed
        });
        let target_texture = renderer
            .device
            .newTextureWithDescriptor(&descriptor)
            .unwrap();
        let command_buffer = renderer
            .render_frame(
                &scene,
                &target_texture,
                size(DevicePixels(16), DevicePixels(16)),
            )
            .unwrap();
        if !uses_shared_storage {
            let blit = command_buffer.blitCommandEncoder().unwrap();
            blit.synchronizeResource(ProtocolObject::from_ref(&*target_texture));
            blit.endEncoding();
        }
        drop(scene);
        drop(renderer);
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(read_texture_to_image(&target_texture).unwrap(), image);
    }
}

/// The pipelines, stencil states and textures an arbitrary-path clip needs.
///
/// Built the first time a scene carries a clip path and kept afterwards. Doing
/// it lazily is what lets the clipped variant of every pipeline exist at all:
/// building them up front would double this renderer's pipeline compilation at
/// startup for a feature nothing in gpui's own UI uses.
///
/// The bill for that is fourteen pipeline states compiled on the render thread
/// during the first frame that clips anything - two mask pipelines and a
/// clipped variant of each of the twelve drawing ones - which on a cold shader
/// cache is a hitch the user sees once. Moving it off the critical path means
/// building them on another thread and handing them over, which is a change to
/// how this renderer owns its state rather than a change to this function.
struct ClipResources {
    stencil_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    cover_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    nonzero_stencil_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    even_odd_stencil_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    nonzero_cover_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    even_odd_cover_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    primitive_pipelines:
        [Retained<ProtocolObject<dyn MTLRenderPipelineState>>; PRIMITIVE_PIPELINES.len()],
    /// The coverage the clipped fragment shaders sample, one tile per clip.
    /// Never cleared: a tile is either written whole by this frame or never
    /// read, because a clip that has no tile says so in its `ClipMask`.
    atlas: Retained<ProtocolObject<dyn MTLTexture>>,
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
            primitive_pipelines: build_primitive_pipelines(device, library, true)?,
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
                descriptor.setSampleCount(PATH_SAMPLE_COUNT as usize);
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

/// The specialization the group composite pipelines take: which operator, and
/// whether the group carried a clip path the composite has to be cut by.
fn group_composite_constants(
    compose: GroupComposeMode,
    clipped: bool,
    backdrop: bool,
) -> Retained<MTLFunctionConstantValues> {
    let constants = clip_constants(clipped);
    let mode = compose as u32;
    // SAFETY: Each value is read for the duration of the call, with the type
    // and index the shaders declare for `COMPOSE_MODE` and `HAS_BACKDROP`.
    unsafe {
        constants.setConstantValue_type_atIndex(
            ptr::NonNull::from(&mode).cast(),
            MTLDataType::UInt,
            1,
        );
        constants.setConstantValue_type_atIndex(
            ptr::NonNull::from(&backdrop).cast(),
            MTLDataType::Bool,
            2,
        );
    }
    constants
}

/// The specialization that picks between the clipped and unclipped variant of a
/// shader. Every function that mentions `CLIPPED` must be given one, and one
/// given to a function that does not is ignored.
fn clip_constants(clipped: bool) -> Retained<MTLFunctionConstantValues> {
    let constants = MTLFunctionConstantValues::new();
    // SAFETY: The value is read for the duration of the call, with the type
    // and index the shaders declare for `CLIPPED`.
    unsafe {
        constants.setConstantValue_type_atIndex(
            ptr::NonNull::from(&clipped).cast(),
            MTLDataType::Bool,
            0,
        );
    }
    constants
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
    descriptor.setRasterSampleCount(PATH_SAMPLE_COUNT as usize);
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
