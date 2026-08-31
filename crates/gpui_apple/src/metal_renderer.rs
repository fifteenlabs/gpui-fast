use crate::metal_atlas::MetalAtlas;
use anyhow::{Context as _, Result};
use block2::RcBlock;
use core_graphics::geometry::CGSize;
use gpui::{
    AtlasTextureId, AtlasTextureKind, AtlasTile, Background, Bounds, BrushExtend, ClipId, ClipPath,
    ClipPathSegment, ComposeMode, ContentMask, DevicePixels, FillRule, GroupSpec, MixMode,
    PaintSurface, Path, PathBrush, Point, PrimitiveBatch, ScaledPixels, Scene, SceneFilter,
    SceneStep, Size, TileId, TransformationMatrix, point, size,
};
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
use image::RgbaImage;
use objc2::runtime::AnyObject;

use core_graphics::color_space::{CGColorSpace, kCGColorSpaceDisplayP3};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    CAMetalLayer, CommandQueue, MTLGPUFamily, MTLPixelFormat, MTLResourceOptions, NSRange,
    NSUInteger,
};
use objc2_core_foundation::CFRetained;
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBufferGetHeight,
    CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
    CVPixelBufferGetWidthOfPlane, kCVPixelFormatType_32BGRA,
    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVReturnSuccess,
};
use parking_lot::Mutex;

use std::{
    cell::Cell, collections::HashMap, ffi::c_void, mem, mem::MaybeUninit, ops::Range, ptr, slice,
    sync::Arc,
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

/// What the clip textures' sides are rounded up to.
///
/// Quantizing them is what keeps a document whose clips move a pixel a frame
/// from recreating textures every frame, and 256 is small enough that the
/// rounding itself wastes little.
const CLIP_TEXTURE_QUANTUM: i32 = 256;
/// The largest either side of the coverage atlas grows to in order to give
/// tiles that would otherwise have to share a shelf more room.
///
/// What the clip machinery keeps resident is the atlas - one R8 texture of this
/// size - plus the working attachments a nesting level is rendered through,
/// which are sized to the largest level rather than to the atlas: one more R8
/// resolve target, and a 4x multisample colour and stencil pair. On Apple
/// silicon the multisample pair is memoryless and costs nothing, so an atlas at
/// this cap whose levels fill it is 16.8 MB + 16.8 MB = 33.6 MB; on an Intel
/// Mac the multisample pair is real memory and the same frame is 167.8 MB.
///
/// Those numbers are why a tile is sized from the part of the window its clip
/// can actually reach rather than from the shape's own bounding box, why the
/// working attachments follow the level rather than the atlas, and why the
/// whole lot is handed back after [`CLIP_TEXTURE_SHRINK_FRAMES`] frames of not
/// being needed.
const CLIP_ATLAS_MAX_SIZE: i32 = 4096;
/// How large a single tile may force the atlas to be, whatever the packing cap
/// says.
///
/// A tile is a clip's on-screen extent, so on a 5K display a full-width element
/// is 5120 device pixels across and cannot be packed under
/// [`CLIP_ATLAS_MAX_SIZE`] at all. Refusing it there would silently drop the
/// shape of every full-width clip on mainstream hardware, so a tile that needs
/// more than the packing cap gets it, up to the largest 2D texture Metal will
/// allocate.
const CLIP_ATLAS_TEXTURE_LIMIT: i32 = 16384;
/// How many consecutive frames have to fit inside smaller clip textures before
/// the renderer gives the memory back.
///
/// Growing is immediate, because a frame that cannot fit its tiles draws the
/// wrong picture. Shrinking waits, because a list scrolling a clipped element
/// in and out of view would otherwise recreate four textures every few frames.
const CLIP_TEXTURE_SHRINK_FRAMES: u32 = 240;
/// How deep clip paths may nest before the innermost ones fall back to their
/// rectangles. Each level is a render pass of its own, and nothing in a
/// document goes anywhere near this.
const MAX_CLIP_DEPTH: usize = 8;
/// How far a flattened clip contour may stray from the curve it came from, in
/// device pixels. Well under what 4x multisampling can resolve.
const CLIP_FLATTEN_TOLERANCE: f32 = 0.1;
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
/// [`CLIP_TEXTURE_QUANTUM`] exists: a group whose bounds move a pixel a frame
/// is then the same allocation rather than a new texture every frame.
const GROUP_TEXTURE_QUANTUM: i32 = 256;
/// How many consecutive frames have to fit inside smaller group targets before
/// the renderer gives the memory back. Growing is immediate; shrinking waits,
/// exactly as [`CLIP_TEXTURE_SHRINK_FRAMES`] does.
const GROUP_TEXTURE_SHRINK_FRAMES: u32 = 240;
/// The most memory the group targets may hold between them.
///
/// A group is sized to its own bounds, so this is only reached by groups that
/// each cover most of the window at several nesting levels at once - on a
/// retina display one full-screen target is already around 28 MB. Past it a
/// group is painted without isolation and says so, which is a wrong picture,
/// but a loud one and a bounded amount of memory.
const GROUP_TARGET_BUDGET_BYTES: usize = 128 * 1024 * 1024;
/// How long a chain of colour matrices one group may carry. CSS `filter` lists
/// are short, and the whole chain travels to the GPU as one small constant.
const MAX_GROUP_COLOR_MATRICES: usize = 8;
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

pub unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: gpui::Size<f32>,
    transparent: bool,
    use_display_p3: bool,
) -> Renderer {
    MetalRenderer::new(context, transparent, use_display_p3)
}

pub struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
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
    metal_buffer: metal::Buffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(
        &mut self,
        device: &metal::Device,
        unified_memory: bool,
    ) -> InstanceBuffer {
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            let options = if unified_memory {
                MTLResourceOptions::StorageModeShared
                    // Buffers are write only which can benefit from the combined cache
                    // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            };

            device.new_buffer(self.buffer_size as u64, options)
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

pub struct MetalRenderer {
    device: metal::Device,
    /// Kept so the clipped pipeline variants can be specialized out of it the
    /// first time a scene actually carries a clip path. Building them eagerly
    /// would double this renderer's pipeline count at startup for a feature
    /// nothing in gpui's own UI uses.
    library: metal::Library,
    clip: Option<ClipResources>,
    clip_atlas_budget: ClipTextureBudget,
    clip_work_budget: ClipTextureBudget,
    group_targets: GroupTargets,
    /// Built the first time a scene actually composites a group, and only for
    /// the operators it uses: there are seven of them times a clipped and an
    /// unclipped variant, and gpui's own UI asks for none.
    group_composite_pipelines: HashMap<(GroupComposeMode, bool), metal::RenderPipelineState>,
    layer: Option<metal::MetalLayer>,
    is_apple_gpu: bool,
    is_unified_memory: bool,
    presents_with_transaction: bool,
    /// For headless rendering, tracks whether output should be opaque
    opaque: bool,
    command_queue: CommandQueue,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    bgra_surfaces_pipeline_state: metal::RenderPipelineState,
    unit_vertices: metal::Buffer,
    /// Bound wherever a path rasterization pass has no image brush to sample: a
    /// fragment shader that declares a `texture2d` argument needs something
    /// bound even in the frames that never read it.
    default_brush_texture: metal::Texture,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: CFRetained<CVMetalTextureCache>,
    path_intermediate_texture: Option<metal::Texture>,
    path_intermediate_msaa_texture: Option<metal::Texture>,
    path_sample_count: u32,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    headless_render_target: Option<metal::Texture>,
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
    ) -> Self {
        let device = Self::create_device();
        let layer = metal::MetalLayer::new();
        Self::configure_layer(&layer, &device, transparent, use_display_p3);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a renderer for a CAMetalLayer owned by a platform view, such as
    /// the backing layer UIKit creates for a view whose `layerClass` is
    /// `CAMetalLayer`. The renderer retains the layer.
    pub fn from_layer(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        layer: &objc2_quartz_core::CAMetalLayer,
        transparent: bool,
    ) -> Self {
        let device = Self::create_device();
        // Both types bind the same Objective-C class, so this only changes which
        // Rust wrapper views the live layer. `to_owned` retains it.
        let layer = unsafe {
            metal::MetalLayerRef::from_ptr(ptr::from_ref(layer).cast_mut().cast::<CAMetalLayer>())
        }
        .to_owned();
        Self::configure_layer(&layer, &device, transparent, false);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    fn configure_layer(
        layer: &metal::MetalLayerRef,
        device: &metal::DeviceRef,
        transparent: bool,
        use_display_p3: bool,
    ) {
        layer.set_device(device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // metal-rs doesn't bind these setters, so view the same object through
        // objc2's typed CAMetalLayer binding.
        let objc2_layer: &objc2_quartz_core::CAMetalLayer = unsafe { &*layer.as_ptr().cast() };
        if use_display_p3 {
            // SAFETY: The constant is extern, from Core Graphics
            let display_p3_name = unsafe { kCGColorSpaceDisplayP3 };
            if let Some(display_p3_color_space) = CGColorSpace::create_with_name(display_p3_name) {
                // SAFETY: `layer` is a CAMetalLayer and the color space remains valid for the call.
                // 'metal-rs' does not expose CAMetalLayer's typed `colorspace` setter.
                // Could be replaced with a native binding via objc2-* in future
                unsafe {
                    let _: () = objc2::msg_send![
                        objc2_layer,
                        setColorspace: display_p3_color_space.as_ptr().cast::<c_void>()
                    ];
                }
            } else {
                log::error!("unable to create the Display P3 color space");
            }
        }
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.set_opaque(!transparent);
        layer.set_maximum_drawable_count(3);
        // Allow texture reading for visual tests and UI automation screenshots
        // (captures without ScreenCaptureKit). Only debug builds pay the
        // presentation cost, even when `test-support` is compiled in.
        #[cfg(all(feature = "test-support", debug_assertions))]
        layer.set_framebuffer_only(false);
        objc2_layer.setAllowsNextDrawableTimeout(false);
        objc2_layer.setNeedsDisplayOnBoundsChange(true);
        // UIKit sizes a view's backing layer itself; only AppKit-hosted
        // layers need to track their superlayer's bounds.
        #[cfg(target_os = "macos")]
        objc2_layer.setAutoresizingMask(
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
    ) -> Self {
        let device = Self::create_device();
        Self::new_internal(device, None, !transparent, instance_buffer_pool)
    }

    #[cfg(target_os = "macos")]
    fn create_device() -> metal::Device {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        if let Some(d) = metal::Device::all()
            .into_iter()
            .min_by_key(|d| (d.is_removable(), !d.is_low_power()))
        {
            d
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
    fn create_device() -> metal::Device {
        Self::system_default_device()
    }

    fn system_default_device() -> metal::Device {
        metal::Device::system_default().unwrap_or_else(|| {
            log::error!("unable to access a compatible graphics device");
            std::process::exit(1);
        })
    }

    fn new_internal(
        device: metal::Device,
        layer: Option<metal::MetalLayer>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Self {
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .new_library_with_source(&SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
            .expect("error building metal library");
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .new_library_with_data(SHADERS_METALLIB)
            .expect("error building metal library");

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
        let is_unified_memory = cfg!(target_os = "ios") || device.has_unified_memory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supports_family(MTLGPUFamily::Apple1);

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices) as u64,
            if is_unified_memory {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            },
        );

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
            false,
        );
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );
        // BGRA surfaces carry premultiplied alpha (they come from GPU
        // renderers like vello, and CoreGraphics can only produce
        // premultiplied BGRA), so blend with source factor One — the
        // path-sprite builder's blend config — rather than gpui's usual
        // straight-alpha SourceAlpha factor.
        let bgra_surfaces_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "bgra_surfaces",
            "surface_vertex",
            "surface_bgra_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );

        let default_brush_texture = build_default_brush_texture(&device);
        let command_queue = device.new_command_queue();
        let supports_shared_storage = cfg!(target_os = "ios") || is_apple_gpu;
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), supports_shared_storage));
        // metal and objc2-metal expose the same Objective-C device; the borrowed
        // protocol object remains valid while `device` owns it.
        let core_video_texture_cache = unsafe {
            let mut cache = ptr::null_mut();
            let status = CVMetalTextureCache::create(
                None,
                None,
                &*device
                    .as_ptr()
                    .cast::<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>(),
                None,
                ptr::NonNull::from(&mut cache),
            );
            assert_eq!(
                status, kCVReturnSuccess,
                "failed to create CoreVideo Metal texture cache"
            );
            CFRetained::from_raw(ptr::NonNull::new(cache).expect("CoreVideo returned a null cache"))
        };

        Self {
            device,
            library,
            clip: None,
            clip_atlas_budget: ClipTextureBudget::default(),
            clip_work_budget: ClipTextureBudget::default(),
            group_targets: GroupTargets::default(),
            group_composite_pipelines: HashMap::new(),
            layer,
            presents_with_transaction: false,
            is_apple_gpu,
            is_unified_memory,
            opaque,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            bgra_surfaces_pipeline_state,
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
        }
    }

    pub fn layer(&self) -> Option<&metal::MetalLayerRef> {
        self.layer.as_ref().map(|l| l.as_ref())
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|l| l.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.set_presents_with_transaction(presents_with_transaction);
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            layer.set_drawable_size(CGSize::new(size.width.0 as f64, size.height.0 as f64));
        }
        self.update_path_intermediate_textures(size);
    }

    fn update_path_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            return;
        }

        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        self.path_intermediate_texture = Some(self.device.new_texture(&texture_descriptor));

        if self.path_sample_count > 1 {
            // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
            // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
            let storage_mode = if self.is_apple_gpu {
                metal::MTLStorageMode::Memoryless
            } else {
                metal::MTLStorageMode::Private
            };

            let msaa_descriptor = texture_descriptor;
            msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
            msaa_descriptor.set_storage_mode(storage_mode);
            msaa_descriptor.set_sample_count(self.path_sample_count as _);
            self.path_intermediate_msaa_texture = Some(self.device.new_texture(&msaa_descriptor));
        } else {
            self.path_intermediate_msaa_texture = None;
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.set_opaque(!transparent);
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
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return;
        };

        let command_buffer = match self.render_frame(scene, drawable.texture(), viewport_size) {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                log::error!("failed to render: {error:#}");
                return;
            }
        };

        if self.presents_with_transaction {
            command_buffer.commit();
            command_buffer.wait_until_scheduled();
            drawable.present();
        } else {
            command_buffer.present_drawable(drawable);
            command_buffer.commit();
        }
    }

    fn render_frame(
        &mut self,
        scene: &Scene,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        let clip_plan = self.prepare_clip_masks(scene, viewport_size);
        let mut writer = InstanceBufferWriter::new(
            &self.device,
            &self.instance_buffer_pool,
            self.is_unified_memory,
        );
        let instance_bindings = write_instances(scene, &clip_plan, &mut writer).with_context(|| {
            format!(
                "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                scene.paths.len(),
                scene.shadows.len(),
                scene.quads.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
            )
        })?;
        let command_buffer = self.draw_primitives_to_texture(
            scene,
            &clip_plan,
            &instance_bindings,
            &mut writer,
            texture,
            viewport_size,
        )?;

        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let instance_buffer = Cell::new(Some(writer.finish()));
        let block = RcBlock::new(move |_: ptr::NonNull<AnyObject>| {
            if let Some(instance_buffer) = instance_buffer.take() {
                instance_buffer_pool.lock().release(instance_buffer);
            }
        });
        // SAFETY: Both pointee types are opaque views of the same Objective-C block pointer ABI.
        unsafe {
            command_buffer.add_completed_handler(&*RcBlock::as_ptr(&block).cast());
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
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = layer
            .next_drawable()
            .ok_or_else(|| anyhow::anyhow!("Failed to get drawable for render_to_image"))?;

        let command_buffer = self.render_frame(scene, drawable.texture(), viewport_size)?;

        // Commit and wait for completion without presenting
        command_buffer.commit();
        command_buffer.wait_until_completed();

        read_texture_to_image(drawable.texture())
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
            // Update path intermediate textures for this size
            self.update_path_intermediate_textures(size);

            // Create an offscreen texture as render target
            let texture_descriptor = metal::TextureDescriptor::new();
            texture_descriptor.set_width(size.width.0 as u64);
            texture_descriptor.set_height(size.height.0 as u64);
            texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            texture_descriptor.set_usage(
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            // Like the atlas, only Apple GPUs can create shared textures on macOS;
            // Intel Macs cannot, even with unified memory. iOS has no managed storage.
            let uses_shared_storage = cfg!(target_os = "ios") || self.is_apple_gpu;
            texture_descriptor.set_storage_mode(if uses_shared_storage {
                metal::MTLStorageMode::Shared
            } else {
                metal::MTLStorageMode::Managed
            });
            let target_texture = self.device.new_texture(&texture_descriptor);

            let command_buffer = self.render_frame(scene, &target_texture, size)?;

            // Managed textures require an explicit blit synchronize before the CPU
            // can read back the rendered data. Without this, get_bytes returns
            // stale zeros.
            if !uses_shared_storage {
                let blit = command_buffer.new_blit_command_encoder();
                blit.synchronize_resource(&target_texture);
                blit.end_encoding();
            }

            // Commit and wait for completion
            command_buffer.commit();
            command_buffer.wait_until_completed();

            read_texture_to_image(&target_texture)
        })
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
            self.update_path_intermediate_textures(size);

            let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
                texture.width() != size.width.0 as u64 || texture.height() != size.height.0 as u64
            });
            if needs_new_target {
                let texture_descriptor = metal::TextureDescriptor::new();
                texture_descriptor.set_width(size.width.0 as u64);
                texture_descriptor.set_height(size.height.0 as u64);
                texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
                texture_descriptor.set_usage(
                    metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
                );
                texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
                self.headless_render_target = Some(self.device.new_texture(&texture_descriptor));
            }
            let target_texture = self
                .headless_render_target
                .clone()
                .expect("just ensured the render target exists");

            let command_buffer = self.render_frame(scene, &target_texture, size)?;

            // Commit without waiting, mirroring presentation to a real window where
            // the CPU doesn't block on the GPU.
            command_buffer.commit();
            Ok(())
        })
    }

    fn draw_primitives_to_texture(
        &mut self,
        scene: &Scene,
        clip_plan: &ClipPlan,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        let alpha = if self.opaque { 1. } else { 0. };

        // Every clip mask this frame samples is rasterized before the first
        // primitive is drawn, so a batch never has to interrupt the main pass
        // to go and build one.
        self.render_clip_masks(clip_plan, writer, command_buffer)?;
        self.group_targets.begin_frame();

        // The scene is walked as steps rather than as batches so that a group
        // boundary can end one render pass and begin another. A scene with no
        // groups in it is one step of one run, which is the same batches in the
        // same order into the same encoder as before.
        let mut target = ActiveTarget {
            texture: texture.to_owned(),
            render_target: RenderTarget::whole(viewport_size),
        };
        let mut command_encoder = new_command_encoder_for_texture(
            command_buffer,
            &target.texture,
            target.render_target,
            Some(metal::MTLClearColor::new(0., 0., 0., alpha)),
        );
        // `None` for a group that had to be painted without a target of its
        // own, which still has to be popped in step with the scene.
        let mut open_groups: Vec<Option<OpenGroup>> = Vec::new();

        for step in scene.steps() {
            let run = match step {
                SceneStep::PushGroup(spec) => {
                    match self.open_group(spec, &target, viewport_size, open_groups.len()) {
                        Some(group) => {
                            command_encoder.end_encoding();
                            target = group.target.clone();
                            command_encoder = new_command_encoder_for_texture(
                                command_buffer,
                                &target.texture,
                                target.render_target,
                                Some(metal::MTLClearColor::new(0., 0., 0., 0.)),
                            );
                            open_groups.push(Some(group));
                        }
                        None => open_groups.push(None),
                    }
                    continue;
                }
                SceneStep::PopGroup => {
                    let group = open_groups
                        .pop()
                        .expect("a scene popped a group it never pushed");
                    let Some(group) = group else {
                        continue;
                    };
                    command_encoder.end_encoding();
                    target = group.parent.clone();
                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        &target.texture,
                        target.render_target,
                        None,
                    );
                    self.composite_group(&group, instance_bindings, command_encoder);
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
                            command_encoder,
                        )
                    }
                    PrimitiveBatch::Quads(range) => {
                        let clipped = scene.quads[range.start].clip.is_clipped();
                        self.draw_quads(
                            range,
                            clipped,
                            instance_bindings,
                            &target.render_target,
                            command_encoder,
                        )
                    }
                    PrimitiveBatch::Paths(range) => {
                        let paths = &scene.paths[range];
                        let clipped = paths.first().is_some_and(|path| path.clip.is_clipped());
                        command_encoder.end_encoding();

                        // Paths are rasterized into a window-sized intermediate
                        // whatever target they are on their way to, so this
                        // pass keeps the viewport it always had; only the copy
                        // out of it lands on the group.
                        let did_draw = self.draw_paths_to_intermediate(
                            paths,
                            clipped,
                            instance_bindings,
                            writer,
                            viewport_size,
                            command_buffer,
                        )?;

                        command_encoder = new_command_encoder_for_texture(
                            command_buffer,
                            &target.texture,
                            target.render_target,
                            None,
                        );

                        if did_draw {
                            if let Err(error) = self.draw_paths_from_intermediate(
                                paths,
                                writer,
                                &target.render_target,
                                viewport_size,
                                command_encoder,
                            ) {
                                command_encoder.end_encoding();
                                return Err(error);
                            }
                        }
                    }
                    PrimitiveBatch::Underlines(range) => {
                        let clipped = scene.underlines[range.start].clip.is_clipped();
                        self.draw_underlines(
                            range,
                            clipped,
                            instance_bindings,
                            &target.render_target,
                            command_encoder,
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
                            &target.render_target,
                            command_encoder,
                        )
                    }
                    PrimitiveBatch::Surfaces(range) => self.draw_surfaces(
                        &scene.surfaces[range.clone()],
                        range.start,
                        instance_bindings,
                        &target.render_target,
                        command_encoder,
                    ),
                    PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
                }
            }
        }

        command_encoder.end_encoding();
        self.group_targets.end_frame();

        Ok(command_buffer.to_owned())
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
    ) -> Option<OpenGroup> {
        let viewport = DeviceRect {
            x: 0,
            y: 0,
            width: viewport_size.width.0,
            height: viewport_size.height.0,
        };
        // A group whose bounds have fallen off the window paints nothing either
        // way, so this is not worth a line in the log - and by the time a spec
        // reaches a renderer its bounds cover everything painted inside it
        // (`Scene::grow_group_to_its_content`), so an empty rectangle here is a
        // group that painted nothing rather than one whose caller left
        // `GroupOptions::bounds` at its default.
        let rect = covering_rect(&spec.bounds).intersect(&viewport);
        if rect.is_empty() {
            return None;
        }
        if depth >= MAX_GROUP_DEPTH {
            self.group_targets.report(format_args!(
                "isolated groups nested more than {MAX_GROUP_DEPTH} deep;                  the innermost are painted without isolation"
            ));
            return None;
        }
        let texture = self.group_targets.acquire(
            &self.device,
            depth,
            rect,
            GROUP_TARGET_BUDGET_BYTES,
            "the group is painted without isolation",
        )?;
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        Some(OpenGroup {
            spec: spec.clone(),
            bounds: rect.bounds(),
            texture_size,
            target: ActiveTarget {
                texture,
                render_target: RenderTarget {
                    size: size(DevicePixels(rect.width), DevicePixels(rect.height)),
                    origin: point(DevicePixels(rect.x), DevicePixels(rect.y)),
                },
            },
            parent: parent.clone(),
        })
    }

    /// Draw a finished group's target onto the target underneath it: one quad
    /// over the group's bounds, blended with the operator the group asked for.
    fn composite_group(
        &mut self,
        group: &OpenGroup,
        instance_bindings: &InstanceBindings,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if group.spec.blend.mix != MixMode::Normal {
            self.group_targets.report(format_args!(
                "the {:?} blend mode is not implemented; the group is composited normally",
                group.spec.blend.mix
            ));
        }
        if let Some(backdrop) = &group.spec.backdrop_filter {
            self.group_targets.report(format_args!(
                "the backdrop filter {backdrop:?} is not implemented; the group is composited \
                 over an unfiltered backdrop"
            ));
        }
        let compose = match GroupComposeMode::of(group.spec.blend.compose) {
            Some(compose) => compose,
            None => {
                self.group_targets.report(format_args!(
                    "the {:?} compose operator is not implemented; the group is                      composited source-over",
                    group.spec.blend.compose
                ));
                GroupComposeMode::SrcOver
            }
        };
        let filter = match GroupFilter::of(group.spec.filter.as_ref()) {
            Some(filter) => filter,
            None => {
                self.group_targets.report(format_args!(
                    "the filter {:?} is not implemented; the group is composited unfiltered",
                    group.spec.filter
                ));
                GroupFilter::NONE
            }
        };
        let clipped = group.spec.clip.is_clipped() && self.clip.is_some();
        let pipeline = self.group_composite_pipeline(compose, clipped);
        command_encoder.set_render_pipeline_state(&pipeline);

        let composite = GroupComposite {
            bounds: group.bounds,
            texture_size: group.texture_size,
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
        };
        command_encoder.set_vertex_buffer(
            GroupCompositeInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            GroupCompositeInputIndex::Composite as u64,
            mem::size_of::<GroupComposite>() as u64,
            &composite as *const GroupComposite as *const _,
        );
        command_encoder.set_vertex_bytes(
            GroupCompositeInputIndex::RenderTarget as u64,
            mem::size_of::<RenderTarget>() as u64,
            &group.parent.render_target as *const RenderTarget as *const _,
        );
        command_encoder.set_fragment_bytes(
            GroupCompositeInputIndex::Composite as u64,
            mem::size_of::<GroupComposite>() as u64,
            &composite as *const GroupComposite as *const _,
        );
        command_encoder.set_fragment_bytes(
            GroupCompositeInputIndex::Filter as u64,
            mem::size_of::<GroupFilter>() as u64,
            &filter as *const GroupFilter as *const _,
        );
        command_encoder.set_fragment_texture(
            GroupCompositeInputIndex::GroupTexture as u64,
            Some(&group.target.texture),
        );
        self.bind_clip_mask(
            self.clip_variant(clipped),
            command_encoder,
            instance_bindings,
            GroupCompositeInputIndex::ClipMasks as u64,
            GroupCompositeInputIndex::ClipAtlas as u64,
        );
        command_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
    }

    /// The composite pipeline for one operator, built the first time a scene
    /// asks for it.
    fn group_composite_pipeline(
        &mut self,
        compose: GroupComposeMode,
        clipped: bool,
    ) -> metal::RenderPipelineState {
        if let Some(pipeline) = self.group_composite_pipelines.get(&(compose, clipped)) {
            return pipeline.clone();
        }
        let pipeline = build_group_composite_pipeline_state(
            &self.device,
            &self.library,
            compose,
            clipped,
            MTLPixelFormat::BGRA8Unorm,
        );
        self.group_composite_pipelines
            .insert((compose, clipped), pipeline.clone());
        pipeline
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        clipped: bool,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<bool> {
        if paths.is_empty() {
            return Ok(false);
        }
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

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

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        let clip = self.clip_variant(clipped);
        command_encoder.set_render_pipeline_state(
            clip.map_or(&self.paths_rasterization_pipeline_state, |clip| {
                &clip.paths_rasterization_pipeline_state
            }),
        );
        if let (Some(clip), Some(clip_id_bindings)) = (clip, clip_id_bindings.as_ref()) {
            command_encoder.set_fragment_buffer(
                PathRasterizationInputIndex::ClipIds as u64,
                Some(&clip_id_bindings.buffer),
                clip_id_bindings.offset as u64,
            );
            command_encoder.set_fragment_buffer(
                PathRasterizationInputIndex::ClipMasks as u64,
                Some(&instance_bindings.clip_masks.buffer),
                instance_bindings.clip_masks.offset as u64,
            );
            command_encoder.set_fragment_texture(
                PathRasterizationInputIndex::ClipAtlas as u64,
                Some(&clip.atlas),
            );
        }
        command_encoder.set_vertex_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instance_bindings.buffer),
            vertex_instance_bindings.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            PathRasterizationInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instance_bindings.buffer),
            vertex_instance_bindings.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            PathRasterizationInputIndex::ContentMasks as u64,
            Some(&content_mask_bindings.buffer),
            content_mask_bindings.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            PathRasterizationInputIndex::Brushes as u64,
            Some(&brush_bindings.buffer),
            brush_bindings.offset as u64,
        );
        command_encoder.set_fragment_bytes(
            PathRasterizationInputIndex::BrushCount as u64,
            mem::size_of_val(&brush_count) as u64,
            &brush_count as *const u32 as *const _,
        );

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
                    .map(|texture_id| self.sprite_atlas.metal_texture(texture_id));
                command_encoder.set_fragment_texture(
                    PathRasterizationInputIndex::BrushAtlas as u64,
                    Some(texture.as_deref().unwrap_or(&self.default_brush_texture)),
                );
                bound_texture = Some(run.texture);
            }
            command_encoder.draw_primitives(
                metal::MTLPrimitiveType::Triangle,
                run.vertices.start,
                run.vertices.end - run.vertices.start,
            );
        }

        command_encoder.end_encoding();
        Ok(true)
    }

    fn draw_shadows(
        &self,
        shadows: Range<usize>,
        clipped: bool,
        instance_bindings: &InstanceBindings,
        render_target: &RenderTarget,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if shadows.is_empty() {
            return;
        }

        let clip = self.clip_variant(clipped);
        command_encoder.set_render_pipeline_state(
            clip.map_or(&self.shadows_pipeline_state, |clip| {
                &clip.shadows_pipeline_state
            }),
        );
        self.bind_clip_mask(
            clip,
            command_encoder,
            instance_bindings,
            ShadowInputIndex::ClipMasks as u64,
            ShadowInputIndex::ClipAtlas as u64,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_bindings.shadows.buffer),
            instance_bindings.shadows.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_bindings.shadows.buffer),
            instance_bindings.shadows.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            ShadowInputIndex::RenderTarget as u64,
            mem::size_of_val(render_target) as u64,
            render_target as *const RenderTarget as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
            shadows.start as u64,
        );
    }

    fn draw_quads(
        &self,
        quads: Range<usize>,
        clipped: bool,
        instance_bindings: &InstanceBindings,
        render_target: &RenderTarget,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if quads.is_empty() {
            return;
        }

        let clip = self.clip_variant(clipped);
        command_encoder.set_render_pipeline_state(
            clip.map_or(&self.quads_pipeline_state, |clip| {
                &clip.quads_pipeline_state
            }),
        );
        self.bind_clip_mask(
            clip,
            command_encoder,
            instance_bindings,
            QuadInputIndex::ClipMasks as u64,
            QuadInputIndex::ClipAtlas as u64,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_bindings.quads.buffer),
            instance_bindings.quads.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_bindings.quads.buffer),
            instance_bindings.quads.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            QuadInputIndex::RenderTarget as u64,
            mem::size_of_val(render_target) as u64,
            render_target as *const RenderTarget as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            quads.len() as u64,
            quads.start as u64,
        );
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        render_target: &RenderTarget,
        intermediate_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> Result<()> {
        let Some(first_path) = paths.first() else {
            return Ok(());
        };
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::RenderTarget as u64,
            mem::size_of_val(render_target) as u64,
            render_target as *const RenderTarget as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&intermediate_size) as u64,
            &intermediate_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if paths.last().unwrap().order == first_path.order {
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
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&sprite_instance_bindings.buffer),
            sprite_instance_bindings.offset as u64,
        );

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        Ok(())
    }

    fn draw_underlines(
        &self,
        underlines: Range<usize>,
        clipped: bool,
        instance_bindings: &InstanceBindings,
        render_target: &RenderTarget,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if underlines.is_empty() {
            return;
        }

        let clip = self.clip_variant(clipped);
        command_encoder.set_render_pipeline_state(
            clip.map_or(&self.underlines_pipeline_state, |clip| {
                &clip.underlines_pipeline_state
            }),
        );
        self.bind_clip_mask(
            clip,
            command_encoder,
            instance_bindings,
            UnderlineInputIndex::ClipMasks as u64,
            UnderlineInputIndex::ClipAtlas as u64,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_bindings.underlines.buffer),
            instance_bindings.underlines.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_bindings.underlines.buffer),
            instance_bindings.underlines.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::RenderTarget as u64,
            mem::size_of_val(render_target) as u64,
            render_target as *const RenderTarget as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
            underlines.start as u64,
        );
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        clipped: bool,
        instance_bindings: &InstanceBindings,
        render_target: &RenderTarget,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        let clip = self.clip_variant(clipped);
        command_encoder.set_render_pipeline_state(
            clip.map_or(&self.monochrome_sprites_pipeline_state, |clip| {
                &clip.monochrome_sprites_pipeline_state
            }),
        );
        self.bind_clip_mask(
            clip,
            command_encoder,
            instance_bindings,
            SpriteInputIndex::ClipMasks as u64,
            SpriteInputIndex::ClipAtlas as u64,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.monochrome_sprites.buffer),
            instance_bindings.monochrome_sprites.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::RenderTarget as u64,
            mem::size_of_val(render_target) as u64,
            render_target as *const RenderTarget as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        // `monochrome_sprite_fragment` reads nothing out of the instance
        // buffer - the vertex stage hands it everything it needs, so that a
        // glyph does not pay for a struct load per fragment - and so the
        // buffer is bound to the vertex stage alone.
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        clipped: bool,
        instance_bindings: &InstanceBindings,
        render_target: &RenderTarget,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        let clip = self.clip_variant(clipped);
        command_encoder.set_render_pipeline_state(
            clip.map_or(&self.polychrome_sprites_pipeline_state, |clip| {
                &clip.polychrome_sprites_pipeline_state
            }),
        );
        self.bind_clip_mask(
            clip,
            command_encoder,
            instance_bindings,
            SpriteInputIndex::ClipMasks as u64,
            SpriteInputIndex::ClipAtlas as u64,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.polychrome_sprites.buffer),
            instance_bindings.polychrome_sprites.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::RenderTarget as u64,
            mem::size_of_val(render_target) as u64,
            render_target as *const RenderTarget as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.polychrome_sprites.buffer),
            instance_bindings.polychrome_sprites.offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        first_surface: usize,
        instance_bindings: &InstanceBindings,
        render_target: &RenderTarget,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        let Some(first_surface_record) = surfaces.first() else {
            return;
        };
        // A batch is homogeneous in whether it is clipped, so the whole run
        // agrees with its first element.
        let clip = self.clip_variant(first_surface_record.clip.is_clipped());
        self.bind_clip_mask(
            clip,
            command_encoder,
            instance_bindings,
            SurfaceInputIndex::ClipMasks as u64,
            SurfaceInputIndex::ClipAtlas as u64,
        );

        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Surfaces as u64,
            Some(&instance_bindings.surfaces.buffer),
            instance_bindings.surfaces.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SurfaceInputIndex::RenderTarget as u64,
            mem::size_of_val(render_target) as u64,
            render_target as *const RenderTarget as *const _,
        );

        for (index, surface) in surfaces.iter().enumerate() {
            let image_buffer = &surface.image_buffer;
            let texture_size = size(
                DevicePixels::from(CVPixelBufferGetWidth(image_buffer) as i32),
                DevicePixels::from(CVPixelBufferGetHeight(image_buffer) as i32),
            );

            let texture = |plane, format, width, height| unsafe {
                let mut texture = ptr::null_mut();
                let status = CVMetalTextureCache::create_texture_from_image(
                    None,
                    &self.core_video_texture_cache,
                    image_buffer,
                    None,
                    format,
                    width,
                    height,
                    plane,
                    ptr::NonNull::from(&mut texture),
                );
                (status == kCVReturnSuccess)
                    .then(|| ptr::NonNull::new(texture))
                    .flatten()
                    .map(|texture| CFRetained::<CVMetalTexture>::from_raw(texture))
                    .ok_or(status)
            };
            let bind = |slot: SurfaceInputIndex, texture: &CFRetained<CVMetalTexture>| {
                command_encoder.set_fragment_texture(slot as u64, unsafe {
                    let texture = CVMetalTextureGetTexture(texture)
                        .expect("CoreVideo returned no Metal texture");
                    Some(metal::TextureRef::from_ptr(
                        objc2::rc::Retained::as_ptr(&texture) as *mut _,
                    ))
                });
            };

            // The CVMetalTexture wrappers must outlive the draw call below:
            // the encoder retains the MTLTextures they expose, and dropping
            // the wrappers at the end of the iteration matches the lifetime
            // the video path has always used.
            let _textures;
            match CVPixelBufferGetPixelFormatType(image_buffer) {
                format if format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange => {
                    command_encoder.set_render_pipeline_state(
                        clip.map_or(&self.surfaces_pipeline_state, |clip| {
                            &clip.surfaces_pipeline_state
                        }),
                    );
                    let plane = |plane, format| {
                        texture(
                            plane,
                            format,
                            CVPixelBufferGetWidthOfPlane(image_buffer, plane),
                            CVPixelBufferGetHeightOfPlane(image_buffer, plane),
                        )
                        .expect("failed to create CoreVideo Metal texture")
                    };
                    let y_texture = plane(0, objc2_metal::MTLPixelFormat::R8Unorm);
                    let cb_cr_texture = plane(1, objc2_metal::MTLPixelFormat::RG8Unorm);
                    bind(SurfaceInputIndex::YTexture, &y_texture);
                    bind(SurfaceInputIndex::CbCrTexture, &cb_cr_texture);
                    _textures = (y_texture, Some(cb_cr_texture));
                }
                format if format == kCVPixelFormatType_32BGRA => {
                    command_encoder.set_render_pipeline_state(
                        clip.map_or(&self.bgra_surfaces_pipeline_state, |clip| {
                            &clip.bgra_surfaces_pipeline_state
                        }),
                    );
                    // Unlike the video path above (whose AVFoundation buffers
                    // are always cache-compatible), BGRA buffers come from
                    // app code — skip the draw instead of panicking the
                    // frame loop if the texture cache rejects one.
                    let bgra_texture = match texture(
                        0,
                        objc2_metal::MTLPixelFormat::BGRA8Unorm,
                        CVPixelBufferGetWidth(image_buffer),
                        CVPixelBufferGetHeight(image_buffer),
                    ) {
                        Ok(texture) => texture,
                        Err(status) => {
                            log::error!(
                                "failed to create Metal texture for BGRA surface \
                                 (CVReturn {status}); skipping draw"
                            );
                            continue;
                        }
                    };
                    bind(SurfaceInputIndex::YTexture, &bgra_texture);
                    _textures = (bgra_texture, None);
                }
                format => {
                    log::error!("unsupported surface pixel format: {format:#x}; skipping draw");
                    continue;
                }
            }

            command_encoder.set_vertex_bytes(
                SurfaceInputIndex::TextureSize as u64,
                mem::size_of_val(&texture_size) as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );

            command_encoder.draw_primitives_instanced_base_instance(
                metal::MTLPrimitiveType::Triangle,
                0,
                6,
                1,
                (first_surface + index) as u64,
            );
        }
    }
    /// The clipped pipeline variants and the coverage atlas, when this batch
    /// is one that needs them.
    fn clip_variant(&self, clipped: bool) -> Option<&ClipResources> {
        if clipped { self.clip.as_ref() } else { None }
    }

    /// Binds the coverage atlas and the per-clip records the clipped variant of
    /// a fragment shader reads. A no-op for an unclipped batch, whose pipeline
    /// declares neither.
    fn bind_clip_mask(
        &self,
        clip: Option<&ClipResources>,
        command_encoder: &metal::RenderCommandEncoderRef,
        instance_bindings: &InstanceBindings,
        masks_index: u64,
        atlas_index: u64,
    ) {
        let Some(clip) = clip else {
            return;
        };
        command_encoder.set_fragment_buffer(
            masks_index,
            Some(&instance_bindings.clip_masks.buffer),
            instance_bindings.clip_masks.offset as u64,
        );
        command_encoder.set_fragment_texture(atlas_index, Some(&clip.atlas));
    }

    /// Works out where in the coverage atlas each of this scene's clip paths
    /// goes, sizing the clip textures and building the clip pipelines if the
    /// scene is the first to ask for either.
    fn prepare_clip_masks(&mut self, scene: &Scene, viewport_size: Size<DevicePixels>) -> ClipPlan {
        if scene.clips.is_empty() {
            // A frame that clips nothing still counts against the budget, so a
            // window that stops clipping - a dialog closing, an email view
            // going away - hands its textures back instead of keeping them for
            // the life of the process.
            if self.clip.is_some() {
                let atlas = self.clip_atlas_budget.floor();
                self.clip_atlas_budget
                    .observe(atlas, Extent::ZERO.quantized());
                let work = self.clip_work_budget.floor();
                self.clip_work_budget
                    .observe(work, Extent::ZERO.quantized());
                self.ensure_clip_resources(atlas, work);
            }
            return ClipPlan::unclipped();
        }
        let plan = ClipPlan::new(scene, viewport_size, self.clip_atlas_budget.floor());
        self.clip_atlas_budget
            .observe(plan.atlas, plan.minimum_atlas);
        let work = self.clip_work_budget.floor().max(plan.work);
        self.clip_work_budget.observe(work, plan.work);
        self.ensure_clip_resources(plan.atlas, work);
        plan
    }

    fn ensure_clip_resources(&mut self, atlas: Extent, work: Extent) {
        let device = self.device.clone();
        match &mut self.clip {
            None => {
                self.clip = Some(ClipResources::new(
                    &device,
                    &self.library,
                    self.is_apple_gpu,
                    atlas,
                    work,
                ));
            }
            Some(clip) => clip.resize(&device, self.is_apple_gpu, atlas, work),
        }
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
        command_buffer: &metal::CommandBufferRef,
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
            let descriptor = metal::RenderPassDescriptor::new();
            let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
            color_attachment.set_texture(Some(&clip.multisample));
            color_attachment.set_resolve_texture(Some(&clip.scratch));
            color_attachment.set_load_action(metal::MTLLoadAction::Clear);
            color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
            let stencil_attachment = descriptor
                .stencil_attachment()
                .context("render pass descriptor has no stencil attachment")?;
            stencil_attachment.set_texture(Some(&clip.stencil));
            stencil_attachment.set_load_action(metal::MTLLoadAction::Clear);
            stencil_attachment.set_clear_stencil(0);
            stencil_attachment.set_store_action(metal::MTLStoreAction::DontCare);

            let encoder = command_buffer.new_render_command_encoder(descriptor);
            encoder.set_viewport(metal::MTLViewport {
                originX: 0.,
                originY: 0.,
                width: clip.work_size.width as f64,
                height: clip.work_size.height as f64,
                znear: 0.,
                zfar: 1.,
            });
            encoder.set_stencil_reference_value(0);

            for &index in &level.clips {
                let Some(planned) = plan.clips[index].as_ref() else {
                    continue;
                };
                let even_odd = planned.fill_rule == FillRule::EvenOdd;
                encoder.set_scissor_rect(metal::MTLScissorRect {
                    x: planned.work.x as NSUInteger,
                    y: planned.work.y as NSUInteger,
                    width: planned.work.width as NSUInteger,
                    height: planned.work.height as NSUInteger,
                });

                if !planned.vertices.is_empty() {
                    encoder.set_render_pipeline_state(&clip.stencil_pipeline);
                    encoder.set_depth_stencil_state(if even_odd {
                        &clip.even_odd_stencil_state
                    } else {
                        &clip.nonzero_stencil_state
                    });
                    encoder.set_vertex_buffer(
                        ClipMaskInputIndex::Vertices as u64,
                        Some(&vertex_bindings.buffer),
                        vertex_bindings.offset as u64,
                    );
                    encoder.set_vertex_bytes(
                        ClipMaskInputIndex::TargetSize as u64,
                        mem::size_of_val(&target_size) as u64,
                        &target_size as *const Size<DevicePixels> as *const _,
                    );
                    encoder.draw_primitives(
                        metal::MTLPrimitiveType::Triangle,
                        planned.vertices.start as u64,
                        planned.vertices.len() as u64,
                    );
                }

                encoder.set_render_pipeline_state(&clip.cover_pipeline);
                encoder.set_depth_stencil_state(if even_odd {
                    &clip.even_odd_cover_state
                } else {
                    &clip.nonzero_cover_state
                });
                encoder.set_vertex_buffer(
                    ClipMaskInputIndex::Vertices as u64,
                    Some(&self.unit_vertices),
                    0,
                );
                encoder.set_vertex_bytes(
                    ClipMaskInputIndex::TargetSize as u64,
                    mem::size_of_val(&target_size) as u64,
                    &target_size as *const Size<DevicePixels> as *const _,
                );
                encoder.set_vertex_bytes(
                    ClipMaskInputIndex::Cover as u64,
                    mem::size_of::<ClipCover>() as u64,
                    &planned.cover as *const ClipCover as *const _,
                );
                encoder.set_fragment_bytes(
                    ClipMaskInputIndex::Cover as u64,
                    mem::size_of::<ClipCover>() as u64,
                    &planned.cover as *const ClipCover as *const _,
                );
                encoder
                    .set_fragment_texture(ClipMaskInputIndex::ClipAtlas as u64, Some(&clip.atlas));
                encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
            }
            encoder.end_encoding();

            let blit = command_buffer.new_blit_command_encoder();
            for &index in &level.clips {
                let Some(planned) = plan.clips[index].as_ref() else {
                    continue;
                };
                blit.copy_from_texture(
                    &clip.scratch,
                    0,
                    0,
                    metal::MTLOrigin {
                        x: planned.work.x as NSUInteger,
                        y: planned.work.y as NSUInteger,
                        z: 0,
                    },
                    metal::MTLSize {
                        width: planned.work.width as NSUInteger,
                        height: planned.work.height as NSUInteger,
                        depth: 1,
                    },
                    &clip.atlas,
                    0,
                    0,
                    metal::MTLOrigin {
                        x: planned.atlas.x as NSUInteger,
                        y: planned.atlas.y as NSUInteger,
                        z: 0,
                    },
                );
            }
            blit.end_encoding();
        }
        Ok(())
    }
}

fn new_command_encoder_for_texture<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    texture: &'a metal::TextureRef,
    render_target: RenderTarget,
    clear_color: Option<metal::MTLClearColor>,
) -> &'a metal::RenderCommandEncoderRef {
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(texture));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);
    if let Some(clear_color) = clear_color {
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(clear_color);
    } else {
        color_attachment.set_load_action(metal::MTLLoadAction::Load);
    }

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    // A group's target is pooled and so may be larger than the group; the
    // viewport is the part of it the group actually occupies, in its top-left
    // corner, and `render_target.origin` is where that sits in the window.
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(render_target.size.width) as f64,
        height: i32::from(render_target.size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
fn read_texture_to_image(texture: &metal::TextureRef) -> Result<RgbaImage> {
    let width = texture.width() as u32;
    let height = texture.height() as u32;
    let bytes_per_row = width as usize * 4;
    let mut pixels = vec![0u8; height as usize * bytes_per_row];

    let region = metal::MTLRegion {
        origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
        size: metal::MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };
    texture.get_bytes(
        pixels.as_mut_ptr() as *mut std::ffi::c_void,
        bytes_per_row as u64,
        region,
        0,
    );

    // Convert BGRA to RGBA (swap B and R channels)
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    RgbaImage::from_raw(width, height, pixels).context("failed to create RgbaImage from pixel data")
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    clipped: bool,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, Some(clip_constants(clipped)))
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, Some(clip_constants(clipped)))
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    clipped: bool,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, Some(clip_constants(clipped)))
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, Some(clip_constants(clipped)))
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
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
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    compose: GroupComposeMode,
    clipped: bool,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let constants = group_composite_constants(compose, clipped);
    let vertex_fn = library
        .get_function("group_composite_vertex", Some(constants))
        .expect("error locating the group composite vertex function");
    let fragment_fn = library
        .get_function(
            "group_composite_fragment",
            Some(group_composite_constants(compose, clipped)),
        )
        .expect("error locating the group composite fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(&format!("group_composite_{compose:?}"));
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    let (source, destination) = compose.blend_factors();
    color_attachment.set_source_rgb_blend_factor(source);
    color_attachment.set_source_alpha_blend_factor(source);
    color_attachment.set_destination_rgb_blend_factor(destination);
    color_attachment.set_destination_alpha_blend_factor(destination);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create the group composite pipeline state")
}

/// A one-texel texture to bind where a path rasterization pass has no image
/// brush. Nothing samples it - the per-path record says the path has no brush -
/// but the argument it fills is declared unconditionally, so it has to exist.
fn build_default_brush_texture(device: &metal::Device) -> metal::Texture {
    let descriptor = metal::TextureDescriptor::new();
    descriptor.set_width(1);
    descriptor.set_height(1);
    descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    descriptor.set_usage(metal::MTLTextureUsage::ShaderRead);
    descriptor.set_storage_mode(metal::MTLStorageMode::Private);
    device.new_texture(&descriptor)
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
    clipped: bool,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, Some(clip_constants(clipped)))
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, Some(clip_constants(clipped)))
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

#[derive(Clone)]
struct InstanceBinding {
    buffer: metal::Buffer,
    offset: usize,
}

struct InstanceBindings {
    quads: InstanceBinding,
    shadows: InstanceBinding,
    underlines: InstanceBinding,
    monochrome_sprites: InstanceBinding,
    polychrome_sprites: InstanceBinding,
    surfaces: InstanceBinding,
    /// One [`ClipMask`] per registered clip, plus the leading `ClipId::NONE`
    /// entry, so a fragment stage can index this by the id it already carries.
    clip_masks: InstanceBinding,
}

fn write_instances(
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

struct InstanceBufferWriter {
    device: metal::Device,
    pool: Arc<Mutex<InstanceBufferPool>>,
    unified_memory: bool,
    filled: Vec<(InstanceBuffer, usize)>,
    current: InstanceBuffer,
    offset: usize,
}

impl InstanceBufferWriter {
    fn new(
        device: &metal::Device,
        pool: &Arc<Mutex<InstanceBufferPool>>,
        unified_memory: bool,
    ) -> Self {
        let current = pool.lock().acquire(device, unified_memory);
        Self {
            device: device.clone(),
            pool: pool.clone(),
            unified_memory,
            filled: Vec::new(),
            current,
            offset: 0,
        }
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
        // Safety: the reservation lies within a buffer this frame owns
        // exclusively, and never overlaps one handed out earlier.
        let values = unsafe {
            let start = (self.current.metal_buffer.contents() as *mut u8).add(offset);
            slice::from_raw_parts_mut(start.cast::<MaybeUninit<T>>(), count)
        };
        Ok((binding, values))
    }

    fn write<T>(&mut self, values: &[T]) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                destination.as_mut_ptr().cast::<T>(),
                values.len(),
            );
        }
        Ok(binding)
    }

    fn write_iter<T>(
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
        let buffer = pool.acquire(&self.device, self.unified_memory);
        drop(pool);

        let filled = mem::replace(&mut self.current, buffer);
        self.filled.push((filled, self.offset));
        self.offset = 0;
        Ok(())
    }

    fn finish(self) -> InstanceBuffer {
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
                buffer.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: *written as NSUInteger,
                });
            }
            if offset > 0 {
                current.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: offset as NSUInteger,
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
enum SpriteInputIndex {
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
enum PathRasterizationInputIndex {
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
    fn whole(size: Size<DevicePixels>) -> Self {
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
    fn blend_factors(self) -> (metal::MTLBlendFactor, metal::MTLBlendFactor) {
        use metal::MTLBlendFactor as Factor;
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

/// A chain of colour matrices applied to a group's result before it is
/// composited.
///
/// Every CSS colour filter - `hue-rotate`, `saturate`, `sepia`, `grayscale`,
/// `opacity`, `invert`, `brightness`, `contrast` - is affine per channel, so a
/// chain of them is this and costs the composite draw nothing but arithmetic:
/// no extra pass, no extra target.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct GroupFilter {
    /// How many of `matrices` are in use.
    pub matrix_count: u32,
    pub pad: [u32; 3],
    /// [`MAX_GROUP_COLOR_MATRICES`] row-major 4x5 matrices, flattened.
    pub matrices: [f32; 160],
}

impl GroupFilter {
    /// No filter at all: the group's result is composited as it came out.
    const NONE: Self = GroupFilter {
        matrix_count: 0,
        pad: [0; 3],
        matrices: [0.; 160],
    };

    fn is_empty(&self) -> bool {
        self.matrix_count == 0
    }

    /// Adds one matrix, or fails when the chain is already
    /// [`MAX_GROUP_COLOR_MATRICES`] long.
    fn push(&mut self, matrix: &[f32; 20]) -> bool {
        let index = self.matrix_count as usize;
        if index >= MAX_GROUP_COLOR_MATRICES {
            return false;
        }
        self.matrices[index * 20..(index + 1) * 20].copy_from_slice(matrix);
        self.matrix_count += 1;
        true
    }
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

/// One step of a [`SceneFilter`], flattened out of the tree it arrives as and
/// with each run of adjacent colour matrices collapsed into a single step.
///
/// Collapsing the matrices is not the same as multiplying them together: the
/// shader still applies them one at a time with a clamp between, because CSS
/// clamps between filter primitives. It is only that a run of them costs one
/// pass rather than one pass each.
#[derive(Clone, Debug)]
enum FilterOp {
    Matrices(GroupFilter),
    Blur { sigma_x: f32, sigma_y: f32 },
    DropShadow { offset: PointF, sigma: f32, color: Hsla },
}

impl FilterOp {
    /// The steps a filter compiles to, in order.
    ///
    /// A run of colour matrices longer than [`MAX_GROUP_COLOR_MATRICES`] is not
    /// a filter this cannot express: it is two steps rather than one, and the
    /// second reads what the first wrote. Nothing here can fail, which is why
    /// there is no longer a "this filter is not implemented" line to log.
    fn flatten(filter: &SceneFilter) -> Vec<FilterOp> {
        fn walk(filter: &SceneFilter, ops: &mut Vec<FilterOp>) {
            match filter {
                SceneFilter::ColorMatrix(matrix) => {
                    if let Some(FilterOp::Matrices(matrices)) = ops.last_mut()
                        && matrices.push(matrix)
                    {
                        return;
                    }
                    let mut matrices = GroupFilter::NONE;
                    matrices.push(matrix);
                    ops.push(FilterOp::Matrices(matrices));
                }
                SceneFilter::Blur { radius_x, radius_y } => ops.push(FilterOp::Blur {
                    sigma_x: radius_x.max(0.),
                    sigma_y: radius_y.max(0.),
                }),
                SceneFilter::DropShadow {
                    offset_x,
                    offset_y,
                    radius,
                    color,
                } => ops.push(FilterOp::DropShadow {
                    offset: point(*offset_x, *offset_y),
                    sigma: radius.max(0.),
                    color: *color,
                }),
                SceneFilter::Chain(filters) => {
                    for filter in filters {
                        walk(filter, ops);
                    }
                }
            }
        }

        let mut ops = Vec::new();
        walk(filter, &mut ops);
        // A blur of zero moves nothing, and a chain can carry one: `blur(0)` is
        // legal CSS and a transition through it passes over it every time.
        ops.retain(|op| !matches!(op, FilterOp::Blur { sigma_x, sigma_y } if *sigma_x <= 0. && *sigma_y <= 0.));
        ops
    }
}

/// The attachment a render pass is drawing into, and where it sits in the
/// window.
#[derive(Clone)]
struct ActiveTarget {
    texture: metal::Texture,
    render_target: RenderTarget,
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
    texture: metal::Texture,
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
    fn acquire(&mut self, device: &metal::Device, rect: DeviceRect) -> Option<FilterImage> {
        let slot = self.in_use.iter().position(|used| !used)?;
        let texture = self.targets.acquire(
            device,
            slot,
            rect,
            FILTER_SCRATCH_BUDGET_BYTES,
            "the filter it was for is dropped and the group composited unfiltered",
        )?;
        self.in_use[slot] = true;
        Some(FilterImage {
            texture_size: size(
                DevicePixels(texture.width() as i32),
                DevicePixels(texture.height() as i32),
            ),
            texture,
            rect,
            slot: Some(slot),
        })
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
    texture: Option<metal::Texture>,
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
        device: &metal::Device,
        depth: usize,
        rect: DeviceRect,
        budget: usize,
        on_refusal: &str,
    ) -> Option<metal::Texture> {
        while self.levels.len() <= depth {
            self.levels.push(GroupTargetLevel::default());
        }
        let required = quantized_group_extent(rect.extent());
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
                return None;
            }
            let descriptor = metal::TextureDescriptor::new();
            descriptor.set_width(wanted.width as u64);
            descriptor.set_height(wanted.height as u64);
            descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            descriptor.set_storage_mode(metal::MTLStorageMode::Private);
            descriptor.set_usage(
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            level.texture = Some(device.new_texture(&descriptor));
            level.allocated = wanted;
        }
        level.texture.clone()
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

/// A group target's extent, rounded up to whole [`GROUP_TEXTURE_QUANTUM`]s.
fn quantized_group_extent(extent: Extent) -> Extent {
    let round = |value: i32| {
        let quanta = (value.max(1) + GROUP_TEXTURE_QUANTUM - 1) / GROUP_TEXTURE_QUANTUM;
        quanta * GROUP_TEXTURE_QUANTUM
    };
    Extent {
        width: round(extent.width),
        height: round(extent.height),
    }
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

/// What a fragment shader needs to ask the coverage atlas how much of a point
/// one clip path lets through.
///
/// One of these per registered clip, plus a leading entry for [`ClipId::NONE`]
/// so the shader can index by the id it already has instead of subtracting one
/// from it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct ClipMask {
    /// The only part of the window this clip can reach, in device pixels: the
    /// content mask it was pushed under - already narrowed to the path's
    /// bounding box - intersected with its parent's tile and with the viewport.
    /// Outside it the clip lets nothing through, and the atlas is not read.
    ///
    /// It is empty for a clip that lets nothing through anywhere, which is how
    /// a path enclosing no area is spelled.
    pub tile: Bounds<ScaledPixels>,
    /// Added to a device position to reach the coverage texel in the atlas.
    pub atlas_offset: PointF,
    /// Zero for a clip that got no tile, whose `tile` rectangle is then the
    /// whole answer.
    pub sampled: u32,
    pub pad: u32,
}

impl ClipMask {
    /// The entry `ClipId::NONE` names.
    ///
    /// Nothing should ever reach it - a batch is homogeneous in whether it is
    /// clipped, so a clipped pipeline only ever draws primitives with a real
    /// clip - and it fails open rather than blanking the window if one does.
    const UNCLIPPED: Self = ClipMask {
        tile: Bounds {
            origin: Point {
                x: ScaledPixels(-1e30),
                y: ScaledPixels(-1e30),
            },
            size: Size {
                width: ScaledPixels(2e30),
                height: ScaledPixels(2e30),
            },
        },
        atlas_offset: Point { x: 0., y: 0. },
        sampled: 0,
        pad: 0,
    };
}

/// The per-clip constants the cover half of stencil-and-cover reads.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct ClipCover {
    /// The clip's tile, in atlas texels.
    pub tile: Bounds<ScaledPixels>,
    /// Added to a position in this tile to reach the same point in the
    /// parent's tile, so nesting intersects rather than replaces.
    pub parent_offset: PointF,
    /// Zero for a clip at the root, whose coverage is its own shape alone.
    pub has_parent: u32,
    pub pad: u32,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
pub struct MetalHeadlessRenderer {
    renderer: MetalRenderer,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl MetalHeadlessRenderer {
    pub fn new() -> Self {
        Self::with_transparency(false)
    }

    /// A headless renderer whose target is cleared to transparent rather than
    /// opaque black, so a test can read the scene's own alpha back.
    pub fn with_transparency(transparent: bool) -> Self {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool, transparent);
        Self { renderer }
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

    #[test]
    fn render_core_video_surface() {
        let attributes = CFDictionary::from_slices(
            &[unsafe { kCVPixelBufferIOSurfacePropertiesKey }],
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
        let image_buffer =
            unsafe { CFRetained::from_raw(ptr::NonNull::new(image_buffer).unwrap()) };
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
            bounds,
            content_mask: ContentMask { bounds },
            image_buffer: image_buffer.clone(),
        });
        drop(image_buffer);
        scene.finish();

        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(Default::default())));
        let image = renderer
            .render_scene_to_image(&scene, size(DevicePixels(16), DevicePixels(16)))
            .unwrap();
        for image::Rgba([red, green, blue, alpha]) in image.pixels() {
            assert!(red.abs_diff(128) <= 2);
            assert!(green.abs_diff(128) <= 2);
            assert!(blue.abs_diff(128) <= 2);
            assert_eq!(*alpha, 255);
        }
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
    stencil_pipeline: metal::RenderPipelineState,
    cover_pipeline: metal::RenderPipelineState,
    nonzero_stencil_state: metal::DepthStencilState,
    even_odd_stencil_state: metal::DepthStencilState,
    nonzero_cover_state: metal::DepthStencilState,
    even_odd_cover_state: metal::DepthStencilState,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    bgra_surfaces_pipeline_state: metal::RenderPipelineState,
    /// The coverage the clipped fragment shaders sample, one tile per clip.
    /// Never cleared: a tile is either written whole by this frame or never
    /// read, because a clip that has no tile says so in its `ClipMask`.
    atlas: metal::Texture,
    /// Where a level's tiles are resolved before being blitted into the atlas.
    scratch: metal::Texture,
    multisample: metal::Texture,
    stencil: metal::Texture,
    atlas_size: Extent,
    /// The size of the three working textures above, which is the size of the
    /// largest nesting level rather than of the atlas.
    work_size: Extent,
}

impl ClipResources {
    fn new(
        device: &metal::Device,
        library: &metal::LibraryRef,
        is_apple_gpu: bool,
        atlas_size: Extent,
        work_size: Extent,
    ) -> Self {
        let atlas = Self::coverage(device, atlas_size);
        let (scratch, multisample, stencil) = Self::working(device, is_apple_gpu, work_size);
        Self {
            stencil_pipeline: build_clip_mask_pipeline_state(
                device,
                library,
                "clip_stencil",
                "clip_stencil_vertex",
                "clip_stencil_fragment",
                false,
            ),
            cover_pipeline: build_clip_mask_pipeline_state(
                device,
                library,
                "clip_cover",
                "clip_cover_vertex",
                "clip_cover_fragment",
                true,
            ),
            nonzero_stencil_state: build_stencil_state(device, FillRule::NonZero),
            even_odd_stencil_state: build_stencil_state(device, FillRule::EvenOdd),
            nonzero_cover_state: build_cover_state(device, FillRule::NonZero),
            even_odd_cover_state: build_cover_state(device, FillRule::EvenOdd),
            paths_rasterization_pipeline_state: build_path_rasterization_pipeline_state(
                device,
                library,
                "paths_rasterization_clipped",
                "path_rasterization_vertex",
                "path_rasterization_fragment",
                MTLPixelFormat::BGRA8Unorm,
                PATH_SAMPLE_COUNT,
                true,
            ),
            shadows_pipeline_state: build_pipeline_state(
                device,
                library,
                "shadows_clipped",
                "shadow_vertex",
                "shadow_fragment",
                MTLPixelFormat::BGRA8Unorm,
                true,
            ),
            quads_pipeline_state: build_pipeline_state(
                device,
                library,
                "quads_clipped",
                "quad_vertex",
                "quad_fragment",
                MTLPixelFormat::BGRA8Unorm,
                true,
            ),
            underlines_pipeline_state: build_pipeline_state(
                device,
                library,
                "underlines_clipped",
                "underline_vertex",
                "underline_fragment",
                MTLPixelFormat::BGRA8Unorm,
                true,
            ),
            monochrome_sprites_pipeline_state: build_pipeline_state(
                device,
                library,
                "monochrome_sprites_clipped",
                "monochrome_sprite_vertex",
                "monochrome_sprite_fragment",
                MTLPixelFormat::BGRA8Unorm,
                true,
            ),
            polychrome_sprites_pipeline_state: build_pipeline_state(
                device,
                library,
                "polychrome_sprites_clipped",
                "polychrome_sprite_vertex",
                "polychrome_sprite_fragment",
                MTLPixelFormat::BGRA8Unorm,
                true,
            ),
            surfaces_pipeline_state: build_pipeline_state(
                device,
                library,
                "surfaces_clipped",
                "surface_vertex",
                "surface_fragment",
                MTLPixelFormat::BGRA8Unorm,
                true,
            ),
            bgra_surfaces_pipeline_state: build_path_sprite_pipeline_state(
                device,
                library,
                "bgra_surfaces_clipped",
                "surface_vertex",
                "surface_bgra_fragment",
                MTLPixelFormat::BGRA8Unorm,
                true,
            ),
            atlas,
            scratch,
            multisample,
            stencil,
            atlas_size,
            work_size,
        }
    }

    /// Reallocates whichever of the two groups of textures has changed size.
    /// The atlas and the working pair move independently: a frame can need a
    /// bigger atlas without needing a bigger level, and usually does.
    fn resize(
        &mut self,
        device: &metal::Device,
        is_apple_gpu: bool,
        atlas_size: Extent,
        work_size: Extent,
    ) {
        if self.atlas_size != atlas_size {
            self.atlas = Self::coverage(device, atlas_size);
            self.atlas_size = atlas_size;
        }
        if self.work_size != work_size {
            let (scratch, multisample, stencil) = Self::working(device, is_apple_gpu, work_size);
            self.scratch = scratch;
            self.multisample = multisample;
            self.stencil = stencil;
            self.work_size = work_size;
        }
    }

    fn coverage(device: &metal::Device, extent: Extent) -> metal::Texture {
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(extent.width as u64);
        descriptor.set_height(extent.height as u64);
        descriptor.set_pixel_format(MTLPixelFormat::R8Unorm);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        device.new_texture(&descriptor)
    }

    /// The resolve target and the multisample pair one nesting level is drawn
    /// through, before its tiles are blitted into the atlas.
    fn working(
        device: &metal::Device,
        is_apple_gpu: bool,
        extent: Extent,
    ) -> (metal::Texture, metal::Texture, metal::Texture) {
        // The multisample pair is written and consumed inside one render pass,
        // so on Apple silicon it never needs backing store at all.
        let transient = if is_apple_gpu {
            metal::MTLStorageMode::Memoryless
        } else {
            metal::MTLStorageMode::Private
        };
        let multisample = |format| {
            let descriptor = metal::TextureDescriptor::new();
            descriptor.set_width(extent.width as u64);
            descriptor.set_height(extent.height as u64);
            descriptor.set_pixel_format(format);
            descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
            descriptor.set_sample_count(PATH_SAMPLE_COUNT as u64);
            descriptor.set_storage_mode(transient);
            descriptor.set_usage(metal::MTLTextureUsage::RenderTarget);
            device.new_texture(&descriptor)
        };
        (
            Self::coverage(device, extent),
            multisample(MTLPixelFormat::R8Unorm),
            multisample(MTLPixelFormat::Stencil8),
        )
    }
}

/// A width and height in whole device pixels: the size of the coverage atlas,
/// or of the working attachments one nesting level is rendered through.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Extent {
    width: i32,
    height: i32,
}

impl Extent {
    const ZERO: Self = Extent {
        width: 0,
        height: 0,
    };

    fn max(self, other: Self) -> Self {
        Extent {
            width: self.width.max(other.width),
            height: self.height.max(other.height),
        }
    }

    /// What a BGRA8 texture of this size costs.
    fn bytes(self) -> usize {
        self.width.max(0) as usize * self.height.max(0) as usize * 4
    }

    /// Rounded up to whole [`CLIP_TEXTURE_QUANTUM`]s, and never to nothing: a
    /// size that wobbles by a pixel is then the same allocation, and a texture
    /// is never asked for with a side of zero.
    fn quantized(self) -> Self {
        let round = |value: i32| {
            let quanta = (value.max(1) + CLIP_TEXTURE_QUANTUM - 1) / CLIP_TEXTURE_QUANTUM;
            quanta * CLIP_TEXTURE_QUANTUM
        };
        Extent {
            width: round(self.width),
            height: round(self.height),
        }
    }
}

/// How large a set of textures is allowed to stay once a frame stops needing
/// them.
///
/// Growing is immediate: a frame that cannot fit what it needs draws the wrong
/// picture. Shrinking waits for `SHRINK_FRAMES` consecutive frames that would
/// all have fitted in something smaller, because recreating the textures is not
/// free and a list scrolling a clipped element in and out of view would
/// otherwise do it every few frames.
#[derive(Default)]
struct TextureBudget<const SHRINK_FRAMES: u32> {
    /// What is currently allocated.
    current: Extent,
    /// The largest a frame has needed since the present run of small frames
    /// began, and so what shrinking would shrink to.
    peak: Extent,
    /// How many consecutive frames have fitted inside `peak`.
    frames: u32,
}

/// The clip textures' budget, shrinking after [`CLIP_TEXTURE_SHRINK_FRAMES`].
type ClipTextureBudget = TextureBudget<CLIP_TEXTURE_SHRINK_FRAMES>;

impl<const SHRINK_FRAMES: u32> TextureBudget<SHRINK_FRAMES> {
    /// The size the next frame starts from: what is already allocated, unless
    /// a long enough run of frames has all fitted inside something smaller.
    fn floor(&mut self) -> Extent {
        if self.frames >= SHRINK_FRAMES {
            self.current = self.peak;
            self.peak = Extent::ZERO;
            self.frames = 0;
        }
        self.current
    }

    /// What the frame allocated, and what it would have been enough to
    /// allocate.
    fn observe(&mut self, allocated: Extent, needed: Extent) {
        self.current = allocated;
        if needed.width < allocated.width || needed.height < allocated.height {
            self.peak = self.peak.max(needed);
            self.frames += 1;
        } else {
            self.peak = Extent::ZERO;
            self.frames = 0;
        }
    }
}

/// A rectangle of whole device pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DeviceRect {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

impl DeviceRect {
    /// A rectangle enclosing nothing, which is what a clip that lets nothing
    /// through gets.
    const EMPTY: Self = DeviceRect {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
    };

    fn is_empty(&self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    fn intersect(&self, other: &Self) -> Self {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = (self.x + self.width).min(other.x + other.width);
        let bottom = (self.y + self.height).min(other.y + other.height);
        Self {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    }

    fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = (self.x + self.width).max(other.x + other.width);
        let bottom = (self.y + self.height).max(other.y + other.height);
        Self {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    }

    fn extent(&self) -> Extent {
        Extent {
            width: self.width.max(0),
            height: self.height.max(0),
        }
    }

    fn bounds(&self) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(self.x as f32), ScaledPixels(self.y as f32)),
            size: size(
                ScaledPixels(self.width.max(0) as f32),
                ScaledPixels(self.height.max(0) as f32),
            ),
        }
    }
}

/// One clip path, flattened and placed, before the atlas has been packed.
struct ClipShape {
    contours: Vec<Vec<PointF>>,
    /// The part of the window this clip can reach: the content mask it was
    /// pushed under - which `Window::with_clip_path` has already tightened to
    /// the path's own bounding box - intersected with its parent's tile and
    /// with the viewport.
    ///
    /// Taking the mask rather than the bare bounding box is what keeps a clip
    /// in a scrolled or narrow container costing a tile of what is on screen.
    /// It is also empty for a clip that lets nothing through, which is not the
    /// same thing as a clip that fell back to its rectangle: see `wants_tile`.
    device: DeviceRect,
    fill_rule: FillRule,
    parent: Option<usize>,
    depth: usize,
    wants_mask: bool,
}

impl ClipShape {
    /// Whether this clip is asking for a tile of the atlas at all.
    fn wants_tile(&self) -> bool {
        self.wants_mask && !self.device.is_empty() && !self.contours.is_empty()
    }
}

/// A clip path that got a tile, and everything the two mask passes need.
struct PlannedClip {
    /// Where the finished coverage lives in the atlas.
    atlas: DeviceRect,
    /// The same tile in the working attachment its level is drawn through,
    /// which is the atlas rectangle less the level's own origin.
    work: DeviceRect,
    vertices: Range<usize>,
    fill_rule: FillRule,
    cover: ClipCover,
}

/// One nesting depth: a render pass that reads the tiles the level above it
/// wrote.
struct ClipLevel {
    /// The clips at this depth that got a tile, in packing order.
    clips: Vec<usize>,
    /// The atlas rectangle their tiles span. Only this much is rendered and
    /// resolved, so a level of one small clip costs one small clip.
    extent: DeviceRect,
}

/// Where every clip path in one scene rasterizes to.
struct ClipPlan {
    /// Indexed by [`ClipId`] itself, entry zero standing for `ClipId::NONE`.
    masks: Vec<ClipMask>,
    /// Indexed by clip index; `None` for a clip that got no tile.
    clips: Vec<Option<PlannedClip>>,
    /// Nesting depths, shallowest first: level `k` is a render pass that reads
    /// the tiles level `k - 1` wrote.
    levels: Vec<ClipLevel>,
    vertices: Vec<PointF>,
    /// The atlas the tiles were packed into.
    atlas: Extent,
    /// The smallest atlas that would have held them. The renderer's budget
    /// watches this, so an atlas grown for one busy frame is given back.
    minimum_atlas: Extent,
    /// The working attachments the mask passes need: the largest level, which
    /// is all a multisample resolve has to cover.
    work: Extent,
}

impl ClipPlan {
    fn unclipped() -> Self {
        Self {
            masks: vec![ClipMask::UNCLIPPED],
            clips: Vec::new(),
            levels: Vec::new(),
            vertices: Vec::new(),
            atlas: Extent::ZERO,
            minimum_atlas: Extent::ZERO,
            work: Extent::ZERO,
        }
    }

    fn new(scene: &Scene, viewport_size: Size<DevicePixels>, floor: Extent) -> Self {
        let viewport = DeviceRect {
            x: 0,
            y: 0,
            width: viewport_size.width.0,
            height: viewport_size.height.0,
        };

        let mut shapes: Vec<ClipShape> = Vec::with_capacity(scene.clips.len());
        let mut too_deep = 0usize;
        for scene_clip in &scene.clips {
            let parent = scene_clip.parent.index();
            let depth = parent.map_or(0, |parent| shapes[parent].depth + 1);
            let parent_tile = parent.map_or(viewport, |parent| shapes[parent].device);
            let reachable = covering_rect(&scene_clip.path.bounds())
                .intersect(&covering_rect(&scene_clip.visible))
                .intersect(&parent_tile);
            let wants_mask = depth < MAX_CLIP_DEPTH;
            if !wants_mask {
                too_deep += 1;
            }
            let contours = if wants_mask && !reachable.is_empty() {
                flatten_clip_path(&scene_clip.path)
            } else {
                Vec::new()
            };
            // A path whose contours enclose no area - a lone `move_to`, or a
            // single line - lets nothing through, and an empty tile is how that
            // is said. It has to be told apart from the two fallbacks, which
            // keep the rectangle because there is still a shape inside it: a
            // clip past the depth limit never asked to be flattened, and one
            // the atlas had no room for flattened to something.
            let device = if wants_mask && contours.is_empty() {
                DeviceRect::EMPTY
            } else {
                reachable
            };
            shapes.push(ClipShape {
                contours,
                device,
                fill_rule: scene_clip.path.fill_rule(),
                parent,
                depth,
                wants_mask,
            });
        }

        // Packed level by level, and tallest first inside a level. A shelf
        // allocator wastes the difference between the tallest tile on a shelf
        // and every other tile on it, so an unsorted run of alternating tall
        // and short tiles doubles the atlas long before its area says it
        // should; and packing a level's tiles together is what keeps the
        // rectangle its render pass has to cover tight around them.
        let mut order: Vec<usize> = (0..shapes.len())
            .filter(|&index| shapes[index].wants_tile())
            .collect();
        order.sort_by_key(|&index| {
            (
                shapes[index].depth,
                std::cmp::Reverse(shapes[index].device.height),
                index,
            )
        });

        let (minimum_atlas, minimum_placements, minimum_fit) = smallest_atlas(&shapes, &order);
        let atlas = minimum_atlas.max(floor);
        let (placements, all_fit) = if atlas == minimum_atlas {
            (minimum_placements, minimum_fit)
        } else {
            pack_clip_tiles(&shapes, &order, atlas)
        };
        if !all_fit {
            log::error!(
                "the {}x{} clip mask atlas could not hold every clip path in this frame; the ones \
                 that did not fit are clipped to their bounding rectangles",
                atlas.width,
                atlas.height
            );
        }
        if too_deep > 0 {
            log::error!(
                "{too_deep} clip paths nested deeper than {MAX_CLIP_DEPTH}; they are clipped to \
                 their bounding rectangles"
            );
        }

        let mut levels: Vec<ClipLevel> = Vec::new();
        for &index in &order {
            let Some((x, y)) = placements[index] else {
                continue;
            };
            let shape = &shapes[index];
            let tile = DeviceRect {
                x,
                y,
                width: shape.device.width,
                height: shape.device.height,
            };
            while levels.len() <= shape.depth {
                levels.push(ClipLevel {
                    clips: Vec::new(),
                    extent: DeviceRect::EMPTY,
                });
            }
            let level = &mut levels[shape.depth];
            level.extent = level.extent.union(&tile);
            level.clips.push(index);
        }
        let work = levels
            .iter()
            .fold(Extent::ZERO, |largest, level| {
                largest.max(level.extent.extent())
            })
            .quantized();

        let mut masks = Vec::with_capacity(shapes.len() + 1);
        masks.push(ClipMask::UNCLIPPED);
        let mut clips = Vec::with_capacity(shapes.len());
        let mut vertices = Vec::new();

        for (index, shape) in shapes.iter().enumerate() {
            let Some((atlas_x, atlas_y)) = placements[index] else {
                masks.push(ClipMask {
                    tile: shape.device.bounds(),
                    atlas_offset: point(0., 0.),
                    sampled: 0,
                    pad: 0,
                });
                clips.push(None);
                continue;
            };

            let tile = DeviceRect {
                x: atlas_x,
                y: atlas_y,
                width: shape.device.width,
                height: shape.device.height,
            };
            let extent = levels[shape.depth].extent;
            let work_tile = DeviceRect {
                x: tile.x - extent.x,
                y: tile.y - extent.y,
                width: tile.width,
                height: tile.height,
            };
            // A device position plus `atlas_offset` is the texel in the atlas
            // that a fragment shader reads; plus `work_offset` it is the same
            // texel in the working attachment this level is drawn through.
            let atlas_offset = point(
                tile.x as f32 - shape.device.x as f32,
                tile.y as f32 - shape.device.y as f32,
            );
            let work_offset = point(
                atlas_offset.x - extent.x as f32,
                atlas_offset.y - extent.y as f32,
            );
            let start = vertices.len();
            for contour in &shape.contours {
                for corner in 1..contour.len().saturating_sub(1) {
                    for at in [contour[0], contour[corner], contour[corner + 1]] {
                        vertices.push(point(at.x + work_offset.x, at.y + work_offset.y));
                    }
                }
            }

            let parent_mask = shape.parent.map(|parent| masks[parent + 1]);
            let (parent_offset, has_parent) = match parent_mask {
                Some(parent) if parent.sampled != 0 => (
                    point(
                        parent.atlas_offset.x - work_offset.x,
                        parent.atlas_offset.y - work_offset.y,
                    ),
                    1,
                ),
                _ => (point(0., 0.), 0),
            };

            masks.push(ClipMask {
                tile: shape.device.bounds(),
                atlas_offset,
                sampled: 1,
                pad: 0,
            });
            clips.push(Some(PlannedClip {
                atlas: tile,
                work: work_tile,
                vertices: start..vertices.len(),
                fill_rule: shape.fill_rule,
                cover: ClipCover {
                    tile: work_tile.bounds(),
                    parent_offset,
                    has_parent,
                    pad: 0,
                },
            }));
        }

        Self {
            masks,
            clips,
            levels,
            vertices,
            atlas,
            minimum_atlas,
            work,
        }
    }
}

/// The smallest atlas this frame's tiles fit in, and where they landed in it.
///
/// Each side starts at what the largest tile in that axis needs and doubles
/// under packing pressure. A tile that on its own wants more than
/// [`CLIP_ATLAS_MAX_SIZE`] - a full-width clip on a 5K display is 5120 device
/// pixels across - still gets an atlas that holds it; the cap is only on how
/// far packing pressure alone may grow one.
fn smallest_atlas(
    shapes: &[ClipShape],
    order: &[usize],
) -> (Extent, Vec<Option<(i32, i32)>>, bool) {
    let required = order.iter().fold(Extent::ZERO, |required, &index| {
        required.max(shapes[index].device.extent())
    });
    let mut atlas = Extent {
        width: required.width.min(CLIP_ATLAS_TEXTURE_LIMIT),
        height: required.height.min(CLIP_ATLAS_TEXTURE_LIMIT),
    }
    .quantized();
    let cap = Extent {
        width: atlas.width.max(CLIP_ATLAS_MAX_SIZE),
        height: atlas.height.max(CLIP_ATLAS_MAX_SIZE),
    };

    let (mut placements, mut all_fit) = pack_clip_tiles(shapes, order, atlas);
    while !all_fit {
        if atlas.height <= atlas.width && atlas.height < cap.height {
            atlas.height = (atlas.height * 2).min(cap.height);
        } else if atlas.width < cap.width {
            atlas.width = (atlas.width * 2).min(cap.width);
        } else {
            break;
        }
        (placements, all_fit) = pack_clip_tiles(shapes, order, atlas);
    }
    (atlas, placements, all_fit)
}

/// Packs each clip in `order` a tile of `atlas`, and says whether they all got
/// one.
///
/// A shelf allocator with no free list: every tile dies at the end of the
/// frame, so the only thing to reclaim is the whole atlas, and reclaiming it is
/// starting the next frame at the origin. `order` is grouped by nesting depth
/// and sorted tallest first within a depth, and a change of depth starts a new
/// shelf, so each level's tiles come out contiguous and its render pass covers
/// only them.
fn pack_clip_tiles(
    shapes: &[ClipShape],
    order: &[usize],
    atlas: Extent,
) -> (Vec<Option<(i32, i32)>>, bool) {
    let mut placements = vec![None; shapes.len()];
    let mut all_fit = true;
    let (mut x, mut y, mut shelf_height) = (0, 0, 0);
    let mut shelf_depth = None;
    for &index in order {
        let shape = &shapes[index];
        let (width, height) = (shape.device.width, shape.device.height);
        if width > atlas.width || height > atlas.height {
            all_fit = false;
            continue;
        }
        if shelf_depth != Some(shape.depth) || x + width > atlas.width {
            x = 0;
            y += shelf_height;
            shelf_height = 0;
            shelf_depth = Some(shape.depth);
        }
        if y + height > atlas.height {
            all_fit = false;
            continue;
        }
        placements[index] = Some((x, y));
        x += width;
        shelf_height = shelf_height.max(height);
    }
    (placements, all_fit)
}

/// How far from the origin a clip path's bounding box is taken seriously.
///
/// A path is arbitrary, so its box can be arbitrary too, and `f32 as i32`
/// saturates: two saturated edges subtracted from one another overflow. Any
/// window is orders of magnitude inside this, and the box is intersected with
/// the viewport straight afterwards.
const CLIP_COORDINATE_LIMIT: f32 = (1 << 24) as f32;

/// The whole device pixels a device-space rectangle touches.
fn covering_rect(bounds: &Bounds<ScaledPixels>) -> DeviceRect {
    let clamp = |value: f32| value.clamp(-CLIP_COORDINATE_LIMIT, CLIP_COORDINATE_LIMIT);
    let x = clamp(bounds.origin.x.0).floor() as i32;
    let y = clamp(bounds.origin.y.0).floor() as i32;
    let right = clamp(bounds.origin.x.0 + bounds.size.width.0).ceil() as i32;
    let bottom = clamp(bounds.origin.y.0 + bounds.size.height.0).ceil() as i32;
    DeviceRect {
        x,
        y,
        width: right - x,
        height: bottom - y,
    }
}

/// Turns a clip path's contours into polylines in device space.
///
/// Flattening the curves outright, rather than handing quadratics to a
/// Loop-Blinn fragment test, is what lets the stencil pass be nothing but
/// triangles: the coverage then comes from multisampling alone and is the same
/// on a curve as on a straight edge, with no per-sample shading to arrange.
fn flatten_clip_path(path: &ClipPath<ScaledPixels>) -> Vec<Vec<PointF>> {
    let mut contours: Vec<Vec<PointF>> = Vec::new();
    let mut contour: Vec<PointF> = Vec::new();
    let mut start = point(0., 0.);
    let mut at = point(0., 0.);

    fn finish(contours: &mut Vec<Vec<PointF>>, contour: Vec<PointF>) {
        // Fewer than three points enclose no area, so they contribute no
        // winding and would only cost a degenerate triangle.
        if contour.len() >= 3 {
            contours.push(contour);
        }
    }

    for segment in path.segments() {
        match segment {
            ClipPathSegment::MoveTo(to) => {
                finish(&mut contours, mem::take(&mut contour));
                at = device_point(to);
                start = at;
                contour.push(at);
            }
            ClipPathSegment::LineTo(to) => {
                at = device_point(to);
                contour.push(at);
            }
            ClipPathSegment::QuadTo { control, to } => {
                let (control, to) = (device_point(control), device_point(to));
                flatten_quadratic(at, control, to, &mut contour);
                at = to;
            }
            ClipPathSegment::CubicTo {
                control1,
                control2,
                to,
            } => {
                let control1 = device_point(control1);
                let control2 = device_point(control2);
                let to = device_point(to);
                flatten_cubic(at, control1, control2, to, &mut contour);
                at = to;
            }
            ClipPathSegment::Close => {
                finish(&mut contours, mem::take(&mut contour));
                // A fill closes every contour anyway, so `Close` only says
                // where anything drawn after it carries on from.
                at = start;
                contour.push(start);
            }
        }
    }
    finish(&mut contours, contour);
    contours
}

fn device_point(at: &Point<ScaledPixels>) -> PointF {
    point(at.x.0, at.y.0)
}

/// How many line segments a curve whose worst-case deviation from its chord is
/// `deviation` needs to stay inside [`CLIP_FLATTEN_TOLERANCE`]. Halving the
/// step quarters the error, so the count goes as its square root.
fn flatten_steps(deviation: f32) -> u32 {
    if !deviation.is_finite() || deviation <= CLIP_FLATTEN_TOLERANCE {
        return 1;
    }
    ((deviation / CLIP_FLATTEN_TOLERANCE).sqrt().ceil() as u32).clamp(1, 256)
}

fn flatten_quadratic(from: PointF, control: PointF, to: PointF, out: &mut Vec<PointF>) {
    let deviation = (from.x - 2. * control.x + to.x).hypot(from.y - 2. * control.y + to.y) / 4.;
    let steps = flatten_steps(deviation);
    for step in 1..=steps {
        let t = step as f32 / steps as f32;
        let inverse = 1. - t;
        out.push(point(
            inverse * inverse * from.x + 2. * inverse * t * control.x + t * t * to.x,
            inverse * inverse * from.y + 2. * inverse * t * control.y + t * t * to.y,
        ));
    }
}

fn flatten_cubic(
    from: PointF,
    control1: PointF,
    control2: PointF,
    to: PointF,
    out: &mut Vec<PointF>,
) {
    let first =
        (from.x - 2. * control1.x + control2.x).hypot(from.y - 2. * control1.y + control2.y);
    let second = (control1.x - 2. * control2.x + to.x).hypot(control1.y - 2. * control2.y + to.y);
    let steps = flatten_steps(0.75 * first.max(second));
    for step in 1..=steps {
        let t = step as f32 / steps as f32;
        let inverse = 1. - t;
        let (a, b, c, d) = (
            inverse * inverse * inverse,
            3. * inverse * inverse * t,
            3. * inverse * t * t,
            t * t * t,
        );
        out.push(point(
            a * from.x + b * control1.x + c * control2.x + d * to.x,
            a * from.y + b * control1.y + c * control2.y + d * to.y,
        ));
    }
}

/// The specialization the group composite pipelines take: which operator, and
/// whether the group carried a clip path the composite has to be cut by.
fn group_composite_constants(
    compose: GroupComposeMode,
    clipped: bool,
) -> metal::FunctionConstantValues {
    let constants = clip_constants(clipped);
    let mode = compose as u32;
    constants.set_constant_value_at_index(
        &mode as *const u32 as *const c_void,
        metal::MTLDataType::UInt,
        1,
    );
    constants
}

/// The specialization that picks between the clipped and unclipped variant of a
/// shader. Every function that mentions `CLIPPED` must be given one, and one
/// given to a function that does not is ignored.
fn clip_constants(clipped: bool) -> metal::FunctionConstantValues {
    let constants = metal::FunctionConstantValues::new();
    constants.set_constant_value_at_index(
        &clipped as *const bool as *const c_void,
        metal::MTLDataType::Bool,
        0,
    );
    constants
}

/// The two halves of stencil-and-cover. Neither blends: the stencil half writes
/// no colour at all, and the cover half replaces the tile's coverage outright.
fn build_clip_mask_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    writes_coverage: bool,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    descriptor.set_raster_sample_count(PATH_SAMPLE_COUNT as u64);
    descriptor.set_stencil_attachment_pixel_format(MTLPixelFormat::Stencil8);
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(MTLPixelFormat::R8Unorm);
    color_attachment.set_blending_enabled(false);
    color_attachment.set_write_mask(if writes_coverage {
        metal::MTLColorWriteMask::all()
    } else {
        metal::MTLColorWriteMask::empty()
    });

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create clip mask render pipeline state")
}

/// Counts winding into the stencil buffer.
///
/// The nonzero rule counts up on one facing and down on the other, so a
/// self-overlapping contour and the interior edges of a triangle fan cancel
/// exactly. The even-odd rule flips a single bit instead, which is the same
/// count taken modulo two.
fn build_stencil_state(device: &metal::DeviceRef, fill_rule: FillRule) -> metal::DepthStencilState {
    let descriptor = metal::DepthStencilDescriptor::new();
    let face = |operation| {
        let face = metal::StencilDescriptor::new();
        face.set_stencil_compare_function(metal::MTLCompareFunction::Always);
        face.set_stencil_failure_operation(metal::MTLStencilOperation::Keep);
        face.set_depth_failure_operation(metal::MTLStencilOperation::Keep);
        face.set_depth_stencil_pass_operation(operation);
        face.set_read_mask(0);
        face.set_write_mask(match fill_rule {
            FillRule::EvenOdd => EVEN_ODD_STENCIL_MASK,
            FillRule::NonZero => u32::MAX,
        });
        face
    };
    match fill_rule {
        FillRule::EvenOdd => {
            descriptor.set_front_face_stencil(Some(&face(metal::MTLStencilOperation::Invert)));
            descriptor.set_back_face_stencil(Some(&face(metal::MTLStencilOperation::Invert)));
        }
        FillRule::NonZero => {
            descriptor
                .set_front_face_stencil(Some(&face(metal::MTLStencilOperation::IncrementWrap)));
            descriptor
                .set_back_face_stencil(Some(&face(metal::MTLStencilOperation::DecrementWrap)));
        }
    }
    device.new_depth_stencil_state(&descriptor)
}

/// Draws the tile wherever the winding says the path covers it, and zeroes the
/// stencil behind itself so the next clip in the same pass starts clean.
fn build_cover_state(device: &metal::DeviceRef, fill_rule: FillRule) -> metal::DepthStencilState {
    let mask = match fill_rule {
        FillRule::EvenOdd => EVEN_ODD_STENCIL_MASK,
        FillRule::NonZero => u32::MAX,
    };
    let descriptor = metal::DepthStencilDescriptor::new();
    let face = || {
        let face = metal::StencilDescriptor::new();
        // The reference value is left at zero, so this is "the winding is not
        // zero" under either rule.
        face.set_stencil_compare_function(metal::MTLCompareFunction::NotEqual);
        face.set_stencil_failure_operation(metal::MTLStencilOperation::Keep);
        face.set_depth_failure_operation(metal::MTLStencilOperation::Keep);
        face.set_depth_stencil_pass_operation(metal::MTLStencilOperation::Zero);
        face.set_read_mask(mask);
        face.set_write_mask(mask);
        face
    };
    descriptor.set_front_face_stencil(Some(&face()));
    descriptor.set_back_face_stencil(Some(&face()));
    device.new_depth_stencil_state(&descriptor)
}

#[cfg(test)]
mod vector_tests {
    use super::*;
    use gpui::px;

    fn viewport(width: i32, height: i32) -> Size<DevicePixels> {
        size(DevicePixels(width), DevicePixels(height))
    }

    fn device_bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(width), ScaledPixels(height)),
        }
    }

    /// A rectangular clip path, already in the device pixels a scene speaks.
    fn box_path(x: f32, y: f32, width: f32, height: f32) -> ClipPath<ScaledPixels> {
        let at = |x: f32, y: f32| point(px(x), px(y));
        ClipPath::builder()
            .move_to(at(x, y))
            .line_to(at(x + width, y))
            .line_to(at(x + width, y + height))
            .line_to(at(x, y + height))
            .close()
            .build()
            .scale(1.0)
    }

    /// Pushes `path` at the root, reachable over exactly its own bounds.
    fn push_root_clip(scene: &mut Scene, path: ClipPath<ScaledPixels>) {
        let visible = path.bounds();
        scene.push_clip(path, visible);
        scene.pop_clip();
    }

    /// What `clip_mask_alpha` in `shaders.metal` computes at a device point,
    /// for the two answers that need no atlas: nothing outside the clip's tile,
    /// and everything inside the tile of a clip that fell back to its
    /// rectangle. `None` where the answer is a texel of the atlas.
    fn coverage_without_the_atlas(plan: &ClipPlan, clip: usize, x: f32, y: f32) -> Option<f32> {
        let tile = plan.masks[clip + 1].tile;
        if x < tile.origin.x.0
            || y < tile.origin.y.0
            || x >= tile.origin.x.0 + tile.size.width.0
            || y >= tile.origin.y.0 + tile.size.height.0
        {
            return Some(0.);
        }
        (plan.masks[clip + 1].sampled == 0).then_some(1.)
    }

    #[test]
    fn a_clip_path_that_encloses_no_area_lets_nothing_through() {
        // A lone `move_to` and a single line both flatten to no contour at all,
        // and a contour is what a mask is rasterized from. Their bounding boxes
        // are not empty, so a clip that answered "no mask, use the rectangle"
        // here would show everything inside the box instead of hiding it.
        for path in [
            ClipPath::builder()
                .move_to(point(px(10.), px(10.)))
                .line_to(point(px(100.), px(100.)))
                .build()
                .scale(1.0),
            ClipPath::builder()
                .move_to(point(px(10.), px(10.)))
                .build()
                .scale(1.0),
        ] {
            let mut scene = Scene::default();
            let visible = device_bounds(10., 10., 90., 90.);
            scene.push_clip(path, visible);
            scene.pop_clip();

            let plan = ClipPlan::new(&scene, viewport(200, 200), Extent::ZERO);
            assert!(
                plan.clips[0].is_none(),
                "a path enclosing no area has nothing to rasterize"
            );
            for (x, y) in [(50., 50.), (10., 10.), (99., 99.)] {
                assert_eq!(
                    coverage_without_the_atlas(&plan, 0, x, y),
                    Some(0.),
                    "({x}, {y}) is inside the bounding box of a path that encloses no area, so \
                     the clip has to hide it"
                );
            }
        }
    }

    #[test]
    fn a_clip_path_past_the_depth_limit_keeps_its_rectangle() {
        // The control for the test above: the other way a clip ends up with no
        // tile is a fallback, and a fallback keeps everything inside its
        // rectangle rather than hiding it.
        let mut scene = Scene::default();
        for _ in 0..=MAX_CLIP_DEPTH {
            let path = box_path(10., 10., 100., 100.);
            let visible = path.bounds();
            scene.push_clip(path, visible);
        }
        let plan = ClipPlan::new(&scene, viewport(200, 200), Extent::ZERO);

        let deepest = MAX_CLIP_DEPTH;
        assert!(
            plan.clips[deepest].is_none(),
            "the depth limit refuses a mask"
        );
        assert_eq!(
            coverage_without_the_atlas(&plan, deepest, 50., 50.),
            Some(1.),
            "a clip that fell back to its rectangle keeps what is inside it"
        );
        assert_eq!(
            coverage_without_the_atlas(&plan, deepest, 5., 5.),
            Some(0.),
            "a clip that fell back to its rectangle still enforces the rectangle"
        );
    }

    #[test]
    fn a_tile_is_the_part_of_the_window_the_clip_can_reach() {
        // The shape is far larger than the container it was pushed inside, as
        // a rounded container in a scrolled list is. Sizing the tile from the
        // shape's own bounding box would ask for a 4096x4096 atlas - 33.6 MB of
        // textures on Apple silicon, 167.8 MB on Intel - to mask a hundred
        // device pixels.
        let mut scene = Scene::default();
        scene.push_clip(
            box_path(0., 0., 4000., 4000.),
            device_bounds(0., 0., 100., 100.),
        );
        scene.pop_clip();

        let plan = ClipPlan::new(&scene, viewport(4000, 4000), Extent::ZERO);
        let tile = plan.masks[1].tile;
        assert_eq!(
            (tile.size.width.0, tile.size.height.0),
            (100., 100.),
            "the tile is the content mask the clip was pushed under, not the whole shape"
        );
        assert_eq!(
            plan.minimum_atlas,
            Extent {
                width: 256,
                height: 256
            }
        );
        assert_eq!(
            plan.work,
            Extent {
                width: 256,
                height: 256
            }
        );
    }

    #[test]
    fn a_tile_covers_every_device_pixel_a_fractional_shape_touches() {
        // A scale factor other than 2 puts a clip at fractional device
        // coordinates - at 1.5 a 51-logical-pixel edge lands on 76.5 - and a
        // tile is whole pixels. Rounding the box inwards would shave the last
        // column off the mask and leave a hairline of unclipped content down
        // the edge of every clip on such a display.
        let at = |x: f32, y: f32| point(px(x), px(y));
        let path = ClipPath::builder()
            .move_to(at(10., 20.))
            .line_to(at(51., 20.))
            .line_to(at(51., 61.))
            .close()
            .build()
            .scale(1.5);
        let mut scene = Scene::default();
        let visible = path.bounds();
        scene.push_clip(path, visible);
        scene.pop_clip();

        let plan = ClipPlan::new(&scene, viewport(200, 200), Extent::ZERO);
        let tile = plan.masks[1].tile;
        assert_eq!(
            (
                tile.origin.x.0,
                tile.origin.y.0,
                tile.size.width.0,
                tile.size.height.0
            ),
            (15., 30., 62., 62.),
            "the tile has to cover 15..76.5 and 30..91.5 in whole device pixels"
        );
    }

    #[test]
    fn a_clip_wider_than_the_packing_cap_still_gets_a_mask() {
        // A full-width element on a 5K display is 5120 device pixels across,
        // which no 4096-square atlas can hold. Falling back to the rectangle
        // there would drop the shape of every full-width clip on mainstream
        // hardware, every frame.
        let mut scene = Scene::default();
        push_root_clip(&mut scene, box_path(0., 0., 5120., 400.));

        let plan = ClipPlan::new(&scene, viewport(5120, 2880), Extent::ZERO);
        assert!(
            plan.clips[0].is_some(),
            "a 5120-wide clip has to be rasterized, not reduced to its rectangle"
        );
        assert!(
            plan.atlas.width >= 5120,
            "the atlas has to be wide enough for the tile, found {:?}",
            plan.atlas
        );
        assert_eq!(
            coverage_without_the_atlas(&plan, 0, 100., 100.),
            None,
            "inside the tile the coverage is a texel of the atlas, not a fallback"
        );
    }

    #[test]
    fn tiles_are_packed_tallest_first() {
        // Twenty tiles alternating tall and short. A shelf holds two of them,
        // and an unsorted shelf is as tall as its tallest tile, so alternating
        // wastes 240 of every 250 rows on half the shelves and needs twice the
        // atlas its tile area calls for.
        let mut scene = Scene::default();
        for index in 0..20 {
            let height = if index % 2 == 0 { 250. } else { 10. };
            push_root_clip(&mut scene, box_path(0., 0., 128., height));
        }

        let plan = ClipPlan::new(&scene, viewport(4096, 4096), Extent::ZERO);
        let tiles = 10 * 128 * 250 + 10 * 128 * 10;
        let atlas = plan.minimum_atlas.width * plan.minimum_atlas.height;
        assert!(
            atlas <= 2 * tiles,
            "{tiles} device pixels of tile needed a {:?} atlas, which is more than twice their \
             area: the shelves are being wasted",
            plan.minimum_atlas
        );
    }

    #[test]
    fn a_working_attachment_is_sized_to_its_level_not_to_the_atlas() {
        // The atlas is already large, from a frame that needed it. A multisample
        // resolve covers its whole attachment, so a level rendered at atlas size
        // would rewrite 16.8 MB to mask one 40x40 clip.
        let mut scene = Scene::default();
        push_root_clip(&mut scene, box_path(0., 0., 40., 40.));

        let large = Extent {
            width: 4096,
            height: 4096,
        };
        let plan = ClipPlan::new(&scene, viewport(4096, 4096), large);
        assert_eq!(plan.atlas, large, "the atlas keeps the size it was given");
        assert_eq!(
            plan.work,
            Extent {
                width: 256,
                height: 256
            },
            "the pass renders only the level's own tiles"
        );
        assert_eq!(
            plan.minimum_atlas,
            Extent {
                width: 256,
                height: 256
            },
            "the budget is told how little would have been enough"
        );
    }

    #[test]
    fn a_nested_clip_reads_its_parents_tile_where_the_parent_actually_is() {
        // `parent_offset` is what makes nesting intersect: the cover pass reads
        // the parent's coverage while writing the child's tile, and the two live
        // at unrelated places in the atlas. Every parent being a rectangle hides
        // an error here, because a rectangle's coverage is 1 everywhere the
        // child can see.
        let mut scene = Scene::default();
        let outer = box_path(20., 30., 300., 200.);
        let outer_visible = outer.bounds();
        scene.push_clip(outer, outer_visible);
        let inner = box_path(40., 60., 60., 50.);
        let inner_visible = inner.bounds();
        scene.push_clip(inner, inner_visible);
        scene.pop_clip();
        scene.pop_clip();

        let plan = ClipPlan::new(&scene, viewport(400, 400), Extent::ZERO);
        assert_eq!(plan.levels.len(), 2, "the two clips are two render passes");
        let parent = plan.clips[0].as_ref().expect("the outer clip wants a tile");
        let child = plan.clips[1].as_ref().expect("the inner clip wants a tile");
        assert_eq!(child.cover.has_parent, 1);

        // A device point inside the child's tile, carried into the working
        // attachment the child's level is drawn through, plus `parent_offset`,
        // has to land on that same device point inside the parent's atlas tile.
        for (x, y) in [(45., 65.), (95., 105.), (70., 80.)] {
            let child_tile = plan.masks[2].tile;
            let parent_tile = plan.masks[1].tile;
            let in_work = (
                x - child_tile.origin.x.0 + child.work.x as f32,
                y - child_tile.origin.y.0 + child.work.y as f32,
            );
            let in_parent_atlas = (
                x - parent_tile.origin.x.0 + parent.atlas.x as f32,
                y - parent_tile.origin.y.0 + parent.atlas.y as f32,
            );
            assert_eq!(
                (
                    in_work.0 + child.cover.parent_offset.x,
                    in_work.1 + child.cover.parent_offset.y
                ),
                in_parent_atlas,
                "the child's cover pass reads the wrong texel of its parent at ({x}, {y})"
            );
        }
    }

    #[test]
    fn a_level_covers_only_its_own_tiles() {
        // Levels are packed into bands of their own, so the rectangle each pass
        // renders and resolves is tight around that level's tiles rather than
        // spanning whichever other tiles happened to land between them.
        let mut scene = Scene::default();
        for _ in 0..4 {
            let path = box_path(0., 0., 100., 100.);
            let visible = path.bounds();
            scene.push_clip(path, visible);
        }
        let plan = ClipPlan::new(&scene, viewport(400, 400), Extent::ZERO);

        assert_eq!(plan.levels.len(), 4);
        for level in &plan.levels {
            assert_eq!(level.clips.len(), 1);
            assert_eq!(
                level.extent.extent(),
                Extent {
                    width: 100,
                    height: 100
                }
            );
        }
        assert_eq!(
            plan.work,
            Extent {
                width: 256,
                height: 256
            }
        );
    }

    #[test]
    fn the_budget_grows_at_once_and_shrinks_only_after_a_run_of_small_frames() {
        let mut budget = ClipTextureBudget::default();
        let small = Extent {
            width: 256,
            height: 256,
        };
        let large = Extent {
            width: 4096,
            height: 2048,
        };

        assert_eq!(budget.floor(), Extent::ZERO);
        budget.observe(large, large);
        assert_eq!(
            budget.floor(),
            large,
            "a frame that needed it all keeps it all"
        );

        for frame in 0..CLIP_TEXTURE_SHRINK_FRAMES {
            let floor = budget.floor();
            assert_eq!(floor, large, "frame {frame} gave the memory back too early");
            budget.observe(floor, small);
        }
        assert_eq!(
            budget.floor(),
            small,
            "a long enough run of small frames has to hand the textures back"
        );
    }

    #[test]
    fn one_busy_frame_resets_the_shrink_countdown() {
        let mut budget = ClipTextureBudget::default();
        let small = Extent {
            width: 256,
            height: 256,
        };
        let large = Extent {
            width: 4096,
            height: 2048,
        };
        budget.observe(large, large);

        for _ in 0..CLIP_TEXTURE_SHRINK_FRAMES - 1 {
            let floor = budget.floor();
            budget.observe(floor, small);
        }
        let floor = budget.floor();
        budget.observe(floor, large);
        for _ in 0..CLIP_TEXTURE_SHRINK_FRAMES - 1 {
            let floor = budget.floor();
            assert_eq!(floor, large, "the countdown has to start again");
            budget.observe(floor, small);
        }
    }
}
