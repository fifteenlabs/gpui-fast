//! Isolated groups: the targets they are drawn into, the filters run over
//! them, and the composite that puts each back onto what is underneath it.
use super::{
    MTLBlendFactor, MTLBlendOperation, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder,
    MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary, MTLLoadAction, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLStorageMode, MTLStoreAction,
    MTLTexture, MTLTextureDescriptor, MTLTextureUsage, MTLViewport, MetalRenderer, PointF,
    ProtocolObject, RenderTarget, Result, Retained, clip_constants, mem, ptr, shader_function,
};
use super::vector_frame::ActiveTarget;
use anyhow::Context as _;
use gpui::{
    Bounds, ClipId, ComposeMode, DevicePixels, DeviceRect, Extent, FilterOp, GroupFilter,
    GroupSpec, Hsla, MixMode, ScaledPixels, Scene, SceneFilter, Size, TextureBudget, point, size,
};
use objc2_foundation::NSString;
use objc2_metal::{MTLBlitCommandEncoder, MTLOrigin, MTLSize};

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

#[repr(C)]
pub(crate) enum GroupCompositeInputIndex {
    /// Buffer slots after the globals every pass binds: the render target the
    /// group is composited onto, and the clip masks.
    Composite = 2,
    Filter = 3,
    /// Texture slots.
    GroupTexture = 4,
    /// The copy of what was underneath the group, and the filtered copy beside
    /// it. Bound, and declared in the shader, only for the `HAS_BACKDROP`
    /// variant.
    BackdropTexture = 5,
    FilteredBackdropTexture = 6,
}

/// What one filter pass reads. Buffers and textures index separately in Metal,
/// so a buffer and a texture naming the same slot is not a collision.
#[repr(C)]
pub(crate) enum FilterInputIndex {
    Pass = 1,
    /// The colour matrices, declared by `filter_matrix_fragment` alone.
    Filter = 2,
    Source = 3,
    /// The blurred alpha a drop shadow draws behind its source, declared by
    /// `filter_drop_shadow_fragment` alone.
    Blurred = 4,
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
pub(crate) enum GroupComposeMode {
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


/// Everything compositing one group needs beyond the group itself.
pub(super) struct PreparedGroup {
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
pub(crate) enum FilterPipeline {
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
pub(super) struct OpenGroup {
    spec: GroupSpec,
    /// The whole device pixels the group covers: the composite's quad, and the
    /// region of the target that holds it.
    bounds: Bounds<ScaledPixels>,
    /// The pooled texture's own size, which the composite needs to turn a
    /// position in `bounds` into a texture coordinate.
    texture_size: Size<DevicePixels>,
    pub(super) target: ActiveTarget,
    /// What the group is composited back onto.
    pub(super) parent: ActiveTarget,
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
pub(crate) struct FilterScratch {
    targets: GroupTargets,
    in_use: [bool; MAX_FILTER_SCRATCH],
}

impl FilterScratch {
    pub(super) fn begin_frame(&mut self) {
        self.targets.begin_frame();
        self.in_use = [false; MAX_FILTER_SCRATCH];
    }

    pub(super) fn end_frame(&mut self) {
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
pub(crate) struct GroupTargets {
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
    pub(super) fn begin_frame(&mut self) {
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

    pub(super) fn end_frame(&mut self) {
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
pub(super) fn scene_has_a_backdrop_filter(scene: &Scene) -> bool {
    scene
        .groups
        .iter()
        .any(|group| group.backdrop_filter.is_some())
}

impl MetalRenderer {
    /// Give a group about to be pushed a render target of its own, or `None`
    /// when it has to be painted straight onto what is underneath it.
    ///
    /// Falling back is a wrong picture - two overlapping children at 0.5 come
    /// out at 0.75 where they overlap, a `DestIn` group paints instead of
    /// masking - so every reason to fall back says so once per frame rather
    /// than quietly drawing something else.
    pub(super) fn open_group(
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
    pub(super) fn prepare_group(
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
        let encoder = begin_filter_pass(command_buffer, destination)?;
        encoder.setRenderPipelineState(pipeline);
        // SAFETY: The slots match the fragment function the pipeline was built
        // with; bytes are copied from live values, and Metal retains the bound
        // textures.
        unsafe {
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
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::TriangleStrip, 0, 4);
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
    pub(super) fn release_prepared(&mut self, prepared: PreparedGroup) {
        self.filter_scratch.release(&prepared.source);
        if let Some(backdrop) = &prepared.backdrop {
            self.filter_scratch.release(&backdrop.original);
            self.filter_scratch.release(&backdrop.filtered);
        }
    }

    /// Draw a finished group's target onto the target underneath it: one quad
    /// over the group's bounds, blended with the operator the group asked for.
    pub(super) fn composite_group(
        &mut self,
        group: &OpenGroup,
        prepared: &PreparedGroup,
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
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&composite).cast(),
                mem::size_of::<GroupComposite>(),
                GroupCompositeInputIndex::Composite as usize,
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
        // SAFETY: A four-vertex strip draws the one composite quad. The pass
        // has already bound the parent's render target and the clip masks.
        unsafe {
            command_encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
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
}

/// Begins a filter pass: a whole-image quad over `destination`, cleared first.
/// The pass has no depth attachment, and the filter pipelines are built for
/// none.
fn begin_filter_pass(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    destination: &FilterImage,
) -> Result<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>> {
    let render_target = destination.render_target();
    let render_pass_descriptor = MTLRenderPassDescriptor::new();
    // SAFETY: Index zero is within the render pass's color attachment array.
    let color_attachment = unsafe {
        render_pass_descriptor
            .colorAttachments()
            .objectAtIndexedSubscript(0)
    };
    color_attachment.setTexture(Some(&destination.texture));
    color_attachment.setStoreAction(MTLStoreAction::Store);
    color_attachment.setLoadAction(MTLLoadAction::Clear);
    color_attachment.setClearColor(MTLClearColor {
        red: 0.,
        green: 0.,
        blue: 0.,
        alpha: 0.,
    });
    let encoder = command_buffer
        .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
        .context("creating filter render command encoder")?;
    encoder.setViewport(MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(render_target.size.width) as f64,
        height: i32::from(render_target.size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    Ok(encoder)
}

/// The pipeline that draws one finished group onto what is underneath it.
///
/// A group target holds premultiplied RGBA, so this blends a premultiplied
/// source rather than [`build_pipeline_state`]'s straight alpha: the primitive
/// pipelines emit `S.rgb` with the coverage in `S.a` and cannot express a
/// premultiplied source at all.
/// Which Porter-Duff operator it is, is these factors together with what
/// `group_composite_fragment` emits, and the two are written to agree.
pub(super) fn build_group_composite_pipeline_state(
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
    // Composited inside a pass of the frame, which carries a depth attachment
    // for the primitive pipelines.
    descriptor.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);

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

