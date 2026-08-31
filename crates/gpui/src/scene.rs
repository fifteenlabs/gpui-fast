// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ClipPath, ContentMask, Corners, Edges, Hsla,
    Pixels, Point, Radians, ScaledPixels, Size, bounds_tree::BoundsTree, point,
};
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
};

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

#[expect(missing_docs)]
pub type DrawOrder = u32;

/// A boolean stored as a `u32` so that GPU-facing structs contain no
/// compiler-inserted padding bytes, which would be undefined behavior to
/// reinterpret as `&[u8]` when writing instance buffers. Guaranteed to be
/// `0` or `1` by construction; shaders read it as a `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PaddedBool32(u32);

impl From<bool> for PaddedBool32 {
    fn from(value: bool) -> Self {
        PaddedBool32(value as u32)
    }
}

/// The clip path a primitive is clipped to: an index into [`Scene::clips`],
/// plus one.
///
/// Zero — [`ClipId::NONE`] — means the primitive is clipped by its
/// [`ContentMask`] alone, which is every primitive gpui's own UI paints. A
/// clip path is strictly opt-in, so carrying this id costs an unclipped scene
/// one word that was padding before and nothing else.
///
/// Every GPU-facing primitive record carries one, so it is `u32`-shaped on
/// purpose: the shaders read it as a plain `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ClipId(pub u32);

impl ClipId {
    /// The id of a primitive that no clip path applies to.
    pub const NONE: ClipId = ClipId(0);

    /// Whether no clip path applies.
    #[inline]
    pub fn is_none(self) -> bool {
        self.0 == ClipId::NONE.0
    }

    /// Whether some clip path applies.
    ///
    /// This, rather than the id itself, is what breaks a batch: see
    /// [`Scene::batches`].
    #[inline]
    pub fn is_clipped(self) -> bool {
        !self.is_none()
    }

    /// The index of this clip in [`Scene::clips`], or `None` for
    /// [`ClipId::NONE`].
    #[inline]
    pub fn index(self) -> Option<usize> {
        (self.0 as usize).checked_sub(1)
    }
}

/// A clip path registered in a [`Scene`], together with the clip it was pushed
/// inside.
///
/// Clips nest, and the intersection of two arbitrary paths is not a path that
/// can be written down cheaply, so a nested clip keeps a pointer to its parent
/// rather than being flattened into it. A rasterizer walks the chain to the
/// root and intersects the masks; a clip whose `parent` is [`ClipId::NONE`] is
/// the whole shape by itself.
#[derive(Clone, Debug)]
pub struct SceneClip {
    /// The shape, in the device pixels the scene speaks.
    pub path: ClipPath<ScaledPixels>,
    /// The clip this one was pushed inside.
    pub parent: ClipId,
    /// The part of the window a primitive inside this clip can reach: the
    /// content mask in force when it was pushed, which
    /// [`Window::with_clip_path`](crate::Window::with_clip_path) has already
    /// tightened to the path's bounding box.
    ///
    /// A rasterizer that gives each clip a tile of a mask atlas sizes the tile
    /// from this rather than from the path's own box, so a clip inside a
    /// narrow or scrolled container costs a tile of what is on screen instead
    /// of a tile of the whole shape.
    pub visible: Bounds<ScaledPixels>,
}

/// An isolated group registered in a [`Scene`]: an index into
/// [`Scene::groups`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct GroupId(pub u32);

/// A run of primitives registered in a [`Scene`]: an index into
/// [`Scene::segments`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct SegmentId(pub u32);

/// The primitives painted between two group boundaries: one contiguous range
/// of each per-kind array.
///
/// Eight ranges are enough to name a subtree because
/// [`Scene::insert_primitive`] appends to those arrays and nothing else does,
/// so primitives land in paint order, and a group is closure-scoped.
/// Everything painted between a [`Scene::push_group`] and its matching
/// [`Scene::pop_group`] therefore occupies a contiguous range of every array,
/// before any sorting - which is why isolating a subtree needs no second
/// scene, no re-batching, and no change to [`Scene::insert_primitive`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[expect(missing_docs)]
pub struct Segment {
    pub shadows: Range<usize>,
    pub quads: Range<usize>,
    pub paths: Range<usize>,
    pub underlines: Range<usize>,
    pub monochrome_sprites: Range<usize>,
    pub subpixel_sprites: Range<usize>,
    pub polychrome_sprites: Range<usize>,
    pub surfaces: Range<usize>,
}

impl Segment {
    /// Whether this run holds no primitives at all.
    pub fn is_empty(&self) -> bool {
        self.shadows.is_empty()
            && self.quads.is_empty()
            && self.paths.is_empty()
            && self.underlines.is_empty()
            && self.monochrome_sprites.is_empty()
            && self.subpixel_sprites.is_empty()
            && self.polychrome_sprites.is_empty()
            && self.surfaces.is_empty()
    }

    /// How many primitives this run holds, across every kind.
    pub fn len(&self) -> usize {
        self.shadows.len()
            + self.quads.len()
            + self.paths.len()
            + self.underlines.len()
            + self.monochrome_sprites.len()
            + self.subpixel_sprites.len()
            + self.polychrome_sprites.len()
            + self.surfaces.len()
    }
}

/// The blend function a group is composited with, mirroring `peniko::Mix`
/// value for value so a painter that speaks peniko converts with a plain
/// `match` and no lookup table.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
#[expect(missing_docs)]
pub enum MixMode {
    #[default]
    Normal = 0,
    Multiply = 1,
    Screen = 2,
    Overlay = 3,
    Darken = 4,
    Lighten = 5,
    ColorDodge = 6,
    ColorBurn = 7,
    HardLight = 8,
    SoftLight = 9,
    Difference = 10,
    Exclusion = 11,
    Hue = 12,
    Saturation = 13,
    Color = 14,
    Luminosity = 15,
    /// Not a blend function: keep the destination's alpha and take the
    /// source's color, which is how a clip is expressed in this model.
    Clip = 128,
}

/// The Porter-Duff operator a group is composited with, mirroring
/// `peniko::Compose` value for value.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
#[expect(missing_docs)]
pub enum ComposeMode {
    Clear = 0,
    Copy = 1,
    Dest = 2,
    #[default]
    SrcOver = 3,
    DestOver = 4,
    SrcIn = 5,
    /// What `mask-image` compiles to: keep the destination where the source
    /// is opaque.
    DestIn = 6,
    SrcOut = 7,
    /// What an inset box shadow compiles to: keep the destination where the
    /// source is *not* opaque.
    DestOut = 8,
    SrcAtop = 9,
    DestAtop = 10,
    Xor = 11,
    Plus = 12,
    PlusLighter = 13,
}

/// How a group's rendered result is combined with what is already on the
/// target underneath it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct BlendMode {
    pub mix: MixMode,
    pub compose: ComposeMode,
}

impl BlendMode {
    /// The mode that changes nothing: paint the group over what is behind it.
    pub const NORMAL: Self = Self {
        mix: MixMode::Normal,
        compose: ComposeMode::SrcOver,
    };

    /// Whether this is [`BlendMode::NORMAL`], and so needs no offscreen target
    /// of its own to be composited correctly.
    pub fn is_normal(self) -> bool {
        self == Self::NORMAL
    }
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// A pixel filter applied to a group's rendered result before it is
/// composited, covering the CSS `filter` functions that are not expressible as
/// a change to the primitives themselves.
///
/// Lengths are in whatever pixel space the value is held in: logical pixels in
/// [`crate::GroupOptions`], device pixels in [`GroupSpec`], which
/// [`crate::Window::with_isolated_group`] converts between with
/// [`SceneFilter::scale`].
#[derive(Clone, Debug, PartialEq)]
pub enum SceneFilter {
    /// An `feColorMatrix`: a row-major 4x5 matrix mapping RGBA plus a
    /// constant column to RGBA. `grayscale`, `sepia`, `saturate`,
    /// `hue-rotate` and `invert` all compile to one of these.
    ///
    /// It acts on *non*-premultiplied colour, as the CSS and SVG filter
    /// specifications define it, so a renderer applying one to a premultiplied
    /// target has to undo the premultiplication first and put it back
    /// afterwards. A chain of them is applied one at a time with a clamp
    /// between stages, which is not the same as their product: `brightness(2)`
    /// followed by `invert(1)` diverges from the collapsed matrix above 0.5.
    ColorMatrix([f32; 20]),
    /// A separable gaussian blur.
    Blur {
        /// Standard deviation along x.
        radius_x: f32,
        /// Standard deviation along y.
        radius_y: f32,
    },
    /// A `drop-shadow()`: the group's alpha, blurred, offset and tinted, drawn
    /// behind the group itself.
    DropShadow {
        /// How far the shadow is offset along x.
        offset_x: f32,
        /// How far the shadow is offset along y.
        offset_y: f32,
        /// The standard deviation of the shadow's blur.
        radius: f32,
        /// The shadow's color.
        color: Hsla,
    },
    /// Several filters applied in order, first to last.
    Chain(Vec<SceneFilter>),
}

impl SceneFilter {
    /// This filter with every length multiplied by `factor`, which is how a
    /// filter written in logical pixels reaches the scene in device pixels.
    /// A color matrix has no lengths in it and is unchanged.
    pub fn scale(&self, factor: f32) -> SceneFilter {
        match self {
            SceneFilter::ColorMatrix(matrix) => SceneFilter::ColorMatrix(*matrix),
            SceneFilter::Blur { radius_x, radius_y } => SceneFilter::Blur {
                radius_x: radius_x * factor,
                radius_y: radius_y * factor,
            },
            SceneFilter::DropShadow {
                offset_x,
                offset_y,
                radius,
                color,
            } => SceneFilter::DropShadow {
                offset_x: offset_x * factor,
                offset_y: offset_y * factor,
                radius: radius * factor,
                color: *color,
            },
            SceneFilter::Chain(filters) => {
                SceneFilter::Chain(filters.iter().map(|f| f.scale(factor)).collect())
            }
        }
    }

    /// Where this filter's result can reach, given an input that paints nothing
    /// outside `input`.
    ///
    /// A blurred group is larger than the group: the gaussian carries the
    /// content's alpha out to [`GAUSSIAN_BLUR_EXTENT`] standard deviations
    /// beyond the ink it was measured from, and a drop shadow carries it out
    /// there *and* offsets it. A target sized to the ink alone would cut all of
    /// that away, so [`Scene::grow_group_to_its_content`] grows a filtered
    /// group by this before the renderer ever sees it.
    ///
    /// A colour matrix does not grow anything. It can turn transparent pixels
    /// opaque - one with a constant column in its alpha row does - and CSS
    /// answers that with a filter region; gpui's group has no region beyond its
    /// bounds, so a matrix reaches exactly as far as the group does.
    pub fn painted_bounds(&self, input: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        fn tail(sigma: f32) -> ScaledPixels {
            ScaledPixels((sigma * GAUSSIAN_BLUR_EXTENT).max(0.))
        }

        match self {
            SceneFilter::ColorMatrix(_) => input,
            SceneFilter::Blur { radius_x, radius_y } => {
                let (x, y) = (tail(*radius_x), tail(*radius_y));
                input.extend(Edges {
                    top: y,
                    right: x,
                    bottom: y,
                    left: x,
                })
            }
            SceneFilter::DropShadow {
                offset_x,
                offset_y,
                radius,
                ..
            } => {
                // The shadow is drawn behind the input, which is still painted
                // where it always was, so the result covers the input and the
                // offset, blurred silhouette both.
                //
                // The tail is added to *both*, not only to the offset copy, and
                // the reason is the intermediate rather than the result. A
                // renderer blurs the input in place and then reads that blurred
                // image back at `p - offset`, which for a `p` near the near
                // corner of the rectangle lands `offset` outside it. What is
                // there is the input's own blur tail, so the tail has to be
                // inside the rectangle: without it the shadow is cut off in a
                // straight line, exactly where its falloff should have been.
                let offset = point(ScaledPixels(*offset_x), ScaledPixels(*offset_y));
                let moved = Bounds {
                    origin: input.origin + offset,
                    size: input.size,
                };
                input.union(&moved).dilate(tail(*radius))
            }
            SceneFilter::Chain(filters) => filters
                .iter()
                .fold(input, |bounds, filter| filter.painted_bounds(bounds)),
        }
    }
}

/// What an isolated group does to the subtree painted inside it.
///
/// A group is drawn to a target of its own and composited once, so `opacity`,
/// `blend` and `filter` apply to the subtree as a whole rather than to each
/// primitive in it - which is the difference between CSS `opacity` on a box
/// and [`crate::Window::with_element_opacity`], where two overlapping children
/// at 0.5 come out at 0.75 where they overlap.
///
/// The Metal renderer composites these; the wgpu and DirectX renderers still
/// draw straight through them, so on those backends a grouped subtree paints
/// un-isolated. See [`Scene::push_group`].
#[derive(Clone, Debug)]
pub struct GroupSpec {
    /// The part of the window the group can paint, in device pixels, already
    /// intersected with the content mask in force: the size of the target it
    /// will be given.
    pub bounds: Bounds<ScaledPixels>,
    /// The clip path in force where the group was pushed. The group's own
    /// primitives carry it too; the composite of the group's result has to be
    /// clipped by it as well.
    pub clip: ClipId,
    /// The alpha the group's result is composited at, `0.0..=1.0`.
    pub opacity: f32,
    /// How the result is combined with what is underneath.
    pub blend: BlendMode,
    /// A filter applied to the group's own result before compositing.
    pub filter: Option<SceneFilter>,
    /// A filter applied to what is already on the target underneath the group,
    /// before the group is drawn over it: CSS `backdrop-filter`.
    pub backdrop_filter: Option<SceneFilter>,
    /// The content mask in force where the group was pushed, in device pixels,
    /// and so the furthest the group's *result* may reach however far its
    /// filter would carry it.
    ///
    /// Everything painted inside the group is already clipped to this: it is
    /// the group's own primitives' content mask, narrowed further by whatever
    /// they push. A blur is what makes it worth recording separately, because
    /// a blur is the one thing that puts colour outside the ink it was measured
    /// from - and an ancestor's `overflow: hidden` clips a filtered result in
    /// CSS exactly as it clips an unfiltered one.
    ///
    /// `None` means unbounded, which is what a group built by hand in a test
    /// gets.
    pub mask: Option<Bounds<ScaledPixels>>,
}

impl Default for GroupSpec {
    fn default() -> Self {
        Self {
            bounds: Bounds::default(),
            clip: ClipId::NONE,
            opacity: 1.0,
            blend: BlendMode::NORMAL,
            filter: None,
            backdrop_filter: None,
            mask: None,
        }
    }
}

impl GroupSpec {
    /// Whether compositing this group is indistinguishable from drawing its
    /// primitives straight onto the target, *given* that nothing inside it
    /// overlaps anything else inside it.
    ///
    /// Opacity is deliberately not part of this: where nothing overlaps,
    /// fading the group's result and fading each primitive in it produce the
    /// same pixels, so the opacity can be folded away rather than isolated.
    /// See [`Scene::pop_group`].
    fn is_foldable(&self) -> bool {
        self.blend.is_normal() && self.filter.is_none() && self.backdrop_filter.is_none()
    }
}

/// Say, once, that a renderer with no isolated-group support has been handed a
/// scene that carries one.
///
/// The wgpu and DirectX renderers walk a scene's [`Scene::batches`] and ignore
/// its group boundaries entirely, so a subtree that asked to be flattened and
/// then faded, blended or filtered is painted one primitive at a time straight
/// onto what is underneath. Two overlapping children at half opacity come out
/// at three quarters where they overlap, a blur does not happen, and a
/// `backdrop-filter` does not happen. That is a wrong picture, and a wrong
/// picture nobody is told about is the worst kind there is.
///
/// `reported` is the caller's own flag and is never cleared, so this is one
/// line per renderer rather than one a frame: whatever puts a group in a scene
/// puts one there every frame for as long as the window is open.
pub fn report_groups_painted_without_isolation(scene: &Scene, reported: &mut bool) {
    if *reported || scene.groups.is_empty() {
        return;
    }
    *reported = true;
    log::error!(
        "this renderer does not composite isolated groups; the {} in this scene \
         are painted without isolation, and any opacity, blend mode or filter on \
         them is applied one primitive at a time or not at all",
        scene.groups.len(),
    );
}

/// One instruction in the order a scene has to be replayed in: see
/// [`Scene::steps`].
#[derive(Debug)]
pub enum SceneStep<'a> {
    /// Draw these primitives, in the order [`Scene::batches`] hands them over.
    Run(&'a Segment),
    /// Start drawing to a target of this group's own.
    PushGroup(&'a GroupSpec),
    /// Composite the innermost open group back onto the target underneath it.
    PopGroup,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SceneStepRecord {
    Run(SegmentId),
    PushGroup(GroupId),
    PopGroup,
}

/// A group [`Scene::push_group`] has opened and [`Scene::pop_group`] has not
/// closed yet, and everything needed to take it back out of the scene if it
/// turns out to be foldable.
struct OpenGroup {
    id: GroupId,
    /// The steps and segments recorded before the run that was closed to make
    /// room for this group: what to rewind to if the group folds away.
    steps_len: usize,
    segments_len: usize,
    /// The open run's start indices before that run was closed, so folding the
    /// group away puts the primitives painted before it and the ones painted
    /// inside it back in one run - which is exactly a scene that never had the
    /// group in it.
    reopen: Segment,
    /// The start indices of the group's own content.
    content: Segment,
    /// How deep the layer stack was at the push. A group may sit inside a
    /// layer, but it may not straddle one: see [`Scene::push_group`].
    layer_depth: usize,
    /// Whether a group pushed inside this one survived its pop. One that
    /// folded away left nothing behind and does not count.
    has_surviving_child: bool,
    /// The union of the final bounds of the groups inside this one that
    /// survived.
    ///
    /// A surviving child is composited onto this group's target as one quad
    /// over its own bounds, and a blur has already grown those past everything
    /// the child's primitives painted. Growing this group to its primitives
    /// alone would leave the child's quad hanging over the edge of the target,
    /// and the tail would be cut off at exactly the point of having widened the
    /// child in the first place.
    child_bounds: Option<Bounds<ScaledPixels>>,
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
    clip_stack: Vec<ClipId>,
    /// The isolated groups this scene records, in the order they were pushed.
    pub groups: Vec<GroupSpec>,
    /// The runs of primitives the groups cut the scene into. A scene with no
    /// groups in it has exactly one, spanning everything.
    pub segments: Vec<Segment>,
    steps: Vec<SceneStepRecord>,
    group_stack: Vec<OpenGroup>,
    /// The start indices of the run being painted into now. Only the `start`
    /// of each range means anything here; the end is wherever the per-kind
    /// array has reached when the run is closed.
    open_run: Segment,
    pub clips: Vec<SceneClip>,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.clip_stack.clear();
        self.groups.clear();
        self.segments.clear();
        self.steps.clear();
        self.group_stack.clear();
        self.open_run = Segment::default();
        self.clips.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    /// Start an isolated group: everything painted until the matching
    /// [`Scene::pop_group`] belongs to it, and is composited onto what is
    /// underneath as one image rather than one primitive at a time.
    ///
    /// # How this costs a group-free scene nothing
    ///
    /// [`Scene::insert_primitive`] appends to the per-kind arrays and is the
    /// only thing that appends to them, so primitives land in paint order, and
    /// a group is closure-scoped. Everything painted inside a group therefore
    /// occupies a contiguous range of every per-kind array *before* any
    /// sorting, and a group can be recorded as eight ranges - a [`Segment`] -
    /// instead of as a scene of its own. All this method does is close the run
    /// in progress and open a new one.
    ///
    /// # Why sorting each run separately is not less correct than sorting once
    ///
    /// [`Scene::finish`] sorts each run's sub-slice rather than the whole
    /// array, so two primitives in different runs are drawn in paint order,
    /// whatever their `order`. That is never worse than the global sort:
    /// `BoundsTree::insert` returns one more than the greatest order among
    /// the bounds a primitive intersects, so a later-painted primitive can only
    /// carry a *lower* order than an earlier one if the tree has established
    /// they are disjoint - and reordering disjoint primitives changes no
    /// pixel. Where two primitives do intersect, the tree's order and paint
    /// order agree, so both schemes draw them the same way round. Segment
    /// ordering is equal-or-more-correct, never less.
    ///
    /// # Layers
    ///
    /// A group may sit inside a [`Scene::push_layer`] and a layer may sit
    /// inside a group, but neither may straddle the other, and
    /// [`Scene::pop_group`] asserts it. Inside a layer no primitive is
    /// inserted into the bounds tree at all - they all take the layer's own
    /// order - so a layer's contents are kept in paint order by the stable
    /// sort alone, and a boundary that cut a layer in half would be splitting
    /// a run whose ordering is only meaningful whole.
    ///
    /// # Deferred draws
    ///
    /// A [`crate::Window::defer_draw`] issued inside a group escapes it: the
    /// deferred element is painted later, at the top level, and so is not part
    /// of the group and receives none of its opacity, blending or filtering.
    /// It cannot happen through [`crate::Window::with_isolated_group`], which
    /// runs in the paint phase while `defer_draw` is prepaint-only.
    pub fn push_group(&mut self, spec: GroupSpec) -> GroupId {
        let reopen = self.open_run.clone();
        let steps_len = self.steps.len();
        let segments_len = self.segments.len();
        self.close_run();

        let id = GroupId(self.groups.len() as u32);
        self.groups.push(spec.clone());
        self.steps.push(SceneStepRecord::PushGroup(id));
        self.group_stack.push(OpenGroup {
            id,
            steps_len,
            segments_len,
            reopen,
            content: self.open_run.clone(),
            layer_depth: self.layer_stack.len(),
            has_surviving_child: false,
            child_bounds: None,
        });
        self.paint_operations.push(PaintOperation::StartGroup(spec));
        id
    }

    /// Close the group the last [`Scene::push_group`] opened.
    ///
    /// A group whose contents provably do not need a target of their own is
    /// dropped here rather than recorded, and its opacity folded into the
    /// primitives it held: group opacity and per-primitive opacity differ only
    /// where two primitives inside the group overlap, so where nothing inside
    /// it overlaps and it asks for nothing but a fade, fading each primitive
    /// gives the same pixels. gpui already folds opacity per primitive
    /// everywhere else, so this keeps today's behaviour exactly where it is
    /// provably right and spends a target only where it is not.
    pub fn pop_group(&mut self) {
        let group = self
            .group_stack
            .pop()
            .expect("Scene::pop_group without a matching Scene::push_group");
        assert_eq!(
            group.layer_depth,
            self.layer_stack.len(),
            "a group may not straddle a layer boundary: it was pushed inside \
             {} layer(s) and popped inside {}",
            group.layer_depth,
            self.layer_stack.len(),
        );

        if !self.fold_group_if_unobservable(&group) {
            self.grow_group_to_its_content(&group);
            self.close_run();
            self.steps.push(SceneStepRecord::PopGroup);
            let bounds = self.groups[group.id.0 as usize].bounds;
            if let Some(parent) = self.group_stack.last_mut() {
                parent.has_surviving_child = true;
                parent.child_bounds = Some(match parent.child_bounds {
                    Some(existing) => existing.union(&bounds),
                    None => bounds,
                });
            }
        }
        self.paint_operations.push(PaintOperation::EndGroup);
    }

    /// How many groups are open right now: zero everywhere gpui's own UI
    /// paints.
    pub fn group_depth(&self) -> usize {
        self.group_stack.len()
    }

    /// The order the scene has to be drawn in: its runs of primitives, and the
    /// group boundaries between them.
    ///
    /// A scene with no groups in it is a single [`SceneStep::Run`] spanning
    /// everything, which is what every current renderer draws by ignoring this
    /// and calling [`Scene::batches`].
    pub fn steps(&self) -> impl Iterator<Item = SceneStep<'_>> + '_ {
        self.steps.iter().map(move |step| match step {
            SceneStepRecord::Run(id) => SceneStep::Run(&self.segments[id.0 as usize]),
            SceneStepRecord::PushGroup(id) => SceneStep::PushGroup(&self.groups[id.0 as usize]),
            SceneStepRecord::PopGroup => SceneStep::PopGroup,
        })
    }

    /// Close the run being painted into and record it, unless it is empty:
    /// nothing is drawn for an empty run, and a group that opens with nothing
    /// painted before it should not cost a step saying so.
    fn close_run(&mut self) {
        let run = Segment {
            shadows: self.open_run.shadows.start..self.shadows.len(),
            quads: self.open_run.quads.start..self.quads.len(),
            paths: self.open_run.paths.start..self.paths.len(),
            underlines: self.open_run.underlines.start..self.underlines.len(),
            monochrome_sprites: self.open_run.monochrome_sprites.start
                ..self.monochrome_sprites.len(),
            subpixel_sprites: self.open_run.subpixel_sprites.start..self.subpixel_sprites.len(),
            polychrome_sprites: self.open_run.polychrome_sprites.start
                ..self.polychrome_sprites.len(),
            surfaces: self.open_run.surfaces.start..self.surfaces.len(),
        };
        self.open_run = self.end_indices();
        if run.is_empty() {
            return;
        }
        let id = SegmentId(self.segments.len() as u32);
        self.segments.push(run);
        self.steps.push(SceneStepRecord::Run(id));
    }

    /// Where every per-kind array currently ends, as an empty run starting
    /// there.
    fn end_indices(&self) -> Segment {
        Segment {
            shadows: self.shadows.len()..self.shadows.len(),
            quads: self.quads.len()..self.quads.len(),
            paths: self.paths.len()..self.paths.len(),
            underlines: self.underlines.len()..self.underlines.len(),
            monochrome_sprites: self.monochrome_sprites.len()..self.monochrome_sprites.len(),
            subpixel_sprites: self.subpixel_sprites.len()..self.subpixel_sprites.len(),
            polychrome_sprites: self.polychrome_sprites.len()..self.polychrome_sprites.len(),
            surfaces: self.surfaces.len()..self.surfaces.len(),
        }
    }

    /// The run a group's own primitives occupy, at the moment it is popped.
    fn group_content(&self, group: &OpenGroup) -> Segment {
        Segment {
            shadows: group.content.shadows.start..self.shadows.len(),
            quads: group.content.quads.start..self.quads.len(),
            paths: group.content.paths.start..self.paths.len(),
            underlines: group.content.underlines.start..self.underlines.len(),
            monochrome_sprites: group.content.monochrome_sprites.start
                ..self.monochrome_sprites.len(),
            subpixel_sprites: group.content.subpixel_sprites.start..self.subpixel_sprites.len(),
            polychrome_sprites: group.content.polychrome_sprites.start
                ..self.polychrome_sprites.len(),
            surfaces: group.content.surfaces.start..self.surfaces.len(),
        }
    }

    /// Take a group back out of the scene when compositing it separately could
    /// not produce a different picture, folding its opacity into the
    /// primitives it held. Returns whether it did.
    ///
    /// Group opacity and per-primitive opacity differ only where two
    /// primitives inside the group overlap - that is the whole of the
    /// difference between CSS `opacity` on a box and
    /// [`crate::Window::with_element_opacity`] - so where nothing inside the
    /// group overlaps anything else inside it, and the group asks for nothing
    /// but opacity, fading each primitive gives the same pixels as fading the
    /// group. gpui already folds opacity per primitive everywhere else, so
    /// this keeps today's behaviour exactly where it is provably right, and
    /// spends a target only where it is not.
    ///
    /// Three things disqualify a group beyond overlap:
    ///
    /// - a blend mode, a filter or a backdrop filter, none of which can be
    ///   expressed per primitive at all;
    /// - a [`PaintSurface`] in it, which has no opacity to fold into;
    /// - a [`SubpixelSprite`] in it, which blends per color channel against
    ///   whatever is behind it, so its coverage is not a single alpha that
    ///   could be scaled;
    /// - a group inside it that was *not* folded away, whose composited result
    ///   is one image that the outer opacity would have to be applied to whole.
    ///
    /// The overlap test uses [`Primitive::painted_bounds`], so a shadow counts
    /// as the rectangle its gaussian tail actually reaches rather than the one
    /// its falloff is measured from: two shadows whose spread boxes are clear
    /// of one another can still overlap where it matters.
    fn fold_group_if_unobservable(&mut self, group: &OpenGroup) -> bool {
        let spec = &self.groups[group.id.0 as usize];
        if group.has_surviving_child || !spec.is_foldable() {
            return false;
        }
        let opacity = spec.opacity;
        let content = self.group_content(group);
        if !content.subpixel_sprites.is_empty() || !content.surfaces.is_empty() {
            return false;
        }
        if !self.content_is_overlap_free(&content) {
            return false;
        }

        debug_assert_eq!(
            self.groups.len() as u32,
            group.id.0 + 1,
            "a group with no surviving child must be the last one registered"
        );
        self.groups.pop();
        self.steps.truncate(group.steps_len);
        self.segments.truncate(group.segments_len);
        self.open_run = group.reopen.clone();
        if opacity < 1.0 {
            self.fold_opacity(&content, opacity);
        }
        true
    }

    /// Whether no two of a run's primitives cover a common device pixel.
    ///
    /// A plane sweep along x, bailing out at the first overlap found, so the
    /// case this is asked about most - a group that has to be isolated because
    /// its contents do overlap - is answered almost immediately. The case it
    /// cannot answer cheaply is a large run of primitives that are pairwise
    /// disjoint but share an x span, a column of rows say, where the active
    /// set never shrinks; rather than let that cost grow quadratically inside
    /// a frame, the scan gives up after a fixed number of comparisons and the
    /// group is isolated. Isolating a group is never wrong, only slower.
    fn content_is_overlap_free(&self, content: &Segment) -> bool {
        const SCAN_BUDGET: usize = 4096;

        let mut boxes = self.content_boxes(content);

        boxes.sort_by(|a, b| {
            a.origin
                .x
                .partial_cmp(&b.origin.x)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut budget = SCAN_BUDGET;
        let mut active: Vec<usize> = Vec::new();
        for index in 0..boxes.len() {
            let bounds = boxes[index];
            active.retain(|&other| boxes[other].right() > bounds.origin.x);
            for &other in &active {
                if budget == 0 {
                    return false;
                }
                budget -= 1;
                if boxes[other].intersects(&bounds) {
                    return false;
                }
            }
            active.push(index);
        }
        true
    }

    /// Every primitive in a run, as the rectangle it can actually paint into:
    /// what it covers, narrowed by its own content mask.
    fn content_boxes(&self, content: &Segment) -> Vec<Bounds<ScaledPixels>> {
        let mut boxes = Vec::with_capacity(content.len());
        for shadow in &self.shadows[content.shadows.clone()] {
            boxes.push(
                shadow
                    .painted_bounds()
                    .intersect(&shadow.content_mask.bounds),
            );
        }
        for quad in &self.quads[content.quads.clone()] {
            boxes.push(quad.bounds.intersect(&quad.content_mask.bounds));
        }
        for path in &self.paths[content.paths.clone()] {
            boxes.push(path.bounds.intersect(&path.content_mask.bounds));
        }
        for underline in &self.underlines[content.underlines.clone()] {
            boxes.push(underline.bounds.intersect(&underline.content_mask.bounds));
        }
        for sprite in &self.monochrome_sprites[content.monochrome_sprites.clone()] {
            boxes.push(sprite.bounds.intersect(&sprite.content_mask.bounds));
        }
        for sprite in &self.subpixel_sprites[content.subpixel_sprites.clone()] {
            boxes.push(sprite.bounds.intersect(&sprite.content_mask.bounds));
        }
        for sprite in &self.polychrome_sprites[content.polychrome_sprites.clone()] {
            boxes.push(sprite.bounds.intersect(&sprite.content_mask.bounds));
        }
        for surface in &self.surfaces[content.surfaces.clone()] {
            boxes.push(surface.bounds.intersect(&surface.content_mask.bounds));
        }
        boxes
    }

    /// Grow a group that is going to be composited so that its target covers
    /// everything painted inside it, not only the bounds its caller named.
    ///
    /// A group's target is what its subtree is drawn into and what the
    /// compositing quad copies back out, so anything painted outside it is cut
    /// away. Nothing about a group is supposed to clip: CSS `opacity` does not,
    /// and neither does folding that opacity into the primitives one at a time
    /// - which is what happens to this very subtree whenever its contents turn
    /// out not to overlap. Left alone, the two paths would paint different
    /// pictures of the same scene depending on which side of the overlap
    /// heuristic it landed, and a shadow's tail or a child that overhangs its
    /// parent's box would vanish on one side and not the other.
    ///
    /// The caller's bounds stay part of the answer rather than being replaced
    /// by the content's: they are what a group with nothing painted in it is
    /// worth, and a caller that sized a group deliberately still gets at least
    /// that much target.
    ///
    /// # Blur
    ///
    /// A filter that blurs reaches past the ink it was measured from, so the
    /// content's own box is not the answer either: the target has to hold the
    /// gaussian tail and a drop shadow's offset as well, or the composite would
    /// cut the blur off at a hard edge - see [`SceneFilter::painted_bounds`].
    /// That growth, and only that growth, is then clipped back to
    /// [`GroupSpec::mask`]: everything inside the group was clipped to it when
    /// it was painted, and a blur is not licence to paint outside an ancestor's
    /// `overflow: hidden`.
    fn grow_group_to_its_content(&mut self, group: &OpenGroup) {
        let content = self.group_content(group);
        let mut painted: Option<Bounds<ScaledPixels>> =
            group.child_bounds.filter(|bounds| !bounds.is_empty());
        for bounds in self.content_boxes(&content) {
            if bounds.is_empty() {
                continue;
            }
            painted = Some(match painted {
                Some(painted) => painted.union(&bounds),
                None => bounds,
            });
        }
        let Some(painted) = painted else {
            return;
        };
        let spec = &mut self.groups[group.id.0 as usize];
        let mut bounds = if spec.bounds.is_empty() {
            painted
        } else {
            spec.bounds.union(&painted)
        };
        if let Some(filter) = &spec.filter {
            bounds = filter.painted_bounds(bounds);
            if let Some(mask) = spec.mask {
                bounds = bounds.intersect(&mask);
            }
        }
        spec.bounds = bounds;
    }

    /// Multiply `opacity` into the alpha of every primitive in a run.
    fn fold_opacity(&mut self, content: &Segment, opacity: f32) {
        for shadow in &mut self.shadows[content.shadows.clone()] {
            shadow.color = shadow.color.opacity(opacity);
        }
        for quad in &mut self.quads[content.quads.clone()] {
            quad.background = quad.background.opacity(opacity);
            quad.border_color = quad.border_color.opacity(opacity);
        }
        for path in &mut self.paths[content.paths.clone()] {
            path.color = path.color.opacity(opacity);
            // A backend that resolves an image brush never reads `color` at
            // all - the brush replaces the fill - so fading the colour alone
            // fades nothing. The fade has to reach the brush's own multiplier
            // too, which is the same fold a polychrome sprite's `opacity` gets
            // below.
            if let Some(brush) = &mut path.brush {
                brush.opacity *= opacity;
            }
        }
        for underline in &mut self.underlines[content.underlines.clone()] {
            underline.color = underline.color.opacity(opacity);
        }
        for sprite in &mut self.monochrome_sprites[content.monochrome_sprites.clone()] {
            sprite.color = sprite.color.opacity(opacity);
        }
        for sprite in &mut self.polychrome_sprites[content.polychrome_sprites.clone()] {
            sprite.opacity *= opacity;
        }
    }

    /// Register `path` and clip every primitive inserted until the matching
    /// [`Scene::pop_clip`] to it, on top of whatever clip is already pushed.
    ///
    /// `visible` is the content mask the clip was pushed under; see
    /// [`SceneClip::visible`].
    pub fn push_clip(
        &mut self,
        path: ClipPath<ScaledPixels>,
        visible: Bounds<ScaledPixels>,
    ) -> ClipId {
        let parent = self.current_clip();
        self.paint_operations
            .push(PaintOperation::PushClip(path.clone(), visible));
        self.clips.push(SceneClip {
            path,
            parent,
            visible,
        });
        let id = ClipId(self.clips.len() as u32);
        self.clip_stack.push(id);
        id
    }

    /// Stop clipping to the clip path the last [`Scene::push_clip`] pushed.
    pub fn pop_clip(&mut self) {
        self.clip_stack.pop();
        self.paint_operations.push(PaintOperation::PopClip);
    }

    /// The clip path primitives are currently being stamped with.
    pub fn current_clip(&self) -> ClipId {
        self.clip_stack.last().copied().unwrap_or(ClipId::NONE)
    }

    /// The registered clip a [`ClipId`] names, or `None` for
    /// [`ClipId::NONE`].
    pub fn clip(&self, id: ClipId) -> Option<&SceneClip> {
        self.clips.get(id.index()?)
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let mut primitive = primitive.into();
        let clipped_bounds = primitive
            .painted_bounds()
            .intersect(&primitive.content_mask().bounds);

        if clipped_bounds.is_empty() {
            return;
        }

        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));
        // Every primitive reaches the scene through here, so stamping the clip
        // here is what keeps `Window::paint_*` free of a clip argument. It is
        // an unconditional overwrite rather than a fill-if-unset because
        // `replay` hands back primitives carrying the *previous* frame's ids,
        // which name clips that no longer exist.
        let clip = self.current_clip();
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                shadow.clip = clip;
                self.shadows.push(*shadow);
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                quad.clip = clip;
                self.quads.push(*quad);
            }
            Primitive::Path(path) => {
                path.order = order;
                path.clip = clip;
                path.id = PathId(self.paths.len());
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                underline.clip = clip;
                self.underlines.push(*underline);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                sprite.clip = clip;
                self.monochrome_sprites.push(*sprite);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.order = order;
                sprite.clip = clip;
                self.subpixel_sprites.push(*sprite);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                sprite.clip = clip;
                self.polychrome_sprites.push(*sprite);
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                surface.clip = clip;
                self.surfaces.push(surface.clone());
            }
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        let operations = &prev_scene.paint_operations[range];
        Self::debug_assert_groups_are_balanced(operations);
        for operation in operations {
            match operation {
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
                PaintOperation::PushClip(path, visible) => {
                    self.push_clip(path.clone(), *visible);
                }
                PaintOperation::PopClip => self.pop_clip(),
                PaintOperation::StartGroup(spec) => {
                    self.push_group(spec.clone());
                }
                PaintOperation::EndGroup => self.pop_group(),
            }
        }
    }

    /// A range of a previous frame's operations is replayed as a unit, so a
    /// group that starts inside it has to end inside it: replaying half a
    /// group would leave the new scene with a group open that nothing closes,
    /// or close one it never opened. The reuse boundaries are element
    /// subtrees, which is exactly where
    /// [`crate::Window::with_isolated_group`] puts its own boundaries, so this
    /// holds by construction - and if a caller ever pushes a group across one,
    /// this says so where it happened rather than at `finish`.
    #[inline]
    fn debug_assert_groups_are_balanced(operations: &[PaintOperation]) {
        #[cfg(debug_assertions)]
        {
            let mut depth = 0i32;
            for operation in operations {
                match operation {
                    PaintOperation::StartGroup(_) => depth += 1,
                    PaintOperation::EndGroup => {
                        depth -= 1;
                        assert!(
                            depth >= 0,
                            "a replayed range closes a group it does not open"
                        );
                    }
                    _ => {}
                }
            }
            assert_eq!(depth, 0, "a replayed range leaves {depth} group(s) open");
        }
        #[cfg(not(debug_assertions))]
        let _ = operations;
    }

    /// Close the run still open and put every run in drawing order.
    ///
    /// Each run is sorted on its own sub-slice rather than the whole array
    /// being sorted at once. A scene with no groups in it has exactly one run
    /// spanning everything, so that is the same stable sort over the same
    /// slice it always was, element for element; a scene with groups draws
    /// across a group boundary in paint order, which is at least as correct as
    /// sorting by `order` would be. [`Scene::push_group`] carries the argument
    /// for why.
    pub fn finish(&mut self) {
        debug_assert!(
            self.group_stack.is_empty(),
            "a scene was finished with {} group(s) still open",
            self.group_stack.len()
        );
        self.close_run();

        for index in 0..self.segments.len() {
            let run = self.segments[index].clone();
            self.shadows[run.shadows].sort_by_key(|shadow| shadow.order);
            self.quads[run.quads].sort_by_key(|quad| quad.order);
            self.paths[run.paths].sort_by_key(|path| path.order);
            self.underlines[run.underlines].sort_by_key(|underline| underline.order);
            self.monochrome_sprites[run.monochrome_sprites]
                .sort_by_key(|sprite| (sprite.order, sprite.tile.texture_id.index));
            self.subpixel_sprites[run.subpixel_sprites]
                .sort_by_key(|sprite| (sprite.order, sprite.tile.texture_id.index));
            self.polychrome_sprites[run.polychrome_sprites]
                .sort_by_key(|sprite| (sprite.order, sprite.tile.texture_id.index));
            self.surfaces[run.surfaces].sort_by_key(|surface| surface.order);
        }
        self.debug_assert_path_ids_are_a_permutation();
    }

    /// [`Scene::insert_primitive`] hands each path the index it was pushed at,
    /// so within one frame the ids are exactly `0..paths.len()`, each used
    /// once. Sorting reorders them - `order` is not monotone in paint order, so
    /// the ids are not sorted afterwards either - but it cannot duplicate one
    /// or push one out of range, and that is what a renderer keying a per-path
    /// side buffer or atlas entry by id depends on. Anything that starts
    /// splitting, merging or re-emitting paths has to keep this true.
    #[inline]
    fn debug_assert_path_ids_are_a_permutation(&self) {
        #[cfg(debug_assertions)]
        {
            let mut seen = vec![false; self.paths.len()];
            for path in &self.paths {
                let index = path.id.0;
                assert!(
                    index < seen.len(),
                    "path id {index} is out of range for a scene of {} paths",
                    seen.len()
                );
                assert!(!seen[index], "two paths in one scene share the id {index}");
                seen[index] = true;
            }
        }
    }

    /// The primitives, grouped into the runs a renderer can issue as one draw
    /// call.
    ///
    /// A batch is a contiguous range of one sorted per-kind array, so it breaks
    /// wherever the next primitive of another kind has to be drawn first —
    /// and, for sprites, wherever the atlas texture changes.
    ///
    /// Clipping breaks a batch too, but only between clipped and unclipped, not
    /// between one clip and another. The clip id travels in the primitive
    /// record itself, so a fragment shader that samples one mask atlas by id
    /// needs no break at all to switch clips; what it cannot switch per
    /// instance is the pipeline, and clipped drawing binds a mask the
    /// unclipped pipeline does not have. Splitting on the id instead would buy
    /// nothing and cost a draw call per clip change. See
    /// [`ClipId::is_clipped`].
    ///
    /// Every primitive in a batch therefore agrees on `clip.is_clipped()`, and
    /// a renderer can read it off the range's first element.
    ///
    /// A batch never spans a group boundary either: the batches of each
    /// [`SceneStep::Run`] are yielded in turn, so a renderer that ignores
    /// groups - which the wgpu and DirectX ones still do - draws exactly what
    /// it drew before, and a renderer that composites them walks
    /// [`Scene::steps`] and takes each run's batches from
    /// [`Scene::run_batches`]. The ranges index the whole per-kind arrays, not
    /// the run's slice of them.
    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        self.segments
            .iter()
            .flat_map(move |run| BatchIterator::over(self, run))
    }

    /// The batches of one run, for a renderer walking [`Scene::steps`] rather
    /// than [`Scene::batches`]: same batches, in the same order, for the
    /// stretch of the scene between two group boundaries.
    pub fn run_batches(&self, run: &Segment) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator::over(self, run)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
    PushClip(ClipPath<ScaledPixels>, Bounds<ScaledPixels>),
    PopClip,
    StartGroup(GroupSpec),
    EndGroup,
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
}

#[expect(missing_docs)]
impl Primitive {
    /// Where the renderer actually puts pixels, which is what the bounds tree
    /// has to be told about: two primitives it believes are disjoint get
    /// unrelated draw orders, and if they in fact overlap they can then be
    /// drawn in the wrong order.
    ///
    /// For every kind but a shadow this is just [`Primitive::bounds`]. A
    /// shadow's record holds the rectangle its SDF is measured from, and the
    /// vertex shader derives the geometry it draws from that - see
    /// [`Shadow::painted_bounds`].
    pub fn painted_bounds(&self) -> Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => shadow.painted_bounds(),
            _ => *self.bounds(),
        }
    }

    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
}

impl<'a> BatchIterator<'a> {
    /// The batches of one run of the scene. The `_start` cursors are indices
    /// into the whole per-kind arrays rather than into the run, so the ranges
    /// this yields address the arrays a renderer uploads.
    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    fn over(scene: &'a Scene, run: &Segment) -> Self {
        BatchIterator {
            shadows_start: run.shadows.start,
            shadows_iter: scene.shadows[run.shadows.clone()].iter().peekable(),
            quads_start: run.quads.start,
            quads_iter: scene.quads[run.quads.clone()].iter().peekable(),
            paths_start: run.paths.start,
            paths_iter: scene.paths[run.paths.clone()].iter().peekable(),
            underlines_start: run.underlines.start,
            underlines_iter: scene.underlines[run.underlines.clone()].iter().peekable(),
            monochrome_sprites_start: run.monochrome_sprites.start,
            monochrome_sprites_iter: scene.monochrome_sprites[run.monochrome_sprites.clone()]
                .iter()
                .peekable(),
            subpixel_sprites_start: run.subpixel_sprites.start,
            subpixel_sprites_iter: scene.subpixel_sprites[run.subpixel_sprites.clone()]
                .iter()
                .peekable(),
            polychrome_sprites_start: run.polychrome_sprites.start,
            polychrome_sprites_iter: scene.polychrome_sprites[run.polychrome_sprites.clone()]
                .iter()
                .peekable(),
            surfaces_start: run.surfaces.start,
            surfaces_iter: scene.surfaces[run.surfaces.clone()].iter().peekable(),
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let clipped = self.shadows_iter.peek().unwrap().clip.is_clipped();
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| {
                        (shadow.order, batch_kind) < max_order_and_kind
                            && shadow.clip.is_clipped() == clipped
                    })
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(shadows_start..shadows_end))
            }
            PrimitiveKind::Quad => {
                let clipped = self.quads_iter.peek().unwrap().clip.is_clipped();
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| {
                        (quad.order, batch_kind) < max_order_and_kind
                            && quad.clip.is_clipped() == clipped
                    })
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(quads_start..quads_end))
            }
            PrimitiveKind::Path => {
                let clipped = self.paths_iter.peek().unwrap().clip.is_clipped();
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| {
                        (path.order, batch_kind) < max_order_and_kind
                            && path.clip.is_clipped() == clipped
                    })
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(paths_start..paths_end))
            }
            PrimitiveKind::Underline => {
                let clipped = self.underlines_iter.peek().unwrap().clip.is_clipped();
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| {
                        (underline.order, batch_kind) < max_order_and_kind
                            && underline.clip.is_clipped() == clipped
                    })
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let clipped = self
                    .monochrome_sprites_iter
                    .peek()
                    .unwrap()
                    .clip
                    .is_clipped();
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                            && sprite.clip.is_clipped() == clipped
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let clipped = self.subpixel_sprites_iter.peek().unwrap().clip.is_clipped();
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                            && sprite.clip.is_clipped() == clipped
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let clipped = self
                    .polychrome_sprites_iter
                    .peek()
                    .unwrap()
                    .clip
                    .is_clipped();
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                            && sprite.clip.is_clipped() == clipped
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::Surface => {
                let clipped = self.surfaces_iter.peek().unwrap().clip.is_clipped();
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| {
                        (surface.order, batch_kind) < max_order_and_kind
                            && surface.clip.is_clipped() == clipped
                    })
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Quads(Range<usize>),
    Paths(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Quads(range) => format!("quads ({})", range.len()),
            Self::Paths(range) => format!("paths ({})", range.len()),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => {
                format!(
                    "monochrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::SubpixelSprites { texture_id, range } => {
                format!(
                    "subpixel sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::PolychromeSprites { texture_id, range } => {
                format!(
                    "polychrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub clip: ClipId,
    pub pad: u32, // keep the record an even number of words
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub clip: ClipId, // also aligns the record to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: PaddedBool32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    /// 0 = drop shadow (rendered outside the element), 1 = inset shadow (rendered inside).
    pub inset: u32,
    pub clip: ClipId, // also aligns the record to 8 bytes
}

/// How many standard deviations of gaussian tail anything blurred is drawn out
/// to before the rest is treated as nothing. Three of them hold 99.7% of the
/// weight.
///
/// This is the figure the shadow vertex shaders already leave room for outside
/// a drop shadow's rectangle, and it is kept in step with `shadow_vertex` in
/// every backend's shader: `shaders.metal`, `shaders.wgsl`, `shaders.hlsl`.
/// [`SceneFilter::painted_bounds`] uses the same one, so a group blurred by one
/// standard deviation and a box shadow blurred by one reach exactly as far as
/// each other.
/// A renderer's blur passes have to use this same figure for how many texels
/// they read: fewer and the tail is cut off inside a target that had room for
/// it, more and the reads land outside the image for nothing.
pub const GAUSSIAN_BLUR_EXTENT: f32 = 3.;

impl Shadow {
    /// The rectangle the renderer actually covers with this shadow, which is
    /// not [`Shadow::bounds`].
    ///
    /// `bounds` is the rectangle the fragment shader measures its signed
    /// distance from, so it cannot be dilated without changing the shape that
    /// gets drawn. The geometry the vertex shader emits is derived from it
    /// instead: a drop shadow's is `bounds` grown by three blur radii, out to
    /// where the gaussian tail has faded, and an inset shadow's is the element
    /// it is painted inside.
    pub fn painted_bounds(&self) -> Bounds<ScaledPixels> {
        if self.inset == 0 {
            self.bounds.dilate(self.blur_radius * GAUSSIAN_BLUR_EXTENT)
        } else {
            self.element_bounds
        }
    }
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }

    /// The transformation that undoes this one, or `None` if it collapses the
    /// plane onto a line or a point and so undoes nothing.
    ///
    /// A brush is authored the way it is drawn - the image goes *there* - but a
    /// fragment shader walks the other way, from the pixel it is shading to the
    /// texel it should read, so something has to invert the matrix. Doing it
    /// here means doing it once per path rather than once per fragment.
    pub fn invert(&self) -> Option<TransformationMatrix> {
        let [[a, b], [c, d]] = self.rotation_scale;
        let determinant = a * d - b * c;
        if determinant == 0. || !determinant.is_finite() {
            return None;
        }
        let inverse_rotation_scale = [
            [d / determinant, -b / determinant],
            [-c / determinant, a / determinant],
        ];
        let [x, y] = self.translation;
        Some(TransformationMatrix {
            rotation_scale: inverse_rotation_scale,
            translation: [
                -(inverse_rotation_scale[0][0] * x + inverse_rotation_scale[0][1] * y),
                -(inverse_rotation_scale[1][0] * x + inverse_rotation_scale[1][1] * y),
            ],
        })
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub clip: ClipId,
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub clip: ClipId, // also aligns the record to 8 bytes
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub clip: ClipId,
    pub tile: AtlasTile,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    // A surface is never uploaded as a packed record — the renderers read this
    // struct field by field — so unlike the other primitives it needs no
    // padding word here, and `SurfaceParams`/`SurfaceBounds` keep their twelve
    // words untouched until a backend has a mask to hand them.
    pub clip: ClipId,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub image_buffer: objc2_core_foundation::CFRetained<objc2_core_video::CVPixelBuffer>,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// What a [`PathBrush`] paints beyond the one copy of its image, along one
/// axis.
///
/// None of these can be a sampler address mode: the atlas packs a tile against
/// its neighbours and [`AtlasTile::padding`] is zero, so wrapping in the atlas
/// wraps into another sprite. A renderer applies these in tile-local
/// coordinates and maps the result into the atlas afterwards.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum BrushExtend {
    /// Clamp to the image's edge texels, so everything outside the one copy is
    /// a smear of its border.
    #[default]
    Pad = 0,
    /// Tile the image.
    Repeat = 1,
    /// Tile the image, mirroring every other copy so the tiles meet edge to
    /// edge.
    Reflect = 2,
}

/// An image a path is filled with, in place of its solid or gradient
/// [`Background`].
///
/// This rides on the path rather than in [`Background`], which is embedded by
/// value in every quad and mirrored by hand in four shader languages:
/// widening that would tax every quad in every scene for something no quad
/// does. A renderer hangs these off a per-path side buffer, indexed the way the
/// per-path content masks already are, so a path that has no brush costs
/// nothing per vertex and a path that has one costs nothing per vertex either.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct PathBrush {
    /// The atlas tile holding the image, in the same polychrome atlas - and
    /// under the same key - that [`crate::Window::paint_image`] uses.
    pub tile: AtlasTile,
    /// Maps a scene position, in device pixels, into brush space, where the
    /// unit square is one copy of the image. The brush transform, the window's
    /// scale factor and the image's own pixel size are all folded into this one
    /// matrix by [`crate::Window::paint_path_with_image`], so a renderer
    /// inverts nothing and multiplies once.
    pub screen_to_brush: TransformationMatrix,
    /// What to paint outside the unit square horizontally.
    pub x_extend: BrushExtend,
    /// What to paint outside the unit square vertically.
    pub y_extend: BrushExtend,
    /// A multiplier on the image's own alpha, carrying the element opacity and
    /// whatever alpha the caller asked for.
    pub opacity: f32,
}

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    // Carried once per path, the way `content_mask` is, rather than per vertex:
    // Metal already hangs the per-path content masks off a side buffer and can
    // hang clip ids off the same one, and wgpu's `PathRasterizationVertex` is
    // 34 words, so a clip id there would cost 36 words on the highest
    // vertex-count primitive gpui has. Whichever a backend picks, it picks it
    // in phase 2; the scene only has to know which clip the path belongs to.
    pub clip: ClipId,
    pub bounds: Bounds<P>,
    pub content_mask: ContentMask<P>,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    /// An image to fill the path with instead of `color`, resolved by
    /// [`crate::Window::paint_path_with_image`]. Deliberately not part of the
    /// `repr(C)` records a renderer uploads: it is read on the CPU into a
    /// per-path side buffer, so it changes no vertex layout and no shader
    /// struct.
    ///
    /// Only the Metal renderer resolves it; the others paint `color` and say
    /// so.
    pub brush: Option<PathBrush>,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            clip: ClipId::NONE,
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            brush: None,
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            clip: self.clip,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
            brush: self.brush,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

/// One vertex of a [`Path`].
///
/// A path carries its content mask once, on the path itself; every renderer
/// reads it from there. Repeating it here would put another 16 bytes and four
/// more multiplies in [`PathVertex::scale`] on the highest vertex-count
/// primitive there is, for a copy nothing ever looks at.
#[derive(Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bounds, ClipPath, Corners, Pixels, Size, size};

    /// Pushes a clip reachable over exactly its own bounds, which is what
    /// `Window::with_clip_path` hands the scene.
    fn push_clip_path(scene: &mut Scene, width: f32) -> ClipId {
        let path = clip_path(width);
        let visible = path.bounds();
        scene.push_clip(path, visible)
    }

    fn clip_path(width: f32) -> ClipPath<ScaledPixels> {
        ClipPath::rounded_rect(
            Bounds {
                origin: point(Pixels(0.), Pixels(0.)),
                size: size(Pixels(width), Pixels(100.)),
            },
            Corners {
                top_left: Size {
                    width: Pixels(20.),
                    height: Pixels(8.),
                },
                ..Default::default()
            },
        )
        .scale(1.0)
    }

    /// A quad far enough from its neighbours that the bounds tree never has to
    /// order two of them relative to one another, so a batch splits only where
    /// this module makes it split.
    fn quad_at(x: f32) -> Quad {
        let bounds = Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(0.)),
            size: size(ScaledPixels(10.), ScaledPixels(10.)),
        };
        Quad {
            bounds,
            content_mask: ContentMask::new(bounds),
            ..Default::default()
        }
    }

    fn underline_at(x: f32) -> Underline {
        let bounds = Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(0.)),
            size: size(ScaledPixels(10.), ScaledPixels(2.)),
        };
        Underline {
            order: 0,
            clip: ClipId::NONE,
            bounds,
            content_mask: ContentMask::new(bounds),
            color: Hsla::default(),
            thickness: ScaledPixels(1.),
            wavy: false.into(),
        }
    }

    /// A drop shadow whose gaussian tail reaches `3 * blur` beyond the rect its
    /// falloff is measured from, under a content mask wide enough not to clip
    /// any of it.
    fn shadow_at(x: f32, blur: f32) -> Shadow {
        let bounds = Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(0.)),
            size: size(ScaledPixels(10.), ScaledPixels(10.)),
        };
        Shadow {
            order: 0,
            clip: ClipId::NONE,
            blur_radius: ScaledPixels(blur),
            bounds,
            corner_radii: Corners::default(),
            content_mask: ContentMask::new(Bounds {
                origin: point(ScaledPixels(-1000.), ScaledPixels(-1000.)),
                size: size(ScaledPixels(2000.), ScaledPixels(2000.)),
            }),
            color: Hsla::default(),
            element_bounds: bounds,
            element_corner_radii: Corners::default(),
            inset: 0,
        }
    }

    fn path_at(x: f32) -> Path<ScaledPixels> {
        let bounds = crate::Bounds {
            origin: point(crate::px(x), crate::px(0.)),
            size: size(crate::px(10.), crate::px(10.)),
        };
        let mut path = Path::new(bounds.origin);
        path.line_to(bounds.top_right());
        path.line_to(bounds.bottom_right());
        path.content_mask = ContentMask::new(bounds);
        path.scale(1.0)
    }

    /// The bounds tree orders a primitive by what it intersects, so it has to be
    /// told the rectangle the renderer actually covers. A shadow's own
    /// `bounds` is the rect its falloff is measured from; the vertex shader
    /// draws three blur radii wider, and the tail out there is opaque enough to
    /// see.
    #[test]
    fn a_blurred_shadow_is_ordered_by_its_tail_not_by_its_inner_rect() {
        let mut scene = Scene::default();
        scene.insert_primitive(quad_at(0.));
        // Inner rect 40..50, clear of the quad at 0..10; painted out to
        // -5..95, which is not.
        scene.insert_primitive(shadow_at(40., 15.));
        scene.finish();

        assert!(
            scene.shadows[0].order > scene.quads[0].order,
            "a shadow painted over a quad must be drawn after it: shadow order \
             {}, quad order {}",
            scene.shadows[0].order,
            scene.quads[0].order,
        );
    }

    #[test]
    fn a_shadow_whose_tail_reaches_nothing_is_still_ordered_beside_it() {
        // The other direction: widening the recorded bounds must not make
        // everything overlap everything.
        let mut scene = Scene::default();
        scene.insert_primitive(quad_at(0.));
        scene.insert_primitive(shadow_at(40., 1.));
        scene.finish();

        assert_eq!(scene.shadows[0].order, scene.quads[0].order);
    }

    /// An inset shadow's geometry is the element it is painted inside, not the
    /// hole its falloff is measured from - a large offset can put that hole
    /// outside the content mask entirely, and culling on it drops a shadow the
    /// shader would have painted.
    #[test]
    fn an_inset_shadow_is_not_culled_by_a_hole_outside_its_content_mask() {
        let element_bounds = Bounds {
            origin: point(ScaledPixels(0.), ScaledPixels(0.)),
            size: size(ScaledPixels(10.), ScaledPixels(10.)),
        };
        let mut shadow = shadow_at(500., 0.);
        shadow.inset = 1;
        shadow.element_bounds = element_bounds;
        shadow.content_mask = ContentMask::new(element_bounds);

        let mut scene = Scene::default();
        scene.insert_primitive(shadow);
        scene.finish();

        assert_eq!(scene.shadows.len(), 1);
    }

    /// What a renderer keying anything by [`PathId`] relies on: within one
    /// frame the ids are `0..paths.len()`, each used once. They are *not*
    /// sorted after `finish` - `order` is not monotone in paint order, so a
    /// path painted later can sort earlier and carry a lower id with it.
    #[test]
    fn path_ids_are_a_permutation_after_finish_even_where_they_are_not_sorted() {
        let mut scene = Scene::default();
        // Two overlapping paths, so the second takes an order above the first,
        // then one clear of both, which takes the lower order and sorts ahead
        // of it carrying the id 2.
        scene.insert_primitive(path_at(0.));
        scene.insert_primitive(path_at(5.));
        scene.insert_primitive(path_at(100.));
        scene.finish();

        let ids: Vec<_> = scene.paths.iter().map(|path| path.id.0).collect();
        assert_eq!(ids, vec![0, 2, 1], "id 2 sorted ahead of id 1");

        let mut sorted = ids;
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2], "the ids are a permutation of 0..3");
    }

    fn quad_batches(scene: &Scene) -> Vec<Range<usize>> {
        scene
            .batches()
            .map(|batch| match batch {
                PrimitiveBatch::Quads(range) => range,
                other => panic!("expected only quad batches, got {}", other.label()),
            })
            .collect()
    }

    /// The "costs nothing" proof: paint the way gpui's own UI does, and the
    /// scene registers no clips, stamps every primitive with `ClipId::NONE`,
    /// and batches exactly as it did before clip paths existed.
    #[test]
    fn an_unclipped_scene_is_one_batch_per_kind_of_clipless_primitives() {
        let mut scene = Scene::default();
        for i in 0..4 {
            scene.insert_primitive(quad_at(i as f32 * 20.));
            scene.insert_primitive(underline_at(i as f32 * 20.));
        }
        scene.finish();

        assert!(scene.clips.is_empty());
        assert!(scene.quads.iter().all(|quad| quad.clip == ClipId::NONE));
        assert!(
            scene
                .underlines
                .iter()
                .all(|underline| underline.clip == ClipId::NONE)
        );

        let batches: Vec<_> = scene.batches().map(|batch| batch.label()).collect();
        assert_eq!(batches, vec!["quads (4)", "underlines (4)"]);
    }

    #[test]
    fn a_batch_breaks_between_clipped_and_unclipped() {
        let mut scene = Scene::default();
        scene.insert_primitive(quad_at(0.));
        push_clip_path(&mut scene, 30.);
        scene.insert_primitive(quad_at(20.));
        scene.insert_primitive(quad_at(40.));
        scene.pop_clip();
        scene.insert_primitive(quad_at(60.));
        scene.finish();

        assert_eq!(quad_batches(&scene), vec![0..1, 1..3, 3..4]);
        let clips: Vec<_> = scene.quads.iter().map(|quad| quad.clip).collect();
        assert_eq!(
            clips,
            vec![ClipId::NONE, ClipId(1), ClipId(1), ClipId::NONE]
        );
    }

    #[test]
    fn a_batch_does_not_break_between_two_different_clips() {
        // The clip id rides in the primitive record, so a fragment shader that
        // samples one mask atlas by id switches clips without a draw call.
        let mut scene = Scene::default();
        push_clip_path(&mut scene, 30.);
        scene.insert_primitive(quad_at(0.));
        scene.pop_clip();
        push_clip_path(&mut scene, 50.);
        scene.insert_primitive(quad_at(20.));
        scene.pop_clip();
        scene.finish();

        assert_eq!(
            scene.quads.iter().map(|quad| quad.clip).collect::<Vec<_>>(),
            vec![ClipId(1), ClipId(2)]
        );
        assert_eq!(quad_batches(&scene), vec![0..2]);
    }

    #[test]
    fn clips_nest_and_unwind() {
        let mut scene = Scene::default();
        assert_eq!(scene.current_clip(), ClipId::NONE);

        let outer = push_clip_path(&mut scene, 30.);
        scene.insert_primitive(quad_at(0.));
        let inner = push_clip_path(&mut scene, 50.);
        scene.insert_primitive(quad_at(20.));
        scene.pop_clip();
        assert_eq!(scene.current_clip(), outer);
        scene.insert_primitive(quad_at(40.));
        scene.pop_clip();
        assert_eq!(scene.current_clip(), ClipId::NONE);
        scene.insert_primitive(quad_at(60.));
        scene.finish();

        assert_eq!(scene.clip(outer).unwrap().parent, ClipId::NONE);
        assert_eq!(scene.clip(inner).unwrap().parent, outer);
        assert!(scene.clip(ClipId::NONE).is_none());
        assert_eq!(
            scene.quads.iter().map(|quad| quad.clip).collect::<Vec<_>>(),
            vec![outer, inner, outer, ClipId::NONE]
        );
    }

    #[test]
    fn clearing_a_scene_forgets_its_clips() {
        let mut scene = Scene::default();
        push_clip_path(&mut scene, 30.);
        scene.insert_primitive(quad_at(0.));
        scene.clear();

        assert!(scene.clips.is_empty());
        assert_eq!(scene.current_clip(), ClipId::NONE);
        scene.insert_primitive(quad_at(0.));
        assert_eq!(scene.quads[0].clip, ClipId::NONE);
    }

    #[test]
    fn replay_re_registers_clips_and_renumbers_their_ids() {
        let mut prev = Scene::default();
        prev.insert_primitive(quad_at(0.));
        push_clip_path(&mut prev, 30.);
        prev.insert_primitive(quad_at(20.));
        prev.pop_clip();
        prev.insert_primitive(quad_at(40.));
        let replayed = 0..prev.len();
        prev.finish();
        assert_eq!(prev.quads[1].clip, ClipId(1));

        // The new scene registers a clip of its own first, so a replayed
        // primitive that kept the id it was given last frame would name the
        // wrong shape.
        let mut next = Scene::default();
        push_clip_path(&mut next, 70.);
        next.insert_primitive(quad_at(60.));
        next.pop_clip();
        next.replay(replayed, &prev);
        next.finish();

        assert_eq!(next.clips.len(), 2);
        assert_eq!(next.clips[1].path, clip_path(30.));
        assert_eq!(next.clips[1].parent, ClipId::NONE);
        assert_eq!(
            next.quads.iter().map(|quad| quad.clip).collect::<Vec<_>>(),
            vec![ClipId(1), ClipId::NONE, ClipId(2), ClipId::NONE]
        );
        assert_eq!(next.current_clip(), ClipId::NONE);
    }

    #[test]
    fn a_replayed_subtree_nests_inside_the_clip_it_is_replayed_under() {
        let mut prev = Scene::default();
        push_clip_path(&mut prev, 30.);
        prev.insert_primitive(quad_at(0.));
        prev.pop_clip();
        let replayed = 0..prev.len();
        prev.finish();

        let mut next = Scene::default();
        let outer = push_clip_path(&mut next, 70.);
        next.replay(replayed, &prev);
        assert_eq!(next.current_clip(), outer);
        next.pop_clip();
        next.finish();

        assert_eq!(next.clips.len(), 2);
        assert_eq!(next.clips[1].parent, outer);
        assert_eq!(next.quads[0].clip, ClipId(2));
    }

    /// A quad with something in it to fade: the default background and border
    /// are transparent, and multiplying zero alpha by anything proves nothing.
    fn filled_quad_at(x: f32) -> Quad {
        let mut quad = quad_at(x);
        quad.background = opaque().into();
        quad.border_color = opaque();
        quad
    }

    fn opaque() -> Hsla {
        Hsla {
            h: 0.,
            s: 1.,
            l: 0.5,
            a: 1.,
        }
    }

    fn tile(id: u32, kind: crate::AtlasTextureKind) -> crate::AtlasTile {
        crate::AtlasTile {
            texture_id: crate::AtlasTextureId { index: 0, kind },
            tile_id: crate::TileId(id),
            padding: 0,
            bounds: Bounds::default(),
        }
    }

    fn monochrome_sprite_at(x: f32, y: f32, tile_id: u32) -> MonochromeSprite {
        let bounds = Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(8.), ScaledPixels(8.)),
        };
        MonochromeSprite {
            order: 0,
            clip: ClipId::NONE,
            bounds,
            content_mask: ContentMask::new(bounds),
            color: opaque(),
            tile: tile(tile_id, crate::AtlasTextureKind::Monochrome),
            transformation: TransformationMatrix::unit(),
        }
    }

    fn subpixel_sprite_at(x: f32) -> SubpixelSprite {
        let bounds = Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(0.)),
            size: size(ScaledPixels(8.), ScaledPixels(8.)),
        };
        SubpixelSprite {
            order: 0,
            clip: ClipId::NONE,
            bounds,
            content_mask: ContentMask::new(bounds),
            color: opaque(),
            tile: tile(0, crate::AtlasTextureKind::Subpixel),
            transformation: TransformationMatrix::unit(),
        }
    }

    fn polychrome_sprite_at(x: f32, y: f32, tile_id: u32) -> PolychromeSprite {
        let bounds = Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(8.), ScaledPixels(8.)),
        };
        PolychromeSprite {
            order: 0,
            clip: ClipId::NONE,
            grayscale: false.into(),
            opacity: 1.,
            bounds,
            content_mask: ContentMask::new(bounds),
            corner_radii: Corners::default(),
            tile: tile(tile_id, crate::AtlasTextureKind::Polychrome),
        }
    }

    #[cfg(target_os = "macos")]
    fn surface_at(x: f32) -> PaintSurface {
        use core_video::pixel_buffer::{CVPixelBuffer, kCVPixelFormatType_32BGRA};

        let bounds = Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(0.)),
            size: size(ScaledPixels(8.), ScaledPixels(8.)),
        };
        PaintSurface {
            order: 0,
            clip: ClipId::NONE,
            bounds,
            content_mask: ContentMask::new(bounds),
            image_buffer: CVPixelBuffer::new(kCVPixelFormatType_32BGRA, 8, 8, None)
                .expect("failed to create a pixel buffer"),
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn surface_at(x: f32) -> PaintSurface {
        let bounds = Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(0.)),
            size: size(ScaledPixels(8.), ScaledPixels(8.)),
        };
        PaintSurface {
            order: 0,
            clip: ClipId::NONE,
            bounds,
            content_mask: ContentMask::new(bounds),
        }
    }

    /// A group that can never be folded away, whatever is painted inside it:
    /// a blend mode cannot be expressed one primitive at a time.
    fn isolated_group() -> GroupSpec {
        GroupSpec {
            blend: BlendMode {
                mix: MixMode::Multiply,
                compose: ComposeMode::SrcOver,
            },
            ..Default::default()
        }
    }

    /// A group that asks for nothing but a subtree-wide fade, which is what
    /// [`Scene::pop_group`] folds away when the subtree does not overlap
    /// itself.
    fn faded_group(opacity: f32) -> GroupSpec {
        GroupSpec {
            opacity,
            ..Default::default()
        }
    }

    fn steps_summary(scene: &Scene) -> Vec<String> {
        scene
            .steps()
            .map(|step| match step {
                SceneStep::Run(run) => format!("run {}", run.len()),
                SceneStep::PushGroup(_) => "push".into(),
                SceneStep::PopGroup => "pop".into(),
            })
            .collect()
    }

    /// A deterministic generator, so a failure is reproducible and the test
    /// needs nothing from outside the crate.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }

        fn upto(&mut self, bound: u64) -> f32 {
            (self.next() % bound) as f32
        }
    }

    macro_rules! sort_keys {
        ($primitives:expr) => {
            $primitives
                .iter()
                .map(|primitive| {
                    (
                        primitive.order,
                        primitive.bounds.origin.x.0.to_bits(),
                        primitive.bounds.origin.y.0.to_bits(),
                    )
                })
                .collect::<Vec<_>>()
        };
    }

    /// The "costs nothing when absent" proof. A scene with no groups in it is
    /// one run spanning everything, so `finish` runs the same stable sort over
    /// the same slice it ran before runs existed, and every per-kind array
    /// comes out element for element identical to a global sort of the same
    /// primitives.
    #[test]
    fn a_scene_with_no_groups_sorts_exactly_as_one_global_sort_does() {
        let mut random = Lcg(0x5eed);
        let mut scene = Scene::default();
        for index in 0..400u64 {
            let x = random.upto(120);
            let y = random.upto(120);
            match index % 6 {
                0 => scene.insert_primitive(filled_quad_at(x)),
                1 => scene.insert_primitive(underline_at(x)),
                2 => scene.insert_primitive(shadow_at(x, random.upto(6))),
                3 => scene.insert_primitive(path_at(x)),
                4 => scene.insert_primitive(monochrome_sprite_at(x, y, (random.upto(4)) as u32)),
                _ => scene.insert_primitive(polychrome_sprite_at(x, y, (random.upto(4)) as u32)),
            }
        }

        let painted_quads = sort_keys!(scene.quads);
        let mut shadows = scene.shadows.clone();
        let mut quads = scene.quads.clone();
        let mut paths = scene.paths.clone();
        let mut underlines = scene.underlines.clone();
        let mut monochrome_sprites = scene.monochrome_sprites.clone();
        let mut polychrome_sprites = scene.polychrome_sprites.clone();
        shadows.sort_by_key(|shadow| shadow.order);
        quads.sort_by_key(|quad| quad.order);
        paths.sort_by_key(|path| path.order);
        underlines.sort_by_key(|underline| underline.order);
        monochrome_sprites.sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        polychrome_sprites.sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));

        scene.finish();

        assert_ne!(
            sort_keys!(quads),
            painted_quads,
            "the corpus has to be one the sort actually reorders, or this \
             proves nothing"
        );
        assert_eq!(scene.segments.len(), 1, "one run spanning everything");
        assert_eq!(scene.steps().count(), 1, "and one step drawing it");
        assert!(scene.groups.is_empty());
        assert_eq!(sort_keys!(scene.shadows), sort_keys!(shadows));
        assert_eq!(sort_keys!(scene.quads), sort_keys!(quads));
        assert_eq!(sort_keys!(scene.paths), sort_keys!(paths));
        assert_eq!(sort_keys!(scene.underlines), sort_keys!(underlines));
        assert_eq!(
            sort_keys!(scene.monochrome_sprites),
            sort_keys!(monochrome_sprites)
        );
        assert_eq!(
            sort_keys!(scene.polychrome_sprites),
            sort_keys!(polychrome_sprites)
        );
    }

    #[test]
    fn a_group_free_scene_batches_exactly_as_it_did_before_groups() {
        let mut scene = Scene::default();
        for i in 0..4 {
            scene.insert_primitive(quad_at(i as f32 * 20.));
            scene.insert_primitive(underline_at(i as f32 * 20.));
        }
        scene.finish();

        assert_eq!(steps_summary(&scene), vec!["run 8"]);
        let batches: Vec<_> = scene.batches().map(|batch| batch.label()).collect();
        assert_eq!(batches, vec!["quads (4)", "underlines (4)"]);
    }

    #[test]
    fn batches_break_at_a_group_boundary() {
        let mut scene = Scene::default();
        scene.insert_primitive(quad_at(0.));
        scene.push_group(isolated_group());
        scene.insert_primitive(quad_at(100.));
        scene.insert_primitive(quad_at(120.));
        scene.pop_group();
        scene.insert_primitive(quad_at(300.));
        scene.finish();

        // The same four quads without the group are one batch: nothing about
        // them breaks it but the boundary.
        let mut ungrouped = Scene::default();
        for x in [0., 100., 120., 300.] {
            ungrouped.insert_primitive(quad_at(x));
        }
        ungrouped.finish();
        assert_eq!(quad_batches(&ungrouped), vec![0..4]);

        assert_eq!(quad_batches(&scene), vec![0..1, 1..3, 3..4]);
        assert_eq!(
            steps_summary(&scene),
            vec!["run 1", "push", "run 2", "pop", "run 1"]
        );
    }

    /// The ranges a batch names index the whole per-kind array, not the run's
    /// slice of it, so a renderer keeps uploading one buffer per kind.
    #[test]
    fn a_batch_inside_a_group_addresses_the_whole_array() {
        let mut scene = Scene::default();
        scene.insert_primitive(underline_at(0.));
        scene.push_group(isolated_group());
        scene.insert_primitive(underline_at(100.));
        scene.pop_group();
        scene.finish();

        let batches: Vec<_> = scene
            .batches()
            .map(|batch| match batch {
                PrimitiveBatch::Underlines(range) => range,
                other => panic!("expected underlines, got {}", other.label()),
            })
            .collect();
        assert_eq!(batches, vec![0..1, 1..2]);
    }

    #[test]
    fn groups_nest_and_unwind() {
        let mut scene = Scene::default();
        assert_eq!(scene.group_depth(), 0);
        scene.push_group(isolated_group());
        scene.insert_primitive(quad_at(0.));
        assert_eq!(scene.group_depth(), 1);
        scene.push_group(isolated_group());
        scene.insert_primitive(quad_at(100.));
        assert_eq!(scene.group_depth(), 2);
        scene.pop_group();
        scene.insert_primitive(quad_at(200.));
        assert_eq!(scene.group_depth(), 1);
        scene.pop_group();
        assert_eq!(scene.group_depth(), 0);
        scene.finish();

        assert_eq!(scene.groups.len(), 2);
        assert_eq!(
            steps_summary(&scene),
            vec!["push", "run 1", "push", "run 1", "pop", "run 1", "pop"]
        );
    }

    #[test]
    fn clearing_a_scene_forgets_its_groups() {
        let mut scene = Scene::default();
        scene.push_group(isolated_group());
        scene.insert_primitive(quad_at(0.));
        scene.pop_group();
        scene.clear();

        assert!(scene.groups.is_empty());
        assert!(scene.segments.is_empty());
        assert_eq!(scene.steps().count(), 0);
        assert_eq!(scene.group_depth(), 0);

        scene.insert_primitive(quad_at(0.));
        scene.finish();
        assert_eq!(steps_summary(&scene), vec!["run 1"]);
    }

    /// A group is rebuilt by replaying it, not copied: the ids it takes are the
    /// ones the new scene has left, and the runs it cuts are cut again against
    /// the new scene's arrays.
    #[test]
    fn replay_rebuilds_a_group_with_its_ids_re_derived() {
        let mut prev = Scene::default();
        prev.insert_primitive(quad_at(0.));
        prev.push_group(isolated_group());
        prev.insert_primitive(quad_at(100.));
        prev.insert_primitive(quad_at(120.));
        prev.pop_group();
        prev.insert_primitive(quad_at(300.));
        let replayed = 0..prev.len();
        prev.finish();

        let mut next = Scene::default();
        next.replay(replayed.clone(), &prev);
        next.finish();

        assert_eq!(next.segments, prev.segments);
        assert_eq!(steps_summary(&next), steps_summary(&prev));
        assert_eq!(next.groups.len(), 1);
        assert_eq!(quad_batches(&next), quad_batches(&prev));

        // Replayed under a group the new scene registered first, the same
        // operations take the next id along rather than the one they carried.
        let mut shifted = Scene::default();
        shifted.push_group(isolated_group());
        shifted.insert_primitive(quad_at(500.));
        shifted.pop_group();
        assert_eq!(shifted.group_depth(), 0);
        shifted.replay(replayed, &prev);
        assert_eq!(shifted.group_depth(), 0);
        shifted.finish();

        assert_eq!(shifted.groups.len(), 2);
        assert_eq!(
            steps_summary(&shifted),
            vec![
                "push", "run 1", "pop", "run 1", "push", "run 2", "pop", "run 1"
            ]
        );
        assert!(matches!(
            shifted.steps().nth(4),
            Some(SceneStep::PushGroup(_))
        ));
    }

    #[test]
    fn a_replayed_subtree_nests_inside_the_group_it_is_replayed_under() {
        let mut prev = Scene::default();
        prev.push_group(isolated_group());
        prev.insert_primitive(quad_at(0.));
        prev.pop_group();
        let replayed = 0..prev.len();
        prev.finish();

        let mut next = Scene::default();
        next.push_group(isolated_group());
        next.replay(replayed, &prev);
        assert_eq!(next.group_depth(), 1);
        next.pop_group();
        next.finish();

        assert_eq!(next.groups.len(), 2);
        assert_eq!(
            steps_summary(&next),
            vec!["push", "push", "run 1", "pop", "pop"]
        );
    }

    #[test]
    #[should_panic(expected = "closes a group it does not open")]
    fn replaying_half_a_group_is_refused() {
        let mut prev = Scene::default();
        prev.push_group(isolated_group());
        prev.insert_primitive(quad_at(0.));
        prev.pop_group();
        let half = 1..prev.len();
        prev.finish();

        let mut next = Scene::default();
        next.replay(half, &prev);
    }

    /// A group asking only for a fade over a subtree that does not overlap
    /// itself is indistinguishable from fading each primitive, which is what
    /// gpui has always done - so the group is taken back out and the scene is
    /// the one it would have been without it.
    #[test]
    fn a_fade_over_content_that_does_not_overlap_is_folded_into_it() {
        let mut scene = Scene::default();
        scene.insert_primitive(filled_quad_at(0.));
        scene.push_group(faded_group(0.5));
        scene.insert_primitive(filled_quad_at(100.));
        scene.insert_primitive(filled_quad_at(200.));
        scene.pop_group();
        scene.insert_primitive(filled_quad_at(300.));
        scene.finish();

        assert!(scene.groups.is_empty(), "the group left nothing behind");
        assert_eq!(scene.segments.len(), 1, "and no boundary either");
        assert_eq!(steps_summary(&scene), vec!["run 4"]);
        assert_eq!(quad_batches(&scene), vec![0..4]);

        let alphas: Vec<_> = scene
            .quads
            .iter()
            .map(|quad| quad.background.solid.a)
            .collect();
        assert_eq!(alphas, vec![1., 0.5, 0.5, 1.]);
        let borders: Vec<_> = scene.quads.iter().map(|quad| quad.border_color.a).collect();
        assert_eq!(borders, vec![1., 0.5, 0.5, 1.]);
    }

    #[test]
    fn a_fade_over_content_that_overlaps_itself_is_kept() {
        let mut scene = Scene::default();
        scene.push_group(faded_group(0.5));
        scene.insert_primitive(filled_quad_at(0.));
        scene.insert_primitive(filled_quad_at(5.));
        scene.pop_group();
        scene.finish();

        assert_eq!(
            scene.groups.len(),
            1,
            "two overlapping children need a target"
        );
        assert_eq!(steps_summary(&scene), vec!["push", "run 2", "pop"]);
        assert!(
            scene.quads.iter().all(|quad| quad.background.solid.a == 1.),
            "and nothing was faded per primitive, which would have been wrong"
        );
    }

    #[test]
    fn a_fade_over_one_primitive_is_folded_whatever_it_is() {
        let mut scene = Scene::default();
        scene.push_group(faded_group(0.25));
        scene.insert_primitive(monochrome_sprite_at(0., 0., 0));
        scene.pop_group();
        scene.finish();

        assert!(scene.groups.is_empty());
        assert_eq!(scene.monochrome_sprites[0].color.a, 0.25);
    }

    #[test]
    fn a_fade_over_a_polychrome_sprite_multiplies_its_own_opacity() {
        let mut scene = Scene::default();
        let mut sprite = polychrome_sprite_at(0., 0., 0);
        sprite.opacity = 0.5;
        scene.push_group(faded_group(0.5));
        scene.insert_primitive(sprite);
        scene.pop_group();
        scene.finish();

        assert!(scene.groups.is_empty());
        assert_eq!(scene.polychrome_sprites[0].opacity, 0.25);
    }

    /// A surface has no alpha of its own to fold a group's opacity into, so a
    /// group holding one is isolated however little else is in it.
    #[test]
    fn a_fade_over_a_surface_is_kept() {
        let mut scene = Scene::default();
        scene.push_group(faded_group(0.5));
        scene.insert_primitive(surface_at(0.));
        scene.pop_group();
        scene.finish();

        assert_eq!(scene.groups.len(), 1);
        assert_eq!(steps_summary(&scene), vec!["push", "run 1", "pop"]);
    }

    /// A subpixel-antialiased glyph blends per color channel against whatever
    /// is behind it, so its coverage is not one alpha that could be scaled.
    #[test]
    fn a_fade_over_a_subpixel_sprite_is_kept() {
        let mut scene = Scene::default();
        scene.push_group(faded_group(0.5));
        scene.insert_primitive(subpixel_sprite_at(0.));
        scene.pop_group();
        scene.finish();

        assert_eq!(scene.groups.len(), 1);
        assert_eq!(scene.subpixel_sprites[0].color.a, 1.);
    }

    /// The overlap test has to use the rectangle a shadow's gaussian tail
    /// actually reaches. Two shadows whose spread boxes are well clear of each
    /// other can still paint over one another out in the tails, and fading
    /// them one at a time would darken the seam.
    #[test]
    fn two_shadows_whose_blur_tails_meet_are_not_folded() {
        let mut overlapping = Scene::default();
        overlapping.push_group(faded_group(0.5));
        // Spread boxes 0..10 and 40..50, painted out to -45..55 and -5..95.
        overlapping.insert_primitive(shadow_at(0., 15.));
        overlapping.insert_primitive(shadow_at(40., 15.));
        overlapping.pop_group();
        overlapping.finish();

        assert_eq!(overlapping.groups.len(), 1, "the tails overlap");
        assert!(
            overlapping
                .shadows
                .iter()
                .all(|shadow| shadow.color.a == 0.)
        );

        // The same two rectangles, blurred by a radius whose tail falls short.
        let mut disjoint = Scene::default();
        disjoint.push_group(faded_group(0.5));
        disjoint.insert_primitive(shadow_at(0., 1.));
        disjoint.insert_primitive(shadow_at(40., 1.));
        disjoint.pop_group();
        disjoint.finish();

        assert!(disjoint.groups.is_empty(), "the tails fall short");
    }

    /// Blending, a filter and a backdrop filter cannot be expressed one
    /// primitive at a time however the primitives are laid out.
    #[test]
    fn a_group_that_asks_for_more_than_a_fade_is_never_folded() {
        for spec in [
            isolated_group(),
            GroupSpec {
                filter: Some(SceneFilter::Blur {
                    radius_x: 2.,
                    radius_y: 2.,
                }),
                ..Default::default()
            },
            GroupSpec {
                backdrop_filter: Some(SceneFilter::ColorMatrix([0.; 20])),
                ..Default::default()
            },
        ] {
            let mut scene = Scene::default();
            scene.push_group(spec.clone());
            scene.insert_primitive(filled_quad_at(0.));
            scene.pop_group();
            scene.finish();
            assert_eq!(scene.groups.len(), 1, "{spec:?} needs a target of its own");
        }
    }

    /// A group holding one quad at `0..10 x 0..10`, filtered and masked as
    /// asked, and the bounds its target came out at.
    fn filtered_group_bounds(
        filter: Option<SceneFilter>,
        mask: Option<Bounds<ScaledPixels>>,
    ) -> Bounds<ScaledPixels> {
        let mut scene = Scene::default();
        scene.push_group(GroupSpec {
            // A blend mode so the group is never folded away: a folded group
            // has no bounds to look at.
            blend: BlendMode {
                mix: MixMode::Multiply,
                compose: ComposeMode::SrcOver,
            },
            filter,
            mask,
            ..Default::default()
        });
        scene.insert_primitive(filled_quad_at(0.));
        scene.pop_group();
        scene.finish();
        assert_eq!(scene.groups.len(), 1, "the group had to survive");
        scene.groups[0].bounds
    }

    fn scaled(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(width), ScaledPixels(height)),
        }
    }

    /// A blurred group is bigger than the ink inside it, and a target sized to
    /// the ink would cut the gaussian tail off at a hard edge.
    #[test]
    fn a_blurred_group_grows_by_the_gaussian_tail() {
        assert_eq!(
            filtered_group_bounds(
                Some(SceneFilter::Blur {
                    radius_x: 4.,
                    radius_y: 1.,
                }),
                None,
            ),
            scaled(-12., -3., 34., 16.),
            "three standard deviations on each axis, the same tail a box \
             shadow of the same standard deviation is drawn out to"
        );
    }

    /// A drop shadow reaches out to its own tail around its offset, and the
    /// group itself is still painted where it always was - and the tail goes
    /// round the un-offset silhouette too, because that is the image the
    /// shadow is read back out of.
    #[test]
    fn a_drop_shadow_group_grows_by_the_offset_and_the_tail() {
        assert_eq!(
            filtered_group_bounds(
                Some(SceneFilter::DropShadow {
                    offset_x: 6.,
                    offset_y: -2.,
                    radius: 2.,
                    color: Hsla::default(),
                }),
                None,
            ),
            // The quad and its offset copy cover 0..16 x -2..10 between them,
            // and three standard deviations is six.
            scaled(-6., -8., 28., 24.),
        );
    }

    /// Blurring is not licence to paint outside an ancestor's clip: everything
    /// inside the group was masked when it was painted, and CSS clips a
    /// filtered result exactly as it clips an unfiltered one.
    #[test]
    fn a_blurred_group_may_not_grow_past_its_content_mask() {
        assert_eq!(
            filtered_group_bounds(
                Some(SceneFilter::Blur {
                    radius_x: 4.,
                    radius_y: 4.,
                }),
                Some(scaled(-5., -1000., 2000., 2000.)),
            ),
            scaled(-5., -12., 27., 34.),
            "the mask cuts the tail off on the left and nowhere else"
        );
    }

    /// The growth belongs to the filter. A group without one is worth exactly
    /// what was painted in it, which is what it has always been worth.
    #[test]
    fn an_unfiltered_group_is_still_the_size_of_its_content() {
        assert_eq!(filtered_group_bounds(None, None), scaled(0., 0., 10., 10.));
        assert_eq!(
            filtered_group_bounds(Some(SceneFilter::ColorMatrix([0.; 20])), None),
            scaled(0., 0., 10., 10.),
            "a colour matrix moves no pixel from where it was"
        );
    }

    /// A chain grows once per link, in order, because a drop shadow of a blur
    /// is a shadow of something that had already spread.
    #[test]
    fn a_chain_grows_through_every_filter_in_it() {
        assert_eq!(
            filtered_group_bounds(
                Some(SceneFilter::Chain(vec![
                    SceneFilter::Blur {
                        radius_x: 1.,
                        radius_y: 1.,
                    },
                    SceneFilter::DropShadow {
                        offset_x: 10.,
                        offset_y: 0.,
                        radius: 1.,
                        color: Hsla::default(),
                    },
                ])),
                None,
            ),
            // The blur takes 0..10 out to -3..13 both ways; that and its
            // offset copy cover -3..23 x -3..13, and the shadow's own three
            // standard deviations go round all of it.
            scaled(-6., -6., 32., 22.),
        );
    }

    /// An outer fade over an inner group that survived has to be applied to the
    /// inner group's composited result, which is one image and not a set of
    /// primitives to fade.
    #[test]
    fn a_fade_around_a_group_that_survived_is_kept() {
        let mut scene = Scene::default();
        scene.push_group(faded_group(0.5));
        scene.push_group(isolated_group());
        scene.insert_primitive(filled_quad_at(0.));
        scene.pop_group();
        scene.pop_group();
        scene.finish();

        assert_eq!(scene.groups.len(), 2);
        assert_eq!(
            steps_summary(&scene),
            vec!["push", "push", "run 1", "pop", "pop"]
        );
    }

    /// An inner group that folded away leaves nothing behind, so the outer one
    /// is judged on the primitives themselves and folds too - and both fades
    /// land on them.
    #[test]
    fn nested_fades_that_both_fold_multiply() {
        let mut scene = Scene::default();
        scene.push_group(faded_group(0.5));
        scene.push_group(faded_group(0.5));
        scene.insert_primitive(filled_quad_at(0.));
        scene.pop_group();
        scene.insert_primitive(filled_quad_at(100.));
        scene.pop_group();
        scene.finish();

        assert!(scene.groups.is_empty());
        assert_eq!(steps_summary(&scene), vec!["run 2"]);
        let alphas: Vec<_> = scene
            .quads
            .iter()
            .map(|quad| quad.background.solid.a)
            .collect();
        assert_eq!(alphas, vec![0.25, 0.5]);
    }

    fn layer_bounds() -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(0.), ScaledPixels(0.)),
            size: size(ScaledPixels(500.), ScaledPixels(500.)),
        }
    }

    /// Inside a layer no primitive is inserted into the bounds tree - they all
    /// take the layer's order - so a layer's contents are held in paint order
    /// by the stable sort alone. A group nested wholly inside a layer is fine,
    /// and so is a layer nested wholly inside a group.
    #[test]
    fn a_group_may_nest_inside_a_layer_and_a_layer_inside_a_group() {
        let mut scene = Scene::default();
        scene.push_layer(layer_bounds());
        scene.push_group(isolated_group());
        scene.insert_primitive(quad_at(0.));
        scene.pop_group();
        scene.pop_layer();

        scene.push_group(isolated_group());
        scene.push_layer(layer_bounds());
        scene.insert_primitive(quad_at(100.));
        scene.pop_layer();
        scene.pop_group();
        scene.finish();

        assert_eq!(scene.groups.len(), 2);
        assert_eq!(
            steps_summary(&scene),
            vec!["push", "run 1", "pop", "push", "run 1", "pop"]
        );
    }

    #[test]
    #[should_panic(expected = "may not straddle a layer boundary")]
    fn a_group_may_not_be_closed_inside_a_layer_it_was_opened_outside() {
        let mut scene = Scene::default();
        scene.push_group(isolated_group());
        scene.push_layer(layer_bounds());
        scene.insert_primitive(quad_at(0.));
        scene.pop_group();
    }

    #[test]
    #[should_panic(expected = "may not straddle a layer boundary")]
    fn a_layer_may_not_be_closed_inside_a_group_it_was_opened_outside() {
        let mut scene = Scene::default();
        scene.push_layer(layer_bounds());
        scene.push_group(isolated_group());
        scene.insert_primitive(quad_at(0.));
        scene.pop_layer();
        scene.pop_group();
    }

    /// A brush's `screen_to_brush` is built by inverting what the caller
    /// authored, so an inverse that is wrong puts every image on screen in the
    /// wrong place at the wrong size.
    #[test]
    fn inverting_a_transformation_undoes_it() {
        let transformation = TransformationMatrix::unit()
            .translate(point(ScaledPixels(30.), ScaledPixels(-12.)))
            .scale(size(2., 4.))
            .rotate(Radians(0.7));
        let inverse = transformation.invert().expect("it is invertible");

        for original in [
            point(Pixels(0.), Pixels(0.)),
            point(Pixels(17.), Pixels(-3.)),
            point(Pixels(-100.5), Pixels(64.25)),
        ] {
            let round_tripped = inverse.apply(transformation.apply(original));
            assert!(
                (round_tripped.x.0 - original.x.0).abs() < 1e-3
                    && (round_tripped.y.0 - original.y.0).abs() < 1e-3,
                "{original:?} came back as {round_tripped:?}"
            );
        }
    }

    #[test]
    fn a_transformation_that_collapses_the_plane_has_no_inverse() {
        assert_eq!(
            TransformationMatrix::unit().scale(size(0., 1.)).invert(),
            None
        );
    }
}
