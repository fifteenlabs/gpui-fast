// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ClipPath, ContentMask, Corners, DevicePixels,
    Edges, HitboxId, Hsla, NativeViewId, Path, PathTiles, Pixels, Point, Radians, ScaledPixels,
    Size, TimedTransformation, bounds_tree::BoundsTree, point, size,
};
use smallvec::SmallVec;
use std::{
    collections::VecDeque, fmt::Debug, iter::Peekable, ops::Range, rc::Rc, slice, time::Duration,
};

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

impl PaddedBool32 {
    /// The wrapped boolean.
    pub fn as_bool(self) -> bool {
        self.0 != 0
    }
}

struct Layer {
    primitives: SmallVec<[Range<usize>; 1]>,
    bounds: Bounds<ScaledPixels>,
    max_kind: PrimitiveKind,
    max_shared_kind: PrimitiveKind,
    /// Whether a native view was painted in the layer, which then can't be
    /// drawn as one: what was painted after the view has to cover it.
    has_native_view: bool,
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

/// The eight per-kind arrays a [`Scene`] appends primitives to, in the one
/// order the whole file spells them.
///
/// The list was written out ten times before this: [`Segment`]'s fields, its
/// `is_empty` and `len`, three ways of building one, the boxes a run covers,
/// and [`BatchIterator`]'s cursors and its eight arms. Adding a ninth kind of
/// primitive meant finding all ten and getting all ten right.
macro_rules! with_primitive_kinds {
    ($callback:ident) => {
        $callback! {
            shadows,
            quads,
            paths,
            underlines,
            monochrome_sprites,
            subpixel_sprites,
            polychrome_sprites,
            surfaces,
            native_views
        }
    };
}

macro_rules! define_segment {
    ($($kind:ident),+ $(,)?) => {
        /// The primitives painted between two group boundaries: one contiguous
        /// range of each per-kind array.
        ///
        /// Eight ranges are enough to name a subtree because
        /// [`Scene::insert_primitive`] appends to those arrays and nothing else
        /// does, so primitives land in paint order, and a group is
        /// closure-scoped. Everything painted between a [`Scene::push_group`]
        /// and its matching [`Scene::pop_group`] therefore occupies a
        /// contiguous range of every array, before any sorting - which is why
        /// isolating a subtree needs no second scene, no re-batching, and no
        /// change to [`Scene::insert_primitive`].
        #[derive(Clone, Debug, Default, PartialEq, Eq)]
        #[expect(missing_docs)]
        pub struct Segment {
            $(pub $kind: Range<usize>,)+
        }

        impl Segment {
            /// Whether this run holds no primitives at all.
            pub fn is_empty(&self) -> bool {
                true $(&& self.$kind.is_empty())+
            }

            /// How many primitives this run holds, across every kind.
            pub fn len(&self) -> usize {
                0 $(+ self.$kind.len())+
            }
        }

        impl Scene {
            /// Every per-kind array from where `start` opened it to where it
            /// ends now: the run painted since `start` was taken.
            fn since(&self, start: &Segment) -> Segment {
                Segment {
                    $($kind: start.$kind.start..self.$kind.len(),)+
                }
            }

            /// Where every per-kind array currently ends, as an empty run
            /// starting there.
            fn end_indices(&self) -> Segment {
                Segment {
                    $($kind: self.$kind.len()..self.$kind.len(),)+
                }
            }
        }
    };
}

with_primitive_kinds!(define_segment);

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

/// How long a chain of colour matrices one group may carry. CSS `filter` lists
/// are short, and the whole chain travels to the GPU as one small constant.
pub const MAX_GROUP_COLOR_MATRICES: usize = 8;

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
    /// Padding, so the struct carries no compiler-inserted bytes and matches
    /// what the shader declares.
    pub pad: [u32; 3],
    /// [`MAX_GROUP_COLOR_MATRICES`] row-major 4x5 matrices, flattened.
    pub matrices: [f32; 160],
}

impl GroupFilter {
    /// No filter at all: the group's result is composited as it came out.
    pub const NONE: Self = GroupFilter {
        matrix_count: 0,
        pad: [0; 3],
        matrices: [0.; 160],
    };

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

/// One step of a [`SceneFilter`], flattened out of the tree it arrives as and
/// with each run of adjacent colour matrices collapsed into a single step.
///
/// Collapsing the matrices is not the same as multiplying them together: the
/// shader still applies them one at a time with a clamp between, because CSS
/// clamps between filter primitives. It is only that a run of them costs one
/// pass rather than one pass each.
#[derive(Clone, Debug)]
pub enum FilterOp {
    /// A run of colour matrices, applied one at a time with a clamp between.
    Matrices(GroupFilter),
    /// A separable gaussian, one pass per axis.
    Blur {
        /// The horizontal standard deviation, in device pixels.
        sigma_x: f32,
        /// The vertical standard deviation, in device pixels.
        sigma_y: f32,
    },
    /// The source blurred, tinted and offset, drawn behind the source.
    DropShadow {
        /// How far the shadow is displaced from the source, in device pixels.
        offset: Point<f32>,
        /// The blur's standard deviation, in device pixels.
        sigma: f32,
        /// What the shadow is tinted.
        color: Hsla,
    },
}

impl FilterOp {
    /// The steps a filter compiles to, in order.
    ///
    /// A run of colour matrices longer than [`MAX_GROUP_COLOR_MATRICES`] is not
    /// a filter this cannot express: it is two steps rather than one, and the
    /// second reads what the first wrote. Nothing here can fail, which is why
    /// there is no longer a "this filter is not implemented" line to log.
    pub fn flatten(filter: &SceneFilter) -> Vec<FilterOp> {
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
    /// This group moved by `delta`, with what it covers.
    pub(crate) fn translated(&self, delta: Point<ScaledPixels>) -> Self {
        let mv = |bounds: Bounds<ScaledPixels>| Bounds {
            origin: bounds.origin + delta,
            size: bounds.size,
        };
        Self {
            bounds: mv(self.bounds),
            mask: self.mask.map(mv),
            ..self.clone()
        }
    }

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

/// Says, once per process, that a path arrived carrying an image or gradient
/// brush this renderer cannot resolve.
///
/// [`crate::Window::paint_path_with_image`] and
/// [`crate::Window::paint_path_with_gradient`] resolve a brush on every backend -
/// the image or the baked ramp lands in the sprite atlas whatever is drawing -
/// but only the Metal renderer reads one back out. Here the path is filled with
/// its `color`: nothing at all for an image brush, and the flat colour halfway
/// along the stop list for a gradient. Either way a picture is quietly losing a
/// background, which is worth a line in the log.
pub fn report_unresolved_path_brush() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        log::warn!(
            "a path carries an image or gradient brush, which this renderer \
             cannot resolve; it is filled with its solid colour instead"
        );
    });
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
    /// The union of the final bounds of the groups inside this one that
    /// survived, and so also whether any of them did: one that folded away
    /// left nothing behind and never reaches here.
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
    pub(crate) primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<usize>,
    pending_layers: VecDeque<Layer>,
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
    pub paths: Vec<PaintedPath>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    pub native_views: Vec<PaintNativeView>,
    animations: Vec<SpriteAnimation>,
    pub(crate) sort_scratch: crate::fast::scene::SortScratch,
    pub(crate) fast_rounded_clip: crate::fast::rounded_clip::SceneClips,
    pub(crate) fast_quad_pieces: Option<smallvec::SmallVec<[Quad; 5]>>,
    pub layers: crate::fast::layers::scene::SceneLayers,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        crate::fast::rounded_clip::clear(self);
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.pending_layers.clear();
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
        crate::fast::layers::scene::SceneLayers::clear(&mut self.layers);
        self.native_views.clear();
        self.animations.clear();
    }

    /// Inserts a sprite that [`Self::advance_animations`] keeps moving. The
    /// sprite's color is taken as its color before the animation's alpha.
    pub(crate) fn insert_animated_sprite(
        &mut self,
        mut sprite: MonochromeSprite,
        mut animation: SpriteAnimation,
    ) {
        sprite.animation = SpriteAnimationId(self.animations.len() as u32);
        animation.base_color = sprite.color;
        if self.try_insert_primitive(sprite.into()) {
            self.animations.push(animation);
        }
    }

    /// How long the window may wait between presents of this scene while only
    /// its animations need a frame: `None` when nothing in it animates, and
    /// zero when an animation wants every frame.
    pub(crate) fn animation_frame_interval(&self) -> Option<Duration> {
        self.animations
            .iter()
            .map(|animation| {
                animation
                    .transformation
                    .min_frame_interval()
                    .unwrap_or_default()
            })
            .min()
    }

    /// Moves every animated sprite to its pose at `elapsed` time on the app's
    /// synced animation clock, without the scene being rebuilt.
    pub(crate) fn advance_animations(&mut self, elapsed: Duration) {
        for animation in &self.animations {
            let Some(sprite) = self.monochrome_sprites.get_mut(animation.sprite_index) else {
                continue;
            };
            let pose = animation.transformation.at(elapsed);
            sprite.transformation = pose.into_matrix(animation.center, animation.scale_factor);
            sprite.color = animation.base_color.opacity(pose.alpha_factor());
        }
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        self.layer_stack.push(self.pending_layers.len());
        self.pending_layers.push_back(Layer {
            primitives: SmallVec::new(),
            bounds: Bounds::default(),
            max_kind: PrimitiveKind::Shadow,
            max_shared_kind: PrimitiveKind::NativeView,
            has_native_view: false,
        });
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        if self.layer_stack.is_empty() {
            while let Some(layer) = self.pending_layers.pop_front() {
                self.insert_layer(layer);
            }
        }
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
    /// [`Scene::pop_group`] asserts it. A layer's primitives are held back
    /// until the outermost layer closes and then share one draw order, drawn
    /// by kind; a group inside a layer cuts the layers open around it at both
    /// of its boundaries (see `split_open_layers`), so the group's run is
    /// contiguous and the parts of the layer before, inside and after it are
    /// drawn in paint order.
    ///
    /// # Deferred draws
    ///
    /// A [`crate::Window::defer_draw`] issued inside a group escapes it: the
    /// deferred element is painted later, at the top level, and so is not part
    /// of the group and receives none of its opacity, blending or filtering.
    /// It cannot happen through [`crate::Window::with_isolated_group`], which
    /// runs in the paint phase while `defer_draw` is prepaint-only.
    pub fn push_group(&mut self, mut spec: GroupSpec) -> GroupId {
        self.split_open_layers();
        // Stamped here for the reason `insert_primitive` stamps primitives: a
        // replayed group carries the previous frame's clip id.
        spec.clip = self.current_clip();
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

        self.split_open_layers();
        // Measured once. Both of the next two want the group's own run and the
        // boxes its primitives paint into, and exactly one of them runs: a
        // group that folds away needed the boxes to prove its contents do not
        // overlap, and one that survives needs them to size its target.
        let content = self.group_content(&group);
        let mut boxes = self.content_boxes(&content);

        if !self.fold_group_if_unobservable(&group, &content, &mut boxes) {
            self.grow_group_to_its_content(&group, &boxes);
            self.close_run();
            self.steps.push(SceneStepRecord::PopGroup);
            let bounds = self.groups[group.id.0 as usize].bounds;
            if let Some(parent) = self.group_stack.last_mut() {
                parent.child_bounds = Some(match parent.child_bounds {
                    Some(existing) => existing.union(&bounds),
                    None => bounds,
                });
            }
        }
        self.paint_operations.push(PaintOperation::EndGroup);
    }

    /// Inserts every layer still waiting to be, and starts each layer still
    /// being painted afresh, so that everything painted so far is in the
    /// per-kind arrays.
    ///
    /// A layer's primitives reach those arrays only once the outermost layer
    /// around them closes, when the layer takes one place in the drawing order
    /// for everything it painted. A group boundary inside a layer cannot wait
    /// that long: a group is a contiguous run of each array, and the run has
    /// to be cut where the boundary is. So the open layers are cut there
    /// instead, each part taking its own place in the drawing order, which
    /// draws the parts in paint order wherever they overlap.
    fn split_open_layers(&mut self) {
        let depth = self.layer_stack.len();
        if depth == 0 {
            return;
        }
        while let Some(layer) = self.pending_layers.pop_front() {
            self.insert_layer(layer);
        }
        self.layer_stack.clear();
        for _ in 0..depth {
            self.layer_stack.push(self.pending_layers.len());
            self.pending_layers.push_back(Layer {
                primitives: SmallVec::new(),
                bounds: Bounds::default(),
                max_kind: PrimitiveKind::Shadow,
                max_shared_kind: PrimitiveKind::NativeView,
                has_native_view: false,
            });
        }
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
    pub(crate) fn close_run(&mut self) {
        let run = self.since(&self.open_run);
        self.open_run = self.end_indices();
        if run.is_empty() {
            return;
        }
        let id = SegmentId(self.segments.len() as u32);
        self.segments.push(run);
        self.steps.push(SceneStepRecord::Run(id));
    }

    /// The run a group's own primitives occupy, at the moment it is popped.
    fn group_content(&self, group: &OpenGroup) -> Segment {
        self.since(&group.content)
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
    fn fold_group_if_unobservable(
        &mut self,
        group: &OpenGroup,
        content: &Segment,
        boxes: &mut [Bounds<ScaledPixels>],
    ) -> bool {
        let spec = &self.groups[group.id.0 as usize];
        if group.child_bounds.is_some() || !spec.is_foldable() {
            return false;
        }
        let opacity = spec.opacity;
        if !content.subpixel_sprites.is_empty()
            || !content.surfaces.is_empty()
            || !content.native_views.is_empty()
        {
            return false;
        }
        if !Self::is_overlap_free(boxes) {
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
            self.fold_opacity(content, opacity);
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
    fn is_overlap_free(boxes: &mut [Bounds<ScaledPixels>]) -> bool {
        const SCAN_BUDGET: usize = 4096;

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
        // A shadow's `bounds` is the area it paints, its blur tail included, so
        // it is measured like everything else; a path's mask is on its paint.
        for path in &self.paths[content.paths.clone()] {
            boxes.push(path.painted_bounds().intersect(&path.content_mask().bounds));
        }
        macro_rules! push_bounds {
            ($($kind:ident),+ $(,)?) => {$(
                for primitive in &self.$kind[content.$kind.clone()] {
                    boxes.push(primitive.bounds.intersect(&primitive.content_mask.bounds));
                }
            )+};
        }
        push_bounds!(
            shadows,
            quads,
            underlines,
            monochrome_sprites,
            subpixel_sprites,
            polychrome_sprites,
            surfaces,
            native_views,
        );
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
    fn grow_group_to_its_content(&mut self, group: &OpenGroup, boxes: &[Bounds<ScaledPixels>]) {
        let mut painted: Option<Bounds<ScaledPixels>> =
            group.child_bounds.filter(|bounds| !bounds.is_empty());
        for bounds in boxes {
            if bounds.is_empty() {
                continue;
            }
            painted = Some(match painted {
                Some(painted) => painted.union(bounds),
                None => *bounds,
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
            path.paint.color = path.paint.color.opacity(opacity);
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

    fn insert_layer(&mut self, layer: Layer) {
        if layer.primitives.is_empty() {
            return;
        }
        if layer.has_native_view {
            self.insert_layer_around_native_views(layer);
            return;
        }
        let order = self.primitive_bounds.insert(layer.bounds);
        for index in layer.primitives.into_iter().flatten() {
            // fast: the painted primitive keeps its draw order too (fast::layers reads it).
            if let PaintOperation::Primitive(primitive) = &mut self.paint_operations[index] {
                crate::fast::scene::set_draw_order(primitive, order);
                let primitive = primitive.clone();
                self.push_primitive(primitive, order);
            }
        }
    }

    /// Inserts a layer that a native view was painted in.
    ///
    /// A layer's primitives share one place in the drawing order, where they
    /// are drawn by kind and not in the order they were painted. A native
    /// view erases what is drawn before it, so the layer is inserted in parts:
    /// what was painted before each view, the view, and what was painted after.
    fn insert_layer_around_native_views(&mut self, layer: Layer) {
        let mut part = Vec::new();
        for index in layer.primitives.into_iter().flatten() {
            let PaintOperation::Primitive(primitive) = &self.paint_operations[index] else {
                continue;
            };
            let primitive = (index, primitive.clone()); // fast: with its index
            if primitive.1.kind() == PrimitiveKind::NativeView {
                self.insert_layer_part(std::mem::take(&mut part));
                self.insert_layer_part(vec![primitive]);
            } else {
                part.push(primitive);
            }
        }
        self.insert_layer_part(part);
    }

    fn insert_layer_part(&mut self, primitives: Vec<(usize, Primitive)>) {
        let Some((_, first)) = primitives.first() else {
            return;
        };
        let mut bounds = first.clipped_bounds();
        let mut max_kind = PrimitiveKind::Shadow;
        let mut max_shared_kind = PrimitiveKind::NativeView;
        for (_, primitive) in &primitives {
            let kind = primitive.kind();
            bounds = bounds.join(&primitive.clipped_bounds());
            max_kind = max_kind.max(kind);
            max_shared_kind = max_shared_kind.min(kind.max_shared_kind());
        }
        let order = self.primitive_bounds.insert(bounds);
        for (index, primitive) in primitives {
            // fast: the painted primitive keeps its draw order too (fast::layers reads it).
            if let PaintOperation::Primitive(painted) = &mut self.paint_operations[index] {
                crate::fast::scene::set_draw_order(painted, order);
            }
            self.push_primitive(primitive, order);
        }
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let Some(primitive) = crate::fast::scene::quad_piece(self, primitive.into()) else {
            return;
        };
        self.try_insert_primitive(primitive);
    }

    /// Returns whether the primitive was kept, rather than clipped away.
    fn try_insert_primitive(&mut self, mut primitive: Primitive) -> bool {
        let clipped_bounds = primitive.clipped_bounds();
        if clipped_bounds.is_empty() {
            return false;
        }
        // Every primitive reaches the scene through here, so stamping the clip
        // here is what keeps `Window::paint_*` free of a clip argument. It is
        // an unconditional overwrite rather than a fill-if-unset because
        // `replay` hands back primitives carrying the *previous* frame's ids,
        // which name clips that no longer exist.
        primitive.set_clip(self.current_clip());

        let kind = primitive.kind();
        if let Some(&layer) = self.layer_stack.last()
            && let Some(layer) = self.pending_layers.get_mut(layer)
        {
            layer.bounds = layer.bounds.join(&clipped_bounds);
            layer.max_kind = layer.max_kind.max(kind);
            layer.max_shared_kind = layer.max_shared_kind.min(kind.max_shared_kind());
            layer.has_native_view |= kind == PrimitiveKind::NativeView;
            let index = self.paint_operations.len();
            match layer.primitives.last_mut() {
                Some(range) if range.end == index => range.end += 1,
                _ => layer.primitives.push(index..index + 1),
            }
        } else {
            let order = self.primitive_bounds.insert(clipped_bounds);
            // fast: the painted primitive keeps its draw order too (fast::layers reads it).
            crate::fast::scene::set_draw_order(&mut primitive, order);
            self.push_primitive(primitive.clone(), order);
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
        true
    }

    // fast: fast::scene::insert_quad_pieces gives a split quad one place in the drawing order.
    pub(crate) fn fast_top_level(&self) -> bool {
        self.layer_stack.is_empty()
    }

    /// Pushes a primitive painted at `order` to its batch and the paint operations.
    pub(crate) fn fast_push_painted(&mut self, primitive: &Primitive, order: DrawOrder) {
        let mut primitive = primitive.clone();
        crate::fast::scene::set_draw_order(&mut primitive, order);
        primitive.set_clip(self.current_clip());
        match &primitive {
            Primitive::Quad(quad) => self.quads.push(*quad),
            Primitive::MonochromeSprite(sprite) => self.monochrome_sprites.push(*sprite),
            Primitive::SubpixelSprite(sprite) => self.subpixel_sprites.push(*sprite),
            Primitive::PolychromeSprite(sprite) => self.polychrome_sprites.push(*sprite),
            _ => self.push_primitive(primitive.clone(), order),
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    pub(crate) fn fast_push_quad(&mut self, quad: Quad, order: DrawOrder) {
        let quad = Quad {
            order,
            clip: self.current_clip(),
            ..quad
        };
        self.quads.push(quad);
        self.paint_operations
            .push(PaintOperation::Primitive(Primitive::Quad(quad)));
    }

    fn push_primitive(&mut self, primitive: Primitive, order: DrawOrder) {
        match primitive {
            Primitive::Shadow(shadow) => self.shadows.push(Shadow { order, ..shadow }),
            Primitive::Quad(quad) => self.quads.push(Quad { order, ..quad }),
            Primitive::Path(mut path) => {
                path.order = order;
                self.paths.push(path);
            }
            Primitive::Underline(underline) => {
                self.underlines.push(Underline { order, ..underline })
            }
            Primitive::MonochromeSprite(sprite) => self
                .monochrome_sprites
                .push(MonochromeSprite { order, ..sprite }),
            Primitive::SubpixelSprite(sprite) => self
                .subpixel_sprites
                .push(SubpixelSprite { order, ..sprite }),
            Primitive::PolychromeSprite(sprite) => self
                .polychrome_sprites
                .push(PolychromeSprite { order, ..sprite }),
            Primitive::Surface(mut surface) => {
                surface.order = order;
                self.surfaces.push(surface);
            }
            Primitive::NativeView(native_view) => self.native_views.push(PaintNativeView {
                order,
                ..native_view
            }),
        }
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        crate::fast::layers::paint::replay_layers(self, range.clone(), prev_scene);
        let mut quads = crate::fast::scene::QuadRun::default(); // fast: split quads insert together
        let operations = &prev_scene.paint_operations[range]; // fast
        Self::debug_assert_groups_are_balanced(operations);
        let mut next = 0; // fast
        while let Some(operation) = operations.get(next) {
            next += 1; // fast
            if let PaintOperation::StartLayer(_) = operation
                && self.layer_stack.is_empty()
                && prev_scene.animations.is_empty()
            {
                quads.flush(self); // fast: a layer replayed in one pass
                if let Some(len) =
                    crate::fast::scene::replay_flat_layer(self, &operations[next - 1..])
                {
                    next += len - 1;
                    continue;
                }
            }
            if let PaintOperation::Primitive(Primitive::Quad(quad)) = operation {
                quads.push(self, *quad); // fast
                continue;
            }
            quads.flush(self); // fast
            // Animation ids index the scene they were inserted into.
            if let PaintOperation::Primitive(Primitive::MonochromeSprite(sprite)) = operation
                && let Some(animation) = prev_scene.animations.get(sprite.animation.0 as usize)
            {
                self.insert_animated_sprite(*sprite, *animation);
                continue;
            }
            self.apply(operation);
        }
        quads.flush(self); // fast
    }

    /// Records one paint operation as painting it did.
    pub(crate) fn apply(&mut self, operation: &PaintOperation) {
        match operation {
            PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
            PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
            PaintOperation::EndLayer => self.pop_layer(),
            PaintOperation::FastRoundedClip(fast_clip) => {
                crate::fast::rounded_clip::record(self, *fast_clip)
            }
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
        debug_assert!(self.layer_stack.is_empty(), "unbalanced scene layers");
        debug_assert!(
            self.group_stack.is_empty(),
            "a scene was finished with {} group(s) still open",
            self.group_stack.len()
        );
        self.close_run();
        crate::fast::scene::sort_in_drawing_order(self);
        self.keep_first_painted_native_views();

        // Sorting moved the sprites, so animations learn where theirs ended up.
        if !self.animations.is_empty() {
            for (index, sprite) in self.monochrome_sprites.iter().enumerate() {
                if let Some(animation) = self.animations.get_mut(sprite.animation.0 as usize) {
                    animation.sprite_index = index;
                }
            }
        }
    }

    /// Drops every occurrence of a native view but the one painted first,
    /// because a platform view can only be in one place.
    ///
    /// That is the occurrence whose hitbox takes the view's mouse input. It
    /// isn't always the first in the drawing order, which depends on what each
    /// occurrence overlaps.
    pub(crate) fn keep_first_painted_native_views(&mut self) {
        let native_views = &self.native_views;
        let is_repeated = native_views.iter().enumerate().any(|(index, native_view)| {
            native_views[..index]
                .iter()
                .any(|earlier| earlier.id == native_view.id)
        });
        if !is_repeated {
            return;
        }
        let mut first_painted: Vec<(NativeViewId, HitboxId)> = Vec::new();
        for operation in &self.paint_operations {
            if let PaintOperation::Primitive(Primitive::NativeView(native_view)) = operation
                && !first_painted.iter().any(|(id, _)| *id == native_view.id)
            {
                first_painted.push((native_view.id, native_view.hitbox_id));
            }
        }
        let kept: Vec<bool> = self
            .native_views
            .iter()
            .map(|native_view| first_painted.contains(&(native_view.id, native_view.hitbox_id)))
            .collect();
        // The runs a scene's groups cut it into index the native views too, so
        // each run's range shrinks by the views dropped from it.
        let kept_before = |index: usize| kept[..index].iter().filter(|kept| **kept).count();
        for segment in &mut self.segments {
            segment.native_views =
                kept_before(segment.native_views.start)..kept_before(segment.native_views.end);
        }
        let mut index = 0;
        self.native_views.retain(|_| {
            index += 1;
            kept[index - 1]
        });
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
        let (native_views, fast_native_indices) =
            crate::fast::rounded_clip_scene::native_batches(self);
        // A scene the isolated groups did not cut is one run over everything,
        // batched exactly as upstream batches it, native views included; one
        // they did cut is batched a run at a time, so no batch crosses a group
        // boundary.
        let whole = (self.segments.len() <= 1).then(|| BatchIterator {
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
            native_views_start: 0,
            native_views_iter: native_views.iter().peekable(),
            fast_native_indices,
        });
        let runs = if whole.is_some() {
            &[][..]
        } else {
            &self.segments[..]
        };
        whole.into_iter().flatten().chain(
            runs.iter()
                .flat_map(move |run| BatchIterator::over(self, run)),
        )
    }

    /// The batches of one run, for a renderer walking [`Scene::steps`] rather
    /// than [`Scene::batches`]: same batches, in the same order, for the
    /// stretch of the scene between two group boundaries.
    pub fn run_batches(&self, run: &Segment) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator::over(self, run)
    }
}

/// The depth step between consecutive quads in a frame with `quad_count` quads.
pub fn quad_depth_scale(quad_count: usize) -> f32 {
    let denominator = (quad_count.min((1 << 24) - 1) + 1).next_power_of_two();
    1.0 / denominator as f32
}

/// Maps a quad's index in [`Scene::quads`] to depth, with greater values closer.
pub fn quad_depth(quad_id: u32, scale: f32) -> f32 {
    ((quad_id + 1) as f32 * scale).min(1.0)
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
    NativeView,
}

impl PrimitiveKind {
    /// We guarantee that overlapping primitives are drawn in paint order.
    /// But, to improve GPU draw batching, the renderer sorts primitives that
    /// share a draw order by kind, then by atlas texture, and only then in
    /// paint order. So, a primitive can only share the draw order of the
    /// topmost primitive it overlaps if that primitive's kind is at most the
    /// one returned here. Otherwise, it needs a new draw order.
    fn max_shared_kind(self) -> Self {
        match self {
            Self::MonochromeSprite => Self::Underline,
            Self::SubpixelSprite => Self::MonochromeSprite,
            Self::PolychromeSprite => Self::SubpixelSprite,
            kind => kind,
        }
    }
}

#[derive(Clone)]
pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
    FastRoundedClip(crate::fast::rounded_clip::ClipOperation),
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
    Path(PaintedPath),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
    NativeView(PaintNativeView),
}

#[expect(missing_docs)]
impl Primitive {
    pub(crate) fn kind(&self) -> PrimitiveKind {
        match self {
            Primitive::Shadow(_) => PrimitiveKind::Shadow,
            Primitive::Quad(_) => PrimitiveKind::Quad,
            Primitive::Path(_) => PrimitiveKind::Path,
            Primitive::Underline(_) => PrimitiveKind::Underline,
            Primitive::MonochromeSprite(_) => PrimitiveKind::MonochromeSprite,
            Primitive::SubpixelSprite(_) => PrimitiveKind::SubpixelSprite,
            Primitive::PolychromeSprite(_) => PrimitiveKind::PolychromeSprite,
            Primitive::Surface(_) => PrimitiveKind::Surface,
            Primitive::NativeView(_) => PrimitiveKind::NativeView,
        }
    }

    /// Stamps the primitive with the clip path it is drawn under. A native
    /// view is a platform view showing through the scene, which no clip path
    /// can shape, so it carries none.
    pub(crate) fn set_clip(&mut self, clip: ClipId) {
        match self {
            Primitive::Shadow(shadow) => shadow.clip = clip,
            Primitive::Quad(quad) => quad.clip = clip,
            Primitive::Path(path) => path.paint.clip = clip,
            Primitive::Underline(underline) => underline.clip = clip,
            Primitive::MonochromeSprite(sprite) => sprite.clip = clip,
            Primitive::SubpixelSprite(sprite) => sprite.clip = clip,
            Primitive::PolychromeSprite(sprite) => sprite.clip = clip,
            Primitive::Surface(surface) => surface.clip = clip,
            Primitive::NativeView(_) => {}
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
            Primitive::NativeView(native_view) => &native_view.bounds,
        }
    }

    fn transformation(&self) -> TransformationMatrix {
        match self {
            Primitive::MonochromeSprite(sprite) => sprite.transformation,
            Primitive::SubpixelSprite(sprite) => sprite.transformation,
            _ => TransformationMatrix::unit(),
        }
    }

    #[inline] // fast: also read by fast::scene
    pub(crate) fn clipped_bounds(&self) -> Bounds<ScaledPixels> {
        self.transformation()
            .transform_bounds(*self.bounds())
            .intersect(&self.content_mask().bounds)
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.paint.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
            Primitive::NativeView(native_view) => &native_view.content_mask,
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
    paths_iter: Peekable<slice::Iter<'a, PaintedPath>>,
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
    native_views_start: usize,
    native_views_iter: Peekable<slice::Iter<'a, PaintNativeView>>,
    fast_native_indices: Option<&'a [usize]>,
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
            native_views_start: run.native_views.start,
            native_views_iter: scene.native_views[run.native_views.clone()]
                .iter()
                .peekable(),
            fast_native_indices: None,
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
            (
                self.paths_iter.peek().map(|path| path.order),
                PrimitiveKind::Path,
            ),
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
            (
                self.native_views_iter
                    .peek()
                    .map(|native_view| native_view.order),
                PrimitiveKind::NativeView,
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

        // The eight arms differ in the array they walk, in whether a batch also
        // has to be homogeneous in its atlas texture, and in which
        // `PrimitiveBatch` comes out. The scan itself - take the first, take
        // every following primitive that is still under the order bound and
        // still on the same side of the clip split, advance the cursor - is one
        // piece of code.
        macro_rules! batch {
            ($iter:ident, $cursor:ident, $variant:ident) => {{
                let clipped = self.$iter.peek().unwrap().clip_id().is_clipped();
                let start = self.$cursor;
                let mut end = start + 1;
                self.$iter.next();
                while self
                    .$iter
                    .next_if(|primitive| {
                        (primitive.order, batch_kind) < max_order_and_kind
                            && primitive.clip_id().is_clipped() == clipped
                    })
                    .is_some()
                {
                    end += 1;
                }
                self.$cursor = end;
                Some(PrimitiveBatch::$variant(start..end))
            }};
        }

        macro_rules! sprite_batch {
            ($iter:ident, $cursor:ident, $variant:ident) => {{
                let texture_id = self.$iter.peek().unwrap().tile.texture_id;
                let clipped = self.$iter.peek().unwrap().clip_id().is_clipped();
                let start = self.$cursor;
                let mut end = start + 1;
                self.$iter.next();
                while self
                    .$iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                            && sprite.clip_id().is_clipped() == clipped
                    })
                    .is_some()
                {
                    end += 1;
                }
                self.$cursor = end;
                Some(PrimitiveBatch::$variant {
                    texture_id,
                    range: start..end,
                })
            }};
        }

        match batch_kind {
            PrimitiveKind::Shadow => batch!(shadows_iter, shadows_start, Shadows),
            PrimitiveKind::Quad => batch!(quads_iter, quads_start, Quads),
            PrimitiveKind::Path => batch!(paths_iter, paths_start, Paths),
            PrimitiveKind::Underline => batch!(underlines_iter, underlines_start, Underlines),
            PrimitiveKind::MonochromeSprite => sprite_batch!(
                monochrome_sprites_iter,
                monochrome_sprites_start,
                MonochromeSprites
            ),
            PrimitiveKind::SubpixelSprite => sprite_batch!(
                subpixel_sprites_iter,
                subpixel_sprites_start,
                SubpixelSprites
            ),
            PrimitiveKind::PolychromeSprite => sprite_batch!(
                polychrome_sprites_iter,
                polychrome_sprites_start,
                PolychromeSprites
            ),
            PrimitiveKind::Surface => batch!(surfaces_iter, surfaces_start, Surfaces),
            PrimitiveKind::NativeView => {
                let native_views_start = self.native_views_start;
                let mut native_views_end = native_views_start + 1;
                self.native_views_iter.next();
                if let Some(indices) = self.fast_native_indices {
                    self.native_views_start = native_views_end;
                    let index = indices[native_views_start];
                    return Some(PrimitiveBatch::NativeViews(index..index + 1));
                }
                while self
                    .native_views_iter
                    .next_if(|native_view| (native_view.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    native_views_end += 1;
                }
                self.native_views_start = native_views_end;
                Some(PrimitiveBatch::NativeViews(
                    native_views_start..native_views_end,
                ))
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
    NativeViews(Range<usize>),
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
            Self::NativeViews(range) => format!("native views ({})", range.len()),
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
    pub shape_bounds: Bounds<ScaledPixels>,
    pub shape_corner_radii: Corners<ScaledPixels>,
    pub inset: PaddedBool32,
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
    pub fn translate(self, point: Point<ScaledPixels>) -> Self {
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

    /// The smallest bounds that contain `bounds` once it is transformed.
    pub(crate) fn transform_bounds(&self, bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        if *self == Self::unit() {
            return bounds;
        }
        let [[xx, xy], [yx, yy]] = self.rotation_scale;
        let [translate_x, translate_y] = self.translation;
        let corners = [
            bounds.origin,
            bounds.top_right(),
            bounds.bottom_left(),
            bounds.bottom_right(),
        ]
        .map(|corner| {
            Point::new(
                ScaledPixels(xx * corner.x.0 + xy * corner.y.0 + translate_x),
                ScaledPixels(yx * corner.x.0 + yy * corner.y.0 + translate_y),
            )
        });
        let [first, rest @ ..] = corners;
        let top_left = rest.iter().fold(first, |min, corner| min.min(corner));
        let bottom_right = rest.iter().fold(first, |max, corner| max.max(corner));
        Bounds::from_corners(top_left, bottom_right)
    }

    /// This transformation applied to a point `delta` back from where it is
    /// asked about: for something that maps positions on a scene, moving
    /// what it maps by `delta` without changing what it maps it to.
    pub fn pretranslated(mut self, delta: Point<ScaledPixels>) -> Self {
        for (row, translation) in self.rotation_scale.iter().zip(&mut self.translation) {
            *translation -= row[0] * delta.x.0 + row[1] * delta.y.0;
        }
        self
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
    pub animation: SpriteAnimationId,
    pub clip: ClipId,
    pub pad: u32, // keep the record an even number of words
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub transformation: TransformationMatrix,
}

/// Refers to an animation of the scene a sprite is inserted into.
///
/// A single `u32` so it fits the sprite's existing padding and leaves the
/// instance layout unchanged on every backend; renderers ignore it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct SpriteAnimationId(u32);

impl SpriteAnimationId {
    /// No animation.
    pub const NONE: Self = Self(u32::MAX);
}

impl Default for SpriteAnimationId {
    fn default() -> Self {
        Self::NONE
    }
}

/// A [`TimedTransformation`] the window applies to a sprite right before
/// every present, so the scene can be presented again without being redrawn.
#[derive(Copy, Clone, Debug)]
pub(crate) struct SpriteAnimation {
    transformation: TimedTransformation,
    center: Point<Pixels>,
    scale_factor: f32,
    base_color: Hsla,
    sprite_index: usize,
}

impl SpriteAnimation {
    pub(crate) fn new(
        transformation: TimedTransformation,
        center: Point<Pixels>,
        scale_factor: f32,
    ) -> Self {
        Self {
            transformation,
            center,
            scale_factor,
            base_color: Hsla::default(),
            sprite_index: usize::MAX,
        }
    }
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
    /// Laid out word for word as a [`MonochromeSprite`], whose shaders the
    /// DirectX renderer draws subpixel sprites with: this is the slot of its
    /// `animation`.
    pub pad: u32,
    pub clip: ClipId,
    pub clip_pad: u32,
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

/// Where a platform view shows through the scene.
///
/// The view sits beneath the window's rendered content, so the renderer
/// erases everything painted earlier within these bounds, and everything
/// painted later covers the view. The view is therefore treated as opaque.
#[derive(Clone, Copy, Debug)]
pub struct PaintNativeView {
    /// The view's position among the scene's other primitives.
    pub order: DrawOrder,
    /// The view these bounds belong to.
    pub id: NativeViewId,
    /// The hitbox that decides whether a mouse position reaches the view.
    pub hitbox_id: HitboxId,
    /// Where the view is laid out, before clipping.
    pub bounds: Bounds<ScaledPixels>,
    /// The clip applied to the view.
    pub content_mask: ContentMask<ScaledPixels>,
    /// The color to show where the view has not caught up with these bounds,
    /// or a transparent color for none.
    pub background: Hsla,
}

impl From<PaintNativeView> for Primitive {
    fn from(native_view: PaintNativeView) -> Self {
        Primitive::NativeView(native_view)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathCurve {
    pub p0: Point<f32>,
    pub p1: Point<f32>,
    pub a: Point<f32>,
    pub b: Point<f32>,
    pub sign: f32,
    pub pad: u32, // align to 8 bytes
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathPaint {
    /// The path's bounds, in device pixels.
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Background,
    pub even_odd: PaddedBool32,
    pub clip: ClipId, // also aligns the record to 8 bytes
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathTile {
    pub paint: u32,
    pub curve_start: u32,
    pub curve_count: u32,
    pub backdrop: i32,
    pub top_left: Point<f32>,
    pub bottom_right: Point<f32>,
    pub reference: Point<f32>,
    pub curves_top_left: Point<f32>,
    pub curves_bottom_right: Point<f32>,
}

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

#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct PaintedPath {
    pub(crate) order: DrawOrder,
    pub(crate) bounds: Bounds<ScaledPixels>,
    pub(crate) tiles: Rc<PathTiles>,
    pub(crate) paint: PathPaint,
    /// An image to fill the path with instead of its `color`, resolved by
    /// [`crate::Window::paint_path_with_image`]. It is not part of the
    /// [`PathPaint`] a renderer uploads for every path: a renderer that
    /// resolves brushes reads it into a side buffer of its own, and the others
    /// paint `color` and say so.
    pub(crate) brush: Option<PathBrush>,
}

#[expect(missing_docs)]
impl PaintedPath {
    /// Bins `path` into tiles, to be filled with `color` within `content_mask`.
    pub fn new(
        path: Path,
        scale_factor: f32,
        content_mask: ContentMask<Pixels>,
        color: Background,
    ) -> Self {
        // fast: on the grid fast::layers draws a path moved into a layer on.
        let content_mask = crate::fast::layers::paint::snap_mask(content_mask.scale(scale_factor));
        Self::with_device_mask(path, scale_factor, content_mask, color)
    }

    /// Bins `path` into tiles, to be filled with `color` within a mask already
    /// in device pixels, which is how a window hands a path the mask it hands
    /// a quad.
    pub fn with_device_mask(
        path: Path,
        scale_factor: f32,
        content_mask: ContentMask<ScaledPixels>,
        color: Background,
    ) -> Self {
        let tiles = crate::fast::path_cache::bin(&path, scale_factor);
        let tiles = crate::fast::path_cache::visible(tiles, content_mask.bounds);
        let paint = PathPaint {
            bounds: tiles.bounds(),
            content_mask,
            color,
            even_odd: path.even_odd().into(),
            clip: ClipId::NONE,
        };
        Self {
            order: 0,
            bounds: tiles.painted_bounds(),
            tiles,
            brush: None,
            paint,
        }
    }

    /// This path, filled with `brush` instead of its colour.
    pub(crate) fn with_brush(mut self, brush: PathBrush) -> Self {
        self.brush = Some(brush);
        self
    }

    /// The image the path is filled with instead of its colour, if any.
    pub fn brush(&self) -> Option<&PathBrush> {
        self.brush.as_ref()
    }

    /// The clip path the path is drawn under.
    pub fn clip(&self) -> ClipId {
        self.paint.clip
    }

    pub(crate) fn order(&self) -> DrawOrder {
        self.order
    }

    /// The bounds of the pixels the path paints, in device pixels
    /// (gpui-fast composition occluders).
    pub fn painted_bounds(&self) -> Bounds<ScaledPixels> {
        self.bounds
    }

    /// The mask the path is drawn within (gpui-fast composition occluders).
    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        &self.paint.content_mask
    }

    /// Moves the path by `delta` device pixels (gpui-fast layers).
    pub(crate) fn translate(&mut self, delta: Point<ScaledPixels>) {
        let move_bounds = |bounds: Bounds<ScaledPixels>| Bounds {
            origin: bounds.origin + delta,
            size: bounds.size,
        };
        self.bounds = move_bounds(self.bounds);
        self.paint.bounds = move_bounds(self.paint.bounds);
        self.paint.content_mask.bounds = move_bounds(self.paint.content_mask.bounds);
        self.tiles = Rc::new(self.tiles.translated(point(delta.x.0, delta.y.0)));
        // A brush maps a scene position to the image, so the image moves with
        // the path only if what it is handed is first moved back.
        if let Some(brush) = &mut self.brush {
            brush.screen_to_brush = brush
                .screen_to_brush
                .translate(point(-delta.x, -delta.y));
        }
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.tiles().len()
    }

    pub fn append_to(
        &self,
        paints: &mut Vec<PathPaint>,
        curves: &mut Vec<PathCurve>,
        tiles: &mut Vec<PathTile>,
    ) {
        let paint_index = paints.len() as u32;
        let curve_start = curves.len() as u32;
        paints.push(self.paint);
        curves.extend_from_slice(self.tiles.curves());
        tiles.extend(self.tiles.tiles().iter().map(|tile| PathTile {
            paint: paint_index,
            curve_start: curve_start + tile.curve_start,
            ..*tile
        }));
    }
}

/// Which clip path a per-kind record is drawn under, which is what cuts a
/// batch: see [`Scene::batches`].
trait ClippedPrimitive {
    fn clip_id(&self) -> ClipId;
}

macro_rules! clipped_primitive {
    ($($kind:ty),+ $(,)?) => {$(
        impl ClippedPrimitive for $kind {
            fn clip_id(&self) -> ClipId {
                self.clip
            }
        }
    )+};
}

clipped_primitive!(
    Shadow,
    Quad,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    PaintSurface,
);

impl ClippedPrimitive for PaintedPath {
    fn clip_id(&self) -> ClipId {
        self.paint.clip
    }
}

impl From<PaintedPath> for Primitive {
    fn from(path: PaintedPath) -> Self {
        Primitive::Path(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point, size};

    fn square(x: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(0.)),
            size: size(ScaledPixels(10.), ScaledPixels(10.)),
        }
    }

    fn quad(x: f32) -> Quad {
        Quad {
            bounds: square(x),
            content_mask: ContentMask { bounds: square(x) },
            ..Default::default()
        }
    }

    fn underline(x: f32) -> Underline {
        Underline {
            order: 0,
            pad: 0,
            bounds: square(x),
            content_mask: ContentMask { bounds: square(x) },
            color: Hsla::default(),
            thickness: ScaledPixels(1.),
            wavy: false.into(),
        }
    }

    fn batch_labels(scene: &mut Scene) -> Vec<String> {
        scene.finish();
        scene.batches().map(|batch| batch.label()).collect()
    }

    #[test]
    fn a_native_view_in_a_layer_is_covered_by_what_the_layer_paints_after_it() {
        let mut scene = Scene::default();
        scene.push_layer(Bounds::default());
        scene.insert_primitive(quad(0.));
        scene.insert_primitive(PaintNativeView {
            order: 0,
            id: NativeViewId::for_test(1),
            hitbox_id: HitboxId::placeholder(),
            bounds: square(0.),
            content_mask: ContentMask { bounds: square(0.) },
            background: Hsla::default(),
        });
        scene.insert_primitive(quad(0.));
        // A quad beside the view shares nothing with it, so it isn't held back.
        scene.insert_primitive(quad(20.));
        scene.pop_layer();
        assert_eq!(
            batch_labels(&mut scene),
            ["quads (1)", "native views (1)", "quads (2)"]
        );
    }

    #[test]
    fn a_native_view_painted_twice_is_kept_where_it_was_painted_first() {
        let native_view = |hitbox_id: HitboxId, x: f32| PaintNativeView {
            order: 0,
            id: NativeViewId::for_test(1),
            hitbox_id,
            bounds: square(x),
            content_mask: ContentMask { bounds: square(x) },
            background: Hsla::default(),
        };
        let first_hitbox = HitboxId::placeholder();
        let second_hitbox = first_hitbox.next();
        let mut scene = Scene::default();
        // The quad can't be drawn with the underline it covers, so the first
        // occurrence, painted over both, is drawn after the second, which
        // overlaps nothing.
        scene.insert_primitive(underline(0.));
        scene.insert_primitive(quad(0.));
        scene.insert_primitive(native_view(first_hitbox, 0.));
        scene.insert_primitive(native_view(second_hitbox, 20.));
        scene.finish();

        let shown = scene
            .native_views
            .iter()
            .map(|native_view| native_view.bounds)
            .collect::<Vec<_>>();
        assert_eq!(shown, [square(0.)]);
    }

    #[test]
    fn layers_are_drawn_in_paint_order() {
        // A quad painted after a layer goes above it.
        let mut scene = Scene::default();
        scene.insert_primitive(quad(0.));
        scene.push_layer(Bounds::default());
        scene.insert_primitive(underline(0.));
        scene.insert_primitive(underline(5.));
        scene.pop_layer();
        scene.insert_primitive(quad(5.));
        assert_eq!(
            batch_labels(&mut scene),
            ["quads (1)", "underlines (2)", "quads (1)"]
        );

        // A layer goes above the layer painted before it.
        let mut scene = Scene::default();
        scene.push_layer(Bounds::default());
        scene.insert_primitive(underline(0.));
        scene.pop_layer();
        scene.push_layer(Bounds::default());
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (1)", "quads (1)"]);
    }

    #[test]
    fn nested_layers_are_drawn_above_their_parents() {
        // A layer has one draw order, so what it paints after a nested layer
        // still goes below that layer.
        let mut scene = Scene::default();
        scene.push_layer(Bounds::default());
        scene.insert_primitive(underline(0.));
        scene.push_layer(Bounds::default());
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        scene.insert_primitive(underline(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (2)", "quads (1)"]);

        // The same holds when the nested layer shares its parent's order.
        let mut scene = Scene::default();
        scene.push_layer(Bounds::default());
        scene.insert_primitive(quad(0.));
        scene.push_layer(Bounds::default());
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        scene.insert_primitive(quad(8.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["quads (3)"]);
        assert_eq!(
            scene
                .quads
                .iter()
                .map(|quad| quad.bounds)
                .collect::<Vec<_>>(),
            [square(0.), square(8.), square(5.)]
        );

        // A layer goes above the layers nested in the layer before it.
        let mut scene = Scene::default();
        scene.push_layer(Bounds::default());
        scene.push_layer(Bounds::default());
        scene.insert_primitive(underline(0.));
        scene.pop_layer();
        scene.pop_layer();
        scene.push_layer(Bounds::default());
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (1)", "quads (1)"]);
    }
}

/// A width and height in whole device pixels: the size of a coverage atlas, or
/// of the working attachments one nesting level is rendered through.
///
/// This is renderer-independent bookkeeping rather than geometry: it is whole
/// texels, it is never negative in practice, and it exists to be handed to a
/// texture allocator. [`Bounds<DevicePixels>`](Bounds) is the geometry type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Extent {
    /// The width, in whole device pixels.
    pub width: i32,
    /// The height, in whole device pixels.
    pub height: i32,
}

impl Extent {
    /// The extent that holds nothing.
    pub const ZERO: Self = Extent {
        width: 0,
        height: 0,
    };

    /// The smallest extent that holds both.
    pub fn max(self, other: Self) -> Self {
        Extent {
            width: self.width.max(other.width),
            height: self.height.max(other.height),
        }
    }

    /// What a BGRA8 texture of this size costs.
    pub fn bytes(self) -> usize {
        self.width.max(0) as usize * self.height.max(0) as usize * 4
    }

    /// Rounded up to whole `quantum`s, and never to nothing: a size that
    /// wobbles by a pixel is then the same allocation, and a texture is never
    /// asked for with a side of zero.
    pub fn quantized_to(self, quantum: i32) -> Self {
        let round = |value: i32| {
            let quanta = (value.max(1) + quantum - 1) / quantum;
            quanta * quantum
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
pub struct TextureBudget<const SHRINK_FRAMES: u32> {
    /// What is currently allocated.
    current: Extent,
    /// The largest a frame has needed since the present run of small frames
    /// began, and so what shrinking would shrink to.
    peak: Extent,
    /// How many consecutive frames have fitted inside `peak`.
    frames: u32,
}

impl<const SHRINK_FRAMES: u32> TextureBudget<SHRINK_FRAMES> {
    /// The size the next frame starts from: what is already allocated, unless
    /// a long enough run of frames has all fitted inside something smaller.
    pub fn floor(&mut self) -> Extent {
        if self.frames >= SHRINK_FRAMES {
            self.current = self.peak;
            self.peak = Extent::ZERO;
            self.frames = 0;
        }
        self.current
    }

    /// What the frame allocated, and what it would have been enough to
    /// allocate.
    pub fn observe(&mut self, allocated: Extent, needed: Extent) {
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
pub struct DeviceRect {
    /// The left edge.
    pub x: i32,
    /// The top edge.
    pub y: i32,
    /// The width; zero or less is empty.
    pub width: i32,
    /// The height; zero or less is empty.
    pub height: i32,
}

/// How far from the origin a rectangle in scaled pixels is taken seriously by
/// [`DeviceRect::covering`].
///
/// A clip path is arbitrary, so its box can be arbitrary too, and `f32 as i32`
/// saturates: two saturated edges subtracted from one another overflow. Any
/// window is orders of magnitude inside this, and such a box is intersected
/// with the viewport straight afterwards.
const COORDINATE_LIMIT: f32 = (1 << 24) as f32;

impl DeviceRect {
    /// A rectangle enclosing nothing, which is what a clip that lets nothing
    /// through gets.
    pub const EMPTY: Self = DeviceRect {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
    };

    /// The whole window, which every other rectangle is clamped inside.
    pub fn viewport(size: Size<DevicePixels>) -> Self {
        DeviceRect {
            x: 0,
            y: 0,
            width: size.width.0,
            height: size.height.0,
        }
    }

    /// The whole device pixels a device-space rectangle touches.
    pub fn covering(bounds: &Bounds<ScaledPixels>) -> Self {
        let clamp = |value: f32| value.clamp(-COORDINATE_LIMIT, COORDINATE_LIMIT);
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

    /// Whether the rectangle encloses no pixel at all.
    pub fn is_empty(&self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    /// The pixels both rectangles hold.
    pub fn intersect(&self, other: &Self) -> Self {
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

    /// The smallest rectangle holding both; an empty rectangle contributes
    /// nothing rather than dragging the result to the origin.
    pub fn union(&self, other: &Self) -> Self {
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

    /// The size alone, with an empty rectangle reading as zero.
    pub fn extent(&self) -> Extent {
        Extent {
            width: self.width.max(0),
            height: self.height.max(0),
        }
    }

    /// The same rectangle as scene geometry.
    pub fn bounds(&self) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(self.x as f32), ScaledPixels(self.y as f32)),
            size: size(
                ScaledPixels(self.width.max(0) as f32),
                ScaledPixels(self.height.max(0) as f32),
            ),
        }
    }
}

#[cfg(test)]
mod fifteen_tests {
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
            pad: 0,
            clip: ClipId::NONE,
            clip_pad: 0,
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
        use objc2_core_foundation::CFRetained;
        use objc2_core_video::{CVPixelBufferCreate, kCVPixelFormatType_32BGRA, kCVReturnSuccess};
        use std::ptr::{self, NonNull};

        let mut image_buffer = ptr::null_mut();
        // SAFETY: The output pointer is writable and no attributes are passed.
        let status = unsafe {
            CVPixelBufferCreate(
                None,
                8,
                8,
                kCVPixelFormatType_32BGRA,
                None,
                NonNull::from(&mut image_buffer),
            )
        };
        assert_eq!(status, kCVReturnSuccess, "failed to create a pixel buffer");
        let image_buffer = NonNull::new(image_buffer).expect("CoreVideo returned no pixel buffer");
        let bounds = Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(0.)),
            size: size(ScaledPixels(8.), ScaledPixels(8.)),
        };
        PaintSurface {
            order: 0,
            clip: ClipId::NONE,
            bounds,
            content_mask: ContentMask::new(bounds),
            // SAFETY: A successful create transfers ownership of the buffer.
            image_buffer: unsafe { CFRetained::from_raw(image_buffer) },
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
