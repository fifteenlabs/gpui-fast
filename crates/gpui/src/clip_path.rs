//! Arbitrary-path clipping.
//!
//! [`ContentMask`](crate::ContentMask) is gpui's cheap clip: a rectangle with
//! four *circular* corner radii, enforced by the hardware scissor plus a signed
//! distance field in the shaders. It costs nothing, and it is what every gpui
//! element uses.
//!
//! It cannot express what CSS asks for. `border-radius` gives each corner an
//! independent horizontal and vertical radius, so a corner is an ellipse
//! quadrant rather than a circle quadrant, and `clip-path` is an arbitrary
//! outline with a fill rule. A [`ClipPath`] carries that shape verbatim:
//! retained contours and a [`FillRule`], with no attempt to recover a rounded
//! rectangle from it.
//!
//! A [`ClipPath`] is deliberately *not* a [`Path`](crate::Path). A `Path` fans
//! triangles from `(start, current, to)` and the pipeline blends them
//! source-over, so a reflex contour paints outside itself and self-overlap
//! accumulates: a `Path` cannot express a non-convex fill at all. Clip shapes
//! are frequently non-convex, so they keep their contours and are rasterized
//! into a mask instead.

use crate::{
    Bounds, Corners, DevicePixels, DeviceRect, Extent, FillRule, Pixels, Point, ScaledPixels,
    Scene, Size, point, px,
};
use std::{fmt::Debug, mem, ops::Range};

/// The circle-to-cubic constant: the control-point offset, as a fraction of the
/// radius, that makes a cubic Bezier approximate a quarter ellipse.
const KAPPA: f32 = 0.552_284_75;

/// The fill rule a clip path uses when nothing says otherwise, and the one CSS
/// and SVG default to.
const DEFAULT_FILL_RULE: FillRule = FillRule::NonZero;

/// What a renderer rounds its clip textures' sides up to.
///
/// Quantizing them is what keeps a document whose clips move a pixel a frame
/// from recreating textures every frame, and 256 is small enough that the
/// rounding itself wastes little.
pub const CLIP_TEXTURE_QUANTUM: i32 = 256;
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

/// How many consecutive frames have to fit inside smaller clip textures before
/// the renderer gives the memory back, as a [`TextureBudget`](crate::TextureBudget).
pub type ClipTextureBudget = crate::TextureBudget<CLIP_TEXTURE_SHRINK_FRAMES>;

/// One step of a [`ClipPath`]'s outline.
///
/// The control points of a curve are kept as authored; nothing is flattened
/// here, so the eventual rasterizer is free to pick a tolerance from the scale
/// it is rasterizing at.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ClipPathSegment<P: Clone + Debug + Default + PartialEq> {
    /// Begin a new contour at this point.
    MoveTo(Point<P>),
    /// A straight line from the current point.
    LineTo(Point<P>),
    /// A quadratic Bezier from the current point.
    QuadTo {
        /// The single control point.
        control: Point<P>,
        /// The end point.
        to: Point<P>,
    },
    /// A cubic Bezier from the current point.
    CubicTo {
        /// The control point leaving the current point.
        control1: Point<P>,
        /// The control point entering `to`.
        control2: Point<P>,
        /// The end point.
        to: Point<P>,
    },
    /// Close the current contour back to the point its `MoveTo` opened it at.
    Close,
}

impl ClipPathSegment<Pixels> {
    fn scale(&self, factor: f32) -> ClipPathSegment<ScaledPixels> {
        match self {
            Self::MoveTo(to) => ClipPathSegment::MoveTo(to.scale(factor)),
            Self::LineTo(to) => ClipPathSegment::LineTo(to.scale(factor)),
            Self::QuadTo { control, to } => ClipPathSegment::QuadTo {
                control: control.scale(factor),
                to: to.scale(factor),
            },
            Self::CubicTo {
                control1,
                control2,
                to,
            } => ClipPathSegment::CubicTo {
                control1: control1.scale(factor),
                control2: control2.scale(factor),
                to: to.scale(factor),
            },
            Self::Close => ClipPathSegment::Close,
        }
    }
}

/// An arbitrary clip shape: retained contours plus a [`FillRule`].
///
/// Build one with [`ClipPath::builder`], or take the common case straight from
/// [`ClipPath::rounded_rect`]. Apply one with
/// [`Window::with_clip_path`](crate::Window::with_clip_path).
#[derive(Clone, Debug, PartialEq)]
pub struct ClipPath<P: Clone + Debug + Default + PartialEq> {
    segments: Vec<ClipPathSegment<P>>,
    fill_rule: FillRule,
    bounds: Bounds<P>,
}

impl<P: Clone + Debug + Default + PartialEq> ClipPath<P> {
    /// The contours, in the order they were authored.
    pub fn segments(&self) -> &[ClipPathSegment<P>] {
        &self.segments
    }

    /// Which points the contours enclose.
    pub fn fill_rule(&self) -> FillRule {
        self.fill_rule
    }

    /// The smallest axis-aligned box containing every point of the path.
    ///
    /// Curves are bounded by their control polygon rather than by the curve
    /// itself, so this can be larger than the tight bound of a curve that
    /// bulges inward. It is never smaller, which is what a culling or scissor
    /// user needs.
    pub fn bounds(&self) -> Bounds<P> {
        self.bounds.clone()
    }

    /// Whether the path has no contours at all, and so clips everything away.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

impl ClipPath<Pixels> {
    /// Start building a path.
    pub fn builder() -> ClipPathBuilder {
        ClipPathBuilder::default()
    }

    /// A rectangle whose corners are elliptical quadrants, one horizontal and
    /// one vertical radius per corner.
    ///
    /// This is the shape `border-radius` describes and the shape
    /// [`ContentMask`](crate::ContentMask), whose radii are a single
    /// `Corners<Pixels>`, structurally cannot hold. Radii are clamped to
    /// non-negative and then scaled down together, by CSS's rule, when two
    /// radii on one edge would overlap.
    ///
    /// The result always has the same ten segments — a `MoveTo`, four
    /// `LineTo`/`CubicTo` pairs going clockwise from the top edge, and a
    /// `Close` — so its shape is a pure function of its inputs. A square corner
    /// simply contributes a degenerate cubic.
    pub fn rounded_rect(bounds: Bounds<Pixels>, radii: Corners<Size<Pixels>>) -> Self {
        let left = bounds.origin.x.0;
        let top = bounds.origin.y.0;
        let width = bounds.size.width.0.max(0.0);
        let height = bounds.size.height.0.max(0.0);
        let right = left + width;
        let bottom = top + height;

        let radius = |size: &Size<Pixels>| (size.width.0.max(0.0), size.height.0.max(0.0));
        let (mut tl, mut tr) = (radius(&radii.top_left), radius(&radii.top_right));
        let (mut br, mut bl) = (radius(&radii.bottom_right), radius(&radii.bottom_left));

        // CSS's overlapping-curves rule: if any edge's two radii together
        // exceed it, every radius shrinks by the same factor, so the corners
        // stay proportional instead of one edge's pair being singled out.
        let ratio = [
            (width, tl.0 + tr.0),
            (width, bl.0 + br.0),
            (height, tl.1 + bl.1),
            (height, tr.1 + br.1),
        ]
        .into_iter()
        .filter(|(_, sum)| *sum > 0.0)
        .map(|(extent, sum)| extent / sum)
        .fold(1.0f32, f32::min);
        if ratio < 1.0 {
            for corner in [&mut tl, &mut tr, &mut br, &mut bl] {
                corner.0 *= ratio;
                corner.1 *= ratio;
            }
        }

        let at = |x: f32, y: f32| point(px(x), px(y));
        ClipPath::builder()
            .move_to(at(left + tl.0, top))
            .line_to(at(right - tr.0, top))
            .cubic_to(
                at(right - tr.0 + tr.0 * KAPPA, top),
                at(right, top + tr.1 - tr.1 * KAPPA),
                at(right, top + tr.1),
            )
            .line_to(at(right, bottom - br.1))
            .cubic_to(
                at(right, bottom - br.1 + br.1 * KAPPA),
                at(right - br.0 + br.0 * KAPPA, bottom),
                at(right - br.0, bottom),
            )
            .line_to(at(left + bl.0, bottom))
            .cubic_to(
                at(left + bl.0 - bl.0 * KAPPA, bottom),
                at(left, bottom - bl.1 + bl.1 * KAPPA),
                at(left, bottom - bl.1),
            )
            .line_to(at(left, top + tl.1))
            .cubic_to(
                at(left, top + tl.1 - tl.1 * KAPPA),
                at(left + tl.0 - tl.0 * KAPPA, top),
                at(left + tl.0, top),
            )
            .close()
            .build()
    }

    /// Scale the path's logical pixels into the device pixels the scene speaks.
    pub fn scale(&self, factor: f32) -> ClipPath<ScaledPixels> {
        ClipPath {
            segments: self
                .segments
                .iter()
                .map(|segment| segment.scale(factor))
                .collect(),
            fill_rule: self.fill_rule,
            bounds: self.bounds.scale(factor),
        }
    }
}

/// Accumulates the contours of a [`ClipPath`].
#[derive(Clone, Debug)]
pub struct ClipPathBuilder {
    segments: Vec<ClipPathSegment<Pixels>>,
    fill_rule: FillRule,
}

impl Default for ClipPathBuilder {
    fn default() -> Self {
        Self {
            segments: Vec::new(),
            fill_rule: DEFAULT_FILL_RULE,
        }
    }
}

impl ClipPathBuilder {
    /// Set which points the finished contours enclose. Defaults to
    /// [`FillRule::NonZero`].
    pub fn fill_rule(mut self, fill_rule: FillRule) -> Self {
        self.fill_rule = fill_rule;
        self
    }

    /// Begin a new contour at `to`.
    pub fn move_to(mut self, to: Point<Pixels>) -> Self {
        self.segments.push(ClipPathSegment::MoveTo(to));
        self
    }

    /// Draw a straight line from the current point to `to`.
    pub fn line_to(mut self, to: Point<Pixels>) -> Self {
        self.segments.push(ClipPathSegment::LineTo(to));
        self
    }

    /// Draw a quadratic Bezier from the current point to `to`.
    pub fn quad_to(mut self, control: Point<Pixels>, to: Point<Pixels>) -> Self {
        self.segments.push(ClipPathSegment::QuadTo { control, to });
        self
    }

    /// Draw a cubic Bezier from the current point to `to`.
    pub fn cubic_to(
        mut self,
        control1: Point<Pixels>,
        control2: Point<Pixels>,
        to: Point<Pixels>,
    ) -> Self {
        self.segments.push(ClipPathSegment::CubicTo {
            control1,
            control2,
            to,
        });
        self
    }

    /// Close the current contour back to the point it started at.
    pub fn close(mut self) -> Self {
        self.segments.push(ClipPathSegment::Close);
        self
    }

    /// Finish the path, computing its bounds once.
    pub fn build(self) -> ClipPath<Pixels> {
        let bounds = bounding_box(&self.segments);
        ClipPath {
            segments: self.segments,
            fill_rule: self.fill_rule,
            bounds,
        }
    }
}

/// The box containing every point the segments name, control points included.
fn bounding_box(segments: &[ClipPathSegment<Pixels>]) -> Bounds<Pixels> {
    let mut min = point(f32::INFINITY, f32::INFINITY);
    let mut max = point(f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut include = |at: &Point<Pixels>| {
        min.x = min.x.min(at.x.0);
        min.y = min.y.min(at.y.0);
        max.x = max.x.max(at.x.0);
        max.y = max.y.max(at.y.0);
    };

    for segment in segments {
        match segment {
            ClipPathSegment::MoveTo(to) | ClipPathSegment::LineTo(to) => include(to),
            ClipPathSegment::QuadTo { control, to } => {
                include(control);
                include(to);
            }
            ClipPathSegment::CubicTo {
                control1,
                control2,
                to,
            } => {
                include(control1);
                include(control2);
                include(to);
            }
            ClipPathSegment::Close => {}
        }
    }

    if min.x > max.x {
        return Bounds::default();
    }
    Bounds::from_corners(point(px(min.x), px(min.y)), point(px(max.x), px(max.y)))
}

/// What a fragment shader needs to ask the coverage atlas how much of a point
/// one clip path lets through.
///
/// One of these per registered clip, plus a leading entry for [`crate::ClipId::NONE`]
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
    pub atlas_offset: Point<f32>,
    /// Zero for a clip that got no tile, whose `tile` rectangle is then the
    /// whole answer.
    pub sampled: u32,
    /// Padding, so the struct carries no compiler-inserted bytes and matches
    /// what the shader declares.
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
    pub parent_offset: Point<f32>,
    /// Zero for a clip at the root, whose coverage is its own shape alone.
    pub has_parent: u32,
    /// Padding, so the struct carries no compiler-inserted bytes and matches
    /// what the shader declares.
    pub pad: u32,
}

/// One clip path, flattened and placed, before the atlas has been packed.
struct ClipShape {
    contours: Vec<Vec<Point<f32>>>,
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
pub struct PlannedClip {
    /// Where the finished coverage lives in the atlas.
    pub atlas: DeviceRect,
    /// The same tile in the working attachment its level is drawn through,
    /// which is the atlas rectangle less the level's own origin.
    pub work: DeviceRect,
    /// This clip's triangles, as a range of [`ClipPlan::vertices`].
    pub vertices: Range<usize>,
    /// Which points its contours enclose, and so how the stencil pass counts.
    pub fill_rule: FillRule,
    /// What the cover half of stencil-and-cover reads.
    pub cover: ClipCover,
}

/// One nesting depth: a render pass that reads the tiles the level above it
/// wrote.
pub struct ClipLevel {
    /// The clips at this depth that got a tile, in packing order.
    pub clips: Vec<usize>,
    /// The atlas rectangle their tiles span. Only this much is rendered and
    /// resolved, so a level of one small clip costs one small clip.
    pub extent: DeviceRect,
}

/// Where every clip path in one scene rasterizes to.
pub struct ClipPlan {
    /// Indexed by [`crate::ClipId`] itself, entry zero standing for `ClipId::NONE`.
    pub masks: Vec<ClipMask>,
    /// Indexed by clip index; `None` for a clip that got no tile.
    pub clips: Vec<Option<PlannedClip>>,
    /// Nesting depths, shallowest first: level `k` is a render pass that reads
    /// the tiles level `k - 1` wrote.
    pub levels: Vec<ClipLevel>,
    /// Every planned clip's triangles, laid end to end in the coordinates of
    /// the working attachment its level is drawn through.
    pub vertices: Vec<Point<f32>>,
    /// The atlas the tiles were packed into.
    pub atlas: Extent,
    /// The smallest atlas that would have held them. The renderer's budget
    /// watches this, so an atlas grown for one busy frame is given back.
    pub minimum_atlas: Extent,
    /// The working attachments the mask passes need: the largest level, which
    /// is all a multisample resolve has to cover.
    pub work: Extent,
}

impl ClipPlan {
    /// The plan for a scene with no clip paths in it at all: no tiles, no
    /// passes, and no textures for a renderer to allocate.
    pub fn unclipped() -> Self {
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

    /// Plan where every clip path in `scene` rasterizes to.
    ///
    /// `floor` is the atlas the renderer already has allocated: the plan never
    /// asks for less than that, so a frame that fits inside last frame's atlas
    /// reuses it rather than shrinking it and growing it back.
    pub fn new(scene: &Scene, viewport_size: Size<DevicePixels>, floor: Extent) -> Self {
        let viewport = DeviceRect::viewport(viewport_size);

        let mut shapes: Vec<ClipShape> = Vec::with_capacity(scene.clips.len());
        let mut too_deep = 0usize;
        for scene_clip in &scene.clips {
            let parent = scene_clip.parent.index();
            let depth = parent.map_or(0, |parent| shapes[parent].depth + 1);
            let parent_tile = parent.map_or(viewport, |parent| shapes[parent].device);
            let reachable = DeviceRect::covering(&scene_clip.path.bounds())
                .intersect(&DeviceRect::covering(&scene_clip.visible))
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
            .quantized_to(CLIP_TEXTURE_QUANTUM);

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
    .quantized_to(CLIP_TEXTURE_QUANTUM);
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

/// Turns a clip path's contours into polylines in device space.
///
/// Flattening the curves outright, rather than handing quadratics to a
/// Loop-Blinn fragment test, is what lets the stencil pass be nothing but
/// triangles: the coverage then comes from multisampling alone and is the same
/// on a curve as on a straight edge, with no per-sample shading to arrange.
fn flatten_clip_path(path: &ClipPath<ScaledPixels>) -> Vec<Vec<Point<f32>>> {
    let mut contours: Vec<Vec<Point<f32>>> = Vec::new();
    let mut contour: Vec<Point<f32>> = Vec::new();
    let mut start = point(0., 0.);
    let mut at = point(0., 0.);

    fn finish(contours: &mut Vec<Vec<Point<f32>>>, contour: Vec<Point<f32>>) {
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

fn device_point(at: &Point<ScaledPixels>) -> Point<f32> {
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

fn flatten_quadratic(
    from: Point<f32>,
    control: Point<f32>,
    to: Point<f32>,
    out: &mut Vec<Point<f32>>,
) {
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
    from: Point<f32>,
    control1: Point<f32>,
    control2: Point<f32>,
    to: Point<f32>,
    out: &mut Vec<Point<f32>>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bounds, DevicePixels, Scene, size};

    fn rect() -> Bounds<Pixels> {
        Bounds {
            origin: point(px(10.), px(20.)),
            size: size(px(200.), px(100.)),
        }
    }

    #[test]
    fn rounded_rect_has_elliptical_corners() {
        let radii = Corners {
            top_left: size(px(30.), px(10.)),
            top_right: size(px(0.), px(0.)),
            bottom_right: size(px(5.), px(40.)),
            bottom_left: size(px(0.), px(0.)),
        };
        let path = ClipPath::rounded_rect(rect(), radii);

        assert_eq!(path.fill_rule(), FillRule::NonZero);
        assert_eq!(path.segments().len(), 10);
        assert_eq!(
            path.segments()[0],
            ClipPathSegment::MoveTo(point(px(40.), px(20.)))
        );
        assert_eq!(path.segments()[9], ClipPathSegment::Close);

        // The left edge stops 10 above the top, and the corner then sweeps 30
        // across to the start point: one quadrant of a 30x10 ellipse, which a
        // single circular radius cannot describe.
        assert_eq!(
            path.segments()[7],
            ClipPathSegment::LineTo(point(px(10.), px(30.)))
        );
        assert_eq!(
            path.segments()[8],
            ClipPathSegment::CubicTo {
                control1: point(px(10.), px(30. - 10. * KAPPA)),
                control2: point(px(10. + 30. - 30. * KAPPA), px(20.)),
                to: point(px(40.), px(20.)),
            }
        );

        // A square corner still emits its cubic, collapsed onto the corner.
        assert_eq!(
            path.segments()[2],
            ClipPathSegment::CubicTo {
                control1: point(px(210.), px(20.)),
                control2: point(px(210.), px(20.)),
                to: point(px(210.), px(20.)),
            }
        );
    }

    #[test]
    fn rounded_rect_bounds_are_the_rect() {
        let radii = Corners {
            top_left: size(px(30.), px(10.)),
            top_right: size(px(12.), px(50.)),
            bottom_right: size(px(5.), px(40.)),
            bottom_left: size(px(60.), px(3.)),
        };
        assert_eq!(ClipPath::rounded_rect(rect(), radii).bounds(), rect());
    }

    #[test]
    fn rounded_rect_clamps_overlapping_radii() {
        // 150 + 150 of horizontal radius on a 200-wide rect: CSS scales every
        // radius by 200 / 300.
        let radii = Corners {
            top_left: size(px(150.), px(30.)),
            top_right: size(px(150.), px(30.)),
            bottom_right: size(px(0.), px(0.)),
            bottom_left: size(px(0.), px(0.)),
        };
        let path = ClipPath::rounded_rect(rect(), radii);
        let ClipPathSegment::MoveTo(start) = path.segments()[0] else {
            unreachable!()
        };
        assert_eq!(start, point(px(110.), px(20.)));
        assert_eq!(path.bounds(), rect());
    }

    #[test]
    fn rounded_rect_with_no_radii_is_the_rect() {
        let path = ClipPath::rounded_rect(rect(), Corners::default());
        assert_eq!(path.segments().len(), 10);
        assert_eq!(path.bounds(), rect());
    }

    #[test]
    fn scaling_scales_points_and_bounds() {
        let path = ClipPath::builder()
            .move_to(point(px(1.), px(2.)))
            .quad_to(point(px(3.), px(4.)), point(px(5.), px(6.)))
            .close()
            .build();
        let scaled = path.scale(2.0);

        assert_eq!(scaled.segments().len(), 3);
        assert_eq!(
            scaled.segments()[1],
            ClipPathSegment::QuadTo {
                control: point(ScaledPixels(6.), ScaledPixels(8.)),
                to: point(ScaledPixels(10.), ScaledPixels(12.)),
            }
        );
        assert_eq!(scaled.bounds(), path.bounds().scale(2.0));
    }

    #[test]
    fn an_empty_builder_builds_an_empty_path() {
        let path = ClipPath::builder().build();
        assert!(path.is_empty());
        assert_eq!(path.bounds(), Bounds::default());
    }

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
