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

use crate::{Bounds, Corners, FillRule, Pixels, Point, ScaledPixels, Size, point, px};
use std::fmt::Debug;

/// The circle-to-cubic constant: the control-point offset, as a fraction of the
/// radius, that makes a cubic Bezier approximate a quarter ellipse.
const KAPPA: f32 = 0.552_284_75;

/// The fill rule a clip path uses when nothing says otherwise, and the one CSS
/// and SVG default to.
const DEFAULT_FILL_RULE: FillRule = FillRule::NonZero;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bounds, size};

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
}
