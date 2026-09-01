//! Multi-stop linear, radial and sweep gradients, baked into a texture and
//! drawn through the image-brush path.
//!
//! [`Background`](crate::Background) expresses a solid colour or a linear
//! gradient with *exactly two* stops. It is eighteen words, embedded by value
//! in every [`Quad`](crate::Quad), and mirrored by hand in four shader
//! languages; widening it to carry a stop list would tax every quad in every
//! scene for something almost no quad does. CSS, meanwhile, routinely asks for
//! three or more stops, for `radial-gradient`, and for
//! `repeating-linear-gradient`.
//!
//! So a gradient here is not a new primitive and not a new shader. It is an
//! image: the stop list is evaluated on the CPU into a small texture, that
//! texture goes into the sprite atlas, and the path is filled with it through
//! the [`PathBrush`](crate::PathBrush) machinery that already exists. The whole
//! feature is a bake plus an affine matrix.
//!
//! # Bake, not shader
//!
//! The alternative was a gradient variant in the fragment shader: a stop buffer,
//! a per-path record naming a kind, and a search per fragment. That would have
//! meant four shader languages to keep in step (Metal, two WGSL dialects and
//! HLSL), a new per-path side buffer beside the brush one, and a stop-list
//! upload path - for a feature whose entire output is a smooth ramp that a
//! texture reproduces to within a quantization step. Baking costs resolution and
//! atlas bytes, both of which are bounded here and stated below; it costs no
//! shader divergence, no new upload path, and it inherits the brush's extend
//! modes, its hand-written bilinear tap and its premultiplication, all of which
//! are already tested.
//!
//! The one place the bake is doing real work rather than avoiding it is the
//! parameter. A linear gradient's parameter is affine in the position, so
//! `screen_to_brush` computes it exactly and the texture is a 1-D ramp - the
//! bake is only quantizing colour, not geometry. A radial gradient's parameter
//! is a distance and a sweep's is an angle; neither is affine, so those bake a
//! 2-D field and the texture is quantizing geometry too. That is why the ramp is
//! 256 texels and cheap while the 2-D bakes are sized to the shape they cover.
//!
//! # Colour
//!
//! Stops are interpolated in the space gpui's own two-stop gradient uses -
//! non-linear sRGB, or Oklab when the caller asks for it, matching `fill_color`
//! in the shaders - and *premultiplied*, which the two-stop shader path is not.
//!
//! The space is a deliberate match: a gradient that blends differently from the
//! one beside it is worse than one that blends the same way as everything else.
//! The premultiplication is a deliberate divergence. Interpolating a colour
//! straight against a stop at `transparent` runs it towards transparent's
//! *black*, so `linear-gradient(rgba(0, 0, 0, .5), transparent)` - which HTML
//! mail is full of - fades through a dark band that no browser paints. Every
//! browser interpolates gradients premultiplied, and so does this.
//!
//! The two-stop shader path still interpolates straight, in four shader
//! languages, under everything gpui's own UI paints; the two agree exactly
//! wherever both stops are opaque, which is every two-stop gradient in gpui
//! today. Bringing it over is a change to `fill_color` in `shaders.metal`,
//! `shaders.wgsl`, `shaders_webgl.wgsl` and `shaders.hlsl` together, and worth
//! doing, but it is a change to what gpui's own UI paints rather than to what
//! this module does.
//!
//! # Memory
//!
//! A linear ramp is [`RAMP_TEXELS`]x1x4 bytes - one kilobyte. A radial or sweep
//! bake is at most [`MAX_FIELD_TEXELS`] square, so a quarter of a megabyte. A
//! window holds [`MAX_GRADIENT_CACHE_BYTES`] of them and then evicts the least
//! recently used one, handing its atlas tile back, so a document with more
//! distinct gradients than that re-bakes as it is scrolled rather than losing
//! its gradients. Only a window whose whole cache is in use in the frame being
//! painted has nothing to evict, and there the gradient is painted flat and the
//! window says so.

use std::f32::consts::TAU;

use crate::{
    Bounds, BrushExtend, ColorSpace, Hsla, LinearColorStop, Pixels, Point, Radians, Rgba, Size,
    TransformationMatrix,
};

/// Texels in a baked linear ramp.
///
/// A linear gradient's parameter is computed exactly by the brush matrix, so
/// these texels quantize colour and nothing else: 256 of them puts a step of the
/// ramp on every distinguishable 8-bit level.
pub const RAMP_TEXELS: u32 = 256;

/// The smallest square a radial or sweep gradient bakes into.
pub const MIN_FIELD_TEXELS: u32 = 64;

/// The largest square a radial or sweep gradient bakes into.
///
/// A 2-D bake quantizes *geometry*, so this is a resolution limit as much as a
/// memory one: a gradient stretched over a box much wider than this is a
/// bilinear magnification of the bake, which is smooth where the stops are
/// smooth and soft where they are sharp.
pub const MAX_FIELD_TEXELS: u32 = 256;

/// The fewest texels a 2-D bake spends on one traversal of the stop list.
///
/// A repeating radial or sweep gradient runs the whole stop list many times
/// across one baked square, and the square's texel count is what has to resolve
/// every one of those runs: at one texel per traversal the bake is not a coarse
/// gradient, it is noise. Four is what a bilinear magnification needs to keep a
/// ramp reading as a ramp, and [`MAX_FIELD_TEXELS`] divided by it is what caps
/// how many traversals a bake will try to hold.
pub const TEXELS_PER_PERIOD: u32 = 4;

/// How many bytes of baked gradient one window holds in its atlas at once.
///
/// A working set, not a ceiling the window runs into and never comes back from:
/// past it the least recently used bake is dropped and its atlas tile given
/// back. It is sized so that a page of distinct gradients fits without
/// thrashing - the largest bake is [`MAX_FIELD_TEXELS`] square, a quarter of a
/// megabyte, so this holds sixty-four of those or thousands of linear ramps.
pub const MAX_GRADIENT_CACHE_BYTES: usize = 16 * 1024 * 1024;

/// Where a [`Gradient`]'s colours run.
///
/// Every geometry here is in the same logical pixels the path is authored in.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GradientKind {
    /// Colours run along the line from `start` to `end`: the first stop sits at
    /// `start`, the last at `end`, and the gradient is constant along every line
    /// perpendicular to that one. CSS `linear-gradient`.
    Linear {
        /// Where the stop at 0 sits.
        start: Point<Pixels>,
        /// Where the stop at 1 sits.
        end: Point<Pixels>,
    },
    /// Colours run outward from `center`, reaching the last stop on the ellipse
    /// with the given radii. CSS `radial-gradient`.
    Radial {
        /// The centre the colours run out from.
        center: Point<Pixels>,
        /// The horizontal and vertical radius at which the parameter reaches 1.
        radii: Size<Pixels>,
    },
    /// Colours run around `center`, starting at `from_angle` and completing the
    /// stop list after `sweep_angle` of rotation. CSS `conic-gradient`.
    ///
    /// Angles are clockwise from the positive x axis, which in gpui's
    /// y-downwards coordinates is clockwise on screen. A CSS
    /// `conic-gradient(from 0deg)` starts at twelve o'clock, so its `from_angle`
    /// is a quarter turn less than the CSS one.
    Sweep {
        /// The centre the colours run around.
        center: Point<Pixels>,
        /// Where the stop at 0 sits, clockwise from the positive x axis.
        from_angle: Radians,
        /// How far the stop list runs before the extend mode takes over. A whole
        /// turn is the CSS default.
        sweep_angle: Radians,
    },
}

impl GradientKind {
    /// Whether every number this geometry is made of is a real one.
    fn is_finite(&self) -> bool {
        let point =
            |point: Point<Pixels>| f32::from(point.x).is_finite() && f32::from(point.y).is_finite();
        match *self {
            GradientKind::Linear { start, end } => point(start) && point(end),
            GradientKind::Radial { center, radii } => {
                point(center)
                    && f32::from(radii.width).is_finite()
                    && f32::from(radii.height).is_finite()
            }
            GradientKind::Sweep {
                center,
                from_angle,
                sweep_angle,
            } => point(center) && from_angle.0.is_finite() && sweep_angle.0.is_finite(),
        }
    }
}

/// Whether every number a rectangle is made of is a real one.
fn bounds_are_finite(bounds: Bounds<Pixels>) -> bool {
    [
        bounds.origin.x,
        bounds.origin.y,
        bounds.size.width,
        bounds.size.height,
    ]
    .into_iter()
    .all(|value| f32::from(value).is_finite())
}

/// A multi-stop gradient: what to paint, not where.
///
/// Hand one to [`crate::Window::paint_path_with_gradient`] along with the path
/// it fills.
#[derive(Clone, Debug, PartialEq)]
pub struct Gradient {
    /// The geometry the parameter is computed from.
    pub kind: GradientKind,
    /// The colour stops, positioned in 0..=1 along the gradient.
    ///
    /// They are read in the order given, with each position raised to at least
    /// its predecessor's the way CSS defines it, so an out-of-order list gives a
    /// hard stop rather than a reversal.
    pub stops: Vec<LinearColorStop>,
    /// What fills the path beyond the stop list.
    ///
    /// [`BrushExtend::Repeat`] on a linear gradient is CSS
    /// `repeating-linear-gradient`; on a sweep it is what makes a whole-turn
    /// conic gradient join up at its seam.
    pub extend: BrushExtend,
    /// The space the stops are interpolated in.
    pub color_space: ColorSpace,
}

impl Gradient {
    /// A linear gradient running from `start` to `end`.
    pub fn linear(
        start: Point<Pixels>,
        end: Point<Pixels>,
        stops: impl Into<Vec<LinearColorStop>>,
    ) -> Self {
        Self {
            kind: GradientKind::Linear { start, end },
            stops: stops.into(),
            extend: BrushExtend::Pad,
            color_space: ColorSpace::default(),
        }
    }

    /// A radial gradient centred on `center`, reaching its last stop at `radii`.
    pub fn radial(
        center: Point<Pixels>,
        radii: Size<Pixels>,
        stops: impl Into<Vec<LinearColorStop>>,
    ) -> Self {
        Self {
            kind: GradientKind::Radial { center, radii },
            stops: stops.into(),
            extend: BrushExtend::Pad,
            color_space: ColorSpace::default(),
        }
    }

    /// A sweep gradient turning about `center`, a whole turn clockwise from
    /// `from_angle`.
    pub fn sweep(
        center: Point<Pixels>,
        from_angle: Radians,
        stops: impl Into<Vec<LinearColorStop>>,
    ) -> Self {
        Self {
            kind: GradientKind::Sweep {
                center,
                from_angle,
                sweep_angle: Radians(TAU),
            },
            stops: stops.into(),
            extend: BrushExtend::Repeat,
            color_space: ColorSpace::default(),
        }
    }

    /// Set what fills the path beyond the stop list.
    pub fn extend(mut self, extend: BrushExtend) -> Self {
        self.extend = extend;
        self
    }

    /// Set the space the stops are interpolated in.
    pub fn color_space(mut self, color_space: ColorSpace) -> Self {
        self.color_space = color_space;
        self
    }

    /// The colour the stop list comes to halfway along: the flat fill a
    /// renderer that cannot resolve a brush paints instead of this gradient,
    /// and the one a degenerate or unbakeable gradient collapses to.
    ///
    /// Worth caching beside a baked ramp rather than asking again per frame:
    /// it is the same colour every time, and arriving at it normalizes the
    /// stop list and converts every stop into the interpolation space - a
    /// `powf` and a `cbrt` apiece under Oklab - for a value the Metal renderer
    /// never reads.
    pub(crate) fn flat_midpoint(&self) -> Hsla {
        sample(&self.normalized_stops(), 0.5, self.color_space).into()
    }

    /// [`Gradient::flat_midpoint`] at `alpha`.
    pub(crate) fn flat_fallback(&self, alpha: f32) -> crate::Background {
        crate::solid_background(self.flat_midpoint()).opacity(alpha)
    }

    /// The stops as the bake reads them: converted once into the interpolation
    /// space, clamped into 0..=1, and made monotone the way CSS makes them.
    fn normalized_stops(&self) -> Vec<(f32, [f32; 4])> {
        let mut highest = 0.0f32;
        self.stops
            .iter()
            .map(|stop| {
                let position = stop.percentage.clamp(0., 1.).max(highest);
                highest = position;
                (
                    position,
                    to_interpolation_space(stop.color, self.color_space),
                )
            })
            .collect()
    }
}

/// Everything a window needs to turn one [`Gradient`] into one brushed path:
/// the texture to bake, the matrix that finds it, and the key both are cached
/// under.
pub(crate) struct GradientPlan {
    pub key: GradientKey,
    pub screen_to_brush: TransformationMatrix,
    pub x_extend: BrushExtend,
    pub y_extend: BrushExtend,
    field: Field,
}

/// What the baked texels mean, and so how the bake walks them.
#[derive(Copy, Clone, Debug, PartialEq)]
enum Field {
    /// One row: texel `i` is the stop list at `(i + 0.5) / width`.
    Ramp,
    /// A square of gradient space, `half_extent` radii either side of the
    /// centre; the parameter is the distance from the middle.
    Radial { half_extent: u32 },
    /// A square centred on the sweep's centre; the parameter is the angle.
    Sweep { from_angle: f32, sweep_angle: f32 },
}

/// What one baked texture is cached under.
///
/// Everything the bake reads is in here and nothing else is, so two paths whose
/// gradients agree share one texture and one atlas tile however far apart on the
/// page they are. Floats go in as bit patterns: a gradient repeated down a
/// document is repeated verbatim, so bitwise equality is exactly the question
/// being asked.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GradientKey {
    field: FieldKey,
    extend: u8,
    color_space: u8,
    width: u32,
    height: u32,
    stops: Vec<[u32; 5]>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum FieldKey {
    Ramp,
    Radial { half_extent: u32 },
    Sweep { from_angle: u32, sweep_angle: u32 },
}

impl GradientKey {
    /// The bytes this key's texture occupies in the atlas.
    pub fn byte_size(&self) -> usize {
        self.width as usize * self.height as usize * 4
    }
}

impl Gradient {
    /// Work out the texture and the matrix for this gradient filling
    /// `path_bounds` at `scale_factor`, or `None` if the geometry is degenerate
    /// - a zero-length axis, a zero radius, a zero sweep - and there is nothing
    ///   to run a parameter along.
    pub(crate) fn plan(
        &self,
        path_bounds: Bounds<Pixels>,
        scale_factor: f32,
    ) -> Option<GradientPlan> {
        if self.stops.is_empty() || !scale_factor.is_finite() || scale_factor <= 0. {
            return None;
        }
        // A NaN passes every ordinary bounds check - `NaN <= EPSILON` is false
        // and so is `NaN > EPSILON` - so a NaN centre or radius would be taken
        // for a well-formed one, baked into a matrix whose every product is
        // NaN, and reach `int2(floor(NaN))` in the brush shader, which Metal
        // leaves undefined. Refusing the whole gradient once, here, is what
        // makes that impossible rather than merely unlikely.
        if !self.kind.is_finite() || !bounds_are_finite(path_bounds) {
            return None;
        }
        match self.kind {
            GradientKind::Linear { start, end } => self.plan_linear(start, end, scale_factor),
            GradientKind::Radial { center, radii } => {
                self.plan_radial(center, radii, path_bounds, scale_factor)
            }
            GradientKind::Sweep {
                center,
                from_angle,
                sweep_angle,
            } => self.plan_sweep(
                center,
                from_angle.0,
                sweep_angle.0,
                path_bounds,
                scale_factor,
            ),
        }
    }

    fn plan_linear(
        &self,
        start: Point<Pixels>,
        end: Point<Pixels>,
        scale_factor: f32,
    ) -> Option<GradientPlan> {
        let (dx, dy) = (f32::from(end.x - start.x), f32::from(end.y - start.y));
        let length_squared = dx * dx + dy * dy;
        if length_squared <= f32::EPSILON {
            return None;
        }
        let (sx, sy) = (f32::from(start.x), f32::from(start.y));
        // A device position `q` becomes the gradient's parameter in one step:
        // divide out the scale factor, project onto the axis, and normalize by
        // its length. The second row is constant, which makes the matrix
        // singular - nothing downstream inverts it, and a one-row texture has
        // nowhere else for the second coordinate to go.
        let screen_to_brush = TransformationMatrix {
            rotation_scale: [
                [
                    dx / (length_squared * scale_factor),
                    dy / (length_squared * scale_factor),
                ],
                [0., 0.],
            ],
            translation: [-(sx * dx + sy * dy) / length_squared, 0.5],
        };
        Some(GradientPlan {
            key: self.key(FieldKey::Ramp, RAMP_TEXELS, 1),
            screen_to_brush,
            x_extend: self.extend,
            y_extend: BrushExtend::Pad,
            field: Field::Ramp,
        })
    }

    fn plan_radial(
        &self,
        center: Point<Pixels>,
        radii: Size<Pixels>,
        path_bounds: Bounds<Pixels>,
        scale_factor: f32,
    ) -> Option<GradientPlan> {
        let (rx, ry) = (f32::from(radii.width), f32::from(radii.height));
        // The finiteness is asked for explicitly rather than left to the
        // comparisons, which a NaN passes: `NaN <= EPSILON` is false, so a NaN
        // radius would be taken for a well-formed one here and reach
        // `int2(floor(NaN))` in the brush shader.
        if !rx.is_finite() || !ry.is_finite() || rx <= f32::EPSILON || ry <= f32::EPSILON {
            return None;
        }
        let (cx, cy) = (f32::from(center.x), f32::from(center.y));
        // How many radii out the path reaches, which is how much of gradient
        // space has to be baked. Under `Pad` the answer is always one: past the
        // last stop the colour is constant, so the square's own edge texels are
        // already that colour and the brush's `Pad` carries it outward forever.
        //
        // The four distances are taken along the axes, and that is enough for
        // the whole rectangle: the bake covers the square `[-h, h]` in *each*
        // coordinate, so what has to be bounded is `|gx|` and `|gy|`
        // separately, and each is largest at an edge. The corner of the square
        // is reached too - its parameter is `h * sqrt(2)`, which the bake
        // computes from the corner texel's own `hypot` - so a corner is a place
        // the ring count runs highest, not a place the square runs out.
        let half_extent = match self.extend {
            BrushExtend::Pad => 1,
            BrushExtend::Repeat | BrushExtend::Reflect => {
                let reach = [
                    (f32::from(path_bounds.left()) - cx).abs() / rx,
                    (f32::from(path_bounds.right()) - cx).abs() / rx,
                    (f32::from(path_bounds.top()) - cy).abs() / ry,
                    (f32::from(path_bounds.bottom()) - cy).abs() / ry,
                ]
                .into_iter()
                .fold(0f32, f32::max);
                // Past this the rings are finer than the bake can hold: the
                // square is `2h` traversals wide and only [`MAX_FIELD_TEXELS`]
                // texels wide, so `h` beyond this would be resolved at fewer
                // than [`TEXELS_PER_PERIOD`] texels a ring. A path that reaches
                // further than the cap pads beyond the last ring the bake
                // holds, which is a smear where the alternative is noise.
                (reach.ceil() as u32).clamp(1, MAX_FIELD_TEXELS / (2 * TEXELS_PER_PERIOD))
            }
        };
        // The square is `2 * half_extent` traversals of the stop list across,
        // and every one of them has to be resolved by the same texels.
        let texels = field_texels(path_bounds, scale_factor, 2 * half_extent);
        let span = 2. * half_extent as f32;
        let screen_to_brush = TransformationMatrix {
            rotation_scale: [
                [1. / (scale_factor * rx * span), 0.],
                [0., 1. / (scale_factor * ry * span)],
            ],
            translation: [0.5 - cx / (rx * span), 0.5 - cy / (ry * span)],
        };
        Some(GradientPlan {
            key: self.key(FieldKey::Radial { half_extent }, texels, texels),
            screen_to_brush,
            // Everything outside the baked square is past the last ring the path
            // can reach, so padding it is padding a colour the extend mode has
            // already had its say about.
            x_extend: BrushExtend::Pad,
            y_extend: BrushExtend::Pad,
            field: Field::Radial { half_extent },
        })
    }

    fn plan_sweep(
        &self,
        center: Point<Pixels>,
        from_angle: f32,
        sweep_angle: f32,
        path_bounds: Bounds<Pixels>,
        scale_factor: f32,
    ) -> Option<GradientPlan> {
        if sweep_angle.abs() <= f32::EPSILON || !sweep_angle.is_finite() {
            return None;
        }
        let (cx, cy) = (f32::from(center.x), f32::from(center.y));
        // An angle does not care how far out it is measured, so the baked square
        // can be any size that covers the path - which means the texture is the
        // same one whatever the path's geometry, and one bake serves every sweep
        // with these stops.
        let half_extent = [
            (f32::from(path_bounds.left()) - cx).abs(),
            (f32::from(path_bounds.right()) - cx).abs(),
            (f32::from(path_bounds.top()) - cy).abs(),
            (f32::from(path_bounds.bottom()) - cy).abs(),
        ]
        .into_iter()
        .fold(0f32, f32::max)
        .max(1.);
        // A sweep whose stop list is shorter than a turn and repeats runs the
        // list `TAU / sweep_angle` times around the circle, and the texels have
        // to resolve every one of those runs the same way a radial's do.
        let periods = match self.extend {
            BrushExtend::Pad => 1,
            BrushExtend::Repeat | BrushExtend::Reflect => {
                ((TAU / sweep_angle.abs()).ceil() as u32).clamp(1, MAX_FIELD_TEXELS)
            }
        };
        let texels = field_texels(path_bounds, scale_factor, periods);
        let span = 2. * half_extent;
        let screen_to_brush = TransformationMatrix {
            rotation_scale: [
                [1. / (scale_factor * span), 0.],
                [0., 1. / (scale_factor * span)],
            ],
            translation: [0.5 - cx / span, 0.5 - cy / span],
        };
        Some(GradientPlan {
            key: self.key(
                FieldKey::Sweep {
                    from_angle: from_angle.to_bits(),
                    sweep_angle: sweep_angle.to_bits(),
                },
                texels,
                texels,
            ),
            screen_to_brush,
            x_extend: BrushExtend::Pad,
            y_extend: BrushExtend::Pad,
            field: Field::Sweep {
                from_angle,
                sweep_angle,
            },
        })
    }

    fn key(&self, field: FieldKey, width: u32, height: u32) -> GradientKey {
        GradientKey {
            field,
            extend: self.extend as u8,
            color_space: self.color_space as u8,
            width,
            height,
            stops: self
                .stops
                .iter()
                .map(|stop| {
                    [
                        stop.color.h.to_bits(),
                        stop.color.s.to_bits(),
                        stop.color.l.to_bits(),
                        stop.color.a.to_bits(),
                        stop.percentage.to_bits(),
                    ]
                })
                .collect(),
        }
    }
}

impl GradientPlan {
    /// The texture's size in texels, which the key already names: it is part of
    /// what a bake is cached under, so holding it twice is one more thing to
    /// keep in step.
    pub(crate) fn size(&self) -> Size<u32> {
        Size {
            width: self.key.width,
            height: self.key.height,
        }
    }

    /// Evaluate the stop list into the straight (un-premultiplied) BGRA the
    /// sprite atlas holds.
    ///
    /// BGRA rather than RGBA because that is what the atlas stores and what
    /// `sample_path_brush` reads back; straight rather than premultiplied
    /// because the path pipeline premultiplies the sample itself, after
    /// multiplying in the brush opacity and the path's coverage.
    pub(crate) fn bake(&self, gradient: &Gradient) -> Vec<u8> {
        let stops = gradient.normalized_stops();
        let (width, height) = (self.key.width, self.key.height);
        let mut bytes = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let parameter = self.parameter(x, y, width, height, gradient.extend);
                let color = sample(&stops, parameter, gradient.color_space);
                bytes.extend_from_slice(&[
                    channel(color.b),
                    channel(color.g),
                    channel(color.r),
                    channel(color.a),
                ]);
            }
        }
        bytes
    }

    /// The stop-list parameter texel `(x, y)` stands for, with the extend mode
    /// already folded in.
    ///
    /// A ramp does not fold it: the brush's own `x_extend` wraps the texel index
    /// itself, which is what makes `repeating-linear-gradient` free. A 2-D field
    /// has to, because the repetition is radial or angular and no texel index
    /// wraps that way.
    fn parameter(&self, x: u32, y: u32, width: u32, height: u32, extend: BrushExtend) -> f32 {
        match self.field {
            Field::Ramp => (x as f32 + 0.5) / width as f32,
            Field::Radial { half_extent } => {
                let gx = ((x as f32 + 0.5) / width as f32 * 2. - 1.) * half_extent as f32;
                let gy = ((y as f32 + 0.5) / height as f32 * 2. - 1.) * half_extent as f32;
                fold(gx.hypot(gy), extend)
            }
            Field::Sweep {
                from_angle,
                sweep_angle,
            } => {
                let gx = (x as f32 + 0.5) / width as f32 * 2. - 1.;
                let gy = (y as f32 + 0.5) / height as f32 * 2. - 1.;
                let turned = (gy.atan2(gx) - from_angle).rem_euclid(TAU);
                fold(turned / sweep_angle, extend)
            }
        }
    }
}

/// The square a radial or sweep gradient bakes into, sized both to the shape it
/// covers and to the number of times the stop list runs across it.
///
/// A gradient in a thumbnail should not cost what one across a page costs, and
/// one across a page should not be visibly coarse. Powers of two so that the
/// same gradient at slightly different sizes still shares a bake.
///
/// `periods` is what a repeating field adds to that: the shape's size says how
/// finely the bake will be magnified, but a bake that runs the stop list twenty
/// times across itself needs texels for twenty ramps whatever size it is drawn
/// at, and a square sized on the shape alone can end up holding a whole
/// traversal in a single texel.
fn field_texels(path_bounds: Bounds<Pixels>, scale_factor: f32, periods: u32) -> u32 {
    let device =
        f32::from(path_bounds.size.width).max(f32::from(path_bounds.size.height)) * scale_factor;
    let wanted = (device.max(1.) as u32)
        .max(periods.saturating_mul(TEXELS_PER_PERIOD))
        .next_power_of_two();
    wanted.clamp(MIN_FIELD_TEXELS, MAX_FIELD_TEXELS)
}

/// Fold a parameter that ran off the end of the stop list back onto 0..=1, the
/// way [`BrushExtend`] says to.
fn fold(parameter: f32, extend: BrushExtend) -> f32 {
    match extend {
        BrushExtend::Pad => parameter.clamp(0., 1.),
        BrushExtend::Repeat => parameter.rem_euclid(1.),
        BrushExtend::Reflect => {
            let wrapped = parameter.rem_euclid(2.);
            if wrapped <= 1. { wrapped } else { 2. - wrapped }
        }
    }
}

/// The colour the stop list holds at `parameter`, in sRGB.
fn sample(stops: &[(f32, [f32; 4])], parameter: f32, color_space: ColorSpace) -> Rgba {
    let parameter = parameter.clamp(0., 1.);
    let Some((first_position, first_color)) = stops.first().copied() else {
        return Rgba {
            r: 0.,
            g: 0.,
            b: 0.,
            a: 0.,
        };
    };
    let mut blended = first_color;
    if parameter > first_position {
        blended = stops.last().expect("there is a first stop").1;
        for pair in stops.windows(2) {
            let ((low, low_color), (high, high_color)) = (pair[0], pair[1]);
            if parameter <= high {
                let span = high - low;
                let t = if span <= f32::EPSILON {
                    1.
                } else {
                    (parameter - low) / span
                };
                blended = interpolate(low_color, high_color, t);
                break;
            }
        }
    }
    from_interpolation_space(blended, color_space)
}

/// Blend two stops `t` of the way from the first to the second, with the colour
/// premultiplied by its alpha and the alpha divided back out afterwards.
///
/// Premultiplied because a straight interpolation runs a colour towards
/// `transparent`'s *black* as the alpha comes down, so
/// `linear-gradient(rgba(0, 0, 0, .5), transparent)` - one of the commonest
/// things in an HTML email, and `to transparent` from any colour with it - fades
/// through a dark band no browser paints. Weighting by alpha keeps the colour
/// where the stops put it and takes only the alpha down, which is what CSS says
/// and what browsers do.
///
/// At zero alpha there is no colour to recover, and the straight blend is used
/// as-is: nothing samples it directly, but the brush's bilinear tap reads the
/// texel next to it, and a texel that quietly turned black would put a dark
/// fringe one texel wide back into the ramp.
fn interpolate(low: [f32; 4], high: [f32; 4], t: f32) -> [f32; 4] {
    let straight = |index: usize| low[index] + (high[index] - low[index]) * t;
    let alpha = straight(3);
    if alpha <= 0. {
        return [straight(0), straight(1), straight(2), 0.];
    }
    let premultiplied = |index: usize| {
        let (low, high) = (low[index] * low[3], high[index] * high[3]);
        (low + (high - low) * t) / alpha
    };
    [premultiplied(0), premultiplied(1), premultiplied(2), alpha]
}

/// A stop's colour in the space the interpolation happens in.
fn to_interpolation_space(color: Hsla, color_space: ColorSpace) -> [f32; 4] {
    let rgba = Rgba::from(color);
    match color_space {
        ColorSpace::Srgb => [rgba.r, rgba.g, rgba.b, rgba.a],
        ColorSpace::Oklab => srgb_to_oklab([rgba.r, rgba.g, rgba.b, rgba.a]),
    }
}

fn from_interpolation_space(color: [f32; 4], color_space: ColorSpace) -> Rgba {
    let [r, g, b, a] = match color_space {
        ColorSpace::Srgb => color,
        ColorSpace::Oklab => oklab_to_srgb(color),
    };
    Rgba { r, g, b, a }
}

/// The same transfer function `srgb_to_linear` uses in the shaders: a plain 2.2
/// power, not the piecewise sRGB curve. Matching it matters more than being
/// right, because a gradient that converts differently from the two-stop one
/// beside it reads as a bug.
fn srgb_to_oklab(color: [f32; 4]) -> [f32; 4] {
    let [r, g, b, a] = color;
    let (r, g, b) = (r.powf(2.2), g.powf(2.2), b.powf(2.2));
    let l = (0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
    let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
        a,
    ]
}

fn oklab_to_srgb(color: [f32; 4]) -> [f32; 4] {
    let [lightness, green_red, blue_yellow, a] = color;
    let l = lightness + 0.396_337_78 * green_red + 0.215_803_76 * blue_yellow;
    let m = lightness - 0.105_561_346 * green_red - 0.063_854_17 * blue_yellow;
    let s = lightness - 0.089_484_18 * green_red - 1.291_485_5 * blue_yellow;
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    let red = 4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s;
    let green = -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s;
    let blue = -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s;
    [
        red.max(0.).powf(1. / 2.2),
        green.max(0.).powf(1. / 2.2),
        blue.max(0.).powf(1. / 2.2),
        a,
    ]
}

fn channel(value: f32) -> u8 {
    (value.clamp(0., 1.) * 255. + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bounds, linear_color_stop, point, px, rgb, rgba, size};

    fn stops() -> Vec<LinearColorStop> {
        vec![
            linear_color_stop(rgb(0xff0000), 0.),
            linear_color_stop(rgb(0x00ff00), 0.5),
            linear_color_stop(rgb(0x0000ff), 1.),
        ]
    }

    fn bounds() -> Bounds<Pixels> {
        Bounds {
            origin: point(px(0.), px(0.)),
            size: size(px(100.), px(100.)),
        }
    }

    #[test]
    fn a_ramp_holds_the_middle_stop_in_its_middle_texel() {
        let gradient = Gradient::linear(point(px(0.), px(0.)), point(px(100.), px(0.)), stops());
        let plan = gradient.plan(bounds(), 1.).expect("a well-formed axis");
        let baked = plan.bake(&gradient);
        let middle = (RAMP_TEXELS / 2) as usize * 4;
        // BGRA: the middle stop is pure green, so the green channel is the only
        // one lit.
        assert!(baked[middle + 1] > 250, "the middle texel is not green");
        assert!(baked[middle] < 5 && baked[middle + 2] < 5);
    }

    #[test]
    fn a_stop_position_moves_the_colour_along_the_ramp() {
        let gradient = Gradient::linear(
            point(px(0.), px(0.)),
            point(px(100.), px(0.)),
            vec![
                linear_color_stop(rgb(0xff0000), 0.),
                linear_color_stop(rgb(0x00ff00), 0.25),
                linear_color_stop(rgb(0x0000ff), 1.),
            ],
        );
        let plan = gradient.plan(bounds(), 1.).expect("a well-formed axis");
        let baked = plan.bake(&gradient);
        let quarter = (RAMP_TEXELS / 4) as usize * 4;
        assert!(baked[quarter + 1] > 250, "the quarter texel is not green");
    }

    #[test]
    fn the_brush_matrix_puts_the_axis_ends_at_zero_and_one() {
        let gradient = Gradient::linear(point(px(10.), px(10.)), point(px(50.), px(30.)), stops());
        let plan = gradient.plan(bounds(), 2.).expect("a well-formed axis");
        // Device pixels in, brush space out: the axis's ends are the unit
        // square's ends.
        let at_start = plan.screen_to_brush.apply(point(px(20.), px(20.)));
        let at_end = plan.screen_to_brush.apply(point(px(100.), px(60.)));
        assert!(f32::from(at_start.x).abs() < 1e-4);
        assert!((f32::from(at_end.x) - 1.).abs() < 1e-4);
    }

    #[test]
    fn a_degenerate_gradient_has_no_plan() {
        let zero_length =
            Gradient::linear(point(px(10.), px(10.)), point(px(10.), px(10.)), stops());
        assert!(zero_length.plan(bounds(), 1.).is_none());
        let no_radius = Gradient::radial(point(px(0.), px(0.)), size(px(0.), px(10.)), stops());
        assert!(no_radius.plan(bounds(), 1.).is_none());
        let no_stops = Gradient::linear(point(px(0.), px(0.)), point(px(10.), px(0.)), vec![]);
        assert!(no_stops.plan(bounds(), 1.).is_none());
    }

    #[test]
    fn a_padded_radial_bakes_one_radius_however_far_the_path_reaches() {
        let gradient = Gradient::radial(point(px(50.), px(50.)), size(px(10.), px(10.)), stops());
        let plan = gradient.plan(bounds(), 1.).expect("a well-formed radius");
        assert_eq!(plan.key.field, FieldKey::Radial { half_extent: 1 });
    }

    #[test]
    fn a_repeating_radial_bakes_out_to_the_path() {
        let gradient = Gradient::radial(point(px(50.), px(50.)), size(px(10.), px(10.)), stops())
            .extend(BrushExtend::Repeat);
        let plan = gradient.plan(bounds(), 1.).expect("a well-formed radius");
        assert_eq!(plan.key.field, FieldKey::Radial { half_extent: 5 });
    }

    #[test]
    fn two_identical_gradients_share_a_key_and_two_different_ones_do_not() {
        let one = Gradient::linear(point(px(0.), px(0.)), point(px(100.), px(0.)), stops());
        let two = Gradient::linear(point(px(0.), px(0.)), point(px(40.), px(40.)), stops());
        let three = one.clone().extend(BrushExtend::Repeat);
        let plans = [
            one.plan(bounds(), 1.).expect("a well-formed axis"),
            two.plan(bounds(), 1.).expect("a well-formed axis"),
            three.plan(bounds(), 1.).expect("a well-formed axis"),
        ];
        assert_eq!(
            plans[0].key, plans[1].key,
            "the same stops down a page bake once, wherever the axis runs"
        );
        assert_ne!(plans[0].key, plans[2].key, "the extend mode is baked in");
    }

    #[test]
    fn a_radial_bake_is_radially_symmetric() {
        let gradient = Gradient::radial(point(px(50.), px(50.)), size(px(50.), px(50.)), stops());
        let plan = gradient.plan(bounds(), 1.).expect("a well-formed radius");
        let baked = plan.bake(&gradient);
        let texels = plan.size.width;
        let at = |x: u32, y: u32| {
            let index = ((y * texels + x) * 4) as usize;
            [baked[index], baked[index + 1], baked[index + 2]]
        };
        let (near, far) = (texels / 4, texels - 1 - texels / 4);
        assert_eq!(at(near, texels / 2), at(far, texels / 2));
        assert_eq!(at(texels / 2, near), at(texels / 2, far));
        assert_ne!(at(texels / 2, texels / 2), at(0, 0));
    }

    #[test]
    fn a_stop_that_fades_to_transparent_keeps_its_colour_on_the_way() {
        // `linear-gradient(red, transparent)`, which HTML mail is full of.
        // Interpolated straight, the colour runs towards transparent's *black*
        // as its alpha falls, so the middle of the ramp is a half-alpha dark
        // red that a browser never paints; interpolated premultiplied, the
        // middle is red at half alpha.
        let gradient = Gradient::linear(
            point(px(0.), px(0.)),
            point(px(100.), px(0.)),
            vec![
                linear_color_stop(rgb(0xff0000), 0.),
                linear_color_stop(rgba(0x00000000), 1.),
            ],
        );
        let baked = gradient
            .plan(bounds(), 1.)
            .expect("a well-formed axis")
            .bake(&gradient);
        let middle = (RAMP_TEXELS / 2) as usize * 4;

        // BGRA, straight: the alpha halves and the colour does not move.
        assert!(
            baked[middle + 2] > 250,
            "the middle of a red-to-transparent ramp came back with a red \
             channel of {}, where a premultiplied interpolation holds it at \
             full red and only takes the alpha down",
            baked[middle + 2]
        );
        assert!(
            baked[middle + 3].abs_diff(128) <= 2,
            "the middle of a red-to-transparent ramp should be half alpha, and \
             is {}",
            baked[middle + 3]
        );
        // The end is transparent whichever way it was interpolated, so the
        // assertion above is not passing on a ramp that never faded at all.
        assert_eq!(baked[baked.len() - 1], 0, "the last texel is transparent");
    }

    #[test]
    fn a_gradient_made_of_nans_has_no_plan() {
        // Every geometric guard is a bound, and a NaN fails every bound in both
        // directions, so an unguarded NaN is taken for a well-formed number and
        // reaches `int2(floor(NaN))` in the brush shader.
        let nan = f32::NAN;
        let degenerate = [
            Gradient::radial(point(px(0.), px(0.)), size(px(nan), px(10.)), stops()),
            Gradient::radial(point(px(nan), px(0.)), size(px(10.), px(10.)), stops()),
            Gradient::linear(point(px(0.), px(0.)), point(px(nan), px(0.)), stops()),
            Gradient::sweep(point(px(nan), px(0.)), Radians(0.), stops()),
            Gradient::sweep(point(px(0.), px(0.)), Radians(nan), stops()),
        ];
        for gradient in degenerate {
            assert!(
                gradient.plan(bounds(), 1.).is_none(),
                "{:?} was planned rather than refused",
                gradient.kind
            );
        }

        let well_formed = Gradient::radial(point(px(0.), px(0.)), size(px(10.), px(10.)), stops());
        assert!(
            well_formed.plan(bounds(), 1.).is_some(),
            "the control has to be planned, or the refusals above prove nothing"
        );
        assert!(
            well_formed
                .plan(
                    Bounds {
                        origin: point(px(nan), px(0.)),
                        size: size(px(100.), px(100.)),
                    },
                    1.
                )
                .is_none(),
            "a NaN reaches the plan through the path's bounds as well"
        );
        assert!(
            well_formed.plan(bounds(), nan).is_none(),
            "and through the scale factor"
        );
    }

    #[test]
    fn a_repeating_radial_bakes_enough_texels_to_resolve_its_rings() {
        // Forty logical pixels across and two-pixel radii: ten rings out to the
        // edge, twenty across the baked square. Sized on the shape alone the
        // square is 64 texels, which is three texels a ring, and the ramp in
        // each of them is whatever two of its stops happen to land on.
        let small = Bounds {
            origin: point(px(0.), px(0.)),
            size: size(px(40.), px(40.)),
        };
        let gradient = Gradient::radial(point(px(20.), px(20.)), size(px(2.), px(2.)), stops())
            .extend(BrushExtend::Repeat);
        let plan = gradient.plan(small, 1.).expect("a well-formed radius");

        let FieldKey::Radial { half_extent } = plan.key.field else {
            panic!("a radial gradient planned something other than a radial field");
        };
        assert_eq!(half_extent, 10, "ten rings out to the path's edge");
        assert!(
            plan.size.width >= 2 * half_extent * TEXELS_PER_PERIOD,
            "a bake {} texels across has to resolve {} traversals of the stop \
             list, which is {} texels each",
            plan.size.width,
            2 * half_extent,
            plan.size.width / (2 * half_extent),
        );
    }

    #[test]
    fn a_repeating_sweep_bakes_enough_texels_for_every_turn_of_its_stop_list() {
        // A stop list that runs thirty-two times around the circle needs texels
        // for thirty-two ramps, however small the shape it is painted on.
        let small = Bounds {
            origin: point(px(0.), px(0.)),
            size: size(px(40.), px(40.)),
        };
        let turns = 32;
        let gradient = Gradient {
            kind: GradientKind::Sweep {
                center: point(px(20.), px(20.)),
                from_angle: Radians(0.),
                sweep_angle: Radians(TAU / turns as f32),
            },
            stops: stops(),
            extend: BrushExtend::Repeat,
            color_space: ColorSpace::default(),
        };
        let plan = gradient.plan(small, 1.).expect("a well-formed sweep");

        assert!(
            plan.size.width >= turns * TEXELS_PER_PERIOD,
            "a sweep whose stop list runs {turns} times around the circle baked \
             into {} texels",
            plan.size.width,
        );
    }

    #[test]
    fn a_padded_gradient_is_still_sized_by_the_shape_it_covers() {
        // The control for the two above: sizing by the repeat count must not
        // have taken over from sizing by the shape, or every gradient would
        // bake at the ceiling.
        let small = Bounds {
            origin: point(px(0.), px(0.)),
            size: size(px(40.), px(40.)),
        };
        let gradient = Gradient::radial(point(px(20.), px(20.)), size(px(2.), px(2.)), stops());
        let plan = gradient.plan(small, 1.).expect("a well-formed radius");
        assert_eq!(plan.size.width, MIN_FIELD_TEXELS);
    }

    #[test]
    fn oklab_and_srgb_disagree_in_the_middle() {
        let axis = (point(px(0.), px(0.)), point(px(100.), px(0.)));
        let two = vec![
            linear_color_stop(rgb(0x000000), 0.),
            linear_color_stop(rgb(0xffffff), 1.),
        ];
        let srgb = Gradient::linear(axis.0, axis.1, two);
        let oklab = srgb.clone().color_space(ColorSpace::Oklab);
        let middle = (RAMP_TEXELS / 2) as usize * 4;
        let srgb_baked = srgb.plan(bounds(), 1.).expect("an axis").bake(&srgb);
        let oklab_baked = oklab.plan(bounds(), 1.).expect("an axis").bake(&oklab);
        assert_ne!(srgb_baked[middle], oklab_baked[middle]);
    }
}
