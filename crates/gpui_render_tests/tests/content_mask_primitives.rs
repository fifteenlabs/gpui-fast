//! What a *rounded* content mask does to every primitive that can land inside
//! one.
//!
//! `content_mask.rs` pins the plain cases: a flat quad and a path. These pin
//! the ones that reach the fragment shaders by a different route - a quad that
//! takes the bordered/rounded branch instead of the flat fast path, an
//! underline, a glyph, a shadow, an image - plus the arithmetic the mask itself
//! has to survive when a caller hands it a radius larger than the mask.
//!
//! Each of those is a pipeline of its own with its own copy of the mask, so
//! one of them can lose the mask without any of the others noticing.
//!
//! Every test here paints white on the harness' black background, so "painted"
//! and "clipped away" are the only two answers a pixel can give.

#![cfg(target_os = "macos")]

mod harness;

use std::sync::Arc;

use gpui::{
    App, BorderStyle, Bounds, BoxShadow, ContentMask, Corners, Pixels, Point, RenderImage,
    TextAlign, TextRun, UnderlineStyle, Window, font, px, quad, size, white,
};
use harness::{WHITE, at, rect, render_inside_content_mask};
use image::{Frame, RgbaImage};

/// The window every test in this file renders, in logical pixels.
const WINDOW: f32 = 200.;

/// A 100x100 mask in the middle of the window, with 30px corners: the same
/// shape `content_mask.rs` uses, so the points below are the same points.
const MASK_LEFT: f32 = 50.;
const MASK_SIDE: f32 = 100.;
const MASK_RADIUS: f32 = 30.;

fn window() -> gpui::Size<Pixels> {
    size(px(WINDOW), px(WINDOW))
}

fn square_mask(corner_radius: f32) -> ContentMask<Pixels> {
    ContentMask {
        bounds: rect(MASK_LEFT, MASK_LEFT, MASK_SIDE, MASK_SIDE),
        corner_radii: Corners::all(px(corner_radius)),
    }
}

/// Points comfortably inside the 100x100 mask, away from every corner arc.
fn well_inside_the_mask() -> Vec<Point<Pixels>> {
    vec![
        at(100., 100.),
        at(80., 80.),
        at(100., 55.),
        at(55., 100.),
        at(145., 100.),
        at(100., 145.),
    ]
}

/// Points inside the mask's rectangle but 35.4px from the nearest arc centre,
/// so a 30px corner has to drop them.
fn cut_off_by_a_30px_corner() -> Vec<Point<Pixels>> {
    vec![at(55., 55.), at(145., 55.), at(55., 145.), at(145., 145.)]
}

#[test]
fn a_bordered_quad_is_clipped_by_a_rounded_mask() {
    // A border makes `quad_fragment` skip its flat fast path, which used to
    // mean the interior of the quad returned before the mask was applied.
    let frame = render_inside_content_mask(window(), square_mask(MASK_RADIUS), |bounds, w, _| {
        w.paint_quad(quad(
            bounds,
            Corners::all(px(0.)),
            white(),
            px(4.),
            white(),
            BorderStyle::Solid,
        ));
    });

    frame.assert_all_painted(
        &well_inside_the_mask(),
        WHITE,
        "a bordered quad inside a rounded mask",
    );
    frame.assert_all_clipped_away(
        &cut_off_by_a_30px_corner(),
        "a bordered quad in the corner a rounded mask cuts away",
    );
}

#[test]
fn a_quad_with_its_own_corner_radii_is_clipped_by_a_rounded_mask() {
    // Its own radii take the quad down the same non-flat branch a border does,
    // while the radii themselves are far away at the window's corners: what is
    // under test is the mask, not the quad's own shape.
    let frame = render_inside_content_mask(window(), square_mask(MASK_RADIUS), |bounds, w, _| {
        w.paint_quad(quad(
            bounds,
            Corners::all(px(8.)),
            white(),
            px(0.),
            white(),
            BorderStyle::Solid,
        ));
    });

    frame.assert_all_painted(
        &well_inside_the_mask(),
        WHITE,
        "a rounded quad inside a rounded mask",
    );
    frame.assert_all_clipped_away(
        &cut_off_by_a_30px_corner(),
        "a rounded quad in the corner a rounded mask cuts away",
    );
}

#[test]
fn a_dashed_border_quad_is_clipped_by_a_rounded_mask() {
    // The dashed branch is the longest way through `quad_fragment`; it has to
    // come out of the far end still masked.
    let frame = render_inside_content_mask(window(), square_mask(MASK_RADIUS), |bounds, w, _| {
        w.paint_quad(quad(
            bounds,
            Corners::all(px(0.)),
            white(),
            px(6.),
            white(),
            BorderStyle::Dashed,
        ));
    });

    frame.assert_all_painted(
        &well_inside_the_mask(),
        WHITE,
        "a dashed-border quad inside a rounded mask",
    );
    frame.assert_all_clipped_away(
        &cut_off_by_a_30px_corner(),
        "a dashed-border quad in the corner a rounded mask cuts away",
    );
}

/// A wide, short mask carrying radii far larger than it can hold. `intersect`
/// produces exactly this - a radius kept from a tall parent, landing on a short
/// result - so the shader has to clamp rather than trust the caller.
fn overlarge_radius_mask() -> ContentMask<Pixels> {
    ContentMask {
        bounds: rect(50., 90., 100., 20.),
        corner_radii: Corners::all(px(40.)),
    }
}

#[test]
fn a_mask_radius_larger_than_the_mask_is_clamped_to_half_its_shortest_side() {
    // 40px corners on a 20px-tall mask clamp to 10px, which makes the mask a
    // 100x20 pill. Unclamped, the four corner arcs overlap and eat most of the
    // interior, leaving a notch on each midline where the quadrants meet.
    let frame = render_inside_content_mask(window(), overlarge_radius_mask(), |bounds, w, _| {
        w.paint_quad(gpui::fill(bounds, white()));
    });

    // The full width of the pill's widest line, straight across the midline
    // the quadrant split runs along.
    frame.assert_all_painted(
        &[
            at(52., 100.),
            at(60., 100.),
            at(75., 100.),
            at(100., 100.),
            at(125., 100.),
            at(140., 100.),
            at(148., 100.),
        ],
        WHITE,
        "the midline of a mask whose radii are larger than it is",
    );

    // Crossing the midline vertically, near the left end: a notch would show
    // up as a gap in this column.
    frame.assert_all_painted(
        &[
            at(55., 96.),
            at(55., 98.),
            at(55., 100.),
            at(55., 102.),
            at(55., 104.),
        ],
        WHITE,
        "a column crossing the midline of an over-rounded mask",
    );

    // One pixel inside the straight top and bottom edges, away from the
    // clamped 10px corners.
    frame.assert_all_painted(
        &[
            at(70., 91.),
            at(100., 91.),
            at(130., 91.),
            at(70., 109.),
            at(100., 109.),
            at(130., 109.),
        ],
        WHITE,
        "the straight edges of an over-rounded mask",
    );

    // The clamped 10px corners are still corners: these four points are
    // 12.7px from the nearest arc centre.
    frame.assert_all_clipped_away(
        &[at(51., 91.), at(149., 91.), at(51., 109.), at(149., 109.)],
        "the clamped corner of an over-rounded mask",
    );

    frame.assert_all_clipped_away(
        &[at(100., 85.), at(100., 115.), at(40., 100.), at(160., 100.)],
        "outside an over-rounded mask",
    );
}

/// A straight white underline lying across the whole window, thick enough to
/// cover the mask's top-left corner arc.
fn paint_underline(bounds: Bounds<Pixels>, window: &mut Window, wavy: bool, thickness: f32) {
    window.paint_underline(
        at(0., 45.),
        bounds.size.width,
        &UnderlineStyle {
            thickness: px(thickness),
            color: Some(white()),
            wavy,
        },
    );
}

#[test]
fn a_straight_underline_is_clipped_by_a_rounded_mask() {
    // Underlined links inside a rounded bubble are the everyday case: the
    // underline reaches the shader through its own pipeline, which has to
    // apply the mask like every other one.
    let frame = render_inside_content_mask(window(), square_mask(MASK_RADIUS), |bounds, w, _| {
        paint_underline(bounds, w, false, 40.);
    });

    // The underline covers y 45..85, so these are inside both it and the mask.
    frame.assert_all_painted(
        &[at(100., 55.), at(100., 80.), at(60., 70.), at(140., 70.)],
        WHITE,
        "an underline inside a rounded mask",
    );
    frame.assert_all_clipped_away(
        &[at(55., 55.), at(145., 55.)],
        "an underline in the corner a rounded mask cuts away",
    );
    frame.assert_all_clipped_away(
        &[at(30., 70.), at(170., 70.)],
        "an underline outside the mask's rectangle",
    );
}

#[test]
fn a_wavy_underline_is_clipped_by_a_rounded_mask() {
    // The wavy branch of the underline shader computes its own alpha and used
    // to return it untouched. Where the wave lands is font-free arithmetic but
    // fiddly, so the rectangular mask renders the same wave and says which
    // pixels it reached: the rounded mask then has to have reached none of
    // them in the corner.
    let corner = rect(MASK_LEFT, MASK_LEFT, 6., 6.);
    let corner_points = 6 * 6;

    let rectangular = render_inside_content_mask(window(), square_mask(0.), |bounds, w, _| {
        paint_underline(bounds, w, true, 10.);
    });
    // The wave covers 33 of these 36 points. A control of "more than nothing"
    // would be satisfied by a single antialiased pixel, which would make the
    // rounded case below a test of almost no area at all.
    assert!(
        rectangular.painted_points(corner) >= corner_points / 2,
        "the wave covered only {} of the corner's {corner_points} points, so the rounded \
         case below would prove almost nothing",
        rectangular.painted_points(corner)
    );

    let rounded = render_inside_content_mask(window(), square_mask(MASK_RADIUS), |bounds, w, _| {
        paint_underline(bounds, w, true, 10.);
    });
    // And the wave still has to be there at all: without this the test passes
    // if the wavy branch stops painting anything.
    rounded.assert_region_painted(
        rect(70., 46., 40., 20.),
        "a wavy underline along the middle of a rounded mask",
    );
    rounded.assert_region_clipped_away(
        corner,
        "a wavy underline in the corner a rounded mask cuts away",
    );
}

/// Paints enough large text to cover the middle of the window several times
/// over, in white.
fn paint_a_block_of_text(window: &mut Window, cx: &mut App) {
    let run = TextRun {
        len: "████████████".len(),
        font: font("Menlo"),
        color: white(),
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    let line = window
        .text_system()
        .shape_line("████████████".into(), px(30.), &[run], None);
    for row in 0..7 {
        line.paint(
            at(20., 30. + row as f32 * 22.),
            px(22.),
            TextAlign::Left,
            None,
            window,
            cx,
        )
        .expect("failed to paint the text");
    }
}

#[test]
fn glyphs_are_clipped_by_a_rounded_mask() {
    // Glyphs are monochrome sprites, and the mask they are clipped by now
    // arrives as flat varyings rather than a per-fragment struct load. Which
    // pixels a font covers is not this test's business, so the rectangular
    // mask establishes that the corner was covered at all and the rounded one
    // has to have cut it away.
    let corner = rect(MASK_LEFT, MASK_LEFT, 6., 6.);

    let rectangular = render_inside_content_mask(window(), square_mask(0.), |_, w, cx| {
        paint_a_block_of_text(w, cx);
    });
    rectangular.assert_region_painted(
        corner,
        "the glyphs a rectangular mask leaves in the corner under test",
    );

    let rounded = render_inside_content_mask(window(), square_mask(MASK_RADIUS), |_, w, cx| {
        paint_a_block_of_text(w, cx);
    });
    rounded.assert_region_painted(
        rect(80., 80., 40., 40.),
        "the glyphs in the middle of a rounded mask",
    );
    rounded.assert_region_clipped_away(corner, "a glyph in the corner a rounded mask cuts away");
}

#[test]
fn a_shadow_is_clipped_by_a_rounded_mask() {
    // A drop shadow reaches the shader through a pipeline of its own, and the
    // everyday case for a rounded mask - a bubble - is full of them.
    let frame = render_inside_content_mask(window(), square_mask(MASK_RADIUS), |bounds, w, _| {
        w.paint_drop_shadows(
            bounds,
            Corners::all(px(0.)),
            &[BoxShadow {
                color: white(),
                offset: Point::default(),
                blur_radius: px(8.),
                spread_radius: px(0.),
                inset: false,
            }],
        );
    });

    // A blurred shadow is only opaque well away from its own edge, and its own
    // edge is the window's, so the points inside the mask are all far from it.
    frame.assert_all_painted(
        &well_inside_the_mask(),
        WHITE,
        "a shadow inside a rounded mask",
    );
    frame.assert_all_clipped_away(
        &cut_off_by_a_30px_corner(),
        "a shadow in the corner a rounded mask cuts away",
    );
}

/// An opaque white image, `side` pixels square. The renderer reads the bytes as
/// BGRA, which for white is the same four bytes either way.
fn white_image(side: u32) -> Arc<RenderImage> {
    let pixels = RgbaImage::from_pixel(side, side, image::Rgba([255, 255, 255, 255]));
    Arc::new(RenderImage::new(vec![Frame::new(pixels)]))
}

#[test]
fn an_image_is_clipped_by_a_rounded_mask() {
    // Images are polychrome sprites: their own pipeline, their own copy of the
    // mask, and the primitive an email body is mostly made of.
    let frame = render_inside_content_mask(window(), square_mask(MASK_RADIUS), |bounds, w, _| {
        w.paint_image(
            bounds,
            bounds,
            Corners::all(px(0.)),
            white_image(64),
            0,
            false,
        )
        .expect("failed to paint the image");
    });

    frame.assert_all_painted(
        &well_inside_the_mask(),
        WHITE,
        "an image inside a rounded mask",
    );
    frame.assert_all_clipped_away(
        &cut_off_by_a_30px_corner(),
        "an image in the corner a rounded mask cuts away",
    );
    frame.assert_all_clipped_away(
        &[at(25., 25.), at(175., 175.), at(100., 20.), at(20., 100.)],
        "an image outside the mask's rectangle",
    );
}
