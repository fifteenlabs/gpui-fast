//! Where a rounded content mask's edge actually falls, to the device pixel.
//!
//! The other files ask whether a primitive is inside or outside a mask at
//! points chosen to be nowhere near the boundary. These ask about the boundary
//! itself: that the arc sits where the caller asked for it rather than a device
//! pixel outside, that a quad and a path under one mask round their corners
//! against the same rectangle, and that a straight edge landing between two
//! device pixels is antialiased rather than snapped.
//!
//! Every mask here is deliberately off the device pixel grid. At the scale
//! factor the harness reports a device pixel is half a logical one, so a mask
//! at a quarter-pixel logical offset lands halfway between two of them.

#![cfg(target_os = "macos")]

mod harness;

use gpui::{ContentMask, Corners, Path, Pixels, px, size, white};
use harness::{BACKGROUND, SCALE_FACTOR, WHITE, at, rect, render_inside_content_mask};

/// The window every test in this file renders, in logical pixels.
const WINDOW: f32 = 200.;

/// The mask's radius, big enough that its arc is nowhere near its straight
/// edges.
const MASK_RADIUS: f32 = 30.;

fn window() -> gpui::Size<Pixels> {
    size(px(WINDOW), px(WINDOW))
}

/// A square mask inset by `inset` on all four sides of the window.
fn mask(inset: f32, corner_radius: f32) -> ContentMask<Pixels> {
    ContentMask {
        bounds: rect(inset, inset, WINDOW - 2. * inset, WINDOW - 2. * inset),
        corner_radii: Corners::all(px(corner_radius)),
    }
}

fn paint_full_window_quad(bounds: gpui::Bounds<Pixels>, window: &mut gpui::Window) {
    window.paint_quad(gpui::fill(bounds, white()));
}

fn paint_full_window_path(bounds: gpui::Bounds<Pixels>, window: &mut gpui::Window) {
    let mut path = Path::new(bounds.origin);
    path.line_to(bounds.top_right());
    path.line_to(bounds.bottom_right());
    path.line_to(bounds.bottom_left());
    window.paint_path(path, white());
}

/// A mask inset by half a device pixel: its edges land on device pixel
/// *centres*, the offset at which expanding the rectangle out to whole device
/// pixels moves each corner arc furthest.
const HALF_DEVICE_PIXEL_INSET: f32 = 50.25;

/// A point on the top-left diagonal that the mask above excludes and the same
/// mask expanded out to whole device pixels would include. It is the centre of
/// one device pixel: 43 device pixels along the diagonal from the arc's centre
/// at 60, which is 60.8 away - outside - where expanding the rectangle by half
/// a device pixel brings it to 60.1, inside.
const JUST_OUTSIDE_THE_ARC: f32 = 58.5;

/// The next device pixel inwards along the same diagonal, which the mask
/// includes outright. Between the two, nothing may be left to interpret.
const JUST_INSIDE_THE_ARC: f32 = 59.;

#[test]
fn a_rounded_masks_arc_is_where_the_caller_put_it_not_a_device_pixel_outside() {
    // The mask a primitive is clipped against used to be the caller's rectangle
    // grown outward to whole device pixels - which is right for a plain
    // rectangle, whose edge is enforced by the hardware clip, and wrong the
    // moment there are radii, because the corner arcs are computed from that
    // very rectangle and move outward with it.
    let frame = render_inside_content_mask(
        window(),
        mask(HALF_DEVICE_PIXEL_INSET, MASK_RADIUS),
        |bounds, w, _| paint_full_window_quad(bounds, w),
    );

    frame.assert_clipped_away(
        at(JUST_OUTSIDE_THE_ARC, JUST_OUTSIDE_THE_ARC),
        "a device pixel the mask's arc excludes, which a mask expanded to whole \
         device pixels would have included",
    );
    frame.assert_painted(
        at(JUST_INSIDE_THE_ARC, JUST_INSIDE_THE_ARC),
        WHITE,
        "the next device pixel inwards, which the arc includes outright",
    );
}

#[test]
fn a_quad_and_a_path_under_one_mask_round_their_corners_the_same_way() {
    // Quads and paths took the mask by different routes: a quad got the
    // rectangle expanded to whole device pixels, a path got it scaled exactly.
    // With a rectangular mask the hardware clip hid the difference. With radii
    // it does not, and a bubble's background and its content disagreed about
    // where the corner was by up to a device pixel.
    let mask = mask(HALF_DEVICE_PIXEL_INSET, MASK_RADIUS);
    let quad = render_inside_content_mask(window(), mask, |bounds, w, _| {
        paint_full_window_quad(bounds, w)
    });
    let path = render_inside_content_mask(window(), mask, |bounds, w, _| {
        paint_full_window_path(bounds, w)
    });

    // The corner under test, one device pixel at a time. Both frames are the
    // whole window painted white, so inside this box the mask is the only thing
    // that can differ between them.
    let mut compared = 0;
    let mut step = 0.;
    while step < 16. {
        let point = at(52. + step, 52. + step);
        let (quad_color, path_color) = (quad.color_at(point), path.color_at(point));
        let drift = quad_color
            .iter()
            .zip(path_color.iter())
            .map(|(q, p)| q.abs_diff(*p))
            .max()
            .unwrap_or(0);
        assert!(
            drift <= 3,
            "the quad and the path disagree by {drift} at {point:?}: the quad is \
             {quad_color:?} and the path is {path_color:?}. Frames written to {} and {}",
            quad.save("quad under a fractional rounded mask").display(),
            path.save("path under a fractional rounded mask").display(),
        );
        compared += 1;
        step += 1. / SCALE_FACTOR;
    }
    assert_eq!(compared, 32);

    // And the corner is a corner in both, or the comparison above is between
    // two blank boxes.
    for frame in [&quad, &path] {
        frame.assert_region_painted(rect(60., 60., 8., 8.), "the inside of the corner");
        frame.assert_region_clipped_away(rect(50.25, 50.25, 5., 5.), "the cut-away corner");
    }
}

/// A mask inset by a quarter of a device pixel, so that its straight edges fall
/// a quarter of a device pixel inside the pixels they cross.
const QUARTER_DEVICE_PIXEL_INSET: f32 = 50.125;

/// A logical point on the mask's left edge, far from either arc. Its device
/// pixel's centre is a quarter of a device pixel inside the mask.
fn on_the_left_edge() -> gpui::Point<Pixels> {
    at(QUARTER_DEVICE_PIXEL_INSET, 100.)
}

#[test]
fn a_rounded_mask_antialiases_the_device_pixel_its_straight_edge_crosses() {
    // Once any radius is non-zero the whole mask goes through the signed
    // distance field, straight edges included, so a device pixel the edge
    // crosses is covered in proportion. This is a quarter of a device pixel in
    // from the edge, which is three quarters covered.
    let frame = render_inside_content_mask(
        window(),
        mask(QUARTER_DEVICE_PIXEL_INSET, MASK_RADIUS),
        |bounds, w, _| paint_full_window_quad(bounds, w),
    );

    frame.assert_painted(
        on_the_left_edge(),
        [191, 191, 191, 255],
        "the device pixel a rounded mask's straight left edge crosses",
    );
}

#[test]
fn a_rectangular_mask_snaps_the_device_pixel_its_straight_edge_crosses() {
    // The same edge without radii takes the fast path: the mask is enforced by
    // the hardware clip against a rectangle grown out to whole device pixels,
    // so the pixel the edge crosses is painted whole. The pair pins the cost of
    // adding a radius, which is that this pixel changes.
    let frame = render_inside_content_mask(
        window(),
        mask(QUARTER_DEVICE_PIXEL_INSET, 0.),
        |bounds, w, _| paint_full_window_quad(bounds, w),
    );

    frame.assert_painted(
        on_the_left_edge(),
        WHITE,
        "the device pixel a rectangular mask's straight left edge crosses",
    );
    assert_eq!(
        frame.color_at(at(QUARTER_DEVICE_PIXEL_INSET - 0.5, 100.)),
        BACKGROUND,
        "the device pixel before it is outside the mask entirely"
    );
}
