//! What a content mask actually does to pixels.
//!
//! `ContentMask`'s own unit tests cover the arithmetic of `intersect` and
//! `scale`. These cover the half nobody can see from Rust: that the mask
//! reaches the shaders, and that a rounded one cuts the corners off both a quad
//! and a path.

#![cfg(target_os = "macos")]

mod harness;

use gpui::{ContentMask, Corners, Path, px, size, white};
use harness::{WHITE, at, rect, render_frame, render_inside_content_mask};

/// The window every test in this file renders, in logical pixels.
const WINDOW: f32 = 200.;

/// The mask: a 100x100 square in the middle of the window.
const MASK_LEFT: f32 = 50.;
const MASK_RIGHT: f32 = 150.;
const MASK_RADIUS: f32 = 30.;

fn mask(corner_radius: f32) -> ContentMask<gpui::Pixels> {
    ContentMask {
        bounds: rect(
            MASK_LEFT,
            MASK_LEFT,
            MASK_RIGHT - MASK_LEFT,
            MASK_RIGHT - MASK_LEFT,
        ),
        corner_radii: Corners::all(px(corner_radius)),
    }
}

fn window() -> gpui::Size<gpui::Pixels> {
    size(px(WINDOW), px(WINDOW))
}

/// Points comfortably inside the mask, away from every corner arc.
fn well_inside_the_mask() -> Vec<gpui::Point<gpui::Pixels>> {
    vec![
        at(100., 100.),
        at(100., 55.),
        at(100., 145.),
        at(55., 100.),
        at(145., 100.),
        at(80., 80.),
    ]
}

/// Points inside the mask's rectangle but outside a 30px corner arc: the
/// nearest arc centre is 35.4px away, so a rounded mask must drop them.
fn cut_off_by_a_30px_corner() -> Vec<gpui::Point<gpui::Pixels>> {
    vec![at(55., 55.), at(145., 55.), at(55., 145.), at(145., 145.)]
}

/// Points outside the mask's rectangle altogether.
fn outside_the_mask_rectangle() -> Vec<gpui::Point<gpui::Pixels>> {
    vec![at(25., 25.), at(175., 175.), at(100., 20.), at(20., 100.)]
}

fn paint_full_window_quad(bounds: gpui::Bounds<gpui::Pixels>, window: &mut gpui::Window) {
    window.paint_quad(gpui::fill(bounds, white()));
}

fn paint_full_window_path(bounds: gpui::Bounds<gpui::Pixels>, window: &mut gpui::Window) {
    let mut path = Path::new(bounds.origin);
    path.line_to(bounds.top_right());
    path.line_to(bounds.bottom_right());
    path.line_to(bounds.bottom_left());
    window.paint_path(path, white());
}

#[test]
fn a_rectangular_mask_clips_a_quad_to_its_rectangle_and_nothing_more() {
    let frame = render_inside_content_mask(window(), mask(0.), |bounds, window, _| {
        paint_full_window_quad(bounds, window)
    });

    frame.assert_all_painted(
        &well_inside_the_mask(),
        WHITE,
        "a quad inside a rectangular mask",
    );
    frame.assert_all_clipped_away(
        &outside_the_mask_rectangle(),
        "a quad outside a rectangular mask",
    );

    // The corners a rounded mask would cut are square here, which is what makes
    // the rounded-mask test below a test of the radii rather than of the
    // rectangle.
    frame.assert_all_painted(
        &cut_off_by_a_30px_corner(),
        WHITE,
        "the square corners of a rectangular mask",
    );

    // The very edge of the mask, a pixel either side.
    frame.assert_painted(at(50.5, 100.), WHITE, "just inside the mask's left edge");
    frame.assert_clipped_away(at(49., 100.), "just outside the mask's left edge");
    frame.assert_painted(at(149.5, 100.), WHITE, "just inside the mask's right edge");
    frame.assert_clipped_away(at(151., 100.), "just outside the mask's right edge");
}

#[test]
fn a_rounded_mask_cuts_the_corners_off_a_quad() {
    let frame = render_inside_content_mask(window(), mask(MASK_RADIUS), |bounds, window, _| {
        paint_full_window_quad(bounds, window)
    });

    frame.assert_all_painted(
        &well_inside_the_mask(),
        WHITE,
        "a quad inside a rounded mask",
    );
    frame.assert_all_clipped_away(
        &cut_off_by_a_30px_corner(),
        "a quad in the corner a rounded mask cuts away",
    );
    frame.assert_all_clipped_away(
        &outside_the_mask_rectangle(),
        "a quad outside a rounded mask",
    );
}

#[test]
fn a_rounded_mask_cuts_the_corners_off_a_path() {
    // Paths take a different route through the renderer than quads: they are
    // rasterized into an intermediate texture first, so the mask has to be
    // applied there. This is the route the email renderer leans on.
    let frame = render_inside_content_mask(window(), mask(MASK_RADIUS), |bounds, window, _| {
        paint_full_window_path(bounds, window)
    });

    frame.assert_all_painted(
        &well_inside_the_mask(),
        WHITE,
        "a path inside a rounded mask",
    );
    frame.assert_all_clipped_away(
        &cut_off_by_a_30px_corner(),
        "a path in the corner a rounded mask cuts away",
    );
    frame.assert_all_clipped_away(
        &outside_the_mask_rectangle(),
        "a path outside a rounded mask",
    );
}

#[test]
fn a_rectangular_mask_clips_a_path_to_its_rectangle_and_nothing_more() {
    let frame = render_inside_content_mask(window(), mask(0.), |bounds, window, _| {
        paint_full_window_path(bounds, window)
    });

    frame.assert_all_painted(
        &well_inside_the_mask(),
        WHITE,
        "a path inside a rectangular mask",
    );
    frame.assert_all_painted(
        &cut_off_by_a_30px_corner(),
        WHITE,
        "the square corners of a rectangular mask",
    );
    frame.assert_all_clipped_away(
        &outside_the_mask_rectangle(),
        "a path outside a rectangular mask",
    );
}

#[test]
fn the_corner_a_rounded_mask_cuts_is_a_smooth_arc_not_a_notch() {
    // Walking the diagonal out of the top-left corner: inside the arc it is
    // painted, outside it is not, and the crossing happens once. A notch or an
    // inverted radius would break the ordering.
    let frame = render_inside_content_mask(window(), mask(MASK_RADIUS), |bounds, window, _| {
        paint_full_window_quad(bounds, window)
    });

    // The arc centre is (80, 80) with radius 30, so the crossing on the
    // diagonal is at 80 - 30/sqrt(2) = 58.8.
    frame.assert_clipped_away(at(56., 56.), "outside the top-left arc");
    frame.assert_painted(at(62., 62.), WHITE, "inside the top-left arc");
    frame.assert_clipped_away(at(50., 50.), "the mask's own top-left corner point");
}

/// A mask `side` logical pixels square with its top-left corner at `origin`.
fn mask_at(origin: (f32, f32), side: f32, corner_radius: f32) -> ContentMask<gpui::Pixels> {
    ContentMask {
        bounds: rect(origin.0, origin.1, side, side),
        corner_radii: Corners::all(px(corner_radius)),
    }
}

/// The four points 5px in from each corner of a `side`-square mask at `origin`.
/// With a 25px radius on a 70px square they are 28.3px from the nearest arc
/// centre, so a rounded mask drops all four and a rectangular one keeps them.
fn corners_of(origin: (f32, f32), side: f32) -> Vec<gpui::Point<gpui::Pixels>> {
    let (x, y) = origin;
    vec![
        at(x + 5., y + 5.),
        at(x + side - 5., y + 5.),
        at(x + 5., y + side - 5.),
        at(x + side - 5., y + side - 5.),
    ]
}

/// Points well inside a `side`-square mask at `origin`, clear of every arc.
fn inside_of(origin: (f32, f32), side: f32) -> Vec<gpui::Point<gpui::Pixels>> {
    let (x, y) = origin;
    let half = side / 2.;
    vec![
        at(x + half, y + half),
        at(x + half, y + 4.),
        at(x + 4., y + half),
        at(x + side - 4., y + half),
        at(x + half, y + side - 4.),
    ]
}

#[test]
fn each_of_several_paths_in_one_frame_is_clipped_by_its_own_mask() {
    // The Metal renderer packs every path in a frame into one vertex buffer and
    // hands the masks alongside it as a second buffer indexed by `path_id`.
    // Nothing else in this crate paints more than one path per frame, so
    // nothing else can tell that indexing apart from "always use the first
    // mask" - or from an off-by-one.
    //
    // The three masks are disjoint, so a path drawn against the wrong one
    // vanishes entirely rather than being subtly wrong. Their radii differ so
    // that the *shape* is per-path too, not just the rectangle.
    const SIDE: f32 = 70.;
    let first = (10., 10.);
    let second = (110., 10.);
    let third = (10., 110.);

    let frame = render_frame(size(px(190.), px(190.)), move |bounds, window, _| {
        for (origin, radius) in [(first, 25.), (second, 25.), (third, 0.)] {
            window.with_content_mask(Some(mask_at(origin, SIDE, radius)), |window| {
                paint_full_window_path(bounds, window);
            });
        }
    });

    frame.assert_all_painted(
        &inside_of(first, SIDE),
        WHITE,
        "the first path, inside the first mask",
    );
    frame.assert_all_clipped_away(
        &corners_of(first, SIDE),
        "the first path, in the corner its own rounded mask cuts away",
    );

    frame.assert_all_painted(
        &inside_of(second, SIDE),
        WHITE,
        "the second path, inside the second mask - blank here means every path \
         was clipped by the first path's mask",
    );
    frame.assert_all_clipped_away(
        &corners_of(second, SIDE),
        "the second path, in the corner its own rounded mask cuts away - painted \
         here means it was clipped by a mask that is not its own",
    );

    // The third mask is square, so its corners must survive: a path clipped by
    // one of the rounded masks instead would lose them.
    frame.assert_all_painted(
        &inside_of(third, SIDE),
        WHITE,
        "the third path, inside the third mask",
    );
    frame.assert_all_painted(
        &corners_of(third, SIDE),
        WHITE,
        "the square corners of the third path's own rectangular mask",
    );

    // And nothing leaked into the quarter no mask covers.
    frame.assert_region_clipped_away(
        rect(110., 110., 70., 70.),
        "the quarter of the window no mask covers",
    );
}
