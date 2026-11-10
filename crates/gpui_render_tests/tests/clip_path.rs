//! What an arbitrary clip path actually does to pixels.
//!
//! `ClipPath`'s own unit tests cover the arithmetic that builds one. These
//! cover the half nobody can see from Rust: that the path reaches the mask
//! rasterizer, and that the shapes a [`ContentMask`](gpui::ContentMask)
//! structurally cannot express - elliptical corners, a non-convex outline, a
//! self-overlapping one, a hole - come out right.
//!
//! Every assertion here is paired with one that fails if clipping is a no-op:
//! the harness paints white over the whole window, so without a working mask
//! every point is white and every "clipped away" assertion fails.

#![cfg(target_os = "macos")]

mod harness;

use gpui::{Bounds, ClipPath, ContentMask, Corners, FillRule, Pixels, Size, px, size, white};
use harness::{TRANSPARENT, WHITE, at, rect, render_frame, render_frame_on_transparent};

/// The window every test in this file renders, in logical pixels.
const WINDOW: f32 = 200.;

fn window() -> Size<Pixels> {
    size(px(WINDOW), px(WINDOW))
}

/// Fills the whole window with white inside `clip`, so any pixel that is not
/// white is a pixel the clip cut away.
fn render_inside_clip_path(clip: ClipPath<Pixels>) -> harness::RenderedFrame {
    render_frame(window(), move |bounds, window, _| {
        window.with_clip_path(&clip, |window| {
            window.paint_quad(gpui::fill(bounds, white()));
        });
    })
}

/// The same picture with no clip at all: every assertion that something was cut
/// away has to fail against this, or it is not testing the clip.
fn render_with_no_clip() -> harness::RenderedFrame {
    render_frame(window(), |bounds, window, _| {
        window.paint_quad(gpui::fill(bounds, white()));
    })
}

fn corner(width: f32, height: f32) -> Size<Pixels> {
    size(px(width), px(height))
}

/// A right triangle with its hypotenuse across the middle of the window: a
/// point is inside when `50 <= y < 120` and `x <= 50 + (y - 50) * 100 / 70`.
///
/// A triangle rather than a rectangle wherever the shape itself is not the
/// point: `with_clip_path` narrows the content mask to the path's bounding box,
/// so a rectangular clip path is reproduced by the mask alone and a test built
/// on one passes with no mask rasterizer at all.
fn diagonal_clip() -> ClipPath<Pixels> {
    ClipPath::builder()
        .move_to(at(50., 50.))
        .line_to(at(150., 120.))
        .line_to(at(50., 120.))
        .close()
        .build()
}

#[test]
fn a_clip_path_that_encloses_no_area_lets_nothing_through() {
    // A path can name points without enclosing any area: a lone move, or a
    // single line. There is no contour to rasterize, and "no mask" is also how
    // a clip that fell back to its bounding rectangle is spelled - so unless
    // the two are told apart, a degenerate path shows everything inside its box
    // instead of hiding it, which is the opposite of what a clip is for.
    for (what, path) in [
        (
            "a line",
            ClipPath::builder()
                .move_to(at(50., 50.))
                .line_to(at(150., 150.))
                .build(),
        ),
        (
            "a lone move",
            ClipPath::builder().move_to(at(50., 50.)).build(),
        ),
        ("an empty path", ClipPath::builder().build()),
    ] {
        let frame = render_inside_clip_path(path);
        frame.assert_region_clipped_away(
            rect(0., 0., WINDOW, WINDOW),
            &format!("the whole window, inside {what} that encloses no area"),
        );
    }

    // The control: the same two points, and a path that does enclose area.
    let unclipped = render_with_no_clip();
    unclipped.assert_all_painted(
        &[at(60., 60.), at(140., 140.)],
        WHITE,
        "the same points with no clip path, which is what makes the above a test",
    );
}

#[test]
fn a_clip_path_cuts_the_alpha_channel_and_not_just_the_colour() {
    // Every other test here renders over opaque black, where a pixel reads back
    // with an alpha of 255 whatever the scene did, so none of them says
    // anything about what a clip does to alpha. Over nothing at all, a clipped
    // pixel has to *be* nothing, and a partly covered one has to come back
    // premultiplied.
    let frame = render_frame_on_transparent(window(), |bounds, window, _| {
        window.with_clip_path(&diagonal_clip(), |window| {
            window.paint_quad(gpui::fill(bounds, white()));
        });
    });

    frame.assert_painted(at(100., 110.), WHITE, "inside a clip path over nothing");
    for outside in [at(100., 60.), at(25., 25.), at(175., 100.)] {
        assert_eq!(
            frame.color_at(outside),
            TRANSPARENT,
            "a clip path has to leave {outside:?} at zero alpha, not at black"
        );
    }

    // At x = 100 the hypotenuse sits at y = 85, so the pixels it crosses are
    // partly covered: premultiplied white is (a, a, a, a).
    let mut partial = Vec::new();
    let mut y = 82.;
    while y <= 88. {
        let [red, green, blue, alpha] = frame.color_at(at(100., y));
        if alpha > 8 && alpha < 247 {
            partial.push((y, alpha));
            assert_eq!(
                [red, green, blue],
                [alpha, alpha, alpha],
                "a partly covered pixel of a clipped white quad has to come back premultiplied"
            );
        }
        y += 0.5;
    }
    assert!(
        !partial.is_empty(),
        "no pixel along the clip's edge came back partly covered over a transparent target"
    );
}

/// A 100x100 square whose top-left corner is a 40x10 ellipse quadrant. Its arc
/// centre is (90, 60), so a point is inside the corner exactly when
/// `((x - 90) / 40)^2 + ((y - 60) / 10)^2 <= 1`.
fn elliptical_corner_clip() -> ClipPath<Pixels> {
    ClipPath::rounded_rect(
        rect(50., 50., 100., 100.),
        Corners {
            top_left: corner(40., 10.),
            ..Default::default()
        },
    )
}

#[test]
fn an_elliptical_corner_cuts_where_the_ellipse_says_and_not_where_a_circle_would() {
    let frame = render_inside_clip_path(elliptical_corner_clip());

    // Outside the ellipse: 1.54 in normalised terms, about seven pixels beyond
    // the arc along that ray.
    frame.assert_all_clipped_away(
        &[at(52., 52.), at(55., 51.), at(51., 55.)],
        "the corner a 40x10 elliptical radius cuts away",
    );

    // Inside the ellipse at 0.81 - but 47.4 from (90, 90), so a *circular* 40px
    // radius, which is all a `Corners<Pixels>` content mask can hold, would
    // have cut it. This point is the whole reason `ClipPath` exists.
    frame.assert_all_painted(
        &[at(55., 58.), at(60., 57.)],
        WHITE,
        "a point a circular radius would have cut and an elliptical one keeps",
    );

    // The other three corners were given no radius, so they stay square.
    frame.assert_all_painted(
        &[at(145., 55.), at(55., 145.), at(145., 145.)],
        WHITE,
        "the square corners of a clip path with one rounded corner",
    );
    frame.assert_all_clipped_away(
        &[at(100., 25.), at(25., 100.)],
        "outside a clip path with one elliptical corner",
    );
}

#[test]
fn a_circular_content_mask_really_would_have_cut_that_point() {
    // The control for the test above, and the reason it cannot be written with
    // a `ContentMask`: the same 40px radius, expressed the only way a content
    // mask can, takes the point the ellipse kept.
    let frame = render_frame(window(), |bounds, window, _| {
        window.with_content_mask(
            Some(ContentMask {
                bounds: rect(50., 50., 100., 100.),
                corner_radii: Corners::all(px(40.)),
            }),
            |window| window.paint_quad(gpui::fill(bounds, white())),
        );
    });

    frame.assert_clipped_away(
        at(55., 58.),
        "a circular 40px corner radius, which cuts what the 40x10 ellipse keeps",
    );
}

/// A rectangle with a notch bitten out of its right-hand side: the outline goes
/// clockwise from the top-left and doubles back through the middle.
///
/// A triangle fan anchored at (50, 50) covers the notch - the triangle
/// (50,50), (90,120), (150,120) contains (120, 100) - so anything that fills
/// this by fanning and blending paints the notch. Stencil counting does not.
fn notched_clip() -> ClipPath<Pixels> {
    ClipPath::builder()
        .move_to(at(50., 50.))
        .line_to(at(150., 50.))
        .line_to(at(150., 80.))
        .line_to(at(90., 80.))
        .line_to(at(90., 120.))
        .line_to(at(150., 120.))
        .line_to(at(150., 150.))
        .line_to(at(50., 150.))
        .close()
        .build()
}

#[test]
fn a_non_convex_clip_path_excludes_its_notch() {
    let frame = render_inside_clip_path(notched_clip());

    frame.assert_all_painted(
        &[
            at(70., 100.),
            at(60., 60.),
            at(120., 60.),
            at(120., 140.),
            at(140., 70.),
        ],
        WHITE,
        "the arms of a notched clip path",
    );
    frame.assert_region_clipped_away(
        rect(95., 85., 50., 30.),
        "the notch bitten out of a non-convex clip path",
    );
    frame.assert_all_clipped_away(
        &[at(120., 100.), at(110., 100.), at(145., 100.)],
        "inside the notch of a non-convex clip path",
    );
}

/// A pentagram: five points, each edge skipping a vertex, so the outline
/// crosses itself and the middle is wound twice.
fn pentagram(fill_rule: FillRule) -> ClipPath<Pixels> {
    let (centre, radius) = (at(100., 100.), 45.);
    let vertex = |index: usize| {
        let angle = (-90. + index as f32 * 144.).to_radians();
        at(
            f32::from(centre.x) + radius * angle.cos(),
            f32::from(centre.y) + radius * angle.sin(),
        )
    };
    let mut builder = ClipPath::builder().fill_rule(fill_rule).move_to(vertex(0));
    for index in 1..5 {
        builder = builder.line_to(vertex(index));
    }
    builder.close().build()
}

#[test]
fn a_self_intersecting_star_fills_its_middle_under_nonzero() {
    let frame = render_inside_clip_path(pentagram(FillRule::NonZero));

    // The middle pentagon is wound twice, which the nonzero rule counts as
    // inside. Its inradius is about 14 logical pixels.
    frame.assert_all_painted(
        &[
            at(100., 100.),
            at(94., 100.),
            at(106., 100.),
            at(100., 105.),
        ],
        WHITE,
        "the doubly-wound middle of a nonzero star",
    );
    // 56 pixels from the centre, so outside the star's own circumcircle.
    frame.assert_all_clipped_away(
        &[at(60., 60.), at(140., 60.), at(60., 140.), at(140., 140.)],
        "between the arms of a star clip path",
    );
    frame.assert_region_painted(rect(96., 58., 8., 8.), "the top arm of a star clip path");
}

#[test]
fn the_same_star_leaves_its_middle_empty_under_even_odd() {
    // The control for the test above: same outline, same winding, other rule.
    // Nothing but the fill rule reaching the rasterizer can produce this
    // difference.
    let frame = render_inside_clip_path(pentagram(FillRule::EvenOdd));

    frame.assert_all_clipped_away(
        &[at(100., 100.), at(94., 100.), at(106., 100.)],
        "the middle of an even-odd star, which the two crossings cancel",
    );
    frame.assert_region_painted(
        rect(96., 58., 8., 8.),
        "the top arm of an even-odd star, which is wound once",
    );
}

/// Two rectangles wound the same way, the smaller inside the larger.
fn nested_rectangles(fill_rule: FillRule) -> ClipPath<Pixels> {
    let ring = |builder: gpui::ClipPathBuilder, bounds: Bounds<Pixels>| {
        builder
            .move_to(bounds.origin)
            .line_to(bounds.top_right())
            .line_to(bounds.bottom_right())
            .line_to(bounds.bottom_left())
            .close()
    };
    let builder = ClipPath::builder().fill_rule(fill_rule);
    let builder = ring(builder, rect(50., 50., 100., 100.));
    ring(builder, rect(80., 80., 40., 40.)).build()
}

#[test]
fn nested_contours_leave_a_hole_under_even_odd() {
    let frame = render_inside_clip_path(nested_rectangles(FillRule::EvenOdd));

    frame.assert_region_clipped_away(
        rect(85., 85., 30., 30.),
        "the hole two nested contours leave under the even-odd rule",
    );
    frame.assert_all_painted(
        &[at(60., 100.), at(140., 100.), at(100., 60.), at(100., 140.)],
        WHITE,
        "the ring around an even-odd hole",
    );
    frame.assert_all_clipped_away(
        &[at(25., 25.), at(175., 175.)],
        "outside a pair of nested contours",
    );
}

#[test]
fn the_same_nested_contours_leave_no_hole_under_nonzero() {
    // Both contours are wound the same way, so the nonzero rule counts the
    // middle twice and keeps it. Without the fill rule reaching the GPU these
    // two tests could not disagree.
    let frame = render_inside_clip_path(nested_rectangles(FillRule::NonZero));

    frame.assert_all_painted(
        &[at(100., 100.), at(90., 90.), at(110., 110.)],
        WHITE,
        "the middle two same-wound contours keep under the nonzero rule",
    );
    frame.assert_all_clipped_away(
        &[at(25., 25.), at(175., 175.)],
        "outside a pair of nested contours",
    );
}

#[test]
fn a_diagonal_clip_edge_is_antialiased() {
    // A clip that could only be applied per-pixel would leave a hard staircase
    // here. The mask is multisampled, so the pixels the edge crosses come back
    // partly covered.
    let frame = render_inside_clip_path(
        ClipPath::builder()
            .move_to(at(50., 50.))
            .line_to(at(150., 120.))
            .line_to(at(50., 120.))
            .close()
            .build(),
    );

    // At x = 100 the diagonal sits at y = 85.
    frame.assert_painted(at(100., 95.), WHITE, "below a diagonal clip edge");
    frame.assert_clipped_away(at(100., 75.), "above a diagonal clip edge");

    let mut partial = Vec::new();
    let mut y = 82.;
    while y <= 88. {
        let [red, _, _, _] = frame.color_at(at(100., y));
        if red > 8 && red < 247 {
            partial.push((y, red));
        }
        y += 0.5;
    }
    assert!(
        !partial.is_empty(),
        "no pixel along the diagonal clip edge came back partly covered, so the mask is not \
         antialiased; the column read {:?}",
        (164..=176)
            .map(|device_y| frame.color_at(at(100., device_y as f32 / 2.))[0])
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_clip_path_stops_applying_after_its_closure() {
    // The clip is a triangle, so the quad painted inside it keeps a shape the
    // content mask alone could not give it, and the quad painted after it keeps
    // its corners.
    let frame = render_frame(window(), |_, window, _| {
        window.with_clip_path(&diagonal_clip(), |window| {
            window.paint_quad(gpui::fill(rect(0., 0., WINDOW, WINDOW), white()));
        });
        window.paint_quad(gpui::fill(rect(110., 130., 60., 60.), white()));
    });

    frame.assert_painted(at(60., 110.), WHITE, "inside the clip path");
    frame.assert_clipped_away(
        at(140., 60.),
        "the part of the clipped quad the triangle cuts",
    );
    frame.assert_clipped_away(at(60., 140.), "the clipped quad beyond the clip path");
    frame.assert_all_painted(
        &[
            at(115., 135.),
            at(165., 185.),
            at(165., 135.),
            at(115., 185.),
        ],
        WHITE,
        "every corner of the quad painted after the clip path was popped",
    );
}
