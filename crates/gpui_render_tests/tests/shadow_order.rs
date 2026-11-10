//! That a blurred shadow is ordered by what it paints, not by the rectangle
//! its falloff is measured from.
//!
//! A drop shadow's record holds the rect the fragment shader takes its signed
//! distance from; the vertex shader draws three blur radii wider than that, and
//! the gaussian tail out there is what you see. If the scene's bounds tree is
//! told only about the inner rect it can believe two shadows are disjoint when
//! they visibly overlap, and hand them draw orders that put the wrong one on
//! top.

#![cfg(target_os = "macos")]

mod harness;

use gpui::{BoxShadow, Corners, black, point, px, size, white};
use harness::{WHITE, at, rect, render_frame};

fn window() -> gpui::Size<gpui::Pixels> {
    size(px(200.), px(100.))
}

fn shadow(color: gpui::Hsla, blur_radius: f32) -> BoxShadow {
    BoxShadow {
        color,
        offset: point(px(0.), px(0.)),
        blur_radius: px(blur_radius),
        spread_radius: px(0.),
        inset: false,
    }
}

/// Paints, in this order:
///
/// - a white quad down the left edge,
/// - an unblurred opaque black shadow overlapping it, which therefore has to be
///   drawn after the quad and so takes a draw order above it,
/// - a blurred white shadow whose own rect is clear of both, but whose tail
///   reaches back across the black one.
///
/// The last one is painted last, so it belongs on top. Ordered by the inner
/// rects alone the blurred shadow looks disjoint from everything, takes the
/// lowest order of the three and ends up underneath the opaque black shadow,
/// which hides it completely.
fn frame() -> harness::RenderedFrame {
    render_frame(window(), |_, window, _| {
        window.paint_quad(gpui::fill(rect(0., 0., 30., 100.), white()));
        window.paint_drop_shadows(
            rect(20., 20., 75., 60.),
            Corners::default(),
            &[shadow(black(), 0.)],
        );
        window.paint_drop_shadows(
            rect(100., 20., 60., 60.),
            Corners::default(),
            &[shadow(white(), 20.)],
        );
    })
}

#[test]
fn a_blurred_shadows_tail_draws_over_a_shadow_painted_before_it() {
    let frame = frame();

    // 7px outside the blurred shadow's rect, and well inside the opaque black
    // one. Whatever the exact falloff is worth here, it is only visible if the
    // blurred shadow was drawn after the black one.
    assert!(
        !frame.is_background(at(93., 50.)),
        "the blurred shadow's tail was painted before the opaque shadow it \
         overlaps, so the black one covered it: found {:?} at (93, 50)\n  \
         frame written to {}",
        frame.color_at(at(93., 50.)),
        frame
            .save("a blurred shadows tail draws over a shadow painted before it")
            .display(),
    );
    frame.assert_region_painted(
        rect(85., 40., 9., 20.),
        "the blurred shadow's tail where it crosses the opaque black shadow",
    );
}

#[test]
fn the_opaque_shadow_really_would_have_hidden_it() {
    // The control for the test above: the black shadow is opaque, so where
    // nothing was drawn on top of it the frame is exactly the background. If it
    // were translucent, "something is visible at 93" would prove nothing.
    let frame = frame();

    assert_eq!(
        frame.color_at(at(35., 50.)),
        harness::BACKGROUND,
        "inside the opaque black shadow, clear of the blurred one's tail"
    );
    frame.assert_painted(at(10., 50.), WHITE, "the quad, clear of both shadows");
}
