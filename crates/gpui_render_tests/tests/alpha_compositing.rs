//! What the renderer's blend state does to the alpha channel.
//!
//! These are the only tests that can see alpha at all: they paint onto a
//! transparent target, so what reads back is the premultiplied RGBA the GPU
//! composited rather than the constant 255 an opaque clear colour supplies.

#![cfg(target_os = "macos")]

mod harness;

use gpui::{hsla, px, size, white};
use harness::{
    HALF_WHITE_ON_NOTHING, THREE_QUARTER_WHITE_ON_NOTHING, TRANSPARENT, WHITE, at, rect,
    render_frame, render_frame_on_transparent,
};

fn window() -> gpui::Size<gpui::Pixels> {
    size(px(200.), px(100.))
}

/// White at half alpha, straight (not premultiplied) - the colour an element
/// hands gpui.
fn half_white() -> gpui::Hsla {
    hsla(0., 0., 1., 0.5)
}

#[test]
fn a_transparent_frame_starts_from_nothing() {
    // The guard for every other test here: if the target were still cleared to
    // opaque black, an assertion about alpha would be reading a constant.
    let frame = render_frame_on_transparent(window(), |_, window, _| {
        window.paint_quad(gpui::fill(rect(10., 10., 40., 40.), white()));
    });

    assert_eq!(
        frame.color_at(at(150., 50.)),
        TRANSPARENT,
        "a pixel nothing was painted at, on a transparent target"
    );
    frame.assert_painted(
        at(30., 30.),
        WHITE,
        "an opaque quad on a transparent target",
    );
}

#[test]
fn an_opaque_frame_still_clears_to_opaque_black() {
    // The transparent mode is an addition, not a migration: the default target
    // is unchanged, which is why the other 23 tests still read as they did.
    let frame = render_frame(window(), |_, window, _| {
        window.paint_quad(gpui::fill(rect(10., 10., 40., 40.), white()));
    });

    assert_eq!(frame.color_at(at(150., 50.)), harness::BACKGROUND);
    assert!(frame.is_background(at(150., 50.)));
}

#[test]
fn one_half_alpha_quad_composites_to_half_alpha() {
    let frame = render_frame_on_transparent(window(), |_, window, _| {
        window.paint_quad(gpui::fill(rect(10., 10., 80., 80.), half_white()));
    });

    frame.assert_painted(
        at(50., 50.),
        HALF_WHITE_ON_NOTHING,
        "one 50% white quad over nothing",
    );
}

#[test]
fn two_overlapping_half_alpha_quads_composite_to_three_quarter_alpha() {
    // Source-over accumulates alpha as `S.a + (1 - S.a) * D.a`, so a second
    // coat of 50% over the first leaves 0.75, not 1.0. A blend state that adds
    // the destination alpha instead saturates here, which is invisible on an
    // opaque target and wrong on every transparent one.
    let frame = render_frame_on_transparent(window(), |_, window, _| {
        window.paint_quad(gpui::fill(rect(10., 10., 100., 80.), half_white()));
        window.paint_quad(gpui::fill(rect(90., 10., 100., 80.), half_white()));
    });

    frame.assert_painted(
        at(50., 50.),
        HALF_WHITE_ON_NOTHING,
        "the left quad, where only it was painted",
    );
    frame.assert_painted(
        at(150., 50.),
        HALF_WHITE_ON_NOTHING,
        "the right quad, where only it was painted",
    );
    frame.assert_painted(
        at(100., 50.),
        THREE_QUARTER_WHITE_ON_NOTHING,
        "where the two 50% quads overlap",
    );
}
