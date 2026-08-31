//! What `Window::with_element_opacity` does to pixels.

#![cfg(target_os = "macos")]

mod harness;

use gpui::{px, size, white};
use harness::{HALF_WHITE, WHITE, at, rect, render_frame};

fn window() -> gpui::Size<gpui::Pixels> {
    size(px(200.), px(100.))
}

#[test]
fn element_opacity_halves_the_intensity_of_what_is_painted_inside_it() {
    // Two identical white quads, side by side. The right one is painted inside
    // a 0.5 opacity layer, so over the black background it must come out half
    // as bright as the left one.
    let frame = render_frame(window(), |_, window, _| {
        window.paint_quad(gpui::fill(rect(10., 10., 80., 80.), white()));
        window.with_element_opacity(Some(0.5), |window| {
            window.paint_quad(gpui::fill(rect(110., 10., 80., 80.), white()));
        });
    });

    frame.assert_painted(at(50., 50.), WHITE, "a quad painted at full opacity");
    frame.assert_painted(
        at(150., 50.),
        HALF_WHITE,
        "the same quad painted inside a 0.5 opacity layer",
    );
}

#[test]
fn element_opacity_halves_a_path_too() {
    let frame = render_frame(window(), |_, window, _| {
        window.with_element_opacity(Some(0.5), |window| {
            let bounds = rect(10., 10., 180., 80.);
            let mut path = gpui::Path::new(bounds.origin);
            path.line_to(bounds.top_right());
            path.line_to(bounds.bottom_right());
            path.line_to(bounds.bottom_left());
            window.paint_path(path, white());
        });
    });

    frame.assert_painted(
        at(100., 50.),
        HALF_WHITE,
        "a path painted inside a 0.5 opacity layer",
    );
}

#[test]
fn element_opacity_nests_multiplicatively() {
    // 0.5 inside 0.5 is 0.25, which over black is a quarter-intensity white.
    let frame = render_frame(window(), |_, window, _| {
        window.with_element_opacity(Some(0.5), |window| {
            window.with_element_opacity(Some(0.5), |window| {
                window.paint_quad(gpui::fill(rect(10., 10., 80., 80.), white()));
            });
        });
    });

    frame.assert_painted(
        at(50., 50.),
        [64, 64, 64, 255],
        "a quad two 0.5 opacity layers deep",
    );
}

#[test]
fn element_opacity_is_scoped_to_the_closure() {
    // Anything painted after the closure returns is at full opacity again.
    let frame = render_frame(window(), |_, window, _| {
        window.with_element_opacity(Some(0.5), |window| {
            window.paint_quad(gpui::fill(rect(10., 10., 80., 80.), white()));
        });
        window.paint_quad(gpui::fill(rect(110., 10., 80., 80.), white()));
    });

    frame.assert_painted(at(50., 50.), HALF_WHITE, "inside the opacity layer");
    frame.assert_painted(at(150., 50.), WHITE, "after the opacity layer returned");
}
