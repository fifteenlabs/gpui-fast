//! Filling a path with a multi-stop gradient: `Window::paint_path_with_gradient`.
//!
//! `Background` carries a linear gradient of exactly two stops. CSS routinely
//! asks for three or more, for stops at positions of their own choosing, for
//! `repeating-linear-gradient`, for `radial-gradient` and for
//! `conic-gradient`; an HTML painter that degrades all of those to a flat
//! colour or a two-stop ramp is visibly wrong on the mail people actually get.
//!
//! Every test here is written so that the two things a gradient can degrade
//! into fail it. A **flat** fill fails because two of the points asserted differ
//! from each other. A **two-stop** ramp between the first and last colour fails
//! because the point in the middle is the middle *stop's* colour, which a
//! two-stop ramp cannot produce. Where a test's claim is about something else -
//! a stop's position, a repetition, a radius, an angle - it says so by asserting
//! a second frame, painted with the feature turned down, disagrees.
//!
//! Colours are written in the RGBA the harness reads back. Nothing here is
//! channel-swapped: unlike an image brush, the caller of a gradient hands over
//! colours, not texels, and the swap into the atlas's BGRA happens in the bake.

#![cfg(target_os = "macos")]

mod harness;

use std::sync::{Arc, Mutex};

use gpui::{
    BrushExtend, ColorSpace, Gradient, LinearColorStop, Path, Pixels, Radians, Window,
    linear_color_stop, point, px, rgb, rgba, size,
};
use harness::{
    at, rect, rect_path, render_frame, render_frame_on_transparent, render_frames,
    render_two_frames,
};

const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];

/// Halfway from red to green and from green to blue, which is what a three-stop
/// ramp holds a quarter and three quarters of the way along.
const RED_GREEN: [u8; 4] = [128, 128, 0, 255];
const GREEN_BLUE: [u8; 4] = [0, 128, 128, 255];

fn window() -> gpui::Size<Pixels> {
    size(px(200.), px(160.))
}

/// The shape every mapping test fills, and the rectangle the gradients run
/// across.
fn shape() -> gpui::Bounds<Pixels> {
    rect(20., 20., 160., 120.)
}

/// Fills `shape` with `gradient` at full alpha: what all but one of the tests
/// below ask for.
fn paint_gradient(window: &mut Window, shape: gpui::Bounds<Pixels>, gradient: &Gradient) {
    window
        .paint_path_with_gradient(rect_path(shape), gradient, 1.)
        .expect("the baked gradient fits in the atlas");
}

/// Red, green, blue at 0, a half and 1: the stop list whose middle a two-stop
/// approximation cannot reach.
fn three_stops() -> Vec<LinearColorStop> {
    vec![
        linear_color_stop(rgb(0xff0000), 0.),
        linear_color_stop(rgb(0x00ff00), 0.5),
        linear_color_stop(rgb(0x0000ff), 1.),
    ]
}

/// The horizontal axis of [`shape`]: the gradient runs left edge to right edge.
fn across_the_shape() -> (gpui::Point<Pixels>, gpui::Point<Pixels>) {
    (
        point(shape().left(), px(0.)),
        point(shape().right(), px(0.)),
    )
}

/// The point a fraction of the way across [`shape`].
fn along(fraction: f32) -> gpui::Point<Pixels> {
    at(20. + 160. * fraction, 80.)
}

fn close(a: [u8; 4], b: [u8; 4], tolerance: u8) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(a, b)| a.abs_diff(*b) <= tolerance)
}

#[test]
fn a_three_stop_linear_gradient_shows_its_middle_stop_in_the_middle() {
    // The test a two-stop approximation cannot pass. Red to blue through green
    // is (128, 0, 128) in the middle if the middle stop is dropped, and green if
    // it is not; and a flat fill of any single colour fails the quarter and
    // three-quarter points, which differ from each other and from the middle.
    let (start, end) = across_the_shape();
    let frame = render_frame(window(), move |_, window, _| {
        paint_gradient(
            window,
            shape(),
            &Gradient::linear(start, end, three_stops()),
        );
    });

    frame.assert_painted(along(0.5), GREEN, "the middle stop, in the middle");
    frame.assert_painted(along(0.25), RED_GREEN, "halfway to the middle stop");
    frame.assert_painted(along(0.75), GREEN_BLUE, "halfway past the middle stop");
    frame.assert_painted(along(0.02), [245, 10, 0, 255], "just past the first stop");
    frame.assert_painted(
        along(0.98),
        [0, 10, 245, 255],
        "just short of the last stop",
    );

    // Said again as a claim about the picture rather than about three colours:
    // whatever a degraded renderer painted, it did not paint three different
    // things.
    let sampled = [along(0.25), along(0.5), along(0.75)].map(|point| frame.color_at(point));
    for (index, colour) in sampled.iter().enumerate() {
        for other in &sampled[index + 1..] {
            assert!(
                !close(*colour, *other, 8),
                "two of the three sampled points came back the same colour, \
                 which is what a flat fill looks like"
            );
        }
    }
}

#[test]
fn a_stop_sits_where_its_position_puts_it() {
    // Frame 0 puts the green stop at a quarter, frame 1 at a half. Nothing else
    // differs. A renderer that read the stop list but ignored the positions -
    // spacing three stops evenly, which is what "0, 0.5, 1" happens to be -
    // makes the two frames identical, which the last assertion refuses.
    let (start, end) = across_the_shape();
    let frames = render_two_frames(window(), move |index, _, window, _| {
        let middle = if index == 0 { 0.25 } else { 0.5 };
        let stops = vec![
            linear_color_stop(rgb(0xff0000), 0.),
            linear_color_stop(rgb(0x00ff00), middle),
            linear_color_stop(rgb(0x0000ff), 1.),
        ];
        paint_gradient(window, shape(), &Gradient::linear(start, end, stops));
    });
    let [quarter, half] = frames;

    quarter.assert_painted(along(0.25), GREEN, "the stop at 0.25, a quarter along");
    quarter.assert_painted(
        along(0.5),
        [0, 170, 85, 255],
        "a third of the way from the stop at 0.25 to the one at 1",
    );
    half.assert_painted(along(0.5), GREEN, "the same stop moved to the middle");
    half.assert_painted(along(0.25), RED_GREEN, "halfway to it");

    for fraction in [0.25, 0.5] {
        assert!(
            !close(
                quarter.color_at(along(fraction)),
                half.color_at(along(fraction)),
                8
            ),
            "moving the middle stop changed nothing at {fraction} of the way \
             along, so stop positions are not reaching the bake"
        );
    }
}

/// A gradient axis a quarter of the shape wide, so the shape runs four periods
/// across it.
fn short_axis() -> (gpui::Point<Pixels>, gpui::Point<Pixels>) {
    (point(px(20.), px(0.)), point(px(60.), px(0.)))
}

/// The point `periods` gradient periods along [`short_axis`].
fn periods_along(periods: f32) -> gpui::Point<Pixels> {
    at(20. + 40. * periods, 80.)
}

#[test]
fn a_repeating_gradient_recurs_and_a_padded_one_does_not() {
    // `repeating-linear-gradient`. The axis ends a quarter of the way across the
    // shape, so everything past that is the extend mode's answer: under Repeat
    // the ramp starts over, under Pad it is the last stop's colour forever.
    let (start, end) = short_axis();
    let frames = render_two_frames(window(), move |index, _, window, _| {
        let extend = if index == 0 {
            BrushExtend::Repeat
        } else {
            BrushExtend::Pad
        };
        let stops = vec![
            linear_color_stop(rgb(0xff0000), 0.),
            linear_color_stop(rgb(0x0000ff), 1.),
        ];
        paint_gradient(
            window,
            shape(),
            &Gradient::linear(start, end, stops).extend(extend),
        );
    });
    let [repeated, padded] = frames;

    // Inside the one period both agree: whatever the extend mode, this is the
    // ramp itself.
    let inside = periods_along(0.25);
    repeated.assert_painted(inside, [191, 0, 64, 255], "a quarter into the first period");
    padded.assert_painted(inside, [191, 0, 64, 255], "the same point, padded");

    // The recurrence, said as an equality: a quarter into the second and fourth
    // periods is a quarter into the first.
    for periods in [1.25, 3.25] {
        let colour = repeated.color_at(periods_along(periods));
        assert!(
            close(colour, [191, 0, 64, 255], 8),
            "a quarter into the period at {periods} came back {colour:?} rather \
             than the colour a quarter into the first one\n  frame written to {}",
            repeated.save("a repeating gradient recurs").display()
        );
        padded.assert_painted(
            periods_along(periods),
            BLUE,
            "the padded control, which holds the last stop past the axis",
        );
    }
}

#[test]
fn a_radial_gradient_is_radially_symmetric() {
    // Equal colours at equal radii and different colours at different radii, in
    // four directions. A linear gradient of any angle fails the first claim -
    // its colour on one side of the centre is not its colour on the other - and
    // a flat fill fails the second.
    let centre = at(100., 80.);
    let frame = render_frame(window(), move |_, window, _| {
        paint_gradient(
            window,
            shape(),
            &Gradient::radial(centre, size(px(60.), px(60.)), three_stops()),
        );
    });

    let around = |radius: f32| {
        [
            at(100. + radius, 80.),
            at(100. - radius, 80.),
            at(100., 80. + radius),
            at(100., 80. - radius),
        ]
        .map(|point| frame.color_at(point))
    };

    let half_way = around(30.);
    for colour in half_way {
        assert!(
            close(colour, half_way[0], 8),
            "four points at the same radius came back {half_way:?}, which is not \
             one colour\n  frame written to {}",
            frame.save("a radial gradient is symmetric").display()
        );
    }
    // And that one colour is the middle stop's: the radius is half the
    // gradient's, so a two-stop red-to-blue approximation would have (128, 0,
    // 128) here. Said as dominance rather than as an exact colour, because a
    // 2-D bake is a bilinear magnification and a stop is a corner of the ramp:
    // the two texels either side of a peak both sit below it.
    let peak = frame.color_at(at(130., 80.));
    assert!(
        peak[1] > 240 && peak[0] < 16 && peak[2] < 16,
        "half the radius came back {peak:?}, where the middle stop is green; a \
         two-stop red-to-blue approximation would have (128, 0, 128) here\n  \
         frame written to {}",
        frame
            .save("a radial gradient reaches its middle stop")
            .display()
    );
    let middle = frame.color_at(centre);
    assert!(
        middle[0] > 240 && middle[1] < 16 && middle[2] < 16,
        "the centre came back {middle:?}, where the first stop is red"
    );

    let near = around(15.);
    let far = around(45.);
    assert!(
        !close(near[0], far[0], 8) && !close(near[0], half_way[0], 8),
        "three different radii came back the same colour, which is what a flat \
         fill looks like: {near:?} {half_way:?} {far:?}"
    );
}

#[test]
fn a_sweep_gradient_turns_with_the_angle_and_not_with_the_radius() {
    // What defines a sweep: the colour depends on the direction from the centre
    // and not on the distance from it. A radial gradient fails the first claim
    // and passes the second inverted; a linear one fails both.
    let centre = at(100., 80.);
    let frame = render_frame(window(), move |_, window, _| {
        let stops = vec![
            linear_color_stop(rgb(0x000000), 0.),
            linear_color_stop(rgb(0xffffff), 1.),
        ];
        paint_gradient(
            window,
            shape(),
            &Gradient::sweep(centre, Radians(-std::f32::consts::FRAC_PI_2), stops),
        );
    });

    // Due east is a quarter turn clockwise from due north, where the ramp
    // starts, so it is a quarter of the way along: mid-ramp, well clear of the
    // seam the whole turn closes at.
    let near_east = frame.color_at(at(130., 80.));
    let far_east = frame.color_at(at(170., 80.));
    assert!(
        close(near_east, far_east, 8),
        "two points due east of the centre at different radii came back \
         {near_east:?} and {far_east:?}, so the sweep is varying with the radius\
         \n  frame written to {}",
        frame
            .save("a sweep gradient turns with the angle")
            .display()
    );
    // A quarter of a black-to-white turn is a quarter grey, and the tolerance
    // is the harness's own: four levels out of 255 is what 8-bit rounding and
    // the bake's own quantization are worth, and on this sweep it is a bit over
    // five degrees of angle. Anything looser stops being a claim about where
    // the gradient points.
    assert!(
        close(near_east, [64, 64, 64, 255], harness::CHANNEL_TOLERANCE),
        "due east is a quarter of the way round a black-to-white sweep, which is \
         a quarter grey, and came back {near_east:?}"
    );

    let west = frame.color_at(at(70., 80.));
    assert!(
        close(west, [191, 191, 191, 255], harness::CHANNEL_TOLERANCE),
        "due west is three quarters of the way round, and came back {west:?}"
    );
    assert!(
        !close(near_east, west, 8),
        "east and west came back the same colour, which is what a flat fill \
         looks like"
    );
}

#[test]
fn a_gradient_fills_a_triangle_and_only_the_triangle() {
    // A gradient rides the brush path, so the coverage the path rasterizer
    // computed still decides which pixels get any of it - and the ramp still
    // runs across the shape's bounding rectangle rather than across the
    // triangle.
    let (start, end) = across_the_shape();
    let frame = render_frame(window(), move |_, window, _| {
        let mut path = Path::new(at(20., 20.));
        path.line_to(at(180., 20.));
        path.line_to(at(100., 140.));
        window
            .paint_path_with_gradient(path, &Gradient::linear(start, end, three_stops()), 1.)
            .expect("the baked ramp fits in the atlas");
    });

    frame.assert_painted(at(100., 60.), GREEN, "the middle stop, inside the triangle");
    frame.assert_painted(at(60., 30.), RED_GREEN, "a quarter along, inside it");
    frame.assert_painted(at(140., 30.), GREEN_BLUE, "three quarters along, inside it");
    frame.assert_all_clipped_away(
        &[at(25., 130.), at(175., 130.)],
        "the corners of the gradient's rectangle that the triangle does not cover",
    );
}

#[test]
fn a_gradient_composites_over_what_is_under_it() {
    // The premultiplication test. The path pipeline blends
    // One/OneMinusSourceAlpha, so a stop's straight alpha has to be multiplied
    // into its colour on the way out. Getting that backwards is invisible over
    // the harness's black - `rgb + 0` and `rgb * 1` agree there - and over blue
    // it is the difference between (128, 0, 128) and (255, 0, 128).
    //
    // Both halves of the alpha path are here: the left shape's alpha is a *stop
    // in the ramp*, which is the new thing a baked gradient has to get right,
    // and the right shape's is the `alpha` argument, which multiplies the baked
    // alpha rather than replacing it.
    let left = rect(20., 20., 60., 120.);
    let right = rect(120., 20., 60., 120.);
    let frame = render_frame(window(), move |bounds, window, _| {
        window.paint_quad(gpui::fill(bounds, gpui::blue()));

        // Five stops, with the alpha held at 1 across the ends: a stop is a
        // corner of the ramp, and a plateau is what makes "the ends are opaque"
        // a claim about a region rather than about one texel.
        let fading = vec![
            linear_color_stop(rgb(0xff0000), 0.),
            linear_color_stop(rgb(0xff0000), 0.3),
            linear_color_stop(rgba(0xff000080), 0.5),
            linear_color_stop(rgb(0xff0000), 0.7),
            linear_color_stop(rgb(0xff0000), 1.),
        ];
        paint_gradient(
            window,
            left,
            &Gradient::linear(
                point(left.left(), px(0.)),
                point(left.right(), px(0.)),
                fading,
            ),
        );

        let opaque = vec![
            linear_color_stop(rgb(0xff0000), 0.),
            linear_color_stop(rgb(0xff0000), 0.5),
            linear_color_stop(rgb(0xff0000), 1.),
        ];
        window
            .paint_path_with_gradient(
                rect_path(right),
                &Gradient::linear(
                    point(right.left(), px(0.)),
                    point(right.right(), px(0.)),
                    opaque,
                ),
                0.5,
            )
            .expect("the baked ramp fits in the atlas");
    });

    frame.assert_painted(
        at(50., 80.),
        [128, 0, 127, 255],
        "the half-alpha stop in the middle of the left ramp, over opaque blue",
    );
    frame.assert_painted(
        at(30., 80.),
        RED,
        "the opaque run at the start of the same ramp, which pins the fade on \
         the stop rather than on gradients being transparent",
    );
    frame.assert_painted(at(70., 80.), RED, "the opaque run at its end");
    frame.assert_painted(
        at(150., 80.),
        [128, 0, 127, 255],
        "an opaque ramp painted at half alpha, over the same blue",
    );
    frame.assert_painted(at(100., 80.), BLUE, "the quad between the two shapes");
}

#[test]
fn a_two_stop_background_still_renders_itself_and_bakes_nothing() {
    // The "costs nothing" proof. gpui's own UI paints two-stop `Background`
    // gradients constantly; they must keep going through the shader that has
    // always drawn them, and must not start putting textures in an atlas that
    // never gives one back.
    //
    // The counts are taken at three moments inside one paint pass, so the
    // zero at the first is not the vacuous zero of a window that never baked
    // anything at all.
    let counts: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = counts.clone();
    let (start, end) = across_the_shape();
    let left = rect(20., 20., 60., 120.);
    let right = rect(120., 20., 60., 120.);
    let frame = render_frame(window(), move |_, window, _| {
        let mut recorded = recorded.lock().expect("the paint callback runs alone");
        recorded.clear();

        window.paint_path(
            rect_path(left),
            gpui::linear_gradient(
                90.,
                gpui::linear_color_stop(rgb(0xff0000), 0.),
                gpui::linear_color_stop(rgb(0x0000ff), 1.),
            ),
        );
        recorded.push(window.baked_gradient_count());

        paint_gradient(window, right, &Gradient::linear(start, end, three_stops()));
        recorded.push(window.baked_gradient_count());

        window.paint_path(
            rect_path(left),
            gpui::linear_gradient(
                90.,
                gpui::linear_color_stop(rgb(0xff0000), 0.),
                gpui::linear_color_stop(rgb(0x0000ff), 1.),
            ),
        );
        recorded.push(window.baked_gradient_count());
    });

    let counts = counts.lock().expect("the paint callback has finished");
    assert_eq!(
        counts.as_slice(),
        &[0, 1, 1],
        "a two-stop Background must bake nothing, before or after a real \
         gradient has baked one thing"
    );

    // The brushed shape in the middle of the interleave drew its own gradient:
    // without this every assertion in this test is about the unbrushed path,
    // and a frame in which the brushed one painted nothing at all passes.
    // `right` spans the last three eighths of the ramp, so three quarters along
    // is inside it and is the halfway colour between the middle stop and the
    // last.
    frame.assert_painted(
        at(140., 80.),
        GREEN_BLUE,
        "three quarters along the brushed three-stop ramp, inside the right \
         shape",
    );
    frame.assert_painted(
        at(170., 80.),
        [0, 32, 223, 255],
        "seven eighths along the same ramp, which a flat fill or a path that \
         lost its brush cannot also be",
    );

    // And it still draws: a 90-degree gradient runs left to right across the
    // path's own bounds, so the shape's own edges are its ends. Checked loosely,
    // because that shader dithers its output by design - the claim here is that
    // the two-stop path still paints its own ramp, not that it paints a
    // particular byte.
    for (point, expected, what) in [
        (
            at(22., 80.),
            [245, 0, 10, 255],
            "the two-stop ramp, near red",
        ),
        (at(50., 80.), [127, 0, 128, 255], "its middle"),
        (at(78., 80.), [7, 0, 248, 255], "near blue"),
    ] {
        let actual = frame.color_at(point);
        assert!(
            close(actual, expected, 8),
            "{what}: expected about {expected:?}, found {actual:?}\n  \
             frame written to {}",
            frame.save(what).display()
        );
    }
}

#[test]
fn the_same_gradient_painted_twice_bakes_once() {
    // The cache's whole claim: a gradient repeated down a page is one bake and
    // one atlas tile, and one that differs in any respect the bake reads is
    // another.
    let counts: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = counts.clone();
    let (start, end) = across_the_shape();
    let frame = render_frame(window(), move |_, window, _| {
        let mut recorded = recorded.lock().expect("the paint callback runs alone");
        recorded.clear();

        for shape in [rect(20., 20., 160., 50.), rect(20., 90., 160., 50.)] {
            paint_gradient(window, shape, &Gradient::linear(start, end, three_stops()));
        }
        recorded.push(window.baked_gradient_count());

        paint_gradient(
            window,
            rect(120., 20., 60., 120.),
            &Gradient::linear(start, end, three_stops()).color_space(ColorSpace::Oklab),
        );
        recorded.push(window.baked_gradient_count());
    });

    let counts = counts.lock().expect("the paint callback has finished");
    assert_eq!(
        counts.as_slice(),
        &[1, 2],
        "the same gradient twice is one bake, and a different interpolation \
         space is a second"
    );

    // Both copies drew, so "one bake" is not "one of them was dropped": the two
    // shapes span the same x, so the ramp puts the same colour in both.
    frame.assert_painted(at(100., 45.), GREEN, "the first copy of the gradient");
    frame.assert_painted(at(100., 115.), GREEN, "the second copy of it");
}

#[test]
fn a_stop_at_transparent_does_not_fade_through_black() {
    // `linear-gradient(red, transparent)` over white, which HTML mail is full
    // of. Interpolating the stops straight runs the colour towards
    // transparent's *black* as the alpha comes down, so the middle of the ramp
    // is a half-alpha dark red and the white underneath shows through it as a
    // grey band; interpolating premultiplied leaves the colour where the stops
    // put it and takes only the alpha down.
    //
    // Over white rather than over the harness's black, because over black the
    // two are the same picture: a colour fading towards black over black is
    // invisible.
    let (start, end) = across_the_shape();
    let frame = render_frame(window(), move |bounds, window, _| {
        window.paint_quad(gpui::fill(bounds, gpui::white()));
        let fading = vec![
            linear_color_stop(rgb(0xff0000), 0.),
            linear_color_stop(rgba(0x00000000), 1.),
        ];
        paint_gradient(window, shape(), &Gradient::linear(start, end, fading));
    });

    frame.assert_painted(
        along(0.5),
        [255, 128, 128, 255],
        "half way along a red-to-transparent ramp over white, where a straight \
         interpolation would have dimmed the red to (191, 128, 128)",
    );
    frame.assert_painted(
        along(0.02),
        [255, 5, 5, 255],
        "just past the red end of the same ramp, two percent of the way into \
         its fade",
    );
    frame.assert_painted(
        along(0.98),
        [255, 250, 250, 255],
        "just short of the transparent end, where two percent of the red is all \
         that is left of it",
    );
}

#[test]
fn a_full_gradient_cache_evicts_rather_than_giving_up_on_gradients() {
    // The cache is a working set, not a latch. A page with more distinct
    // gradients than fit is a page that re-bakes as it scrolls; before, the
    // window that had baked its last byte painted every gradient it had not
    // already seen as a flat colour, for the rest of its life.
    //
    // Frame 0 fills the cache to the byte. Frame 1 paints no gradient at all,
    // which is what lets frame 0's tiles stop being live - a tile the frame
    // being painted or the frame on its way to the GPU still points at cannot
    // be given back, or the next bake is allocated over pixels something is
    // still reading. Frame 2 then asks for a gradient the window has never
    // baked.
    const TEXELS: usize = gpui::MAX_FIELD_TEXELS as usize;
    let fills_the_cache = gpui::MAX_GRADIENT_CACHE_BYTES / (TEXELS * TEXELS * 4);
    let counts: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = counts.clone();
    let centre = at(100., 80.);

    let frames = render_frames(window(), 3, move |index, _, window, _| {
        match index {
            // Distinct only in one stop's colour, so every one of them is a
            // separate bake of the largest size a field takes.
            0 => {
                for bake in 0..fills_the_cache {
                    let stops = vec![
                        linear_color_stop(rgb(0x800000 | bake as u32), 0.),
                        linear_color_stop(rgb(0x00ff00), 0.5),
                        linear_color_stop(rgb(0x0000ff), 1.),
                    ];
                    paint_gradient(
                        window,
                        shape(),
                        &Gradient::radial(centre, size(px(60.), px(60.)), stops),
                    );
                }
            }
            1 => window.paint_quad(gpui::fill(shape(), gpui::blue())),
            _ => {
                paint_gradient(
                    window,
                    shape(),
                    &Gradient::radial(centre, size(px(60.), px(60.)), three_stops()),
                );
            }
        }
        recorded
            .lock()
            .expect("the paint callback runs alone")
            .push(window.baked_gradient_count());
    });

    let counts = counts.lock().expect("the paint callbacks have finished");
    assert_eq!(
        counts.as_slice(),
        &[fills_the_cache, fills_the_cache, fills_the_cache + 1],
        "a cache with nothing left in it has to evict and bake, not give up: \
         the third frame's gradient was never baked"
    );

    // And the gradient it baked is the one that was asked for. A window that
    // gave up paints the stop list's halfway colour - the middle stop, green -
    // flat across the whole shape, so the centre is where the two differ.
    let last = &frames[2];
    let middle = last.color_at(centre);
    assert!(
        middle[0] > 240 && middle[1] < 16 && middle[2] < 16,
        "the centre of the gradient baked after an eviction came back \
         {middle:?}, where its first stop is red; flat would be green here\n  \
         frame written to {}",
        last.save("a gradient baked after an eviction").display()
    );
    last.assert_painted(
        at(130., 80.),
        GREEN,
        "half the radius out, where the middle stop is",
    );
}

#[test]
fn a_gradient_carries_the_alpha_of_its_stops() {
    // Every other test in this file paints onto the harness's opaque black,
    // where the alpha that comes back is 255 whatever the pipeline emitted.
    // Here the same ramp is painted onto nothing at all, so the premultiplied
    // RGBA the brush pipeline actually wrote is what is read back: a stop at
    // half alpha has to arrive as a half-alpha *and* half-strength red.
    let (start, end) = across_the_shape();
    let frame = render_frame_on_transparent(window(), move |_, window, _| {
        let fading = vec![
            linear_color_stop(rgba(0xff000080), 0.),
            linear_color_stop(rgba(0xff000080), 0.5),
            linear_color_stop(rgb(0xff0000), 1.),
        ];
        paint_gradient(window, shape(), &Gradient::linear(start, end, fading));
    });

    frame.assert_painted(
        along(0.25),
        [128, 0, 0, 128],
        "the half-alpha plateau of a ramp painted over nothing",
    );
    frame.assert_painted(
        along(0.98),
        [250, 0, 0, 250],
        "just short of its opaque end, which pins the alpha above on the stop \
         rather than on gradients being transparent",
    );
    frame.assert_painted(
        at(10., 80.),
        harness::TRANSPARENT,
        "outside the shape, where nothing was painted",
    );
}
