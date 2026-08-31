//! What a [`gpui::SceneFilter`] that is not a colour matrix does to pixels:
//! a separable gaussian blur, a drop shadow, and a filter applied to what is
//! behind a group rather than to the group.
//!
//! Every test here is written so that it fails when the thing it names stops
//! happening rather than when nothing happens at all. The trap these tests are
//! built against is the one the earlier work fell into: a grayscale-on-opaque
//! assertion that still passed with the shader broken, because an opaque pixel
//! hides a premultiplication bug. So a blur is checked against a *different*
//! implementation of the same integral - gpui's own box shadow - rather than
//! against a number this file worked out; each axis of the blur is checked
//! separately, so skipping a pass is not silently half right; and the cases
//! with partial alpha are here because the ones without cannot see the bug.

#![cfg(target_os = "macos")]

mod harness;

use std::sync::Mutex;

use gpui::{
    Bounds, BoxShadow, Corners, GroupOptions, Pixels, SceneFilter, Size, point, px, red, size,
    white,
};
use harness::{
    BACKGROUND, RenderedFrame, TRANSPARENT, WHITE, at, rect, render_frame,
    render_frame_on_transparent,
};

/// A window with room for a square and three standard deviations of tail on
/// every side of it.
fn window() -> Size<Pixels> {
    size(px(240.), px(120.))
}

/// The standard deviation every blur here uses, in logical pixels. Sixteen
/// device pixels: wide enough that a half-device-pixel sampling offset is a
/// thirtieth of it, and narrow enough that its tail fits in [`window`].
const SIGMA: f32 = 8.;

/// The square the blur tests blur. Its edges are more than three standard
/// deviations from the window's, so nothing here is a test of what happens when
/// a tail runs out of window.
fn square() -> Bounds<Pixels> {
    rect(80., 30., 80., 60.)
}

/// The alpha channel at a logical point.
fn alpha_at(frame: &RenderedFrame, x: f32, y: f32) -> u8 {
    frame.color_at(at(x, y))[3]
}

/// How far a blurred alpha may sit from what the box shadow shader makes of the
/// same standard deviation.
///
/// The two compute the same integral by different means - this one sums the
/// gaussian at whole texels and normalizes by the weight it found, the shadow
/// shader integrates the error function along x and takes a four-step Riemann
/// sum along y - so they are not expected to agree to the byte. They are
/// expected to agree to a twentieth, which a wrong reading of what the radius
/// *means* could not: taking the radius for twice the standard deviation, the
/// usual mistake, moves the value one standard deviation out from 40 to 79.
const WEIGHT_TOLERANCE: u8 = 13;

// ------------------------------------------------------------------- blur --

fn blurred_square(sigma_x: f32, sigma_y: f32) -> RenderedFrame {
    render_frame_on_transparent(window(), move |_, window, _| {
        window.with_isolated_group(
            GroupOptions {
                bounds: square(),
                filter: Some(SceneFilter::Blur {
                    radius_x: sigma_x,
                    radius_y: sigma_y,
                }),
                ..Default::default()
            },
            |window| {
                window.paint_quad(gpui::fill(square(), white()));
            },
        );
    })
}

/// The same square's alpha, blurred by gpui's own drop shadow shader at the
/// same standard deviation: an independent implementation of the thing under
/// test.
fn shadowed_square() -> RenderedFrame {
    render_frame_on_transparent(window(), move |_, window, _| {
        window.paint_drop_shadows(
            square(),
            Corners::default(),
            &[BoxShadow {
                color: white(),
                offset: point(px(0.), px(0.)),
                blur_radius: px(SIGMA),
                spread_radius: px(0.),
                inset: false,
            }],
        );
    })
}

#[test]
fn a_group_blur_weighs_the_same_as_a_box_shadow_of_the_same_standard_deviation() {
    let blurred = blurred_square(SIGMA, SIGMA);
    let shadow = shadowed_square();

    // Across the square's right edge, from two standard deviations inside it to
    // two outside. The middle of the window vertically, where the top and
    // bottom edges are three and three quarter standard deviations away and
    // contribute a factor of one.
    for offset in [-2. * SIGMA, -SIGMA, 0., SIGMA, 2. * SIGMA] {
        let x = 160. + offset;
        let ours = alpha_at(&blurred, x, 60.);
        let theirs = alpha_at(&shadow, x, 60.);
        assert!(
            ours.abs_diff(theirs) <= WEIGHT_TOLERANCE,
            "{offset} logical pixels from the edge the group blur left {ours} \
             where the box shadow of the same standard deviation left {theirs}"
        );
    }

    // And it is a profile rather than a constant, so the agreement above is
    // agreement about a shape.
    let inside = alpha_at(&blurred, 160. - 2. * SIGMA, 60.);
    let outside = alpha_at(&blurred, 160. + 2. * SIGMA, 60.);
    assert!(
        inside > 200 && outside < 40,
        "two standard deviations either side of the edge came out at {inside} \
         and {outside}, which is not a falloff"
    );
    // Half the weight sits either side of the edge itself.
    let edge = alpha_at(&blurred, 160., 60.);
    assert!(
        edge.abs_diff(128) <= WEIGHT_TOLERANCE,
        "the edge of a blurred square came out at {edge}, not half"
    );
}

#[test]
fn a_blur_runs_along_both_axes() {
    // One pass along x and one along y. A blur that ran only the first would
    // look right along one edge and be untouched along the other, which is why
    // this asks about both.
    let both = blurred_square(SIGMA, SIGMA);
    let across = alpha_at(&both, 160. + SIGMA, 60.);
    let down = alpha_at(&both, 120., 90. + SIGMA);
    assert!(
        across.abs_diff(40) <= WEIGHT_TOLERANCE,
        "one standard deviation past the right edge came out at {across}"
    );
    assert!(
        down.abs_diff(40) <= WEIGHT_TOLERANCE,
        "one standard deviation past the bottom edge came out at {down}"
    );
}

#[test]
fn a_blur_along_one_axis_leaves_the_other_alone() {
    // The control for the test above: with a standard deviation of zero along
    // y, the horizontal edges stay exactly where they were, and a blur that
    // ran both passes whatever it was asked for would soften them.
    let horizontal = blurred_square(SIGMA, 0.);
    horizontal.assert_painted(
        at(120., 90. + SIGMA),
        TRANSPARENT,
        "below a square blurred along x alone",
    );
    horizontal.assert_painted(
        at(120., 90. - SIGMA),
        WHITE,
        "inside a square blurred along x alone",
    );
    let spread = alpha_at(&horizontal, 160. + SIGMA, 60.);
    assert!(
        spread.abs_diff(40) <= WEIGHT_TOLERANCE,
        "the axis that was blurred came out at {spread}"
    );

    let vertical = blurred_square(0., SIGMA);
    vertical.assert_painted(
        at(160. + SIGMA, 60.),
        TRANSPARENT,
        "beside a square blurred along y alone",
    );
    vertical.assert_painted(
        at(160. - SIGMA, 60.),
        WHITE,
        "inside a square blurred along y alone",
    );
    let spread = alpha_at(&vertical, 120., 90. + SIGMA);
    assert!(
        spread.abs_diff(40) <= WEIGHT_TOLERANCE,
        "the axis that was blurred came out at {spread}"
    );
}

#[test]
fn a_blur_runs_on_premultiplied_colour() {
    // A group's target holds premultiplied colour, and `feGaussianBlur` is
    // defined over premultiplied colour, so the two agree and the pass reads
    // and writes the texels as they stand.
    //
    // Over an opaque square that is invisible, which is exactly the trap: every
    // channel is already its own premultiplied self. It is the *edge* that can
    // tell, because there the square is being mixed with the transparent black
    // around it. Premultiplied, red at half coverage is (128, 0, 0, 128) and
    // the red channel tracks the alpha. Straight, the red would be averaged
    // with the canvas's black first and the pixel would read (64, 0, 0, 128) -
    // the halo every renderer that gets this wrong draws round its edges.
    let frame = render_frame_on_transparent(window(), |_, window, _| {
        window.with_isolated_group(
            GroupOptions {
                bounds: square(),
                filter: Some(SceneFilter::Blur {
                    radius_x: SIGMA,
                    radius_y: SIGMA,
                }),
                ..Default::default()
            },
            |window| {
                window.paint_quad(gpui::fill(square(), red()));
            },
        );
    });

    let edge = frame.color_at(at(160., 60.));
    assert!(
        edge[3].abs_diff(128) <= WEIGHT_TOLERANCE,
        "the edge of a blurred red square came out at alpha {}",
        edge[3]
    );
    assert!(
        edge[0].abs_diff(edge[3]) <= 4,
        "a premultiplied red at alpha {} has red {}, and this pixel is {edge:?}",
        edge[3],
        edge[3]
    );
    assert!(
        edge[1] <= 4 && edge[2] <= 4,
        "blurring red produced {edge:?}, which is not red any more"
    );
}

#[test]
fn a_blur_of_partial_alpha_keeps_the_alpha_it_started_from() {
    // The same claim where the *input* is already translucent, which is where a
    // pass that divided the alpha out and put it back the wrong way round would
    // show. Half-alpha red reaches the group's target as (128, 0, 0, 128), and
    // the interior of a region far larger than the kernel has to come back
    // exactly as it went in: a normalized gaussian over a constant field is
    // that constant.
    let frame = render_frame_on_transparent(window(), |_, window, _| {
        window.with_isolated_group(
            GroupOptions {
                bounds: square(),
                filter: Some(SceneFilter::Blur {
                    radius_x: SIGMA,
                    radius_y: SIGMA,
                }),
                ..Default::default()
            },
            |window| {
                window.paint_quad(gpui::fill(square(), red().opacity(0.5)));
            },
        );
    });

    frame.assert_painted(
        at(120., 60.),
        [128, 0, 0, 128],
        "the middle of a blurred half-transparent red square",
    );
    // And at its edge, half of that again, colour and coverage together.
    let edge = frame.color_at(at(160., 60.));
    assert!(
        edge[3].abs_diff(64) <= WEIGHT_TOLERANCE && edge[0].abs_diff(edge[3]) <= 4,
        "the edge of a blurred half-transparent red square came out {edge:?}"
    );
}

#[test]
fn a_blurred_edge_differs_from_the_unblurred_edge_in_the_same_place() {
    // The control against the easiest false positive there is: an assertion
    // that something was painted, passing on a frame where nothing was blurred
    // at all. Same scene, same group, same points; the only difference is the
    // standard deviation.
    let blurred = blurred_square(SIGMA, SIGMA);
    let sharp = blurred_square(0., 0.);

    sharp.assert_painted(at(158., 60.), WHITE, "just inside an unblurred edge");
    sharp.assert_painted(at(162., 60.), TRANSPARENT, "just outside an unblurred edge");

    let inside = alpha_at(&blurred, 158., 60.);
    let outside = alpha_at(&blurred, 162., 60.);
    assert!(
        (130..=200).contains(&inside),
        "just inside a blurred edge the alpha was {inside}, which is either the \
         unblurred square or nothing at all"
    );
    assert!(
        (60..=125).contains(&outside),
        "just outside a blurred edge the alpha was {outside}"
    );
}

#[test]
fn a_blur_reaches_past_the_bounds_the_group_asked_for() {
    // `GroupOptions::bounds` is the square itself, and a blurred group is
    // larger than its content: the scene grows the target by the gaussian's
    // tail, so the tail has somewhere to land. Without that growth the group's
    // target would stop at the square and the blur would end in a hard edge at
    // exactly the place it was supposed to soften.
    let blurred = blurred_square(SIGMA, SIGMA);
    let sharp = blurred_square(0., 0.);

    // Between one and two standard deviations outside the right edge, which is
    // outside the bounds the caller named by any reading.
    let beyond = rect(160. + SIGMA, 55., SIGMA, 10.);
    blurred.assert_region_painted(beyond, "the tail of a blur past the group's bounds");
    sharp.assert_region_clipped_away(beyond, "the same region with no blur asked for");
}

#[test]
fn a_blur_that_runs_out_of_target_fades_rather_than_smearing() {
    // The content mask in force is the furthest a group's result may reach, so
    // a square against the window's left edge is blurred inside a target whose
    // own left edge is the window's, and the pass reads past it. What it has to
    // find there is the transparent black the group was painted on: a group's
    // filter is defined over the group's own rendering, and there is nothing
    // outside that rendering. Finding a copy of the last column instead would
    // drag the square's ink along the whole margin rather than fading it, which
    // is the difference between a blur and a smear.
    let bounds = rect(0., 30., 80., 60.);
    let frame = render_frame_on_transparent(window(), move |_, window, _| {
        window.with_isolated_group(
            GroupOptions {
                bounds,
                filter: Some(SceneFilter::Blur {
                    radius_x: SIGMA,
                    radius_y: SIGMA,
                }),
                ..Default::default()
            },
            |window| {
                window.paint_quad(gpui::fill(bounds, white()));
            },
        );
    });

    // Half a device pixel inside the window's left edge, where the kernel finds
    // ink on one side of itself and nothing at all on the other.
    let edge = alpha_at(&frame, 0., 60.);
    assert!(
        edge.abs_diff(128) <= WEIGHT_TOLERANCE,
        "against the edge of the target the blur left {edge}; at 255 it has \
         smeared the last column outwards instead of fading"
    );
    // And an ordinary edge of the same square, for the same reading.
    let inner = alpha_at(&frame, 80., 60.);
    assert!(
        inner.abs_diff(128) <= WEIGHT_TOLERANCE,
        "the square's own right edge came out at {inner}"
    );
}

#[test]
fn a_blurred_group_inside_another_group_keeps_its_tail() {
    // A group that survives is composited onto its parent's target as one quad
    // over its own bounds, and a blur has already grown those past everything
    // its primitives painted. A parent grown to its primitives alone would hang
    // that quad over the edge of its target and cut the tail off there, undoing
    // the growth at one remove.
    let frame = render_frame_on_transparent(window(), |_, window, _| {
        window.with_isolated_group(
            GroupOptions {
                bounds: square(),
                opacity: 0.5,
                ..Default::default()
            },
            |window| {
                window.with_isolated_group(
                    GroupOptions {
                        bounds: square(),
                        filter: Some(SceneFilter::Blur {
                            radius_x: SIGMA,
                            radius_y: SIGMA,
                        }),
                        ..Default::default()
                    },
                    |window| {
                        window.paint_quad(gpui::fill(square(), white()));
                    },
                );
            },
        );
    });

    frame.assert_region_painted(
        rect(160. + SIGMA, 55., SIGMA, 10.),
        "the tail of a blur inside an enclosing group",
    );
}

#[test]
fn a_chain_applies_a_blur_and_a_colour_matrix_in_order() {
    // Two filters, one of which needs a pass of its own and one of which the
    // composite folds into its own fragment stage. Both have to happen.
    let frame = render_frame_on_transparent(window(), |_, window, _| {
        window.with_isolated_group(
            GroupOptions {
                bounds: square(),
                filter: Some(SceneFilter::Chain(vec![
                    SceneFilter::Blur {
                        radius_x: SIGMA,
                        radius_y: SIGMA,
                    },
                    SceneFilter::ColorMatrix(GRAYSCALE),
                ])),
                ..Default::default()
            },
            |window| {
                window.paint_quad(gpui::fill(square(), red()));
            },
        );
    });

    // Grey, so the matrix ran; and 54 is red's luminance, so it ran on red
    // rather than on something the blur had already turned to mud.
    frame.assert_painted(
        at(120., 60.),
        [54, 54, 54, 255],
        "the middle of a blurred, greyed red square",
    );
    // Still blurred, so the matrix did not replace the blur.
    let edge = alpha_at(&frame, 160., 60.);
    assert!(
        edge.abs_diff(128) <= WEIGHT_TOLERANCE,
        "the edge of the blurred, greyed square came out at {edge}"
    );
}

/// `grayscale(1)` as CSS defines it.
const GRAYSCALE: [f32; 20] = [
    0.2126, 0.7152, 0.0722, 0., 0., //
    0.2126, 0.7152, 0.0722, 0., 0., //
    0.2126, 0.7152, 0.0722, 0., 0., //
    0., 0., 0., 1., 0.,
];

/// The luminance of pure red, as a byte.
const GRAY_RED: [u8; 4] = [54, 54, 54, 255];

// ------------------------------------------------------------ drop shadow --

/// The drop shadow's own standard deviation and offset, both large enough that
/// the offset is several standard deviations: a shadow drawn without its offset
/// then reaches nowhere near where this one does.
const SHADOW_SIGMA: f32 = 6.;
const SHADOW_OFFSET: f32 = 30.;

/// A white square with a red shadow under it, `SHADOW_OFFSET` down and right.
fn drop_shadowed_square() -> RenderedFrame {
    let bounds = rect(80., 25., 80., 50.);
    render_frame_on_transparent(window(), move |_, window, _| {
        window.with_isolated_group(
            GroupOptions {
                bounds,
                filter: Some(SceneFilter::DropShadow {
                    offset_x: SHADOW_OFFSET,
                    offset_y: SHADOW_OFFSET,
                    radius: SHADOW_SIGMA,
                    color: red(),
                }),
                ..Default::default()
            },
            |window| {
                window.paint_quad(gpui::fill(bounds, white()));
            },
        );
    })
}

#[test]
fn a_drop_shadow_lands_where_its_offset_puts_it() {
    let frame = drop_shadowed_square();

    // The shadow's own rectangle is the square moved thirty logical pixels down
    // and right: 110..190 x 55..105. Well inside it, and outside the square.
    frame.assert_painted(
        at(170., 85.),
        [255, 0, 0, 255],
        "inside a drop shadow offset down and to the right",
    );

    // Above the square, where a shadow drawn without its offset would have
    // reached: five logical pixels past the top edge is under a standard
    // deviation, so an unoffset shadow would leave a fifth of its colour here.
    frame.assert_painted(
        at(120., 20.),
        TRANSPARENT,
        "above the square, where only an unoffset shadow would reach",
    );
}

#[test]
fn a_drop_shadow_is_drawn_behind_its_source_in_its_own_colour() {
    let frame = drop_shadowed_square();

    // The square itself is untouched: the shadow goes underneath it, and an
    // opaque source hides it completely.
    frame.assert_painted(at(120., 50.), WHITE, "the square a drop shadow sits under");

    // Where the shadow shows, it is the colour the filter named rather than the
    // colour of what cast it. The square is white; this is red.
    let shadow = frame.color_at(at(170., 85.));
    assert!(
        shadow[0] > 200 && shadow[1] < 20 && shadow[2] < 20,
        "the shadow of a white square came out {shadow:?} rather than the red \
         the filter asked for"
    );
}

#[test]
fn a_drop_shadow_blurs_its_source_alpha_at_the_standard_deviation_it_was_given() {
    let frame = drop_shadowed_square();

    // Across the shadow rectangle's own right edge at x = 190, far from its
    // top and bottom. The same falloff the group blur has, because it is the
    // same pass: half the weight at the edge, a sixth one standard deviation
    // out, a fortieth two out.
    for (offset, expected) in [(-SHADOW_SIGMA, 215u8), (0., 128), (SHADOW_SIGMA, 40)] {
        let found = alpha_at(&frame, 190. + offset, 80.);
        assert!(
            found.abs_diff(expected) <= WEIGHT_TOLERANCE,
            "{offset} logical pixels from the shadow's edge the alpha was \
             {found} where a gaussian of {SHADOW_SIGMA} leaves {expected}"
        );
    }

    // And across its top edge at y = 55, at an x the square itself does not
    // cover. This is the near side, and the near side is where a target grown
    // only far enough for the shadow's *result* cuts the falloff off in a
    // straight line: the shadow is read back out of the blurred silhouette at
    // `p - offset`, which from here lands thirty logical pixels above the
    // square, and the square's own blur tail has to still be there.
    for (offset, expected) in [(-SHADOW_SIGMA, 40u8), (0., 128), (SHADOW_SIGMA, 215)] {
        let found = alpha_at(&frame, 170., 55. + offset);
        assert!(
            found.abs_diff(expected) <= WEIGHT_TOLERANCE,
            "{offset} logical pixels from the shadow's top edge the alpha was \
             {found} where a gaussian of {SHADOW_SIGMA} leaves {expected}"
        );
    }
}

#[test]
fn a_drop_shadow_of_no_radius_is_the_silhouette_itself() {
    // The degenerate case the blur passes are skipped for. It still has to be
    // the source's alpha, offset and tinted, rather than nothing.
    let bounds = rect(80., 25., 80., 50.);
    let frame = render_frame_on_transparent(window(), move |_, window, _| {
        window.with_isolated_group(
            GroupOptions {
                bounds,
                filter: Some(SceneFilter::DropShadow {
                    offset_x: SHADOW_OFFSET,
                    offset_y: SHADOW_OFFSET,
                    radius: 0.,
                    color: red(),
                }),
                ..Default::default()
            },
            |window| {
                window.paint_quad(gpui::fill(bounds, white()));
            },
        );
    });

    frame.assert_painted(
        at(170., 85.),
        [255, 0, 0, 255],
        "inside an unblurred drop shadow",
    );
    // A hard edge, since nothing blurred it.
    frame.assert_painted(
        at(192., 85.),
        TRANSPARENT,
        "two logical pixels past an unblurred shadow's edge",
    );
}

// --------------------------------------------------------- backdrop filter --

#[test]
fn a_backdrop_filter_filters_what_is_behind_the_group_and_nothing_else() {
    let frame = render_frame(window(), |_, window, _| {
        window.paint_quad(gpui::fill(rect(0., 0., 240., 120.), red()));
        window.with_isolated_group(
            GroupOptions {
                bounds: rect(80., 30., 80., 60.),
                backdrop_filter: Some(SceneFilter::ColorMatrix(GRAYSCALE)),
                ..Default::default()
            },
            |_| {},
        );
    });

    frame.assert_painted(at(120., 60.), GRAY_RED, "the red behind the group");
    frame.assert_painted(
        at(20., 60.),
        [255, 0, 0, 255],
        "the red beside the group, which the filter must not reach",
    );
    frame.assert_painted(
        at(120., 100.),
        [255, 0, 0, 255],
        "the red below the group, which the filter must not reach",
    );
}

#[test]
fn a_backdrop_filter_is_faded_in_by_the_group_s_own_coverage() {
    // The filter belongs to the group, so the group's opacity applies to it:
    // half a group is half a filter, and the backdrop underneath comes out
    // halfway between what it was and what the filter made of it.
    //
    // This is what the composite needs the *unfiltered* copy of the backdrop
    // for. A composite that only had the filtered copy would have to write it
    // whole, and this pixel would come out fully grey.
    let frame = render_frame(window(), |_, window, _| {
        window.paint_quad(gpui::fill(rect(0., 0., 240., 120.), red()));
        window.with_isolated_group(
            GroupOptions {
                bounds: rect(80., 30., 80., 60.),
                opacity: 0.5,
                backdrop_filter: Some(SceneFilter::ColorMatrix(GRAYSCALE)),
                ..Default::default()
            },
            |_| {},
        );
    });

    // Halfway from (255, 0, 0) to (54, 54, 54).
    frame.assert_painted(
        at(120., 60.),
        [154, 27, 27, 255],
        "the red behind a half-strength grayscale backdrop filter",
    );
}

#[test]
fn a_backdrop_blur_softens_what_is_behind_the_group_and_not_beside_it() {
    // A hard edge across the window, and a group over the top of it. Under the
    // group the edge is blurred; at the same x, below the group, it is not.
    // That pairing is the control: an assertion that the pixel under the group
    // is grey would pass on a frame where the whole window had been blurred, or
    // where nothing had.
    let frame = render_frame(window(), |_, window, _| {
        window.paint_quad(gpui::fill(rect(0., 0., 120., 120.), white()));
        window.with_isolated_group(
            GroupOptions {
                bounds: rect(40., 10., 160., 40.),
                backdrop_filter: Some(SceneFilter::Blur {
                    radius_x: SIGMA,
                    radius_y: 0.,
                }),
                ..Default::default()
            },
            |_| {},
        );
    });

    let under = frame.color_at(at(120. + SIGMA, 30.))[0];
    let beside = frame.color_at(at(120. + SIGMA, 80.))[0];
    assert_eq!(
        beside, BACKGROUND[0],
        "the edge below the group was softened, so the filter reached further \
         than the group did"
    );
    assert!(
        under.abs_diff(40) <= WEIGHT_TOLERANCE,
        "one standard deviation past the edge, under a blurring group, the \
         backdrop came out at {under}"
    );

    // And the bright side of the edge dimmed by as much as the dark side
    // brightened, which is what a blur does and a grey wash does not.
    let under = frame.color_at(at(120. - SIGMA, 30.))[0];
    let beside = frame.color_at(at(120. - SIGMA, 80.))[0];
    assert_eq!(beside, WHITE[0], "the white below the group");
    assert!(
        under.abs_diff(215) <= WEIGHT_TOLERANCE,
        "one standard deviation inside the edge, under a blurring group, the \
         backdrop came out at {under}"
    );
}

#[test]
fn a_backdrop_blur_copies_more_of_the_backdrop_than_the_group_covers() {
    // A backdrop blur draws on what is *beside* the group as well as on what is
    // under it - that is what a blur is - so the copy taken of the backdrop
    // reaches a gaussian tail further out than the group does.
    //
    // The group here starts exactly where the backdrop turns from white to
    // black, so the first column under it has to come back half way between the
    // two: every one of the white texels that make up that half is outside the
    // group. A copy cropped to the group would have nothing but black to read,
    // and would leave the column black.
    let frame = render_frame(window(), |_, window, _| {
        window.paint_quad(gpui::fill(rect(0., 0., 120., 120.), white()));
        window.with_isolated_group(
            GroupOptions {
                bounds: rect(120., 30., 80., 60.),
                backdrop_filter: Some(SceneFilter::Blur {
                    radius_x: SIGMA,
                    radius_y: 0.,
                }),
                ..Default::default()
            },
            |_| {},
        );
    });

    let straddling = frame.color_at(at(120., 60.))[0];
    assert!(
        straddling.abs_diff(124) <= WEIGHT_TOLERANCE,
        "the group's first column came out at {straddling}; at 0 the copy of \
         the backdrop stopped where the group did and the white beside it was \
         never read"
    );
    let further = frame.color_at(at(120. + SIGMA, 60.))[0];
    assert!(
        further.abs_diff(40) <= WEIGHT_TOLERANCE,
        "one standard deviation into the group the backdrop came out at {further}"
    );

    // And the filter reaches no further than the group: the same column below
    // the group is the hard edge it always was, and the column beside it is the
    // white it always was.
    frame.assert_painted(
        at(120., 100.),
        BACKGROUND,
        "the same column below the group",
    );
    frame.assert_painted(at(119., 60.), WHITE, "the column beside the group");
}

#[test]
fn a_backdrop_blur_duplicates_the_edge_where_the_copy_runs_out() {
    // The copy can only reach as far as the thing it is a copy of. Against the
    // window's own edge there is nothing further to take, and what the filter
    // effects specification calls for there is edge duplication: the pixels
    // past the edge are not empty, they are merely not in the copy, and reading
    // transparent black there would ring the window's edge with a band that is
    // in nothing behind it.
    //
    // This is the opposite rule from the one a group's own filter follows, and
    // the reason they differ is that a group's rendering really does stop.
    let bounds = rect(0., 30., 80., 60.);
    let frame = render_frame(window(), move |_, window, _| {
        window.paint_quad(gpui::fill(rect(0., 0., 240., 120.), white()));
        window.with_isolated_group(
            GroupOptions {
                bounds,
                backdrop_filter: Some(SceneFilter::Blur {
                    radius_x: SIGMA,
                    radius_y: SIGMA,
                }),
                ..Default::default()
            },
            |_| {},
        );
    });

    for y in [40., 60., 80.] {
        frame.assert_painted(
            at(0., y),
            WHITE,
            "against the window's edge, under a group blurring a white backdrop",
        );
    }
}

// ------------------------------------------------------------ the counters --

/// The process's one logger, so a test can read what the renderer said.
struct CapturedLog(Mutex<Vec<String>>);

impl log::Log for CapturedLog {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        self.0
            .lock()
            .expect("the capturing logger is never poisoned")
            .push(record.args().to_string());
    }

    fn flush(&self) {}
}

fn captured_log() -> &'static CapturedLog {
    static LOG: std::sync::OnceLock<&'static CapturedLog> = std::sync::OnceLock::new();
    LOG.get_or_init(|| {
        let logger: &'static CapturedLog = Box::leak(Box::new(CapturedLog(Mutex::new(Vec::new()))));
        log::set_logger(logger).expect("nothing else in this binary installs a logger");
        log::set_max_level(log::LevelFilter::Trace);
        logger
    })
}

fn lines_since(clear: impl FnOnce()) -> Vec<String> {
    let log = captured_log();
    log.0
        .lock()
        .expect("the capturing logger is never poisoned")
        .clear();
    clear();
    log.0
        .lock()
        .expect("the capturing logger is never poisoned")
        .clone()
}

#[test]
fn a_blur_wider_than_one_pass_can_read_says_that_it_was_approximated() {
    // A standard deviation of forty logical pixels is eighty device pixels, and
    // its tail is two hundred and forty texels - more than one pass reads. The
    // pass keeps the support and widens its step, which is a coarser sum of the
    // same integral rather than a shorter one, and it is still an
    // approximation. The rule this file is written to is that an approximation
    // is counted out loud.
    let lines = lines_since(|| {
        blurred_square(40., 40.);
    });
    assert!(
        lines.iter().any(|line| line.contains("integrated every")),
        "a blur too wide to read every texel of said nothing about it: {lines:?}"
    );

    // And a blur that fits says nothing, so the line above is about the width
    // rather than about blurring at all.
    let lines = lines_since(|| {
        blurred_square(SIGMA, SIGMA);
    });
    assert!(
        !lines.iter().any(|line| line.contains("integrated every")),
        "a blur narrow enough to read every texel reported an approximation it \
         did not make: {lines:?}"
    );
}
