//! What an isolated group does to pixels: the subtree painted inside it is
//! drawn to a target of its own and composited once.
//!
//! Every test here is paired with a control that fails when nothing is
//! isolated. The load-bearing one is the first: group opacity and
//! [`gpui::Window::with_element_opacity`] agree everywhere except where two
//! children overlap, so the overlap is the only place a pixel can say which of
//! the two actually happened.

#![cfg(target_os = "macos")]

mod harness;

use std::sync::{Arc, Mutex};

use gpui::{
    App, BlendMode, Bounds, BoxShadow, ClipPath, ComposeMode, Corners, GroupOptions, MixMode, Path,
    Pixels, RenderImage, SceneFilter, Size, TextAlign, TextRun, UnderlineStyle, Window, font,
    point, px, red, size, white,
};
use harness::{
    GRAY_RED, GRAYSCALE, HALF_WHITE_ON_NOTHING, RenderedFrame, THREE_QUARTER_WHITE_ON_NOTHING,
    TRANSPARENT, WHITE, at, lines_since, rect, render_frame, render_frame_on_transparent,
    white_surface,
};
use image::{Frame, RgbaImage};

/// Records what `paint` left in the scene, so a test can say whether the group
/// it painted was composited or folded away.
///
/// Two groups that produce the same pixels are the whole point of folding, so
/// no assertion about pixels can tell which of the two happened: a test whose
/// subject is compositing has to ask.
fn isolated_groups(recorded: &Arc<Mutex<Vec<usize>>>, window: &Window) {
    let mut recorded = recorded.lock().expect("the paint callback runs alone");
    recorded.clear();
    recorded.push(window.isolated_group_count());
}

#[track_caller]
fn assert_isolated(recorded: &Arc<Mutex<Vec<usize>>>, expected: usize, what: &str) {
    let recorded = recorded.lock().expect("the paint callback has finished");
    assert_eq!(
        recorded.as_slice(),
        &[expected],
        "{what}: the scene kept {recorded:?} isolated groups"
    );
}

fn window() -> Size<Pixels> {
    size(px(200.), px(100.))
}

/// A group that only composites its subtree with `compose`.
fn composed(bounds: Bounds<Pixels>, compose: ComposeMode) -> GroupOptions {
    GroupOptions {
        bounds,
        blend: BlendMode {
            mix: MixMode::Normal,
            compose,
        },
        ..Default::default()
    }
}

// ---------------------------------------------------------------- opacity --

/// Two 60x60 quads that share the strip `x in 60..80`.
fn overlapping_quads(window: &mut gpui::Window) {
    window.paint_quad(gpui::fill(rect(20., 20., 60., 60.), white()));
    window.paint_quad(gpui::fill(rect(60., 20., 60., 60.), white()));
}

/// Where the two quads overlap, where only the first covers, and where only
/// the second does.
const OVERLAP: (f32, f32) = (70., 50.);
const FIRST_ONLY: (f32, f32) = (40., 50.);
const SECOND_ONLY: (f32, f32) = (100., 50.);

#[test]
fn a_group_at_half_opacity_halves_the_overlap_too() {
    // The whole point of a group. Two opaque children at 0.5 composite to 0.5
    // everywhere they cover, including where they cover each other, because
    // the group is flattened before the opacity is applied.
    let frame = render_frame_on_transparent(window(), |_, window, _| {
        // Bounds that are not the window's, so the group's target has an
        // origin of its own and the coordinate change is exercised.
        window.with_group_opacity(rect(15., 15., 110., 70.), 0.5, overlapping_quads);
    });

    frame.assert_painted(
        at(FIRST_ONLY.0, FIRST_ONLY.1),
        HALF_WHITE_ON_NOTHING,
        "one child of a 0.5 group",
    );
    frame.assert_painted(
        at(SECOND_ONLY.0, SECOND_ONLY.1),
        HALF_WHITE_ON_NOTHING,
        "the other child of a 0.5 group",
    );
    frame.assert_painted(
        at(OVERLAP.0, OVERLAP.1),
        HALF_WHITE_ON_NOTHING,
        "where the two children of a 0.5 group overlap",
    );
}

#[test]
fn element_opacity_does_not_halve_the_overlap() {
    // The control for the test above: fading each primitive on its own lets
    // them composite against each other, so the overlap comes out at
    // `0.5 + 0.5 * 0.5 = 0.75`. If a group did the same thing, the assertion
    // above would be passing for the wrong reason.
    let frame = render_frame_on_transparent(window(), |_, window, _| {
        window.with_element_opacity(Some(0.5), overlapping_quads);
    });

    frame.assert_painted(
        at(FIRST_ONLY.0, FIRST_ONLY.1),
        HALF_WHITE_ON_NOTHING,
        "one child at 0.5 element opacity",
    );
    frame.assert_painted(
        at(OVERLAP.0, OVERLAP.1),
        THREE_QUARTER_WHITE_ON_NOTHING,
        "where two children at 0.5 element opacity overlap",
    );
}

#[test]
fn nested_groups_each_composite_once() {
    // 0.5 inside 0.5 is 0.25 everywhere the inner group covers, overlap
    // included - not 0.4375, which is what two rounds of per-primitive fading
    // would leave there.
    let frame = render_frame_on_transparent(window(), |_, window, _| {
        window.with_group_opacity(rect(10., 10., 120., 80.), 0.5, |window| {
            window.with_group_opacity(rect(15., 15., 110., 70.), 0.5, overlapping_quads);
        });
    });

    frame.assert_painted(
        at(FIRST_ONLY.0, FIRST_ONLY.1),
        [64, 64, 64, 64],
        "one child two 0.5 groups deep",
    );
    frame.assert_painted(
        at(OVERLAP.0, OVERLAP.1),
        [64, 64, 64, 64],
        "where two children two 0.5 groups deep overlap",
    );
}

#[test]
fn nested_element_opacity_does_not() {
    let frame = render_frame_on_transparent(window(), |_, window, _| {
        window.with_element_opacity(Some(0.5), |window| {
            window.with_element_opacity(Some(0.5), overlapping_quads);
        });
    });

    frame.assert_painted(
        at(FIRST_ONLY.0, FIRST_ONLY.1),
        [64, 64, 64, 64],
        "one child two 0.5 element-opacity layers deep",
    );
    frame.assert_painted(
        at(OVERLAP.0, OVERLAP.1),
        [112, 112, 112, 112],
        "where two children two 0.5 element-opacity layers deep overlap",
    );
}

// ------------------------------------------------------------- compositing --

/// An opaque white bar across the middle of the window, and a group covering
/// its left half only, composited with `compose`.
fn bar_with_group_over_its_left_half(compose: ComposeMode) -> RenderedFrame {
    render_frame_on_transparent(window(), move |_, window, _| {
        window.paint_quad(gpui::fill(rect(20., 20., 160., 60.), white()));
        window.with_isolated_group(composed(rect(20., 20., 80., 60.), compose), |window| {
            window.paint_quad(gpui::fill(rect(20., 20., 80., 60.), white()));
        });
    })
}

#[test]
fn dest_out_erases_what_is_beneath_it() {
    let frame = bar_with_group_over_its_left_half(ComposeMode::DestOut);

    frame.assert_painted(at(60., 50.), TRANSPARENT, "under a DestOut group");
    frame.assert_painted(at(140., 50.), WHITE, "beside a DestOut group");
}

#[test]
fn src_over_paints_where_dest_out_erases() {
    // The control: the same group, composited normally, leaves the bar where
    // it was. Without isolation `DestOut` degrades to exactly this.
    let frame = bar_with_group_over_its_left_half(ComposeMode::SrcOver);

    frame.assert_painted(at(60., 50.), WHITE, "under a source-over group");
    frame.assert_painted(at(140., 50.), WHITE, "beside a source-over group");
}

/// A right triangle over the box `50..150` square: a point is inside it when
/// `x + y <= 200`. A triangle rather than a rectangle because `with_clip_path`
/// narrows the content mask to the path's bounding box, so a rectangular clip
/// is reproduced by the mask alone and says nothing about coverage.
fn triangle_clip() -> ClipPath<Pixels> {
    ClipPath::builder()
        .move_to(at(50., 50.))
        .line_to(at(150., 50.))
        .line_to(at(50., 150.))
        .close()
        .build()
}

/// A white square under a `DestIn` group whose contents cover only the left
/// half of it, optionally inside [`triangle_clip`].
fn square_masked_by_a_group(
    compose: ComposeMode,
    clipped: bool,
    groups: &Arc<Mutex<Vec<usize>>>,
) -> RenderedFrame {
    let recorded = groups.clone();
    render_frame_on_transparent(size(px(200.), px(200.)), move |_, window, _| {
        window.paint_quad(gpui::fill(rect(20., 20., 160., 160.), white()));
        let group = |window: &mut Window| {
            window.with_isolated_group(composed(rect(50., 50., 100., 100.), compose), |window| {
                window.paint_quad(gpui::fill(rect(50., 50., 50., 100.), white()));
            });
        };
        if clipped {
            let clip = triangle_clip();
            window.with_clip_path(&clip, group);
        } else {
            group(window);
        }
        isolated_groups(&recorded, window);
    })
}

fn square_masked_by_a_clipped_group(compose: ComposeMode) -> RenderedFrame {
    square_masked_by_a_group(compose, true, &Arc::new(Mutex::new(Vec::new())))
}

#[test]
fn dest_in_keeps_only_what_the_mask_covers() {
    let frame = square_masked_by_a_clipped_group(ComposeMode::DestIn);

    frame.assert_painted(at(60., 60.), WHITE, "where a DestIn mask is opaque");
    frame.assert_painted(
        at(120., 60.),
        TRANSPARENT,
        "inside a DestIn group but outside its mask",
    );
}

#[test]
fn dest_in_leaves_what_is_outside_its_clip_alone() {
    // (140, 140) is inside the group's bounds and inside the clip path's
    // bounding box, but outside the triangle itself, so the clip's coverage
    // there is zero. `DestIn` has to leave the destination exactly as it found
    // it - which is what emitting `1 - c * (1 - S.a)` buys and what emitting
    // `c * S.a` would get wrong, erasing the square instead.
    let groups = Arc::new(Mutex::new(Vec::new()));
    let frame = square_masked_by_a_group(ComposeMode::DestIn, true, &groups);
    assert_isolated(&groups, 1, "a DestIn group cannot be folded away");

    frame.assert_painted(
        at(140., 140.),
        WHITE,
        "inside a DestIn group's bounds but outside its clip path",
    );

    // The control, at the very same point: the same group with no clip path
    // over it. Its coverage there is one rather than zero and its own alpha is
    // zero, so `DestIn` erases - which is what says the composite really does
    // reach this point, and so that the assertion above is about the clip's
    // coverage rather than about a point no compositing happened at.
    let unclipped = square_masked_by_a_group(ComposeMode::DestIn, false, &groups);
    assert_isolated(&groups, 1, "the unclipped control is a group too");
    unclipped.assert_painted(
        at(140., 140.),
        TRANSPARENT,
        "the same point under an unclipped DestIn group, which has to erase it",
    );
}

// ----------------------------------------------------------------- filters --

fn red_square_in_a_group(filter: Option<SceneFilter>) -> RenderedFrame {
    render_frame(window(), move |_, window, _| {
        window.with_isolated_group(
            GroupOptions {
                bounds: rect(20., 20., 160., 60.),
                filter: filter.clone(),
                ..Default::default()
            },
            |window| {
                window.paint_quad(gpui::fill(rect(20., 20., 160., 60.), red()));
            },
        )
    })
}

#[test]
fn a_colour_matrix_filter_applies_to_the_whole_group() {
    let frame = red_square_in_a_group(Some(SceneFilter::ColorMatrix(GRAYSCALE)));
    frame.assert_painted(at(100., 50.), GRAY_RED, "a red quad in a grayscale group");
}

#[test]
fn the_same_group_without_the_filter_stays_red() {
    let frame = red_square_in_a_group(None);
    frame.assert_painted(
        at(100., 50.),
        [255, 0, 0, 255],
        "a red quad in an unfiltered group",
    );
}

/// `invert(1)` as CSS defines it: every channel taken from one, alpha left
/// alone. Unlike [`GRAYSCALE`] it has a constant column, which is what makes it
/// notice whether the colour it was handed was premultiplied.
const INVERT: [f32; 20] = [
    -1., 0., 0., 0., 1., //
    0., -1., 0., 0., 1., //
    0., 0., -1., 0., 1., //
    0., 0., 0., 1., 0.,
];

#[test]
fn a_colour_matrix_runs_on_unpremultiplied_colour() {
    // A group's target holds premultiplied colour and a CSS colour matrix is
    // defined on straight colour, so the shader has to divide the alpha out,
    // run the chain, and put it back.
    //
    // Over an opaque quad that round trip is invisible: with an alpha of one it
    // divides and multiplies by one. It is invisible over a translucent one too
    // for any matrix without a constant column, since scaling the input of a
    // linear map scales its output. `invert` on a half-transparent quad is
    // neither: straight, red inverts to cyan and comes back as premultiplied
    // (0, 128, 128, 128); premultiplied, (0.5, 0, 0) inverts to (0.5, 1, 1),
    // which is not a premultiplied colour at all and reads as (128, 255, 255,
    // 128).
    let frame = render_frame_on_transparent(window(), |_, window, _| {
        window.with_isolated_group(
            GroupOptions {
                bounds: rect(20., 20., 160., 60.),
                filter: Some(SceneFilter::ColorMatrix(INVERT)),
                ..Default::default()
            },
            |window| {
                window.paint_quad(gpui::fill(rect(20., 20., 160., 60.), red().opacity(0.5)));
            },
        );
    });

    frame.assert_painted(
        at(100., 50.),
        [0, 128, 128, 128],
        "a half-transparent red quad in an inverting group",
    );
}

// --------------------------------------------------------------- fallbacks --

/// A bar, and a `DestOut` group over its left half, both painted `depth`
/// isolated groups deep.
///
/// Every enclosing group survives into the scene: one of them has a child that
/// survives, and that is enough to keep the whole chain from folding away.
fn dest_out_nested(depth: usize) -> RenderedFrame {
    render_frame_on_transparent(window(), move |_, window, _| {
        fn inner(window: &mut gpui::Window, remaining: usize) {
            if remaining > 0 {
                window.with_isolated_group(
                    GroupOptions {
                        bounds: rect(0., 0., 200., 100.),
                        ..Default::default()
                    },
                    |window| inner(window, remaining - 1),
                );
                return;
            }
            window.paint_quad(gpui::fill(rect(20., 20., 160., 60.), white()));
            window.with_isolated_group(
                composed(rect(20., 20., 80., 60.), ComposeMode::DestOut),
                |window| {
                    window.paint_quad(gpui::fill(rect(20., 20., 80., 60.), white()));
                },
            );
        }
        inner(window, depth);
    })
}

#[test]
fn a_group_at_the_depth_limit_still_composites() {
    // Seven enclosing groups put the `DestOut` group at depth seven, the last
    // depth that gets a target of its own.
    let frame = dest_out_nested(7);
    frame.assert_painted(at(60., 50.), TRANSPARENT, "a DestOut group at depth 7");
    frame.assert_painted(at(140., 50.), WHITE, "beside it");
}

#[test]
fn a_group_past_the_depth_limit_is_painted_without_isolation() {
    // One deeper, and there is no target left to give it: it is painted
    // straight onto what is underneath, which for `DestOut` means painting
    // where it should have erased. The renderer says so once per frame; this
    // is the picture that goes with the log line.
    let frame = dest_out_nested(8);
    frame.assert_painted(
        at(60., 50.),
        WHITE,
        "a DestOut group past the depth limit, painted inline",
    );
    frame.assert_painted(at(140., 50.), WHITE, "beside it");
}

// ------------------------------------------------------- offset coordinates --

/// The window the coordinate tests render into: taller than [`window`], to
/// leave room for a subtree with one of everything in it.
fn tall_window() -> Size<Pixels> {
    size(px(200.), px(160.))
}

/// The bounds the group claims: not the window's, so its target has an origin
/// of its own and every stage inside it is shading a fragment whose framebuffer
/// coordinates no longer agree with the scene position it is evaluating masks,
/// radii, gaussians and brush matrices against.
fn offset_group() -> Bounds<Pixels> {
    rect(30., 20., 150., 120.)
}

/// A sixteen-texel red image, in the straight BGRA the sprite atlas holds.
fn red_image() -> Arc<RenderImage> {
    Arc::new(RenderImage::new([Frame::new(RgbaImage::from_pixel(
        16,
        16,
        image::Rgba([0, 0, 255, 255]),
    ))]))
}

/// One primitive of every kind that can land inside a group, all inside
/// [`offset_group`].
///
/// Each kind reaches the GPU through a pipeline of its own, and each pipeline
/// has to be told where the target it is drawing into sits in the window. A
/// path is the riskiest of them: it is rasterized into a window-sized
/// intermediate, in an encoder that has to be stopped and started again, and
/// then copied out by a sprite stage that has to find it in there afterwards.
fn one_of_everything(_: Bounds<Pixels>, window: &mut Window, cx: &mut App) {
    window.paint_drop_shadows(
        rect(70., 40., 60., 20.),
        Corners::all(px(8.)),
        &[BoxShadow {
            color: white(),
            offset: point(px(0.), px(0.)),
            blur_radius: px(5.),
            spread_radius: px(0.),
            inset: false,
        }],
    );
    window.paint_quad(gpui::PaintQuad {
        bounds: rect(70., 40., 60., 20.),
        corner_radii: Corners::all(px(8.)).map(|radius| *radius),
        background: white().into(),
        border_widths: gpui::Edges::default(),
        border_color: white(),
        border_style: gpui::BorderStyle::default(),
    });

    // A triangle, so the claim covers the rasterizer's coverage and not only
    // the rectangle it was drawn in.
    let mut path = Path::new(at(40., 70.));
    path.line_to(at(90., 70.));
    path.line_to(at(40., 110.));
    window.paint_path(path, white());

    window.paint_underline(
        at(100., 80.),
        px(60.),
        &UnderlineStyle {
            thickness: px(2.),
            color: Some(white()),
            wavy: false,
        },
    );

    window
        .paint_image(
            rect(120., 90., 40., 40.),
            rect(120., 90., 40., 40.),
            Corners::default(),
            red_image(),
            0,
            false,
        )
        .expect("the image goes into the atlas");

    window.paint_surface(rect(40., 115., 30., 20.), white_surface(32));

    let text = "████";
    let run = TextRun {
        len: text.len(),
        font: font("Menlo"),
        color: white(),
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    window
        .text_system()
        .shape_line(text.into(), px(20.), &[run], None)
        .paint(at(40., 25.), px(20.), TextAlign::Left, None, window, cx)
        .expect("failed to paint the text");
}

#[test]
fn a_group_with_an_origin_of_its_own_paints_where_it_would_have_anyway() {
    // A group target is only as large as the group, so it starts at an offset
    // in the window and every fragment stage inside it is evaluating its masks,
    // corner radii, gaussians and brush matrices against a position the
    // framebuffer no longer agrees with. If any vertex stage forgot to carry
    // the scene position across, this picture moves or deforms.
    let groups = Arc::new(Mutex::new(Vec::new()));
    let recorded = groups.clone();
    let grouped = render_frame(tall_window(), move |bounds, window, cx| {
        window.with_isolated_group(
            GroupOptions {
                bounds: offset_group(),
                // Not a fade, and not a blend: the two frames have to be
                // comparable pixel for pixel, so the only thing this group does
                // is be a group.
                ..Default::default()
            },
            |window| one_of_everything(bounds, window, cx),
        );
        isolated_groups(&recorded, window);
    });
    let ungrouped = render_frame(tall_window(), one_of_everything);

    // Without this the comparison is satisfied by construction: a group that
    // folded away *is* the ungrouped frame, and every pixel would agree because
    // nothing was composited at all.
    assert_isolated(
        &groups,
        1,
        "the shadow's tail overlaps the quad it is behind, so this group cannot \
         be folded away",
    );

    let (mut compared, mut differing) = (0usize, Vec::new());
    for y in 0..grouped.image().height() {
        for x in 0..grouped.image().width() {
            let (a, b) = (
                grouped.image().get_pixel(x, y).0,
                ungrouped.image().get_pixel(x, y).0,
            );
            compared += 1;
            if a.iter().zip(b.iter()).any(|(a, b)| a.abs_diff(*b) > 2) {
                differing.push(((x, y), a, b));
            }
        }
    }
    assert!(compared > 0);
    assert!(
        differing.is_empty(),
        "{} of {compared} pixels differ between a group with an origin of its \
         own and the same content painted straight onto the window; first few: {:?}",
        differing.len(),
        &differing[..differing.len().min(5)],
    );

    // And every kind of primitive really was painted, or the comparison above
    // is a comparison of two blank pictures.
    for (region, what) in [
        (rect(70., 40., 60., 20.), "the rounded quad and its shadow"),
        (rect(45., 75., 20., 20.), "the path"),
        (rect(100., 79., 60., 4.), "the underline"),
        (rect(125., 95., 30., 30.), "the image"),
        (rect(45., 120., 20., 10.), "the surface"),
        (rect(40., 10., 60., 20.), "the text"),
    ] {
        ungrouped.assert_region_painted(region, what);
        grouped.assert_region_painted(region, what);
    }
}

// ------------------------------------------------------------- group bounds --

/// Two 60x60 quads, the first inside the bounds every test below gives its
/// group and the second reaching well outside them.
fn one_quad_inside_the_bounds_and_one_outside(window: &mut gpui::Window) {
    window.paint_quad(gpui::fill(rect(20., 20., 60., 60.), white()));
    window.paint_quad(gpui::fill(rect(60., 20., 100., 60.), white()));
}

/// The same two quads, moved apart so that nothing overlaps and the group folds.
fn two_disjoint_quads_one_outside_the_bounds(window: &mut gpui::Window) {
    window.paint_quad(gpui::fill(rect(20., 20., 40., 60.), white()));
    window.paint_quad(gpui::fill(rect(100., 20., 60., 60.), white()));
}

/// Bounds that hold the first quad and cut the second in half.
fn tight_bounds() -> Bounds<Pixels> {
    rect(15., 15., 70., 70.)
}

/// A point inside the second quad and well outside [`tight_bounds`].
const OUTSIDE_THE_BOUNDS: (f32, f32) = (140., 50.);

#[test]
fn a_group_does_not_clip_what_is_painted_outside_its_bounds() {
    // A group's bounds size its target; they are not a clip. CSS `opacity` does
    // not clip, and a group that folds away has no target to clip to - so a
    // group that clipped would paint one picture or the other depending on
    // whether its contents happened to overlap, which is a heuristic no
    // document can see.
    let groups = Arc::new(Mutex::new(Vec::new()));
    let recorded = groups.clone();
    let composited = render_frame_on_transparent(window(), move |_, window, _| {
        window.with_group_opacity(
            tight_bounds(),
            0.5,
            one_quad_inside_the_bounds_and_one_outside,
        );
        isolated_groups(&recorded, window);
    });
    assert_isolated(
        &groups,
        1,
        "the two quads overlap, so this group has to be composited",
    );

    composited.assert_painted(
        at(FIRST_ONLY.0, FIRST_ONLY.1),
        HALF_WHITE_ON_NOTHING,
        "the part of a composited group that is inside its bounds",
    );
    composited.assert_painted(
        at(OUTSIDE_THE_BOUNDS.0, OUTSIDE_THE_BOUNDS.1),
        HALF_WHITE_ON_NOTHING,
        "the part of a composited group that reaches outside the bounds it was \
         given, which is not a clip",
    );
    composited.assert_painted(
        at(OVERLAP.0, OVERLAP.1),
        HALF_WHITE_ON_NOTHING,
        "and it really was composited: the overlap is at 0.5 rather than at the \
         0.75 two separately faded quads leave there",
    );

    // The same claim on the other side of the heuristic: the same bounds, the
    // same overhang, contents that do not overlap, and so a group that folds
    // into its primitives instead of being composited. The two have to agree.
    let recorded = groups.clone();
    let folded = render_frame_on_transparent(window(), move |_, window, _| {
        window.with_group_opacity(
            tight_bounds(),
            0.5,
            two_disjoint_quads_one_outside_the_bounds,
        );
        isolated_groups(&recorded, window);
    });
    assert_isolated(&groups, 0, "quads that do not overlap fold away");
    folded.assert_painted(
        at(OUTSIDE_THE_BOUNDS.0, OUTSIDE_THE_BOUNDS.1),
        HALF_WHITE_ON_NOTHING,
        "the part of a folded group that reaches outside its bounds",
    );
}

#[test]
fn a_group_whose_opacity_is_not_a_number_is_composited_at_full_strength() {
    // A NaN is not something a document should produce, but `f32::clamp` hands
    // one straight back and a NaN coverage multiplies every channel of the
    // group into a NaN. Full strength is what the scene already does with one -
    // `opacity < 1.` is false for a NaN, so folding leaves the primitives alone
    // - so it is what compositing has to do too.
    let frame = render_frame_on_transparent(window(), |_, window, _| {
        window.with_isolated_group(
            GroupOptions {
                bounds: rect(15., 15., 110., 70.),
                opacity: f32::NAN,
                ..Default::default()
            },
            overlapping_quads,
        );
    });

    frame.assert_painted(
        at(OVERLAP.0, OVERLAP.1),
        WHITE,
        "where the two children of a group with a NaN opacity overlap",
    );
    frame.assert_painted(
        at(FIRST_ONLY.0, FIRST_ONLY.1),
        WHITE,
        "one child of a group with a NaN opacity",
    );
}

// ----------------------------------------------------------- what is dropped --

#[test]
fn a_backdrop_filter_is_applied_rather_than_dropped_with_a_line_in_the_log() {
    // `GroupOptions::backdrop_filter` used to be recorded by the scene and
    // applied by nobody, and the only promise `with_isolated_group` could keep
    // was that the renderer said so. It is applied now, so the promise to keep
    // is the opposite one: the filter reaches the pixels *and* nothing is
    // logged, because a line in the log now would mean something else had gone
    // wrong.
    let (frame, lines) = lines_since(|| {
        render_frame(window(), |_, window, _| {
            window.paint_quad(gpui::fill(rect(0., 0., 200., 100.), red()));
            window.with_isolated_group(
                GroupOptions {
                    bounds: rect(20., 20., 160., 60.),
                    backdrop_filter: Some(SceneFilter::ColorMatrix(GRAYSCALE)),
                    ..Default::default()
                },
                |_| {},
            );
        })
    });

    assert!(
        !lines.iter().any(|line| line.contains("backdrop")),
        "the renderer said something about the backdrop filter it was supposed \
         to have applied: {lines:?}"
    );

    frame.assert_painted(
        at(100., 50.),
        GRAY_RED,
        "the red under a group whose backdrop filter greys it",
    );
    frame.assert_painted(
        at(10., 50.),
        [255, 0, 0, 255],
        "the red beside the group, which the filter does not reach",
    );
}
