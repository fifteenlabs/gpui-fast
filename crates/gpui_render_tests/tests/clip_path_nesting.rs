//! What happens when clip paths meet each other, a content mask, or unclipped
//! content in the same frame.
//!
//! A clip is a tile of a coverage atlas, and the atlas is where a whole class
//! of bugs lives that a single clip cannot show: a nested clip reading the
//! wrong parent tile, two clips whose tiles collide, an unclipped batch left
//! holding the previous batch's binding. Each test here pairs its claim with a
//! control that only differs by the thing under test.
//!
//! Nothing here nests a *rectangle* inside anything. `with_clip_path` narrows
//! the content mask to the path's bounding box, so for a rectangular path the
//! mask alone reproduces the picture and a test built out of rectangles passes
//! against a renderer with no mask rasterizer in it at all. Every clip below is
//! notched, wedged or triangular, and every assertion is one the bounding boxes
//! cannot make true.

#![cfg(target_os = "macos")]

mod harness;

use gpui::{ClipPath, ContentMask, Corners, FillRule, Pixels, Size, px, size, white};
use harness::{RenderedFrame, WHITE, at, rect, render_frame, render_two_frames};

/// The window every test in this file renders, in logical pixels.
const WINDOW: f32 = 200.;

fn window() -> Size<Pixels> {
    size(px(WINDOW), px(WINDOW))
}

/// A rectangular clip path, for the one place a rectangle is the point: a
/// window onto a *parent* whose shape varies underneath it.
fn box_clip(x: f32, y: f32, width: f32, height: f32) -> ClipPath<Pixels> {
    let bounds = rect(x, y, width, height);
    ClipPath::builder()
        .move_to(bounds.origin)
        .line_to(bounds.top_right())
        .line_to(bounds.bottom_right())
        .line_to(bounds.bottom_left())
        .close()
        .build()
}

/// A triangle with its right angle at `(x, y)` and legs of `width` and
/// `height`, so a point is inside when it is above the hypotenuse.
fn corner_triangle(x: f32, y: f32, width: f32, height: f32) -> ClipPath<Pixels> {
    ClipPath::builder()
        .move_to(at(x, y))
        .line_to(at(x + width, y))
        .line_to(at(x, y + height))
        .close()
        .build()
}

/// The square (50, 50) to (150, 150) with the box (90, 80) to (150, 120) bitten
/// out of its right-hand side.
///
/// A point is inside when `50 <= x <= 150` and `50 <= y <= 150` and it is not
/// in the notch. The notch is the whole reason this shape is here: it is a
/// place *inside* the bounding box where the clip lets nothing through, so a
/// child that samples its parent wrongly - or not at all - shows it.
fn notched_column() -> ClipPath<Pixels> {
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

/// A wedge opening to the bottom right: `(30, 60)`, `(170, 60)`, `(170, 200)`.
/// A point is inside when `y >= 60`, `x <= 170` and `x > y - 30`.
fn wedge() -> ClipPath<Pixels> {
    ClipPath::builder()
        .move_to(at(30., 60.))
        .line_to(at(170., 60.))
        .line_to(at(170., 200.))
        .close()
        .build()
}

/// Paints the whole window white inside every clip in `clips`, each pushed
/// inside the last.
fn render_inside_nested_clips(clips: Vec<ClipPath<Pixels>>) -> RenderedFrame {
    render_frame(window(), move |bounds, window, _| {
        paint_inside(&clips, bounds, window);
    })
}

fn paint_inside(
    clips: &[ClipPath<Pixels>],
    bounds: gpui::Bounds<Pixels>,
    window: &mut gpui::Window,
) {
    match clips.split_first() {
        None => window.paint_quad(gpui::fill(bounds, white())),
        Some((first, rest)) => {
            window.with_clip_path(first, |window| paint_inside(rest, bounds, window))
        }
    }
}

/// Inside both the notched column and the wedge.
const INSIDE_BOTH: [(f32, f32); 7] = [
    (120., 70.),
    (145., 70.),
    (140., 140.),
    (88., 100.),
    (120., 78.),
    (120., 122.),
    (114., 140.),
];

/// Inside the wedge, but in the notch the column bites out of itself. Every one
/// of these is inside the *bounding box* of both clips, so nothing but the
/// column's real shape can cut them.
const IN_THE_NOTCH: [(f32, f32); 5] = [
    (92., 100.),
    (120., 100.),
    (145., 100.),
    (120., 82.),
    (120., 118.),
];

/// Inside the column, but outside the wedge's hypotenuse or above its top edge.
const OUTSIDE_THE_WEDGE: [(f32, f32); 4] = [(106., 140.), (60., 55.), (55., 140.), (60., 100.)];

fn points(from: &[(f32, f32)]) -> Vec<gpui::Point<Pixels>> {
    from.iter().map(|&(x, y)| at(x, y)).collect()
}

#[test]
fn nesting_two_clip_paths_intersects_their_shapes() {
    let both = render_inside_nested_clips(vec![notched_column(), wedge()]);

    both.assert_all_painted(
        &points(&INSIDE_BOTH),
        WHITE,
        "inside both a notched clip path and the wedge nested in it",
    );
    both.assert_all_clipped_away(
        &points(&IN_THE_NOTCH),
        "the notch the outer clip path bites out from under the inner one",
    );
    both.assert_all_clipped_away(
        &points(&OUTSIDE_THE_WEDGE),
        "the part of the outer clip path the inner one cuts",
    );
    both.assert_all_clipped_away(
        &points(&[(25., 25.), (180., 180.), (40., 160.), (170., 40.)]),
        "outside both of two nested clip paths",
    );
}

#[test]
fn each_of_those_clips_alone_paints_strictly_more() {
    // Without these the test above would pass against a renderer that clipped
    // to nothing at all, or that ignored one of the two clips entirely.
    let column = render_inside_nested_clips(vec![notched_column()]);
    column.assert_all_painted(
        &points(&OUTSIDE_THE_WEDGE),
        WHITE,
        "the parts of the notched column only the wedge cuts",
    );
    column.assert_all_clipped_away(
        &points(&IN_THE_NOTCH),
        "the notch, which the column cuts on its own",
    );

    let wedge_alone = render_inside_nested_clips(vec![wedge()]);
    wedge_alone.assert_all_painted(
        &points(&IN_THE_NOTCH),
        WHITE,
        "the notch, which only the column cuts",
    );
    wedge_alone.assert_all_clipped_away(
        &points(&OUTSIDE_THE_WEDGE),
        "the parts the wedge cuts on its own",
    );
    wedge_alone.assert_all_painted(
        &points(&INSIDE_BOTH),
        WHITE,
        "the wedge on its own keeps everything the pair keeps",
    );
}

#[test]
fn a_nested_clip_samples_its_parent_where_the_parent_actually_is() {
    // The hardest arithmetic in the atlas: the child's cover pass reads the
    // parent's coverage while writing its own tile, and the two tiles are in
    // different places - different nesting levels are packed into bands of
    // their own, tens of pixels apart. Every probe below is two logical pixels
    // from an edge of the *parent's* notch, seen through a small child window
    // that has no such edge of its own, so an offset that is wrong by more than
    // that moves the notch and the picture changes.
    let window_onto_the_notch = box_clip(85., 70., 45., 60.);
    let frame = render_inside_nested_clips(vec![notched_column(), window_onto_the_notch.clone()]);

    frame.assert_all_painted(
        &points(&[(88., 100.), (120., 78.), (120., 122.), (128., 75.)]),
        WHITE,
        "two logical pixels outside the parent's notch, seen through a nested clip",
    );
    frame.assert_all_clipped_away(
        &points(&[(92., 100.), (120., 82.), (120., 118.), (128., 100.)]),
        "two logical pixels inside the parent's notch, seen through a nested clip",
    );
    // The child's own rectangle is still enforced, so the test cannot pass by
    // the child having been dropped and the parent drawn alone.
    frame.assert_all_clipped_away(
        &points(&[(60., 100.), (140., 75.), (100., 60.), (100., 140.)]),
        "outside the nested clip's own window onto its parent",
    );

    // The control: the same window with no parent under it is white
    // throughout, so every "clipped away" above is the parent's shape and
    // nothing else.
    let alone = render_inside_nested_clips(vec![window_onto_the_notch]);
    alone.assert_all_painted(
        &points(&[(92., 100.), (120., 82.), (120., 118.), (128., 100.)]),
        WHITE,
        "the same points with no parent clip, which is what makes the above a test",
    );
}

#[test]
fn three_clip_paths_nest_down_to_their_common_part() {
    // The third is a triangle across the top left: inside when x + y < 200.
    let upper_left = ClipPath::builder()
        .move_to(at(40., 40.))
        .line_to(at(160., 40.))
        .line_to(at(40., 160.))
        .close()
        .build();
    let frame = render_inside_nested_clips(vec![notched_column(), wedge(), upper_left]);

    frame.assert_all_painted(
        &points(&[(120., 70.), (88., 100.), (120., 78.)]),
        WHITE,
        "the part three nested clip paths share",
    );
    frame.assert_all_clipped_away(
        &points(&[(140., 140.), (120., 122.), (114., 140.)]),
        "the part only the third of three nested clip paths cuts",
    );
    frame.assert_all_clipped_away(
        &points(&IN_THE_NOTCH),
        "the notch, which the first of three nested clip paths cuts",
    );
    frame.assert_all_clipped_away(
        &points(&OUTSIDE_THE_WEDGE),
        "the part the second of three nested clip paths cuts",
    );

    // The control: without the third, the points it cuts are painted.
    let two = render_inside_nested_clips(vec![notched_column(), wedge()]);
    two.assert_all_painted(
        &points(&[(140., 140.), (120., 122.), (114., 140.)]),
        WHITE,
        "the points the third clip path cuts, with only the first two pushed",
    );
}

#[test]
fn a_clip_path_inside_a_rounded_content_mask_takes_both() {
    // The clip is a triangle: inside when x + y < 200. The mask is a square
    // with 30px circular corners, which is a shape the clip path here does not
    // have and the mask does.
    let triangle = corner_triangle(50., 50., 100., 100.);
    let frame = render_frame(window(), move |bounds, window, _| {
        window.with_content_mask(
            Some(ContentMask {
                bounds: rect(50., 50., 100., 100.),
                corner_radii: Corners::all(px(30.)),
            }),
            |window| {
                window.with_clip_path(&triangle, |window| {
                    window.paint_quad(gpui::fill(bounds, white()));
                })
            },
        );
    });

    frame.assert_all_painted(
        &points(&[(70., 70.), (100., 60.), (60., 100.), (90., 90.)]),
        WHITE,
        "inside both a rounded content mask and a triangular clip path",
    );
    // Cut by the mask's 30px corner, inside the clip path's triangle: the arc
    // is centred on (80, 80) and (55, 55) is 35 away from it.
    frame.assert_clipped_away(
        at(55., 55.),
        "the corner the content mask cuts from a clipped quad",
    );
    // Cut by the triangle, well inside the mask.
    frame.assert_all_clipped_away(
        &points(&[(130., 130.), (140., 100.), (100., 140.)]),
        "the part of a rounded content mask the clip path's hypotenuse cuts",
    );
}

#[test]
fn a_rounded_content_mask_inside_a_clip_path_takes_both() {
    // The other nesting order. The mask is intersected into the clip's own
    // tightened mask rather than the other way round, which is a different
    // code path through `with_content_mask`.
    let triangle = corner_triangle(50., 50., 100., 100.);
    let frame = render_frame(window(), move |bounds, window, _| {
        window.with_clip_path(&triangle, |window| {
            window.with_content_mask(
                Some(ContentMask {
                    bounds: rect(50., 50., 100., 100.),
                    corner_radii: Corners::all(px(30.)),
                }),
                |window| window.paint_quad(gpui::fill(bounds, white())),
            )
        });
    });

    frame.assert_all_painted(
        &points(&[(70., 70.), (100., 60.), (60., 100.), (90., 90.)]),
        WHITE,
        "inside both a triangular clip path and a rounded content mask",
    );
    frame.assert_clipped_away(
        at(55., 55.),
        "the corner the content mask cuts, with the clip path outside it",
    );
    frame.assert_all_clipped_away(
        &points(&[(130., 130.), (140., 100.), (100., 140.)]),
        "the part the clip path cuts, with the content mask inside it",
    );
}

#[test]
fn two_clip_paths_at_the_same_depth_do_not_borrow_each_others_tiles() {
    // Both clips are pushed at the root, so both tiles are written in one pass,
    // and their bounding boxes overlap on screen while their shapes do not. If
    // the atlas handed them the same texels, or if the second read the first's,
    // the overlap would come out painted.
    let upper_left = corner_triangle(20., 20., 100., 100.);
    let lower_right = ClipPath::builder()
        .move_to(at(180., 180.))
        .line_to(at(80., 180.))
        .line_to(at(180., 80.))
        .close()
        .build();
    let frame = render_frame(window(), move |_, window, _| {
        window.with_clip_path(&upper_left, |window| {
            window.paint_quad(gpui::fill(rect(0., 0., WINDOW, WINDOW), white()));
        });
        window.with_clip_path(&lower_right, |window| {
            window.paint_quad(gpui::fill(rect(0., 0., WINDOW, WINDOW), white()));
        });
    });

    frame.assert_all_painted(
        &points(&[(40., 40.), (30., 80.), (160., 160.), (170., 120.)]),
        WHITE,
        "two clip paths at the same depth, whose bounding boxes overlap on screen",
    );
    // Inside both bounding boxes and inside neither shape: the whole overlap.
    frame.assert_region_clipped_away(
        rect(96., 96., 8., 8.),
        "the middle both of two same-depth clip paths exclude",
    );
    frame.assert_all_clipped_away(
        &points(&[(115., 30.), (30., 115.), (85., 170.), (170., 85.)]),
        "inside one same-depth clip path's bounding box but outside its shape",
    );
    frame.assert_all_clipped_away(
        &points(&[(10., 10.), (190., 190.)]),
        "outside both of two same-depth clip paths",
    );
}

#[test]
fn unclipped_content_between_two_clipped_draws_is_untouched() {
    // Clipped and unclipped quads alternate, so the batches alternate too. A
    // renderer that left the mask bound across the split, or that failed to
    // split at all, would clip the middle quad to the first quad's path.
    let frame = render_frame(window(), |_, window, _| {
        window.with_clip_path(&corner_triangle(10., 10., 50., 50.), |window| {
            window.paint_quad(gpui::fill(rect(0., 0., 70., 70.), white()));
        });
        window.paint_quad(gpui::fill(rect(70., 10., 120., 50.), white()));
        window.with_clip_path(&corner_triangle(10., 130., 50., 50.), |window| {
            window.paint_quad(gpui::fill(rect(0., 120., 70., 70.), white()));
        });
        window.paint_quad(gpui::fill(rect(70., 140., 120., 50.), white()));
    });

    frame.assert_all_painted(
        &points(&[(20., 20.), (50., 15.), (20., 140.), (50., 135.)]),
        WHITE,
        "the clipped quads of an alternating frame",
    );
    frame.assert_all_clipped_away(
        &points(&[(50., 50.), (55., 40.), (50., 170.), (55., 160.), (65., 65.)]),
        "the parts of the clipped quads their triangles cut away",
    );
    // The unclipped quads reach every corner of their own rectangles: nothing
    // from either clip may touch them.
    frame.assert_all_painted(
        &points(&[
            (75., 15.),
            (185., 15.),
            (75., 55.),
            (185., 55.),
            (130., 35.),
            (75., 145.),
            (185., 185.),
            (130., 165.),
        ]),
        WHITE,
        "the unclipped quads painted between two clipped ones",
    );
}

#[test]
fn a_clip_path_nested_past_the_depth_limit_falls_back_to_its_rectangle() {
    // Nine levels: the renderer rasterizes masks for eight and clips the ninth
    // to its own rectangle, which is already intersected with everything
    // outside it. The shape is lost, the bound is not, and it logs.
    let mut clips = Vec::new();
    for level in 0..9 {
        let inset = level as f32 * 2.;
        clips.push(box_clip(40. + inset, 40. + inset, 120., 120.));
    }
    // The ninth is a triangle, so a rasterized mask and a rectangle fallback
    // disagree about its bottom-left corner.
    clips.push(
        ClipPath::builder()
            .move_to(at(60., 60.))
            .line_to(at(150., 60.))
            .line_to(at(150., 150.))
            .close()
            .build(),
    );
    let frame = render_inside_nested_clips(clips.clone());

    frame.assert_all_painted(
        &points(&[(145., 140.), (120., 100.), (145., 70.)]),
        WHITE,
        "inside a clip path past the nesting depth limit",
    );
    // Inside the triangle's bounding rectangle but outside the triangle: this
    // is the shape the fallback gives up.
    frame.assert_painted(
        at(70., 140.),
        WHITE,
        "the corner a depth-limited clip path keeps because it fell back to its rectangle",
    );
    // The rectangle is still enforced, and so is every clip above it.
    frame.assert_all_clipped_away(
        &points(&[(50., 100.), (100., 50.), (170., 100.), (100., 170.)]),
        "outside the rectangle a depth-limited clip path fell back to",
    );

    // The control: the same triangle, two levels shallower so it lands inside
    // the limit, gets a real mask and the corner above is cut.
    let shallow = render_inside_nested_clips(clips[2..].to_vec());
    shallow.assert_clipped_away(
        at(70., 140.),
        "the corner a clip path inside the depth limit cuts, which the fallback keeps",
    );
    shallow.assert_painted(
        at(145., 140.),
        WHITE,
        "inside a triangular clip path within the depth limit",
    );
}

#[test]
fn two_clip_paths_in_one_batch_are_told_apart_by_their_ids() {
    // The two quads are disjoint, so the bounds tree gives them the same draw
    // order and they are drawn as one batch. A batch splits on clipped versus
    // unclipped and nothing finer, so this is the case where one draw call has
    // to distinguish two clips out of the primitive records themselves: a
    // renderer that read the clip off the batch would cut the second quad away
    // entirely, since it lies outside the first quad's clip.
    let frame = render_frame(window(), |_, window, _| {
        window.with_clip_path(&corner_triangle(20., 20., 50., 50.), |window| {
            window.paint_quad(gpui::fill(rect(10., 10., 80., 80.), white()));
        });
        window.with_clip_path(&corner_triangle(130., 130., 50., 50.), |window| {
            window.paint_quad(gpui::fill(rect(120., 120., 80., 80.), white()));
        });
    });

    frame.assert_all_painted(
        &points(&[(30., 30.), (60., 25.), (140., 140.), (170., 135.)]),
        WHITE,
        "two quads in one batch, each clipped by its own triangle",
    );
    frame.assert_all_clipped_away(
        &points(&[(60., 60.), (170., 170.), (15., 15.), (185., 185.)]),
        "the parts each of two same-batch clip paths cuts from its own quad",
    );
}

#[test]
fn an_even_odd_clip_keeps_its_hole_under_a_nested_clip() {
    // The fill rule has to survive nesting: the outer clip is a ring - two
    // same-wound contours, which even-odd leaves hollow - and the inner clip is
    // a triangle across it. Nothing but the outer path's own rule reaching the
    // rasterizer can empty the middle, and nothing but the nesting can cut the
    // rest of the ring.
    let ring = |fill_rule| {
        let outline = |builder: gpui::ClipPathBuilder, bounds: gpui::Bounds<Pixels>| {
            builder
                .move_to(bounds.origin)
                .line_to(bounds.top_right())
                .line_to(bounds.bottom_right())
                .line_to(bounds.bottom_left())
                .close()
        };
        let builder = ClipPath::builder().fill_rule(fill_rule);
        let builder = outline(builder, rect(40., 40., 120., 120.));
        outline(builder, rect(80., 80., 40., 40.)).build()
    };
    let across = corner_triangle(30., 30., 150., 150.);

    let frame = render_inside_nested_clips(vec![ring(FillRule::EvenOdd), across.clone()]);
    frame.assert_region_clipped_away(
        rect(85., 85., 30., 30.),
        "the hole an even-odd outer clip keeps under a nested clip",
    );
    frame.assert_all_painted(
        &points(&[(60., 60.), (130., 45.), (45., 130.), (100., 50.)]),
        WHITE,
        "the ring an even-odd outer clip keeps under a nested clip",
    );
    frame.assert_all_clipped_away(
        &points(&[(140., 140.), (150., 100.), (100., 150.)]),
        "the part of an even-odd ring the nested clip cuts",
    );

    // The control: the same nesting under the nonzero rule fills the middle,
    // so the hole above is the fill rule and not the nesting.
    let filled = render_inside_nested_clips(vec![ring(FillRule::NonZero), across]);
    filled.assert_all_painted(
        &points(&[(100., 90.), (90., 100.), (95., 95.)]),
        WHITE,
        "the middle the nonzero rule keeps, which even-odd empties",
    );
}

#[test]
fn a_clip_atlas_that_has_to_grow_still_clips_every_path() {
    // Thirty clips in one frame, each a triangle in its own cell of a grid.
    // Their tiles do not fit the atlas a single clip gets, so the renderer
    // grows it - and because the first frame gets a small one, the second frame
    // takes the resize path rather than the allocate path, which is what
    // happens in production the first time a document with many clips is
    // painted.
    const COLUMNS: usize = 5;
    const ROWS: usize = 6;
    fn cell(index: usize) -> (f32, f32) {
        let (column, row) = (index % COLUMNS, index / COLUMNS);
        (column as f32 * 40., row as f32 * 33.)
    }

    let [one, many] = render_two_frames(window(), move |frame, _, window, _| {
        let count = if frame == 0 { 1 } else { COLUMNS * ROWS };
        for index in 0..count {
            let (x, y) = cell(index);
            window.with_clip_path(&corner_triangle(x, y, 36., 30.), |window| {
                window.paint_quad(gpui::fill(rect(0., 0., WINDOW, WINDOW), white()));
            });
        }
    });

    let (x, y) = cell(0);
    one.assert_painted(
        at(x + 6., y + 6.),
        WHITE,
        "the single clip of the frame before the atlas grew",
    );

    for index in 0..COLUMNS * ROWS {
        let (x, y) = cell(index);
        many.assert_painted(
            at(x + 6., y + 6.),
            WHITE,
            &format!("inside clip {index} of a frame that made the atlas grow"),
        );
        many.assert_clipped_away(
            at(x + 30., y + 24.),
            &format!("beyond the hypotenuse of clip {index} of a frame that made the atlas grow"),
        );
    }
}

#[test]
fn a_clip_path_reaching_off_screen_clips_what_is_on_screen() {
    // The path runs far past the window on two sides and into negative
    // coordinates, so its bounding box is much larger than the viewport and its
    // origin is outside it. The tile is the intersection, and the shape has to
    // arrive in it at the right offset: a tile taken from the raw box, or an
    // offset taken from it, puts the hypotenuse in the wrong place.
    //
    // The triangle's hypotenuse runs from (300, -100) to (-100, 300), so a
    // point is inside when x + y < 200.
    let frame = render_inside_nested_clips(vec![corner_triangle(-100., -100., 400., 400.)]);

    frame.assert_all_painted(
        &points(&[(5., 5.), (100., 5.), (5., 100.), (50., 50.)]),
        WHITE,
        "the on-screen part of a clip path that starts far off screen",
    );
    frame.assert_all_clipped_away(
        &points(&[(150., 100.), (100., 150.), (195., 195.), (130., 130.)]),
        "the part of an off-screen clip path's shape that falls outside it on screen",
    );

    // A clip entirely off screen paints nothing at all, and does not take the
    // window down with it.
    let gone = render_inside_nested_clips(vec![corner_triangle(-900., -900., 400., 400.)]);
    gone.assert_region_clipped_away(
        rect(0., 0., WINDOW, WINDOW),
        "a clip path entirely off screen",
    );
}
