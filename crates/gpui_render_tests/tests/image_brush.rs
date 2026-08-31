//! Filling a path with an image: `Window::paint_path_with_image`.
//!
//! An image brush is the one thing a solid or gradient `Background` cannot
//! express, and the thing an HTML painter needs most: a `background-image` that
//! repeats is a single tile drawn over and over across whatever shape the box
//! turned out to be. Every test here is written so that a renderer that ignored
//! the brush and fell back to the path's `Background` - which for a brushed path
//! is nothing at all - fails it, and so that a renderer that resolved the brush
//! but got the mapping, the extend mode, the atlas neighbour, the texture
//! binding or the premultiplication wrong fails a different one.
//!
//! The images are authored channel-swapped: the sprite atlas holds BGRA, so
//! `bgra` below is the one place a colour is flipped, and every expectation in
//! the file is written in the RGBA the harness reads back.

#![cfg(target_os = "macos")]

mod harness;

use std::sync::{Arc, Mutex};

use gpui::{
    Bounds, BrushExtend, ContentMask, Corners, MAX_BRUSH_IMAGE_SIZE, Path, Pixels, RenderImage,
    ScaledPixels, TransformationMatrix, point, px, size,
};
use harness::{at, rect, render_frame, render_frame_on_transparent, render_two_frames};
use image::{Frame, Rgba, RgbaImage};

/// The four colours of the test image, in the RGBA the harness reads back.
const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const YELLOW: [u8; 4] = [255, 255, 0, 255];
const CYAN: [u8; 4] = [0, 255, 255, 255];
const MAGENTA: [u8; 4] = [255, 0, 255, 255];

/// The test image's own pixel size.
///
/// Sixteen texels rather than the two the four quadrants strictly need: a
/// bilinear tap reaches one texel either side of where it lands, which on a 2x2
/// image is the entire image, so every sample would be a blend of all four
/// colours and no assertion could name one. At sixteen, only the two texel rows
/// on each quadrant seam blend, and a sample taken near a quadrant's middle is
/// that quadrant's colour outright.
const IMAGE_SIZE: u32 = 16;

/// The atlas holds BGRA and the harness reads RGBA, so a test image is authored
/// with its channels swapped. Doing it here means every expectation in the file
/// can be written the way it will be read.
fn bgra(rgba: [u8; 4]) -> Rgba<u8> {
    Rgba([rgba[2], rgba[1], rgba[0], rgba[3]])
}

/// Four distinct quadrants: red top-left, green top-right, blue bottom-left,
/// yellow bottom-right. A brush that maps its image the wrong way round reads
/// as the wrong colour rather than as a blur.
fn quadrant_image(size: u32) -> Arc<RenderImage> {
    let mut buffer = RgbaImage::new(size, size);
    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        *pixel = bgra(match (x < size / 2, y < size / 2) {
            (true, true) => RED,
            (false, true) => GREEN,
            (true, false) => BLUE,
            (false, false) => YELLOW,
        });
    }
    Arc::new(RenderImage::new([Frame::new(buffer)]))
}

fn solid_image(colour: [u8; 4], size: u32) -> Arc<RenderImage> {
    Arc::new(RenderImage::new([Frame::new(RgbaImage::from_pixel(
        size,
        size,
        bgra(colour),
    ))]))
}

/// A solid image that need not be square, for the size guard.
fn solid_image_of(colour: [u8; 4], width: u32, height: u32) -> Arc<RenderImage> {
    Arc::new(RenderImage::new([Frame::new(RgbaImage::from_pixel(
        width,
        height,
        bgra(colour),
    ))]))
}

/// The brush transform that lays one copy of an `image_size`-pixel image over
/// `bounds`.
///
/// `paint_path_with_image` takes the transform that maps the image's own pixel
/// rectangle into the coordinate space the path is in, so this is the whole of
/// what a `background-image` at a position and a size asks for.
fn brush_over(bounds: Bounds<Pixels>, image_size: u32) -> TransformationMatrix {
    TransformationMatrix::unit()
        .translate(point(
            ScaledPixels(f32::from(bounds.origin.x)),
            ScaledPixels(f32::from(bounds.origin.y)),
        ))
        .scale(size(
            f32::from(bounds.size.width) / image_size as f32,
            f32::from(bounds.size.height) / image_size as f32,
        ))
}

fn rect_path(bounds: Bounds<Pixels>) -> Path<Pixels> {
    let mut path = Path::new(bounds.origin);
    path.line_to(bounds.top_right());
    path.line_to(bounds.bottom_right());
    path.line_to(bounds.bottom_left());
    path
}

fn window() -> gpui::Size<Pixels> {
    size(px(200.), px(160.))
}

/// The shape the mapping tests fill, and the rectangle they lay one copy of the
/// image over.
fn shape() -> Bounds<Pixels> {
    rect(20., 20., 160., 120.)
}

/// The middles of the four quadrants of [`shape`].
fn quadrant_centres() -> [gpui::Point<Pixels>; 4] {
    [at(60., 50.), at(140., 50.), at(60., 110.), at(140., 110.)]
}

#[test]
fn an_image_brush_fills_a_triangle_and_only_the_triangle() {
    // The shape half: a brush is a fill, not a rectangle blit, so the coverage
    // the path rasterizer computed still has to be what decides which pixels
    // get any of the image at all.
    let image = quadrant_image(IMAGE_SIZE);
    let frame = render_frame(window(), move |_, window, _| {
        let mut path = Path::new(at(20., 20.));
        path.line_to(at(180., 20.));
        path.line_to(at(100., 140.));
        window
            .paint_path_with_image(
                path,
                image.clone(),
                0,
                brush_over(shape(), IMAGE_SIZE),
                BrushExtend::Pad,
                BrushExtend::Pad,
                1.,
            )
            .expect("the brush image fits in the atlas");
    });

    // Inside the triangle, and inside the image's top-left quadrant. A renderer
    // that dropped the brush paints the path in its `Background`, which is
    // nothing, and this is the assertion that catches it.
    frame.assert_painted(at(60., 60.), RED, "inside the brushed triangle");
    frame.assert_all_clipped_away(
        &[at(25., 130.), at(175., 130.)],
        "the corners of the brush's rectangle that the triangle does not cover",
    );
}

#[test]
fn the_image_maps_across_the_shape() {
    // The core test. If the brush were ignored and the path fell back to its
    // `Background`, all four of these would be one colour - the background's -
    // rather than four.
    let image = quadrant_image(IMAGE_SIZE);
    let frame = render_frame(window(), move |_, window, _| {
        window
            .paint_path_with_image(
                rect_path(shape()),
                image.clone(),
                0,
                brush_over(shape(), IMAGE_SIZE),
                BrushExtend::Pad,
                BrushExtend::Pad,
                1.,
            )
            .expect("the brush image fits in the atlas");
    });

    let [top_left, top_right, bottom_left, bottom_right] = quadrant_centres();
    frame.assert_painted(top_left, RED, "the image's top-left quadrant");
    frame.assert_painted(top_right, GREEN, "the image's top-right quadrant");
    frame.assert_painted(bottom_left, BLUE, "the image's bottom-left quadrant");
    frame.assert_painted(bottom_right, YELLOW, "the image's bottom-right quadrant");

    let colours = quadrant_centres().map(|point| frame.color_at(point));
    for (index, colour) in colours.iter().enumerate() {
        for other in &colours[index + 1..] {
            assert_ne!(
                colour, other,
                "two of the four sampled quadrants came back the same colour, \
                 which is what a fill that ignored the image looks like"
            );
        }
    }
}

#[test]
fn the_brush_transform_is_honoured() {
    // Frame 0 lays the image over the shape; frame 1 hands the same path the
    // unit transform, which leaves the image sixteen logical pixels wide in the
    // window's corner, nowhere near the shape. Under `Pad` every point of the
    // shape then reads the image's bottom-right corner texel, so the two frames
    // agree at none of the four points - and a renderer that dropped the
    // transform would make them agree at all four.
    let image = quadrant_image(IMAGE_SIZE);
    let frames = render_two_frames(window(), move |index, _, window, _| {
        let transform = if index == 0 {
            brush_over(shape(), IMAGE_SIZE)
        } else {
            TransformationMatrix::unit()
        };
        window
            .paint_path_with_image(
                rect_path(shape()),
                image.clone(),
                0,
                transform,
                BrushExtend::Pad,
                BrushExtend::Pad,
                1.,
            )
            .expect("the brush image fits in the atlas");
    });
    let [transformed, untransformed] = frames;

    let [top_left, top_right, bottom_left, bottom_right] = quadrant_centres();
    transformed.assert_painted(top_left, RED, "the transformed brush, top-left");
    transformed.assert_painted(top_right, GREEN, "the transformed brush, top-right");
    transformed.assert_painted(bottom_left, BLUE, "the transformed brush, bottom-left");
    transformed.assert_painted(bottom_right, YELLOW, "the transformed brush, bottom-right");

    for point in quadrant_centres() {
        untransformed.assert_painted(
            point,
            YELLOW,
            "the untransformed brush, whose image ends long before this point and pads",
        );
    }
    // Every point but the last: padding the untransformed image clamps to its
    // bottom-right texel, which is the same yellow the transformed brush puts
    // in its bottom-right quadrant, so that one point agrees by construction.
    for point in &quadrant_centres()[..3] {
        assert_ne!(
            transformed.color_at(*point),
            untransformed.color_at(*point),
            "the two brush transforms put the same colour at {point:?}, \
             so the transform is not reaching the shader"
        );
    }
}

/// The window the extend-mode tests render: wide enough for the shape to run
/// two image widths across.
fn wide_window() -> gpui::Size<Pixels> {
    size(px(340.), px(160.))
}

/// One copy of the image, a hundred logical pixels square, at the shape's
/// top-left.
fn one_tile() -> Bounds<Pixels> {
    rect(20., 20., 100., 120.)
}

#[test]
fn pad_and_repeat_disagree_outside_the_image() {
    // Both frames sample a point a quarter of a tile past the image's right
    // edge. Padding clamps it to the image's right-hand column; repeating wraps
    // it back to the left-hand one. An extend mode that was stubbed out - or
    // done with a sampler address mode that clamps either way - makes the two
    // frames agree, which is what the final assertion refuses.
    let image = quadrant_image(IMAGE_SIZE);
    let shape = rect(20., 20., 200., 120.);
    let frames = render_two_frames(wide_window(), move |index, _, window, _| {
        let extend = if index == 0 {
            BrushExtend::Pad
        } else {
            BrushExtend::Repeat
        };
        window
            .paint_path_with_image(
                rect_path(shape),
                image.clone(),
                0,
                brush_over(one_tile(), IMAGE_SIZE),
                extend,
                BrushExtend::Pad,
                1.,
            )
            .expect("the brush image fits in the atlas");
    });
    let [padded, repeated] = frames;

    // Inside the one copy of the image both agree: whatever the extend mode,
    // this is the image itself.
    padded.assert_painted(at(45., 50.), RED, "inside the image, padded");
    repeated.assert_painted(at(45., 50.), RED, "inside the image, repeated");

    let past_the_edge_top = at(145., 50.);
    let past_the_edge_bottom = at(145., 110.);
    padded.assert_painted(
        past_the_edge_top,
        GREEN,
        "a quarter of a tile past the image's right edge, padded",
    );
    padded.assert_painted(
        past_the_edge_bottom,
        YELLOW,
        "a quarter of a tile past the image's right edge, padded",
    );
    repeated.assert_painted(
        past_the_edge_top,
        RED,
        "a quarter of a tile past the image's right edge, repeated",
    );
    repeated.assert_painted(
        past_the_edge_bottom,
        BLUE,
        "a quarter of a tile past the image's right edge, repeated",
    );
}

#[test]
fn repeat_recurs_at_the_period_of_the_image() {
    let image = quadrant_image(IMAGE_SIZE);
    let shape = rect(20., 20., 300., 120.);
    let frame = render_frame(size(px(340.), px(160.)), move |_, window, _| {
        window
            .paint_path_with_image(
                rect_path(shape),
                image.clone(),
                0,
                brush_over(one_tile(), IMAGE_SIZE),
                BrushExtend::Repeat,
                BrushExtend::Pad,
                1.,
            )
            .expect("the brush image fits in the atlas");
    });

    // A quarter of the way into the first, second and third copies.
    for (copy, x) in [45., 145., 245.].into_iter().enumerate() {
        frame.assert_painted(
            at(x, 50.),
            RED,
            &format!("a quarter of the way into copy {copy}"),
        );
        frame.assert_painted(
            at(x, 110.),
            BLUE,
            &format!("a quarter of the way down copy {copy}"),
        );
    }
    // Three quarters of the way into each copy is the other column, so the
    // assertions above are not passing on a brush that lost its transform and
    // is painting one colour everywhere.
    for x in [95., 195., 295.] {
        frame.assert_painted(
            at(x, 50.),
            GREEN,
            "three quarters of the way into a repeated copy",
        );
    }
}

#[test]
fn a_repeated_brush_never_reaches_its_neighbour_in_the_atlas() {
    // The atlas packs sprites against one another and `AtlasTile::padding` is
    // zero, so an extend mode done with a sampler address mode wraps - or
    // clamps - into whatever was allocated next door. Frame 0 puts a red and a
    // green image in the atlas side by side and proves the green one is really
    // there and really renders green; frame 1 tiles the red one many times over
    // and looks for any trace of it.
    let red = solid_image(RED, IMAGE_SIZE);
    let green = solid_image(GREEN, IMAGE_SIZE);
    let shape = rect(20., 20., 300., 120.);
    let frames = render_two_frames(size(px(340.), px(160.)), move |index, _, window, _| {
        if index == 0 {
            window
                .paint_image(
                    rect(20., 20., 100., 100.),
                    rect(20., 20., 100., 100.),
                    Corners::default(),
                    red.clone(),
                    0,
                    false,
                )
                .expect("the red image goes into the atlas");
            window
                .paint_image(
                    rect(140., 20., 100., 100.),
                    rect(140., 20., 100., 100.),
                    Corners::default(),
                    green.clone(),
                    0,
                    false,
                )
                .expect("the green image goes into the atlas");
            return;
        }
        window
            .paint_path_with_image(
                rect_path(shape),
                red.clone(),
                0,
                // A twenty-pixel tile, so the shape crosses the image's seam
                // fourteen times over and every one of them is sampled below.
                brush_over(rect(20., 20., 20., 20.), IMAGE_SIZE),
                BrushExtend::Repeat,
                BrushExtend::Repeat,
                1.,
            )
            .expect("the brush image fits in the atlas");
    });
    let [uploads, tiled] = frames;

    uploads.assert_painted(at(70., 70.), RED, "the red image, painted as an image");
    uploads.assert_painted(
        at(190., 70.),
        GREEN,
        "the green image, painted as an image - without which \"no green\" below \
         could pass by there being no green to find",
    );

    // Every device pixel of the tiled shape, inset by a pixel so the path's own
    // antialiased edge is not in the scan.
    let inset = rect(21., 21., 298., 118.);
    let step = 1. / harness::SCALE_FACTOR;
    let mut y = f32::from(inset.origin.y);
    while y < f32::from(inset.bottom()) {
        let mut x = f32::from(inset.origin.x);
        while x < f32::from(inset.right()) {
            let colour = tiled.color_at(at(x, y));
            // Red outright, not merely "not green": the tile is solid, so every
            // pixel of it is the image's one colour unless something else got
            // in - the sprite next door, the empty atlas around it, or the
            // window background showing through a brush that never resolved.
            // The tile is one flat colour and every tap inside it reads that
            // colour outright, so this is written as "red" rather than as "red
            // enough": a tolerance wide enough to admit an eighth of a
            // neighbouring sprite is a tolerance wide enough to admit the bug.
            assert!(
                colour[0] >= 250 && colour[1] <= 5 && colour[2] <= 5,
                "the tiled red brush shows {colour:?} at ({x}, {y}), where every \
                 pixel should be red: a seam that samples across the tile is \
                 reading the atlas's next sprite\n  \
                 frame written to {}",
                tiled
                    .save("a repeated brush reached its atlas neighbour")
                    .display()
            );
            x += step;
        }
        y += step;
    }
}

#[test]
fn a_rounded_content_mask_clips_a_brushed_path() {
    let image = quadrant_image(IMAGE_SIZE);
    let mask = ContentMask {
        bounds: rect(50., 40., 100., 100.),
        corner_radii: Corners::all(px(30.)),
    };
    let frame = render_frame(window(), move |_, window, _| {
        window.with_content_mask(Some(mask), |window| {
            window
                .paint_path_with_image(
                    rect_path(shape()),
                    image.clone(),
                    0,
                    brush_over(shape(), IMAGE_SIZE),
                    BrushExtend::Pad,
                    BrushExtend::Pad,
                    1.,
                )
                .expect("the brush image fits in the atlas");
        });
    });

    // Well inside the mask - it is the mask's own bottom-right corner arc
    // centre, so no radius can cut it - and well clear of the image's quadrant
    // seams, which cross at (100, 80).
    frame.assert_painted(
        at(120., 110.),
        YELLOW,
        "the brushed path inside the rounded mask",
    );
    frame.assert_all_clipped_away(
        &[at(55., 45.), at(145., 45.), at(55., 135.), at(145., 135.)],
        "the corners a 30px radius cuts off the mask",
    );
    frame.assert_all_clipped_away(
        &[at(30., 30.), at(170., 130.)],
        "inside the path, outside the mask's rectangle",
    );
}

#[test]
fn two_brushes_on_different_atlas_textures_each_show_their_own_image() {
    // One atlas texture can be bound per draw, so the batch has to be cut into
    // runs wherever the texture changes. The first image is exactly the size of
    // the atlas's standard texture, so it takes one to itself and the second
    // lands on another; a renderer that bound one texture for the whole batch
    // paints one of these two shapes with the other's image.
    let filler = solid_image(MAGENTA, 1024);
    let second = solid_image(CYAN, IMAGE_SIZE);
    let left = rect(20., 20., 60., 120.);
    let right = rect(120., 20., 60., 120.);
    let frame = render_frame(window(), move |_, window, _| {
        window
            .paint_path_with_image(
                rect_path(left),
                filler.clone(),
                0,
                brush_over(left, 1024),
                BrushExtend::Pad,
                BrushExtend::Pad,
                1.,
            )
            .expect("a 1024-square image is the largest brush the atlas takes");
        window
            .paint_path_with_image(
                rect_path(right),
                second.clone(),
                0,
                brush_over(right, IMAGE_SIZE),
                BrushExtend::Pad,
                BrushExtend::Pad,
                1.,
            )
            .expect("the second brush image fits in the atlas");
    });

    frame.assert_painted(
        at(50., 80.),
        MAGENTA,
        "the path brushed with the image that filled the first atlas texture",
    );
    frame.assert_painted(
        at(150., 80.),
        CYAN,
        "the path brushed with the image on the second atlas texture",
    );
}

#[test]
fn a_brushed_path_composites_over_what_is_under_it() {
    // The premultiplication test, and the reason it is painted over a saturated
    // colour rather than the harness's black: the path pipeline blends
    // One/OneMinusSourceAlpha, so the brush's straight-alpha sample has to be
    // multiplied by its own alpha on the way out. Getting that backwards is
    // invisible over black - `rgb + 0` and `rgb * 1` are the same thing there -
    // and over blue it is the difference between (128, 0, 128) and (255, 0,
    // 128).
    let image = solid_image(RED, IMAGE_SIZE);
    let left = rect(20., 20., 60., 120.);
    let right = rect(120., 20., 60., 120.);
    let frame = render_frame(window(), move |bounds, window, _| {
        window.paint_quad(gpui::fill(bounds, gpui::blue()));
        for (shape, alpha) in [(left, 0.5), (right, 1.)] {
            window
                .paint_path_with_image(
                    rect_path(shape),
                    image.clone(),
                    0,
                    brush_over(shape, IMAGE_SIZE),
                    BrushExtend::Pad,
                    BrushExtend::Pad,
                    alpha,
                )
                .expect("the brush image fits in the atlas");
        }
    });

    frame.assert_painted(
        at(50., 80.),
        [128, 0, 128, 255],
        "a half-alpha red brush over an opaque blue quad",
    );
    frame.assert_painted(
        at(150., 80.),
        RED,
        "the full-alpha control, which pins the colour the half-alpha one is half of",
    );
    frame.assert_painted(
        at(100., 80.),
        BLUE,
        "the quad between the two brushed paths",
    );
}

#[test]
fn element_opacity_dims_a_brushed_path() {
    let image = solid_image(RED, IMAGE_SIZE);
    let left = rect(20., 20., 60., 120.);
    let right = rect(120., 20., 60., 120.);
    let frame = render_frame(window(), move |_, window, _| {
        window.with_element_opacity(Some(0.5), |window| {
            window
                .paint_path_with_image(
                    rect_path(left),
                    image.clone(),
                    0,
                    brush_over(left, IMAGE_SIZE),
                    BrushExtend::Pad,
                    BrushExtend::Pad,
                    1.,
                )
                .expect("the brush image fits in the atlas");
        });
        window
            .paint_path_with_image(
                rect_path(right),
                image.clone(),
                0,
                brush_over(right, IMAGE_SIZE),
                BrushExtend::Pad,
                BrushExtend::Pad,
                1.,
            )
            .expect("the brush image fits in the atlas");
    });

    frame.assert_painted(
        at(50., 80.),
        [128, 0, 0, 255],
        "a brushed path inside a 0.5 opacity layer, over black",
    );
    frame.assert_painted(
        at(150., 80.),
        RED,
        "the same brushed path outside the opacity layer",
    );
}

#[test]
fn an_oversize_brush_image_is_refused_rather_than_given_an_atlas_texture_of_its_own() {
    // The atlas sizes a new texture to whatever it is first asked to hold and
    // never evicts one, so an unbounded brush image is an unbounded, permanent
    // allocation. A pixel over the limit has to come back as an error the
    // caller can log and recover from - the same failure path `paint_image`
    // already has - rather than as tens of megabytes nobody asked for.
    let too_wide = solid_image_of(MAGENTA, MAX_BRUSH_IMAGE_SIZE.0 as u32 + 1, IMAGE_SIZE);
    let widest_allowed = solid_image_of(CYAN, MAX_BRUSH_IMAGE_SIZE.0 as u32, IMAGE_SIZE);
    let outcomes: Arc<Mutex<Vec<Result<(), String>>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = outcomes.clone();
    let left = rect(20., 20., 60., 120.);
    let right = rect(120., 20., 60., 120.);
    let frame = render_frame(window(), move |bounds, window, _| {
        let mut recorded = recorded.lock().expect("the paint callback runs alone");
        recorded.clear();
        // Something under both shapes, so that "the refused path painted
        // nothing" is a claim about the picture rather than one that a window
        // nobody painted into would satisfy on its own.
        window.paint_quad(gpui::fill(bounds, gpui::blue()));
        for (shape, image) in [(left, &too_wide), (right, &widest_allowed)] {
            recorded.push(
                window
                    .paint_path_with_image(
                        rect_path(shape),
                        image.clone(),
                        0,
                        brush_over(shape, IMAGE_SIZE),
                        BrushExtend::Pad,
                        BrushExtend::Pad,
                        1.,
                    )
                    .map_err(|error| error.to_string()),
            );
        }
    });

    let outcomes = outcomes.lock().expect("the paint callback has finished");
    let refusal = outcomes[0]
        .as_ref()
        .expect_err("an image a pixel over the limit must be refused");
    assert!(
        refusal.contains("image brush may be at most"),
        "the refusal should say what the limit is, and said {refusal:?}"
    );
    outcomes[1]
        .as_ref()
        .expect("an image exactly at the limit is still allowed");

    frame.assert_painted(
        at(50., 80.),
        BLUE,
        "under the path whose brush was refused, which has to be left exactly \
         as it was rather than filled with the transparent black a path's \
         colour defaults to",
    );
    frame.assert_painted(
        at(150., 80.),
        CYAN,
        "the path whose brush was exactly at the limit, which pins the refusal \
         on the size rather than on brushes having stopped working",
    );
}

#[test]
fn the_image_lands_where_the_transform_puts_it_to_within_a_pixel() {
    // [`the_image_maps_across_the_shape`] samples the middles of the four
    // quadrants, which a brush transform could be thirty logical pixels out
    // and still pass. Here the image has one texel per logical pixel, so the
    // seams the quadrants meet at are a pixel wide, and every sample is taken
    // two pixels from the crossing at (100, 80): a transform out by more than
    // that reads the wrong quadrant.
    let image = quadrant_image(160);
    let frame = render_frame(window(), move |_, window, _| {
        window
            .paint_path_with_image(
                rect_path(shape()),
                image.clone(),
                0,
                brush_over(shape(), 160),
                BrushExtend::Pad,
                BrushExtend::Pad,
                1.,
            )
            .expect("the brush image fits in the atlas");
    });

    frame.assert_painted(at(98., 78.), RED, "two pixels inside the top-left quadrant");
    frame.assert_painted(
        at(102., 78.),
        GREEN,
        "two pixels inside the top-right quadrant",
    );
    frame.assert_painted(
        at(98., 82.),
        BLUE,
        "two pixels inside the bottom-left quadrant",
    );
    frame.assert_painted(
        at(102., 82.),
        YELLOW,
        "two pixels inside the bottom-right quadrant",
    );
}

#[test]
fn reflect_mirrors_every_other_copy_and_repeat_does_not() {
    // `BrushExtend::Reflect`, which nothing else in these tests exercises. The
    // shape runs three tiles across, so the second copy is the mirrored one and
    // the third is the right way round again: under `Repeat` the same quarter of
    // every copy is the same colour, and under `Reflect` the second copy's
    // quarters are swapped.
    let image = quadrant_image(IMAGE_SIZE);
    let shape = rect(20., 20., 300., 120.);
    let frames = render_two_frames(wide_window(), move |index, _, window, _| {
        let extend = if index == 0 {
            BrushExtend::Repeat
        } else {
            BrushExtend::Reflect
        };
        window
            .paint_path_with_image(
                rect_path(shape),
                image.clone(),
                0,
                brush_over(one_tile(), IMAGE_SIZE),
                extend,
                BrushExtend::Pad,
                1.,
            )
            .expect("the brush image fits in the atlas");
    });
    let [repeated, reflected] = frames;

    // The first copy is the image itself either way.
    repeated.assert_painted(at(45., 50.), RED, "a quarter into the first copy, repeated");
    reflected.assert_painted(
        at(45., 50.),
        RED,
        "a quarter into the first copy, reflected",
    );

    // The second copy: `Repeat` starts the image again, `Reflect` runs it
    // backwards, so its left quarter is the image's right-hand column.
    repeated.assert_painted(at(145., 50.), RED, "a quarter into the second copy");
    reflected.assert_painted(
        at(145., 50.),
        GREEN,
        "a quarter into the mirrored second copy, which is the image's \
         right-hand column",
    );
    repeated.assert_painted(at(195., 50.), GREEN, "three quarters into the second copy");
    reflected.assert_painted(
        at(195., 50.),
        RED,
        "three quarters into the mirrored second copy",
    );

    // And the third copy is the right way round again, which is what makes this
    // a mirror rather than a reversal.
    repeated.assert_painted(at(245., 50.), RED, "a quarter into the third copy");
    reflected.assert_painted(
        at(245., 50.),
        RED,
        "a quarter into the third copy, which `Reflect` puts back the right way \
         round",
    );
}

#[test]
fn a_brush_carries_the_alpha_of_the_image_it_samples() {
    // Every other test in this file paints an opaque image onto an opaque
    // target, where the alpha the brush pipeline emits is 255 whatever it did.
    // Here a half-transparent image is painted onto nothing at all, so what
    // comes back is the premultiplied RGBA the pipeline actually wrote: a brush
    // that emitted straight colour, or the image's alpha unmultiplied into it,
    // reads differently in both the colour and the alpha channel.
    let translucent = solid_image([255, 0, 0, 128], IMAGE_SIZE);
    let opaque = solid_image(RED, IMAGE_SIZE);
    let left = rect(20., 20., 60., 120.);
    let right = rect(120., 20., 60., 120.);
    let frame = render_frame_on_transparent(window(), move |_, window, _| {
        for (shape, image) in [(left, &translucent), (right, &opaque)] {
            window
                .paint_path_with_image(
                    rect_path(shape),
                    image.clone(),
                    0,
                    brush_over(shape, IMAGE_SIZE),
                    BrushExtend::Pad,
                    BrushExtend::Pad,
                    1.,
                )
                .expect("the brush image fits in the atlas");
        }
    });

    frame.assert_painted(
        at(50., 80.),
        [128, 0, 0, 128],
        "a half-transparent red brush over nothing, premultiplied",
    );
    frame.assert_painted(
        at(150., 80.),
        [255, 0, 0, 255],
        "the opaque control, which pins the alpha on the image rather than on \
         brushes being transparent",
    );
    frame.assert_painted(
        at(100., 80.),
        harness::TRANSPARENT,
        "between the two shapes, where nothing was painted",
    );
}

#[test]
fn a_group_at_half_opacity_fades_the_image_a_path_is_brushed_with() {
    // `<div style="opacity: .5">` around a `background-image`, which is what
    // the HTML painter emits. A group holding one brushed path cannot overlap
    // itself, so it always folds away and its opacity is folded into the
    // primitives - and the fill a folded brush ends up with is the brush's own
    // multiplier, not `Path::color`, which a renderer that resolves a brush
    // never reads.
    let image = solid_image(RED, IMAGE_SIZE);
    let shape = rect(20., 20., 60., 120.);
    let groups: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = groups.clone();
    let frame = render_frame(window(), move |bounds, window, _| {
        window.paint_quad(gpui::fill(bounds, gpui::blue()));
        window.with_group_opacity(shape, 0.5, |window| {
            window
                .paint_path_with_image(
                    rect_path(shape),
                    image.clone(),
                    0,
                    brush_over(shape, IMAGE_SIZE),
                    BrushExtend::Pad,
                    BrushExtend::Pad,
                    1.,
                )
                .expect("the brush image fits in the atlas");
        });
        let mut recorded = recorded.lock().expect("the paint callback runs alone");
        recorded.clear();
        recorded.push(window.isolated_group_count());
    });

    assert_eq!(
        groups
            .lock()
            .expect("the paint callback has finished")
            .as_slice(),
        &[0],
        "a group holding one path has nothing to overlap, so it folds away - \
         which is the path this test is about"
    );
    frame.assert_painted(
        at(50., 80.),
        [128, 0, 128, 255],
        "a red brush inside a 0.5 group, over an opaque blue quad",
    );
}
