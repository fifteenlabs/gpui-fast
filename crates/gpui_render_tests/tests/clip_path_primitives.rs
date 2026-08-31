//! What a clip path does to every primitive that can land inside one.
//!
//! Each primitive kind reaches the GPU through a pipeline of its own, and each
//! pipeline has a clipped variant of its own, so any one of them can lose the
//! mask without the others noticing. Every test paints the same diagonal clip
//! over the same shape and pairs it with the unclipped control: without the
//! control an assertion that a corner is empty would also pass if the primitive
//! stopped being painted at all.

#![cfg(target_os = "macos")]

mod harness;

use std::sync::Arc;

use core_foundation::{base::TCFType, dictionary::CFDictionary, string::CFString};
use core_video::pixel_buffer::{
    CVPixelBuffer, CVPixelBufferKeys, kCVPixelFormatType_32BGRA,
    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
};
use gpui::{
    App, BorderStyle, Bounds, BoxShadow, ClipPath, Corners, Path, Pixels, Point, RenderImage, Size,
    TextAlign, TextRun, UnderlineStyle, Window, font, px, quad, size, white,
};
use harness::{WHITE, at, rect, render_frame};
use image::{Frame, RgbaImage};

/// The window every test in this file renders, in logical pixels.
const WINDOW: f32 = 200.;

fn window() -> Size<Pixels> {
    size(px(WINDOW), px(WINDOW))
}

/// A right triangle covering the upper right of the window: a point is inside
/// when `30 <= y < x <= 170`.
///
/// A triangle rather than a rectangle so that the fallback a clip takes when it
/// cannot get a mask - its bounding rectangle - would fail these tests too.
fn diagonal_clip() -> ClipPath<Pixels> {
    ClipPath::builder()
        .move_to(at(30., 30.))
        .line_to(at(170., 30.))
        .line_to(at(170., 170.))
        .close()
        .build()
}

/// Points inside the triangle, and well clear of its edges.
const KEPT: [(f32, f32); 4] = [(140., 50.), (150., 90.), (160., 150.), (120., 45.)];

/// Points inside the triangle's bounding rectangle but below its hypotenuse.
const CUT: [(f32, f32); 4] = [(50., 140.), (80., 150.), (45., 120.), (140., 160.)];

fn points(from: &[(f32, f32)]) -> Vec<Point<Pixels>> {
    from.iter().map(|&(x, y)| at(x, y)).collect()
}

/// Renders `paint` twice - once inside [`diagonal_clip`] and once with no clip
/// at all - and asserts the clip took exactly `cut` and left exactly `kept`.
#[track_caller]
fn assert_clipped(
    what: &str,
    kept: &[(f32, f32)],
    cut: &[(f32, f32)],
    paint: fn(Bounds<Pixels>, &mut Window, &mut App),
) {
    let clipped = render_frame(window(), move |bounds, window, cx| {
        window.with_clip_path(&diagonal_clip(), |window| paint(bounds, window, cx));
    });
    clipped.assert_all_painted(&points(kept), WHITE, &format!("{what} inside a clip path"));
    clipped.assert_all_clipped_away(&points(cut), &format!("{what} where the clip path cuts it"));

    let unclipped = render_frame(window(), move |bounds, window, cx| {
        paint(bounds, window, cx)
    });
    unclipped.assert_all_painted(
        &points(cut),
        WHITE,
        &format!("{what} with no clip path, which is what makes the above a test"),
    );
}

#[test]
fn a_flat_quad_is_clipped_by_a_clip_path() {
    // The fast path through `quad_fragment`: no border, no radii, so the
    // shader returns before most of it runs.
    assert_clipped("a flat quad", &KEPT, &CUT, |bounds, window, _| {
        window.paint_quad(gpui::fill(bounds, white()));
    });
}

#[test]
fn a_bordered_quad_is_clipped_by_a_clip_path() {
    // A border takes the quad down the long branch instead.
    assert_clipped("a bordered quad", &KEPT, &CUT, |bounds, window, _| {
        window.paint_quad(quad(
            bounds,
            Corners::all(px(0.)),
            white(),
            px(4.),
            white(),
            BorderStyle::Solid,
        ));
    });
}

#[test]
fn a_rounded_quad_is_clipped_by_a_clip_path() {
    // The quad's own radii are at the window's corners, far from every point
    // under test: what is being checked is the clip, not the quad's shape.
    assert_clipped("a rounded quad", &KEPT, &CUT, |bounds, window, _| {
        window.paint_quad(quad(
            bounds,
            Corners::all(px(8.)),
            white(),
            px(0.),
            white(),
            BorderStyle::Solid,
        ));
    });
}

/// A band thick enough to cross the clip's hypotenuse, so the same points work
/// on both sides of it.
const UNDERLINE_KEPT: [(f32, f32); 3] = [(140., 50.), (150., 100.), (130., 60.)];
const UNDERLINE_CUT: [(f32, f32); 3] = [(50., 60.), (60., 100.), (45., 50.)];

#[test]
fn an_underline_is_clipped_by_a_clip_path() {
    assert_clipped(
        "an underline",
        &UNDERLINE_KEPT,
        &UNDERLINE_CUT,
        |bounds, window, _| {
            window.paint_underline(
                at(0., 40.),
                bounds.size.width,
                &UnderlineStyle {
                    thickness: px(80.),
                    color: Some(white()),
                    wavy: false,
                },
            );
        },
    );
}

#[test]
fn a_shadow_is_clipped_by_a_clip_path() {
    // A drop shadow has its own pipeline, and its geometry is derived in the
    // vertex stage from a rectangle it is not drawn at, so it is the one
    // primitive whose device position could plausibly not line up with the
    // atlas.
    assert_clipped("a shadow", &KEPT, &CUT, |bounds, window, _| {
        window.paint_drop_shadows(
            bounds,
            Corners::all(px(0.)),
            &[BoxShadow {
                color: white(),
                offset: Point::default(),
                blur_radius: px(8.),
                spread_radius: px(0.),
                inset: false,
            }],
        );
    });
}

/// An opaque white image. The renderer reads the bytes as BGRA, which for white
/// is the same four bytes either way.
fn white_image(side: u32) -> Arc<RenderImage> {
    let pixels = RgbaImage::from_pixel(side, side, image::Rgba([255, 255, 255, 255]));
    Arc::new(RenderImage::new(vec![Frame::new(pixels)]))
}

#[test]
fn an_image_is_clipped_by_a_clip_path() {
    // Images are polychrome sprites, and an email body is mostly made of them.
    assert_clipped("an image", &KEPT, &CUT, |bounds, window, _| {
        window
            .paint_image(
                bounds,
                bounds,
                Corners::all(px(0.)),
                white_image(64),
                0,
                false,
            )
            .expect("failed to paint the image");
    });
}

#[test]
fn a_path_is_clipped_by_a_clip_path() {
    // A path is rasterized into an intermediate texture and copied back out,
    // so its clip has to be applied during the rasterization - the copy applies
    // neither the clip nor the content mask.
    assert_clipped("a path", &KEPT, &CUT, |bounds, window, _| {
        let mut path = Path::new(bounds.origin);
        path.line_to(bounds.top_right());
        path.line_to(bounds.bottom_right());
        path.line_to(bounds.bottom_left());
        window.paint_path(path, white());
    });
}

/// Paints enough large text to cover the middle of the window several times
/// over, in white.
fn paint_a_block_of_text(_: Bounds<Pixels>, window: &mut Window, cx: &mut App) {
    let text = "████████████";
    let run = TextRun {
        len: text.len(),
        font: font("Menlo"),
        color: white(),
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    let line = window
        .text_system()
        .shape_line(text.into(), px(30.), &[run], None);
    for row in 0..8 {
        line.paint(
            at(20., 20. + row as f32 * 22.),
            px(22.),
            TextAlign::Left,
            None,
            window,
            cx,
        )
        .expect("failed to paint the text");
    }
}

#[test]
fn glyphs_are_clipped_by_a_clip_path() {
    // Glyphs are monochrome sprites, whose fragment stage never loads the
    // sprite record: its clip id arrives as a flat varying that only exists in
    // the clipped variant of the pipeline.
    //
    // Which pixels a font covers is not this test's business, so the claims are
    // about regions: the unclipped frame establishes that the region below the
    // hypotenuse was covered at all, and the clipped one has to have emptied
    // it.
    let below = rect(45., 120., 30., 30.);
    let above = rect(120., 45., 30., 30.);

    let unclipped = render_frame(window(), |bounds, window, cx| {
        paint_a_block_of_text(bounds, window, cx)
    });
    unclipped.assert_region_painted(
        below,
        "the glyphs an unclipped frame puts below the diagonal",
    );
    unclipped.assert_region_painted(
        above,
        "the glyphs an unclipped frame puts above the diagonal",
    );

    let clipped = render_frame(window(), |bounds, window, cx| {
        window.with_clip_path(&diagonal_clip(), |window| {
            paint_a_block_of_text(bounds, window, cx)
        });
    });
    clipped.assert_region_painted(above, "the glyphs a clip path keeps above its hypotenuse");
    clipped.assert_region_clipped_away(below, "the glyphs a clip path cuts below its hypotenuse");
}

/// An opaque white BGRA pixel buffer, backed by an IOSurface so the Metal
/// texture cache will accept it.
fn white_surface(side: usize) -> CVPixelBuffer {
    let io_surface_properties = CFDictionary::<CFString, CFString>::from_CFType_pairs(&[]);
    let attributes = CFDictionary::from_CFType_pairs(&[(
        CFString::from(CVPixelBufferKeys::IOSurfaceProperties),
        io_surface_properties.as_CFType(),
    )]);
    let buffer = CVPixelBuffer::new(kCVPixelFormatType_32BGRA, side, side, Some(&attributes))
        .expect("failed to create a pixel buffer for the surface");
    assert_eq!(buffer.lock_base_address(0), 0, "failed to lock the buffer");
    // SAFETY: the buffer is locked, so its base address is valid for
    // `bytes_per_row * height` bytes, and each row's first `side * 4` bytes are
    // the pixels.
    unsafe {
        let base = buffer.get_base_address() as *mut u8;
        let stride = buffer.get_bytes_per_row();
        for row in 0..side {
            std::ptr::write_bytes(base.add(row * stride), 0xff, side * 4);
        }
    }
    assert_eq!(
        buffer.unlock_base_address(0),
        0,
        "failed to unlock the buffer"
    );
    buffer
}

#[test]
fn a_bgra_surface_is_clipped_by_a_clip_path() {
    // A surface is drawn one instance at a time out of a record the renderer
    // builds itself, so its clip id travels by a route no other primitive uses.
    assert_clipped("a BGRA surface", &KEPT, &CUT, |bounds, window, _| {
        window.paint_surface(bounds, white_surface(64));
    });
}

/// An opaque white biplanar YCbCr pixel buffer, backed by an IOSurface.
///
/// The BGRA surface above goes down the `bgra_surfaces` pipeline; this one goes
/// down `surfaces`, whose fragment stage does the YCbCr conversion and blends
/// straight rather than premultiplied alpha. They are two pipelines with two
/// clipped variants, and only one of them was covered.
fn white_ycbcr_surface(side: usize) -> CVPixelBuffer {
    let io_surface_properties = CFDictionary::<CFString, CFString>::from_CFType_pairs(&[]);
    let attributes = CFDictionary::from_CFType_pairs(&[(
        CFString::from(CVPixelBufferKeys::IOSurfaceProperties),
        io_surface_properties.as_CFType(),
    )]);
    let buffer = CVPixelBuffer::new(
        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        side,
        side,
        Some(&attributes),
    )
    .expect("failed to create a biplanar pixel buffer for the surface");
    assert_eq!(buffer.lock_base_address(0), 0, "failed to lock the buffer");
    // SAFETY: the buffer is locked, so each plane's base address is valid for
    // `bytes_per_row_of_plane * height_of_plane` bytes. Plane 0 is luma and
    // plane 1 is interleaved chroma at half resolution; full-range white is
    // luma 255 with both chroma channels at 128.
    unsafe {
        for (plane, value) in [(0usize, 0xffu8), (1, 0x80)] {
            let base = buffer.get_base_address_of_plane(plane) as *mut u8;
            let stride = buffer.get_bytes_per_row_of_plane(plane);
            for row in 0..buffer.get_height_of_plane(plane) {
                std::ptr::write_bytes(base.add(row * stride), value, stride);
            }
        }
    }
    assert_eq!(
        buffer.unlock_base_address(0),
        0,
        "failed to unlock the buffer"
    );
    buffer
}

#[test]
fn a_ycbcr_surface_is_clipped_by_a_clip_path() {
    assert_clipped("a YCbCr surface", &KEPT, &CUT, |bounds, window, _| {
        window.paint_surface(bounds, white_ycbcr_surface(64));
    });
}
