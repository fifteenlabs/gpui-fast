//! A headless GPU render harness: paint a gpui scene, get real pixels back.
//!
//! `render_frame` opens an invisible window backed by the platform's headless
//! renderer (Metal, on macOS), runs the given paint callback inside a real
//! paint pass, and reads the rendered texture back as an image. The returned
//! [`RenderedFrame`] carries assertions that name what they check, so a failure
//! reads as a statement about the picture rather than about a byte array.
//!
//! `render_frame_on_transparent` is the same thing over a cleared-to-nothing
//! target. Over opaque black every pixel reads back with an alpha of 255, so
//! nothing painted onto it can say anything about alpha; a transparent target
//! gives back the premultiplied RGBA the scene actually composited, which is
//! what a test of blending or of group compositing has to look at.
//!
//! Nothing here fakes the renderer. If the GPU is unavailable the harness
//! panics with the underlying error instead of substituting a software path,
//! because a passing test on a stub would say nothing about the shaders.

#![allow(dead_code, reason = "each test binary uses a different part of this")]

use std::{rc::Rc, sync::Arc};

use gpui::{
    App, AppContext as _, Bounds, Context, HeadlessAppContext, IntoElement, Pixels, Point, Render,
    Size, Styled, Window, canvas, point, px, size,
};
use image::RgbaImage;

/// The scale factor gpui's test window reports. Logical coordinates handed to
/// the assertions are multiplied by this to find the pixel in the image.
pub const SCALE_FACTOR: f32 = 2.0;

/// How far a channel may drift from the expected colour before an assertion
/// that something *was* painted fails. Antialiasing and 8-bit rounding both
/// land inside this.
///
/// It deliberately does not apply to the other direction. "Nothing was painted
/// here" is [`RenderedFrame::is_background`], an exact comparison: the harness
/// clears to a flat colour and paints in opaque white, so a clipped pixel is
/// bit-for-bit the clear colour, and a tolerance there would let a 1.6%
/// antialiased fringe through the assertion whose whole job is to catch it.
pub const CHANNEL_TOLERANCE: u8 = 4;

/// The colour the headless renderer clears to by default: opaque black.
/// Anything a content mask cuts away shows this.
pub const BACKGROUND: [u8; 4] = [0, 0, 0, 255];

/// The colour a transparent-clear frame starts from: nothing at all.
pub const TRANSPARENT: [u8; 4] = [0, 0, 0, 0];

/// Opaque white, the colour these tests paint with.
pub const WHITE: [u8; 4] = [255, 255, 255, 255];

/// Opaque white painted at half opacity over [`BACKGROUND`].
pub const HALF_WHITE: [u8; 4] = [128, 128, 128, 255];

/// White at half alpha over [`TRANSPARENT`], premultiplied, which is how the
/// renderer writes and reads it back: `0.5 * 255` in every channel.
pub const HALF_WHITE_ON_NOTHING: [u8; 4] = [128, 128, 128, 128];

/// Two coats of [`HALF_WHITE_ON_NOTHING`] composited source-over: alpha
/// `0.5 + 0.5 * 0.5 = 0.75`, and the premultiplied colour alongside it.
pub const THREE_QUARTER_WHITE_ON_NOTHING: [u8; 4] = [191, 191, 191, 191];

/// Paints `paint` into an invisible window of `window_size` logical pixels and
/// returns what the GPU actually produced.
///
/// The callback runs in gpui's paint phase with the window's full bounds, so it
/// can call any of `Window`'s paint APIs directly.
pub fn render_frame(
    window_size: Size<Pixels>,
    paint: impl Fn(Bounds<Pixels>, &mut Window, &mut App) + 'static,
) -> RenderedFrame {
    render_frame_onto(window_size, BACKGROUND, paint)
}

/// Paints `paint` onto a fully transparent target and returns the premultiplied
/// RGBA the GPU composited, alpha included.
///
/// Use this for anything whose claim is about alpha. [`render_frame`] clears to
/// opaque black, so its alpha channel is 255 everywhere no matter what the
/// scene did and an assertion on it proves nothing.
pub fn render_frame_on_transparent(
    window_size: Size<Pixels>,
    paint: impl Fn(Bounds<Pixels>, &mut Window, &mut App) + 'static,
) -> RenderedFrame {
    render_frame_onto(window_size, TRANSPARENT, paint)
}

/// Paints two frames into one window and returns both.
///
/// One window is one renderer, and a renderer is where everything that outlives
/// a frame lives: the clip mask atlas and the textures around it are sized on
/// the first frame and resized on the second. A test that opens a window per
/// frame can never see that happen.
pub fn render_two_frames(
    window_size: Size<Pixels>,
    paint: impl Fn(usize, Bounds<Pixels>, &mut Window, &mut App) + 'static,
) -> [RenderedFrame; 2] {
    let paint = Rc::new(paint);
    render_frames_onto(
        window_size,
        BACKGROUND,
        2,
        move |frame, bounds, window, cx| paint(frame, bounds, window, cx),
    )
    .try_into()
    .map_err(|_| ())
    .expect("two frames were asked for")
}

/// Paints `count` frames into one window and returns all of them.
///
/// The same window and the same renderer throughout, so anything that outlives
/// a frame - a pooled target, an atlas, a cache with an eviction policy - is
/// carried from one frame to the next the way it is in a running app.
pub fn render_frames(
    window_size: Size<Pixels>,
    count: usize,
    paint: impl Fn(usize, Bounds<Pixels>, &mut Window, &mut App) + 'static,
) -> Vec<RenderedFrame> {
    render_frames_onto(window_size, BACKGROUND, count, paint)
}

fn render_frame_onto(
    window_size: Size<Pixels>,
    background: [u8; 4],
    paint: impl Fn(Bounds<Pixels>, &mut Window, &mut App) + 'static,
) -> RenderedFrame {
    let mut frames =
        render_frames_onto(window_size, background, 1, move |_, bounds, window, cx| {
            paint(bounds, window, cx)
        });
    frames.pop().expect("one frame was asked for")
}

fn render_frames_onto(
    window_size: Size<Pixels>,
    background: [u8; 4],
    count: usize,
    paint: impl Fn(usize, Bounds<Pixels>, &mut Window, &mut App) + 'static,
) -> Vec<RenderedFrame> {
    let transparent = background == TRANSPARENT;
    let text_system = Arc::new(gpui_macos::MacTextSystem::new());
    let mut cx = HeadlessAppContext::with_platform(text_system, Arc::new(()), move || {
        let renderer = gpui_platform::current_headless_renderer_with_transparency(transparent);
        assert!(
            renderer.is_some(),
            "no headless renderer for this platform; these tests need a real GPU"
        );
        renderer
    });

    let paint: Rc<dyn Fn(usize, Bounds<Pixels>, &mut Window, &mut App)> = Rc::new(paint);
    let window = cx
        .open_window(window_size, |_, cx| {
            cx.new(|_| PaintRoot { paint, frame: 0 })
        })
        .expect("failed to open the headless window");
    cx.run_until_parked();

    let mut frames = Vec::with_capacity(count);
    for index in 0..count {
        if index > 0 {
            window
                .update(&mut cx, |root, _, cx| {
                    root.frame = index;
                    cx.notify();
                })
                .expect("failed to advance to the next frame");
            cx.run_until_parked();
        }
        let image = cx
            .capture_screenshot(window.into())
            .expect("failed to render the scene to an image");

        assert_eq!(
            (image.width(), image.height()),
            (
                (f32::from(window_size.width) * SCALE_FACTOR) as u32,
                (f32::from(window_size.height) * SCALE_FACTOR) as u32
            ),
            "the captured image is not the size of the window"
        );
        frames.push(RenderedFrame { image, background });
    }
    frames
}

/// A convenience for the common shape of these tests: paint one thing inside a
/// content mask. `mask` is intersected with the window's own mask, exactly as
/// any element's would be.
pub fn render_inside_content_mask(
    window_size: Size<Pixels>,
    mask: gpui::ContentMask<Pixels>,
    paint: impl Fn(Bounds<Pixels>, &mut Window, &mut App) + 'static,
) -> RenderedFrame {
    render_frame(window_size, move |bounds, window, cx| {
        window.with_content_mask(Some(mask), |window| paint(bounds, window, cx));
    })
}

struct PaintRoot {
    paint: Rc<dyn Fn(usize, Bounds<Pixels>, &mut Window, &mut App)>,
    frame: usize,
}

impl Render for PaintRoot {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let paint = self.paint.clone();
        let frame = self.frame;
        canvas(
            |_, _, _| (),
            move |bounds, _, window, cx| paint(frame, bounds, window, cx),
        )
        .size_full()
    }
}

/// The pixels a frame produced, with assertions that read as claims about the
/// picture.
pub struct RenderedFrame {
    image: RgbaImage,
    background: [u8; 4],
}

impl RenderedFrame {
    /// The raw image, for a test that needs something the assertions don't cover.
    pub fn image(&self) -> &RgbaImage {
        &self.image
    }

    /// The colour at a logical point, as RGBA bytes.
    pub fn color_at(&self, at: Point<Pixels>) -> [u8; 4] {
        let (x, y) = self.device_pixel(at);
        self.image.get_pixel(x, y).0
    }

    /// Asserts the given logical point was painted in `expected`, within
    /// [`CHANNEL_TOLERANCE`] per channel.
    #[track_caller]
    pub fn assert_painted(&self, at: Point<Pixels>, expected: [u8; 4], what: &str) {
        let actual = self.color_at(at);
        if !within_tolerance(actual, expected) {
            panic!(
                "{what}: expected {expected:?} at {}, found {actual:?}{}",
                describe(at),
                self.dump_hint(what),
            );
        }
    }

    /// The colour this frame was cleared to, and so what "nothing was painted
    /// here" looks like in it.
    pub fn background(&self) -> [u8; 4] {
        self.background
    }

    /// Whether the given logical point still shows the window background
    /// exactly: the one definition of "nothing was painted here" the whole
    /// harness uses.
    pub fn is_background(&self, at: Point<Pixels>) -> bool {
        self.color_at(at) == self.background
    }

    /// How many of the *device* pixels in `region` the frame painted something
    /// at.
    ///
    /// The grid is one device pixel, not one logical pixel: at a scale factor
    /// of 2 a logical grid samples every second device pixel, and the whole
    /// point of `assert_region_clipped_away` is to catch a one-pixel artefact
    /// that a coarser grid would step straight over.
    pub fn painted_points(&self, region: Bounds<Pixels>) -> usize {
        let step = 1. / SCALE_FACTOR;
        let mut painted = 0;
        let mut y = f32::from(region.origin.y);
        while y < f32::from(region.bottom()) {
            let mut x = f32::from(region.origin.x);
            while x < f32::from(region.right()) {
                if !self.is_background(at(x, y)) {
                    painted += 1;
                }
                x += step;
            }
            y += step;
        }
        painted
    }

    /// Asserts nothing reached the given logical point: it still shows the
    /// window background. This is the assertion a clip is supposed to make true.
    #[track_caller]
    pub fn assert_clipped_away(&self, at: Point<Pixels>, what: &str) {
        let actual = self.color_at(at);
        if !self.is_background(at) {
            let background = self.background;
            panic!(
                "{what}: expected the background {background:?} at {}, found {actual:?} - \
                 something was painted where the mask should have clipped it{}",
                describe(at),
                self.dump_hint(what),
            );
        }
    }

    /// Asserts every one of the given logical points was painted in `expected`.
    #[track_caller]
    pub fn assert_all_painted(&self, at: &[Point<Pixels>], expected: [u8; 4], what: &str) {
        for point in at {
            self.assert_painted(*point, expected, what);
        }
    }

    /// Asserts every one of the given logical points still shows the background.
    #[track_caller]
    pub fn assert_all_clipped_away(&self, at: &[Point<Pixels>], what: &str) {
        for point in at {
            self.assert_clipped_away(*point, what);
        }
    }

    /// Asserts nothing at all reached `region`, and says how much did when
    /// something has.
    #[track_caller]
    pub fn assert_region_clipped_away(&self, region: Bounds<Pixels>, what: &str) {
        let painted = self.painted_points(region);
        if painted > 0 {
            panic!(
                "{what}: {painted} of the points in {region:?} were painted, where the mask \
                 should have clipped every one of them{}",
                self.dump_hint(what),
            );
        }
    }

    /// Asserts something reached `region`. Every "the mask cut this away"
    /// assertion needs one of these beside it, or it can pass by nothing having
    /// been painted at all.
    #[track_caller]
    pub fn assert_region_painted(&self, region: Bounds<Pixels>, what: &str) {
        if self.painted_points(region) == 0 {
            panic!(
                "{what}: nothing at all was painted in {region:?}, so an emptier region \
                 elsewhere proves nothing{}",
                self.dump_hint(what),
            );
        }
    }

    /// Writes the frame to `target/gpui_render_tests/<name>.png` and returns the
    /// path, for eyeballing a failure. Assertion failures call this themselves.
    pub fn save(&self, name: &str) -> std::path::PathBuf {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/gpui_render_tests");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(format!("{}.png", slug(name)));
        let _ = self.image.save(&path);
        path
    }

    fn dump_hint(&self, what: &str) -> String {
        format!("\n  frame written to {}", self.save(what).display())
    }

    fn device_pixel(&self, at: Point<Pixels>) -> (u32, u32) {
        let x = (f32::from(at.x) * SCALE_FACTOR) as u32;
        let y = (f32::from(at.y) * SCALE_FACTOR) as u32;
        assert!(
            x < self.image.width() && y < self.image.height(),
            "{} is outside the {}x{} frame",
            describe(at),
            self.image.width(),
            self.image.height()
        );
        (x, y)
    }
}

fn within_tolerance(actual: [u8; 4], expected: [u8; 4]) -> bool {
    actual
        .iter()
        .zip(expected.iter())
        .all(|(a, b)| a.abs_diff(*b) <= CHANNEL_TOLERANCE)
}

fn describe(at: Point<Pixels>) -> String {
    format!(
        "({}, {}) logical / ({}, {}) device",
        f32::from(at.x),
        f32::from(at.y),
        f32::from(at.x) * SCALE_FACTOR,
        f32::from(at.y) * SCALE_FACTOR
    )
}

fn slug(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

/// A logical point, spelled shorter than `point(px(x), px(y))`.
pub fn at(x: f32, y: f32) -> Point<Pixels> {
    point(px(x), px(y))
}

/// A logical rectangle, spelled shorter.
pub fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<Pixels> {
    Bounds {
        origin: at(x, y),
        size: size(px(width), px(height)),
    }
}
