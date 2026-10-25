//! Tests of images dropped while something still draws them. See
//! [`crate::fast::image_drops`].

use crate::{
    AppContext as _, Context, Entity, ImageSource, IntoElement, ParentElement as _, Render,
    RenderImage, Styled as _, TestAppContext, Window, WindowHandle, div, img, px,
};
use image::{Frame, ImageBuffer, Rgba};
use smallvec::SmallVec;
use std::sync::Arc;

fn image() -> Arc<RenderImage> {
    let frame = Frame::new(ImageBuffer::from_pixel(4, 4, Rgba([255, 0, 0, 255])));
    Arc::new(RenderImage::new(SmallVec::from_elem(frame, 1)))
}

struct Counter(u32);
impl Render for Counter {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().child(format!("{}", self.0))
    }
}

/// Shows `image` while `shown`, beside a nested view: notifying the nested
/// view draws this one again from last frame, without rendering it.
struct Shows {
    image: Arc<RenderImage>,
    shown: bool,
    counter: Entity<Counter>,
}
impl Render for Shows {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let mut root = div().child(self.counter.clone());
        if self.shown {
            root = root.child(img(ImageSource::Render(self.image.clone())).size(px(10.)));
        }
        root
    }
}

fn open(cx: &mut TestAppContext, image: &Arc<RenderImage>) -> WindowHandle<Shows> {
    let image = image.clone();
    let window = cx.add_window(move |_, cx| Shows {
        image,
        shown: true,
        counter: cx.new(|_| Counter(0)),
    });
    cx.run_until_parked();
    window
}

fn in_atlas(cx: &mut TestAppContext, window: WindowHandle<Shows>, image: &RenderImage) -> bool {
    window
        .update(cx, |_, window, _| window.has_image_atlas_entry(image))
        .unwrap()
}

#[test]
fn an_image_dropped_while_drawn_keeps_its_tile_until_nothing_draws_it() {
    let mut cx = TestAppContext::single();
    let image = image();
    let window = open(&mut cx, &image);
    assert!(in_atlas(&mut cx, window, &image));

    window
        .update(&mut cx, |_, window, _| {
            assert!(window.image_drawn(&image));
            window.drop_image(image.clone()).unwrap();
        })
        .unwrap();
    assert!(
        in_atlas(&mut cx, window, &image),
        "a drawn image's tile is kept"
    );

    window
        .update(&mut cx, |shows, _, cx| {
            shows.counter.update(cx, |counter, cx| {
                counter.0 += 1;
                cx.notify();
            })
        })
        .unwrap();
    cx.run_until_parked();
    assert!(
        in_atlas(&mut cx, window, &image),
        "a frame that draws the image again from last frame keeps its tile"
    );

    window
        .update(&mut cx, |shows, _, cx| {
            shows.shown = false;
            cx.notify();
        })
        .unwrap();
    cx.run_until_parked();
    window
        .update(&mut cx, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();
    assert!(
        !in_atlas(&mut cx, window, &image),
        "the tile is given back once nothing draws the image"
    );
}

#[test]
fn an_image_nothing_draws_is_dropped_at_once() {
    let mut cx = TestAppContext::single();
    let image = image();
    let window = open(&mut cx, &image);
    for _ in 0..2 {
        window
            .update(&mut cx, |shows, _, cx| {
                shows.shown = false;
                cx.notify();
            })
            .unwrap();
        cx.run_until_parked();
        window
            .update(&mut cx, |_, window, _| window.refresh())
            .unwrap();
        cx.run_until_parked();
    }
    window
        .update(&mut cx, |_, window, _| {
            assert!(!window.image_drawn(&image));
            window.drop_image(image.clone()).unwrap();
        })
        .unwrap();
    assert!(!in_atlas(&mut cx, window, &image));
}
