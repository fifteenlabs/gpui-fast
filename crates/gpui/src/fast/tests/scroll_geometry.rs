//! Tests of views reading the geometry of a scroll container another view
//! lays out. See [`crate::fast::scroll_geometry`].

use std::{cell::RefCell, rc::Rc};

use crate::{
    AppContext as _, Context, Entity, InteractiveElement as _, IntoElement, ParentElement as _,
    Pixels, Point, Render, ScrollHandle, StatefulInteractiveElement as _, Styled as _,
    TestAppContext, Window, div, point, px,
};

/// What a reader saw of the handle.
#[derive(Clone, Debug, PartialEq)]
struct Seen {
    max_offset: Point<Pixels>,
    bottom_item: usize,
    bounds_height: Pixels,
}

/// Shows what it read of a handle. Drawn before the scroll container.
struct Reader {
    handle: ScrollHandle,
    seen: Rc<RefCell<Vec<Seen>>>,
}
impl Render for Reader {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.seen.borrow_mut().push(Seen {
            max_offset: self.handle.max_offset(),
            bottom_item: self.handle.bottom_item(),
            bounds_height: self.handle.bounds().size.height,
        });
        div().w(px(10.)).h(px(10.))
    }
}

/// A scroll container of `rows` 20 px rows, `height` tall.
struct Rows {
    handle: ScrollHandle,
    rows: usize,
    height: f32,
}
impl Render for Rows {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("rows")
            .overflow_y_scroll()
            .track_scroll(&self.handle)
            .w(px(100.))
            .h(px(self.height))
            .children((0..self.rows).map(|_| div().h(px(20.))))
    }
}

struct Page {
    reader: Entity<Reader>,
    rows: Entity<Rows>,
    counter: Entity<Counter>,
}
impl Render for Page {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .child(self.reader.clone())
            .child(self.rows.clone())
            .child(self.counter.clone())
    }
}

struct Counter(u32);
impl Render for Counter {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().child(format!("{}", self.0))
    }
}

/// A view reading the geometry of a scroll container drawn after it is
/// built again once the container's rows or height change, with nothing
/// else changed.
#[test]
fn a_view_reading_scroll_geometry_is_built_again_when_it_changes() {
    for retention in [false, true] {
        let mut cx = TestAppContext::single();
        let handle = ScrollHandle::new();
        let seen = Rc::new(RefCell::new(Vec::new()));
        let reader = cx.new(|_| Reader {
            handle: handle.clone(),
            seen: seen.clone(),
        });
        let rows = cx.new(|_| Rows {
            handle: handle.clone(),
            rows: 10,
            height: 110.,
        });
        let counter = cx.new(|_| Counter(0));
        let window = cx.add_window({
            let rows = rows.clone();
            let counter = counter.clone();
            move |_, _| Page {
                reader,
                rows,
                counter,
            }
        });
        window
            .update(&mut cx, |_, window, _| window.set_view_retention(retention))
            .unwrap();
        cx.run_until_parked();
        // Draw what the reader read of the first layout.
        window
            .update(&mut cx, |_, window, _| window.refresh())
            .unwrap();
        cx.run_until_parked();
        assert_eq!(
            seen.borrow().last(),
            Some(&Seen {
                max_offset: point(px(0.), px(90.)),
                bottom_item: 5,
                bounds_height: px(110.),
            })
        );

        for (rows_count, height, expected) in [
            (
                20,
                110.,
                Seen {
                    max_offset: point(px(0.), px(290.)),
                    bottom_item: 5,
                    bounds_height: px(110.),
                },
            ),
            (
                20,
                210.,
                Seen {
                    max_offset: point(px(0.), px(190.)),
                    bottom_item: 10,
                    bounds_height: px(210.),
                },
            ),
        ] {
            cx.update(|cx| {
                rows.update(cx, |rows, cx| {
                    rows.rows = rows_count;
                    rows.height = height;
                    cx.notify();
                })
            });
            cx.run_until_parked();
            if retention {
                // The window asks the platform for a frame; a test draws it
                // on the next update.
                cx.update(|_| {});
                assert_eq!(
                    seen.borrow().last(),
                    Some(&expected),
                    "a frame follows the change: {rows_count} rows, {height} px tall"
                );
            }
            // Upstream shows the change once anything is notified.
            cx.update(|cx| counter.update(cx, |_, cx| cx.notify()));
            cx.run_until_parked();
            assert_eq!(
                seen.borrow().last(),
                Some(&expected),
                "retention {retention}: {rows_count} rows, {height} px tall"
            );
        }
    }
}

/// A container whose geometry nothing read asks for no frame when it
/// changes.
#[test]
fn scroll_geometry_nothing_read_asks_for_no_frame() {
    let mut cx = TestAppContext::single();
    let window = cx.add_window(|_, _| Rows {
        handle: ScrollHandle::new(),
        rows: 10,
        height: 100.,
    });
    cx.run_until_parked();
    window
        .update(&mut cx, |rows, _, cx| {
            rows.rows = 20;
            cx.notify();
        })
        .unwrap();
    cx.run_until_parked();
    let dirty = window
        .update(&mut cx, |_, window, _| window.invalidator.is_dirty())
        .unwrap();
    assert!(!dirty, "a change nothing read asks for no frame");
}

/// Shows a scroll container tracking a handle, counting its renders.
struct Tracks {
    handle: ScrollHandle,
    renders: Rc<RefCell<usize>>,
}
impl Render for Tracks {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        *self.renders.borrow_mut() += 1;
        div()
            .id("tracked")
            .overflow_y_scroll()
            .track_scroll(&self.handle)
            .h(px(50.))
            .children((0..5).map(|_| div().h(px(20.))))
    }
}

struct TracksPage {
    tracks: Entity<Tracks>,
    counter: Entity<Counter>,
}
impl Render for TracksPage {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().child(self.tracks.clone()).child(self.counter.clone())
    }
}

/// A handle marked changed builds the view whose element tracks it again
/// with the next frame.
#[test]
fn a_handle_marked_changed_builds_the_view_tracking_it_again() {
    let mut cx = TestAppContext::single();
    let handle = ScrollHandle::new();
    let renders = Rc::new(RefCell::new(0));
    let tracks = cx.new(|_| Tracks {
        handle: handle.clone(),
        renders: renders.clone(),
    });
    let counter = cx.new(|_| Counter(0));
    cx.add_window({
        let counter = counter.clone();
        move |_, _| TracksPage { tracks, counter }
    });
    cx.run_until_parked();
    let notify_counter = |cx: &mut TestAppContext| {
        cx.update(|cx| counter.update(cx, |_, cx| cx.notify()));
        cx.run_until_parked();
    };
    notify_counter(&mut cx);
    let before = *renders.borrow();

    handle.mark_changed();
    notify_counter(&mut cx);
    assert_eq!(
        *renders.borrow(),
        before + 1,
        "the view tracking it is built again"
    );
    notify_counter(&mut cx);
    assert_eq!(
        *renders.borrow(),
        before + 1,
        "and then drawn from last frame"
    );
}
