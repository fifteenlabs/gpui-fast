use crate::{
    AppContext as _, Context, Entity, IntoElement, ParentElement as _, Render, StyleRefinement,
    Styled as _, TestAppContext, Window, WindowHandle, canvas, div, px,
};
use std::{cell::Cell, rc::Rc};

struct Label {
    builds: Rc<Cell<usize>>,
    add_fonts_in_prepaint: Rc<Cell<bool>>,
}

impl Render for Label {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.builds.set(self.builds.get() + 1);
        let add_fonts = self.add_fonts_in_prepaint.clone();
        div()
            .size_full()
            .child("first cached line")
            .child(div().child("second cached line"))
            .child(
                canvas(
                    move |_, _, cx| {
                        if add_fonts.replace(false) {
                            cx.text_system().add_fonts(Vec::new()).unwrap();
                        }
                    },
                    |_, _, _, _| {},
                )
                .size(px(10.)),
            )
    }
}

struct Page {
    header: u32,
    label: Entity<Label>,
}

impl Render for Page {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size(px(300.))
            .child(format!("header {}", self.header))
            .child(div().child(format!("subheader {}", self.header)))
            .child(div().child(format!("caption {}", self.header)))
            .child(
                self.label
                    .clone()
                    .cached(StyleRefinement::default().w(px(200.)).h(px(60.))),
            )
    }
}

struct Fixture {
    window: WindowHandle<Page>,
    builds: Rc<Cell<usize>>,
    add_fonts_in_prepaint: Rc<Cell<bool>>,
}

fn fixture(cx: &mut TestAppContext, add_fonts_in_first_prepaint: bool) -> Fixture {
    let builds = Rc::new(Cell::new(0));
    let add_fonts_in_prepaint = Rc::new(Cell::new(add_fonts_in_first_prepaint));
    let window = cx.add_window({
        let builds = builds.clone();
        let add_fonts_in_prepaint = add_fonts_in_prepaint.clone();
        move |_, cx| Page {
            header: 0,
            label: cx.new(|_| Label {
                builds,
                add_fonts_in_prepaint,
            }),
        }
    });
    Fixture {
        window,
        builds,
        add_fonts_in_prepaint,
    }
}

fn draw(cx: &mut TestAppContext, window: WindowHandle<Page>) -> Vec<String> {
    cx.update_window(window.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
        window.describe_rendered_frame()
    })
    .unwrap()
}

fn change_header_and_draw(cx: &mut TestAppContext, window: WindowHandle<Page>) -> Vec<String> {
    window
        .update(cx, |page, _, cx| {
            page.header += 1;
            cx.notify();
        })
        .unwrap();
    draw(cx, window)
}

fn add_fonts(cx: &mut TestAppContext) {
    cx.update(|cx| cx.text_system().add_fonts(Vec::new()).unwrap());
}

fn reused_any(cx: &mut TestAppContext, window: WindowHandle<Page>) -> bool {
    cx.update_window(window.into(), |_, window, _| {
        window.rendered_frame.retained.reused_any()
    })
    .unwrap()
}

#[test]
fn a_view_reused_after_fonts_were_added_between_frames_keeps_its_ranges() {
    let mut cx = TestAppContext::single();
    let fixture = fixture(&mut cx, false);
    draw(&mut cx, fixture.window);
    let first = change_header_and_draw(&mut cx, fixture.window);
    assert!(reused_any(&mut cx, fixture.window));
    let builds = fixture.builds.get();

    add_fonts(&mut cx);
    for _ in 0..3 {
        let frame = change_header_and_draw(&mut cx, fixture.window);
        assert!(reused_any(&mut cx, fixture.window));
        assert_eq!(frame.len(), first.len());
    }
    assert_eq!(fixture.builds.get(), builds);
}

#[test]
fn a_view_reused_after_fonts_were_added_while_it_was_prepainted_keeps_its_ranges() {
    let mut cx = TestAppContext::single();
    let fixture = fixture(&mut cx, true);
    let first = draw(&mut cx, fixture.window);
    assert!(!fixture.add_fonts_in_prepaint.get());
    let builds = fixture.builds.get();

    for _ in 0..3 {
        let frame = change_header_and_draw(&mut cx, fixture.window);
        assert!(reused_any(&mut cx, fixture.window));
        assert_eq!(frame.len(), first.len());
    }
    assert_eq!(fixture.builds.get(), builds);
}
