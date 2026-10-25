//! Tests of whether a focus handle holds the focused one, asked while the
//! window draws. See [`crate::fast::focus_queries`].

use std::{cell::RefCell, rc::Rc};

use crate::{
    AppContext as _, Context, Entity, FocusHandle, InteractiveElement as _, IntoElement,
    ParentElement as _, Render, TestAppContext, Window, div,
};

/// Shows whether its handle holds the focused one, around a dialog.
struct Container {
    handle: FocusHandle,
    dialog: Entity<Dialog>,
    seen: Rc<RefCell<Vec<bool>>>,
}
impl Render for Container {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.seen
            .borrow_mut()
            .push(self.handle.contains_focused(window, cx));
        div().track_focus(&self.handle).child(self.dialog.clone())
    }
}

/// Shows its input once open.
struct Dialog {
    input: FocusHandle,
    open: bool,
}
impl Render for Dialog {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let mut root = div();
        if self.open {
            root = root.child(div().track_focus(&self.input));
        }
        root
    }
}

struct Counter(u32);
impl Render for Counter {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().child(format!("{}", self.0))
    }
}

struct Page {
    container: Entity<Container>,
    counter: Entity<Counter>,
}
impl Render for Page {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .child(self.container.clone())
            .child(self.counter.clone())
    }
}

/// A view asking whether it holds the focus, when focus moves to an element
/// shown in the same update, is told so once that element is drawn.
#[test]
fn a_view_holding_focus_moved_into_it_in_one_update_shows_it() {
    for retention in [false, true] {
        let mut cx = TestAppContext::single();
        let seen = Rc::new(RefCell::new(Vec::new()));
        let input = cx.update(|cx| cx.focus_handle());
        let dialog = cx.new(|_| Dialog {
            input: input.clone(),
            open: false,
        });
        let container = cx.new(|cx| Container {
            handle: cx.focus_handle(),
            dialog: dialog.clone(),
            seen: seen.clone(),
        });
        let counter = cx.new(|_| Counter(0));
        let window = cx.add_window({
            let counter = counter.clone();
            move |_, _| Page { container, counter }
        });
        window
            .update(&mut cx, |_, window, _| window.set_view_retention(retention))
            .unwrap();
        cx.run_until_parked();
        assert_eq!(seen.borrow().last(), Some(&false));

        window
            .update(&mut cx, |_, window, cx| {
                dialog.update(cx, |dialog, cx| {
                    dialog.open = true;
                    cx.notify();
                });
                window.focus(&input, cx);
            })
            .unwrap();
        cx.run_until_parked();
        if retention {
            assert_eq!(
                seen.borrow().last(),
                Some(&true),
                "the container is told once the input is drawn"
            );
        }
        // Upstream shows it once anything is notified.
        cx.update(|cx| counter.update(cx, |_, cx| cx.notify()));
        cx.run_until_parked();
        assert_eq!(
            seen.borrow().last(),
            Some(&true),
            "retention {retention}: the container shows the input focused inside it"
        );
    }
}
