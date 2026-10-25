//! Tests of views reading the window's own state, which no view records
//! reading: a change to it refreshes every view. See
//! [`crate::Window::refresh_for_state`].

use crate::{
    AppContext as _, Context, Entity, FocusHandle, InteractiveElement as _, IntoElement,
    ParentElement as _, Render, Size, Styled as _, TestAppContext, Window, WindowAppearance,
    WindowHandle, canvas, div, px, size,
};
use std::{cell::RefCell, rc::Rc};

struct Counter(u32);
impl Render for Counter {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().child(format!("{}", self.0))
    }
}

fn bump(cx: &mut TestAppContext, counter: &Entity<Counter>) {
    cx.update(|cx| {
        counter.update(cx, |counter, cx| {
            counter.0 += 1;
            cx.notify();
        })
    });
    cx.run_until_parked();
}

fn with_retention<V: 'static>(cx: &mut TestAppContext, window: WindowHandle<V>, retention: bool) {
    window
        .update(cx, |_, window, _| window.set_view_retention(retention))
        .unwrap();
    cx.run_until_parked();
}

/// Shows what it read of the window's appearance beside a nested view:
/// notifying the nested view draws this one again from last frame.
struct ShowsAppearance {
    seen: Rc<RefCell<Vec<WindowAppearance>>>,
    counter: Entity<Counter>,
}
impl Render for ShowsAppearance {
    fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.seen.borrow_mut().push(window.appearance());
        div().child(self.counter.clone())
    }
}

#[test]
fn a_view_reading_the_appearance_is_built_again_when_it_changes() {
    for retention in [false, true] {
        let mut cx = TestAppContext::single();
        let seen = Rc::new(RefCell::new(Vec::new()));
        let counter = cx.new(|_| Counter(0));
        let window = cx.add_window({
            let seen = seen.clone();
            let counter = counter.clone();
            move |_, _| ShowsAppearance { seen, counter }
        });
        with_retention(&mut cx, window, retention);

        let test_window = cx.test_window(window.into());
        cx.update(|_| test_window.simulate_appearance_change(WindowAppearance::Dark));
        cx.run_until_parked();
        bump(&mut cx, &counter);

        assert_eq!(
            seen.borrow().last(),
            Some(&WindowAppearance::Dark),
            "retention {retention}: the view shows the new appearance"
        );
    }
}

/// Shows whether a handle is focused. Drawn before [`Focuses`], which moves
/// focus to the handle as it renders.
struct ShowsFocus {
    handle: FocusHandle,
    seen: Rc<RefCell<Vec<bool>>>,
}
impl Render for ShowsFocus {
    fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.seen.borrow_mut().push(self.handle.is_focused(window));
        div().track_focus(&self.handle)
    }
}

struct Focuses {
    handle: FocusHandle,
    focus: bool,
}
impl Render for Focuses {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if self.focus {
            window.focus(&self.handle, cx);
        }
        div()
    }
}

struct FocusRoot {
    shows: Entity<ShowsFocus>,
    focuses: Entity<Focuses>,
    counter: Entity<Counter>,
}
impl Render for FocusRoot {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .child(self.shows.clone())
            .child(self.focuses.clone())
            .child(self.counter.clone())
    }
}

/// A view drawn before focus moves in the same frame, from a later view's
/// render, is built again once that frame is drawn.
#[test]
fn a_view_drawn_before_focus_moved_in_its_frame_is_built_again() {
    for retention in [false, true] {
        let mut cx = TestAppContext::single();
        let seen = Rc::new(RefCell::new(Vec::new()));
        let handle = cx.update(|cx| cx.focus_handle());
        let counter = cx.new(|_| Counter(0));
        let shows = cx.new(|_| ShowsFocus {
            handle: handle.clone(),
            seen: seen.clone(),
        });
        let focuses = cx.new(|_| Focuses {
            handle: handle.clone(),
            focus: false,
        });
        let window = cx.add_window({
            let focuses = focuses.clone();
            let counter = counter.clone();
            move |_, _| FocusRoot {
                shows,
                focuses,
                counter,
            }
        });
        with_retention(&mut cx, window, retention);
        assert_eq!(seen.borrow().last(), Some(&false));

        cx.update(|cx| {
            focuses.update(cx, |focuses, cx| {
                focuses.focus = true;
                cx.notify();
            })
        });
        cx.run_until_parked();
        bump(&mut cx, &counter);

        assert_eq!(
            seen.borrow().last(),
            Some(&true),
            "retention {retention}: the view shows the handle focused"
        );
    }
}

/// Shows the viewport size it read, and resizes the window once, from a
/// prepaint listener drawn after it.
struct ResizesFromPrepaint {
    seen: Rc<RefCell<Vec<Size<crate::Pixels>>>>,
    resize_to: Rc<RefCell<Option<Size<crate::Pixels>>>>,
}
impl Render for ResizesFromPrepaint {
    fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.seen.borrow_mut().push(window.viewport_size());
        let resize_to = self.resize_to.clone();
        div().child(
            canvas(
                move |_, window, _| {
                    if let Some(size) = resize_to.borrow_mut().take() {
                        window.resize(size);
                    }
                },
                |_, _, _, _| {},
            )
            .size(px(1.)),
        )
    }
}

#[test]
fn a_window_resized_while_it_draws_draws_again_at_its_new_size() {
    for retention in [false, true] {
        let mut cx = TestAppContext::single();
        let seen = Rc::new(RefCell::new(Vec::new()));
        let resize_to = Rc::new(RefCell::new(None));
        let window = cx.add_window({
            let seen = seen.clone();
            let resize_to = resize_to.clone();
            move |_, _| ResizesFromPrepaint { seen, resize_to }
        });
        with_retention(&mut cx, window, retention);

        let new_size = size(px(321.), px(123.));
        *resize_to.borrow_mut() = Some(new_size);
        window.update(&mut cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();
        assert!(resize_to.borrow().is_none(), "the window was resized");
        // The window asks the platform for a frame; a test draws it on the
        // next update.
        cx.update(|_| {});

        assert_eq!(
            seen.borrow().last(),
            Some(&new_size),
            "retention {retention}: a frame at the new size follows, with nothing else changed"
        );
    }
}
