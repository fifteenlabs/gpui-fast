//! Tests of the values an element hands to what is drawn inside it through
//! the app.

use std::{cell::RefCell, rc::Rc};

use crate::{
    AnyElement, App, AppContext as _, Bounds, Context, Element, ElementId, Entity, GlobalElementId,
    InspectorElementId, IntoElement, LayoutId, ParentElement as _, Pixels, Render, Styled as _,
    TestAppContext, Window, WindowHandle, canvas, deferred, div, px,
};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Shade(u8);

/// Draws its child inheriting `shade`, as a theme scope does.
struct Inheriting {
    shade: Shade,
    child: AnyElement,
}

impl Inheriting {
    fn within<R>(&mut self, cx: &mut App, f: impl FnOnce(&mut AnyElement, &mut App) -> R) -> R {
        let outer = cx.set_inherited(Some(self.shade));
        let result = f(&mut self.child, cx);
        cx.set_inherited(outer);
        result
    }
}

impl Element for Inheriting {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        (
            self.within(cx, |child, cx| child.request_layout(window, cx)),
            (),
        )
    }

    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        self.within(cx, |child, cx| child.prepaint(window, cx));
    }

    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut (),
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        self.within(cx, |child, cx| child.paint(window, cx));
    }
}

type Seen = Rc<RefCell<Vec<Option<Shade>>>>;

/// Notes the shade it inherits each time it renders, prepaints and paints.
struct Swatch {
    seen: Seen,
}

impl Render for Swatch {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.seen.borrow_mut().push(cx.inherited::<Shade>());
        let prepainted = self.seen.clone();
        let painted = self.seen.clone();
        canvas(
            move |_, _, cx| prepainted.borrow_mut().push(cx.inherited::<Shade>()),
            move |_, _, _, cx| painted.borrow_mut().push(cx.inherited::<Shade>()),
        )
        .size(px(10.))
    }
}

struct Page {
    swatch: Entity<Swatch>,
    deferred: bool,
    renders: usize,
}

impl Render for Page {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders += 1;
        let swatch = div().size(px(10.)).child(self.swatch.clone());
        let child = if self.deferred {
            deferred(swatch).into_any_element()
        } else {
            swatch.into_any_element()
        };
        div().size_full().child(Inheriting {
            shade: Shade(3),
            child,
        })
    }
}

fn page(cx: &mut TestAppContext, deferred: bool) -> (WindowHandle<Page>, Entity<Swatch>, Seen) {
    let seen = Seen::default();
    let window = cx.add_window(|_, cx| Page {
        swatch: cx.new(|_| Swatch { seen: seen.clone() }),
        deferred,
        renders: 0,
    });
    let swatch = window.update(cx, |page, _, _| page.swatch.clone()).unwrap();
    (window, swatch, seen)
}

fn draw(cx: &mut TestAppContext, window: WindowHandle<Page>) {
    cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
        .unwrap();
}

#[test]
fn a_view_built_again_on_its_own_inherits_what_it_was_drawn_with() {
    let mut cx = TestAppContext::single();
    let (window, swatch, seen) = page(&mut cx, false);
    draw(&mut cx, window);
    swatch.update(&mut cx, |_, cx| cx.notify());
    draw(&mut cx, window);

    let renders = window.update(&mut cx, |page, _, _| page.renders).unwrap();
    assert_eq!(
        renders, 1,
        "the page is drawn from last frame around the swatch"
    );
    assert_eq!(*seen.borrow(), [Some(Shade(3)); 6]);
    assert_eq!(cx.update(|cx| cx.inherited::<Shade>()), None);
}

#[test]
fn a_deferred_view_inherits_what_was_set_where_it_was_deferred() {
    let mut cx = TestAppContext::single();
    let (window, swatch, seen) = page(&mut cx, true);
    draw(&mut cx, window);
    swatch.update(&mut cx, |_, cx| cx.notify());
    draw(&mut cx, window);

    assert!(seen.borrow().len() >= 2, "{:?}", seen.borrow());
    assert!(
        seen.borrow().iter().all(|shade| *shade == Some(Shade(3))),
        "{:?}",
        seen.borrow()
    );
    assert_eq!(cx.update(|cx| cx.inherited::<Shade>()), None);
}
