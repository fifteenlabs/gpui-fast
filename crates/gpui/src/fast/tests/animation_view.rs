//! Tests of animations drawn in a view of their own. See
//! [`crate::fast::animation_view`].

use crate::{
    Animation, AnimationExt as _, AnyWindowHandle, App, AppContext as _, Bounds, Context, Element,
    ElementId, GlobalElementId, Hsla, InspectorElementId, IntoElement, LayoutId,
    ParentElement as _, Pixels, Render, RenderOnce, Style, Styled as _, TestAppContext, Window,
    WindowHandle, div, hsla, px, size,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    time::{Duration, Instant},
};

/// How often the elements of a page were asked for each phase.
#[derive(Default)]
struct Calls {
    layouts: Cell<usize>,
    prepaints: Cell<usize>,
    paints: Cell<usize>,
}

impl Calls {
    fn take(&self) -> (usize, usize, usize) {
        (
            self.layouts.take(),
            self.prepaints.take(),
            self.paints.take(),
        )
    }
}

/// An 8 px square that counts each phase it is drawn in.
struct Probe(Rc<Calls>);

impl IntoElement for Probe {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for Probe {
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
        self.0.layouts.set(self.0.layouts.get() + 1);
        let mut style = Style::default();
        style.size = size(px(8.).into(), px(8.).into());
        (window.request_layout(style, None, cx), ())
    }

    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut (),
        _: &mut Window,
        _: &mut App,
    ) {
        self.0.prepaints.set(self.0.prepaints.get() + 1);
    }

    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        _: &mut (),
        window: &mut Window,
        _: &mut App,
    ) {
        self.0.paints.set(self.0.paints.get() + 1);
        window.paint_quad(crate::fill(bounds, hsla(0.6, 0.5, 0.5, 1.)));
    }
}

/// The animated element: a bar whose colour's alpha, and optionally width,
/// follow the animation.
#[derive(Clone, IntoElement)]
struct Swatch {
    hue: f32,
    width: Pixels,
    grows: bool,
    /// Whether the animation repeats, rather than running once for 100 ms.
    repeats: bool,
    alpha: f32,
    /// The value the animator was last called with, each time it was.
    seen: Rc<RefCell<Vec<f32>>>,
}

const SWATCH_HEIGHT: f32 = 8.;

impl Swatch {
    fn color(&self) -> Hsla {
        hsla(self.hue, 0.8, 0.5, self.alpha)
    }
}

impl RenderOnce for Swatch {
    fn render(self, _: &mut Window, _: &mut App) -> impl IntoElement {
        div()
            .w(self.width)
            .h(px(SWATCH_HEIGHT))
            .flex_none()
            .bg(self.color())
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Placement {
    /// As `with_animation` alone places it: its frames notify the page.
    InPage,
    /// With `in_own_view`.
    OwnView,
}

/// A page of probes with one animated swatch among them, and one probe after
/// it in a row of its own whose position shows the swatch's width.
struct Page {
    placement: Placement,
    probes: usize,
    calls: Rc<Calls>,
    marker: Rc<Calls>,
    renders: Rc<Cell<usize>>,
    swatch: Swatch,
}

impl Render for Page {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        let grows = self.swatch.grows;
        let base_width = self.swatch.width;
        let animation = if self.swatch.repeats {
            Animation::new(Duration::from_secs(1)).repeat()
        } else {
            Animation::new(Duration::from_millis(100))
        };
        let animated =
            self.swatch
                .clone()
                .with_animation("swatch", animation, move |mut swatch, delta| {
                    swatch.seen.borrow_mut().push(delta);
                    swatch.alpha = 0.1 + 0.9 * delta;
                    if grows {
                        swatch.width = base_width + px(100. * delta);
                    }
                    swatch
                });
        let animated = match self.placement {
            Placement::InPage => animated.into_any_element(),
            Placement::OwnView => animated.in_own_view().into_any_element(),
        };
        div()
            .flex()
            .flex_wrap()
            .w(px(400.))
            .children((0..self.probes).map(|_| Probe(self.calls.clone())))
            .child(
                div()
                    .flex()
                    .w(px(400.))
                    .child(animated)
                    .child(Probe(self.marker.clone())),
            )
    }
}

struct Fixture {
    cx: TestAppContext,
    window: WindowHandle<Page>,
    calls: Rc<Calls>,
    marker: Rc<Calls>,
    renders: Rc<Cell<usize>>,
    seen: Rc<RefCell<Vec<f32>>>,
}

impl Fixture {
    fn new(placement: Placement, probes: usize, grows: bool) -> Self {
        Self::with_animation(placement, probes, grows, true)
    }

    fn with_animation(placement: Placement, probes: usize, grows: bool, repeats: bool) -> Self {
        let mut cx = TestAppContext::single();
        let calls = Rc::new(Calls::default());
        let marker = Rc::new(Calls::default());
        let renders = Rc::new(Cell::new(0));
        let seen = Rc::new(RefCell::new(Vec::new()));
        let window = cx.add_window({
            let (calls, marker, renders, seen) =
                (calls.clone(), marker.clone(), renders.clone(), seen.clone());
            move |_, _| Page {
                placement,
                probes,
                calls,
                marker,
                renders,
                swatch: Swatch {
                    hue: 0.,
                    width: px(20.),
                    grows,
                    repeats,
                    alpha: 1.,
                    seen,
                },
            }
        });
        let mut fixture = Fixture {
            cx,
            window,
            calls,
            marker,
            renders,
            seen,
        };
        fixture.frame();
        fixture.frame();
        fixture.calls.take();
        fixture.marker.take();
        fixture.renders.take();
        fixture.seen.borrow_mut().clear();
        fixture
    }

    fn any_window(&self) -> AnyWindowHandle {
        self.window.into()
    }

    /// Lets 16 ms pass, delivers the animation frames the window asked for,
    /// as the platform's next frame does, and draws the frame.
    fn frame(&mut self) {
        self.cx.executor().advance_clock(Duration::from_millis(16));
        self.cx
            .update_window(self.any_window(), |_, window, cx| {
                window.simulate_next_frame(cx);
                window.draw(cx).clear(cx);
            })
            .unwrap();
    }

    /// The swatch's quad last drawn: its bounds and colour.
    fn swatch(&mut self) -> (Bounds<Pixels>, Hsla) {
        self.cx
            .update_window(self.any_window(), |_, window, _| {
                let scale = window.scale_factor();
                let quads: Vec<_> = window
                    .rendered_frame
                    .scene
                    .quads
                    .iter()
                    .filter(|quad| quad.background.solid.s > 0.7)
                    .map(|quad| {
                        let bounds = quad.bounds;
                        (
                            Bounds::new(
                                crate::point(
                                    px(bounds.origin.x.0 / scale),
                                    px(bounds.origin.y.0 / scale),
                                ),
                                size(
                                    px(bounds.size.width.0 / scale),
                                    px(bounds.size.height.0 / scale),
                                ),
                            ),
                            quad.background.solid,
                        )
                    })
                    .collect();
                assert_eq!(quads.len(), 1, "one swatch is drawn");
                quads[0]
            })
            .unwrap()
    }

    /// Where the probe after the swatch was last laid out.
    fn marker_x(&mut self) -> Pixels {
        self.cx
            .update_window(self.any_window(), |_, window, _| {
                let scale = window.scale_factor();
                let probe = window
                    .rendered_frame
                    .scene
                    .quads
                    .iter()
                    .filter(|quad| quad.background.solid.s <= 0.7)
                    .max_by(|a, b| a.bounds.origin.y.0.total_cmp(&b.bounds.origin.y.0))
                    .expect("the marker is drawn")
                    .bounds;
                px(probe.origin.x.0 / scale)
            })
            .unwrap()
    }
}

/// Without `in_own_view`, every frame of an animation renders the page it is
/// in and asks every element of it for its layout, prepaint and paint again.
/// This is what `in_own_view` is for.
#[test]
fn an_animation_in_a_page_builds_the_whole_page_every_frame() {
    let mut page = Fixture::new(Placement::InPage, 50, false);
    for _ in 0..5 {
        page.frame();
        assert_eq!(page.renders.take(), 1);
        assert_eq!(page.calls.take(), (50, 50, 50));
    }
}

/// With `in_own_view`, an animation's frames build only the animated
/// element: the page around it is not rendered, and none of its other
/// elements is laid out, prepainted or painted again. The animation still
/// advances, and its element is painted with each new value.
#[test]
fn an_animation_in_its_own_view_builds_only_itself() {
    let mut page = Fixture::new(Placement::OwnView, 50, false);
    let (bounds, mut color) = page.swatch();
    for _ in 0..5 {
        page.frame();
        assert_eq!(page.renders.take(), 0, "the page is not rendered");
        assert_eq!(page.calls.take(), (0, 0, 0), "nor are its elements drawn");
        assert_eq!(page.marker.take(), (0, 0, 0));
        assert_eq!(page.seen.borrow_mut().drain(..).count(), 1);
        let (now_bounds, now_color) = page.swatch();
        assert_eq!(now_bounds, bounds, "the swatch is drawn where it was");
        assert!(
            now_color.a > color.a,
            "the swatch is painted at the animation's new value: {} after {}",
            now_color.a,
            color.a
        );
        color = now_color;
    }
}

/// The animation goes on from where it was when the page around it is built
/// again, and shows at once what the page now hands it.
#[test]
fn a_page_built_again_hands_its_own_view_animation_what_it_now_draws() {
    let mut page = Fixture::new(Placement::OwnView, 10, false);
    for _ in 0..3 {
        page.frame();
    }
    let before = *page.seen.borrow().last().unwrap();
    let window = page.window;
    window
        .update(&mut page.cx, |page, _, cx| {
            page.swatch.hue = 0.3;
            cx.notify();
        })
        .unwrap();
    page.frame();
    assert_eq!(page.renders.take(), 1);
    let (_, color) = page.swatch();
    assert_eq!(
        color.h, 0.3,
        "the swatch is drawn in the colour handed to it"
    );
    let after = *page.seen.borrow().last().unwrap();
    assert!(
        after > before,
        "the animation is not started again: {after} after {before}"
    );
    page.seen.borrow_mut().clear();
    page.frame();
    assert_eq!(page.renders.take(), 0);
    assert_eq!(page.swatch().1.h, 0.3, "and keeps it on its own frames");
}

/// A page built again after its animation has finished draws the
/// animation's element with what it now hands it on that same frame, though
/// nothing notifies the animation's own view.
#[test]
fn a_page_built_again_after_its_animation_finished_draws_it_anew() {
    let mut page = Fixture::with_animation(Placement::OwnView, 10, false, false);
    for _ in 0..10 {
        page.frame();
    }
    assert_eq!(page.swatch().1.a, 1., "the animation has finished");
    page.renders.take();
    page.seen.borrow_mut().clear();
    page.frame();
    assert_eq!(page.renders.take(), 0);
    assert!(
        page.seen.borrow().is_empty(),
        "nothing is animated any more"
    );
    let window = page.window;
    window
        .update(&mut page.cx, |page, _, cx| {
            page.swatch.hue = 0.3;
            cx.notify();
        })
        .unwrap();
    page.frame();
    assert_eq!(page.renders.take(), 1);
    assert_eq!(
        page.swatch().1.h,
        0.3,
        "the swatch is drawn in its new colour"
    );
    assert_eq!(page.swatch().1.a, 1., "at the value the animation ended at");
}

/// An animation that changes its element's size still lays out what is
/// around it: the page is built again for it, and the probe after it moves.
#[test]
fn an_own_view_animation_that_changes_its_size_lays_out_the_page_again() {
    for placement in [Placement::InPage, Placement::OwnView] {
        let mut page = Fixture::new(placement, 10, true);
        let mut x = page.marker_x();
        for _ in 0..4 {
            page.frame();
            let (bounds, _) = page.swatch();
            let now = page.marker_x();
            assert!(now > x, "{placement:?}: the probe after the swatch moves");
            assert_eq!(
                page.renders.take(),
                1,
                "{placement:?}: the page is built again"
            );
            assert_eq!(
                now,
                bounds.origin.x + bounds.size.width,
                "{placement:?}: it is laid out right after the swatch"
            );
            x = now;
        }
    }
}

/// What a frame of a spinner costs on a page of 2,000 elements, with and
/// without `in_own_view`. Run with
/// `cargo test -p gpui --lib --release animation_view::measure -- --ignored --nocapture`.
#[test]
#[ignore]
fn measure() {
    const PROBES: usize = 2000;
    const FRAMES: u32 = 200;
    for placement in [Placement::InPage, Placement::OwnView] {
        let mut page = Fixture::new(placement, PROBES, false);
        let start = Instant::now();
        for _ in 0..FRAMES {
            page.frame();
        }
        let elapsed = start.elapsed() / FRAMES;
        let (layouts, prepaints, paints) = page.calls.take();
        println!(
            "{placement:?}: {elapsed:?} a frame, {} page renders, {} / {} / {} element layouts / prepaints / paints a frame",
            page.renders.take() as f64 / FRAMES as f64,
            layouts as f64 / FRAMES as f64,
            prepaints as f64 / FRAMES as f64,
            paints as f64 / FRAMES as f64,
        );
    }
}
