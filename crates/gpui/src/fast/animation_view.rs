//! Animations drawn in a view of their own, so that their frames build
//! nothing but the animated element.
//!
//! An [`AnimationElement`] asks for its next frame with
//! [`Window::request_animation_frame`], which notifies the view the element
//! is drawn in. That view is built again on every frame the animation runs:
//! its whole `render`, and the request for layout, prepaint and paint of
//! every element it holds, however little of it the animation changes. A
//! spinner in a list row, or in a page of components, builds the whole page
//! sixty times a second.
//!
//! An element can only be built again without the view around it if the code
//! that made it can run again without that view; GPUI's elements are values
//! taken by the frame they are drawn in, so the view's `render` is the only
//! code that can. [`AnimationElement::in_own_view`] keeps a copy of the
//! element before it is animated, and draws the animation in a view of its
//! own ([`AnimationHost`]) that builds the animated element from that copy.
//! The animation's frames then notify that view, not the one around it, and
//! the retained-view machinery does the rest: the view around it is drawn
//! from last frame around it ([`crate::fast::splice`]), with only the
//! animation built again in the gap it left. When what the animation changes
//! is only painted — opacity, a transformation, a colour — the animated
//! element asks for the layout it asked for last frame, its layout node is
//! left as it was, and nothing around it is laid out again. When it changes
//! its size, the gap asks for another layout and the view around it is built
//! after all, as it is today.
//!
//! The view around the animation hands it a new copy each time it is built,
//! and the animation is built again with it in the same frame: what that view
//! gives the animated element (a colour, a size) is shown as soon as it
//! changes.

use std::{cell::RefCell, rc::Rc};

use crate::{
    AnimationElement, AnyElement, AnyView, App, AppContext as _, Context, Element, ElementId,
    Entity, GlobalElementId, InspectorElementId, IntoElement, LayoutId, Pixels, Render, Window,
};

/// Builds the animated element, from the copy of the element it animates.
type Build = Rc<dyn Fn() -> AnyElement>;

impl<E: IntoElement + Clone + 'static> AnimationElement<E> {
    /// Draws this animation in a view of its own, so that each of its frames
    /// builds only the animated element rather than the whole view it is
    /// drawn in.
    ///
    /// Without it, every frame of a running animation renders the view the
    /// animation is in again, and asks every element of that view for its
    /// layout, prepaint and paint again. With it, the view around the
    /// animation is drawn from the last frame, and the animated element is
    /// built, laid out at the node it kept, prepainted and painted where it
    /// was. An animation that changes its element's size still works: the
    /// view around it is then built again too.
    ///
    /// The element is cloned for every frame, before the animator is applied
    /// to it. Inside the animated element, [`Window::current_view`] is the
    /// animation's own view.
    pub fn in_own_view(self) -> impl IntoElement {
        let AnimationElement {
            id,
            element,
            animations,
            animator,
        } = self;
        let element = element.expect("an animation element holds its element until it is drawn");
        let animator: Rc<dyn Fn(E, usize, f32) -> E> = Rc::from(animator);
        let animation_id = id.clone();
        let build: Build = Rc::new(move || {
            let animator = animator.clone();
            AnimationElement {
                id: animation_id.clone(),
                element: Some(element.clone()),
                animations: animations.clone(),
                animator: Box::new(move |element, ix, delta| animator(element, ix, delta)),
            }
            .into_any_element()
        });
        OwnViewAnimation { id, build }
    }
}

/// The view an animation placed with [`AnimationElement::in_own_view`] is
/// drawn in. It is rendered by the latest [`Build`] the view around it
/// handed over.
pub(crate) struct AnimationHost {
    build: Rc<RefCell<Build>>,
}

impl Render for AnimationHost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let build = self.build.borrow().clone();
        build()
    }
}

/// The element [`AnimationElement::in_own_view`] returns: it keeps the
/// animation's view in its element state and draws it.
struct OwnViewAnimation {
    id: ElementId,
    build: Build,
}

/// What an [`OwnViewAnimation`] keeps from one frame to the next: its view,
/// and where that view finds the element to build.
struct HostState {
    host: Entity<AnimationHost>,
    build: Rc<RefCell<Build>>,
}

impl Element for OwnViewAnimation {
    type RequestLayoutState = AnyElement;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        Some(self.id.clone())
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let build = self.build.clone();
        let host = window.with_element_state(
            global_id.expect("an animation has an id"),
            |state: Option<HostState>, _| {
                // Handed over outside the entity, so that the view building
                // this one does not count as reading the animation's view,
                // nor the animation's view as changed by it.
                let state = match state {
                    Some(state) => {
                        *state.build.borrow_mut() = build;
                        state
                    }
                    None => {
                        let build = Rc::new(RefCell::new(build));
                        HostState {
                            host: cx.new(|_| AnimationHost {
                                build: build.clone(),
                            }),
                            build,
                        }
                    }
                };
                (state.host.clone(), state)
            },
        );
        // The view around it is being built, and may have handed over an
        // element that draws differently: the animation is built again with
        // it rather than drawn from last frame.
        window.dirty_views.insert(host.entity_id());
        let mut element = AnyView::from(host).into_any_element();
        (element.request_layout(window, cx), element)
    }

    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: crate::Bounds<Pixels>,
        element: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) {
        element.prepaint(window, cx);
    }

    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: crate::Bounds<Pixels>,
        element: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        element.paint(window, cx);
    }
}
