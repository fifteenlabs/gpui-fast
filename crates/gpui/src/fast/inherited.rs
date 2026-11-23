//! Values an element hands to everything drawn inside it that is read from
//! the app rather than the window, such as which of several themes a part of
//! the window is drawn in.
//!
//! An element sets one with [`App::set_inherited`] around its children's
//! request for layout, prepaint and paint, and puts back the one it got. What
//! is drawn away from that element still inherits the values it was drawn
//! with, as it inherits the text style: a nested view built again on its own
//! inside a view drawn from last frame (see [`crate::fast::splice`]) and a
//! deferred draw. A view drawn from last frame is reused only where it
//! inherits what it did then.

use std::{
    any::{Any, TypeId},
    cell::RefCell,
    rc::Rc,
};

use smallvec::SmallVec;

use crate::{App, Window};

trait InheritedValue: Any {
    fn as_any(&self) -> &dyn Any;
    fn same(&self, other: &dyn InheritedValue) -> bool;
}

impl<T: Copy + PartialEq + 'static> InheritedValue for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn same(&self, other: &dyn InheritedValue) -> bool {
        other.as_any().downcast_ref::<T>() == Some(self)
    }
}

/// The values set at some point of the element tree, one per type.
#[derive(Clone, Default)]
pub(crate) struct InheritedValues(SmallVec<[(TypeId, Rc<dyn InheritedValue>); 2]>);

impl PartialEq for InheritedValues {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self.0.iter().all(|(type_id, value)| {
                other.0.iter().any(|(other_type, other_value)| {
                    other_type == type_id && value.same(&**other_value)
                })
            })
    }
}

/// The app's values, as the element being drawn inherits them.
#[derive(Default)]
pub(crate) struct AppInherited(RefCell<InheritedValues>);

impl App {
    /// The `T` the element being drawn inherits, as an element around it set
    /// it with [`App::set_inherited`].
    pub fn inherited<T: Copy + PartialEq + 'static>(&self) -> Option<T> {
        self.fast_inherited
            .0
            .borrow()
            .0
            .iter()
            .find(|(type_id, _)| *type_id == TypeId::of::<T>())
            .and_then(|(_, value)| value.as_any().downcast_ref::<T>().copied())
    }

    /// Sets the `T` what is drawn from here on inherits, or clears it with
    /// `None`, returning the one it inherited before for the caller to put
    /// back once its children are drawn. A view nested in them that is built
    /// again on its own, and an element they defer, are drawn with the value
    /// set here.
    pub fn set_inherited<T: Copy + PartialEq + 'static>(&self, value: Option<T>) -> Option<T> {
        let mut values = self.fast_inherited.0.borrow_mut();
        let position = values
            .0
            .iter()
            .position(|(type_id, _)| *type_id == TypeId::of::<T>());
        let previous = position
            .and_then(|position| values.0[position].1.as_any().downcast_ref::<T>().copied());
        if previous == value {
            return previous;
        }
        match (position, value) {
            (Some(position), Some(value)) => values.0[position].1 = Rc::new(value),
            (Some(position), None) => {
                values.0.remove(position);
            }
            (None, Some(value)) => values.0.push((TypeId::of::<T>(), Rc::new(value))),
            (None, None) => {}
        }
        previous
    }
}

/// What the element being drawn inherits.
pub(crate) fn current(cx: &App) -> InheritedValues {
    cx.fast_inherited.0.borrow().clone()
}

/// Whether the element being drawn inherits `values`, compared in place
/// rather than through a copy of what it inherits.
pub(crate) fn inherits(cx: &App, values: &InheritedValues) -> bool {
    *cx.fast_inherited.0.borrow() == *values
}

/// Draws from here on with `values`, returning what to put back with
/// [`leave`].
pub(crate) fn enter(cx: &App, values: &InheritedValues) -> InheritedValues {
    cx.fast_inherited.0.replace(values.clone())
}

/// Puts back what [`enter`] replaced.
pub(crate) fn leave(cx: &App, previous: InheritedValues) {
    cx.fast_inherited.0.replace(previous);
}

/// Notes what the element just deferred with [`Window::defer_draw`]
/// inherits, for it to be drawn with that.
pub(crate) fn note_deferred(window: &mut Window, cx: &App) {
    if let Some(deferred_draw) = window.next_frame.deferred_draws.last_mut() {
        deferred_draw.fast_inherited = current(cx);
    }
}
