//! Views reading the geometry of a scroll container another view lays out.
//!
//! A scroll container writes its bounds, its children's bounds and how far
//! it can scroll into its [`crate::ScrollHandle`] as it is prepainted, and a
//! view reads them back through the handle's getters as it renders, as a
//! "jump to bottom" button beside a list does. Upstream, such a view is
//! rendered again whenever anything in the window is notified, and shows
//! what the last frame laid out. A retained view is rendered again only when
//! something it read changed, and a write made in prepaint notifies nothing.
//!
//! So the handle counts changes of its geometry in a version of its own,
//! which the getters note a read of, and a change that a render read asks
//! for another frame, in which what read it is built again. A change nothing
//! read asks for nothing.

use std::{
    cell::{Cell, RefCell},
    hash::{Hash, Hasher},
    rc::Rc,
};

use collections::FxHasher;

use crate::{
    Bounds, Pixels, ScrollHandleState, Window, fast::dependencies::StateVersion,
    fast::layers::invalidate,
};

/// The geometry a scroll handle's container last laid out, as far as its
/// readers can tell it changed.
#[derive(Default, Debug)]
pub(crate) struct ScrollGeometry {
    version: StateVersion,
    /// A hash of the geometry last laid out.
    laid_out: Cell<u64>,
    /// Whether a render read the geometry since it last changed.
    read: Cell<bool>,
}

impl ScrollGeometry {
    fn note_read(&self) {
        if invalidate::recording_offset_reads() {
            invalidate::note_offset_read(&self.version);
            self.read.set(true);
        }
    }
}

/// Notes the geometry the container of the handle with `state` was laid out
/// with, asking for another frame if it changed since a render read it.
pub(crate) fn laid_out(window: &mut Window, state: &ScrollHandleState) {
    let geometry = &state.geometry;
    let mut hasher = FxHasher::default();
    (&state.child_bounds, state.bounds, state.max_offset).hash(&mut hasher);
    let hash = hasher.finish();
    if geometry.laid_out.replace(hash) == hash {
        return;
    }
    geometry.version.bump();
    if geometry.read.take() {
        window.invalidator.set_dirty(true);
    }
}

/// Notes a read of the offset and the geometry of the handle with `state`.
pub(crate) fn note_read(state: &ScrollHandleState) {
    invalidate::note_offset_read(&state.version);
    state.geometry.note_read();
}

/// The bounds of the handle with `state`, `bounds` as it was laid out, as its
/// container is painted now, noting a read of its geometry.
pub(crate) fn bounds(
    state: &Rc<RefCell<ScrollHandleState>>,
    bounds: Bounds<Pixels>,
) -> Bounds<Pixels> {
    state.borrow().geometry.note_read();
    crate::fast::layers::input::moved(state, Some(bounds)).unwrap_or_default()
}
