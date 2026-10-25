//! Whether a focus handle holds the focused one, asked while a window draws.
//!
//! [`crate::FocusHandle::contains_focused`] and
//! [`crate::FocusHandle::within_focused`] answer from the last frame's
//! dispatch tree, as the frame being drawn has none yet. When focus moves to
//! an element shown in the same update, as a dialog's input is, the view
//! around it asks before the element is in any tree and is told no. Upstream,
//! that view is rendered again with the next notification anywhere in the
//! window, and asks again. A retained view is not.
//!
//! So the questions views ask while the window draws are kept, with their
//! answers, and asked again of the frame drawn. The view that asked one whose
//! answer changed is notified.

use std::cell::RefCell;

use crate::{App, EntityId, FocusId, Window};

/// A question a view asked while the window drew: whether `parent` holds
/// `child`.
struct FocusQuery {
    view: EntityId,
    parent: FocusId,
    child: FocusId,
    answer: bool,
}

/// The questions asked while the window draws.
#[derive(Default)]
pub(crate) struct FocusQueries(RefCell<Vec<FocusQuery>>);

/// Whether `parent` holds `child` in the last frame drawn, kept to be asked
/// again once the frame being drawn is, if a view asked it while drawing.
pub(crate) fn contains(window: &Window, parent: FocusId, child: FocusId) -> bool {
    let answer = window
        .rendered_frame
        .dispatch_tree
        .focus_contains(parent, child);
    if !window.invalidator.not_drawing()
        && let Some(view) = window.rendered_entity_stack.last()
    {
        window.focus_queries.0.borrow_mut().push(FocusQuery {
            view: *view,
            parent,
            child,
            answer,
        });
    }
    answer
}

/// Asks the questions asked while the window drew of the frame it drew,
/// notifying each view told an answer that is no longer so.
pub(crate) fn finish_frame(window: &mut Window, cx: &mut App) {
    let queries = window.focus_queries.0.take();
    let mut changed: Vec<EntityId> = queries
        .into_iter()
        .filter(|query| {
            window
                .rendered_frame
                .dispatch_tree
                .focus_contains(query.parent, query.child)
                != query.answer
        })
        .map(|query| query.view)
        .collect();
    changed.sort_unstable();
    changed.dedup();
    for view in changed {
        cx.notify(view);
    }
}
