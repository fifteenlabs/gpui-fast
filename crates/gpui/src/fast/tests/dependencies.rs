//! Tests of how retained subtrees record what they read. See
//! [`crate::fast::dependencies`].

use crate::fast::dependencies::merge_sorted;
use std::rc::Rc;

#[test]
fn merging_sorted_lists_shares_one_that_holds_the_other() {
    let all: Rc<[u32]> = Rc::from([1, 3, 5, 7]);
    let some: Rc<[u32]> = Rc::from([3, 7]);
    let empty: Rc<[u32]> = Rc::from([]);

    assert!(Rc::ptr_eq(&merge_sorted(&all, &some), &all));
    assert!(Rc::ptr_eq(&merge_sorted(&some, &all), &all));
    assert!(Rc::ptr_eq(&merge_sorted(&all, &empty), &all));
    assert!(Rc::ptr_eq(&merge_sorted(&empty, &some), &some));

    let other: Rc<[u32]> = Rc::from([2, 3, 8]);
    assert_eq!(&*merge_sorted(&all, &other), &[1, 2, 3, 5, 7, 8]);
    assert_eq!(&*merge_sorted(&other, &all), &[1, 2, 3, 5, 7, 8]);
}

/// A subtree built again, frame after frame, reading what it read before,
/// takes the list of what it read that it had, rather than a new one.
#[test]
fn a_subtree_built_again_reading_the_same_takes_the_list_it_had() {
    use crate::{
        AppContext as _, Context, Entity, IntoElement, ParentElement as _, Render, TestAppContext,
        Window, div,
    };

    struct Shared(u32);
    struct Reader(Entity<Shared>);
    impl Render for Reader {
        fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            div().child(format!("{}", self.0.read(cx).0))
        }
    }
    struct Readers(Vec<Entity<Reader>>);
    impl Render for Readers {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().children(self.0.iter().cloned())
        }
    }

    let mut cx = TestAppContext::single();
    let window = cx.add_window(|_, cx| {
        let shared = cx.new(|_| Shared(0));
        Readers((0..3).map(|_| cx.new(|_| Reader(shared.clone()))).collect())
    });
    let lists = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            // Every reader built again, reading what it read before.
            window.refresh();
            window.draw(cx).clear(cx);
            window
                .rendered_frame
                .retained
                .records
                .iter()
                .filter(|record| record.own_dependencies.entities.len() == 2)
                .map(|record| record.own_dependencies.entities.clone())
                .collect::<Vec<_>>()
        })
        .unwrap()
    };
    let first = lists(&mut cx);
    assert_eq!(first.len(), 3, "each reader reads itself and the model");
    let second = lists(&mut cx);
    assert_eq!(second.len(), 3);
    for (first, second) in first.iter().zip(&second) {
        assert!(
            Rc::ptr_eq(first, second),
            "a subtree built again reading the same should take the list it had"
        );
    }
}

/// A view can read an entity it does not hold, as a store's view reads an
/// entry the store holds. Taking the entry out and notifying it draws the
/// view again, though the entry is released before the window draws, and the
/// released entry is forgotten once the window has drawn.
#[test]
fn a_view_that_read_an_entity_released_since_is_built_again() {
    use crate::{
        AppContext as _, Context, Entity, IntoElement, Render, TestAppContext, Window, div,
    };
    use std::cell::RefCell;

    struct Value(u32);
    type Held = Rc<RefCell<Option<Entity<Value>>>>;
    struct Probe {
        held: Held,
        seen: Rc<RefCell<Vec<Option<u32>>>>,
    }
    impl Render for Probe {
        fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            let value = self.held.borrow().clone();
            self.seen
                .borrow_mut()
                .push(value.map(|value| value.read(cx).0));
            div()
        }
    }

    let mut cx = TestAppContext::single();
    let held: Held = Rc::default();
    let seen = Rc::new(RefCell::new(Vec::new()));
    cx.update(|cx| *held.borrow_mut() = Some(cx.new(|_| Value(1))));
    cx.add_window({
        let held = held.clone();
        let seen = seen.clone();
        move |_, _| Probe { held, seen }
    });
    cx.run_until_parked();
    assert_eq!(seen.borrow().last(), Some(&Some(1)));

    cx.update(|cx| {
        let value = held.borrow_mut().take().expect("the value is held");
        value.update(cx, |_, cx| cx.notify());
    });
    cx.run_until_parked();
    assert_eq!(seen.borrow().last(), Some(&None));
    assert_eq!(
        cx.update(|cx| crate::fast::dependencies::released_count(cx)),
        0,
        "the released value is forgotten once the window has drawn"
    );
}
