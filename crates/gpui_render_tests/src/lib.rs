//! Pixel-level tests for the gpui renderer.
//!
//! This crate is deliberately empty. gpui cannot depend on `gpui_platform`
//! (that would be a dependency cycle), so a test that drives a real GPU
//! renderer cannot live inside gpui itself and needs a crate of its own.
//!
//! Everything lives under `tests/`: `tests/harness/` renders a gpui scene
//! through the platform's headless renderer and hands back an image with
//! assertions attached, and the test binaries beside it use that harness.
//! All of it is behind `[dev-dependencies]`, so `gpui/test-support` never
//! leaks into a real build.
