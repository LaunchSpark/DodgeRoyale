//! Small, independent foundations for `DodgeRoyale`.

pub mod collision;
#[cfg(not(target_arch = "wasm32"))]
pub mod compute;
#[cfg(not(target_arch = "wasm32"))]
pub mod database;
pub mod enemy;
pub mod enemy_population;
pub mod enemy_types;
pub mod model;
pub mod observation;
pub mod rng;
pub mod scale;
pub mod simulation;
pub mod torus;
pub mod tween;

#[cfg(any(feature = "graphics", test))]
mod art;
#[cfg(any(feature = "graphics", test))]
mod camera_math;
#[cfg(any(feature = "graphics", test))]
mod glyphs;
pub mod motion;
pub mod settings;
pub mod shatter;

#[cfg(feature = "graphics")]
pub mod game;
