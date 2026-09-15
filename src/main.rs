#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
mod startup;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> color_eyre::eyre::Result<bevy::app::AppExit> {
    native::run()
}

#[cfg(all(target_arch = "wasm32", feature = "graphics"))]
fn main() {
    dodge_royale::game::build_app().run();
}

#[cfg(all(target_arch = "wasm32", not(feature = "graphics")))]
compile_error!("Browser builds require the graphics feature (enabled by default).");
