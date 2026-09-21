//! Screen states and the white wipe that separates them.

use bevy::prelude::*;

use super::art::ink;

/// Which screen the game is showing.
#[derive(States, Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Screen {
    #[default]
    Menu,
    Config,
    Playing,
}

/// The embedded viewer owns movement and restarts; the normal game remains interactive.
#[derive(Resource, Default)]
pub(super) struct WatchMode(pub bool);

/// Marks entities owned by the menu, removed when the menu is left.
#[derive(Component)]
pub(super) struct MenuEntity;

/// Marks entities owned by gameplay, removed when gameplay is left.
#[derive(Component)]
pub(super) struct GameEntity;

/// The overlay that sweeps down the screen between states.
#[derive(Component)]
struct Wipe;

/// The single overlay entity, mutable in both transform and visibility.
type Overlay<'w, 's> = Single<
    'w,
    's,
    (&'static mut Transform, &'static mut Visibility),
    (With<Wipe>, Without<Camera2d>),
>;

/// How far the overlay travels, chosen to cover any reasonable viewport.
const WIPE_SPAN: f32 = 2_400.0;
/// Seconds the overlay takes to cover the screen, and again to clear it.
const WIPE_SECONDS: f32 = 0.22;

/// Progress of the wipe, if one is running.
#[derive(Resource, Default)]
pub(super) struct Transition {
    /// `None` when idle; otherwise 0.0 covering through 2.0 fully cleared.
    progress: Option<f32>,
    /// The screen to switch to at the moment the overlay covers everything.
    target: Option<Screen>,
}

impl Transition {
    /// Begin a wipe towards `target`, ignoring repeat requests.
    pub(super) const fn start(&mut self, target: Screen) {
        if self.progress.is_none() {
            self.progress = Some(0.0);
            self.target = Some(target);
        }
    }

    /// Whether a wipe is currently running.
    pub(super) const fn is_running(&self) -> bool {
        self.progress.is_some()
    }
}

pub(super) struct ScreenPlugin;

impl Plugin for ScreenPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<Screen>()
            .init_resource::<WatchMode>()
            .init_resource::<Transition>()
            .add_systems(Startup, spawn_wipe)
            .add_systems(Update, advance_wipe);
    }
}

/// The overlay is a child of no camera, so it is parked far in front instead.
fn spawn_wipe(mut commands: Commands) {
    commands.spawn((
        Wipe,
        Sprite::from_color(ink(), Vec2::new(WIPE_SPAN * 2.0, WIPE_SPAN)),
        Transform::from_xyz(0.0, WIPE_SPAN, 900.0),
        Visibility::Hidden,
    ));
}

/// Drive the overlay down the screen, switching state at full cover.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Bevy injects system parameters by value"
)]
fn advance_wipe(
    time: Res<Time>,
    mut transition: ResMut<Transition>,
    mut next: ResMut<NextState<Screen>>,
    camera: Single<&Transform, With<Camera2d>>,
    overlay: Overlay,
) {
    let (mut overlay_transform, mut visibility) = overlay.into_inner();
    let Some(progress) = transition.progress else {
        *visibility = Visibility::Hidden;
        return;
    };

    let advanced = progress + time.delta_secs() / WIPE_SECONDS;
    // Crossing 1.0 is the moment the screen is fully covered.
    if progress < 1.0
        && advanced >= 1.0
        && let Some(target) = transition.target.take()
    {
        next.set(target);
    }
    if advanced >= 2.0 {
        transition.progress = None;
        *visibility = Visibility::Hidden;
        return;
    }

    transition.progress = Some(advanced);
    *visibility = Visibility::Visible;
    // 0.0 parks the overlay above the view, 1.0 covers it, 2.0 clears below.
    let offset = WIPE_SPAN * (1.0 - advanced);
    overlay_transform.translation.x = camera.translation.x;
    overlay_transform.translation.y = camera.translation.y + offset;
}
