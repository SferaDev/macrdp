//! Keeps the last `SCDisplay` SCK listed for each display, so a capture can start
//! without listing SCK's shareable content.
//!
//! The first listing after a display wake blocks for seconds until SCK catches
//! up with WindowServer, while a filter built from an `SCDisplay` listed earlier
//! needs no listing. A cached display is used
//! only while CoreGraphics still reports it online at the size SCK listed; a
//! capture that fails to start from one falls back to a fresh listing. The
//! cache is refreshed in the background on every wake or display change.
//! `MACRDP_SCK_DISPLAY_CACHE=0` turns it off.

/// Displays remembered at most; a listing adds the displays it saw.
const CAPACITY: usize = 16;

/// How the capture got its `SCDisplay`, for the connect timeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    Cached,
    Listed,
    /// The cached display failed to start a capture and SCK was listed again.
    Relisted,
}

impl Path {
    pub fn as_str(self) -> &'static str {
        match self {
            Path::Cached => "cached",
            Path::Listed => "listed",
            Path::Relisted => "relisted",
        }
    }
}

struct Entry<T> {
    id: u32,
    size_pts: (u32, u32),
    value: T,
}

/// Displays by id with the point size SCK listed them at, newest last.
pub struct Cache<T> {
    entries: Vec<Entry<T>>,
}

impl<T: Clone> Cache<T> {
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds what a listing saw; a display it did not list is kept, since an
    /// asleep display drops out of SCK's list.
    pub fn remember(&mut self, listed: impl IntoIterator<Item = (u32, (u32, u32), T)>) {
        for (id, size_pts, value) in listed {
            self.entries.retain(|e| e.id != id);
            self.entries.push(Entry {
                id,
                size_pts,
                value,
            });
        }
        let excess = self.entries.len().saturating_sub(CAPACITY);
        self.entries.drain(..excess);
    }

    /// The display listed as `id`, if CoreGraphics reports it online
    /// (`live_size_pts`) at the size it was listed at.
    pub fn get(&self, id: u32, live_size_pts: Option<(u32, u32)>) -> Option<T> {
        let live = live_size_pts?;
        self.entries
            .iter()
            .find(|e| e.id == id && e.size_pts == live)
            .map(|e| e.value.clone())
    }

    pub fn forget(&mut self, id: u32) {
        self.entries.retain(|e| e.id != id);
    }
}

/// On unless `MACRDP_SCK_DISPLAY_CACHE` turns it off.
pub fn enabled_by(env: Option<&str>) -> bool {
    !matches!(
        env.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// The displays as the wake watcher polls them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    pub asleep: bool,
    pub online: Vec<u32>,
}

/// Whether SCK should be listed again: the displays woke, or changed while awake.
pub fn needs_refresh(prev: &Snapshot, now: &Snapshot) -> bool {
    !now.asleep && (prev.asleep || prev.online != now.online)
}

#[cfg(target_os = "macos")]
mod macos {
    use super::*;
    use screencapturekit::shareable_content::SCDisplay;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    const WATCH_POLL: Duration = Duration::from_secs(1);

    static CACHE: Mutex<Cache<SCDisplay>> = Mutex::new(Cache::new());

    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| enabled_by(std::env::var("MACRDP_SCK_DISPLAY_CACHE").ok().as_deref()))
    }

    fn cache() -> std::sync::MutexGuard<'static, Cache<SCDisplay>> {
        CACHE.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn remember(displays: &[SCDisplay]) {
        if enabled() {
            cache().remember(
                displays
                    .iter()
                    .map(|d| (d.display_id(), (d.width(), d.height()), d.clone())),
            );
        }
    }

    fn live_size_pts(id: u32) -> Option<(u32, u32)> {
        let b = core_graphics::display::CGDisplay::new(id).bounds();
        (b.size.width > 0.0 && b.size.height > 0.0)
            .then(|| (b.size.width.round() as u32, b.size.height.round() as u32))
    }

    pub fn lookup(id: u32) -> Option<SCDisplay> {
        if !enabled() {
            return None;
        }
        let live = live_size_pts(id);
        cache().get(id, live)
    }

    pub fn forget(id: u32) {
        cache().forget(id);
    }

    fn snapshot() -> Snapshot {
        let mut online = core_graphics::display::CGDisplay::active_displays().unwrap_or_default();
        online.sort_unstable();
        Snapshot {
            asleep: crate::virtual_display::any_display_asleep(),
            online,
        }
    }

    /// Lists SCK at startup and again whenever the displays wake or change.
    pub fn start_watcher() {
        static STARTED: AtomicBool = AtomicBool::new(false);
        if !enabled() || STARTED.swap(true, Ordering::Relaxed) {
            return;
        }
        let spawned = std::thread::Builder::new()
            .name("sck-display-cache".into())
            .spawn(|| {
                let mut prev = snapshot();
                if !prev.asleep {
                    crate::sck_stream::warm_up("startup");
                }
                loop {
                    std::thread::sleep(WATCH_POLL);
                    let now = snapshot();
                    if needs_refresh(&prev, &now) {
                        crate::sck_stream::warm_up(if prev.asleep {
                            "display wake"
                        } else {
                            "display change"
                        });
                    }
                    prev = now;
                }
            });
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "could not start the SCK display cache watcher");
        }
    }
}

#[cfg(target_os = "macos")]
pub use macos::{forget, lookup, remember, start_watcher};

#[cfg(test)]
mod tests {
    use super::*;

    fn listing(entries: &[(u32, u32, u32, &'static str)]) -> Vec<(u32, (u32, u32), &'static str)> {
        entries
            .iter()
            .map(|&(id, w, h, v)| (id, (w, h), v))
            .collect()
    }

    #[test]
    fn a_listed_display_is_cached_while_online_at_its_listed_size() {
        let mut c = Cache::new();
        c.remember(listing(&[(207, 1512, 949, "vd"), (1, 1728, 1117, "panel")]));
        assert_eq!(c.get(207, Some((1512, 949))), Some("vd"));
        assert_eq!(c.get(1, Some((1728, 1117))), Some("panel"));
        assert_eq!(c.get(207, None), None, "offline");
        assert_eq!(
            c.get(207, Some((1920, 1080))),
            None,
            "re-moded since listed"
        );
        assert_eq!(
            c.get(208, Some((1512, 949))),
            None,
            "re-registered under a new id"
        );
    }

    #[test]
    fn a_new_listing_replaces_an_entry_and_keeps_displays_it_missed() {
        let mut c = Cache::new();
        c.remember(listing(&[
            (207, 1512, 949, "old"),
            (1, 1728, 1117, "panel"),
        ]));
        c.remember(listing(&[(207, 1920, 1080, "new")]));
        assert_eq!(c.get(207, Some((1920, 1080))), Some("new"));
        assert_eq!(c.get(207, Some((1512, 949))), None);
        assert_eq!(c.get(1, Some((1728, 1117))), Some("panel"));
    }

    #[test]
    fn a_forgotten_display_is_listed_again() {
        let mut c = Cache::new();
        c.remember(listing(&[(207, 1512, 949, "vd")]));
        c.forget(207);
        assert_eq!(c.get(207, Some((1512, 949))), None);
    }

    #[test]
    fn the_cache_keeps_the_newest_displays_only() {
        let mut c = Cache::new();
        let ids: Vec<u32> = (0..CAPACITY as u32 + 4).collect();
        c.remember(ids.iter().map(|&id| (id, (10, 10), id)));
        assert_eq!(c.get(0, Some((10, 10))), None);
        assert_eq!(c.get(3, Some((10, 10))), None);
        assert_eq!(c.get(4, Some((10, 10))), Some(4));
        assert_eq!(c.entries.len(), CAPACITY);
    }

    #[test]
    fn the_switch_is_on_unless_turned_off() {
        assert!(enabled_by(None));
        assert!(enabled_by(Some("1")));
        assert!(!enabled_by(Some("0")));
        assert!(!enabled_by(Some(" Off ")));
        assert!(!enabled_by(Some("false")));
    }

    #[test]
    fn sck_is_listed_again_on_a_wake_or_a_change_while_awake() {
        let awake = |ids: &[u32]| Snapshot {
            asleep: false,
            online: ids.to_vec(),
        };
        let asleep = |ids: &[u32]| Snapshot {
            asleep: true,
            online: ids.to_vec(),
        };
        assert!(!needs_refresh(&awake(&[207]), &awake(&[207])));
        assert!(needs_refresh(&asleep(&[207]), &awake(&[207])));
        assert!(needs_refresh(&awake(&[207]), &awake(&[208])));
        assert!(!needs_refresh(&awake(&[207]), &asleep(&[207])));
        assert!(!needs_refresh(&asleep(&[207]), &asleep(&[208])));
        assert!(needs_refresh(&asleep(&[207]), &awake(&[208])));
    }
}
