//! Keeps the displays awake while a client is connected, even with `--allow-sleep`.
//!
//! A sleeping display (the lock screen turns it off after 30 s idle) makes WindowServer
//! re-register the virtual display on wake, which stops the capture under a live session.
//! The hold and a wake start on the TCP accept, so the displays are up before the capture starts.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
mod iokit {
    use core_foundation::base::TCFType;
    use core_foundation::string::{CFString, CFStringRef};

    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IOPMAssertionCreateWithName(
            kind: CFStringRef,
            level: u32,
            name: CFStringRef,
            id: *mut u32,
        ) -> i32;
        fn IOPMAssertionRelease(id: u32) -> i32;
    }

    const LEVEL_ON: u32 = 255;

    pub fn hold() -> Option<u32> {
        let kind = CFString::from_static_string("PreventUserIdleDisplaySleep");
        let name = CFString::from_static_string("macrdp: remote session active");
        let mut id = 0;
        let rc = unsafe {
            IOPMAssertionCreateWithName(
                kind.as_concrete_TypeRef(),
                LEVEL_ON,
                name.as_concrete_TypeRef(),
                &mut id,
            )
        };
        (rc == 0).then_some(id)
    }

    pub fn release(id: u32) {
        unsafe { IOPMAssertionRelease(id) };
    }
}

/// How long an accepted connection holds the displays while it authenticates.
const ACCEPT_HOLD: Duration = Duration::from_secs(10);
/// How long an authenticated client holds the displays before its session starts.
const CONNECT_HOLD: Duration = Duration::from_secs(30);
/// How often one peer's accepts may wake the displays, so a port scan can't keep them on.
const EARLY_WAKE_INTERVAL: Duration = Duration::from_secs(30);
/// Peers remembered by the early-wake limiter; past this, new peers don't wake.
const EARLY_WAKE_PEERS: usize = 256;
/// How long an early wake watches for the displays to come up.
const WAKE_TIMEOUT: Duration = Duration::from_secs(5);
const WAKE_POLL: Duration = Duration::from_millis(20);
/// How long the hold outlasts a session, so a capture rebuild's 1 -> 0 -> 1 flap keeps it.
const LINGER: Duration = Duration::from_secs(10);

#[derive(Default)]
struct Holds {
    session: bool,
    until: Option<Instant>,
    assertion: Option<u32>,
}

impl Holds {
    fn wanted(&self, now: Instant) -> bool {
        self.session || self.until.is_some_and(|t| now < t)
    }

    fn extend(&mut self, now: Instant, by: Duration) {
        let t = now + by;
        self.until = Some(self.until.map_or(t, |u| u.max(t)));
    }
}

static HOLDS: Mutex<Holds> = Mutex::new(Holds {
    session: false,
    until: None,
    assertion: None,
});

fn update(f: impl FnOnce(&mut Holds, Instant)) {
    let mut holds = HOLDS.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    f(&mut holds, now);
    let wanted = holds.wanted(now);
    #[cfg(target_os = "macos")]
    match (wanted, holds.assertion) {
        (true, None) => {
            holds.assertion = iokit::hold();
            match holds.assertion {
                Some(id) => {
                    tracing::info!(assertion = id, "keeping the displays awake for the session")
                }
                None => tracing::warn!("could not create the display-awake assertion"),
            }
        }
        (false, Some(id)) => {
            iokit::release(id);
            holds.assertion = None;
            tracing::info!(assertion = id, "released the display-awake assertion");
        }
        _ => {}
    }
    #[cfg(not(target_os = "macos"))]
    let _ = wanted;
}

fn recheck_after(delay: Duration) {
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        update(|_, _| {});
    });
}

/// Lets each peer wake the displays at most once per [`EARLY_WAKE_INTERVAL`].
#[derive(Default)]
struct EarlyWakes {
    last: HashMap<IpAddr, Instant>,
}

impl EarlyWakes {
    fn allow(&mut self, ip: IpAddr, now: Instant) -> bool {
        self.last
            .retain(|_, t| now.saturating_duration_since(*t) < EARLY_WAKE_INTERVAL);
        if self.last.contains_key(&ip) || self.last.len() >= EARLY_WAKE_PEERS {
            return false;
        }
        self.last.insert(ip, now);
        true
    }
}

static EARLY_WAKES: Mutex<Option<EarlyWakes>> = Mutex::new(None);

/// Called on a TCP accept the auth guard let through, before TLS and CredSSP:
/// wakes the displays like a keypress (never unlocks) and holds them awake.
pub fn client_connecting(ip: IpAddr) {
    let allowed = EARLY_WAKES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(EarlyWakes::default)
        .allow(ip.to_canonical(), Instant::now());
    if allowed {
        update(|h, now| h.extend(now, ACCEPT_HOLD));
        recheck_after(ACCEPT_HOLD);
    }
    let _ = std::thread::Builder::new()
        .name("display-wake".into())
        .spawn(move || watch_wake(allowed));
}

/// Wakes the displays when `wake` and marks the timeline once they are up.
fn watch_wake(wake: bool) {
    let asleep = crate::virtual_display::any_display_asleep();
    if wake {
        crate::virtual_display::wake_display();
    }
    if !asleep {
        crate::connect_timeline::mark(crate::connect_timeline::Step::Awake);
    }
    if !asleep || !wake {
        return;
    }
    let started = Instant::now();
    while crate::virtual_display::any_display_asleep() && started.elapsed() < WAKE_TIMEOUT {
        std::thread::sleep(WAKE_POLL);
    }
    let waited_ms = started.elapsed().as_millis() as u64;
    if crate::virtual_display::any_display_asleep() {
        tracing::warn!(
            waited_ms,
            "the displays did not wake for a connecting client"
        );
        return;
    }
    crate::connect_timeline::mark(crate::connect_timeline::Step::Awake);
    tracing::info!(waited_ms, "woke the displays for a connecting client");
    #[cfg(target_os = "macos")]
    crate::sck_stream::warm_up();
}

/// Called once a client authenticates, ahead of its capture: holds the displays
/// awake and starts waking them, so the capture finds them up.
pub fn client_authenticated() {
    update(|h, now| h.extend(now, CONNECT_HOLD));
    recheck_after(CONNECT_HOLD);
    if crate::virtual_display::any_display_asleep() {
        crate::virtual_display::wake_display();
    }
}

/// Called on the first client's arrival.
pub fn session_started() {
    update(|h, _| h.session = true);
}

/// Called when the last client leaves.
pub fn session_ended() {
    update(|h, now| {
        h.session = false;
        h.extend(now, LINGER);
    });
    recheck_after(LINGER);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hold_lasts_while_a_session_runs_or_until_its_deadline() {
        let t0 = Instant::now();
        let mut h = Holds::default();
        assert!(!h.wanted(t0));
        h.extend(t0, CONNECT_HOLD);
        assert!(h.wanted(t0 + CONNECT_HOLD - Duration::from_millis(1)));
        assert!(!h.wanted(t0 + CONNECT_HOLD));
        h.session = true;
        assert!(h.wanted(t0 + CONNECT_HOLD * 10));
    }

    #[test]
    fn a_peer_wakes_the_displays_once_per_interval() {
        let t0 = Instant::now();
        let a = IpAddr::from([10, 0, 0, 1]);
        let b = IpAddr::from([10, 0, 0, 2]);
        let mut w = EarlyWakes::default();
        assert!(w.allow(a, t0));
        assert!(!w.allow(a, t0 + Duration::from_secs(1)));
        assert!(w.allow(b, t0 + Duration::from_secs(1)));
        assert!(!w.allow(a, t0 + EARLY_WAKE_INTERVAL - Duration::from_millis(1)));
        assert!(w.allow(a, t0 + EARLY_WAKE_INTERVAL));
    }

    #[test]
    fn a_flood_of_peers_cannot_keep_waking_the_displays() {
        let t0 = Instant::now();
        let mut w = EarlyWakes::default();
        let peers = (0..=EARLY_WAKE_PEERS as u32).map(|i| IpAddr::from(i.to_be_bytes()));
        let allowed = peers.filter(|&ip| w.allow(ip, t0)).count();
        assert_eq!(allowed, EARLY_WAKE_PEERS);
        // Forgotten once the interval passes.
        assert!(w.allow(IpAddr::from([192, 168, 0, 1]), t0 + EARLY_WAKE_INTERVAL));
    }

    #[test]
    fn a_shorter_extension_never_cuts_a_longer_hold() {
        let t0 = Instant::now();
        let mut h = Holds::default();
        h.extend(t0, CONNECT_HOLD);
        h.extend(t0, LINGER);
        assert!(h.wanted(t0 + LINGER + Duration::from_secs(1)));
    }
}
