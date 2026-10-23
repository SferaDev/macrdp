//! Server-side key repeat.
//!
//! macOS does not auto-repeat key-downs posted with `CGEventPost`, so holding a
//! key on the client produced one character unless the client sent repeats
//! itself, and then at the client's rate. `Repeater` decides when the held key
//! repeats; the input handler swallows the client's own repeats so the two
//! never stack. Pure state, no FFI, so it is unit-tested on any target; the
//! macOS driver thread lives in `input.rs`.

use std::time::{Duration, Instant};

/// The Mac's repeat settings (System Settings > Keyboard).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RepeatTiming {
    pub delay: Duration,
    pub interval: Duration,
}

impl RepeatTiming {
    /// Settings past this read as "repeat off" (macOS stores Off as a huge value).
    const OFF_THRESHOLD: Duration = Duration::from_secs(5);
    const MIN_INTERVAL: Duration = Duration::from_millis(10);

    /// `None` when key repeat is turned off or the values are unusable.
    pub(crate) fn from_secs(delay: f64, interval: f64) -> Option<Self> {
        if !(delay.is_finite() && interval.is_finite()) || delay < 0.0 || interval <= 0.0 {
            return None;
        }
        let delay = Duration::from_secs_f64(delay);
        let interval = Duration::from_secs_f64(interval);
        if delay >= Self::OFF_THRESHOLD || interval >= Self::OFF_THRESHOLD {
            return None;
        }
        Some(Self {
            delay,
            interval: interval.max(Self::MIN_INTERVAL),
        })
    }
}

/// While a client that repeats keys stops sending repeats for this long without
/// a key-up, it lost track of the key (focus moved away on the client), so the
/// server stops repeating too. Covers the slowest macOS initial delay (1.8 s).
pub(crate) const CLIENT_SILENCE_LIMIT: Duration = Duration::from_secs(2);

/// Ceiling for a client that never repeats: it gives no sign of life while a
/// key is held, so a key-up lost to a dropped link would repeat until TCP gives
/// up. Thirty seconds of a held key is far past any real use.
const MAX_UNCONFIRMED_RUN: Duration = Duration::from_secs(30);

/// How far behind its grid the repeat may fall and still catch up: a few
/// repeats at the fastest setting, so ordinary timer slack never lowers the
/// rate while a real stall (sleep, a hung post) does not type a burst.
const CATCH_UP_LIMIT: Duration = Duration::from_millis(100);

struct Held<A> {
    vk: u16,
    action: A,
    pressed_at: Instant,
    next_fire: Instant,
    last_client_signal: Instant,
}

pub(crate) struct Repeater<A> {
    held: Option<Held<A>>,
    /// The client has sent at least one repeat on this connection, so its
    /// silence while a key is held means the key is no longer held.
    client_repeats: bool,
}

impl<A> Default for Repeater<A> {
    fn default() -> Self {
        Self {
            held: None,
            client_repeats: false,
        }
    }
}

impl<A: Clone> Repeater<A> {
    /// A fresh key-down. Any key already repeating stops, as on a hardware
    /// keyboard. `action` is `None` for a key that must not repeat (a consumed
    /// hotkey, a dead key) and `timing` is `None` when repeat is off.
    pub(crate) fn press(
        &mut self,
        vk: u16,
        action: Option<A>,
        timing: Option<RepeatTiming>,
        now: Instant,
    ) {
        self.held = match (action, timing) {
            (Some(action), Some(timing)) => Some(Held {
                vk,
                action,
                pressed_at: now,
                next_fire: now + timing.delay,
                last_client_signal: now,
            }),
            _ => None,
        };
    }

    /// The client sent another key-down for a key it already holds.
    pub(crate) fn client_repeat(&mut self, vk: u16, now: Instant) {
        self.client_repeats = true;
        if let Some(held) = self.held.as_mut().filter(|h| h.vk == vk) {
            held.last_client_signal = now;
        }
    }

    pub(crate) fn release(&mut self, vk: u16) {
        if self.held.as_ref().is_some_and(|h| h.vk == vk) {
            self.held = None;
        }
    }

    /// Stop repeating (a modifier changed, input state is being reset).
    pub(crate) fn stop(&mut self) {
        self.held = None;
    }

    /// A new connection: nothing is held and the new client may not repeat.
    pub(crate) fn reset_connection(&mut self) {
        self.held = None;
        self.client_repeats = false;
    }

    /// When the driver should next call `fire`.
    pub(crate) fn deadline(&self) -> Option<Instant> {
        self.held.as_ref().map(|h| h.next_fire)
    }

    /// The action to post if a repeat is due at `now`, scheduling the next one.
    /// Repeats stay on the press's grid, so a wake-up that is late by less than
    /// `CATCH_UP_LIMIT` costs no repeats: the missed ones are due at once. A
    /// longer stall posts one repeat and restarts the grid, not a burst.
    pub(crate) fn fire(&mut self, now: Instant, timing: RepeatTiming) -> Option<A> {
        let held = self.held.as_mut()?;
        if now < held.next_fire {
            return None;
        }
        let abandoned = if self.client_repeats {
            now.duration_since(held.last_client_signal) > CLIENT_SILENCE_LIMIT
        } else {
            now.duration_since(held.pressed_at) > MAX_UNCONFIRMED_RUN
        };
        if abandoned {
            self.held = None;
            return None;
        }
        held.next_fire = if now.duration_since(held.next_fire) > CATCH_UP_LIMIT {
            now + timing.interval / 2
        } else {
            held.next_fire + timing.interval
        };
        Some(held.action.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: RepeatTiming = RepeatTiming {
        delay: Duration::from_millis(225),
        interval: Duration::from_millis(30),
    };

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// Fire at every deadline up to `until`, returning the times it posted.
    fn run(r: &mut Repeater<u16>, start: Instant, until: Duration) -> Vec<Duration> {
        let mut posted = Vec::new();
        while let Some(at) = r.deadline() {
            if at > start + until {
                break;
            }
            if r.fire(at, T).is_some() {
                posted.push(at - start);
            }
        }
        posted
    }

    #[test]
    fn held_key_repeats_after_the_delay_at_the_interval() {
        let t0 = Instant::now();
        let mut r = Repeater::default();
        r.press(0x00, Some(0x00), Some(T), t0);
        assert_eq!(r.fire(t0 + ms(224), T), None, "nothing before the delay");
        let posted = run(&mut r, t0, ms(300));
        assert_eq!(posted, vec![ms(225), ms(255), ms(285)]);
    }

    #[test]
    fn release_stops_only_the_repeating_key() {
        let t0 = Instant::now();
        let mut r = Repeater::default();
        r.press(0x00, Some(0x00), Some(T), t0);
        r.release(0x01);
        assert!(
            r.deadline().is_some(),
            "another key's release leaves it repeating"
        );
        r.release(0x00);
        assert_eq!(r.deadline(), None);
    }

    #[test]
    fn a_new_key_takes_over_and_a_non_repeating_key_stops_the_old_one() {
        let t0 = Instant::now();
        let mut r = Repeater::default();
        r.press(0x00, Some(0x00), Some(T), t0);
        r.press(0x0B, Some(0x0B), Some(T), t0 + ms(100));
        assert_eq!(r.fire(t0 + ms(325), T), Some(0x0B));
        r.press(0x30, None, Some(T), t0 + ms(400));
        assert_eq!(r.deadline(), None);
    }

    #[test]
    fn repeat_off_never_schedules() {
        let mut r = Repeater::default();
        r.press(0x00, Some(0x00), None, Instant::now());
        assert_eq!(r.deadline(), None);
    }

    #[test]
    fn client_repeats_do_not_add_server_repeats() {
        // The client repeats every 30 ms from 250 ms; the server's cadence is
        // unchanged by them, so there is one repeat per interval, not two.
        let t0 = Instant::now();
        let mut r = Repeater::default();
        r.press(0x00, Some(0x00), Some(T), t0);
        let mut posted = 0;
        let mut client_at = ms(250);
        let mut now = ms(0);
        while now <= ms(600) {
            if now >= client_at {
                r.client_repeat(0x00, t0 + now);
                client_at += ms(30);
            }
            if r.fire(t0 + now, T).is_some() {
                posted += 1;
            }
            now += ms(1);
        }
        // 225, 255, ..., 585: 13 repeats.
        assert_eq!(posted, 13);
    }

    #[test]
    fn a_repeating_client_going_quiet_stops_the_repeat() {
        let t0 = Instant::now();
        let mut r = Repeater::default();
        r.press(0x00, Some(0x00), Some(T), t0);
        r.client_repeat(0x00, t0 + ms(250));
        assert!(
            r.fire(t0 + ms(2240), T).is_some(),
            "still within the silence limit"
        );
        assert_eq!(r.fire(t0 + ms(2300), T), None);
        assert_eq!(r.deadline(), None, "abandoned keys stop for good");
    }

    #[test]
    fn a_client_known_to_repeat_must_confirm_each_new_key() {
        let t0 = Instant::now();
        let mut r = Repeater::default();
        r.press(0x00, Some(0x00), Some(T), t0);
        r.client_repeat(0x00, t0 + ms(250));
        r.release(0x00);
        // Next key, never confirmed by a client repeat: lost key-up.
        let t1 = t0 + ms(1000);
        r.press(0x01, Some(0x01), Some(T), t1);
        assert!(r.fire(t1 + ms(1990), T).is_some());
        assert_eq!(r.fire(t1 + ms(2010), T), None);
    }

    #[test]
    fn a_client_that_never_repeats_is_capped() {
        let t0 = Instant::now();
        let mut r = Repeater::default();
        r.press(0x00, Some(0x00), Some(T), t0);
        assert!(r.fire(t0 + ms(29_990), T).is_some());
        assert_eq!(r.fire(t0 + ms(30_010), T), None);
    }

    #[test]
    fn reset_connection_forgets_that_the_client_repeats() {
        let t0 = Instant::now();
        let mut r = Repeater::default();
        r.client_repeat(0x00, t0);
        r.reset_connection();
        r.press(0x00, Some(0x00), Some(T), t0);
        assert!(
            r.fire(t0 + ms(5_000), T).is_some(),
            "no silence limit for a new client"
        );
    }

    #[test]
    fn a_late_wake_up_posts_one_repeat_not_a_burst() {
        let t0 = Instant::now();
        let mut r = Repeater::default();
        r.press(0x00, Some(0x00), Some(T), t0);
        assert!(r.fire(t0 + ms(500), T).is_some());
        assert_eq!(r.fire(t0 + ms(500), T), None);
        assert!(r.deadline().unwrap() >= t0 + ms(515));
    }

    /// What tycho reads for KeyRepeat=2, InitialKeyRepeat=15: NSEvent and the
    /// HID system both count the defaults in 1/60 s ticks.
    fn tycho() -> RepeatTiming {
        RepeatTiming::from_secs(15.0 / 60.0, 2.0 / 60.0).unwrap()
    }

    /// Drive the repeater like the repeat thread, with a timer that wakes
    /// `slack` after each deadline, and count the repeats posted while the key
    /// is held for `hold`.
    fn repeats_while_held(timing: RepeatTiming, slack: Duration, hold: Duration) -> usize {
        let t0 = Instant::now();
        let mut r = Repeater::default();
        r.press(0x26, Some(0x26), Some(timing), t0);
        let mut now = t0;
        let mut posted = 0;
        while let Some(at) = r.deadline() {
            if at > now {
                now = at + slack;
            }
            if now > t0 + hold {
                break;
            }
            if r.fire(now, timing).is_some() {
                posted += 1;
            }
        }
        posted
    }

    #[test]
    fn a_held_key_types_at_the_macs_rate() {
        // A 1.5 s hold: the key-down, then repeats at 250 ms + k * 33.3 ms.
        let hold = ms(1500);
        let expected = 38;
        assert_eq!(repeats_while_held(tycho(), Duration::ZERO, hold), expected);
        // Timer slack: 5 ms is the default-QoS leeway, 56 ms what a
        // utility-clamped launchd agent measured on tycho. Without catching up,
        // 56 ms of slack posted one repeat per wake-up, under half the rate.
        for slack in [ms(1), ms(5), ms(20), ms(56), ms(90)] {
            let n = repeats_while_held(tycho(), slack, hold);
            assert!(
                (expected - 3..=expected).contains(&n),
                "{n} repeats with {slack:?} of timer slack, want about {expected}"
            );
        }
    }

    #[test]
    fn a_late_wake_up_catches_up_on_the_grid() {
        let t0 = Instant::now();
        let mut r = Repeater::default();
        r.press(0x00, Some(0x00), Some(T), t0);
        let now = t0 + ms(285);
        assert!(r.fire(now, T).is_some());
        assert_eq!(r.deadline(), Some(t0 + ms(255)), "the missed repeat is due");
        assert!(r.fire(now, T).is_some());
        assert_eq!(r.deadline(), Some(t0 + ms(285)));
        assert!(r.fire(now, T).is_some());
        assert_eq!(r.deadline(), Some(t0 + ms(315)), "back on time");
    }

    #[test]
    fn timing_from_settings() {
        let t = tycho();
        assert_eq!(t.delay.as_micros(), 250_000);
        assert_eq!(t.interval.as_micros(), 33_333);
        let t = RepeatTiming::from_secs(0.225, 0.03).unwrap();
        assert_eq!(t.delay, ms(225));
        assert_eq!(t.interval, ms(30));
        assert_eq!(
            RepeatTiming::from_secs(0.2, 0.001).unwrap().interval,
            ms(10)
        );
        assert_eq!(RepeatTiming::from_secs(4500.0, 0.03), None, "Off");
        assert_eq!(RepeatTiming::from_secs(0.2, f64::NAN), None);
        assert_eq!(RepeatTiming::from_secs(0.2, 0.0), None);
    }
}
