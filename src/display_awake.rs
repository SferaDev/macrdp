//! Keeps the displays awake while a client is connected, even with `--allow-sleep`.
//!
//! A sleeping display (the lock screen turns it off after 30 s idle) makes WindowServer
//! re-register the virtual display on wake, which stops the capture under a live session.

use std::sync::Mutex;

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

static ASSERTION: Mutex<Option<u32>> = Mutex::new(None);

/// Called on the first client's arrival.
pub fn session_started() {
    #[cfg(target_os = "macos")]
    {
        let mut held = ASSERTION.lock().unwrap_or_else(|e| e.into_inner());
        if held.is_none() {
            *held = iokit::hold();
            match *held {
                Some(id) => {
                    tracing::info!(assertion = id, "keeping the displays awake for the session")
                }
                None => tracing::warn!("could not create the display-awake assertion"),
            }
        }
    }
}

/// Called when the last client leaves.
pub fn session_ended() {
    #[cfg(target_os = "macos")]
    if let Some(id) = ASSERTION.lock().unwrap_or_else(|e| e.into_inner()).take() {
        iokit::release(id);
        tracing::info!(assertion = id, "released the display-awake assertion");
    }
}
