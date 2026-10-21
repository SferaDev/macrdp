//! A multi-monitor session as one desktop. The client describes its monitors
//! relative to its primary, which may put some at negative coordinates; the
//! session desktop is their bounding box, and everything placed on it (EGFX
//! surface origins, pointer input) uses coordinates shifted so the box starts
//! at 0,0.

use ironrdp_acceptor::ClientMonitor;

/// One client monitor, placed on the session desktop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Monitor {
    /// Top-left corner on the session desktop, in client pixels.
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub height: u16,
    /// Backing pixels per point for this monitor's virtual display.
    pub scale: u32,
    pub primary: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonitorLayout {
    pub width: u16,
    pub height: u16,
    /// In the client's order, which puts the primary first.
    pub monitors: Vec<Monitor>,
}

impl MonitorLayout {
    /// The layout for two or more client monitors, or `None` for one monitor
    /// or a layout that doesn't fit the protocol's 16-bit desktop.
    pub fn from_client(client: &[ClientMonitor]) -> Option<Self> {
        if client.len() < 2 {
            return None;
        }
        let left = client.iter().map(|m| m.monitor.left).min()?;
        let top = client.iter().map(|m| m.monitor.top).min()?;
        let right = client.iter().map(|m| m.monitor.right).max()?;
        let bottom = client.iter().map(|m| m.monitor.bottom).max()?;
        let monitors = client
            .iter()
            .map(|c| {
                let m = &c.monitor;
                // Right and bottom are inclusive.
                let width = u16::try_from(m.right - m.left + 1).ok()?;
                let height = u16::try_from(m.bottom - m.top + 1).ok()?;
                Some(Monitor {
                    x: u16::try_from(m.left - left).ok()?,
                    y: u16::try_from(m.top - top).ok()?,
                    width,
                    height,
                    scale: crate::capture::virtual_display_scale(
                        width,
                        height,
                        c.desktop_scale_factor.unwrap_or(100),
                    ),
                    primary: m.flags.contains(ironrdp_pdu::gcc::MonitorFlags::PRIMARY),
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            width: u16::try_from(right - left + 1).ok()?,
            height: u16::try_from(bottom - top + 1).ok()?,
            monitors,
        })
    }

    /// The monitor containing a point on the session desktop, falling back to
    /// the nearest one for a point in a gap between monitors.
    pub fn monitor_at(&self, x: u16, y: u16) -> usize {
        let distance = |m: &Monitor| {
            let dx = (i32::from(m.x) - i32::from(x))
                .max(i32::from(x) - (i32::from(m.x) + i32::from(m.width) - 1))
                .max(0);
            let dy = (i32::from(m.y) - i32::from(y))
                .max(i32::from(y) - (i32::from(m.y) + i32::from(m.height) - 1))
                .max(0);
            dx * dx + dy * dy
        };
        (0..self.monitors.len())
            .min_by_key(|&i| distance(&self.monitors[i]))
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use ironrdp_pdu::gcc::{Monitor as GccMonitor, MonitorFlags};

    use super::*;

    fn client(
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
        primary: bool,
        scale: u32,
    ) -> ClientMonitor {
        ClientMonitor {
            monitor: GccMonitor {
                left,
                top,
                right,
                bottom,
                flags: if primary {
                    MonitorFlags::PRIMARY
                } else {
                    MonitorFlags::empty()
                },
            },
            desktop_scale_factor: Some(scale),
        }
    }

    /// Windows App for Mac with an ultrawide primary, a Retina panel to its
    /// lower right, and a portrait monitor to its upper left.
    fn three_monitors() -> MonitorLayout {
        MonitorLayout::from_client(&[
            client(0, 0, 3439, 1439, true, 100),
            client(3440, 800, 6463, 2697, false, 200),
            client(-1080, -227, -1, 1692, false, 100),
        ])
        .unwrap()
    }

    #[test]
    fn shifts_the_layout_so_the_bounding_box_starts_at_zero() {
        let layout = three_monitors();
        assert_eq!((layout.width, layout.height), (7544, 2925));
        let placed: Vec<_> = layout
            .monitors
            .iter()
            .map(|m| (m.x, m.y, m.width, m.height, m.scale, m.primary))
            .collect();
        assert_eq!(
            placed,
            [
                (1080, 227, 3440, 1440, 1, true),
                (4520, 1027, 3024, 1898, 2, false),
                (0, 0, 1080, 1920, 1, false)
            ]
        );
    }

    #[test]
    fn one_monitor_is_not_a_layout() {
        assert_eq!(
            MonitorLayout::from_client(&[client(0, 0, 1919, 1079, true, 100)]),
            None
        );
    }

    #[test]
    fn monitor_at_hit_tests_and_falls_back_to_the_nearest() {
        let layout = three_monitors();
        assert_eq!(layout.monitor_at(1080, 227), 0);
        assert_eq!(layout.monitor_at(5000, 2000), 1);
        assert_eq!(layout.monitor_at(10, 10), 2);
        // Below the primary, left of the Retina panel: nearest is the primary.
        assert_eq!(layout.monitor_at(3000, 2800), 0);
    }
}
