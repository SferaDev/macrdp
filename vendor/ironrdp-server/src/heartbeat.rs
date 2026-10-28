//! Server Heartbeat PDU ([MS-RDPBCGR] 2.2.16.1), so a client can tell a live
//! server from a dead link that still looks connected.
//!
//! [MS-RDPBCGR]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpbcgr/86db1222-6b5c-4001-9923-6b292e1ccb47

use core::time::Duration;

use ironrdp_core::{Encode, EncodeResult, WriteCursor, encode_vec, ensure_size};
use ironrdp_pdu::gcc::ClientEarlyCapabilityFlags;
use ironrdp_pdu::mcs::SendDataIndication;
use ironrdp_pdu::rdp::headers::{BasicSecurityHeader, BasicSecurityHeaderFlags};
use ironrdp_pdu::x224::X224;

/// Heartbeat timing announced to the client in every PDU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeartbeatConfig {
    /// Seconds between heartbeats; also the send interval.
    pub period_secs: u8,
    /// Missed heartbeats that should trigger a client-side warning.
    pub count1: u8,
    /// Further missed heartbeats that should trigger a client-side reconnect.
    pub count2: u8,
}

impl HeartbeatConfig {
    pub fn interval(&self) -> Duration {
        Duration::from_secs(u64::from(self.period_secs.max(1)))
    }
}

/// The security header and heartbeat fields carried in the MCS user data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeartbeatPdu {
    pub period: u8,
    pub count1: u8,
    pub count2: u8,
}

impl HeartbeatPdu {
    const NAME: &'static str = "HeartbeatPdu";
    const FIELDS_SIZE: usize = 4;
}

impl From<HeartbeatConfig> for HeartbeatPdu {
    fn from(c: HeartbeatConfig) -> Self {
        Self {
            period: c.period_secs.max(1),
            count1: c.count1,
            count2: c.count2,
        }
    }
}

impl Encode for HeartbeatPdu {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ensure_size!(in: dst, size: self.size());
        BasicSecurityHeader {
            flags: BasicSecurityHeaderFlags::HEARTBEAT,
        }
        .encode(dst)?;
        dst.write_u8(0); // reserved
        dst.write_u8(self.period);
        dst.write_u8(self.count1);
        dst.write_u8(self.count2);
        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        BasicSecurityHeader::FIXED_PART_SIZE + Self::FIELDS_SIZE
    }
}

/// The channel to send heartbeats on: the MCS message channel, and only for a
/// client that announced `RNS_UD_CS_SUPPORT_HEARTBEAT_PDU`.
pub fn heartbeat_channel(
    early_capability: Option<ClientEarlyCapabilityFlags>,
    message_channel_id: Option<u16>,
) -> Option<u16> {
    early_capability
        .filter(|f| f.contains(ClientEarlyCapabilityFlags::SUPPORT_HEART_BEAT_PDU))
        .and(message_channel_id)
}

/// The full TPKT frame of one Heartbeat PDU.
pub fn encode_heartbeat(pdu: HeartbeatPdu, user_channel_id: u16, message_channel_id: u16) -> anyhow::Result<Vec<u8>> {
    let indication = SendDataIndication {
        initiator_id: user_channel_id,
        channel_id: message_channel_id,
        user_data: encode_vec(&pdu)?.into(),
    };
    Ok(encode_vec(&X224(indication))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_data_matches_the_spec_layout() {
        let pdu = HeartbeatPdu {
            period: 2,
            count1: 3,
            count2: 3,
        };
        let bytes = encode_vec(&pdu).unwrap();
        // flags SEC_HEARTBEAT (LE), flagsHi 0, reserved, period, count1, count2.
        assert_eq!(bytes, [0x00, 0x40, 0x00, 0x00, 0x00, 0x02, 0x03, 0x03]);
    }

    #[test]
    fn frame_is_a_send_data_indication_on_the_message_channel() {
        let pdu = HeartbeatPdu {
            period: 2,
            count1: 3,
            count2: 3,
        };
        let frame = encode_heartbeat(pdu, 1007, 1010).unwrap();
        let decoded = ironrdp_core::decode::<X224<SendDataIndication<'_>>>(&frame).unwrap().0;
        assert_eq!(decoded.initiator_id, 1007);
        assert_eq!(decoded.channel_id, 1010);
        assert_eq!(&decoded.user_data[..], &encode_vec(&pdu).unwrap()[..]);
        // TPKT version 3, then the frame length.
        assert_eq!(frame[0], 3);
        assert_eq!(usize::from(u16::from_be_bytes([frame[2], frame[3]])), frame.len());
    }

    #[test]
    fn only_a_supporting_client_with_a_message_channel_gets_heartbeats() {
        let hb = ClientEarlyCapabilityFlags::SUPPORT_HEART_BEAT_PDU;
        let other = ClientEarlyCapabilityFlags::SUPPORT_ERR_INFO_PDU;
        assert_eq!(heartbeat_channel(Some(hb | other), Some(1010)), Some(1010));
        assert_eq!(heartbeat_channel(Some(other), Some(1010)), None);
        assert_eq!(heartbeat_channel(Some(hb), None), None);
        assert_eq!(heartbeat_channel(None, Some(1010)), None);
    }

    #[test]
    fn period_is_never_zero() {
        let c = HeartbeatConfig {
            period_secs: 0,
            count1: 1,
            count2: 1,
        };
        assert_eq!(HeartbeatPdu::from(c).period, 1);
        assert_eq!(c.interval(), Duration::from_secs(1));
    }
}
