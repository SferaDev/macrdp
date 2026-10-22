//! PS/2 Set 1 scancode → macOS virtual keycode mapping.
//!
//! Split out of `input.rs` so the input handler reads as logic rather than a
//! ~130-line data table. Pure data + pure functions — no FFI, no shared state.

const VK_ANSI_GRAVE: u16 = 0x32;
const VK_ISO_SECTION: u16 = 0x0A;

/// PS/2 Set 1 scancode → macOS virtual keycode, by physical position.
///
/// Reference scancodes: SDL2_scancode.h, kbdscan.txt. Reference virtual
/// keycodes: <HIToolbox/Events.h> (kVK_*). Media keys and IME composition
/// need follow-on work.
///
/// `iso` is whether the Mac translates with an ISO keyboard type. A PC sends
/// 0x29 for the key left of 1 and 0x56 for the key left of Z (`<>` on most
/// European layouts). macOS swaps the two on ISO hardware: there the key left
/// of 1 is `kVK_ISO_Section` and `<>` is `kVK_ANSI_Grave`, and every Apple ISO
/// layout (Spanish, German, French, British) is drawn for that order.
pub(crate) fn scancode_to_cgkeycode(scancode: u8, extended: bool, iso: bool) -> Option<u16> {
    if !extended {
        match scancode {
            0x29 => return Some(if iso { VK_ISO_SECTION } else { VK_ANSI_GRAVE }),
            0x56 => return Some(if iso { VK_ANSI_GRAVE } else { VK_ISO_SECTION }),
            _ => {}
        }
    }
    let entry = if extended {
        SCANCODE_EXTENDED[scancode as usize]
    } else {
        SCANCODE_NORMAL[scancode as usize]
    };
    if entry == NONE {
        None
    } else {
        Some(entry)
    }
}

// Sentinel for "no mapping" — 0xFFFF is well outside the kVK_* range, so we
// can pack the table as a flat `[u16; 256]` and avoid the niche-less
// `Option<u16>` representation (which would still work, just larger and
// slightly worse for branch prediction).
const NONE: u16 = 0xFFFF;
const SCANCODE_NORMAL: [u16; 256] = build_scancode_normal();
const SCANCODE_EXTENDED: [u16; 256] = build_scancode_extended();

const fn build_scancode_normal() -> [u16; 256] {
    let mut t = [NONE; 256];
    t[0x01] = 0x35; // Esc
    t[0x02] = 0x12; // 1
    t[0x03] = 0x13; // 2
    t[0x04] = 0x14; // 3
    t[0x05] = 0x15; // 4
    t[0x06] = 0x17; // 5
    t[0x07] = 0x16; // 6
    t[0x08] = 0x1A; // 7
    t[0x09] = 0x1C; // 8
    t[0x0A] = 0x19; // 9
    t[0x0B] = 0x1D; // 0
    t[0x0C] = 0x1B; // -
    t[0x0D] = 0x18; // =
    t[0x0E] = 0x33; // Backspace
    t[0x0F] = 0x30; // Tab
    t[0x10] = 0x0C; // Q
    t[0x11] = 0x0D; // W
    t[0x12] = 0x0E; // E
    t[0x13] = 0x0F; // R
    t[0x14] = 0x11; // T
    t[0x15] = 0x10; // Y
    t[0x16] = 0x20; // U
    t[0x17] = 0x22; // I
    t[0x18] = 0x1F; // O
    t[0x19] = 0x23; // P
    t[0x1A] = 0x21; // [
    t[0x1B] = 0x1E; // ]
    t[0x1C] = 0x24; // Enter
    t[0x1D] = 0x3B; // Left Ctrl
    t[0x1E] = 0x00; // A
    t[0x1F] = 0x01; // S
    t[0x20] = 0x02; // D
    t[0x21] = 0x03; // F
    t[0x22] = 0x05; // G
    t[0x23] = 0x04; // H
    t[0x24] = 0x26; // J
    t[0x25] = 0x28; // K
    t[0x26] = 0x25; // L
    t[0x27] = 0x29; // ;
    t[0x28] = 0x27; // '
                    // 0x29 and 0x56 depend on the keyboard type; see `scancode_to_cgkeycode`.
    t[0x2A] = 0x38; // Left Shift
    t[0x2B] = 0x2A; // backslash
    t[0x2C] = 0x06; // Z
    t[0x2D] = 0x07; // X
    t[0x2E] = 0x08; // C
    t[0x2F] = 0x09; // V
    t[0x30] = 0x0B; // B
    t[0x31] = 0x2D; // N
    t[0x32] = 0x2E; // M
    t[0x33] = 0x2B; // ,
    t[0x34] = 0x2F; // .
    t[0x35] = 0x2C; // /
    t[0x36] = 0x3C; // Right Shift
    t[0x37] = 0x43; // numpad *
    t[0x38] = 0x3A; // Left Alt (Option)
    t[0x39] = 0x31; // Space
    t[0x3A] = 0x39; // CapsLock
    t[0x3B] = 0x7A; // F1
    t[0x3C] = 0x78; // F2
    t[0x3D] = 0x63; // F3
    t[0x3E] = 0x76; // F4
    t[0x3F] = 0x60; // F5
    t[0x40] = 0x61; // F6
    t[0x41] = 0x62; // F7
    t[0x42] = 0x64; // F8
    t[0x43] = 0x65; // F9
    t[0x44] = 0x6D; // F10
    t[0x45] = 0x47; // Num Lock → kVK_ANSI_KeypadClear, the key in its place on a Mac
    t[0x47] = 0x59; // numpad 7
    t[0x48] = 0x5B; // numpad 8
    t[0x49] = 0x5C; // numpad 9
    t[0x4A] = 0x4E; // numpad -
    t[0x4B] = 0x56; // numpad 4
    t[0x4C] = 0x57; // numpad 5
    t[0x4D] = 0x58; // numpad 6
    t[0x4E] = 0x45; // numpad +
    t[0x4F] = 0x53; // numpad 1
    t[0x50] = 0x54; // numpad 2
    t[0x51] = 0x55; // numpad 3
    t[0x52] = 0x52; // numpad 0
    t[0x53] = 0x41; // numpad .
    t[0x57] = 0x67; // F11
    t[0x58] = 0x6F; // F12
    t[0x59] = 0x51; // numpad = → kVK_ANSI_KeypadEquals
    t[0x64] = 0x69; // F13
    t[0x65] = 0x6B; // F14
    t[0x66] = 0x71; // F15
    t[0x67] = 0x6A; // F16
    t[0x68] = 0x40; // F17
    t[0x69] = 0x4F; // F18
    t[0x6A] = 0x50; // F19
    t[0x6B] = 0x5A; // F20
    t[0x70] = 0x68; // Katakana/Hiragana → kVK_JIS_Kana
    t[0x73] = 0x5E; // JIS Ro, ABNT2 /? → kVK_JIS_Underscore
    t[0x79] = 0x68; // Henkan (right of Space) → kVK_JIS_Kana
    t[0x7B] = 0x66; // Muhenkan (left of Space) → kVK_JIS_Eisu
    t[0x7D] = 0x5D; // JIS Yen → kVK_JIS_Yen
    t[0x7E] = 0x5F; // JIS / ABNT2 keypad comma → kVK_JIS_KeypadComma
    t[0xF1] = 0x66; // LANG2 → kVK_JIS_Eisu
    t[0xF2] = 0x68; // LANG1 → kVK_JIS_Kana
    t
}

const fn build_scancode_extended() -> [u16; 256] {
    let mut t = [NONE; 256];
    t[0x1C] = 0x4C; // numpad Enter → kVK_ANSI_KeypadEnter
    t[0x1D] = 0x3E; // right Ctrl → kVK_RightControl
    t[0x35] = 0x4B; // numpad / → kVK_ANSI_KeypadDivide
    t[0x38] = 0x3D; // right Alt (Option) → kVK_RightOption
    t[0x47] = 0x73; // Home → kVK_Home
    t[0x48] = 0x7E; // Up arrow
    t[0x49] = 0x74; // PageUp
    t[0x4B] = 0x7B; // Left arrow
    t[0x4D] = 0x7C; // Right arrow
    t[0x4F] = 0x77; // End
    t[0x50] = 0x7D; // Down arrow
    t[0x51] = 0x79; // PageDown
    t[0x52] = 0x72; // Insert (no macOS equivalent — map to Help)
    t[0x53] = 0x75; // Delete (forward delete)
    t[0x5B] = 0x37; // left GUI / Windows → kVK_Command
    t[0x5C] = 0x36; // right GUI → kVK_RightCommand
    t[0x5D] = 0x6E; // Menu / Apps → kVK_ContextualMenu
    t
}

/// True for any macOS vk that hardware would mark with
/// `CGEventFlagNumericPad` — keypad digits, operators, decimal,
/// Enter, Clear. The cluster sits in the kVK_ANSI_Keypad* range
/// 0x41..=0x5C with two gaps (0x46, 0x48) where macOS reserves
/// no usage; we just exclude those.
pub(crate) fn is_numeric_pad_vk(vk: u16) -> bool {
    matches!(
        vk,
        0x41 | 0x43
            | 0x45
            | 0x47
            | 0x4B
            | 0x4C
            | 0x4E
            | 0x51
            | 0x52
            | 0x53
            | 0x54
            | 0x55
            | 0x56
            | 0x57
            | 0x58
            | 0x59
            | 0x5B
            | 0x5C
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(scancode: u8, extended: bool) -> Option<u16> {
        let ansi = scancode_to_cgkeycode(scancode, extended, false);
        let iso = scancode_to_cgkeycode(scancode, extended, true);
        if !matches!(scancode, 0x29 | 0x56) || extended {
            assert_eq!(ansi, iso, "only 0x29 and 0x56 depend on the keyboard type");
        }
        ansi
    }

    #[test]
    fn normal_letters_and_modifiers_map() {
        // Spot-check the ANSI block: scancode → kVK_*.
        assert_eq!(map(0x1E, false), Some(0x00)); // A
        assert_eq!(map(0x10, false), Some(0x0C)); // Q
        assert_eq!(map(0x39, false), Some(0x31)); // Space
        assert_eq!(map(0x1C, false), Some(0x24)); // Enter
        assert_eq!(map(0x1D, false), Some(0x3B)); // Left Ctrl
        assert_eq!(map(0x2A, false), Some(0x38)); // Left Shift
    }

    #[test]
    fn extended_flag_selects_a_different_table() {
        // 0x1D is Left Ctrl normally but Right Ctrl when extended.
        assert_eq!(map(0x1D, false), Some(0x3B));
        assert_eq!(map(0x1D, true), Some(0x3E));
        // 0x48 is numpad-8 normally but the Up arrow when extended.
        assert_eq!(map(0x48, false), Some(0x5B));
        assert_eq!(map(0x48, true), Some(0x7E));
        // GUI / Windows key only exists in the extended table.
        assert_eq!(map(0x5B, true), Some(0x37));
        assert_eq!(map(0x5B, false), None);
    }

    #[test]
    fn unmapped_scancodes_return_none() {
        assert_eq!(map(0x00, false), None);
        assert_eq!(map(0xFF, false), None);
        assert_eq!(map(0x46, false), None); // gap (ScrollLock)
        assert_eq!(map(0x00, true), None);
        assert_eq!(map(0xFF, true), None);
    }

    #[test]
    fn numeric_pad_classification() {
        assert!(is_numeric_pad_vk(0x52)); // numpad 0
        assert!(is_numeric_pad_vk(0x59)); // numpad 7
        assert!(is_numeric_pad_vk(0x4C)); // numpad Enter
        assert!(!is_numeric_pad_vk(0x00)); // A
        assert!(!is_numeric_pad_vk(0x24)); // Return
        assert!(!is_numeric_pad_vk(0x46)); // reserved gap in the keypad range
    }

    #[test]
    fn iso_keys_follow_the_macs_keyboard_type() {
        // ISO (every Apple ISO layout): left of 1 is kVK_ISO_Section, `<>` is kVK_ANSI_Grave.
        assert_eq!(scancode_to_cgkeycode(0x29, false, true), Some(0x0A));
        assert_eq!(scancode_to_cgkeycode(0x56, false, true), Some(0x32));
        // ANSI and JIS: the key left of 1 is kVK_ANSI_Grave.
        assert_eq!(scancode_to_cgkeycode(0x29, false, false), Some(0x32));
        assert_eq!(scancode_to_cgkeycode(0x56, false, false), Some(0x0A));
        // E0 29 and E0 56 are not keys.
        assert_eq!(scancode_to_cgkeycode(0x29, true, true), None);
        assert_eq!(scancode_to_cgkeycode(0x56, true, false), None);
    }

    #[test]
    fn jis_and_abnt2_keys_map() {
        assert_eq!(map(0x73, false), Some(0x5E)); // Ro / ABNT2 /?
        assert_eq!(map(0x7D, false), Some(0x5D)); // Yen
        assert_eq!(map(0x7E, false), Some(0x5F)); // keypad comma
        assert_eq!(map(0x70, false), Some(0x68)); // Katakana/Hiragana → Kana
        assert_eq!(map(0x79, false), Some(0x68)); // Henkan → Kana
        assert_eq!(map(0x7B, false), Some(0x66)); // Muhenkan → Eisu
        assert_eq!(map(0xF1, false), Some(0x66)); // LANG2 → Eisu
        assert_eq!(map(0xF2, false), Some(0x68)); // LANG1 → Kana
    }

    #[test]
    fn function_keypad_and_menu_keys_map() {
        let f13_to_f20 = [0x69, 0x6B, 0x71, 0x6A, 0x40, 0x4F, 0x50, 0x5A];
        for (i, vk) in f13_to_f20.into_iter().enumerate() {
            assert_eq!(map(0x64 + i as u8, false), Some(vk), "F{}", 13 + i);
        }
        assert_eq!(map(0x45, false), Some(0x47)); // Num Lock → keypad Clear
        assert_eq!(map(0x59, false), Some(0x51)); // keypad =
        assert!(is_numeric_pad_vk(0x47) && is_numeric_pad_vk(0x51));
        assert_eq!(map(0x5D, true), Some(0x6E)); // Menu → kVK_ContextualMenu
    }

    #[test]
    fn no_two_scancodes_collide_except_the_japanese_ime_aliases() {
        // Kana and Eisu each have two PC spellings (Henkan/Katakana and LANG1;
        // Muhenkan and LANG2); everything else is one key to one keycode.
        let aliases = [0x66u16, 0x68];
        for iso in [false, true] {
            let mut seen = std::collections::HashMap::new();
            for extended in [false, true] {
                for sc in 0u8..=0xFF {
                    if let Some(vk) = scancode_to_cgkeycode(sc, extended, iso) {
                        if let Some(prev) = seen.insert(vk, (sc, extended)) {
                            assert!(
                                aliases.contains(&vk),
                                "vk 0x{vk:02X} from both {prev:?} and {:?}",
                                (sc, extended)
                            );
                        }
                    }
                }
            }
        }
    }
}
