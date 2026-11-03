//! Hand-specialized scanner for the o200k regex family: o200k_base
//! (GPT-4o, gpt-oss), the Nemotron-3 variant, and the Kimi (moonshotai K2)
//! variant. o200k and Kimi are wired into dispatch (see
//! [`super::fast_split::FastSplitScheme::from_pattern`]). Nemotron-3 is
//! covered by [`advance_pos`]'s const parameters
//! (`<false, false, true, false>`) but has no pattern const or `advance`
//! wrapper yet, so it is not routed.
//!
//! Faithfully ported from gigatoken
//! (<https://github.com/marcelroed/gigatoken>), MIT License, Copyright (c)
//! 2026 Marcel Rød — its `pretokenize/fast/o200k_family.rs` scalar walker
//! and `pretokenize/unicode.rs` class tables. Shares low-level primitives
//! (`decode_cp`, byte predicates, `scan_numbers_max`) with the sibling
//! [`super::fast_split`] module.
//!
//! The patterns share the shape
//!
//! ```text
//! HAN-RUN?|
//!  [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+SUF?|
//!  [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*SUF?|
//!  \p{N}{1,3} or \p{N}| ?[^\s\p{L}\p{N}]+TAIL|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! where `SUF = (?i:'s|'t|'re|'ve|'m|'ll|'d)` exists in o200k and Kimi
//! (`CONTRACTIONS`), the digit group is `\p{N}{1,3}` for o200k/Kimi vs
//! `\p{N}` for Nemotron (`DIGITS3`), the absorbed punct tail is `[\r\n/]*`
//! for o200k/Nemotron vs `[\r\n]*` for Kimi (`SLASH`), and Kimi alone
//! (`HAN`) prepends `[\p{Han}]+` and intersects both letter brackets with
//! `[^\p{Han}]`. See the differential tests at the bottom.

use std::sync::LazyLock;

use super::fast_split::{decode_cp, is_ascii_ws, is_digit, is_letter, scan_numbers_max};

/// The Kimi (moonshotai Kimi-K2 family) split pattern, as it appears in
/// `tokenizer.json`.
pub(crate) const KIMI_PATTERN: &str = "[\\p{Han}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// The o200k_base split pattern (GPT-4o, gpt-oss), as it appears in
/// `tokenizer.json`.
pub(crate) const O200K_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Advance past one Kimi-scheme token starting at `pos`.
/// `pos` must be `< bytes.len()`; returns `end` with `pos < end <= len`.
#[inline(always)]
pub(super) fn kimi_advance(bytes: &[u8], pos: usize) -> usize {
    // Kimi = o200k with contractions, \p{N}{1,3}, no `/` tail, Han runs.
    advance_pos::<true, true, false, true>(bytes, pos)
}

/// Advance past one o200k-scheme token starting at `pos`.
/// `pos` must be `< bytes.len()`; returns `end` with `pos < end <= len`.
#[inline(always)]
pub(super) fn o200k_advance(bytes: &[u8], pos: usize) -> usize {
    // o200k = contractions, \p{N}{1,3}, `[\r\n/]*` tail, no Han runs.
    advance_pos::<true, true, true, false>(bytes, pos)
}

// ---------------------------------------------------------------------------
// Character classes
// ---------------------------------------------------------------------------

/// The o200k letter-run classes: `Upper` is Lu|Lt (only in the first
/// bracket), `Lower` is Ll (only in the second), `Caseless` is Lm|Lo (in
/// both). Marks (`\p{M}`) are their own class: they join letter runs like
/// `Caseless` but, being outside `\p{L}`, also continue `[^\s\p{L}\p{N}]+`.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum O200kCharClass {
    Upper = 0,
    Lower = 1,
    Caseless = 2,
    Mark = 3,
    Number = 4,
    Whitespace = 5,
    Other = 6,
}

/// The Kimi classes: the o200k classes with `\p{Han}` split out. Han runs
/// get their own leading alternative and are excluded from both letter
/// brackets, but the general-category rules are otherwise Han-blind (a Han
/// numeral still counts toward `\p{N}{1,3}`, a Han symbol still spans a
/// `[^\s\p{L}\p{N}]+` run). Each Han variant records the base class it
/// behaves as outside a Han run.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum KimiCharClass {
    Upper = 0,
    Lower = 1,
    Caseless = 2,
    Mark = 3,
    Number = 4,
    Whitespace = 5,
    Other = 6,
    Han = 7,
    HanNumber = 8,
    HanOther = 9,
}

impl KimiCharClass {
    #[inline(always)]
    fn base(self) -> O200kCharClass {
        match self {
            KimiCharClass::Upper => O200kCharClass::Upper,
            KimiCharClass::Lower => O200kCharClass::Lower,
            KimiCharClass::Caseless | KimiCharClass::Han => O200kCharClass::Caseless,
            KimiCharClass::Mark => O200kCharClass::Mark,
            KimiCharClass::Number | KimiCharClass::HanNumber => O200kCharClass::Number,
            KimiCharClass::Whitespace => O200kCharClass::Whitespace,
            KimiCharClass::Other | KimiCharClass::HanOther => O200kCharClass::Other,
        }
    }

    #[inline(always)]
    fn is_han(self) -> bool {
        self as u8 >= KimiCharClass::Han as u8
    }
}

// ---------------------------------------------------------------------------
// Packed class tables (4 bits per codepoint, 2 per byte, ~544 KiB each)
// ---------------------------------------------------------------------------

/// One `O200kCharClass` byte per codepoint — the unpacked form both table
/// builders start from.
fn o200k_classes_unpacked() -> Vec<u8> {
    use icu_properties::CodePointMapData;
    use icu_properties::CodePointSetData;
    use icu_properties::props::{GeneralCategory, GeneralCategoryGroup, WhiteSpace};

    const N: usize = 0x11_0000;
    let mut classes = vec![O200kCharClass::Other as u8; N];
    let gc = CodePointMapData::<GeneralCategory>::new();
    for (category, class) in [
        (GeneralCategory::UppercaseLetter, O200kCharClass::Upper),
        (GeneralCategory::TitlecaseLetter, O200kCharClass::Upper),
        (GeneralCategory::LowercaseLetter, O200kCharClass::Lower),
        (GeneralCategory::ModifierLetter, O200kCharClass::Caseless),
        (GeneralCategory::OtherLetter, O200kCharClass::Caseless),
    ] {
        for range in gc.iter_ranges_for_value(category) {
            classes[*range.start() as usize..=*range.end() as usize].fill(class as u8);
        }
    }
    for (group, class) in [
        (GeneralCategoryGroup::Mark, O200kCharClass::Mark),
        (GeneralCategoryGroup::Number, O200kCharClass::Number),
    ] {
        for range in gc.iter_ranges_for_group(group) {
            classes[*range.start() as usize..=*range.end() as usize].fill(class as u8);
        }
    }
    for range in CodePointSetData::new::<WhiteSpace>().iter_ranges() {
        classes[*range.start() as usize..=*range.end() as usize]
            .fill(O200kCharClass::Whitespace as u8);
    }
    classes
}

fn pack_nibbles(classes: &[u8]) -> Box<[u8]> {
    let mut packed = vec![0u8; classes.len() / 2].into_boxed_slice();
    for (i, chunk) in classes.chunks_exact(2).enumerate() {
        packed[i] = chunk[0] | (chunk[1] << 4);
    }
    packed
}

static KIMI_CLASS_TABLE: LazyLock<Box<[u8]>> = LazyLock::new(build_kimi_class_table);

fn build_kimi_class_table() -> Box<[u8]> {
    use icu_properties::CodePointMapData;
    use icu_properties::props::Script;

    let mut classes = o200k_classes_unpacked();
    let script = CodePointMapData::<Script>::new();
    for range in script.iter_ranges_for_value(Script::Han) {
        for cp in *range.start()..=*range.end() {
            let slot = &mut classes[cp as usize];
            *slot = if *slot == O200kCharClass::Number as u8 {
                KimiCharClass::HanNumber as u8
            } else if *slot == O200kCharClass::Other as u8 || *slot == O200kCharClass::Mark as u8 {
                // Marks land in HanOther: `&&[^\p{Han}]` evicts them from the
                // letter brackets, leaving only their punct-run role.
                KimiCharClass::HanOther as u8
            } else {
                // Letters (Lo/Lm); no Han char is Lu/Lt/Ll/Whitespace.
                KimiCharClass::Han as u8
            };
        }
    }
    pack_nibbles(&classes)
}

/// Classify a codepoint for the Kimi scheme with one table load. `cp` must
/// be `<= 0x10FFFF` (guaranteed by [`decode_cp`]'s clamp).
#[inline(always)]
fn kimi_class_of(cp: u32) -> KimiCharClass {
    debug_assert!(cp <= 0x10_FFFF);
    // SAFETY: cp <= 0x10FFFF, so (cp >> 1) < 0x110000/2 == table.len().
    let byte = unsafe { *KIMI_CLASS_TABLE.get_unchecked((cp >> 1) as usize) };
    match (byte >> ((cp & 1) << 2)) & 0xF {
        0 => KimiCharClass::Upper,
        1 => KimiCharClass::Lower,
        2 => KimiCharClass::Caseless,
        3 => KimiCharClass::Mark,
        4 => KimiCharClass::Number,
        5 => KimiCharClass::Whitespace,
        6 => KimiCharClass::Other,
        7 => KimiCharClass::Han,
        8 => KimiCharClass::HanNumber,
        _ => KimiCharClass::HanOther,
    }
}

static O200K_CLASS_TABLE: LazyLock<Box<[u8]>> =
    LazyLock::new(|| pack_nibbles(&o200k_classes_unpacked()));

/// Classify a codepoint for the o200k scheme (non-Kimi) with one table load.
#[inline(always)]
fn o200k_class_of(cp: u32) -> O200kCharClass {
    debug_assert!(cp <= 0x10_FFFF);
    let byte = unsafe { *O200K_CLASS_TABLE.get_unchecked((cp >> 1) as usize) };
    match (byte >> ((cp & 1) << 2)) & 0xF {
        0 => O200kCharClass::Upper,
        1 => O200kCharClass::Lower,
        2 => O200kCharClass::Caseless,
        3 => O200kCharClass::Mark,
        4 => O200kCharClass::Number,
        5 => O200kCharClass::Whitespace,
        _ => O200kCharClass::Other,
    }
}

#[inline(always)]
fn is_upper_ascii(b: u8) -> bool {
    b.wrapping_sub(b'A') < 26
}

#[inline(always)]
fn is_tail_byte<const SLASH: bool>(b: u8) -> bool {
    b == b'\r' || b == b'\n' || (SLASH && b == b'/')
}

/// Effective o200k class of `cp` plus whether it is a `[\p{Han}]+` member
/// (always false off the Kimi scheme).
#[inline(always)]
fn family_class_of<const HAN: bool>(cp: u32) -> (O200kCharClass, bool) {
    if HAN {
        let k = kimi_class_of(cp);
        (k.base(), k.is_han())
    } else {
        (o200k_class_of(cp), false)
    }
}

// ---------------------------------------------------------------------------
// Scalar walker
// ---------------------------------------------------------------------------

/// Scan state of a case-structured letter run under leftmost-greedy
/// backtracking. `U`: still inside `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*`;
/// `last_cl_end` is the end offset of the last caseless/mark char (0 =
/// none) — the backtrack split point if the run ends on a strict-upper
/// tail. `L`: inside `[\p{Ll}\p{Lm}\p{Lo}\p{M}]+`, where a strict-upper
/// char ends the token unconditionally.
#[derive(Clone, Copy)]
enum CaseState {
    U { last_cl_end: usize },
    L,
}

#[inline(always)]
fn ascii_letter_state(b: u8) -> CaseState {
    if is_upper_ascii(b) {
        CaseState::U { last_cl_end: 0 }
    } else {
        CaseState::L
    }
}

/// If the char at `pos` is a letter-run member (`\p{L}` or `\p{M}`, minus
/// Han under Kimi), return (offset past it, initial scan state).
#[inline(always)]
fn letter_run_first<const HAN: bool>(bytes: &[u8], pos: usize) -> Option<(usize, CaseState)> {
    let &b = bytes.get(pos)?;
    if is_letter(b) {
        return Some((pos + 1, ascii_letter_state(b)));
    }
    if b >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        match family_class_of::<HAN>(cp) {
            (_, true) => {}
            (O200kCharClass::Upper, _) => return Some((pos + l, CaseState::U { last_cl_end: 0 })),
            (O200kCharClass::Lower, _) => return Some((pos + l, CaseState::L)),
            (O200kCharClass::Caseless | O200kCharClass::Mark, _) => {
                return Some((
                    pos + l,
                    CaseState::U {
                        last_cl_end: pos + l,
                    },
                ));
            }
            _ => {}
        }
    }
    None
}

/// Letter-run continuation with the o200k casing phase automaton. In phase
/// U a strict-upper continues and a strict-lower switches to phase L; in
/// phase L a strict-upper ends the token. A run ending while still in
/// phase U backtracks to the last caseless/mark char, splitting off the
/// trailing strict-upper run ("AxxB" -> `Axx|B` for caseless x,
/// "HTTPResponse" one token).
#[inline(always)]
fn scan_case_run<const HAN: bool>(bytes: &[u8], mut pos: usize, mut st: CaseState) -> usize {
    let len = bytes.len();
    loop {
        while pos < len {
            let b = unsafe { *bytes.get_unchecked(pos) };
            if is_upper_ascii(b) {
                if matches!(st, CaseState::L) {
                    return pos;
                }
                pos += 1;
            } else if is_letter(b) {
                st = CaseState::L;
                pos += 1;
            } else {
                break;
            }
        }
        if pos < len && unsafe { *bytes.get_unchecked(pos) } >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, pos) };
            match family_class_of::<HAN>(cp) {
                (_, true) => break,
                (O200kCharClass::Upper, _) => {
                    if matches!(st, CaseState::L) {
                        return pos;
                    }
                    pos += l;
                }
                (O200kCharClass::Lower, _) => {
                    st = CaseState::L;
                    pos += l;
                }
                (O200kCharClass::Caseless | O200kCharClass::Mark, _) => {
                    pos += l;
                    if let CaseState::U {
                        ref mut last_cl_end,
                    } = st
                    {
                        *last_cl_end = pos;
                    }
                }
                _ => break,
            }
            continue;
        }
        break;
    }
    match st {
        CaseState::U { last_cl_end } if last_cl_end != 0 => last_cl_end,
        _ => pos,
    }
}

/// Attach a `(?i:'s|'t|'re|'ve|'m|'ll|'d)?` suffix to a letter token
/// ending at `end`, when the scheme has contractions.
#[inline(always)]
fn try_suffix<const CONTRACTIONS: bool>(bytes: &[u8], end: usize) -> usize {
    if !CONTRACTIONS || bytes.get(end) != Some(&b'\'') {
        return end;
    }
    match bytes.get(end + 1).map(u8::to_ascii_lowercase) {
        Some(b's' | b'd' | b'm' | b't') => end + 2,
        Some(b'l') if bytes.get(end + 2).map(u8::to_ascii_lowercase) == Some(b'l') => end + 3,
        Some(b'v') if bytes.get(end + 2).map(u8::to_ascii_lowercase) == Some(b'e') => end + 3,
        Some(b'r') if bytes.get(end + 2).map(u8::to_ascii_lowercase) == Some(b'e') => end + 3,
        // U+017F LATIN SMALL LETTER LONG S folds to 's' under `(?i)`.
        Some(0xC5) if bytes.get(end + 2) == Some(&0xBF) => end + 3,
        _ => end,
    }
}

/// `[^\s\p{L}\p{N}]+` continuation (punct, symbols, marks, controls).
#[inline(always)]
fn scan_punct_from<const HAN: bool>(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        while p < len {
            let b = unsafe { *bytes.get_unchecked(p) };
            if b >= 0x80 {
                break;
            }
            if is_letter(b) || is_digit(b) || is_ascii_ws(b) {
                return p;
            }
            p += 1;
        }
        if p < len {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if matches!(
                family_class_of::<HAN>(cp).0,
                O200kCharClass::Other | O200kCharClass::Mark
            ) {
                p += l;
                continue;
            }
        }
        return p;
    }
}

/// The `[\r\n/]*` (or Kimi `[\r\n]*`) tail absorbed after a punct run.
#[inline(always)]
fn scan_tail<const SLASH: bool>(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() && is_tail_byte::<SLASH>(unsafe { *bytes.get_unchecked(pos) }) {
        pos += 1;
    }
    pos
}

/// `[\p{Han}]+` from `pos` (chars of any Han class).
#[inline(always)]
fn scan_han_run(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if b < 0x80 {
            return pos;
        }
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        if !kimi_class_of(cp).is_han() {
            return pos;
        }
        pos += l;
    }
    pos
}

/// Whitespace-led token: `\s*[\r\n]+` | `\s+(?!\S)` | `\s+`, in priority.
#[inline(always)]
fn ws_token_end<const HAN: bool>(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let mut p = start;
    let mut last_nl_end = 0usize;
    let mut last_char_start = start;
    while p < len {
        let b = unsafe { *bytes.get_unchecked(p) };
        if b == b'\r' || b == b'\n' {
            last_char_start = p;
            p += 1;
            last_nl_end = p;
        } else if is_ascii_ws(b) {
            last_char_start = p;
            p += 1;
        } else if b >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if family_class_of::<HAN>(cp).0 == O200kCharClass::Whitespace {
                last_char_start = p;
                p += l;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if last_nl_end != 0 {
        return last_nl_end;
    }
    if p >= len {
        return p;
    }
    if last_char_start > start {
        return last_char_start;
    }
    p
}

/// `\p{N}{1,3}` or `\p{N}` starting at a digit char ending at `first_end`.
#[inline(always)]
fn digit_token_end<const DIGITS3: bool>(bytes: &[u8], first_end: usize) -> usize {
    if DIGITS3 {
        scan_numbers_max::<3>(bytes, first_end, 1)
    } else {
        first_end
    }
}

/// Advance past one token starting at `pos`. `pos` must be `< bytes.len()`.
#[inline(always)]
fn advance_pos<
    const CONTRACTIONS: bool,
    const DIGITS3: bool,
    const SLASH: bool,
    const HAN: bool,
>(
    bytes: &[u8],
    pos: usize,
) -> usize {
    let b0 = unsafe { *bytes.get_unchecked(pos) };

    // Hot path 1: ASCII letter run (empty prefix)
    if is_letter(b0) {
        let e = scan_case_run::<HAN>(bytes, pos + 1, ascii_letter_state(b0));
        return try_suffix::<CONTRACTIONS>(bytes, e);
    }

    // Hot path 2: space prefix
    if b0 == b' ' {
        let Some(&b1) = bytes.get(pos + 1) else {
            return pos + 1;
        };
        if is_letter(b1) {
            let e = scan_case_run::<HAN>(bytes, pos + 2, ascii_letter_state(b1));
            return try_suffix::<CONTRACTIONS>(bytes, e);
        }
        if b1 < 0x80 {
            if is_digit(b1) {
                return pos + 1;
            }
            if is_ascii_ws(b1) {
                return ws_token_end::<HAN>(bytes, pos);
            }
            let p = scan_punct_from::<HAN>(bytes, pos + 2);
            return scan_tail::<SLASH>(bytes, p);
        }
        let (cp, l) = unsafe { decode_cp(bytes, pos + 1) };
        let p1 = pos + 1 + l;
        return match family_class_of::<HAN>(cp) {
            // A Han letter can neither join a letter run nor extend
            // ` ?[^\s\p{L}\p{N}]+`: the space is a lone `\s+` token and the
            // Han run starts after it.
            (O200kCharClass::Caseless, true) => pos + 1,
            (O200kCharClass::Upper, _) => try_suffix::<CONTRACTIONS>(
                bytes,
                scan_case_run::<HAN>(bytes, p1, CaseState::U { last_cl_end: 0 }),
            ),
            (O200kCharClass::Lower, _) => {
                try_suffix::<CONTRACTIONS>(bytes, scan_case_run::<HAN>(bytes, p1, CaseState::L))
            }
            (O200kCharClass::Caseless | O200kCharClass::Mark, _) => try_suffix::<CONTRACTIONS>(
                bytes,
                scan_case_run::<HAN>(bytes, p1, CaseState::U { last_cl_end: p1 }),
            ),
            (O200kCharClass::Whitespace, _) => ws_token_end::<HAN>(bytes, pos),
            (O200kCharClass::Number, _) => pos + 1,
            (O200kCharClass::Other, _) => {
                scan_tail::<SLASH>(bytes, scan_punct_from::<HAN>(bytes, p1))
            }
        };
    }

    // Non-ASCII first char
    if b0 >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        let p0 = pos + l;
        let (class, han) = family_class_of::<HAN>(cp);
        if HAN && han {
            return scan_han_run(bytes, p0);
        }
        return match class {
            O200kCharClass::Upper => try_suffix::<CONTRACTIONS>(
                bytes,
                scan_case_run::<HAN>(bytes, p0, CaseState::U { last_cl_end: 0 }),
            ),
            O200kCharClass::Lower => {
                try_suffix::<CONTRACTIONS>(bytes, scan_case_run::<HAN>(bytes, p0, CaseState::L))
            }
            O200kCharClass::Caseless | O200kCharClass::Mark => try_suffix::<CONTRACTIONS>(
                bytes,
                scan_case_run::<HAN>(bytes, p0, CaseState::U { last_cl_end: p0 }),
            ),
            O200kCharClass::Number => digit_token_end::<DIGITS3>(bytes, p0),
            class => {
                if let Some((e, st)) = letter_run_first::<HAN>(bytes, p0) {
                    return try_suffix::<CONTRACTIONS>(bytes, scan_case_run::<HAN>(bytes, e, st));
                }
                if class == O200kCharClass::Whitespace {
                    ws_token_end::<HAN>(bytes, pos)
                } else {
                    scan_tail::<SLASH>(bytes, scan_punct_from::<HAN>(bytes, p0))
                }
            }
        };
    }

    // ASCII digit
    if is_digit(b0) {
        return digit_token_end::<DIGITS3>(bytes, pos + 1);
    }

    // \r and \n are excluded from the letter-run prefix
    if b0 == b'\r' || b0 == b'\n' {
        return ws_token_end::<HAN>(bytes, pos);
    }

    // Other ASCII whitespace (\t, \x0b, \x0c) may prefix a letter run
    if is_ascii_ws(b0) {
        if let Some((e, st)) = letter_run_first::<HAN>(bytes, pos + 1) {
            return try_suffix::<CONTRACTIONS>(bytes, scan_case_run::<HAN>(bytes, e, st));
        }
        return ws_token_end::<HAN>(bytes, pos);
    }

    // ASCII punct/symbol/control (incl. `'`: o200k has no standalone
    // contraction alternative, so a leading apostrophe is ordinary punct /
    // a letter-run prefix: "'sound" is one token)
    if let Some((e, st)) = letter_run_first::<HAN>(bytes, pos + 1) {
        return try_suffix::<CONTRACTIONS>(bytes, scan_case_run::<HAN>(bytes, e, st));
    }
    scan_tail::<SLASH>(bytes, scan_punct_from::<HAN>(bytes, pos + 1))
}

// ---------------------------------------------------------------------------
// Tests: differential against fancy-regex on the exact Kimi pattern
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn regex_spans(pattern: &str, s: &str) -> Vec<(usize, usize)> {
        let re = fancy_regex::Regex::new(pattern).unwrap();
        re.find_iter(s)
            .map(|m| {
                let m = m.unwrap();
                (m.start(), m.end())
            })
            .collect()
    }

    fn fast_spans(advance: impl Fn(&[u8], usize) -> usize, s: &str) -> Vec<(usize, usize)> {
        let bytes = s.as_bytes();
        let mut spans = Vec::new();
        let mut pos = 0;
        while pos < bytes.len() {
            let end = advance(bytes, pos);
            assert!(
                end > pos && end <= bytes.len(),
                "no progress at {pos} in {s:?}"
            );
            spans.push((pos, end));
            pos = end;
        }
        spans
    }

    fn assert_parity(s: &str) {
        assert_eq!(
            fast_spans(kimi_advance, s),
            regex_spans(KIMI_PATTERN, s),
            "kimi split mismatch on {s:?}"
        );
    }

    fn assert_o200k_parity(s: &str) {
        assert_eq!(
            fast_spans(o200k_advance, s),
            regex_spans(O200K_PATTERN, s),
            "o200k split mismatch on {s:?}"
        );
    }

    #[test]
    fn kimi_edge_cases() {
        let cases = [
            "",
            "hello world",
            "camelCase HTTPResponse AxxB",
            "don't can'ts it's we'Ve THEY'RE",
            "'sound 3'ts x''y",
            "123 4567 89",
            "2024-07-25 1.5e-7 0x7f",
            "  leading   spaces\ttab\r\n\r\n",
            "trailing spaces   ",
            "punct!!! ...\n// slash/tail",
            "café ñiño über MÜNCHEN",
            "日本語のテキスト",
            "中文abc123混合",
            "你好，世界！ Hello",
            "漢字とかな カタカナ",
            "한국어 텍스트",
            "русский ТЕКСТ",
            "emoji \u{1F600}\u{1F680}\u{1F680} test",
            "\u{00A0}nbsp\u{2028}line\u{3000}ideo",
            "combining e\u{301}\u{301} mark",
            "Han中numeral\u{3007}mix",
            "々repeat 〆ideo",
            "½ ⅷ roman ٣ arabic",
            "a\u{017F}b 'ſ longs",
            "MixedÑCase café'll",
            "tab\tHTTPServer\tcamelX",
            "  中文 mixed 日本",
            "\u{16FF0}reading mark",
            "ABC123中DEF",
            "'''",
            ".\n//comment",
        ];
        for case in cases {
            assert_parity(case);
        }
    }

    #[test]
    fn o200k_edge_cases() {
        // o200k = Kimi without Han runs and with a `[\r\n/]*` punct tail
        // (Han chars are ordinary letters/symbols here; `/` joins a tail).
        let cases = [
            "",
            "hello world",
            "camelCase HTTPResponse AxxB",
            "don't can'ts it's we'Ve THEY'RE",
            "'sound 3'ts x''y",
            "123 4567 89 2024-07-25 1.5e-7",
            "  leading   spaces\ttab\r\n\r\n",
            "punct!!! ...\n// slash/tail path/to/file",
            "café ñiño über MÜNCHEN",
            "日本語 中文 漢字", // Han is NOT special in o200k
            "中文abc123混合",
            "½ ⅷ ٣ roman",
            "a\u{017F}b 'ſ longs",
            "combining e\u{301}\u{301} mark",
            "emoji \u{1F600}\u{1F680} test",
            "path/./x .\n//c",
            "'''",
        ];
        for case in cases {
            assert_o200k_parity(case);
        }
    }

    #[test]
    fn family_fuzz_parity() {
        let pool: Vec<char> = "abcXYZ z09'\"-.,!?/ \t\r\n\
             \u{00A0}\u{2028}\u{3000}\u{017F}\
             éñÜАЖ日本中语字々〆〇\u{3007}\u{16FF0}\
             한¡½٣๒Ⅷ😀🚀\u{301}\u{5BF}\u{10FFFD}"
            .chars()
            .collect();
        let mut rng: u64 = 0xC0FFEE_1234_5678;
        for round in 0..6000 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let len = (rng >> 48) as usize % 48;
            let mut s = String::new();
            for _ in 0..len {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                s.push(pool[(rng >> 33) as usize % pool.len()]);
            }
            assert_eq!(
                fast_spans(kimi_advance, &s),
                regex_spans(KIMI_PATTERN, &s),
                "kimi fuzz mismatch round {round} on {s:?}"
            );
            assert_eq!(
                fast_spans(o200k_advance, &s),
                regex_spans(O200K_PATTERN, &s),
                "o200k fuzz mismatch round {round} on {s:?}"
            );
        }
    }

    #[test]
    fn kimi_matches_regex_on_corpus() {
        super::super::fast_split::assert_corpus_matches_regex("kimi", KIMI_PATTERN, kimi_advance);
    }

    #[test]
    fn o200k_matches_regex_on_corpus() {
        super::super::fast_split::assert_corpus_matches_regex(
            "o200k",
            O200K_PATTERN,
            o200k_advance,
        );
    }
}
