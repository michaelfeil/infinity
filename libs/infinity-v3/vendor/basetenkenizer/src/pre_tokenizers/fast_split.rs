//! Hand-specialized scanners for known pre-tokenization split regexes.
//!
//! The split regexes of common tokenizers are known ahead of time, so instead
//! of running a general regex engine (fancy-regex / PCRE2) we can recognize
//! the exact pattern string at [`Split`](super::Split) construction and
//! dispatch to a scanner hand-written for that pattern: a first-byte dispatch
//! plus SWAR run scans, roughly an order of magnitude faster than PCRE2 JIT
//! on the same pattern.
//!
//! Scalar scanning technique and the Qwen2 walker are adapted from
//! gigatoken (<https://github.com/marcelroed/gigatoken>),
//! MIT License, Copyright (c) 2026 Marcel Rød.
//!
//! Correctness contract: for a recognized pattern, iterating
//! [`FastSplitScheme::advance`] from 0 must produce exactly the spans of
//! `regex.find_iter(text)` for the original pattern (whose alternations
//! cover every char, so consecutive matches tile the text with no gaps).
//! The differential tests at the bottom of this file enforce that against
//! fancy-regex on hand-picked edge cases and a deterministic fuzz corpus;
//! any new scheme added here needs the same treatment.

use std::sync::LazyLock;

/// The Qwen2/Qwen3 split pattern, as it appears in `tokenizer.json`.
pub(crate) const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The Qwen2 scheme with number runs of up to three chars (`\p{N}{1,3}`
/// instead of `\p{N}`): GLM-4/GLM-5, and dolma2 / OLMo 2/3.
pub(crate) const QWEN2_N3_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// A split regex with a hand-specialized scanner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastSplitScheme {
    /// [`QWEN2_PATTERN`]: Qwen2/Qwen3 and derivatives.
    Qwen2,
    /// [`QWEN2_N3_PATTERN`]: Qwen2 with `\p{N}{1,3}` — GLM, dolma2/OLMo.
    Qwen2N3,
    /// The Kimi (moonshotai K2 family) o200k-family pattern.
    Kimi,
    /// The o200k_base pattern (GPT-4o, gpt-oss).
    O200k,
}

impl FastSplitScheme {
    /// Recognize a split regex source string. Only byte-for-byte known
    /// patterns qualify; anything else takes the generic engines.
    pub fn from_pattern(pattern: &str) -> Option<Self> {
        match pattern {
            QWEN2_PATTERN => Some(Self::Qwen2),
            QWEN2_N3_PATTERN => Some(Self::Qwen2N3),
            super::fast_split_o200k::KIMI_PATTERN => Some(Self::Kimi),
            super::fast_split_o200k::O200K_PATTERN => Some(Self::O200k),
            _ => None,
        }
    }

    /// Advance past one pretoken starting at `pos`. Requires
    /// `pos < bytes.len()`; always returns `end > pos` and `end <= len`,
    /// on a UTF-8 char boundary for valid UTF-8 input.
    #[inline(always)]
    pub fn advance(self, bytes: &[u8], pos: usize) -> usize {
        match self {
            Self::Qwen2 => qwen2_advance::<1>(bytes, pos),
            Self::Qwen2N3 => qwen2_advance::<3>(bytes, pos),
            Self::Kimi => super::fast_split_o200k::kimi_advance(bytes, pos),
            Self::O200k => super::fast_split_o200k::o200k_advance(bytes, pos),
        }
    }

    /// A chunk boundary near `target` that no pretoken can cross, for
    /// parallel scanning of large inputs: right after a `\n` whose next byte
    /// is non-whitespace ASCII. Safe because no alternative reaches backward
    /// across such a point — letter-run prefixes exclude `\r\n`, a whitespace
    /// run containing a newline always ends at its last newline, and a
    /// punctuation run's absorbed tail is only `[\r\n]` (all whitespace, so
    /// excluded by the non-whitespace check).
    ///
    /// The one exception is the **O200k** scheme, whose punct tail is
    /// `[\r\n/]*`: a punct token can absorb the `\n` *and* a following `/`
    /// (e.g. `.\n/` is one token), so a boundary placed between the `\n` and
    /// the `/` would be crossed — the preceding chunk would scan past it and
    /// the next chunk would re-scan the `/`, producing overlapping ranges.
    /// So `/` is not a safe boundary-follower for O200k. Returns `None` when
    /// no safe boundary exists in `[target, limit)`.
    pub fn find_safe_boundary(self, bytes: &[u8], target: usize, limit: usize) -> Option<usize> {
        // O200k's `[\r\n/]*` tail lets a punct token cross a newline into a `/`.
        let slash_unsafe = matches!(self, Self::O200k);
        let mut p = target.max(1);
        while p < limit {
            if bytes[p - 1] == b'\n' {
                let b = bytes[p];
                if b < 0x80 && !is_ascii_ws(b) && !(slash_unsafe && b == b'/') {
                    return Some(p);
                }
            }
            p += 1;
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Unicode classification: packed 2-bit class per codepoint
// ---------------------------------------------------------------------------

/// Character class as used by the split regexes: `\p{L}`, `\p{N}`, `\s`
/// (the Unicode White_Space property), and everything else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum CharClass {
    Letter = 0,
    Number = 1,
    Whitespace = 2,
    Other = 3,
}

/// 2 bits per codepoint, 4 codepoints per byte (~272 KiB). One L1/L2 load
/// replaces an ICU trie walk per non-ASCII char; only the cache lines for
/// scripts actually present in the input stay resident.
static CLASS_TABLE: LazyLock<Box<[u8]>> = LazyLock::new(build_class_table);

fn build_class_table() -> Box<[u8]> {
    use icu_properties::props::{GeneralCategory, GeneralCategoryGroup, WhiteSpace};
    use icu_properties::{CodePointMapData, CodePointSetData};

    const N: usize = 0x11_0000;
    let mut classes = vec![CharClass::Other as u8; N];
    let gc = CodePointMapData::<GeneralCategory>::new();
    for (group, class) in [
        (GeneralCategoryGroup::Letter, CharClass::Letter),
        (GeneralCategoryGroup::Number, CharClass::Number),
    ] {
        for range in gc.iter_ranges_for_group(group) {
            classes[*range.start() as usize..=*range.end() as usize].fill(class as u8);
        }
    }
    // White_Space is disjoint from GC Letter/Number, so fill order is moot.
    for range in CodePointSetData::new::<WhiteSpace>().iter_ranges() {
        classes[*range.start() as usize..=*range.end() as usize].fill(CharClass::Whitespace as u8);
    }

    let mut packed = vec![0u8; N / 4].into_boxed_slice();
    for (cp, class) in classes.iter().enumerate() {
        packed[cp >> 2] |= class << ((cp & 3) * 2);
    }
    packed
}

/// Class of a codepoint. `cp` must be `<= 0x10FFFF`, which [`decode_cp`]
/// guarantees via its clamp.
#[inline(always)]
fn class_of(cp: u32) -> CharClass {
    debug_assert!(cp <= 0x10_FFFF);
    let table = &**CLASS_TABLE;
    // SAFETY: cp <= 0x10FFFF, so cp >> 2 < table.len() == 0x110000 / 4.
    let byte = unsafe { *table.get_unchecked((cp >> 2) as usize) };
    match (byte >> ((cp & 3) * 2)) & 3 {
        0 => CharClass::Letter,
        1 => CharClass::Number,
        2 => CharClass::Whitespace,
        _ => CharClass::Other,
    }
}

// ---------------------------------------------------------------------------
// Branchless byte predicates
// ---------------------------------------------------------------------------

#[inline(always)]
pub(super) fn is_letter(b: u8) -> bool {
    (b | 0x20).wrapping_sub(b'a') < 26
}

#[inline(always)]
pub(super) fn is_digit(b: u8) -> bool {
    b.wrapping_sub(b'0') < 10
}

#[inline(always)]
pub(super) fn is_ascii_ws(b: u8) -> bool {
    b == b' ' || b.wrapping_sub(9) < 5
}

// ---------------------------------------------------------------------------
// UTF-8 decode tolerant of invalid input
// ---------------------------------------------------------------------------

/// The codepoint reported for byte sequences that cannot be decoded within
/// bounds (truncated tails) or that assemble past the Unicode range.
/// U+10FFFF is an unassigned noncharacter: class `Other`.
const CP_INVALID: u32 = 0x10_FFFF;

/// Decode one non-ASCII scalar. Requires only `pos < bytes.len()` and
/// `bytes[pos] >= 0x80`; invalid bytes decode deterministically, never read
/// past `bytes.len()`, and the returned length never overruns it.
#[inline(always)]
pub(super) unsafe fn decode_cp(bytes: &[u8], pos: usize) -> (u32, usize) {
    if pos + 4 > bytes.len() {
        return decode_cp_near_end(bytes, pos);
    }
    unsafe {
        let b0 = *bytes.get_unchecked(pos) as u32;
        let b1 = (*bytes.get_unchecked(pos + 1) & 0x3F) as u32;
        if b0 < 0xE0 {
            return (((b0 & 0x1F) << 6) | b1, 2);
        }
        let b2 = (*bytes.get_unchecked(pos + 2) & 0x3F) as u32;
        if b0 < 0xF0 {
            return (((b0 & 0x0F) << 12) | (b1 << 6) | b2, 3);
        }
        let b3 = (*bytes.get_unchecked(pos + 3) & 0x3F) as u32;
        (
            (((b0 & 0x07) << 18) | (b1 << 12) | (b2 << 6) | b3).min(CP_INVALID),
            4,
        )
    }
}

/// [`decode_cp`]'s slow path for `pos + 4 > bytes.len()`: identical results
/// for complete sequences; a sequence truncated by the buffer end consumes
/// exactly the remaining bytes and yields [`CP_INVALID`].
#[cold]
#[inline(never)]
fn decode_cp_near_end(bytes: &[u8], pos: usize) -> (u32, usize) {
    let len = bytes.len();
    let b0 = bytes[pos] as u32;
    let need = if b0 < 0xE0 {
        2
    } else if b0 < 0xF0 {
        3
    } else {
        4
    };
    if pos + need > len {
        return (CP_INVALID, len - pos);
    }
    let b1 = (bytes[pos + 1] & 0x3F) as u32;
    if need == 2 {
        return (((b0 & 0x1F) << 6) | b1, 2);
    }
    let b2 = (bytes[pos + 2] & 0x3F) as u32;
    if need == 3 {
        return (((b0 & 0x0F) << 12) | (b1 << 6) | b2, 3);
    }
    let b3 = (bytes[pos + 3] & 0x3F) as u32;
    (
        (((b0 & 0x07) << 18) | (b1 << 12) | (b2 << 6) | b3).min(CP_INVALID),
        4,
    )
}

// ---------------------------------------------------------------------------
// SWAR letter scan
// ---------------------------------------------------------------------------

const HI: u64 = 0x8080_8080_8080_8080;

/// High bit set in each lane that is NOT an ASCII letter.
#[inline(always)]
fn swar64_letter_nonmask(word: u64) -> u64 {
    let lowered = word | 0x2020_2020_2020_2020;
    let ge_a = (lowered | HI).wrapping_sub(0x6161_6161_6161_6161);
    let le_z = 0xFAFA_FAFA_FAFA_FAFA_u64.wrapping_sub(lowered);
    !(ge_a & le_z) & HI
}

/// Advance `pos` past ASCII letters, 8 bytes per iteration.
#[inline(always)]
fn swar_scan_letters(bytes: &[u8], mut pos: usize) -> usize {
    let len = bytes.len();
    while pos + 8 <= len {
        // SAFETY: pos + 8 <= len.
        let word = unsafe { (bytes.as_ptr().add(pos) as *const u64).read_unaligned() };
        if word & HI != 0 {
            break;
        }
        let nonletter = swar64_letter_nonmask(word);
        if nonletter != 0 {
            return pos + nonletter.to_le().trailing_zeros() as usize / 8;
        }
        pos += 8;
    }
    while pos < len {
        // SAFETY: pos < len.
        let b = unsafe { *bytes.get_unchecked(pos) };
        if is_letter(b) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

// ---------------------------------------------------------------------------
// Shared run scans
// ---------------------------------------------------------------------------

/// `\p{L}+` continuation: past ASCII letters via SWAR, then non-ASCII
/// letters via the class table.
#[inline(always)]
fn scan_letters_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        p = swar_scan_letters(bytes, p);
        if p < len && unsafe { *bytes.get_unchecked(p) } >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if class_of(cp) == CharClass::Letter {
                p += l;
                continue;
            }
        }
        return p;
    }
}

/// `[^\s\p{L}\p{N}]+` continuation.
#[inline(always)]
fn scan_other_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        while p < len {
            // SAFETY: p < len.
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
            if class_of(cp) == CharClass::Other {
                p += l;
                continue;
            }
        }
        return p;
    }
}

/// `[\r\n]*`: past a run of CR/LF bytes.
#[inline(always)]
fn scan_newlines(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        // SAFETY: pos < len.
        let b = unsafe { *bytes.get_unchecked(pos) };
        if b == b'\r' || b == b'\n' {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

/// `\p{N}{1,MAX}` continuation: extend a number run that already matched
/// `consumed` chars to at most `MAX` chars total. ASCII digits and Unicode
/// `\p{N}` count alike.
#[inline(always)]
pub(super) fn scan_numbers_max<const MAX: u32>(
    bytes: &[u8],
    mut pos: usize,
    mut consumed: u32,
) -> usize {
    let len = bytes.len();
    while consumed < MAX && pos < len {
        // SAFETY: pos < len.
        let b = unsafe { *bytes.get_unchecked(pos) };
        if is_digit(b) {
            pos += 1;
            consumed += 1;
            continue;
        }
        if b >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, pos) };
            if class_of(cp) == CharClass::Number {
                pos += l;
                consumed += 1;
                continue;
            }
        }
        break;
    }
    pos
}

/// If the char at `pos` is `\p{L}`, return the offset just past it.
#[inline(always)]
fn letter_end_at(bytes: &[u8], pos: usize) -> Option<usize> {
    let &b = bytes.get(pos)?;
    if is_letter(b) {
        return Some(pos + 1);
    }
    if b >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        if class_of(cp) == CharClass::Letter {
            return Some(pos + l);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Qwen2 scheme
// ---------------------------------------------------------------------------

/// Whitespace-led token starting at `start`: the alternatives
/// `\s*[\r\n]+` | `\s+(?!\S)` | `\s+`, in that priority.
/// Precondition: the letter-prefix (`[^\r\n\p{L}\p{N}]?\p{L}+`) and
/// space+punct (` ?[^\s\p{L}\p{N}]+...`) alternatives were ruled out.
#[inline(always)]
fn ws_token_end(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let mut p = start;
    let mut last_nl_end = 0usize; // 0 = run contains no \r\n
    let mut last_char_start = start;
    while p < len {
        // SAFETY: p < len.
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
            if class_of(cp) == CharClass::Whitespace {
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
        return last_nl_end; // `\s*[\r\n]+`: through the last newline, even at EOS
    }
    if p >= len {
        return p; // `\s+(?!\S)`: lookahead succeeds at EOS
    }
    if last_char_start > start {
        return last_char_start; // `\s+(?!\S)`: all but the last ws char
    }
    p // `\s+`: single whitespace char before content
}

/// Advance past one Qwen2-scheme token starting at `pos`.
/// `pos` must be < `bytes.len()`.
///
/// `MAX_DIGITS` is the number-run cap: 1 for the Qwen2 pattern's `\p{N}`,
/// 3 for the GLM/dolma2 variant's `\p{N}{1,3}` (the only difference
/// between the two patterns).
#[inline(always)]
fn qwen2_advance<const MAX_DIGITS: u32>(bytes: &[u8], pos: usize) -> usize {
    // SAFETY: pos < len per contract.
    let b0 = unsafe { *bytes.get_unchecked(pos) };

    // Hot path 1: ASCII letter — `\p{L}+` with empty prefix
    if is_letter(b0) {
        return scan_letters_from(bytes, pos + 1);
    }

    // Hot path 2: space prefix
    if b0 == b' ' {
        let Some(&b1) = bytes.get(pos + 1) else {
            return pos + 1; // trailing lone space (`\s+(?!\S)` at EOS)
        };
        if is_letter(b1) {
            return scan_letters_from(bytes, pos + 2); // " word"
        }
        if b1 < 0x80 {
            if is_digit(b1) {
                return pos + 1; // numbers never absorb the space
            }
            if is_ascii_ws(b1) {
                return ws_token_end(bytes, pos);
            }
            // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
            let p = scan_other_from(bytes, pos + 2);
            return scan_newlines(bytes, p);
        }
        let (cp, l) = unsafe { decode_cp(bytes, pos + 1) };
        let p1 = pos + 1 + l;
        match class_of(cp) {
            CharClass::Letter => return scan_letters_from(bytes, p1),
            CharClass::Whitespace => return ws_token_end(bytes, pos),
            CharClass::Number => return pos + 1,
            CharClass::Other => {
                let p = scan_other_from(bytes, p1);
                return scan_newlines(bytes, p);
            }
        }
    }

    // Non-ASCII
    if b0 >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        let p0 = pos + l;
        let class = class_of(cp);
        if class == CharClass::Letter {
            return scan_letters_from(bytes, p0);
        }
        if class == CharClass::Number {
            if MAX_DIGITS == 1 {
                return p0; // `\p{N}`: exactly one char
            }
            return scan_numbers_max::<MAX_DIGITS>(bytes, p0, 1);
        }
        // Any non-letter/number char except \r\n may prefix a letter run
        if let Some(p) = letter_end_at(bytes, p0) {
            return scan_letters_from(bytes, p);
        }
        if class == CharClass::Whitespace {
            return ws_token_end(bytes, pos);
        }
        let p = scan_other_from(bytes, p0);
        return scan_newlines(bytes, p);
    }

    // ASCII digit: `\p{N}` (one char) or `\p{N}{1,3}` (up to MAX_DIGITS)
    if is_digit(b0) {
        if MAX_DIGITS == 1 {
            return pos + 1;
        }
        return scan_numbers_max::<MAX_DIGITS>(bytes, pos + 1, 1);
    }

    // Apostrophe: case-insensitive contractions
    if b0 == b'\'' {
        match bytes.get(pos + 1).map(u8::to_ascii_lowercase) {
            Some(b's' | b'd' | b'm' | b't') => return pos + 2,
            Some(b'l') if bytes.get(pos + 2).map(u8::to_ascii_lowercase) == Some(b'l') => {
                return pos + 3;
            }
            Some(b'v') if bytes.get(pos + 2).map(u8::to_ascii_lowercase) == Some(b'e') => {
                return pos + 3;
            }
            Some(b'r') if bytes.get(pos + 2).map(u8::to_ascii_lowercase) == Some(b'e') => {
                return pos + 3;
            }
            _ => {}
        }
        // U+017F LATIN SMALL LETTER LONG S case-folds to 's' under `(?i)`
        if bytes.get(pos + 1) == Some(&0xC5) && bytes.get(pos + 2) == Some(&0xBF) {
            return pos + 3;
        }
        // Not a contraction: `'` can still prefix a letter run
        if let Some(p) = letter_end_at(bytes, pos + 1) {
            return scan_letters_from(bytes, p);
        }
        let p = scan_other_from(bytes, pos + 1);
        return scan_newlines(bytes, p);
    }

    // \r and \n are excluded from the letter-run prefix
    if b0 == b'\r' || b0 == b'\n' {
        return ws_token_end(bytes, pos);
    }

    // Other ASCII whitespace (\t, \x0b, \x0c) may prefix a letter run
    if is_ascii_ws(b0) {
        if let Some(p) = letter_end_at(bytes, pos + 1) {
            return scan_letters_from(bytes, p);
        }
        return ws_token_end(bytes, pos);
    }

    // ASCII punctuation/symbol
    if let Some(p) = letter_end_at(bytes, pos + 1) {
        return scan_letters_from(bytes, p);
    }
    let p = scan_other_from(bytes, pos + 1);
    scan_newlines(bytes, p)
}

// ---------------------------------------------------------------------------
// Tests: differential against fancy-regex on the original pattern
// ---------------------------------------------------------------------------

/// Assert a scheme's scanner produces byte-identical spans to `fancy-regex`
/// on the exact `pattern`, over the whole vendored multilingual corpus
/// (~2 MB real prose: English + 8 non-Latin scripts; see
/// `tests/fixtures/PROVENANCE.md`). Compares token-by-token in lockstep so a
/// divergence reports the exact byte offset and surrounding tokens rather
/// than a giant Vec diff. Shared by the o200k-family schemes' tests.
///
/// Since the scanner only replaces the split stage — byte-level and BPE
/// downstream are untouched — span-identity over a large real corpus is
/// end-to-end token-id equivalence for the change.
#[cfg(test)]
pub(crate) fn assert_corpus_matches_regex(
    label: &str,
    pattern: &str,
    advance: impl Fn(&[u8], usize) -> usize,
) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/corpus_multilingual.txt"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read corpus: {e}"));
    let bytes = text.as_bytes();
    let re = fancy_regex::Regex::new(pattern).unwrap();
    let mut re_iter = re.find_iter(&text);
    let mut pos = 0usize;
    let mut idx = 0usize;
    let mut recent: Vec<(&str, &str)> = Vec::new();
    while pos < bytes.len() {
        let end = advance(bytes, pos);
        assert!(
            end > pos && end <= bytes.len(),
            "{label}: no progress at byte {pos}"
        );
        let fast = &text[pos..end];
        match re_iter.next() {
            Some(m) => {
                let m = m.expect("regex match error");
                let re_str = &text[m.start()..m.end()];
                if recent.len() == 8 {
                    recent.remove(0);
                }
                recent.push((fast, re_str));
                assert_eq!(
                    fast, re_str,
                    "{label}: mismatch at token {idx} (byte {pos}): fast={fast:?} regex={re_str:?}\n  recent (fast,regex): {recent:?}"
                );
            }
            None => panic!("{label}: scanner produced extra token at {idx}: {fast:?}"),
        }
        pos = end;
        idx += 1;
    }
    assert!(
        re_iter.next().is_none(),
        "{label}: regex produced tokens past where the scanner stopped"
    );
    eprintln!(
        "{label}: {idx} tokens match over {} bytes of corpus",
        bytes.len()
    );
}

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

    fn fast_spans(scheme: FastSplitScheme, s: &str) -> Vec<(usize, usize)> {
        let bytes = s.as_bytes();
        let mut spans = Vec::new();
        let mut pos = 0;
        while pos < bytes.len() {
            let end = scheme.advance(bytes, pos);
            assert!(
                end > pos && end <= bytes.len(),
                "no progress at {pos} in {s:?}"
            );
            spans.push((pos, end));
            pos = end;
        }
        spans
    }

    fn assert_parity(scheme: FastSplitScheme, pattern: &str, s: &str) {
        assert_eq!(
            fast_spans(scheme, s),
            regex_spans(pattern, s),
            "split mismatch on {s:?}"
        );
    }

    #[test]
    fn qwen2_edge_cases() {
        let cases = [
            "",
            "hello",
            " hello",
            "hello world",
            "  hello",
            "   hello",
            "\thello",
            "\t hello",
            " \thello",
            "hello\n",
            "hello \n",
            "hello\r\n\r\n",
            "hello  \r\n  world",
            "don't",
            "DON'T",
            "it's it'S IT'Ll we'VE they're I'd I'm",
            "c'mon 'quoted' rock'n'roll",
            "'s't'll've're'd'm",
            "a1b2c3",
            "123",
            "1 2 34",
            " 5",
            "3.14159",
            "hello, world! (parens) [brackets]",
            "  ,,,  ",
            "!!!\n\n",
            "?!\r",
            "trailing spaces   ",
            "trailing tab\t",
            "\n\nleading",
            "\r\n",
            " ",
            "\t",
            "múltiple ñiño café über",
            "日本語のテキスト",
            "русский текст",
            "한국어 텍스트",
            "e\u{301}xpose\u{301}", // combining acute: Mark is not \p{L}
            "\u{00A0}nbsp\u{00A0}\u{00A0}", // NBSP is White_Space
            "\u{2028}line\u{2029}sep",
            "\u{1F600} emoji \u{1F680}\u{1F680}",
            "١٢٣ arabic digits", // \p{N} single-char matches
            "๑๒๓ thai digits",
            "Ⅷ roman numeral", // Nl is \p{N}
            "'\u{017F} long s contraction",
            "mixed 'ſ fold",
            "tab\tbetween\twords",
            "\x0b\x0c vertical form",
            "space before ünïcode",
            "@user #tag $5 100%",
            "a'll b'LL c'Ll",
            "'''",
            "''s",
            "x''y",
            "-prefix word",
            ".hidden",
            "\u{3000}ideographic space",
            "za\u{017F}ka", // long s inside a letter run
        ];
        let extra_digit_cases = [
            "123456",
            "12 345 6789",
            " 12",
            "a123b",
            "١٢٣٤ arabic digits",
            "๑๒๓๔๕ thai digits",
            "1٢3٤5",
            "x99999y",
            "2024-07-25 12:34:56",
        ];
        for scheme in [
            (FastSplitScheme::Qwen2, QWEN2_PATTERN),
            (FastSplitScheme::Qwen2N3, QWEN2_N3_PATTERN),
        ] {
            for case in cases.iter().chain(extra_digit_cases.iter()) {
                assert_parity(scheme.0, scheme.1, case);
            }
        }
    }

    #[test]
    fn qwen2_fuzz_parity() {
        // Deterministic LCG fuzz over a pool that exercises every branch:
        // ASCII letters/digits/punct, all ASCII whitespace, CRLF runs,
        // contraction heads, non-ASCII letters/digits/whitespace/symbols,
        // combining marks, and multi-byte boundary cases.
        let pool: Vec<char> = "abcXYZ z09'\"-.,!? \t\r\n\
             \u{00A0}\u{2028}\u{3000}\u{017F}\
             éñÜ日本語һЖ¡½٣๒Ⅷ😀🚀\u{301}\u{10FFFD}"
            .chars()
            .collect();
        let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
        for round in 0..4000 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let len = (rng >> 48) as usize % 64;
            let mut s = String::new();
            for _ in 0..len {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                s.push(pool[(rng >> 33) as usize % pool.len()]);
            }
            for (scheme, pattern) in [
                (FastSplitScheme::Qwen2, QWEN2_PATTERN),
                (FastSplitScheme::Qwen2N3, QWEN2_N3_PATTERN),
            ] {
                let fast = fast_spans(scheme, &s);
                let reference = regex_spans(pattern, &s);
                assert_eq!(fast, reference, "fuzz mismatch round {round} on {s:?}");
            }
        }
    }

    #[test]
    fn qwen2_matches_regex_on_corpus() {
        super::assert_corpus_matches_regex("qwen2", QWEN2_PATTERN, |b, p| qwen2_advance::<1>(b, p));
    }

    #[test]
    fn qwen2n3_matches_regex_on_corpus() {
        super::assert_corpus_matches_regex("qwen2n3", QWEN2_N3_PATTERN, |b, p| {
            qwen2_advance::<3>(b, p)
        });
    }

    #[test]
    fn o200k_safe_boundary_excludes_slash_after_newline() {
        // `.\n/foo\nbar`: byte 2 is right after `\n` and is `/`. O200k's
        // punct tail `[\r\n/]*` lets `.` absorb `\n/`, so byte 2 is an unsafe
        // boundary and O200k must skip to the next `\n`-led non-slash point
        // (byte 7, after the second `\n`, before `b`). Schemes with a `[\r\n]`
        // tail may use the slash boundary.
        let bytes = b".\n/foo\nbar";
        assert_eq!(
            FastSplitScheme::O200k.find_safe_boundary(bytes, 0, bytes.len()),
            Some(7)
        );
        for scheme in [
            FastSplitScheme::Qwen2,
            FastSplitScheme::Qwen2N3,
            FastSplitScheme::Kimi,
        ] {
            assert_eq!(
                scheme.find_safe_boundary(bytes, 0, bytes.len()),
                Some(2),
                "{scheme:?} has a [\\r\\n] tail; slash boundary is safe"
            );
        }
    }

    #[test]
    fn recognizes_only_known_patterns() {
        use super::super::fast_split_o200k::{KIMI_PATTERN, O200K_PATTERN};

        assert_eq!(
            FastSplitScheme::from_pattern(QWEN2_PATTERN),
            Some(FastSplitScheme::Qwen2)
        );
        assert_eq!(
            FastSplitScheme::from_pattern(QWEN2_N3_PATTERN),
            Some(FastSplitScheme::Qwen2N3)
        );
        assert_eq!(
            FastSplitScheme::from_pattern(KIMI_PATTERN),
            Some(FastSplitScheme::Kimi)
        );
        assert_eq!(
            FastSplitScheme::from_pattern(O200K_PATTERN),
            Some(FastSplitScheme::O200k)
        );

        // Guard: the Kimi K2/K3 (knext) tiktoken `pat_str`, assembled here
        // exactly as `tokenization_kimi.py` builds it, must equal
        // `KIMI_PATTERN` and route to the Kimi scanner. knext ships as a
        // tiktoken model (no tokenizer.json); its `pat_str` reaches
        // `from_pattern` verbatim through the tiktoken -> tokenizer.json
        // conversion, so an upstream regex change would otherwise silently
        // drop it to the slow PCRE2 path. This literal is the failing
        // tripwire for that.
        let knext_pat_str = [
            r"[\p{Han}]+",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"\p{N}{1,3}",
            r" ?[^\s\p{L}\p{N}]+[\r\n]*",
            r"\s*[\r\n]+",
            r"\s+(?!\S)",
            r"\s+",
        ]
        .join("|");
        assert_eq!(knext_pat_str, KIMI_PATTERN, "Kimi K2/K3 pat_str drifted");
        assert_eq!(
            FastSplitScheme::from_pattern(&knext_pat_str),
            Some(FastSplitScheme::Kimi)
        );

        // One char off -> no specialization.
        assert_eq!(
            FastSplitScheme::from_pattern(&QWEN2_PATTERN.replace("'t", "'x")),
            None
        );
        assert_eq!(FastSplitScheme::from_pattern(""), None);
    }
}
