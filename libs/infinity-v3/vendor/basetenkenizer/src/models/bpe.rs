// Copyright 2026 fastokens contributors
// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0
//
// Derived from Crusoe fastokens and modified by Baseten.

use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use serde::Deserialize;
use serde_json::Value;

use super::Result;
use crate::pre_tokenizers::BYTE_TO_CHAR;

type TokenId = u32;
type ParsedMergeMap = HashMap<(u32, u32), (u32, u32)>;
type DecompositionMergeMap = HashMap<(u32, u32), (u32, u32), FxBuildHasher>;
type Vocab = HashMap<String, u32>;

#[derive(Clone, Copy, Eq, PartialEq)]
struct TokenEntry {
    id: TokenId,
    whole_token: bool,
}

const INVALID_TOKEN: u32 = u32::MAX;

#[inline(always)]
pub(crate) fn pretoken_hash(bytes: &[u8]) -> u64 {
    let mut hash = bytes.len() as u64;
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        hash = hash
            .wrapping_add(u64::from_ne_bytes(chunk.try_into().unwrap()))
            .wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    for &byte in chunks.remainder() {
        hash = hash
            .wrapping_add(byte as u64)
            .wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    if hash == EMPTY_SLOT { 1 } else { hash }
}

/// Pretokens with at most this many initial (per-byte) symbols use the
/// stack-resident linear-scan merge instead of the heap. Sized so the stack
/// arrays fit in registers/L1 and `u8` linked-list indices stay in range.
const SMALL_MERGE_MAX: usize = 32;

/// Open-addressing hash table for merge lookups.
#[derive(Clone, PartialEq)]
struct MergeMap {
    mask: usize,
    keys: Vec<u64>,
    vals: Vec<u32>,
}

const EMPTY_KEY: u64 = u64::MAX;

impl MergeMap {
    fn new() -> Self {
        Self {
            mask: 0,
            keys: Vec::new(),
            vals: Vec::new(),
        }
    }

    fn from_parsed(parsed: &ParsedMergeMap) -> Self {
        if parsed.is_empty() {
            return Self::new();
        }
        // ~50% load factor.
        let capacity = (parsed.len() * 2).next_power_of_two();
        let mask = capacity - 1;
        let mut keys = vec![EMPTY_KEY; capacity];
        let mut vals = vec![0u32; capacity];

        for (&(t1, t2), &(_rank, merged_id)) in parsed {
            let key = pack_pair(t1, t2);
            let mut idx = fx_hash(key) as usize & mask;
            loop {
                if keys[idx] == EMPTY_KEY {
                    keys[idx] = key;
                    vals[idx] = merged_id;
                    break;
                }
                idx = (idx + 1) & mask;
            }
        }

        Self { mask, keys, vals }
    }

    /// Look up the merged token ID for a pair.
    #[inline(always)]
    fn get(&self, t1: u32, t2: u32) -> Option<u32> {
        if self.keys.is_empty() {
            return None;
        }
        let key = pack_pair(t1, t2);
        let mut idx = fx_hash(key) as usize & self.mask;
        loop {
            let k = unsafe { *self.keys.get_unchecked(idx) };
            if k == key {
                return Some(unsafe { *self.vals.get_unchecked(idx) });
            }
            if k == EMPTY_KEY {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    fn len(&self) -> usize {
        self.keys.iter().filter(|&&k| k != EMPTY_KEY).count()
    }
}

#[inline(always)]
fn pack_pair(t1: u32, t2: u32) -> u64 {
    (t1 as u64) << 32 | t2 as u64
}

#[inline(always)]
fn fx_hash(key: u64) -> u64 {
    key.wrapping_mul(0x517cc1b727220a95)
}

/// FxHash-based [`BuildHasher`] for the token cache.
#[derive(Clone, Default)]
struct FxBuildHasher;

impl std::hash::BuildHasher for FxBuildHasher {
    type Hasher = FxStrHasher;
    fn build_hasher(&self) -> FxStrHasher {
        FxStrHasher(0)
    }
}

struct FxStrHasher(u64);

impl std::hash::Hasher for FxStrHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut state = self.0;
        let mut i = 0;
        while i + 8 <= bytes.len() {
            let word = u64::from_ne_bytes(bytes[i..i + 8].try_into().unwrap());
            state = state.wrapping_add(word).wrapping_mul(0x517cc1b727220a95);
            i += 8;
        }
        while i < bytes.len() {
            state = state
                .wrapping_add(bytes[i] as u64)
                .wrapping_mul(0x517cc1b727220a95);
            i += 1;
        }
        self.0 = state;
    }
}

type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

const FLAT_CACHE_BITS: usize = 16;
const EMPTY_SLOT: u64 = 0;

#[derive(Clone, Copy)]
#[repr(C)]
struct CacheSlot {
    hash: u64,
    offset: u32,
    len: u16,
    key_len: u16,
    key_offset: u32,
}

struct FlatCache {
    bpe_id: usize,
    mask: usize,
    max_load: usize,
    slots: Vec<CacheSlot>,
    pool: Vec<u32>,
    key_pool: Vec<u8>,
    count: usize,
}

impl FlatCache {
    fn new() -> Self {
        // Keep only the slot table eager. Reserving the old 256/512 KiB pools
        // here multiplied that memory by every worker even before a cache miss.
        Self::with_bits(FLAT_CACHE_BITS)
    }

    /// A flat cache with `1 << bits` slots. The thread-local L1 uses
    /// [`FLAT_CACHE_BITS`]; the shared-cache shards use fewer bits each.
    fn with_bits(bits: usize) -> Self {
        let size = 1usize << bits;
        Self {
            bpe_id: 0,
            mask: size - 1,
            max_load: size * 3 / 4,
            slots: vec![
                CacheSlot {
                    hash: EMPTY_SLOT,
                    offset: 0,
                    len: 0,
                    key_len: 0,
                    key_offset: 0,
                };
                size
            ],
            pool: Vec::new(),
            key_pool: Vec::new(),
            count: 0,
        }
    }

    fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.hash = EMPTY_SLOT;
        }
        self.pool.clear();
        self.key_pool.clear();
        self.count = 0;
    }

    #[inline(always)]
    fn hash_str(s: &str) -> u64 {
        pretoken_hash(s.as_bytes())
    }

    #[inline(always)]
    fn get_hashed(&self, key: &str, hash: u64, out: &mut Vec<u32>) -> bool {
        let key_bytes = key.as_bytes();
        let mut idx = hash as usize & self.mask;
        loop {
            let slot = unsafe { self.slots.get_unchecked(idx) };
            if slot.hash == hash {
                let ks = slot.key_offset as usize;
                let ke = ks + slot.key_len as usize;
                if unsafe { self.key_pool.get_unchecked(ks..ke) } == key_bytes {
                    let start = slot.offset as usize;
                    let end = start + slot.len as usize;
                    out.extend_from_slice(unsafe { self.pool.get_unchecked(start..end) });
                    return true;
                }
            }
            if slot.hash == EMPTY_SLOT {
                return false;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn insert_hashed(&mut self, key: &str, hash: u64, ids: &[u32]) {
        debug_assert_ne!(hash, EMPTY_SLOT);
        if self.count >= self.max_load {
            self.clear();
        }
        let key_bytes = key.as_bytes();
        let mut idx = hash as usize & self.mask;
        loop {
            let slot = unsafe { self.slots.get_unchecked(idx) };
            let h = slot.hash;
            if h == EMPTY_SLOT {
                let Ok(len) = u16::try_from(ids.len()) else {
                    return;
                };
                let Ok(key_len) = u16::try_from(key_bytes.len()) else {
                    return;
                };
                self.count += 1;
                let offset = self.pool.len() as u32;
                self.pool.extend_from_slice(ids);
                let key_offset = self.key_pool.len() as u32;
                self.key_pool.extend_from_slice(key_bytes);
                let slot = unsafe { self.slots.get_unchecked_mut(idx) };
                slot.hash = hash;
                slot.offset = offset;
                slot.len = len;
                slot.key_offset = key_offset;
                slot.key_len = key_len;
                return;
            }
            if h == hash {
                let ks = slot.key_offset as usize;
                let ke = ks + slot.key_len as usize;
                if unsafe { self.key_pool.get_unchecked(ks..ke) } == key_bytes {
                    // Updating in place appends the new IDs and leaves the old
                    // range unreachable. The load-bound clear cycle reclaims
                    // the pool without moving offsets for other live slots.
                    let Ok(len) = u16::try_from(ids.len()) else {
                        return;
                    };
                    let offset = self.pool.len() as u32;
                    self.pool.extend_from_slice(ids);
                    let slot = unsafe { self.slots.get_unchecked_mut(idx) };
                    slot.offset = offset;
                    slot.len = len;
                    return;
                }
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

thread_local! {
    static TL_BPE_CACHE: RefCell<FlatCache> = RefCell::new(FlatCache::new());
    static TL_FUSED_CACHE: RefCell<FlatCache> = RefCell::new(FlatCache::new());
}

const CACHE_SHARDS: usize = 64;

/// Slot bits per shared-cache shard: 64 shards x 1024 slots = 64k entries.
const SHARED_SHARD_BITS: usize = 10;

/// Cross-thread token cache: [`CACHE_SHARDS`] mutex-guarded [`FlatCache`]
/// shards. Compared to the previous `HashMap<String, Vec<u32>>` shards this
/// is allocation-free per insert (keys and ids are copied into per-shard
/// pools that retain capacity across clears) and bounded (a shard clears at
/// 3/4 load instead of growing forever), which both removes the two heap
/// allocations per cold pretoken and fixes unbounded memory growth on
/// diverse long-running traffic.
struct SharedCache {
    shards: Vec<Mutex<Option<FlatCache>>>,
}

impl SharedCache {
    fn new() -> Self {
        Self {
            shards: (0..CACHE_SHARDS).map(|_| Mutex::new(None)).collect(),
        }
    }

    /// Shard selector. Uses the TOP bits of the same hash `FlatCache`
    /// indexes slots with (low bits), so the two are independent.
    #[inline]
    fn shard_index(hash: u64) -> usize {
        (hash >> (64 - 6)) as usize & (CACHE_SHARDS - 1)
    }

    #[inline]
    fn get_hashed(&self, key: &str, hash: u64, out: &mut Vec<u32>) -> bool {
        self.shards[Self::shard_index(hash)]
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|cache| cache.get_hashed(key, hash, out))
    }

    #[inline]
    fn insert_hashed(&self, key: &str, hash: u64, ids: &[u32]) {
        self.shards[Self::shard_index(hash)]
            .lock()
            .unwrap()
            .get_or_insert_with(|| FlatCache::with_bits(SHARED_SHARD_BITS))
            .insert_hashed(key, hash, ids);
    }
}

/// Raw deserialization helper.
#[derive(Deserialize)]
struct RawBpe {
    #[serde(default)]
    vocab: Vocab,
    #[serde(default)]
    merges: Vec<Value>,
    #[allow(dead_code)]
    dropout: Option<f64>,
    #[allow(dead_code)]
    unk_token: Option<String>,
    #[allow(dead_code)]
    continuing_subword_prefix: Option<String>,
    #[allow(dead_code)]
    end_of_word_suffix: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    fuse_unk: bool,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    ignore_merges: bool,
}

/// Monotonic counter for unique Bpe instance IDs.
static BPE_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// Entry in the BPE merge priority queue.
/// `key = (rank << 32) | pos`, `val = (left_c << 32) | right_c`.
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(C)]
struct MergeEntry {
    key: u64,
    val: u64,
}

impl MergeEntry {
    #[inline(always)]
    fn new(rank: u32, pos: u32, left_c: u32, right_c: u32) -> Self {
        Self {
            key: (rank as u64) << 32 | pos as u64,
            val: (left_c as u64) << 32 | right_c as u64,
        }
    }

    #[inline(always)]
    fn pos(&self) -> u32 {
        self.key as u32
    }

    #[inline(always)]
    fn left_c(&self) -> u32 {
        (self.val >> 32) as u32
    }

    #[inline(always)]
    fn right_c(&self) -> u32 {
        self.val as u32
    }
}

impl Ord for MergeEntry {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

impl PartialOrd for MergeEntry {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Symbol in the merge linked list.
#[derive(Clone, Copy)]
struct MergeSymbol {
    c: u32,
    prev: i32,
    next: i32,
}

struct MergeScratch {
    symbols: Vec<MergeSymbol>,
    heap: BinaryHeap<Reverse<MergeEntry>>,
    heap_buf: Vec<Reverse<MergeEntry>>,
}

impl MergeScratch {
    fn new() -> Self {
        Self {
            symbols: Vec::new(),
            heap: BinaryHeap::new(),
            heap_buf: Vec::new(),
        }
    }
}

thread_local! {
    static TL_MERGE_SCRATCH: RefCell<MergeScratch> = RefCell::new(MergeScratch::new());
}

/// Interleaved slot for the ranked merge map (16 bytes).
#[derive(Clone, Copy)]
#[repr(C)]
struct RankedMergeSlot {
    key: u64,
    rank: u32,
    id: u32,
}

/// Open-addressing hash table storing `(left_id, right_id) → (rank, merged_id)`.
#[derive(Clone)]
struct RankedMergeMap {
    mask: usize,
    slots: Vec<RankedMergeSlot>,
}

impl RankedMergeMap {
    fn from_parsed(parsed: &ParsedMergeMap) -> Self {
        if parsed.is_empty() {
            return Self {
                mask: 0,
                slots: Vec::new(),
            };
        }
        let capacity = (parsed.len() * 2).next_power_of_two();
        let mask = capacity - 1;
        let mut slots = vec![
            RankedMergeSlot {
                key: EMPTY_KEY,
                rank: 0,
                id: 0
            };
            capacity
        ];

        for (&(t1, t2), &(rank, merged_id)) in parsed {
            let key = pack_pair(t1, t2);
            let mut idx = fx_hash(key) as usize & mask;
            loop {
                if slots[idx].key == EMPTY_KEY {
                    slots[idx] = RankedMergeSlot {
                        key,
                        rank,
                        id: merged_id,
                    };
                    break;
                }
                idx = (idx + 1) & mask;
            }
        }

        Self { mask, slots }
    }

    /// Look up the rank and merged token ID for a pair.
    #[inline(always)]
    fn get(&self, t1: u32, t2: u32) -> Option<(u32, u32)> {
        if self.slots.is_empty() {
            return None;
        }
        let key = pack_pair(t1, t2);
        let mut idx = fx_hash(key) as usize & self.mask;
        loop {
            let slot = unsafe { self.slots.get_unchecked(idx) };
            if slot.key == key {
                return Some((slot.rank, slot.id));
            }
            if slot.key == EMPTY_KEY {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

/// CSR adjacency structure for merge pair discovery.
#[derive(Clone)]
struct MergeAdjacency {
    offsets: Vec<u32>,
    data: Vec<(u32, u32, u32)>, // (neighbor, rank, new_id)
}

impl MergeAdjacency {
    fn from_parsed(parsed: &ParsedMergeMap, vocab_size: usize) -> Self {
        let mut counts = vec![0u32; vocab_size];
        for &(left, _right) in parsed.keys() {
            counts[left as usize] += 1;
        }

        let mut offsets = Vec::with_capacity(vocab_size + 1);
        offsets.push(0u32);
        let mut running = 0u32;
        for &c in &counts {
            running += c;
            offsets.push(running);
        }

        let mut data = vec![(0u32, 0u32, 0u32); running as usize];
        let mut write_pos = offsets[..vocab_size].to_vec();
        for (&(left, right), &(rank, merged_id)) in parsed {
            let idx = write_pos[left as usize] as usize;
            data[idx] = (right, rank, merged_id);
            write_pos[left as usize] += 1;
        }

        for i in 0..vocab_size {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            data[start..end].sort_unstable_by_key(|&(neighbor, _, _)| neighbor);
        }

        Self { offsets, data }
    }

    #[inline(always)]
    fn get(&self, left: u32, right: u32) -> Option<(u32, u32)> {
        let start = unsafe { *self.offsets.get_unchecked(left as usize) } as usize;
        let end = unsafe { *self.offsets.get_unchecked(left as usize + 1) } as usize;
        let slice = unsafe { self.data.get_unchecked(start..end) };
        match slice.binary_search_by_key(&right, |&(n, _, _)| n) {
            Ok(idx) => {
                let entry = unsafe { slice.get_unchecked(idx) };
                Some((entry.1, entry.2))
            }
            Err(_) => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(try_from = "RawBpe")]
pub struct Bpe {
    #[serde(skip)]
    id: usize,
    // One eligible fused encode unit may claim affinity for this BPE instance.
    // Segmented/chunked APIs race eligible units against this gate, so one
    // chunk wins and its siblings use contiguous scheduling. Cache eviction
    // does not re-arm the gate.
    #[serde(skip)]
    content_affinity_available: AtomicBool,
    merge_map: MergeMap,
    unmerge_map: Vec<(TokenId, TokenId)>,
    token_lens: Vec<u16>,
    shared_cache: SharedCache,
    fused_shared_cache: SharedCache,
    id_to_token: Vec<Arc<str>>,
    token_to_id: FxHashMap<Arc<str>, TokenEntry>,
    raw_token_to_id: Option<Arc<FxHashMap<Vec<u8>, u32>>>,
    byte_to_initial_token: [u32; 256],
    ranked_merge_map: RankedMergeMap,
    byte_pair_initial: Vec<(u32, u32)>,
    merge_adj: MergeAdjacency,
    orphan: Vec<bool>,
    ignore_merges: bool,
    byte_fallback_ids: Option<Arc<[Option<u32>; 256]>>,
}

impl TryFrom<RawBpe> for Bpe {
    type Error = String;

    fn try_from(raw: RawBpe) -> Result<Self> {
        let merge_map = parse_merges(&raw.vocab, &raw.merges)?;
        let mut bpe = Self::new(&raw.vocab, merge_map)?;
        bpe.ignore_merges = raw.ignore_merges;
        if raw.byte_fallback {
            bpe.byte_fallback_ids = Some(Arc::new(std::array::from_fn(|byte| {
                raw.vocab.get(&format!("<0x{byte:02X}>")).copied()
            })));
        }
        if bpe.ignore_merges {
            bpe.raw_token_to_id = Some(Arc::new(raw_bytelevel_vocab(&raw.vocab)));
        }
        Ok(bpe)
    }
}

fn raw_bytelevel_vocab(vocab: &Vocab) -> FxHashMap<Vec<u8>, u32> {
    let mut char_to_byte = [None; 324];
    for (byte, &ch) in BYTE_TO_CHAR.iter().enumerate() {
        char_to_byte[ch as usize] = Some(byte as u8);
    }

    let mut raw = HashMap::with_capacity_and_hasher(vocab.len(), FxBuildHasher);
    for (token, &id) in vocab {
        let mut bytes = Vec::with_capacity(token.chars().count());
        let mut valid = true;
        for ch in token.chars() {
            match char_to_byte.get(ch as usize).copied().flatten() {
                Some(byte) => bytes.push(byte),
                None => {
                    valid = false;
                    break;
                }
            }
        }
        if valid {
            raw.insert(bytes, id);
        }
    }
    raw
}

enum Decomposition {
    Pair(TokenId, TokenId),
    CharsNotInVocab,
    Stuck,
}

fn encoding_decomposition(
    text: &str,
    char_to_id: &FxHashMap<char, TokenId>,
    merge_map: &DecompositionMergeMap,
    tokens: &mut Vec<TokenId>,
) -> Decomposition {
    tokens.clear();
    for ch in text.chars() {
        match char_to_id.get(&ch) {
            Some(&tid) => tokens.push(tid),
            None => return Decomposition::CharsNotInVocab,
        }
    }

    if tokens.len() < 2 {
        return Decomposition::CharsNotInVocab;
    }

    while tokens.len() > 2 {
        let mut best_rank = u32::MAX;
        let mut best_pos = usize::MAX;
        let mut best_new = 0;
        for i in 0..tokens.len() - 1 {
            let pair = (tokens[i], tokens[i + 1]);
            if let Some(&(rank, new_id)) = merge_map.get(&pair)
                && rank < best_rank
            {
                best_rank = rank;
                best_pos = i;
                best_new = new_id;
            }
        }
        if best_pos == usize::MAX {
            return Decomposition::Stuck;
        }
        tokens[best_pos] = best_new;
        tokens.remove(best_pos + 1);
    }

    Decomposition::Pair(tokens[0], tokens[1])
}

fn parse_merges(vocab: &Vocab, merges: &[Value]) -> Result<ParsedMergeMap> {
    let mut merge_map = ParsedMergeMap::new();
    for (rank, entry) in merges.iter().enumerate() {
        let (left, right) = parse_merge_entry(entry)?;
        let &left_id = vocab
            .get(left)
            .ok_or_else(|| format!("merge token not in vocab: {left:?}"))?;
        let &right_id = vocab
            .get(right)
            .ok_or_else(|| format!("merge token not in vocab: {right:?}"))?;
        let merged = format!("{left}{right}");
        let &merged_id = vocab
            .get(&merged)
            .ok_or_else(|| format!("merged token not in vocab: {merged:?}"))?;
        merge_map.insert((left_id, right_id), (rank as u32, merged_id));
    }
    Ok(merge_map)
}

fn parse_merge_entry(entry: &Value) -> Result<(&str, &str)> {
    match entry {
        Value::String(s) => {
            let (left, right) = s
                .split_once(' ')
                .ok_or_else(|| format!("invalid merge entry (no space): {s:?}"))?;
            Ok((left, right))
        }
        Value::Array(arr) if arr.len() == 2 => {
            let left = arr[0]
                .as_str()
                .ok_or_else(|| format!("merge element not a string: {:?}", arr[0]))?;
            let right = arr[1]
                .as_str()
                .ok_or_else(|| format!("merge element not a string: {:?}", arr[1]))?;
            Ok((left, right))
        }
        _ => Err(format!("unrecognized merge entry format: {entry:?}")),
    }
}

impl Bpe {
    pub fn new(vocab: &Vocab, merge_map: ParsedMergeMap) -> Result<Self> {
        if vocab.is_empty() {
            return Err("cannot build Bpe with empty vocabulary".into());
        }

        let mut reverse_by_id: HashMap<u32, &str, FxBuildHasher> =
            HashMap::with_capacity_and_hasher(vocab.len(), FxBuildHasher);
        for (text, &id) in vocab {
            reverse_by_id.insert(id, text.as_str());
        }
        let max_token = match reverse_by_id.keys().copied().max() {
            Some(max_token) => max_token,
            None => return Err("cannot build Bpe with empty vocabulary".into()),
        };
        let unique_ids = reverse_by_id.len();
        if max_token as u64 + 1 != unique_ids as u64 {
            let mut ids: Vec<u32> = reverse_by_id.keys().copied().collect();
            ids.sort_unstable();
            let mut expected = 0u32;
            for id in ids {
                if id != expected {
                    return Err(format!(
                        "non-contiguous tokens - token {expected} is missing"
                    ));
                }
                expected = expected.saturating_add(1);
            }
            return Err(format!(
                "non-contiguous tokens - token {expected} is missing"
            ));
        }
        let mut reverse_vocab: Vec<Option<&str>> = vec![None; unique_ids];
        for (&id, text) in &reverse_by_id {
            reverse_vocab[id as usize] = Some(*text);
        }
        let id_to_token: Vec<Arc<str>> = reverse_vocab
            .iter()
            .enumerate()
            .map(|(t, text)| {
                text.map(Arc::<str>::from)
                    .ok_or_else(|| format!("non-contiguous tokens - token {t} is missing"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if id_to_token.iter().any(|text| text.is_empty()) {
            return Err(
                "error building DAAC: InvalidArgumentError: pattern.len() must be >= 1".into(),
            );
        }

        let mut char_to_id = HashMap::with_capacity_and_hasher(324, FxBuildHasher);
        for (text, &id) in vocab {
            let mut chars = text.chars();
            if let (Some(ch), None) = (chars.next(), chars.next()) {
                char_to_id.insert(ch, id);
            }
        }
        let mut decomposition_merge_map =
            HashMap::with_capacity_and_hasher(merge_map.len(), FxBuildHasher);
        decomposition_merge_map.extend(merge_map.iter().map(|(&pair, &value)| (pair, value)));

        let vocab_size = id_to_token.len();
        let mut unmerge_map = (0..vocab_size)
            .map(|t| (t as u32, t as u32))
            .collect::<Vec<_>>();
        let mut is_orphan = vec![false; vocab_size];
        let mut decomposition_tokens = Vec::new();
        for (tid, text) in id_to_token.iter().enumerate() {
            if text.chars().count() < 2 {
                continue;
            }
            match encoding_decomposition(
                text,
                &char_to_id,
                &decomposition_merge_map,
                &mut decomposition_tokens,
            ) {
                Decomposition::Pair(left, right) => {
                    unmerge_map[tid] = (left, right);
                }
                Decomposition::Stuck => {
                    is_orphan[tid] = true;
                }
                Decomposition::CharsNotInVocab => {}
            }
        }

        let token_lens: Vec<u16> = id_to_token
            .iter()
            .enumerate()
            .map(|(t, text)| {
                u16::try_from(text.len())
                    .map_err(|_| format!("token {t} length {} exceeds u16::MAX", text.len()))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let flat_merge_map = MergeMap::from_parsed(&merge_map);
        let ranked_merge_map = RankedMergeMap::from_parsed(&merge_map);

        let mut byte_to_initial_token = [INVALID_TOKEN; 256];
        for byte_val in 0u16..256 {
            let ch = BYTE_TO_CHAR[byte_val as usize];
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            if let Some(&id) = vocab.get(s) {
                byte_to_initial_token[byte_val as usize] = id;
            }
        }

        // Pre-compute initial byte-pair merges (256×256 table).
        let mut byte_pair_initial = vec![(u32::MAX, 0u32); 65536];
        for b1 in 0u16..256 {
            let t1 = byte_to_initial_token[b1 as usize];
            if t1 == INVALID_TOKEN {
                continue;
            }
            for b2 in 0u16..256 {
                let t2 = byte_to_initial_token[b2 as usize];
                if t2 == INVALID_TOKEN {
                    continue;
                }
                if let Some((rank, new_id)) = ranked_merge_map.get(t1, t2) {
                    byte_pair_initial[b1 as usize * 256 + b2 as usize] = (rank, new_id);
                }
            }
        }

        let merge_adj = MergeAdjacency::from_parsed(&merge_map, vocab_size);

        let mut token_to_id = HashMap::with_capacity_and_hasher(vocab.len(), FxBuildHasher);
        for (text, &id) in vocab {
            let (key, whole_token) = match id_to_token.get(id as usize) {
                Some(existing) if existing.as_ref() == text => {
                    (Arc::clone(existing), !is_orphan[id as usize])
                }
                _ => (Arc::<str>::from(text.as_str()), false),
            };
            token_to_id.insert(key, TokenEntry { id, whole_token });
        }

        Ok(Self {
            id: BPE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            content_affinity_available: AtomicBool::new(true),
            merge_map: flat_merge_map,
            unmerge_map,
            token_lens,
            shared_cache: SharedCache::new(),
            fused_shared_cache: SharedCache::new(),
            id_to_token,
            token_to_id,
            raw_token_to_id: None,
            byte_to_initial_token,
            ranked_merge_map,
            byte_pair_initial,
            merge_adj,
            orphan: is_orphan,
            ignore_merges: false,
            byte_fallback_ids: None,
        })
    }

    pub fn is_compatible_token_pair(&self, mut t1: TokenId, mut t2: TokenId) -> bool {
        if t1 == INVALID_TOKEN {
            return false;
        }

        let mut limit = u32::MAX;
        loop {
            if let Some(t) = self.merge_map.get(t1, t2)
                && t < limit
            {
                return false;
            }

            if t1 > t2 {
                limit = t1;
                t1 = self.unmerge_map[t1 as usize].1;
                if t1 == limit {
                    limit = t2 + 1;
                    t2 = self.unmerge_map[t2 as usize].0;
                    if t2 + 1 == limit {
                        return true;
                    }
                }
            } else {
                limit = t2 + 1;
                t2 = self.unmerge_map[t2 as usize].0;
                if t2 + 1 == limit {
                    limit = t1;
                    t1 = self.unmerge_map[t1 as usize].1;
                    if t1 == limit {
                        return true;
                    }
                }
            }
        }
    }

    fn next_match(&self, input: &str) -> Option<TokenId> {
        self.token_to_id
            .get(input)
            .and_then(|entry| entry.whole_token.then_some(entry.id))
    }

    pub fn tokenize(&self, input: &str) -> Result<Vec<TokenId>> {
        let mut out = Vec::new();
        self.tokenize_into(input, &mut out)?;
        Ok(out)
    }

    #[inline(always)]
    pub fn tokenize_into(&self, input: &str, out: &mut Vec<u32>) -> Result<()> {
        if input.is_empty() {
            return Ok(());
        }

        if let Some(token) = self.next_match(input)
            && self.token_lens[token as usize] as usize == input.len()
        {
            out.push(token);
            return Ok(());
        }

        let bpe_id = self.id;
        let hash = FlatCache::hash_str(input);
        let hit = TL_BPE_CACHE.with(|c| {
            let c = c.borrow();
            if c.bpe_id != bpe_id {
                return false;
            }
            c.get_hashed(input, hash, out)
        });
        if hit {
            return Ok(());
        }

        let start = out.len();
        if self.shared_cache.get_hashed(input, hash, out) {
            TL_BPE_CACHE.with(|c| {
                let mut c = c.borrow_mut();
                if c.bpe_id != bpe_id {
                    c.bpe_id = bpe_id;
                    c.clear();
                }
                c.insert_hashed(input, hash, &out[start..]);
            });
            return Ok(());
        }

        self.merge_all_encoded_into(input, out)?;

        let ids = &out[start..];
        TL_BPE_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.bpe_id != bpe_id {
                c.bpe_id = bpe_id;
                c.clear();
            }
            c.insert_hashed(input, hash, ids);
        });
        self.shared_cache.insert_hashed(input, hash, ids);

        Ok(())
    }

    /// Priority-queue BPE merge on already-encoded (ByteLevel) text.
    fn merge_all_encoded_into(&self, input: &str, out: &mut Vec<u32>) -> Result<()> {
        if input.is_empty() {
            return Ok(());
        }

        TL_MERGE_SCRATCH.with(|s| {
            let mut scratch = s.borrow_mut();
            scratch.symbols.clear();
            scratch.heap.clear();

            let mut n = 0usize;
            for ch in input.chars() {
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                let mut fallback = [0; 4];
                let ids = if let Some(entry) = self.token_to_id.get(s) {
                    std::slice::from_ref(&entry.id)
                } else if let Some(byte_ids) = &self.byte_fallback_ids {
                    for (index, byte) in s.bytes().enumerate() {
                        fallback[index] = byte_ids[byte as usize]
                            .ok_or_else(|| format!("fallback byte {byte:02X} not in vocabulary"))?;
                    }
                    &fallback[..s.len()]
                } else {
                    return Err(format!("character {ch:?} not in vocabulary"));
                };
                for &id in ids {
                    scratch.symbols.push(MergeSymbol {
                        c: id,
                        prev: if n == 0 { -1 } else { (n - 1) as i32 },
                        next: -1,
                    });
                    if n > 0 {
                        scratch.symbols[n - 1].next = n as i32;
                    }
                    n += 1;
                }
            }

            if n == 1 {
                out.push(scratch.symbols[0].c);
                return Ok(());
            }

            self.init_merge_heap(&mut scratch, n);
            self.run_merge_loop(&mut scratch, out);
            Ok(())
        })
    }

    /// Linear-scan BPE merge for short pretokens (`n <= SMALL_MERGE_MAX`
    /// initial symbols). Avoids the `BinaryHeap` entirely: a stack-resident
    /// doubly-linked list plus a per-position rank array, find-min by a short
    /// scan over stack `u32`s, merge (O(1) pointer update), then refresh only
    /// the two neighbor pairs. At these sizes this beats the heap's
    /// sift/stale-entry traffic and does zero heap allocation.
    ///
    /// Produces the identical token sequence as [`run_merge_loop`]: both
    /// process the globally lowest-`(rank, pos)` active pair each step (the
    /// heap's `MergeEntry` key is `(rank << 32) | pos`; the scan's strict
    /// `<` keeps the leftmost/lowest-`pos` position on ties). Enforced by
    /// `merge_small_matches_heap` differential test. `ids[..n]` are the
    /// per-byte initial token ids.
    fn merge_small_raw(
        &self,
        bytes: &[u8],
        ids: &mut [u32; SMALL_MERGE_MAX],
        n: usize,
        out: &mut Vec<u32>,
    ) {
        let mut next = [0u8; SMALL_MERGE_MAX];
        let mut prev = [0u8; SMALL_MERGE_MAX];
        let mut ranks = [u32::MAX; SMALL_MERGE_MAX];
        let mut new_ids = [0u32; SMALL_MERGE_MAX];
        for i in 0..n {
            next[i] = (i + 1) as u8;
            prev[i] = (i as u8).wrapping_sub(1); // prev[0] = 255 (>= n): sentinel
        }
        // Round-1 ranks via the dense byte-pair table: one direct-indexed
        // load per pair instead of a CSR neighbor scan.
        for i in 0..n - 1 {
            let (rank, new_id) =
                self.byte_pair_initial[bytes[i] as usize * 256 + bytes[i + 1] as usize];
            if rank != u32::MAX {
                ranks[i] = rank;
                new_ids[i] = new_id;
            }
        }
        loop {
            let mut best = u32::MAX;
            let mut best_i = 0usize;
            for (i, &rank) in ranks[..n - 1].iter().enumerate() {
                if rank < best {
                    best = rank;
                    best_i = i;
                }
            }
            if best == u32::MAX {
                break;
            }
            let i = best_i;
            ids[i] = new_ids[i];
            let dead = next[i] as usize;
            let new_right = next[dead] as usize;
            next[i] = new_right as u8;
            ranks[dead] = u32::MAX;
            if new_right < n {
                prev[new_right] = i as u8;
                match self.merge_adj.get(ids[i], ids[new_right]) {
                    Some((rank, new_id)) => {
                        ranks[i] = rank;
                        new_ids[i] = new_id;
                    }
                    None => ranks[i] = u32::MAX,
                }
            } else {
                ranks[i] = u32::MAX;
            }
            let left = prev[i] as usize;
            if left < n {
                match self.merge_adj.get(ids[left], ids[i]) {
                    Some((rank, new_id)) => {
                        ranks[left] = rank;
                        new_ids[left] = new_id;
                    }
                    None => ranks[left] = u32::MAX,
                }
            }
        }
        let mut i = 0usize;
        while i < n {
            out.push(ids[i]);
            i = next[i] as usize;
        }
    }

    /// `ignore_merges` whole-pretoken lookup against the raw-byte inverse of
    /// the ByteLevel vocabulary, avoiding per-lookup remapping and allocation.
    #[inline]
    fn whole_pretoken_id(&self, raw: &str) -> Option<u32> {
        debug_assert!(
            self.raw_token_to_id.is_some(),
            "ignore_merges without raw vocab index"
        );
        self.raw_token_to_id
            .as_ref()
            .and_then(|vocab| vocab.get(raw.as_bytes()).copied())
    }

    /// Priority-queue BPE merge on raw (pre-ByteLevel) bytes.
    fn merge_all_raw_into(&self, raw_input: &str, out: &mut Vec<u32>) -> Result<()> {
        if raw_input.is_empty() {
            return Ok(());
        }

        // Short pretokens (the common case) skip the heap + thread-local
        // scratch and use the stack-resident linear-scan merge. `n` initial
        // symbols == byte length, so the tier is known up front.
        let bytes = raw_input.as_bytes();
        if bytes.len() <= SMALL_MERGE_MAX {
            let n = bytes.len();
            let mut ids = [0u32; SMALL_MERGE_MAX];
            for (i, &byte) in bytes.iter().enumerate() {
                let id = self.byte_to_initial_token[byte as usize];
                if id == INVALID_TOKEN {
                    return Err(format!("byte 0x{byte:02x} has no token in vocabulary"));
                }
                ids[i] = id;
            }
            if n == 1 {
                out.push(ids[0]);
            } else {
                self.merge_small_raw(bytes, &mut ids, n, out);
            }
            return Ok(());
        }

        self.merge_all_raw_heap_into(raw_input, out)
    }

    /// Reference heap implementation used for long pretokens and as the
    /// correctness oracle for the short linear-scan tier.
    fn merge_all_raw_heap_into(&self, raw_input: &str, out: &mut Vec<u32>) -> Result<()> {
        TL_MERGE_SCRATCH.with(|s| {
            let mut scratch = s.borrow_mut();
            scratch.symbols.clear();
            scratch.heap.clear();
            scratch.heap_buf.clear();

            let bytes = raw_input.as_bytes();
            let n = bytes.len();
            let mut prev_byte = 0u8;
            for (i, &byte) in bytes.iter().enumerate() {
                let id = self.byte_to_initial_token[byte as usize];
                if id == INVALID_TOKEN {
                    return Err(format!("byte 0x{byte:02x} has no token in vocabulary"));
                }
                scratch.symbols.push(MergeSymbol {
                    c: id,
                    prev: if i == 0 { -1 } else { (i - 1) as i32 },
                    next: if i == n - 1 { -1 } else { (i + 1) as i32 },
                });
                // Check pair with previous byte via pre-computed table.
                if i > 0 {
                    let (rank, _new_id) =
                        self.byte_pair_initial[prev_byte as usize * 256 + byte as usize];
                    if rank != u32::MAX {
                        scratch.heap_buf.push(Reverse(MergeEntry::new(
                            rank,
                            (i - 1) as u32,
                            self.byte_to_initial_token[prev_byte as usize],
                            id,
                        )));
                    }
                }
                prev_byte = byte;
            }

            if n == 1 {
                out.push(scratch.symbols[0].c);
                return Ok(());
            }

            // Bulk heapify.
            let mut tmp = std::mem::take(&mut scratch.heap_buf);
            scratch.heap.extend(tmp.drain(..));
            scratch.heap_buf = tmp;

            self.run_merge_loop(&mut scratch, out);

            Ok(())
        })
    }

    /// Seed the priority queue with all initial adjacent pairs.
    #[inline(always)]
    fn init_merge_heap(&self, scratch: &mut MergeScratch, n: usize) {
        let symbols = &scratch.symbols;
        scratch.heap.extend((0..n - 1).filter_map(|i| {
            let left = symbols[i].c;
            let right = symbols[i + 1].c;
            self.merge_adj
                .get(left, right)
                .map(|(rank, _new_id)| Reverse(MergeEntry::new(rank, i as u32, left, right)))
        }));
    }

    #[inline(always)]
    fn run_merge_loop(&self, scratch: &mut MergeScratch, out: &mut Vec<u32>) {
        let symbols = &mut scratch.symbols;
        let heap = &mut scratch.heap;

        while let Some(Reverse(entry)) = heap.pop() {
            let pos = entry.pos() as usize;
            let sym = symbols[pos];

            // Stale-entry check.
            let left_c = entry.left_c();
            let right_c = entry.right_c();
            if sym.c != left_c {
                continue;
            }
            let next_idx = sym.next;
            if next_idx < 0 {
                continue;
            }
            let next_idx = next_idx as usize;
            let next_sym = symbols[next_idx];
            if next_sym.c != right_c {
                continue;
            }

            // Derive new_id from adjacency list.
            let new_id = match self.merge_adj.get(left_c, right_c) {
                Some((_, nid)) => nid,
                None => continue,
            };

            // Merge: left symbol absorbs right.
            symbols[pos].c = new_id;
            symbols[pos].next = next_sym.next;
            if next_sym.next >= 0 {
                symbols[next_sym.next as usize].prev = pos as i32;
            }
            symbols[next_idx].c = INVALID_TOKEN;

            // Discover new adjacent pairs.
            if sym.prev >= 0 {
                let prev_c = symbols[sym.prev as usize].c;
                if let Some((rank, _)) = self.merge_adj.get(prev_c, new_id) {
                    heap.push(Reverse(MergeEntry::new(
                        rank,
                        sym.prev as u32,
                        prev_c,
                        new_id,
                    )));
                }
            }
            let new_next = symbols[pos].next;
            if new_next >= 0 {
                let next_c = symbols[new_next as usize].c;
                if let Some((rank, _)) = self.merge_adj.get(new_id, next_c) {
                    heap.push(Reverse(MergeEntry::new(rank, pos as u32, new_id, next_c)));
                }
            }
        }

        let mut i: i32 = 0;
        while i >= 0 {
            let sym = symbols[i as usize];
            out.push(sym.c);
            i = sym.next;
        }
    }

    #[inline(always)]
    pub fn tokenize_into_fused(&self, raw_input: &str, out: &mut Vec<u32>) -> Result<()> {
        if raw_input.is_empty() {
            return Ok(());
        }

        let bpe_id = self.id;
        let hash = FlatCache::hash_str(raw_input);
        let hit = TL_FUSED_CACHE.with(|c| {
            let c = c.borrow();
            if c.bpe_id != bpe_id {
                return false;
            }
            c.get_hashed(raw_input, hash, out)
        });
        if hit {
            return Ok(());
        }

        let start = out.len();
        if self.fused_shared_cache.get_hashed(raw_input, hash, out) {
            TL_FUSED_CACHE.with(|c| {
                let mut c = c.borrow_mut();
                if c.bpe_id != bpe_id {
                    c.bpe_id = bpe_id;
                    c.clear();
                }
                c.insert_hashed(raw_input, hash, &out[start..]);
            });
            return Ok(());
        }

        if self.ignore_merges
            && let Some(id) = self.whole_pretoken_id(raw_input)
        {
            out.push(id);
            return Ok(());
        }

        self.merge_all_raw_into(raw_input, out)?;

        let ids = &out[start..];
        TL_FUSED_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.bpe_id != bpe_id {
                c.bpe_id = bpe_id;
                c.clear();
            }
            c.insert_hashed(raw_input, hash, ids);
        });
        self.fused_shared_cache.insert_hashed(raw_input, hash, ids);

        Ok(())
    }

    pub fn tokenize_batch_fused(
        &self,
        buffer: &str,
        splits: &[crate::pre_tokenized::Split],
        out: &mut Vec<u32>,
    ) -> Result<()> {
        self.tokenize_batch_fused_impl::<false, _>(buffer, splits.iter(), out, &mut Vec::new())
    }

    pub(crate) fn tokenize_batch_fused_by_indices(
        &self,
        buffer: &str,
        splits: &[crate::pre_tokenized::Split],
        indices: &[u32],
        out: &mut Vec<u32>,
        ends: &mut Vec<usize>,
    ) -> Result<()> {
        self.tokenize_batch_fused_impl::<true, _>(
            buffer,
            indices.iter().map(|&index| &splits[index as usize]),
            out,
            ends,
        )
    }

    fn tokenize_batch_fused_impl<'a, const RECORD_ENDS: bool, I>(
        &self,
        buffer: &str,
        splits: I,
        out: &mut Vec<u32>,
        ends: &mut Vec<usize>,
    ) -> Result<()>
    where
        I: Iterator<Item = &'a crate::pre_tokenized::Split>,
    {
        let bpe_id = self.id;
        TL_FUSED_CACHE.with(|c| {
            let mut cache = c.borrow_mut();
            if cache.bpe_id != bpe_id {
                cache.bpe_id = bpe_id;
                cache.clear();
            }

            for split in splits {
                if let Some(id) = split.token_id {
                    out.push(id);
                } else if !split.range.is_empty() {
                    let text = &buffer[split.range.clone()];
                    if text.is_empty() {
                        if RECORD_ENDS {
                            ends.push(out.len());
                        }
                        continue;
                    }

                    let hash = FlatCache::hash_str(text);
                    if cache.get_hashed(text, hash, out) {
                        if RECORD_ENDS {
                            ends.push(out.len());
                        }
                        continue;
                    }

                    let start = out.len();
                    if self.fused_shared_cache.get_hashed(text, hash, out) {
                        cache.insert_hashed(text, hash, &out[start..]);
                        if RECORD_ENDS {
                            ends.push(out.len());
                        }
                        continue;
                    }

                    if self.ignore_merges
                        && let Some(id) = self.whole_pretoken_id(text)
                    {
                        out.push(id);
                        cache.insert_hashed(text, hash, &out[start..]);
                        if RECORD_ENDS {
                            ends.push(out.len());
                        }
                        continue;
                    }

                    self.merge_all_raw_into(text, out)?;

                    cache.insert_hashed(text, hash, &out[start..]);
                    self.fused_shared_cache
                        .insert_hashed(text, hash, &out[start..]);
                }
                if RECORD_ENDS {
                    ends.push(out.len());
                }
            }
            Ok(())
        })
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(Arc::as_ref)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).map(|entry| entry.id)
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub(crate) fn take_content_affinity(&self) -> bool {
        self.content_affinity_available
            .swap(false, Ordering::Relaxed)
    }
}

impl Clone for Bpe {
    fn clone(&self) -> Self {
        Self {
            id: BPE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            content_affinity_available: AtomicBool::new(true),
            merge_map: self.merge_map.clone(),
            unmerge_map: self.unmerge_map.clone(),
            token_lens: self.token_lens.clone(),
            shared_cache: SharedCache::new(),
            fused_shared_cache: SharedCache::new(),
            id_to_token: self.id_to_token.clone(),
            token_to_id: self.token_to_id.clone(),
            raw_token_to_id: self.raw_token_to_id.clone(),
            byte_to_initial_token: self.byte_to_initial_token,
            ranked_merge_map: self.ranked_merge_map.clone(),
            byte_pair_initial: self.byte_pair_initial.clone(),
            merge_adj: self.merge_adj.clone(),
            orphan: self.orphan.clone(),
            ignore_merges: self.ignore_merges,
            byte_fallback_ids: self.byte_fallback_ids.clone(),
        }
    }
}

impl fmt::Debug for Bpe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bpe")
            .field("vocab_size", &self.token_lens.len())
            .field("merges", &self.merge_map.len())
            .finish()
    }
}

impl PartialEq for Bpe {
    fn eq(&self, other: &Self) -> bool {
        self.id_to_token == other.id_to_token
            && self.merge_map == other.merge_map
            && self.unmerge_map == other.unmerge_map
            && self.token_lens == other.token_lens
            && self.orphan == other.orphan
            && self.ignore_merges == other.ignore_merges
            && self.byte_fallback_ids == other.byte_fallback_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_structs::ModelConfig;

    fn test_bpe() -> Bpe {
        let vocab: Vocab = [
            ("a", 0),
            ("b", 1),
            ("c", 2),
            ("d", 3),
            ("ab", 4),
            ("cd", 5),
            ("abcd", 6),
        ]
        .into_iter()
        .map(|(s, id)| (s.to_string(), id))
        .collect();

        let merges: Vec<Value> = vec![
            Value::String("a b".into()),
            Value::String("c d".into()),
            Value::String("ab cd".into()),
        ];

        let merge_map = parse_merges(&vocab, &merges).unwrap();
        Bpe::new(&vocab, merge_map).unwrap()
    }

    #[test]
    fn empty_input() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn single_char() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("a").unwrap(), vec![0]);
        assert_eq!(bpe.tokenize("d").unwrap(), vec![3]);
    }

    #[test]
    fn simple_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("ab").unwrap(), vec![4]);
        assert_eq!(bpe.tokenize("cd").unwrap(), vec![5]);
    }

    #[test]
    fn chained_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("abcd").unwrap(), vec![6]);
    }

    #[test]
    fn partial_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("abc").unwrap(), vec![4, 2]);
    }

    #[test]
    fn repeated_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("abab").unwrap(), vec![4, 4]);
    }

    #[test]
    fn content_affinity_is_available_once_per_bpe() {
        let bpe = test_bpe();
        assert!(bpe.take_content_affinity());
        assert!(!bpe.take_content_affinity());
    }

    #[test]
    fn merge_small_matches_heap() {
        let bpe = test_bpe();
        let alphabet = [b'a', b'b', b'c', b'd'];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;

        for len in 1..=SMALL_MERGE_MAX {
            for _ in 0..512 {
                let mut bytes = Vec::with_capacity(len);
                for _ in 0..len {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    bytes.push(alphabet[state as usize & 3]);
                }
                let input = std::str::from_utf8(&bytes).unwrap();
                let mut small = Vec::new();
                let mut heap = Vec::new();

                bpe.merge_all_raw_into(input, &mut small).unwrap();
                bpe.merge_all_raw_heap_into(input, &mut heap).unwrap();
                assert_eq!(small, heap, "short merge mismatch for {input:?}");
            }
        }
    }

    #[test]
    fn merge_small_matches_heap_on_multilingual_corpus() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("vendored_tokenizers/kimi-k2.5/tokenizer.json");
        let tokenizer = crate::Tokenizer::from_file(&path).unwrap();
        let crate::Model::Bpe(bpe) = tokenizer.model();
        let corpus = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/corpus_multilingual.txt"),
        )
        .unwrap()
        .replace("<|endoftext|>", " ");
        let mut pts = tokenizer.build_pre_tokenized(&corpus);
        tokenizer
            .split_only
            .as_ref()
            .expect("Kimi should use the fused split path")
            .pre_tokenize(&mut pts)
            .unwrap();

        let mut seen = std::collections::HashSet::new();
        let mut checked_non_ascii = false;
        for split in pts.splits() {
            let input = pts.split_text(split);
            if input.is_empty() || input.len() > SMALL_MERGE_MAX || !seen.insert(input) {
                continue;
            }

            let mut small = Vec::new();
            let mut heap = Vec::new();
            bpe.merge_all_raw_into(input, &mut small).unwrap();
            bpe.merge_all_raw_heap_into(input, &mut heap).unwrap();
            assert_eq!(small, heap, "short merge mismatch for {input:?}");
            checked_non_ascii |= !input.is_ascii();
        }

        assert!(
            seen.len() > 1_000,
            "expected broad production pretoken coverage"
        );
        assert!(checked_non_ascii, "expected non-ASCII pretokens");
    }

    #[test]
    fn deserialize_from_json() {
        let json = serde_json::json!({
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2},
            "merges": ["a b"]
        });
        let config: ModelConfig = serde_json::from_value(json).unwrap();
        assert!(matches!(config, ModelConfig::Bpe(_)));
    }

    #[test]
    fn deserialize_array_merges() {
        let json = serde_json::json!({
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2},
            "merges": [["a", "b"]]
        });
        let config: ModelConfig = serde_json::from_value(json).unwrap();
        let ModelConfig::Bpe(bpe) = config else {
            panic!("expected Bpe variant");
        };
        assert_eq!(bpe.tokenize("ab").unwrap(), vec![2]);
    }

    #[test]
    fn raw_bytelevel_vocab_reverses_encoded_tokens() {
        let encoded = [
            BYTE_TO_CHAR[b' ' as usize],
            BYTE_TO_CHAR[b'h' as usize],
            BYTE_TO_CHAR[0xff],
        ]
        .iter()
        .collect::<String>();
        let vocab = HashMap::from([(encoded, 42), ("\u{1f642}".to_string(), 99)]);

        let raw = raw_bytelevel_vocab(&vocab);
        assert_eq!(raw.get(&[b' ', b'h', 0xff][..]), Some(&42));
        assert!(!raw.values().any(|&id| id == 99));
    }

    #[test]
    fn cache_returns_same_result() {
        let vocab: Vocab = [("a", 0), ("b", 1), ("ab", 2)]
            .into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();
        let merges = vec![Value::String("a b".into())];
        let merge_map = parse_merges(&vocab, &merges).unwrap();
        let bpe = Bpe::new(&vocab, merge_map).unwrap();

        let first = bpe.tokenize("ab").unwrap();
        let second = bpe.tokenize("ab").unwrap();
        assert_eq!(first, second);
        assert_eq!(first, vec![2]);
    }

    #[test]
    fn orphan_token_does_not_shortcut_character_merge() {
        let vocab: Vocab = [("a", 0), ("b", 1), ("c", 2), ("abc", 3)]
            .into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();
        let bpe = Bpe::new(&vocab, parse_merges(&vocab, &[]).unwrap()).unwrap();

        assert_eq!(bpe.tokenize("a").unwrap(), vec![0]);
        assert_eq!(bpe.tokenize("abc").unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn non_contiguous_vocab_ids_are_rejected() {
        let vocab: Vocab = [("a", 0), ("c", 2)]
            .into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();

        let error = Bpe::new(&vocab, parse_merges(&vocab, &[]).unwrap()).unwrap_err();
        assert!(error.contains("non-contiguous tokens - token 1 is missing"));
    }

    #[test]
    fn sparse_vocab_ids_are_rejected_without_allocating_the_id_range() {
        let vocab: Vocab = [("a", 0), ("b", u32::MAX)]
            .into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();
        let error = Bpe::new(&vocab, parse_merges(&vocab, &[]).unwrap()).unwrap_err();
        assert_eq!(error, "non-contiguous tokens - token 1 is missing");

        let vocab: Vocab = [("a", 10)]
            .into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();
        let error = Bpe::new(&vocab, parse_merges(&vocab, &[]).unwrap()).unwrap_err();
        assert_eq!(error, "non-contiguous tokens - token 0 is missing");
    }

    #[test]
    fn equality_includes_canonical_token_text_and_survives_clone() {
        let vocab_a: Vocab = [("aa", 0), ("bb", 1)]
            .into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();
        let vocab_b: Vocab = [("cc", 0), ("dd", 1)]
            .into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();
        let bpe_a = Bpe::new(&vocab_a, parse_merges(&vocab_a, &[]).unwrap()).unwrap();
        let bpe_a_again = Bpe::new(&vocab_a, parse_merges(&vocab_a, &[]).unwrap()).unwrap();
        let bpe_b = Bpe::new(&vocab_b, parse_merges(&vocab_b, &[]).unwrap()).unwrap();

        assert_eq!(bpe_a, bpe_a.clone());
        assert_eq!(bpe_a, bpe_a_again);
        assert_ne!(bpe_a, bpe_b);
    }

    #[test]
    fn empty_vocab_token_is_rejected_like_daac() {
        let vocab: Vocab = [("", 0), ("a", 1)]
            .into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();
        let error = Bpe::new(&vocab, parse_merges(&vocab, &[]).unwrap()).unwrap_err();
        assert_eq!(
            error,
            "error building DAAC: InvalidArgumentError: pattern.len() must be >= 1"
        );
    }

    #[test]
    fn duplicate_vocab_ids_keep_all_token_lookups() {
        let vocab: Vocab = [("a", 0), ("b", 0), ("c", 1)]
            .into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();
        let bpe = Bpe::new(&vocab, parse_merges(&vocab, &[]).unwrap()).unwrap();

        assert_eq!(bpe.token_to_id("a"), Some(0));
        assert_eq!(bpe.token_to_id("b"), Some(0));
        assert_eq!(bpe.tokenize("a").unwrap(), vec![0]);
        assert_eq!(bpe.tokenize("b").unwrap(), vec![0]);
    }

    #[test]
    fn shared_cache_allocates_shards_lazily() {
        let cache = SharedCache::new();
        assert!(
            cache
                .shards
                .iter()
                .all(|shard| shard.lock().unwrap().is_none())
        );

        let hash = FlatCache::hash_str("piece");
        cache.insert_hashed("piece", hash, &[1, 2]);
        assert_eq!(
            cache
                .shards
                .iter()
                .filter(|shard| shard.lock().unwrap().is_some())
                .count(),
            1
        );
        let mut out = Vec::new();
        assert!(cache.get_hashed("piece", hash, &mut out));
        assert_eq!(out, [1, 2]);
    }
}
