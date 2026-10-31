// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Bounded LRU reuse of backend workspaces.
//!
//! `reserved_bytes` accounts for caller-owned memory (the active workspace);
//! `cached_bytes` accounts for entries here. While the reservation alone fits,
//! operations keep their sum within `byte_budget` using saturating arithmetic.
//! `reserve` and `put` evict least-recently used entries and return every
//! displaced key/value for the owner to drop or synchronize. A reservation above
//! budget empties the cache; the excess remains active. `put` rejects values that
//! cannot fit even in an empty cache. Callers supply all byte sizes.

use std::collections::VecDeque;

// Seven inactive arenas plus the engine's active arena retain at most 8 shapes.
pub(crate) const CAPACITY: usize = 7;
pub(crate) const BYTE_BUDGET: usize = 256 * 1024 * 1024;

struct Entry<K, V> {
    key: K,
    value: V,
    bytes: usize,
}

pub(crate) struct WorkspaceCache<K, V> {
    entries: VecDeque<Entry<K, V>>,
    capacity: usize,
    byte_budget: usize,
    cached_bytes: usize,
    reserved_bytes: usize,
}

impl<K, V> WorkspaceCache<K, V> {
    pub(crate) fn new() -> Self {
        Self::with_limits(CAPACITY, BYTE_BUDGET)
    }

    pub(crate) fn with_limits(capacity: usize, byte_budget: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity,
            byte_budget,
            cached_bytes: 0,
            reserved_bytes: 0,
        }
    }

    fn total_with(&self, extra: usize) -> usize {
        self.cached_bytes
            .saturating_add(self.reserved_bytes)
            .saturating_add(extra)
    }

    fn evict_to_budget(&mut self) -> Vec<(K, V)> {
        let mut spilled = Vec::new();
        while !self.entries.is_empty() && self.total_with(0) > self.byte_budget {
            if let Some(old) = self.entries.pop_front() {
                self.cached_bytes = self.cached_bytes.saturating_sub(old.bytes);
                spilled.push((old.key, old.value));
            }
        }
        spilled
    }

    pub(crate) fn reserve(&mut self, bytes: usize) -> Vec<(K, V)> {
        self.reserved_bytes = bytes;
        self.evict_to_budget()
    }

    pub(crate) fn drain(&mut self) -> Vec<(K, V)> {
        self.cached_bytes = 0;
        self.reserved_bytes = 0;
        self.entries
            .drain(..)
            .map(|entry| (entry.key, entry.value))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn access(&mut self, key: &K) -> Option<&V>
    where
        K: PartialEq,
    {
        let pos = self.entries.iter().position(|entry| &entry.key == key)?;
        let entry = self.entries.remove(pos)?;
        self.entries.push_back(entry);
        self.entries.back().map(|entry| &entry.value)
    }

    #[cfg(test)]
    pub(crate) fn take(&mut self, key: &K) -> Option<V>
    where
        K: PartialEq,
    {
        let pos = self.entries.iter().position(|entry| &entry.key == key)?;
        let entry = self.entries.remove(pos)?;
        self.cached_bytes = self.cached_bytes.saturating_sub(entry.bytes);
        Some(entry.value)
    }

    /// Reuse the smallest retained allocation that can serve this request.
    /// The predicate concerns capacity, not an exact logical shape.
    pub(crate) fn take_best_fit(&mut self, mut fits: impl FnMut(&K) -> bool) -> Option<V> {
        let pos = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| fits(&entry.key))
            .min_by_key(|(_, entry)| entry.bytes)
            .map(|(pos, _)| pos)?;
        let entry = self.entries.remove(pos)?;
        self.cached_bytes = self.cached_bytes.saturating_sub(entry.bytes);
        Some(entry.value)
    }

    pub(crate) fn put(&mut self, key: K, value: V, bytes: usize) -> Vec<(K, V)>
    where
        K: PartialEq,
    {
        let mut spilled = Vec::new();
        if self.capacity == 0
            || bytes > self.byte_budget
            || self.reserved_bytes.saturating_add(bytes) > self.byte_budget
        {
            spilled.push((key, value));
            return spilled;
        }
        if let Some(pos) = self.entries.iter().position(|entry| entry.key == key) {
            if let Some(old) = self.entries.remove(pos) {
                self.cached_bytes = self.cached_bytes.saturating_sub(old.bytes);
                spilled.push((old.key, old.value));
            }
        }
        while !self.entries.is_empty()
            && (self.entries.len() >= self.capacity || self.total_with(bytes) > self.byte_budget)
        {
            if let Some(old) = self.entries.pop_front() {
                self.cached_bytes = self.cached_bytes.saturating_sub(old.bytes);
                spilled.push((old.key, old.value));
            }
        }
        self.entries.push_back(Entry { key, value, bytes });
        self.cached_bytes = self.cached_bytes.saturating_add(bytes);
        spilled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_smallest_fitting_arena_and_charges_its_actual_size() {
        let mut cache = WorkspaceCache::with_limits(7, 1000);
        assert!(cache.put(128, "large", 600).is_empty());
        assert!(cache.put(64, "medium", 300).is_empty());
        assert!(cache.put(8, "small", 80).is_empty());
        assert_eq!(
            cache.take_best_fit(|capacity| *capacity >= 32),
            Some("medium")
        );
        assert_eq!(cache.cached_bytes, 680);
        let evicted = cache.reserve(300);
        assert!(evicted.is_empty());
        assert_eq!(cache.total_with(0), 980);
        assert_eq!(
            cache.take_best_fit(|capacity| *capacity >= 64),
            Some("large")
        );
        // Reusing a 600-byte arena for a smaller request must reserve 600,
        // not the 300-byte request estimate. The inactive 300-byte arena fits.
        assert!(cache.reserve(600).is_empty());
        assert!(cache.put(64, "medium", 300).is_empty());
        assert_eq!(cache.total_with(0), 980);
        assert!(cache.take_best_fit(|capacity| *capacity >= 256).is_none());
    }

    #[test]
    fn shape_switches_retain_other_cached_workspaces() {
        let mut cache = WorkspaceCache::with_limits(8, 1024);
        let first = (10, 1, 1, 4);
        let second = (20, 2, 1, 8);
        assert!(cache.put(first, "first", 10).is_empty());
        assert!(cache.put(second, "second", 20).is_empty());
        assert_eq!(cache.take(&second), Some("second"));
        assert!(cache.put(second, "second", 20).is_empty());
        assert_eq!(cache.take(&first), Some("first"));
        assert_eq!(cache.take(&second), Some("second"));
    }

    #[test]
    fn evicts_lru_and_accounts_for_active_reservation() {
        let mut cache = WorkspaceCache::with_limits(3, 1000);
        let a = (1, 1, 1, 1);
        let b = (2, 2, 2, 2);
        let c = (3, 3, 3, 3);
        let d = (4, 4, 4, 4);
        assert!(cache.put(a, "a", 30).is_empty());
        assert!(cache.put(b, "b", 30).is_empty());
        assert!(cache.put(c, "c", 30).is_empty());
        assert_eq!(cache.access(&a), Some(&"a"));
        assert_eq!(cache.put(d, "d", 30), vec![(b, "b")]);
        assert_eq!(cache.take(&a), Some("a"));
        assert_eq!(cache.take(&c), Some("c"));

        let mut cache = WorkspaceCache::with_limits(8, 100);
        assert!(cache.reserve(40).is_empty());
        assert!(cache.put(a, "a", 30).is_empty());
        assert_eq!(cache.put(b, "b", 40), vec![(a, "a")]);
        assert_eq!(cache.put(c, "c", 30), vec![(b, "b")]);
        assert_eq!(cache.put(d, "d", 61), vec![(d, "d")]);
        assert_eq!(cache.take(&c), Some("c"));
    }

    #[test]
    fn oversized_workspace_is_returned_and_never_evicts_cache() {
        let mut cache = WorkspaceCache::with_limits(8, 100);
        let small = (1, 1, 1, 1);
        let huge = (2, 2, 2, 2);
        assert!(cache.put(small, "small", 20).is_empty());
        assert_eq!(cache.put(huge, "huge", 101), vec![(huge, "huge")]);
        assert_eq!(cache.take(&small), Some("small"));
        assert!(cache.put(small, "small", 20).is_empty());
        assert_eq!(cache.reserve(101), vec![(small, "small")]);
        assert_eq!(cache.put(small, "small", 20), vec![(small, "small")]);
    }
}
