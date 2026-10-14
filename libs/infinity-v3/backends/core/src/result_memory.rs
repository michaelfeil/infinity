//! Process-wide admission for owned host inference results.
use crate::BackendError;
use nohash_hasher::IntMap;
use std::ops::{Deref, DerefMut};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, OnceLock,
};

#[derive(Debug)]
pub struct ResultPool {
    limit: usize,
    used: AtomicUsize,
}
impl ResultPool {
    pub fn new(limit: usize) -> Result<Arc<Self>, BackendError> {
        if limit == 0 {
            return Err(BackendError::Start(
                "Host result budget must be positive".into(),
            ));
        }
        Ok(Arc::new(Self {
            limit,
            used: AtomicUsize::new(0),
        }))
    }
    fn acquire_bytes(&self, bytes: usize) -> Result<(), BackendError> {
        self.used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|next| *next <= self.limit)
            })
            .map_err(|_| {
                BackendError::Overloaded("Aggregate host result memory exhausted".into())
            })?;
        Ok(())
    }
    pub fn reserve(self: &Arc<Self>, bytes: usize) -> Result<ResultLease, BackendError> {
        self.acquire_bytes(bytes)?;
        Ok(ResultLease(Arc::new(Reservation {
            pool: self.clone(),
            bytes,
            parent: None,
        })))
    }
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }
}
#[derive(Debug)]
struct Reservation {
    pool: Arc<ResultPool>,
    bytes: usize,
    parent: Option<ResultLease>,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.pool.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
#[derive(Debug, Clone)]
pub struct ResultLease(Arc<Reservation>);
impl ResultLease {
    pub fn reserve_more(&self, bytes: usize) -> Result<Self, BackendError> {
        let mut next = self.0.pool.reserve(bytes)?;
        Arc::get_mut(&mut next.0).unwrap().parent = Some(self.clone());
        Ok(next)
    }
}
static GLOBAL: OnceLock<Result<Arc<ResultPool>, String>> = OnceLock::new();
pub fn global_pool() -> Result<Arc<ResultPool>, BackendError> {
    GLOBAL
        .get_or_init(|| {
            let limit = std::env::var("INFINITY_HOST_RESULTS_MAX_BYTES")
                .map(|v| v.parse::<usize>().map_err(|e| e.to_string()))
                .unwrap_or(Ok(512 << 20))?;
            ResultPool::new(limit).map_err(|e| e.to_string())
        })
        .clone()
        .map_err(BackendError::Start)
}

/// Payload is destroyed before its reservation. Cloning a lease does not
/// duplicate its payload charge; transformations reserve their own allocation.
#[derive(Debug)]
pub struct ResultMap<T> {
    values: IntMap<usize, T>,
    pub memory: Option<ResultLease>,
}
impl<T> Default for ResultMap<T> {
    fn default() -> Self {
        Self {
            values: IntMap::default(),
            memory: None,
        }
    }
}
impl<T> From<IntMap<usize, T>> for ResultMap<T> {
    fn from(values: IntMap<usize, T>) -> Self {
        Self {
            values,
            memory: None,
        }
    }
}
impl<T> Deref for ResultMap<T> {
    type Target = IntMap<usize, T>;
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}
impl<T> DerefMut for ResultMap<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}
pub struct ResultIntoIter<T> {
    values: std::collections::hash_map::IntoIter<usize, T>,
    _memory: Option<ResultLease>,
}
impl<T> Iterator for ResultIntoIter<T> {
    type Item = (usize, T);
    fn next(&mut self) -> Option<Self::Item> {
        self.values.next()
    }
}
impl<T> IntoIterator for ResultMap<T> {
    type Item = (usize, T);
    type IntoIter = ResultIntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        ResultIntoIter {
            values: self.values.into_iter(),
            _memory: self.memory,
        }
    }
}
impl<'a, T> IntoIterator for &'a ResultMap<T> {
    type Item = (&'a usize, &'a T);
    type IntoIter = std::collections::hash_map::Iter<'a, usize, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}
pub trait HasResultMemory {
    fn result_memory(&self) -> Option<ResultLease>;
}
impl<T> HasResultMemory for ResultMap<T> {
    fn result_memory(&self) -> Option<ResultLease> {
        self.memory.clone()
    }
}
impl HasResultMemory for std::vec::IntoIter<crate::DecisionOutput> {
    fn result_memory(&self) -> Option<ResultLease> {
        self.as_slice().first().and_then(|v| v.memory.clone())
    }
}

/// HTTP handoff owner: Core responses retain their own leases; this additional
/// owner bridges handlers that move plain vectors into wire-format structures.
#[derive(Debug, Default)]
struct RequestReservations {
    native: Vec<ResultLease>,
    // One inline accumulator avoids allocating an Arc and growing the native
    // lease vector for every derived string, score row, or response field.
    derived: Option<Reservation>,
}
#[derive(Debug, Default)]
pub struct RequestResults(Mutex<RequestReservations>);
impl RequestResults {
    pub fn retain(&self, memory: Option<ResultLease>) {
        if let Some(memory) = memory {
            self.0.lock().unwrap().native.push(memory);
        }
    }
    pub fn reserve(&self, bytes: usize) -> Result<(), BackendError> {
        if bytes == 0 {
            return Ok(());
        }
        let mut reservations = self.0.lock().unwrap();
        if let Some(derived) = reservations.derived.as_mut() {
            let total = derived.bytes.checked_add(bytes).ok_or_else(|| {
                BackendError::Overloaded("Derived result memory size overflow".into())
            })?;
            derived.pool.acquire_bytes(bytes)?;
            derived.bytes = total;
        } else if let Some(reference) = reservations.native.first().cloned() {
            reference.0.pool.acquire_bytes(bytes)?;
            reservations.derived = Some(Reservation {
                pool: reference.0.pool.clone(),
                bytes,
                parent: Some(reference),
            });
        }
        Ok(())
    }
}
tokio::task_local! { pub static REQUEST_RESULTS: Arc<RequestResults>; }
pub fn retain_for_request(memory: Option<ResultLease>) {
    let _ = REQUEST_RESULTS.try_with(|owner| owner.retain(memory));
}
pub fn reserve_for_request(bytes: usize) -> Result<(), BackendError> {
    REQUEST_RESULTS
        .try_with(|owner| owner.reserve(bytes))
        .unwrap_or(Ok(()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn held_outputs_and_transform_overlap_admit_and_recover() {
        let pool = ResultPool::new(96).unwrap();
        let first = pool.reserve(64).unwrap();
        let first_payload = vec![0u8; 64];
        let response = first.clone();
        assert!(matches!(pool.reserve(64), Err(BackendError::Overloaded(_))));
        let transform = first.reserve_more(32).unwrap();
        let transform_payload = vec![0u8; 32];
        drop(first);
        assert_eq!(pool.used(), 96);
        drop(transform_payload);
        drop(transform);
        assert_eq!(pool.used(), 64);
        drop(first_payload);
        drop(response);
        assert_eq!(pool.used(), 0);
        assert!(pool.reserve(96).is_ok());
        assert!(pool.reserve(usize::MAX).is_err());
        let native = pool.reserve(16).unwrap();
        let request = Arc::new(RequestResults::default());
        request.retain(Some(native.clone()));
        for _ in 0..64 {
            request.reserve(1).unwrap();
        }
        {
            let state = request.0.lock().unwrap();
            assert_eq!(state.native.len(), 1);
            assert_eq!(state.derived.as_ref().unwrap().bytes, 64);
        }
        assert!(request.reserve(17).is_err());
        assert_eq!(pool.used(), 80);
        assert!(request.reserve(usize::MAX).is_err());
        assert_eq!(pool.used(), 80);
        let response = request.clone();
        drop(native);
        drop(request);
        assert_eq!(pool.used(), 80);
        drop(response);
        assert_eq!(pool.used(), 0);
        assert!(pool.reserve(96).is_ok());
    }
}
