// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Pure-CPU deferred owner retirement.
//!
//! A persistent [`CompletionClock`] belongs to one existing host-I/O event
//! slot. Each submission takes one monotonic [`CompletionTicket`] from that
//! clock; the slot is not reusable until the actual same-stream event or stream
//! fence has completed. [`RetirementQueue`] keeps submitted owners and their
//! charged bytes until that proof is observed, so no elapsed-time or polling
//! heuristic can release scratch early. No CUDA, event, query, or timing API is
//! reached from this leaf.

use infinity_v3_backend_core::BackendError;
use std::{
    mem,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

// Two host-I/O slots times the two submit-path owner classes (Workspace and
// Dense) leaves four live pending owners; the extra slots absorb an owner that
// completed but was not yet collected. The array is inline on the cold Engine,
// so no submission allocates queue storage.
const RETIREMENT_CAPACITY: usize = 8;

fn infer(error: &str) -> BackendError {
    BackendError::Inference(format!("Oxide work retirement: {error}"))
}

/// Monotonic generation source owned by one persistent host-I/O event slot.
///
/// The completed watermark is advanced only by an unsafe caller that has
/// observed the real same-stream fence. A later generation completing also
/// proves every earlier generation on that clock, which is why completion is
/// represented as a monotonic watermark rather than a set of ids.
pub(crate) struct CompletionClock {
    issued: AtomicU64,
    completed: AtomicU64,
}

impl CompletionClock {
    pub(crate) fn new() -> Self {
        Self {
            issued: AtomicU64::new(0),
            completed: AtomicU64::new(0),
        }
    }

    #[cfg(test)]
    fn with_issued(issued: u64) -> Self {
        Self {
            issued: AtomicU64::new(issued),
            completed: AtomicU64::new(0),
        }
    }

    /// Issues one ticket for `stream_identity` without wrapping or panicking.
    pub(crate) fn ticket(
        self: &Arc<Self>,
        stream_identity: usize,
    ) -> Result<CompletionTicket, BackendError> {
        let previous = self
            .issued
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |issued| {
                issued.checked_add(1)
            })
            .map_err(|_| infer("completion generation overflow"))?;
        Ok(CompletionTicket {
            clock: Arc::clone(self),
            generation: previous + 1,
            stream_identity,
        })
    }
}

/// One deferred owner's identity. Cloning shares the clock; no GPU work is
/// created or inspected.
#[derive(Clone)]
pub(crate) struct CompletionTicket {
    clock: Arc<CompletionClock>,
    generation: u64,
    stream_identity: usize,
}

impl CompletionTicket {
    pub(crate) fn stream_identity(&self) -> usize {
        self.stream_identity
    }

    /// Returns proof only after this generation's completion watermark was
    /// published. A future generation is never completed by an older proof.
    pub(crate) fn completed(&self) -> Option<CompletedTicket> {
        if self.clock.completed.load(Ordering::Acquire) >= self.generation {
            Some(CompletedTicket {
                clock: Arc::clone(&self.clock),
                generation: self.generation,
                stream_identity: self.stream_identity,
            })
        } else {
            None
        }
    }

    /// Full request identity: same clock, generation, and stream slot.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn same_request(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.clock, &other.clock)
            && self.generation == other.generation
            && self.stream_identity == other.stream_identity
    }

    /// Publishes completion for this ticket and every older generation.
    ///
    /// # Safety
    ///
    /// Call only after the actual same-stream work that covers this ticket is
    /// fenced: the corresponding `Event::synchronize` or `Stream::synchronize`
    /// must have returned `Ok`. Never call it from elapsed-time bookkeeping, an
    /// unrelated stream, or a different host-I/O slot. A forged or premature
    /// publish can let a queue drop owners whose buffers are still in use.
    pub(crate) unsafe fn mark_completed_after_fence(&self) {
        self.clock
            .completed
            .fetch_max(self.generation, Ordering::AcqRel);
    }
}

/// Opaque proof that one ticket's actual fence completed. It is produced only
/// by [`CompletionTicket::completed`].
#[derive(Clone)]
pub(crate) struct CompletedTicket {
    clock: Arc<CompletionClock>,
    generation: u64,
    stream_identity: usize,
}

impl CompletedTicket {
    pub(crate) fn matches(&self, ticket: &CompletionTicket) -> bool {
        Arc::ptr_eq(&self.clock, &ticket.clock)
            && self.generation == ticket.generation
            && self.stream_identity == ticket.stream_identity
    }

    pub(crate) fn stream_identity(&self) -> usize {
        self.stream_identity
    }
}

struct Entry<V> {
    value: V,
    bytes: usize,
    ticket: Option<CompletionTicket>,
}

/// Fixed-capacity, heap-free deferred owner queue.
///
/// Bytes stay charged and values stay owned until either a matching completed
/// ticket is popped, the owner explicitly drains after a real fence, or Drop
/// quarantines the entry. An entry with no ticket can never be popped as
/// completed.
pub(crate) struct RetirementQueue<V> {
    entries: [Option<Entry<V>>; RETIREMENT_CAPACITY],
    bytes: usize,
    len: usize,
}

impl<V> RetirementQueue<V> {
    pub(crate) fn new() -> Self {
        Self {
            entries: [const { None }; RETIREMENT_CAPACITY],
            bytes: 0,
            len: 0,
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Adds one owner and its charged bytes. On a full queue or byte-accounting
    /// overflow the original value is returned untouched.
    pub(crate) fn try_push(
        &mut self,
        value: V,
        bytes: usize,
        ticket: Option<CompletionTicket>,
    ) -> Result<(), (BackendError, V)> {
        let total = match self.bytes.checked_add(bytes) {
            Some(total) => total,
            None => return Err((infer("retirement byte accounting overflow"), value)),
        };
        if self.len == RETIREMENT_CAPACITY {
            return Err((
                BackendError::Overloaded("work retirement queue full".into()),
                value,
            ));
        }
        let Some(index) = self.entries.iter().position(|entry| entry.is_none()) else {
            return Err((
                BackendError::Overloaded("work retirement queue full".into()),
                value,
            ));
        };
        self.entries[index] = Some(Entry {
            value,
            bytes,
            ticket,
        });
        self.bytes = total;
        self.len += 1;
        Ok(())
    }

    /// Removes one completed owner, even when it is not at the head. A ticket
    /// with no proof, or no ticket at all, is left in place.
    pub(crate) fn pop_completed(&mut self) -> Option<(V, CompletedTicket)> {
        for index in 0..RETIREMENT_CAPACITY {
            let Some(entry) = self.entries[index].as_ref() else {
                continue;
            };
            let Some(completed) = entry.ticket.as_ref().and_then(CompletionTicket::completed)
            else {
                continue;
            };
            let Some(entry) = self.entries[index].take() else {
                continue;
            };
            self.release_accounting(entry.bytes);
            return Some((entry.value, completed));
        }
        None
    }

    /// Removes one owner without interpreting completion proof.
    ///
    /// Only the owning caller may use this after it has actually drained the
    /// producing stream for this owner, or when it will immediately
    /// `mem::forget` the returned value as quarantine. Dropping an unproven
    /// value can return pinned or device-backed state to a pool too early.
    pub(crate) fn pop_any(&mut self) -> Option<(V, Option<CompletionTicket>)> {
        for index in 0..RETIREMENT_CAPACITY {
            let Some(entry) = self.entries[index].take() else {
                continue;
            };
            self.release_accounting(entry.bytes);
            return Some((entry.value, entry.ticket));
        }
        None
    }

    fn release_accounting(&mut self, bytes: usize) {
        self.bytes = self.bytes.saturating_sub(bytes);
        self.len = self.len.saturating_sub(1);
    }
}

impl<V> Drop for RetirementQueue<V> {
    fn drop(&mut self) {
        // Safe quarantine: an unproven owner must not be dropped implicitly.
        // Forget each remaining value; the CPU-only ticket metadata may drop.
        // Explicit cleanup must prove the actual fence before dropping.
        // Metrics stay with the caller's budget owner so an unknown caller
        // cannot double count this quarantine.
        for slot in &mut self.entries {
            if let Some(Entry { value, .. }) = slot.take() {
                mem::forget(value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct DropSpy {
        dropped: Arc<AtomicUsize>,
        id: usize,
    }

    impl DropSpy {
        fn new(dropped: Arc<AtomicUsize>, id: usize) -> Self {
            Self { dropped, id }
        }
    }

    impl Drop for DropSpy {
        fn drop(&mut self) {
            self.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn pending_ticket_does_not_release_the_owner_or_bytes() {
        let clock = Arc::new(CompletionClock::new());
        let ticket = clock.ticket(3).unwrap();
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut queue = RetirementQueue::new();
        assert!(queue
            .try_push(DropSpy::new(dropped.clone(), 1), 40, Some(ticket))
            .is_ok());
        assert_eq!(queue.bytes(), 40);
        assert_eq!(queue.len(), 1);
        assert!(queue.pop_completed().is_none());
        assert_eq!(queue.bytes(), 40);
        assert_eq!(queue.len(), 1);
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn old_generation_completion_does_not_complete_a_future_ticket() {
        let clock = Arc::new(CompletionClock::new());
        let old = clock.ticket(1).unwrap();
        let future = clock.ticket(1).unwrap();
        // SAFETY: this test stands in for an actual same-stream fence.
        unsafe { old.mark_completed_after_fence() };
        let proof = old.completed().expect("old generation was marked");
        assert!(proof.matches(&old));
        assert!(proof.clone().matches(&old));
        assert!(!proof.matches(&future));
        assert_eq!(proof.stream_identity(), 1);
        assert!(future.completed().is_none());
        assert!(old.same_request(&old.clone()));
        assert!(!old.same_request(&future));
    }

    #[test]
    fn slot_and_stream_identity_cannot_forge_completion_proof() {
        let first = Arc::new(CompletionClock::new());
        let second = Arc::new(CompletionClock::new());
        let first_ticket = first.ticket(7).unwrap();
        let second_ticket = second.ticket(7).unwrap();
        let other_stream = first.ticket(8).unwrap();
        // SAFETY: this test stands in for an actual same-stream fence.
        unsafe { first_ticket.mark_completed_after_fence() };
        let proof = first_ticket.completed().unwrap();
        assert!(proof.matches(&first_ticket));
        assert!(!proof.matches(&second_ticket));
        assert_eq!(proof.stream_identity(), 7);
        assert!(second_ticket.completed().is_none());
        assert!(other_stream.completed().is_none());
        assert!(!first_ticket.same_request(&second_ticket));
        assert!(!first_ticket.same_request(&other_stream));
    }

    #[test]
    fn completed_entries_retire_out_of_order_with_their_own_owners_and_bytes() {
        let a_clock = Arc::new(CompletionClock::new());
        let b_clock = Arc::new(CompletionClock::new());
        let c_clock = Arc::new(CompletionClock::new());
        let a = a_clock.ticket(1).unwrap();
        let b = b_clock.ticket(1).unwrap();
        let c = c_clock.ticket(1).unwrap();
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut queue = RetirementQueue::new();
        assert!(queue
            .try_push(DropSpy::new(dropped.clone(), 1), 10, Some(a.clone()))
            .is_ok());
        assert!(queue
            .try_push(DropSpy::new(dropped.clone(), 2), 20, Some(b.clone()))
            .is_ok());
        assert!(queue
            .try_push(DropSpy::new(dropped.clone(), 3), 30, Some(c.clone()))
            .is_ok());

        // SAFETY: this test stands in for an actual same-stream fence.
        unsafe { b.mark_completed_after_fence() };
        let (value, proof) = queue.pop_completed().expect("middle entry completed");
        assert_eq!(value.id, 2);
        assert!(proof.matches(&b));
        assert_eq!(queue.bytes(), 40);
        assert_eq!(queue.len(), 2);
        drop(value);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);

        // SAFETY: this test stands in for an actual same-stream fence.
        unsafe { a.mark_completed_after_fence() };
        let (value, _) = queue.pop_completed().expect("head entry completed");
        assert_eq!(value.id, 1);
        drop(value);

        // SAFETY: this test stands in for an actual same-stream fence.
        unsafe { c.mark_completed_after_fence() };
        let (value, _) = queue.pop_completed().expect("tail entry completed");
        assert_eq!(value.id, 3);
        drop(value);
        assert_eq!(queue.bytes(), 0);
        assert_eq!(queue.len(), 0);
        assert!(queue.pop_completed().is_none());
        assert_eq!(dropped.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn bytes_remain_charged_until_the_actual_proof_is_taken() {
        let clock = Arc::new(CompletionClock::new());
        let first = clock.ticket(2).unwrap();
        let second = clock.ticket(2).unwrap();
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut queue = RetirementQueue::new();
        assert!(queue
            .try_push(DropSpy::new(dropped.clone(), 1), 11, Some(first.clone()))
            .is_ok());
        assert!(queue
            .try_push(DropSpy::new(dropped.clone(), 2), 22, Some(second.clone()))
            .is_ok());
        // SAFETY: this test stands in for an actual same-stream fence.
        unsafe { first.mark_completed_after_fence() };
        let (value, _) = queue.pop_completed().unwrap();
        drop(value);
        assert_eq!(queue.bytes(), 22);
        assert_eq!(queue.len(), 1);
        // SAFETY: this test stands in for an actual same-stream fence.
        unsafe { second.mark_completed_after_fence() };
        let (value, _) = queue.pop_completed().unwrap();
        drop(value);
        assert_eq!(queue.bytes(), 0);
        assert_eq!(queue.len(), 0);
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn full_queue_returns_the_owned_value_for_failed_submit_rollback() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut queue = RetirementQueue::new();
        for id in 0..RETIREMENT_CAPACITY {
            assert!(queue
                .try_push(DropSpy::new(dropped.clone(), id), 1, None)
                .is_ok());
        }
        let (error, value) = queue
            .try_push(DropSpy::new(dropped.clone(), 99), 1, None)
            .unwrap_err();
        assert!(matches!(error, BackendError::Overloaded(_)));
        assert_eq!(value.id, 99);
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        drop(value);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        assert_eq!(queue.len(), RETIREMENT_CAPACITY);
        assert_eq!(queue.bytes(), RETIREMENT_CAPACITY);
    }

    #[test]
    fn byte_overflow_returns_the_owned_value() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut queue = RetirementQueue::new();
        assert!(queue
            .try_push(DropSpy::new(dropped.clone(), 1), usize::MAX, None)
            .is_ok());
        let (error, value) = queue
            .try_push(DropSpy::new(dropped.clone(), 2), 1, None)
            .unwrap_err();
        assert!(matches!(error, BackendError::Inference(_)));
        assert_eq!(value.id, 2);
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        drop(value);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        assert_eq!(queue.bytes(), usize::MAX);
    }

    #[test]
    fn an_unproven_none_ticket_is_never_released() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut queue = RetirementQueue::new();
        assert!(queue
            .try_push(DropSpy::new(dropped.clone(), 1), 7, None)
            .is_ok());
        for _ in 0..4 {
            assert!(queue.pop_completed().is_none());
        }
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.bytes(), 7);
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn queue_drop_quarantines_unproven_owners_instead_of_dropping_them() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let clock = Arc::new(CompletionClock::new());
        {
            let mut queue = RetirementQueue::new();
            assert!(queue
                .try_push(
                    DropSpy::new(dropped.clone(), 1),
                    11,
                    Some(clock.ticket(1).unwrap())
                )
                .is_ok());
            assert!(queue
                .try_push(DropSpy::new(dropped.clone(), 2), 22, None)
                .is_ok());
            assert_eq!(queue.bytes(), 33);
        }
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn pop_any_returns_the_owner_and_its_ticket_for_explicit_cleanup() {
        let clock = Arc::new(CompletionClock::new());
        let ticket = clock.ticket(5).unwrap();
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut queue = RetirementQueue::new();
        assert!(queue
            .try_push(DropSpy::new(dropped.clone(), 1), 9, Some(ticket.clone()))
            .is_ok());
        let (value, returned) = queue.pop_any().unwrap();
        assert_eq!(queue.bytes(), 0);
        assert_eq!(queue.len(), 0);
        assert!(returned.as_ref().unwrap().same_request(&ticket));
        // SAFETY: this test stands in for an actual same-stream fence.
        unsafe { ticket.mark_completed_after_fence() };
        assert!(returned.unwrap().completed().is_some());
        drop(value);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn generation_overflow_is_a_typed_error_without_wrapping_or_panicking() {
        let exhausted = Arc::new(CompletionClock::with_issued(u64::MAX));
        assert!(matches!(
            exhausted.ticket(1),
            Err(BackendError::Inference(_))
        ));

        let last = Arc::new(CompletionClock::with_issued(u64::MAX - 1));
        let ticket = last.ticket(4).unwrap();
        assert!(last.ticket(4).is_err());
        // SAFETY: this test stands in for an actual same-stream fence.
        unsafe { ticket.mark_completed_after_fence() };
        assert!(ticket.completed().is_some());
    }
}
