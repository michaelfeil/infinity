// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Owned pinned input/output leases for ordered-stream deferred submissions.
use super::{cache, capacity_fits, infer, product, Engine, WorkKey};
use cuda_core::{Context, Event, PinnedHostBuffer, Stream};
use infinity_v3_backend_core::BackendError;
use std::{
    mem,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    },
};

// Free slots retain ordinary I/O buffers across shape changes. A slot larger
// than this is kept only for its exact capacity, so one exceptional request
// cannot pin host memory indefinitely after traffic returns to normal sizes.
const PIN_RETAIN_BYTES: usize = cache::BYTE_BUDGET / 4;

pub(crate) fn input_capacity(key: WorkKey) -> Result<usize, BackendError> {
    product(key.0, 3)?
        .checked_add(key.1)
        .and_then(|n| n.checked_add(key.2))
        .ok_or_else(|| infer("Input staging layout overflow"))
}
pub(crate) fn pinned_bytes(key: WorkKey, output_elements: usize) -> Result<usize, BackendError> {
    product(input_capacity(key)?, 4)?
        .checked_add(product(output_elements, 4)?)
        .ok_or_else(|| infer("Host I/O staging overflow"))
}
fn slot_fits(capacity: WorkKey, request: WorkKey, bytes: usize) -> bool {
    capacity_fits(capacity, request, bytes) && (bytes <= PIN_RETAIN_BYTES || capacity == request)
}
fn host_slot_fits(
    capacity: WorkKey,
    capacity_output: usize,
    request: WorkKey,
    request_output: usize,
    bytes: usize,
) -> bool {
    slot_fits(capacity, request, bytes) && request_output <= capacity_output
}

pub(crate) struct HostIo {
    key: WorkKey,
    output_elements: usize,
    pub(crate) input: PinnedHostBuffer<u32>,
    pub(crate) output: PinnedHostBuffer<f32>,
    pub(crate) event: Event,
    // Declared last so physical buffers are freed before their accounting.
    resident: HostIoBytes,
}

struct HostIoBytes(usize);
impl HostIoBytes {
    fn replace(&mut self, bytes: usize) {
        super::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        super::LIVE_ARENA_BYTES.fetch_sub(self.0, Ordering::Relaxed);
        self.0 = bytes;
        super::record_live_arena_bytes();
    }
}
impl Drop for HostIoBytes {
    fn drop(&mut self) {
        super::LIVE_ARENA_BYTES.fetch_sub(self.0, Ordering::Relaxed);
        super::record_live_arena_bytes();
    }
}
impl HostIo {
    fn empty(context: &Arc<Context>) -> Result<Self, BackendError> {
        Ok(Self {
            key: (0, 0, 0),
            output_elements: 0,
            input: PinnedHostBuffer::zeroed(context, 0).map_err(infer)?,
            output: PinnedHostBuffer::zeroed(context, 0).map_err(infer)?,
            event: context.new_event_with_flags(None).map_err(infer)?,
            resident: HostIoBytes(0),
        })
    }
    pub(crate) fn pool(context: &Arc<Context>) -> Result<Arc<SlotPool<Self>>, BackendError> {
        Ok(SlotPool::new([
            Self::empty(context)?,
            Self::empty(context)?,
        ]))
    }
    fn bytes(&self) -> usize {
        self.input
            .num_bytes()
            .saturating_add(self.output.num_bytes())
    }
    fn fits(&self, key: WorkKey, output_elements: usize) -> bool {
        host_slot_fits(
            self.key,
            self.output_elements,
            key,
            output_elements,
            self.bytes(),
        )
    }
    fn allocate(
        context: &Arc<Context>,
        key: WorkKey,
        output_elements: usize,
    ) -> Result<(PinnedHostBuffer<u32>, PinnedHostBuffer<f32>), BackendError> {
        Ok((
            PinnedHostBuffer::zeroed(context, input_capacity(key)?).map_err(infer)?,
            PinnedHostBuffer::zeroed(context, output_elements).map_err(infer)?,
        ))
    }
    fn replace(
        &mut self,
        key: WorkKey,
        output_elements: usize,
        input: PinnedHostBuffer<u32>,
        output: PinnedHostBuffer<f32>,
    ) {
        self.key = key;
        self.output_elements = output_elements;
        self.input = input;
        self.output = output;
        self.resident.replace(self.bytes());
    }
}
impl SlotBytes for HostIo {
    fn slot_bytes(&self) -> usize {
        self.bytes()
    }
    fn shrink(&mut self) {
        if self.bytes() == 0 {
            return;
        }
        if let Ok((input, output)) = HostIo::allocate(self.input.context(), (0, 0, 0), 0) {
            self.replace((0, 0, 0), 0, input, output);
        }
    }
}

struct PoolState<S> {
    free: Vec<S>,
    closed: bool,
    leases: usize,
    exclusive: bool,
}
pub(crate) struct SlotPool<S> {
    state: Mutex<PoolState<S>>,
    ready: Condvar,
    poison: Arc<AtomicBool>,
    pinned: AtomicUsize,
    budget: usize,
}
impl<S> SlotPool<S> {
    pub(crate) fn new(slots: impl IntoIterator<Item = S>) -> Arc<Self>
    where
        S: SlotBytes,
    {
        Self::with_budget(slots, cache::BYTE_BUDGET)
    }
    pub(crate) fn with_budget(slots: impl IntoIterator<Item = S>, budget: usize) -> Arc<Self>
    where
        S: SlotBytes,
    {
        let slots: Vec<S> = slots.into_iter().collect();
        let pinned = slots
            .iter()
            .map(SlotBytes::slot_bytes)
            .fold(0usize, usize::saturating_add);
        Arc::new(Self {
            state: Mutex::new(PoolState {
                free: slots,
                closed: false,
                leases: 0,
                exclusive: false,
            }),
            ready: Condvar::new(),
            poison: Arc::new(AtomicBool::new(false)),
            pinned: AtomicUsize::new(pinned),
            budget,
        })
    }
    pub(crate) fn acquire(&self) -> Result<S, BackendError> {
        let mut state = lock(&self.state);
        loop {
            if state.closed || self.poison.load(Ordering::Acquire) {
                return Err(BackendError::Unhealthy);
            }
            if !state.exclusive {
                if let Some(slot) = state.free.pop() {
                    state.leases += 1;
                    return Ok(slot);
                }
            }
            state = self
                .ready
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }
    pub(crate) fn release(&self, slot: S)
    where
        S: SlotBytes,
    {
        let mut state = lock(&self.state);
        if state.closed || self.poison.load(Ordering::Acquire) {
            self.retire(slot);
        } else {
            state.free.push(slot);
        }
        state.leases = state.leases.saturating_sub(1);
        if state.leases == 0 {
            state.exclusive = false;
        }
        drop(state);
        self.ready.notify_all();
    }
    pub(crate) fn close(&self)
    where
        S: SlotBytes,
    {
        self.poison.store(true, Ordering::Release);
        let mut state = lock(&self.state);
        state.closed = true;
        let free = mem::take(&mut state.free);
        drop(state);
        for slot in free {
            self.retire(slot);
        }
        self.ready.notify_all();
    }
    pub(crate) fn poisoned(&self) -> bool {
        self.poison.load(Ordering::Acquire)
    }
    pub(crate) fn pinned_bytes(&self) -> usize {
        self.pinned.load(Ordering::Acquire)
    }
    pub(crate) fn pinned_after(&self, old: usize, new: usize) -> Result<usize, BackendError> {
        self.pinned_bytes()
            .checked_sub(old)
            .and_then(|bytes| bytes.checked_add(new))
            .ok_or_else(|| infer("Host I/O staging overflow"))
    }
    pub(crate) fn grow(&self, old: usize, new: usize, exclusive: bool) -> Result<(), BackendError> {
        let mut state = lock(&self.state);
        if state.closed || self.poison.load(Ordering::Acquire) {
            return Err(BackendError::Unhealthy);
        }
        let total = self.pinned_after(old, new)?;
        if total > self.budget && !exclusive {
            return Err(BackendError::Overloaded(
                "Host I/O staging budget exhausted".into(),
            ));
        }
        if exclusive && state.leases != 1 {
            return Err(BackendError::Overloaded(
                "Host I/O staging requires an exclusive lease".into(),
            ));
        }
        self.adjust_pinned(old, new)?;
        state.exclusive |= exclusive;
        Ok(())
    }
    pub(crate) fn active_leases(&self) -> usize {
        lock(&self.state).leases
    }
    pub(crate) fn shrink_free(&self) -> usize
    where
        S: SlotBytes,
    {
        let mut state = lock(&self.state);
        if state.closed || self.poison.load(Ordering::Acquire) {
            return 0;
        }
        let mut reclaimed = 0usize;
        for slot in &mut state.free {
            let old = slot.slot_bytes();
            if old == 0 {
                continue;
            }
            slot.shrink();
            let new = slot.slot_bytes();
            if new < old {
                reclaimed = reclaimed.saturating_add(old - new);
            }
        }
        drop(state);
        if reclaimed > 0 {
            let _ = self
                .pinned
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |total| {
                    Some(total.saturating_sub(reclaimed))
                });
        }
        reclaimed
    }
    fn adjust_pinned(&self, old: usize, new: usize) -> Result<(), BackendError> {
        self.pinned
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |bytes| {
                bytes.checked_sub(old)?.checked_add(new)
            })
            .map_err(|_| infer("Host I/O staging overflow"))?;
        Ok(())
    }
    fn retire(&self, slot: S)
    where
        S: SlotBytes,
    {
        let bytes = slot.slot_bytes();
        drop(slot);
        let _ = self
            .pinned
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |total| {
                Some(total.saturating_sub(bytes))
            });
    }
}
fn lock<T>(value: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    value
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(crate) trait SlotBytes {
    fn slot_bytes(&self) -> usize;
    fn shrink(&mut self) {}
}

pub(crate) struct HostIoLease {
    slot: Option<HostIo>,
    pool: Arc<SlotPool<HostIo>>,
    _engine: Arc<Mutex<Engine>>,
    stream: Arc<Stream>,
    started: bool,
    recorded: bool,
    drained: bool,
    exclusive: bool,
}
impl HostIoLease {
    pub(crate) fn new(
        slot: HostIo,
        pool: Arc<SlotPool<HostIo>>,
        engine: Arc<Mutex<Engine>>,
        stream: Arc<Stream>,
    ) -> Self {
        Self {
            slot: Some(slot),
            pool,
            _engine: engine,
            stream,
            started: false,
            recorded: false,
            drained: false,
            exclusive: false,
        }
    }
    pub(crate) fn bytes(&self) -> Result<usize, BackendError> {
        Ok(self.slot()?.bytes())
    }
    pub(crate) fn required_bytes(
        &self,
        key: WorkKey,
        output_elements: usize,
    ) -> Result<Option<usize>, BackendError> {
        if self.slot()?.fits(key, output_elements) {
            Ok(None)
        } else {
            pinned_bytes(key, output_elements).map(Some)
        }
    }
    pub(crate) fn mark_exclusive(&mut self) {
        self.exclusive = true;
    }
    pub(crate) fn discard_for_resize(
        &mut self,
        key: WorkKey,
        output_elements: usize,
    ) -> Result<(), BackendError> {
        let slot = self
            .slot
            .as_mut()
            .ok_or_else(|| infer("Missing host I/O lease"))?;
        if !slot.fits(key, output_elements) && slot.bytes() > 0 {
            let old = slot.bytes();
            slot.shrink();
            self.pool.adjust_pinned(old, slot.bytes())?;
        }
        Ok(())
    }
    pub(crate) fn prepare(
        &mut self,
        key: WorkKey,
        output_elements: usize,
        exclusive: bool,
    ) -> Result<(), BackendError> {
        self.discard_for_resize(key, output_elements)?;
        let Self {
            slot, pool, stream, ..
        } = self;
        let slot = slot
            .as_mut()
            .ok_or_else(|| infer("Missing host I/O lease"))?;
        if slot.fits(key, output_elements) {
            return Ok(());
        }
        let base = slot.bytes();
        let new = pinned_bytes(key, output_elements)?;
        pool.grow(base, new, exclusive)?;
        let (input, output) = match HostIo::allocate(stream.context(), key, output_elements) {
            Ok(buffers) => buffers,
            Err(error) => {
                let _ = pool.adjust_pinned(new, base);
                return Err(error);
            }
        };
        slot.replace(key, output_elements, input, output);
        Ok(())
    }
    pub(crate) fn slot(&self) -> Result<&HostIo, BackendError> {
        self.slot
            .as_ref()
            .ok_or_else(|| infer("Missing host I/O lease"))
    }
    pub(crate) fn slot_mut(&mut self) -> Result<&mut HostIo, BackendError> {
        self.slot
            .as_mut()
            .ok_or_else(|| infer("Missing host I/O lease"))
    }
    pub(crate) fn begin(&mut self) {
        self.started = true;
    }
    pub(crate) fn record(&mut self) -> Result<(), BackendError> {
        self.started = true;
        let slot = self
            .slot
            .as_ref()
            .ok_or_else(|| infer("Missing host I/O lease"))?;
        slot.event.record(&self.stream).map_err(infer)?;
        self.recorded = true;
        Ok(())
    }
    pub(crate) fn wait(&mut self) -> Result<(), BackendError> {
        if self.drained {
            return Ok(());
        }
        let complete = match self.slot.as_ref() {
            Some(slot) if self.recorded => slot.event.synchronize().map_err(infer),
            Some(_) if self.started => self.stream.synchronize().map_err(infer),
            Some(_) => Ok(()),
            None => Err(infer("Missing host I/O lease")),
        };
        match complete {
            Ok(()) => {
                self.drained = true;
                Ok(())
            }
            Err(_) => {
                self.retain_uncertain();
                Err(BackendError::Unhealthy)
            }
        }
    }
    pub(crate) fn output(&self, elements: usize) -> Result<&[f32], BackendError> {
        if !self.drained {
            return Err(infer("Host I/O output has not completed"));
        }
        self.slot()?
            .output
            .as_slice()
            .get(..elements)
            .ok_or_else(|| infer("Output exceeds host I/O arena"))
    }
    pub(crate) fn poison(&self) {
        self.pool.close();
    }
    fn retain_uncertain(&mut self) {
        self.pool.close();
        if let Some(slot) = self.slot.take() {
            metrics::counter!("oxide_arena_retained_bytes_total").increment(slot.bytes() as u64);
            std::mem::forget(slot);
        }
    }
    fn shrink_idle(&mut self) {
        let Self {
            slot, pool, stream, ..
        } = self;
        if pool.poisoned() {
            return;
        }
        let Some(slot) = slot.as_mut() else {
            return;
        };
        if slot.bytes() == 0 || (!self.exclusive && slot.bytes() <= PIN_RETAIN_BYTES) {
            return;
        }
        let old = slot.bytes();
        let Ok((input, output)) = HostIo::allocate(stream.context(), (0, 0, 0), 0) else {
            return;
        };
        slot.replace((0, 0, 0), 0, input, output);
        // Keep accounting conservative until the old physical pins are freed;
        // a submission can project its reservation concurrently with completion.
        let _ = pool.adjust_pinned(old, 0);
    }
}
impl Drop for HostIoLease {
    fn drop(&mut self) {
        if self.slot.is_none() {
            return;
        }
        if !self.drained {
            let complete = match self.slot.as_ref() {
                Some(slot) if self.recorded => slot.event.synchronize().is_ok(),
                Some(_) if self.started => self.stream.synchronize().is_ok(),
                Some(_) => true,
                None => false,
            };
            if !complete {
                self.retain_uncertain();
                return;
            }
            self.drained = true;
        }
        self.shrink_idle();
        if let Some(slot) = self.slot.take() {
            self.pool.release(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, thread, time::Duration};

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct Slot(u32);
    impl SlotBytes for Slot {
        fn slot_bytes(&self) -> usize {
            0
        }
    }

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct SizedSlot(usize);
    impl SlotBytes for SizedSlot {
        fn slot_bytes(&self) -> usize {
            self.0
        }
        fn shrink(&mut self) {
            self.0 = 0;
        }
    }

    fn wait_for<T>(receiver: &mpsc::Receiver<T>) -> T {
        receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("test thread did not respond")
    }

    #[test]
    fn pool_hands_out_exactly_two_leases_and_reuses_a_returned_one() {
        let pool = SlotPool::new([Slot(1), Slot(2)]);
        let first = pool.acquire().unwrap();
        let second = pool.acquire().unwrap();
        assert_eq!(pool.active_leases(), 2);
        let (started, waiting) = mpsc::channel();
        let (result, receiver) = mpsc::channel();
        let waiter = {
            let pool = pool.clone();
            thread::spawn(move || {
                started.send(()).unwrap();
                result.send(pool.acquire()).unwrap();
            })
        };
        wait_for(&waiting);
        assert!(receiver.recv_timeout(Duration::from_millis(20)).is_err());
        pool.release(first);
        let third = wait_for(&receiver).unwrap();
        assert!(second != third);
        waiter.join().unwrap();
        pool.release(second);
        pool.release(third);
    }

    #[test]
    fn canceled_lease_drop_returns_its_slot_and_wakes_a_waiter() {
        struct Lease {
            pool: Arc<SlotPool<Slot>>,
            slot: Option<Slot>,
        }
        impl Drop for Lease {
            fn drop(&mut self) {
                if let Some(slot) = self.slot.take() {
                    self.pool.release(slot);
                }
            }
        }

        let pool = SlotPool::new([Slot(1)]);
        let lease = Lease {
            pool: pool.clone(),
            slot: Some(pool.acquire().unwrap()),
        };
        let (started, waiting) = mpsc::channel();
        let (result, receiver) = mpsc::channel();
        let waiter = {
            let pool = pool.clone();
            thread::spawn(move || {
                started.send(()).unwrap();
                result.send(pool.acquire()).unwrap();
            })
        };
        wait_for(&waiting);
        assert!(receiver.recv_timeout(Duration::from_millis(20)).is_err());
        drop(lease);
        assert_eq!(wait_for(&receiver).unwrap(), Slot(1));
        waiter.join().unwrap();
    }

    #[test]
    fn close_and_poison_wake_blocked_acquirers() {
        let pool = SlotPool::new([Slot(1), Slot(2)]);
        let first = pool.acquire().unwrap();
        let second = pool.acquire().unwrap();
        let (started, waiting) = mpsc::channel();
        let (result, receiver) = mpsc::channel();
        let waiter = {
            let pool = pool.clone();
            thread::spawn(move || {
                started.send(()).unwrap();
                result.send(pool.acquire()).unwrap();
            })
        };
        wait_for(&waiting);
        thread::sleep(Duration::from_millis(20));
        pool.close();
        let error = wait_for(&receiver).unwrap_err();
        assert!(matches!(error, BackendError::Unhealthy));
        assert!(pool.poisoned());
        waiter.join().unwrap();
        pool.release(first);
        pool.release(second);
    }

    #[test]
    fn closed_pool_drops_free_and_returned_leases_without_reuse() {
        struct DropSpy(Arc<AtomicUsize>);
        impl SlotBytes for DropSpy {
            fn slot_bytes(&self) -> usize {
                0
            }
        }
        impl Drop for DropSpy {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let dropped = Arc::new(AtomicUsize::new(0));
        let pool = SlotPool::new([DropSpy(dropped.clone()), DropSpy(dropped.clone())]);
        let first = pool.acquire().unwrap();
        pool.close();
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        pool.release(first);
        assert_eq!(dropped.load(Ordering::Relaxed), 2);
        assert!(matches!(pool.acquire(), Err(BackendError::Unhealthy)));
    }

    #[test]
    fn output_capacity_is_independent_of_the_backbone_key() {
        let key = (32, 4, 16);
        let exact = pinned_bytes(key, 45).unwrap();
        let max = pinned_bytes(key, 16 * 7).unwrap();
        let input = input_capacity(key).unwrap();
        assert_eq!(exact, input * 4 + 45 * 4);
        assert!(exact < max);
        assert!(host_slot_fits(key, 45, key, 45, exact));
        assert!(!host_slot_fits(key, 45, key, 46, exact));
        assert!(!host_slot_fits(key, 45, key, 16 * 7, exact));
        assert!(host_slot_fits((32, 4, 16), 77, (32, 4, 8), 45, 1024));
        assert!(!host_slot_fits((32, 4, 8), 45, (32, 4, 16), 45, 77));
    }

    #[test]
    fn modest_slots_are_reused_but_oversized_slots_follow_their_exact_shape() {
        let capacity = (32, 4, 2);
        let smaller = (32, 2, 1);
        assert!(slot_fits(capacity, smaller, 1024));
        let oversized = PIN_RETAIN_BYTES + 1;
        assert!(slot_fits(capacity, capacity, oversized));
        assert!(!slot_fits(capacity, smaller, oversized));
        assert!(!slot_fits((16, 2, 1), smaller, 1024));
    }

    #[test]
    fn normal_growth_is_bounded_but_exclusive_growth_can_exceed_the_budget() {
        let pool = SlotPool::with_budget([SizedSlot(0), SizedSlot(0)], 100);
        let first = pool.acquire().unwrap();
        let second = pool.acquire().unwrap();
        assert_eq!(pool.active_leases(), 2);
        pool.grow(0, 60, false).unwrap();
        assert_eq!(pool.pinned_bytes(), 60);
        assert!(matches!(
            pool.grow(0, 60, false),
            Err(BackendError::Overloaded(_))
        ));
        assert_eq!(pool.pinned_bytes(), 60);
        assert!(matches!(
            pool.grow(0, 60, true),
            Err(BackendError::Overloaded(_))
        ));
        pool.release(second);
        pool.grow(0, 60, true).unwrap();
        assert_eq!(pool.pinned_bytes(), 120);
        pool.release(first);
    }

    #[test]
    fn exclusive_growth_blocks_checkout_until_the_lease_returns() {
        let pool = SlotPool::with_budget([SizedSlot(0), SizedSlot(0)], 100);
        let mut first = pool.acquire().unwrap();
        pool.grow(0, 101, true).unwrap();
        first.0 = 101;
        let (result, receiver) = mpsc::channel();
        let waiter = {
            let pool = pool.clone();
            thread::spawn(move || result.send(pool.acquire()).unwrap())
        };
        assert!(receiver.recv_timeout(Duration::from_millis(20)).is_err());
        assert_eq!(pool.active_leases(), 1);
        // HostIoLease releases exceptional pins before publishing the slot.
        first.shrink();
        pool.adjust_pinned(101, 0).unwrap();
        pool.release(first);
        let acquired = wait_for(&receiver).unwrap();
        waiter.join().unwrap();
        pool.release(acquired);
        assert_eq!(pool.active_leases(), 0);
    }

    #[test]
    fn idle_free_slots_are_shrunk_before_growth_is_charged() {
        let pool = SlotPool::with_budget([SizedSlot(80), SizedSlot(80)], 100);
        assert_eq!(pool.pinned_bytes(), 160);
        assert_eq!(pool.shrink_free(), 160);
        assert_eq!(pool.pinned_bytes(), 0);
        let first = pool.acquire().unwrap();
        let second = pool.acquire().unwrap();
        assert_eq!(first.0, 0);
        assert_eq!(second.0, 0);
        pool.release(first);
        pool.release(second);
    }

    #[test]
    fn pinned_accounting_charges_only_the_active_resize() {
        let pool: Arc<SlotPool<SizedSlot>> = SlotPool::new(std::iter::empty());
        pool.adjust_pinned(0, 100).unwrap();
        assert_eq!(pool.pinned_bytes(), 100);
        pool.adjust_pinned(100, 40).unwrap();
        assert_eq!(pool.pinned_bytes(), 40);
        assert!(pool.adjust_pinned(100, 0).is_err());
        assert_eq!(pool.pinned_bytes(), 40);
    }
}
