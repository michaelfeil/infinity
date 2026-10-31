//! Shared queued-compute admission budget. Running work is excluded.
use crate::{queue_actor::ServiceTier, TextEmbeddingsError};
use std::sync::{
    atomic::{AtomicI64, Ordering},
    Arc,
};

/// Cumulative ceilings in tokens, ordered flex <= default <= priority.
#[derive(Debug, Clone, Copy)]
pub struct QueueTokenLimits {
    pub flex: i64,
    pub default: i64,
    pub priority: i64,
}
impl QueueTokenLimits {
    pub fn from_thousands(value: &str) -> Result<Self, String> {
        let values = value
            .split(',')
            .map(|v| {
                v.trim()
                    .parse::<i64>()
                    .ok()
                    .filter(|n| *n >= 0)
                    .and_then(|n| n.checked_mul(1000))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(
                "Queue token limits must be nonnegative integers in thousands without overflow",
            )?;
        match values.as_slice() {
            [flex, default, priority] if flex <= default && default <= priority => Ok(Self {
                flex: *flex,
                default: *default,
                priority: *priority,
            }),
            _ => Err("Expected flex,default,priority queue limits in ascending order".into()),
        }
    }
    fn for_tier(self, tier: ServiceTier) -> i64 {
        match tier {
            ServiceTier::Flex => self.flex,
            ServiceTier::Default => self.default,
            ServiceTier::Priority => self.priority,
        }
    }
}

/// Every admitted charge is owned by a reservation until dispatch or cancellation.
#[derive(Debug, Default)]
pub struct QueueBudget {
    model: String,
    queued: AtomicI64,
}
impl QueueBudget {
    pub(crate) fn new(model: String) -> Self {
        Self {
            model,
            ..Self::default()
        }
    }
    pub fn queued_tokens(&self) -> i64 {
        self.queued.load(Ordering::Relaxed)
    }
    pub(crate) fn reserve(
        self: &Arc<Self>,
        tokens: i64,
        tier: ServiceTier,
        limits: Option<QueueTokenLimits>,
    ) -> Result<QueueLease, TextEmbeddingsError> {
        if tokens < 0 {
            return Err(TextEmbeddingsError::Validation(
                "Queue token charge must be nonnegative".into(),
            ));
        }
        let limit = limits.map_or(i64::MAX, |limits| limits.for_tier(tier));
        // One atomic operation checks the tier ceiling and reserves capacity.
        // Relaxed suffices: these atomics track numbers, not payload publication.
        self.queued
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |queued| {
                queued.checked_add(tokens).filter(|next| *next <= limit)
            })
            .map_err(|queued| {
                // Use the failed admission's snapshot, including this charge.
                // Only expose tier names, never limits or queue/token counts.
                let higher: &[ServiceTier] = match tier {
                    ServiceTier::Flex => &[ServiceTier::Default, ServiceTier::Priority],
                    ServiceTier::Default => &[ServiceTier::Priority],
                    ServiceTier::Priority => &[],
                };
                let available_tiers = higher
                    .iter()
                    .copied()
                    .filter(|candidate| {
                        queued.checked_add(tokens).is_some_and(|total| {
                            total <= limits.map_or(i64::MAX, |limits| limits.for_tier(*candidate))
                        })
                    })
                    .collect();
                TextEmbeddingsError::QueueFull {
                    tier: tier.as_str(),
                    available_tiers,
                }
            })?;
        metrics::gauge!("infinity_queue_compute_tokens", "model" => self.model.clone())
            .increment(tokens as f64);
        // Publish ownership only after charging. Every successfully constructed
        // lease must drop, including on failed enqueue and task cancellation.
        Ok(QueueLease(Arc::new(Reservation {
            budget: self.clone(),
            tokens: AtomicI64::new(tokens),
        })))
    }
}

#[derive(Debug)]
struct Reservation {
    budget: Arc<QueueBudget>,
    tokens: AtomicI64,
}
impl Reservation {
    fn release(&self) {
        // Caller cancellation and queue dispatch share ownership. Exactly one
        // can take the charge; all later releases take zero. Never clamp or skip
        // a subtraction: each successful admission has one matching return.
        let tokens = self.tokens.swap(0, Ordering::Relaxed);
        if tokens != 0 {
            self.budget.queued.fetch_sub(tokens, Ordering::Relaxed);
            // Balanced deltas commute, unlike concurrent gauge.set snapshots.
            metrics::gauge!("infinity_queue_compute_tokens", "model" => self.budget.model.clone())
                .decrement(tokens as f64);
        }
    }
}
/// Retain while awaiting inference; dropping it cancels the queue reservation.
/// The caller and queue own separate guards over the same charge. Either caller
/// cancellation, queue discard, or dispatch returns it exactly once.
#[must_use = "Hold the lease until inference completes; dropping it releases queue capacity"]
#[derive(Debug)]
pub struct QueueLease(Arc<Reservation>);
impl QueueLease {
    pub(crate) fn queue_guard(&self) -> Arc<Self> {
        Arc::new(Self(self.0.clone()))
    }
    pub(crate) fn release(&self) {
        self.0.release();
    }
}
impl Drop for QueueLease {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejection_hints_include_the_incoming_charge_and_only_higher_tiers() {
        use ServiceTier::{Default, Flex, Priority};
        let budget = Arc::new(QueueBudget::default());
        let limits = Some(QueueTokenLimits::from_thousands("5,10,12").unwrap());
        let _live = budget.reserve(9000, Priority, limits).unwrap();
        for (tier, tokens, expected) in [
            (Flex, 1000, vec![Default, Priority]),
            (Flex, 2000, vec![Priority]),
            (Default, 3000, vec![Priority]),
            (Default, 3001, vec![]),
            (Priority, 3001, vec![]),
        ] {
            let err = budget.reserve(tokens, tier, limits).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!("Queue is full for the {} tier", tier.as_str())
            );
            let TextEmbeddingsError::QueueFull {
                available_tiers, ..
            } = err
            else {
                panic!("expected queue rejection")
            };
            assert_eq!(available_tiers, expected);
            assert_eq!(budget.queued_tokens(), 9000);
        }
    }

    #[test]
    fn tiers_share_one_total_and_release_once() {
        let budget = Arc::new(QueueBudget::default());
        let limits = Some(QueueTokenLimits::from_thousands("5,10,12").unwrap());
        let flex = budget.reserve(5000, ServiceTier::Flex, limits).unwrap();
        assert!(budget.reserve(1, ServiceTier::Flex, limits).is_err());
        let standard = budget.reserve(5000, ServiceTier::Default, limits).unwrap();
        assert!(budget.reserve(1, ServiceTier::Default, limits).is_err());
        let priority = budget.reserve(2000, ServiceTier::Priority, limits).unwrap();
        assert!(budget.reserve(1, ServiceTier::Priority, limits).is_err());
        flex.release();
        flex.release();
        drop(flex);
        assert_eq!(budget.queued_tokens(), 7000);
        drop(standard);
        drop(priority);
        assert_eq!(budget.queued_tokens(), 0);
        assert!(QueueTokenLimits::from_thousands("12,10,5").is_err());
        assert!(QueueTokenLimits::from_thousands("18446744073709551615,1,2").is_err());
    }
    #[test]
    fn racing_dispatch_and_cancellation_return_only_their_owned_charge() {
        let budget = Arc::new(QueueBudget::default());
        let live = budget.reserve(10, ServiceTier::Default, None).unwrap();
        for _ in 0..100 {
            let caller = budget.reserve(11, ServiceTier::Default, None).unwrap();
            let queued = caller.queue_guard();
            let barrier = Arc::new(std::sync::Barrier::new(3));
            std::thread::scope(|scope| {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    drop(caller);
                });
                scope.spawn(move || {
                    barrier.wait();
                    queued.release();
                    queued.release();
                    drop(queued);
                });
                barrier.wait();
            });
            assert_eq!(budget.queued_tokens(), 10);
        }
        drop(live);
        assert_eq!(budget.queued_tokens(), 0);
        assert!(budget.reserve(-1, ServiceTier::Default, None).is_err());
        assert!(QueueTokenLimits::from_thousands("-1,1,2").is_err());
    }

    #[test]
    fn dropping_one_group_member_keeps_the_reservation_until_the_last_member() {
        let budget = Arc::new(QueueBudget::default());
        let live = budget.reserve(10, ServiceTier::Default, None).unwrap();
        let caller = budget.reserve(20, ServiceTier::Default, None).unwrap();
        let first = caller.queue_guard();
        let second = first.clone();
        drop(first);
        assert_eq!(budget.queued_tokens(), 30);
        // A discarded group returns capacity even if its caller has not yet
        // polled the failed response. Later caller cancellation is a no-op.
        drop(second);
        assert_eq!(budget.queued_tokens(), 10);
        drop(caller);
        assert_eq!(budget.queued_tokens(), 10);
        drop(live);
        assert_eq!(budget.queued_tokens(), 0);
    }

    #[test]
    fn rejected_overflow_preserves_live_charges_and_capacity_recovers() {
        let budget = Arc::new(QueueBudget::default());
        let large = budget
            .reserve(i64::MAX - 10, ServiceTier::Default, None)
            .unwrap();
        let small = budget.reserve(10, ServiceTier::Default, None).unwrap();
        assert!(budget.reserve(1, ServiceTier::Default, None).is_err());
        assert_eq!(budget.queued_tokens(), i64::MAX);
        drop(large);
        assert_eq!(budget.queued_tokens(), 10);
        let next = budget.reserve(100, ServiceTier::Default, None).unwrap();
        drop(small);
        assert_eq!(budget.queued_tokens(), 100);
        drop(next);
        assert_eq!(budget.queued_tokens(), 0);
    }

    #[test]
    fn concurrent_admission_and_release_leave_no_drift() {
        let budget = Arc::new(QueueBudget::default());
        let live = budget.reserve(777, ServiceTier::Default, None).unwrap();
        let limits = Some(QueueTokenLimits {
            flex: 2000,
            default: 3000,
            priority: 4000,
        });
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let budget = &budget;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    for iteration in 0..2000 {
                        let tier = [
                            ServiceTier::Flex,
                            ServiceTier::Default,
                            ServiceTier::Priority,
                        ][(worker + iteration) % 3];
                        if let Ok(caller) = budget.reserve(100 + worker as i64, tier, limits) {
                            let queued = caller.queue_guard();
                            if iteration % 2 == 0 {
                                queued.release();
                            }
                            drop(caller);
                            drop(queued);
                        }
                    }
                });
            }
        });
        assert_eq!(budget.queued_tokens(), 777);
        drop(live);
        assert_eq!(budget.queued_tokens(), 0);
    }

    #[test]
    fn racing_admissions_cannot_exceed_capacity() {
        let budget = Arc::new(QueueBudget::default());
        let barrier = Arc::new(std::sync::Barrier::new(32));
        let workers: Vec<_> = (0..32)
            .map(|_| {
                let budget = budget.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    budget.reserve(
                        1000,
                        ServiceTier::Priority,
                        Some(QueueTokenLimits {
                            flex: 0,
                            default: 0,
                            priority: 12000,
                        }),
                    )
                })
            })
            .collect();
        let leases: Vec<_> = workers
            .into_iter()
            .filter_map(|t| t.join().unwrap().ok())
            .collect();
        assert_eq!(leases.len(), 12);
        assert_eq!(budget.queued_tokens(), 12000);
        drop(leases);
        assert_eq!(budget.queued_tokens(), 0);
    }
}
