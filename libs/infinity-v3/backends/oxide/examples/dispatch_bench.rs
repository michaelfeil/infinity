// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Host-only timing after one CUDA allocation setup; no GPU work in timed loops.
use cuda_core::{Context, DeviceBuffer};
use infinity_backend_oxide::{
    math::{GemmArguments, LinearPlan},
    tensor::Bf16,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    error::Error,
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

struct CountingAllocator;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn main() -> Result<(), Box<dyn Error>> {
    let device = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "0".into())
        .parse()?;
    let context = Context::new(device)?;
    let stream = context.new_stream()?;
    let (rows, input, width) = (128usize, 384usize, 1152usize);
    let x = DeviceBuffer::<Bf16>::zeroed(&stream, rows * input)?;
    let w = DeviceBuffer::<Bf16>::zeroed(&stream, width * input)?;
    let mut y = DeviceBuffer::<Bf16>::zeroed(&stream, rows * width)?;
    let plan = LinearPlan::<Bf16>::contiguous(rows, input, width)?;
    let count = 1_000_000usize;
    let mut samples = vec![];
    for round in 0..11 {
        // Alternate ordering to limit frequency/thermal drift.
        for prepared in if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let allocs = ALLOCATIONS.load(Ordering::Relaxed);
            let started = Instant::now();
            if prepared {
                for _ in 0..count {
                    let (x, w, y, p) = black_box((&x, &w, &mut y, &plan));
                    black_box(p.bind(x, w, y)?);
                }
            } else {
                for _ in 0..count {
                    let (x, w, y, dims) = black_box((&x, &w, &mut y, (rows, input, width)));
                    let (rows, input, width) = dims;
                    black_box(GemmArguments {
                        x: x.cu_deviceptr(),
                        weight: w.cu_deviceptr(),
                        output: y.cu_deviceptr(),
                        rows: i32::try_from(rows)?,
                        input: i32::try_from(input)?,
                        width: i32::try_from(width)?,
                        x_stride: i32::try_from(input)?,
                        weight_stride: i32::try_from(input)?,
                        output_stride: i32::try_from(width)?,
                        cuda_type: 14,
                    });
                }
            }
            let ns = started.elapsed().as_nanos() as f64 / count as f64;
            let allocations = ALLOCATIONS.load(Ordering::Relaxed) - allocs;
            if allocations != 0 {
                return Err("dispatch allocated in timed loop".into());
            }
            if round > 0 {
                samples.push((prepared, ns));
            }
        }
    }
    let median = |prepared| {
        let mut values = samples
            .iter()
            .filter(|(p, _)| *p == prepared)
            .map(|(_, ns)| *ns)
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        values
            .get(values.len() / 2)
            .copied()
            .ok_or("missing timing samples")
    };
    let (raw, checked) = (median(false)?, median(true)?);
    println!(
        "{}",
        serde_json::json!({"raw_argument_ns":raw,"checked_tensor_bind_ns":checked,"added_ns":checked-raw,"allocations_per_call":0,"samples_per_path":10,"iterations_per_sample":count,"scope":"CPU binding only; excludes CUDA launch and GPU execution"})
    );
    Ok(())
}
