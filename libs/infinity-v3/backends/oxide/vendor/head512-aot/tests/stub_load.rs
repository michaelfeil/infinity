// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
use infinity_head512_aot::{
    DType, DeviceRegion, Error, Head512Batch, Head512Spec, Head512Work, KernelVariant, Provider,
    Window, HEAD_DIM,
};
use libloading::Library;
use std::path::PathBuf;

fn stub_root(variable: &str) -> PathBuf {
    PathBuf::from(std::env::var(variable).unwrap_or_else(|_| panic!("{variable}")))
}

fn batch() -> Head512Batch {
    Head512Batch {
        q: 0x1000,
        k: 0x2000,
        v: 0x3000,
        q_offsets: 0x4000,
        q_offsets_bytes: 16,
        k_offsets: 0x5000,
        k_offsets_bytes: 16,
        total_q: 7,
        total_k: 9,
        batch: 3,
        max_q: 5,
        max_k: 9,
        heads: 8,
        kv_heads: 2,
        scale: 1.0,
        dtype: DType::Bf16,
        causal: true,
        window: Window::Global,
        strides: [
            8 * 512,
            512,
            1,
            2 * 544,
            544,
            1,
            2 * 544,
            544,
            1,
            8 * 512,
            512,
            1,
        ],
    }
}

fn work() -> Head512Work {
    Head512Work::new(
        DeviceRegion::new(0x6000, 7 * 8 * 512 * 2).unwrap(),
        DeviceRegion::new(0x7000, 7 * 8 * 4).unwrap(),
    )
}

fn global_causal() -> Head512Spec {
    Head512Spec {
        dtype: DType::Bf16,
        causal: true,
        window: Window::Global,
    }
}

#[test]
#[ignore = "set HEAD512_STUB_ROOT to the test-only stub bundle"]
fn loads_prepares_and_enqueues_through_the_real_c_abi_boundary() {
    let root = stub_root("HEAD512_STUB_ROOT");
    let mut provider = Provider::load_for_arch(&root, "sm_90").unwrap();
    assert_eq!(provider.capabilities().head_dim, HEAD_DIM);
    assert_eq!(provider.kernel_variant(), KernelVariant::Native);
    assert!(matches!(
        provider.supports(global_causal()),
        Err(Error::Unprepared)
    ));
    let mut work = work();
    assert!(matches!(
        unsafe { provider.enqueue(&batch(), &mut work, 0) },
        Err(Error::Unprepared)
    ));

    unsafe {
        provider.prepare_current_context().unwrap();
        provider.prepare_current_context().unwrap();
    }
    assert!(provider.supports(global_causal()).unwrap());
    let layout = unsafe { provider.enqueue(&batch(), &mut work, 0) }.unwrap();
    assert_eq!(layout.out_elements, 7 * 8 * 512);
    assert_eq!(layout.out_physical_elements, 7 * 8 * 512);
    assert_eq!(layout.out_bytes, 7 * 8 * 512 * 2);

    let library_path = root.join("sm_90").join("libinfinity_head512_sm_90.so");
    let library = unsafe { Library::new(library_path) }.unwrap();
    let prepare_calls: libloading::Symbol<unsafe extern "C" fn() -> i32> =
        unsafe { library.get(b"infinity_head512_stub_prepare_calls\0") }.unwrap();
    assert_eq!(unsafe { prepare_calls() }, 1);
}

#[test]
#[ignore = "set HEAD512_STUB_ROOT to the test-only stub bundle"]
fn selects_native_compact_and_qualification_artifacts_cold() {
    let root = stub_root("HEAD512_STUB_ROOT");
    let native = Provider::load_for_device(
        &root,
        "sm_90",
        infinity_head512_aot::NATIVE_DYNAMIC_SHARED_MEMORY_BYTES,
    )
    .unwrap();
    assert_eq!(native.kernel_variant(), KernelVariant::Native);
    assert_eq!(
        native.capabilities().dynamic_shared_memory_bytes,
        infinity_head512_aot::NATIVE_DYNAMIC_SHARED_MEMORY_BYTES
    );

    let compact = Provider::load_for_device(
        &root,
        "sm_120",
        infinity_head512_aot::COMPACT_DYNAMIC_SHARED_MEMORY_BYTES,
    )
    .unwrap();
    assert_eq!(compact.kernel_variant(), KernelVariant::Compact);
    assert_eq!(compact.architecture(), "sm_120");
    assert_eq!(
        compact.capabilities().dynamic_shared_memory_bytes,
        infinity_head512_aot::COMPACT_DYNAMIC_SHARED_MEMORY_BYTES
    );

    let qualification = Provider::load_compact_qualification(
        &root,
        infinity_head512_aot::COMPACT_DYNAMIC_SHARED_MEMORY_BYTES,
    )
    .unwrap();
    assert_eq!(qualification.kernel_variant(), KernelVariant::Compact);
    assert_eq!(qualification.architecture(), "sm_90");
    assert_eq!(
        qualification.library_name(),
        "libinfinity_head512_sm_90_compact.so"
    );
}

#[test]
#[ignore = "set HEAD512_STUB_ROOT to the test-only stub bundle"]
fn v2_bundle_has_no_old_abi1_exports() {
    let root = stub_root("HEAD512_STUB_ROOT");
    let library_path = root.join("sm_90").join("libinfinity_head512_sm_90.so");
    let library = unsafe { Library::new(library_path) }.unwrap();
    for symbol in [
        b"infinity_head512_abi_v1\0".as_slice(),
        b"infinity_head512_variant_v1\0".as_slice(),
        b"infinity_head512_info_v1\0".as_slice(),
        b"infinity_head512_error_v1\0".as_slice(),
        b"infinity_head512_enqueue_v1\0".as_slice(),
        b"infinity_head512_prepare_v1\0".as_slice(),
    ] {
        assert!(unsafe { library.get::<unsafe extern "C" fn()>(symbol) }.is_err());
    }
}

#[test]
#[ignore = "set HEAD512_OLD_ABI_STUB_ROOT to a v1-only test stub bundle"]
fn rejects_old_abi_bundle_before_inference() {
    let root = stub_root("HEAD512_OLD_ABI_STUB_ROOT");
    let error = match Provider::load_for_arch(&root, "sm_90") {
        Ok(_) => panic!("old ABI bundle was accepted"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::Library(_)), "{error}");
    assert!(error.to_string().contains("abi_v2"), "{error}");
}

#[test]
#[ignore = "set HEAD512_MISSING_PREPARE_STUB_ROOT to a bundle without prepare_v2"]
fn rejects_bundle_without_mandatory_prepare_symbol() {
    let root = stub_root("HEAD512_MISSING_PREPARE_STUB_ROOT");
    let error = match Provider::load_for_arch(&root, "sm_90") {
        Ok(_) => panic!("bundle without prepare_v2 was accepted"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::Library(_)), "{error}");
    assert!(error.to_string().contains("prepare_v2"), "{error}");
}

#[test]
#[ignore = "set HEAD512_FAIL_PREPARE_STUB_ROOT to a stub that fails cold prepare"]
fn prepare_failure_poisons_provider_and_rejects_work() {
    let root = stub_root("HEAD512_FAIL_PREPARE_STUB_ROOT");
    let mut provider = Provider::load_for_arch(&root, "sm_90").unwrap();
    assert!(matches!(
        unsafe { provider.prepare_current_context() },
        Err(Error::Provider(_))
    ));
    assert!(matches!(
        unsafe { provider.prepare_current_context() },
        Err(Error::Provider(_))
    ));
    assert!(matches!(
        provider.supports(global_causal()),
        Err(Error::Provider(_))
    ));
    let mut work = work();
    assert!(matches!(
        unsafe { provider.enqueue(&batch(), &mut work, 0) },
        Err(Error::Provider(_))
    ));
}
