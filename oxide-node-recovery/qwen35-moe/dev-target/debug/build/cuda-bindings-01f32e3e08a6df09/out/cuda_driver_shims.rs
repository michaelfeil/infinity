#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGetErrorString(
    error: CUresult,
    pStr: *mut *const ::std::os::raw::c_char,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGetErrorString {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(error, pStr) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGetErrorName(
    error: CUresult,
    pStr: *mut *const ::std::os::raw::c_char,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGetErrorName {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(error, pStr) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuInit(Flags: ::std::os::raw::c_uint) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuInit {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(Flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDriverGetVersion(
    driverVersion: *mut ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDriverGetVersion {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(driverVersion) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGet(
    device: *mut CUdevice,
    ordinal: ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGet {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(device, ordinal) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetCount(
    count: *mut ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetCount {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(count) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetName(
    name: *mut ::std::os::raw::c_char,
    len: ::std::os::raw::c_int,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetName {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(name, len, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetUuid_v2(
    uuid: *mut CUuuid,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetUuid_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(uuid, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetLuid(
    luid: *mut ::std::os::raw::c_char,
    deviceNodeMask: *mut ::std::os::raw::c_uint,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetLuid {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(luid, deviceNodeMask, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceTotalMem_v2(
    bytes: *mut usize,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceTotalMem_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(bytes, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetTexture1DLinearMaxWidth(
    maxWidthInElements: *mut usize,
    format: CUarray_format,
    numChannels: ::std::os::raw::c_uint,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetTexture1DLinearMaxWidth {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(maxWidthInElements, format, numChannels, dev) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetAttribute(
    pi: *mut ::std::os::raw::c_int,
    attrib: CUdevice_attribute,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pi, attrib, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetHostAtomicCapabilities(
    capabilities: *mut ::std::os::raw::c_uint,
    operations: *const CUatomicOperation,
    count: ::std::os::raw::c_uint,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetHostAtomicCapabilities {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(capabilities, operations, count, dev) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetNvSciSyncAttributes(
    nvSciSyncAttrList: *mut ::std::os::raw::c_void,
    dev: CUdevice,
    flags: ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetNvSciSyncAttributes {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(nvSciSyncAttrList, dev, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceSetMemPool(
    dev: CUdevice,
    pool: CUmemoryPool,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceSetMemPool {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dev, pool) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetMemPool(
    pool: *mut CUmemoryPool,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetMemPool {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pool, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetDefaultMemPool(
    pool_out: *mut CUmemoryPool,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetDefaultMemPool {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pool_out, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetExecAffinitySupport(
    pi: *mut ::std::os::raw::c_int,
    type_: CUexecAffinityType,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetExecAffinitySupport {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pi, type_, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuFlushGPUDirectRDMAWrites(
    target: CUflushGPUDirectRDMAWritesTarget,
    scope: CUflushGPUDirectRDMAWritesScope,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuFlushGPUDirectRDMAWrites {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(target, scope) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetProperties(
    prop: *mut CUdevprop,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetProperties {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(prop, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceComputeCapability(
    major: *mut ::std::os::raw::c_int,
    minor: *mut ::std::os::raw::c_int,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceComputeCapability {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(major, minor, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDevicePrimaryCtxRetain(
    pctx: *mut CUcontext,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDevicePrimaryCtxRetain {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pctx, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDevicePrimaryCtxRelease_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDevicePrimaryCtxSetFlags_v2(
    dev: CUdevice,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDevicePrimaryCtxSetFlags_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dev, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDevicePrimaryCtxGetState(
    dev: CUdevice,
    flags: *mut ::std::os::raw::c_uint,
    active: *mut ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDevicePrimaryCtxGetState {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dev, flags, active) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDevicePrimaryCtxReset_v2(dev: CUdevice) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDevicePrimaryCtxReset_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxCreate_v4(
    pctx: *mut CUcontext,
    ctxCreateParams: *mut CUctxCreateParams,
    flags: ::std::os::raw::c_uint,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxCreate_v4 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(pctx, ctxCreateParams, flags, dev) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxDestroy_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(ctx) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxPushCurrent_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(ctx) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxPopCurrent_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pctx) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxSetCurrent {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(ctx) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxGetCurrent {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pctx) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxGetDevice {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(device) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxGetDevice_v2(
    device: *mut CUdevice,
    ctx: CUcontext,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxGetDevice_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(device, ctx) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxGetFlags(flags: *mut ::std::os::raw::c_uint) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxGetFlags {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxSetFlags(flags: ::std::os::raw::c_uint) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxSetFlags {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxGetId(
    ctx: CUcontext,
    ctxId: *mut ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxGetId {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(ctx, ctxId) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxSynchronize() -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxSynchronize {
                Ok(loaded_fn) => unsafe { (*loaded_fn)() }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxSynchronize_v2(ctx: CUcontext) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxSynchronize_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(ctx) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxSetLimit(limit: CUlimit, value: usize) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxSetLimit {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(limit, value) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxGetLimit(pvalue: *mut usize, limit: CUlimit) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxGetLimit {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pvalue, limit) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxGetCacheConfig(pconfig: *mut CUfunc_cache) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxGetCacheConfig {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pconfig) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxSetCacheConfig(config: CUfunc_cache) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxSetCacheConfig {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(config) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxGetApiVersion(
    ctx: CUcontext,
    version: *mut ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxGetApiVersion {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(ctx, version) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxGetStreamPriorityRange(
    leastPriority: *mut ::std::os::raw::c_int,
    greatestPriority: *mut ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxGetStreamPriorityRange {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(leastPriority, greatestPriority) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxResetPersistingL2Cache() -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxResetPersistingL2Cache {
                Ok(loaded_fn) => unsafe { (*loaded_fn)() }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxGetExecAffinity(
    pExecAffinity: *mut CUexecAffinityParam,
    type_: CUexecAffinityType,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxGetExecAffinity {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pExecAffinity, type_) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxRecordEvent(hCtx: CUcontext, hEvent: CUevent) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxRecordEvent {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hCtx, hEvent) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxWaitEvent(hCtx: CUcontext, hEvent: CUevent) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxWaitEvent {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hCtx, hEvent) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxAttach(
    pctx: *mut CUcontext,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxAttach {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pctx, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxDetach(ctx: CUcontext) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxDetach {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(ctx) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxGetSharedMemConfig(
    pConfig: *mut CUsharedconfig,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxGetSharedMemConfig {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pConfig) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxSetSharedMemConfig(config: CUsharedconfig) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxSetSharedMemConfig {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(config) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuModuleLoad(
    module: *mut CUmodule,
    fname: *const ::std::os::raw::c_char,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuModuleLoad {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(module, fname) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuModuleLoadData(
    module: *mut CUmodule,
    image: *const ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuModuleLoadData {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(module, image) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuModuleLoadDataEx(
    module: *mut CUmodule,
    image: *const ::std::os::raw::c_void,
    numOptions: ::std::os::raw::c_uint,
    options: *mut CUjit_option,
    optionValues: *mut *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuModuleLoadDataEx {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(module, image, numOptions, options, optionValues)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuModuleLoadFatBinary(
    module: *mut CUmodule,
    fatCubin: *const ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuModuleLoadFatBinary {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(module, fatCubin) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuModuleUnload(hmod: CUmodule) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuModuleUnload {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hmod) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuModuleGetLoadingMode(
    mode: *mut CUmoduleLoadingMode,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuModuleGetLoadingMode {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(mode) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuModuleGetFunction(
    hfunc: *mut CUfunction,
    hmod: CUmodule,
    name: *const ::std::os::raw::c_char,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuModuleGetFunction {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hfunc, hmod, name) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuModuleGetFunctionCount(
    count: *mut ::std::os::raw::c_uint,
    mod_: CUmodule,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuModuleGetFunctionCount {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(count, mod_) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuModuleEnumerateFunctions(
    functions: *mut CUfunction,
    numFunctions: ::std::os::raw::c_uint,
    mod_: CUmodule,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuModuleEnumerateFunctions {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(functions, numFunctions, mod_) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuModuleGetGlobal_v2(
    dptr: *mut CUdeviceptr,
    bytes: *mut usize,
    hmod: CUmodule,
    name: *const ::std::os::raw::c_char,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuModuleGetGlobal_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dptr, bytes, hmod, name) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLinkCreate_v2(
    numOptions: ::std::os::raw::c_uint,
    options: *mut CUjit_option,
    optionValues: *mut *mut ::std::os::raw::c_void,
    stateOut: *mut CUlinkState,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLinkCreate_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(numOptions, options, optionValues, stateOut) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLinkAddData_v2(
    state: CUlinkState,
    type_: CUjitInputType,
    data: *mut ::std::os::raw::c_void,
    size: usize,
    name: *const ::std::os::raw::c_char,
    numOptions: ::std::os::raw::c_uint,
    options: *mut CUjit_option,
    optionValues: *mut *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLinkAddData_v2 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            state,
                            type_,
                            data,
                            size,
                            name,
                            numOptions,
                            options,
                            optionValues,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLinkAddFile_v2(
    state: CUlinkState,
    type_: CUjitInputType,
    path: *const ::std::os::raw::c_char,
    numOptions: ::std::os::raw::c_uint,
    options: *mut CUjit_option,
    optionValues: *mut *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLinkAddFile_v2 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            state,
                            type_,
                            path,
                            numOptions,
                            options,
                            optionValues,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLinkComplete(
    state: CUlinkState,
    cubinOut: *mut *mut ::std::os::raw::c_void,
    sizeOut: *mut usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLinkComplete {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(state, cubinOut, sizeOut) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLinkDestroy(state: CUlinkState) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLinkDestroy {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(state) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuModuleGetTexRef(
    pTexRef: *mut CUtexref,
    hmod: CUmodule,
    name: *const ::std::os::raw::c_char,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuModuleGetTexRef {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pTexRef, hmod, name) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuModuleGetSurfRef(
    pSurfRef: *mut CUsurfref,
    hmod: CUmodule,
    name: *const ::std::os::raw::c_char,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuModuleGetSurfRef {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pSurfRef, hmod, name) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLibraryLoadData(
    library: *mut CUlibrary,
    code: *const ::std::os::raw::c_void,
    jitOptions: *mut CUjit_option,
    jitOptionsValues: *mut *mut ::std::os::raw::c_void,
    numJitOptions: ::std::os::raw::c_uint,
    libraryOptions: *mut CUlibraryOption,
    libraryOptionValues: *mut *mut ::std::os::raw::c_void,
    numLibraryOptions: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLibraryLoadData {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            library,
                            code,
                            jitOptions,
                            jitOptionsValues,
                            numJitOptions,
                            libraryOptions,
                            libraryOptionValues,
                            numLibraryOptions,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLibraryLoadFromFile(
    library: *mut CUlibrary,
    fileName: *const ::std::os::raw::c_char,
    jitOptions: *mut CUjit_option,
    jitOptionsValues: *mut *mut ::std::os::raw::c_void,
    numJitOptions: ::std::os::raw::c_uint,
    libraryOptions: *mut CUlibraryOption,
    libraryOptionValues: *mut *mut ::std::os::raw::c_void,
    numLibraryOptions: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLibraryLoadFromFile {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            library,
                            fileName,
                            jitOptions,
                            jitOptionsValues,
                            numJitOptions,
                            libraryOptions,
                            libraryOptionValues,
                            numLibraryOptions,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLibraryUnload(library: CUlibrary) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLibraryUnload {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(library) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLibraryGetKernel(
    pKernel: *mut CUkernel,
    library: CUlibrary,
    name: *const ::std::os::raw::c_char,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLibraryGetKernel {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pKernel, library, name) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLibraryGetKernelCount(
    count: *mut ::std::os::raw::c_uint,
    lib: CUlibrary,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLibraryGetKernelCount {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(count, lib) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLibraryEnumerateKernels(
    kernels: *mut CUkernel,
    numKernels: ::std::os::raw::c_uint,
    lib: CUlibrary,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLibraryEnumerateKernels {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(kernels, numKernels, lib) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLibraryGetModule(
    pMod: *mut CUmodule,
    library: CUlibrary,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLibraryGetModule {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pMod, library) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuKernelGetFunction(
    pFunc: *mut CUfunction,
    kernel: CUkernel,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuKernelGetFunction {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pFunc, kernel) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuKernelGetLibrary(
    pLib: *mut CUlibrary,
    kernel: CUkernel,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuKernelGetLibrary {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pLib, kernel) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLibraryGetGlobal(
    dptr: *mut CUdeviceptr,
    bytes: *mut usize,
    library: CUlibrary,
    name: *const ::std::os::raw::c_char,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLibraryGetGlobal {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dptr, bytes, library, name) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLibraryGetManaged(
    dptr: *mut CUdeviceptr,
    bytes: *mut usize,
    library: CUlibrary,
    name: *const ::std::os::raw::c_char,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLibraryGetManaged {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dptr, bytes, library, name) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLibraryGetUnifiedFunction(
    fptr: *mut *mut ::std::os::raw::c_void,
    library: CUlibrary,
    symbol: *const ::std::os::raw::c_char,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLibraryGetUnifiedFunction {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(fptr, library, symbol) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuKernelGetAttribute(
    pi: *mut ::std::os::raw::c_int,
    attrib: CUfunction_attribute,
    kernel: CUkernel,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuKernelGetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pi, attrib, kernel, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuKernelSetAttribute(
    attrib: CUfunction_attribute,
    val: ::std::os::raw::c_int,
    kernel: CUkernel,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuKernelSetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(attrib, val, kernel, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuKernelSetCacheConfig(
    kernel: CUkernel,
    config: CUfunc_cache,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuKernelSetCacheConfig {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(kernel, config, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuKernelGetName(
    name: *mut *const ::std::os::raw::c_char,
    hfunc: CUkernel,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuKernelGetName {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(name, hfunc) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuKernelGetParamInfo(
    kernel: CUkernel,
    paramIndex: usize,
    paramOffset: *mut usize,
    paramSize: *mut usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuKernelGetParamInfo {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(kernel, paramIndex, paramOffset, paramSize) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemGetInfo_v2(
    free: *mut usize,
    total: *mut usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemGetInfo_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(free, total) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemAlloc_v2(
    dptr: *mut CUdeviceptr,
    bytesize: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemAlloc_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dptr, bytesize) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemAllocPitch_v2(
    dptr: *mut CUdeviceptr,
    pPitch: *mut usize,
    WidthInBytes: usize,
    Height: usize,
    ElementSizeBytes: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemAllocPitch_v2 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            dptr,
                            pPitch,
                            WidthInBytes,
                            Height,
                            ElementSizeBytes,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemFree_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dptr) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemGetAddressRange_v2(
    pbase: *mut CUdeviceptr,
    psize: *mut usize,
    dptr: CUdeviceptr,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemGetAddressRange_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pbase, psize, dptr) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemAllocHost_v2(
    pp: *mut *mut ::std::os::raw::c_void,
    bytesize: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemAllocHost_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pp, bytesize) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemFreeHost(p: *mut ::std::os::raw::c_void) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemFreeHost {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(p) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemHostAlloc(
    pp: *mut *mut ::std::os::raw::c_void,
    bytesize: usize,
    Flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemHostAlloc {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pp, bytesize, Flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemHostGetDevicePointer_v2(
    pdptr: *mut CUdeviceptr,
    p: *mut ::std::os::raw::c_void,
    Flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemHostGetDevicePointer_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pdptr, p, Flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemHostGetFlags(
    pFlags: *mut ::std::os::raw::c_uint,
    p: *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemHostGetFlags {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pFlags, p) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemAllocManaged(
    dptr: *mut CUdeviceptr,
    bytesize: usize,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemAllocManaged {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dptr, bytesize, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceRegisterAsyncNotification(
    device: CUdevice,
    callbackFunc: CUasyncCallback,
    userData: *mut ::std::os::raw::c_void,
    callback: *mut CUasyncCallbackHandle,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceRegisterAsyncNotification {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(device, callbackFunc, userData, callback) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceUnregisterAsyncNotification(
    device: CUdevice,
    callback: CUasyncCallbackHandle,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceUnregisterAsyncNotification {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(device, callback) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetByPCIBusId(
    dev: *mut CUdevice,
    pciBusId: *const ::std::os::raw::c_char,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetByPCIBusId {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dev, pciBusId) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetPCIBusId(
    pciBusId: *mut ::std::os::raw::c_char,
    len: ::std::os::raw::c_int,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetPCIBusId {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pciBusId, len, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuIpcGetEventHandle(
    pHandle: *mut CUipcEventHandle,
    event: CUevent,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuIpcGetEventHandle {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pHandle, event) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuIpcOpenEventHandle(
    phEvent: *mut CUevent,
    handle: CUipcEventHandle,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuIpcOpenEventHandle {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(phEvent, handle) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuIpcGetMemHandle(
    pHandle: *mut CUipcMemHandle,
    dptr: CUdeviceptr,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuIpcGetMemHandle {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pHandle, dptr) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuIpcOpenMemHandle_v2(
    pdptr: *mut CUdeviceptr,
    handle: CUipcMemHandle,
    Flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuIpcOpenMemHandle_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pdptr, handle, Flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuIpcCloseMemHandle(dptr: CUdeviceptr) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuIpcCloseMemHandle {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dptr) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemHostRegister_v2(
    p: *mut ::std::os::raw::c_void,
    bytesize: usize,
    Flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemHostRegister_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(p, bytesize, Flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemHostUnregister(
    p: *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemHostUnregister {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(p) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpy(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    ByteCount: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpy {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dst, src, ByteCount) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyPeer(
    dstDevice: CUdeviceptr,
    dstContext: CUcontext,
    srcDevice: CUdeviceptr,
    srcContext: CUcontext,
    ByteCount: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyPeer {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            dstDevice,
                            dstContext,
                            srcDevice,
                            srcContext,
                            ByteCount,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyHtoD_v2(
    dstDevice: CUdeviceptr,
    srcHost: *const ::std::os::raw::c_void,
    ByteCount: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyHtoD_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dstDevice, srcHost, ByteCount) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyDtoH_v2(
    dstHost: *mut ::std::os::raw::c_void,
    srcDevice: CUdeviceptr,
    ByteCount: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyDtoH_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dstHost, srcDevice, ByteCount) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyDtoD_v2(
    dstDevice: CUdeviceptr,
    srcDevice: CUdeviceptr,
    ByteCount: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyDtoD_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dstDevice, srcDevice, ByteCount) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyDtoA_v2(
    dstArray: CUarray,
    dstOffset: usize,
    srcDevice: CUdeviceptr,
    ByteCount: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyDtoA_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(dstArray, dstOffset, srcDevice, ByteCount) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyAtoD_v2(
    dstDevice: CUdeviceptr,
    srcArray: CUarray,
    srcOffset: usize,
    ByteCount: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyAtoD_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(dstDevice, srcArray, srcOffset, ByteCount) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyHtoA_v2(
    dstArray: CUarray,
    dstOffset: usize,
    srcHost: *const ::std::os::raw::c_void,
    ByteCount: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyHtoA_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(dstArray, dstOffset, srcHost, ByteCount) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyAtoH_v2(
    dstHost: *mut ::std::os::raw::c_void,
    srcArray: CUarray,
    srcOffset: usize,
    ByteCount: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyAtoH_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(dstHost, srcArray, srcOffset, ByteCount) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyAtoA_v2(
    dstArray: CUarray,
    dstOffset: usize,
    srcArray: CUarray,
    srcOffset: usize,
    ByteCount: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyAtoA_v2 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(dstArray, dstOffset, srcArray, srcOffset, ByteCount)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpy2D_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpy2D_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pCopy) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpy2DUnaligned_v2(
    pCopy: *const CUDA_MEMCPY2D,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpy2DUnaligned_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pCopy) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpy3D_v2(pCopy: *const CUDA_MEMCPY3D) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpy3D_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pCopy) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpy3DPeer(pCopy: *const CUDA_MEMCPY3D_PEER) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpy3DPeer {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pCopy) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyAsync(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyAsync {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dst, src, ByteCount, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyPeerAsync(
    dstDevice: CUdeviceptr,
    dstContext: CUcontext,
    srcDevice: CUdeviceptr,
    srcContext: CUcontext,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyPeerAsync {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            dstDevice,
                            dstContext,
                            srcDevice,
                            srcContext,
                            ByteCount,
                            hStream,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyHtoDAsync_v2(
    dstDevice: CUdeviceptr,
    srcHost: *const ::std::os::raw::c_void,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyHtoDAsync_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(dstDevice, srcHost, ByteCount, hStream) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyDtoHAsync_v2(
    dstHost: *mut ::std::os::raw::c_void,
    srcDevice: CUdeviceptr,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyDtoHAsync_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(dstHost, srcDevice, ByteCount, hStream) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyDtoDAsync_v2(
    dstDevice: CUdeviceptr,
    srcDevice: CUdeviceptr,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyDtoDAsync_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(dstDevice, srcDevice, ByteCount, hStream) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyHtoAAsync_v2(
    dstArray: CUarray,
    dstOffset: usize,
    srcHost: *const ::std::os::raw::c_void,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyHtoAAsync_v2 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(dstArray, dstOffset, srcHost, ByteCount, hStream)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyAtoHAsync_v2(
    dstHost: *mut ::std::os::raw::c_void,
    srcArray: CUarray,
    srcOffset: usize,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyAtoHAsync_v2 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(dstHost, srcArray, srcOffset, ByteCount, hStream)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpy2DAsync_v2(
    pCopy: *const CUDA_MEMCPY2D,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpy2DAsync_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pCopy, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpy3DAsync_v2(
    pCopy: *const CUDA_MEMCPY3D,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpy3DAsync_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pCopy, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpy3DPeerAsync(
    pCopy: *const CUDA_MEMCPY3D_PEER,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpy3DPeerAsync {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pCopy, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpyBatchAsync_v2(
    dsts: *mut CUdeviceptr,
    srcs: *mut CUdeviceptr,
    sizes: *mut usize,
    count: usize,
    attrs: *mut CUmemcpyAttributes,
    attrsIdxs: *mut usize,
    numAttrs: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpyBatchAsync_v2 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            dsts,
                            srcs,
                            sizes,
                            count,
                            attrs,
                            attrsIdxs,
                            numAttrs,
                            hStream,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemcpy3DBatchAsync_v2(
    numOps: usize,
    opList: *mut CUDA_MEMCPY3D_BATCH_OP,
    flags: ::std::os::raw::c_ulonglong,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemcpy3DBatchAsync_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(numOps, opList, flags, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemsetD8_v2(
    dstDevice: CUdeviceptr,
    uc: ::std::os::raw::c_uchar,
    N: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemsetD8_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dstDevice, uc, N) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemsetD16_v2(
    dstDevice: CUdeviceptr,
    us: ::std::os::raw::c_ushort,
    N: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemsetD16_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dstDevice, us, N) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemsetD32_v2(
    dstDevice: CUdeviceptr,
    ui: ::std::os::raw::c_uint,
    N: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemsetD32_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dstDevice, ui, N) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemsetD2D8_v2(
    dstDevice: CUdeviceptr,
    dstPitch: usize,
    uc: ::std::os::raw::c_uchar,
    Width: usize,
    Height: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemsetD2D8_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(dstDevice, dstPitch, uc, Width, Height) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemsetD2D16_v2(
    dstDevice: CUdeviceptr,
    dstPitch: usize,
    us: ::std::os::raw::c_ushort,
    Width: usize,
    Height: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemsetD2D16_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(dstDevice, dstPitch, us, Width, Height) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemsetD2D32_v2(
    dstDevice: CUdeviceptr,
    dstPitch: usize,
    ui: ::std::os::raw::c_uint,
    Width: usize,
    Height: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemsetD2D32_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(dstDevice, dstPitch, ui, Width, Height) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemsetD8Async(
    dstDevice: CUdeviceptr,
    uc: ::std::os::raw::c_uchar,
    N: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemsetD8Async {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dstDevice, uc, N, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemsetD16Async(
    dstDevice: CUdeviceptr,
    us: ::std::os::raw::c_ushort,
    N: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemsetD16Async {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dstDevice, us, N, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemsetD32Async(
    dstDevice: CUdeviceptr,
    ui: ::std::os::raw::c_uint,
    N: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemsetD32Async {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dstDevice, ui, N, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemsetD2D8Async(
    dstDevice: CUdeviceptr,
    dstPitch: usize,
    uc: ::std::os::raw::c_uchar,
    Width: usize,
    Height: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemsetD2D8Async {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(dstDevice, dstPitch, uc, Width, Height, hStream)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemsetD2D16Async(
    dstDevice: CUdeviceptr,
    dstPitch: usize,
    us: ::std::os::raw::c_ushort,
    Width: usize,
    Height: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemsetD2D16Async {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(dstDevice, dstPitch, us, Width, Height, hStream)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemsetD2D32Async(
    dstDevice: CUdeviceptr,
    dstPitch: usize,
    ui: ::std::os::raw::c_uint,
    Width: usize,
    Height: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemsetD2D32Async {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(dstDevice, dstPitch, ui, Width, Height, hStream)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuArrayCreate_v2(
    pHandle: *mut CUarray,
    pAllocateArray: *const CUDA_ARRAY_DESCRIPTOR,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuArrayCreate_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pHandle, pAllocateArray) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuArrayGetDescriptor_v2(
    pArrayDescriptor: *mut CUDA_ARRAY_DESCRIPTOR,
    hArray: CUarray,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuArrayGetDescriptor_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pArrayDescriptor, hArray) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuArrayGetSparseProperties(
    sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES,
    array: CUarray,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuArrayGetSparseProperties {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(sparseProperties, array) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMipmappedArrayGetSparseProperties(
    sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES,
    mipmap: CUmipmappedArray,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMipmappedArrayGetSparseProperties {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(sparseProperties, mipmap) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuArrayGetMemoryRequirements(
    memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS,
    array: CUarray,
    device: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuArrayGetMemoryRequirements {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(memoryRequirements, array, device) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMipmappedArrayGetMemoryRequirements(
    memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS,
    mipmap: CUmipmappedArray,
    device: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMipmappedArrayGetMemoryRequirements {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(memoryRequirements, mipmap, device) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuArrayGetPlane(
    pPlaneArray: *mut CUarray,
    hArray: CUarray,
    planeIdx: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuArrayGetPlane {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pPlaneArray, hArray, planeIdx) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuArrayDestroy(hArray: CUarray) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuArrayDestroy {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hArray) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuArray3DCreate_v2(
    pHandle: *mut CUarray,
    pAllocateArray: *const CUDA_ARRAY3D_DESCRIPTOR,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuArray3DCreate_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pHandle, pAllocateArray) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuArray3DGetDescriptor_v2(
    pArrayDescriptor: *mut CUDA_ARRAY3D_DESCRIPTOR,
    hArray: CUarray,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuArray3DGetDescriptor_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pArrayDescriptor, hArray) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMipmappedArrayCreate(
    pHandle: *mut CUmipmappedArray,
    pMipmappedArrayDesc: *const CUDA_ARRAY3D_DESCRIPTOR,
    numMipmapLevels: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMipmappedArrayCreate {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(pHandle, pMipmappedArrayDesc, numMipmapLevels)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMipmappedArrayGetLevel(
    pLevelArray: *mut CUarray,
    hMipmappedArray: CUmipmappedArray,
    level: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMipmappedArrayGetLevel {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(pLevelArray, hMipmappedArray, level) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMipmappedArrayDestroy(
    hMipmappedArray: CUmipmappedArray,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMipmappedArrayDestroy {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hMipmappedArray) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemGetHandleForAddressRange(
    handle: *mut ::std::os::raw::c_void,
    dptr: CUdeviceptr,
    size: usize,
    handleType: CUmemRangeHandleType,
    flags: ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemGetHandleForAddressRange {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(handle, dptr, size, handleType, flags) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemBatchDecompressAsync(
    paramsArray: *mut CUmemDecompressParams,
    count: usize,
    flags: ::std::os::raw::c_uint,
    errorIndex: *mut usize,
    stream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemBatchDecompressAsync {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(paramsArray, count, flags, errorIndex, stream)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemAddressReserve(
    ptr: *mut CUdeviceptr,
    size: usize,
    alignment: usize,
    addr: CUdeviceptr,
    flags: ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemAddressReserve {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(ptr, size, alignment, addr, flags) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemAddressFree {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(ptr, size) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemCreate(
    handle: *mut CUmemGenericAllocationHandle,
    size: usize,
    prop: *const CUmemAllocationProp,
    flags: ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemCreate {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(handle, size, prop, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemRelease(handle: CUmemGenericAllocationHandle) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemRelease {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(handle) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemMap(
    ptr: CUdeviceptr,
    size: usize,
    offset: usize,
    handle: CUmemGenericAllocationHandle,
    flags: ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemMap {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(ptr, size, offset, handle, flags) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemMapArrayAsync(
    mapInfoList: *mut CUarrayMapInfo,
    count: ::std::os::raw::c_uint,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemMapArrayAsync {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(mapInfoList, count, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemUnmap(ptr: CUdeviceptr, size: usize) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemUnmap {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(ptr, size) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemSetAccess(
    ptr: CUdeviceptr,
    size: usize,
    desc: *const CUmemAccessDesc,
    count: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemSetAccess {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(ptr, size, desc, count) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemGetAccess(
    flags: *mut ::std::os::raw::c_ulonglong,
    location: *const CUmemLocation,
    ptr: CUdeviceptr,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemGetAccess {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(flags, location, ptr) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemExportToShareableHandle(
    shareableHandle: *mut ::std::os::raw::c_void,
    handle: CUmemGenericAllocationHandle,
    handleType: CUmemAllocationHandleType,
    flags: ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemExportToShareableHandle {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(shareableHandle, handle, handleType, flags) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemImportFromShareableHandle(
    handle: *mut CUmemGenericAllocationHandle,
    osHandle: *mut ::std::os::raw::c_void,
    shHandleType: CUmemAllocationHandleType,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemImportFromShareableHandle {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(handle, osHandle, shHandleType) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemGetAllocationGranularity(
    granularity: *mut usize,
    prop: *const CUmemAllocationProp,
    option: CUmemAllocationGranularity_flags,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemGetAllocationGranularity {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(granularity, prop, option) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemGetAllocationPropertiesFromHandle(
    prop: *mut CUmemAllocationProp,
    handle: CUmemGenericAllocationHandle,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemGetAllocationPropertiesFromHandle {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(prop, handle) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemRetainAllocationHandle(
    handle: *mut CUmemGenericAllocationHandle,
    addr: *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemRetainAllocationHandle {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(handle, addr) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemFreeAsync(
    dptr: CUdeviceptr,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemFreeAsync {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dptr, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemAllocAsync(
    dptr: *mut CUdeviceptr,
    bytesize: usize,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemAllocAsync {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dptr, bytesize, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemPoolTrimTo(
    pool: CUmemoryPool,
    minBytesToKeep: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemPoolTrimTo {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pool, minBytesToKeep) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemPoolSetAttribute(
    pool: CUmemoryPool,
    attr: CUmemPool_attribute,
    value: *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemPoolSetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pool, attr, value) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemPoolGetAttribute(
    pool: CUmemoryPool,
    attr: CUmemPool_attribute,
    value: *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemPoolGetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pool, attr, value) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemPoolSetAccess(
    pool: CUmemoryPool,
    map: *const CUmemAccessDesc,
    count: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemPoolSetAccess {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pool, map, count) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemPoolGetAccess(
    flags: *mut CUmemAccess_flags,
    memPool: CUmemoryPool,
    location: *mut CUmemLocation,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemPoolGetAccess {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(flags, memPool, location) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemPoolCreate(
    pool: *mut CUmemoryPool,
    poolProps: *const CUmemPoolProps,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemPoolCreate {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pool, poolProps) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemPoolDestroy(pool: CUmemoryPool) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemPoolDestroy {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pool) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemGetDefaultMemPool(
    pool_out: *mut CUmemoryPool,
    location: *mut CUmemLocation,
    type_: CUmemAllocationType,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemGetDefaultMemPool {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pool_out, location, type_) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemGetMemPool(
    pool: *mut CUmemoryPool,
    location: *mut CUmemLocation,
    type_: CUmemAllocationType,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemGetMemPool {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pool, location, type_) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemSetMemPool(
    location: *mut CUmemLocation,
    type_: CUmemAllocationType,
    pool: CUmemoryPool,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemSetMemPool {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(location, type_, pool) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemAllocFromPoolAsync(
    dptr: *mut CUdeviceptr,
    bytesize: usize,
    pool: CUmemoryPool,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemAllocFromPoolAsync {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dptr, bytesize, pool, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemPoolExportToShareableHandle(
    handle_out: *mut ::std::os::raw::c_void,
    pool: CUmemoryPool,
    handleType: CUmemAllocationHandleType,
    flags: ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemPoolExportToShareableHandle {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(handle_out, pool, handleType, flags) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemPoolImportFromShareableHandle(
    pool_out: *mut CUmemoryPool,
    handle: *mut ::std::os::raw::c_void,
    handleType: CUmemAllocationHandleType,
    flags: ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemPoolImportFromShareableHandle {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(pool_out, handle, handleType, flags) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemPoolExportPointer(
    shareData_out: *mut CUmemPoolPtrExportData,
    ptr: CUdeviceptr,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemPoolExportPointer {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(shareData_out, ptr) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemPoolImportPointer(
    ptr_out: *mut CUdeviceptr,
    pool: CUmemoryPool,
    shareData: *mut CUmemPoolPtrExportData,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemPoolImportPointer {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(ptr_out, pool, shareData) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMulticastCreate(
    mcHandle: *mut CUmemGenericAllocationHandle,
    prop: *const CUmulticastObjectProp,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMulticastCreate {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(mcHandle, prop) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMulticastAddDevice(
    mcHandle: CUmemGenericAllocationHandle,
    dev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMulticastAddDevice {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(mcHandle, dev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMulticastBindMem(
    mcHandle: CUmemGenericAllocationHandle,
    mcOffset: usize,
    memHandle: CUmemGenericAllocationHandle,
    memOffset: usize,
    size: usize,
    flags: ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMulticastBindMem {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            mcHandle,
                            mcOffset,
                            memHandle,
                            memOffset,
                            size,
                            flags,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMulticastBindMem_v2(
    mcHandle: CUmemGenericAllocationHandle,
    dev: CUdevice,
    mcOffset: usize,
    memHandle: CUmemGenericAllocationHandle,
    memOffset: usize,
    size: usize,
    flags: ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMulticastBindMem_v2 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            mcHandle,
                            dev,
                            mcOffset,
                            memHandle,
                            memOffset,
                            size,
                            flags,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMulticastBindAddr(
    mcHandle: CUmemGenericAllocationHandle,
    mcOffset: usize,
    memptr: CUdeviceptr,
    size: usize,
    flags: ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMulticastBindAddr {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(mcHandle, mcOffset, memptr, size, flags) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMulticastBindAddr_v2(
    mcHandle: CUmemGenericAllocationHandle,
    dev: CUdevice,
    mcOffset: usize,
    memptr: CUdeviceptr,
    size: usize,
    flags: ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMulticastBindAddr_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(mcHandle, dev, mcOffset, memptr, size, flags) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMulticastUnbind(
    mcHandle: CUmemGenericAllocationHandle,
    dev: CUdevice,
    mcOffset: usize,
    size: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMulticastUnbind {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(mcHandle, dev, mcOffset, size) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMulticastGetGranularity(
    granularity: *mut usize,
    prop: *const CUmulticastObjectProp,
    option: CUmulticastGranularity_flags,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMulticastGetGranularity {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(granularity, prop, option) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuPointerGetAttribute(
    data: *mut ::std::os::raw::c_void,
    attribute: CUpointer_attribute,
    ptr: CUdeviceptr,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuPointerGetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(data, attribute, ptr) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemPrefetchAsync_v2(
    devPtr: CUdeviceptr,
    count: usize,
    location: CUmemLocation,
    flags: ::std::os::raw::c_uint,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemPrefetchAsync_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(devPtr, count, location, flags, hStream) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemAdvise_v2(
    devPtr: CUdeviceptr,
    count: usize,
    advice: CUmem_advise,
    location: CUmemLocation,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemAdvise_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(devPtr, count, advice, location) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemPrefetchBatchAsync(
    dptrs: *mut CUdeviceptr,
    sizes: *mut usize,
    count: usize,
    prefetchLocs: *mut CUmemLocation,
    prefetchLocIdxs: *mut usize,
    numPrefetchLocs: usize,
    flags: ::std::os::raw::c_ulonglong,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemPrefetchBatchAsync {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            dptrs,
                            sizes,
                            count,
                            prefetchLocs,
                            prefetchLocIdxs,
                            numPrefetchLocs,
                            flags,
                            hStream,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemDiscardBatchAsync(
    dptrs: *mut CUdeviceptr,
    sizes: *mut usize,
    count: usize,
    flags: ::std::os::raw::c_ulonglong,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemDiscardBatchAsync {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(dptrs, sizes, count, flags, hStream) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemDiscardAndPrefetchBatchAsync(
    dptrs: *mut CUdeviceptr,
    sizes: *mut usize,
    count: usize,
    prefetchLocs: *mut CUmemLocation,
    prefetchLocIdxs: *mut usize,
    numPrefetchLocs: usize,
    flags: ::std::os::raw::c_ulonglong,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemDiscardAndPrefetchBatchAsync {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            dptrs,
                            sizes,
                            count,
                            prefetchLocs,
                            prefetchLocIdxs,
                            numPrefetchLocs,
                            flags,
                            hStream,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemRangeGetAttribute(
    data: *mut ::std::os::raw::c_void,
    dataSize: usize,
    attribute: CUmem_range_attribute,
    devPtr: CUdeviceptr,
    count: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemRangeGetAttribute {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(data, dataSize, attribute, devPtr, count) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuMemRangeGetAttributes(
    data: *mut *mut ::std::os::raw::c_void,
    dataSizes: *mut usize,
    attributes: *mut CUmem_range_attribute,
    numAttributes: usize,
    devPtr: CUdeviceptr,
    count: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuMemRangeGetAttributes {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            data,
                            dataSizes,
                            attributes,
                            numAttributes,
                            devPtr,
                            count,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuPointerSetAttribute(
    value: *const ::std::os::raw::c_void,
    attribute: CUpointer_attribute,
    ptr: CUdeviceptr,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuPointerSetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(value, attribute, ptr) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuPointerGetAttributes(
    numAttributes: ::std::os::raw::c_uint,
    attributes: *mut CUpointer_attribute,
    data: *mut *mut ::std::os::raw::c_void,
    ptr: CUdeviceptr,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuPointerGetAttributes {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(numAttributes, attributes, data, ptr) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamCreate(
    phStream: *mut CUstream,
    Flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamCreate {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(phStream, Flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamCreateWithPriority(
    phStream: *mut CUstream,
    flags: ::std::os::raw::c_uint,
    priority: ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamCreateWithPriority {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(phStream, flags, priority) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamGetPriority(
    hStream: CUstream,
    priority: *mut ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamGetPriority {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, priority) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamGetDevice(
    hStream: CUstream,
    device: *mut CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamGetDevice {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, device) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamGetFlags(
    hStream: CUstream,
    flags: *mut ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamGetFlags {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamGetId(
    hStream: CUstream,
    streamId: *mut ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamGetId {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, streamId) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamGetCtx(
    hStream: CUstream,
    pctx: *mut CUcontext,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamGetCtx {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, pctx) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamGetCtx_v2(
    hStream: CUstream,
    pCtx: *mut CUcontext,
    pGreenCtx: *mut CUgreenCtx,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamGetCtx_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, pCtx, pGreenCtx) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamWaitEvent(
    hStream: CUstream,
    hEvent: CUevent,
    Flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamWaitEvent {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, hEvent, Flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamAddCallback(
    hStream: CUstream,
    callback: CUstreamCallback,
    userData: *mut ::std::os::raw::c_void,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamAddCallback {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(hStream, callback, userData, flags) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamBeginCapture_v2(
    hStream: CUstream,
    mode: CUstreamCaptureMode,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamBeginCapture_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, mode) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamBeginCaptureToGraph(
    hStream: CUstream,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    dependencyData: *const CUgraphEdgeData,
    numDependencies: usize,
    mode: CUstreamCaptureMode,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamBeginCaptureToGraph {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            hStream,
                            hGraph,
                            dependencies,
                            dependencyData,
                            numDependencies,
                            mode,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuThreadExchangeStreamCaptureMode(
    mode: *mut CUstreamCaptureMode,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuThreadExchangeStreamCaptureMode {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(mode) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamEndCapture(
    hStream: CUstream,
    phGraph: *mut CUgraph,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamEndCapture {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, phGraph) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamIsCapturing(
    hStream: CUstream,
    captureStatus: *mut CUstreamCaptureStatus,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamIsCapturing {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, captureStatus) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamGetCaptureInfo_v3(
    hStream: CUstream,
    captureStatus_out: *mut CUstreamCaptureStatus,
    id_out: *mut cuuint64_t,
    graph_out: *mut CUgraph,
    dependencies_out: *mut *const CUgraphNode,
    edgeData_out: *mut *const CUgraphEdgeData,
    numDependencies_out: *mut usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamGetCaptureInfo_v3 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            hStream,
                            captureStatus_out,
                            id_out,
                            graph_out,
                            dependencies_out,
                            edgeData_out,
                            numDependencies_out,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamUpdateCaptureDependencies_v2(
    hStream: CUstream,
    dependencies: *mut CUgraphNode,
    dependencyData: *const CUgraphEdgeData,
    numDependencies: usize,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamUpdateCaptureDependencies_v2 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            hStream,
                            dependencies,
                            dependencyData,
                            numDependencies,
                            flags,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamAttachMemAsync(
    hStream: CUstream,
    dptr: CUdeviceptr,
    length: usize,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamAttachMemAsync {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, dptr, length, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamQuery(hStream: CUstream) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamQuery {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamSynchronize(hStream: CUstream) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamSynchronize {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamDestroy_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamCopyAttributes(
    dst: CUstream,
    src: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamCopyAttributes {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dst, src) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamGetAttribute(
    hStream: CUstream,
    attr: CUstreamAttrID,
    value_out: *mut CUstreamAttrValue,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamGetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, attr, value_out) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamSetAttribute(
    hStream: CUstream,
    attr: CUstreamAttrID,
    value: *const CUstreamAttrValue,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamSetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, attr, value) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuEventCreate(
    phEvent: *mut CUevent,
    Flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuEventCreate {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(phEvent, Flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuEventRecord {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hEvent, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuEventRecordWithFlags(
    hEvent: CUevent,
    hStream: CUstream,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuEventRecordWithFlags {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hEvent, hStream, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuEventQuery(hEvent: CUevent) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuEventQuery {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hEvent) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuEventSynchronize(hEvent: CUevent) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuEventSynchronize {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hEvent) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuEventDestroy_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hEvent) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuEventElapsedTime_v2(
    pMilliseconds: *mut f32,
    hStart: CUevent,
    hEnd: CUevent,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuEventElapsedTime_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pMilliseconds, hStart, hEnd) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuImportExternalMemory(
    extMem_out: *mut CUexternalMemory,
    memHandleDesc: *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuImportExternalMemory {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(extMem_out, memHandleDesc) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuExternalMemoryGetMappedBuffer(
    devPtr: *mut CUdeviceptr,
    extMem: CUexternalMemory,
    bufferDesc: *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuExternalMemoryGetMappedBuffer {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(devPtr, extMem, bufferDesc) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuExternalMemoryGetMappedMipmappedArray(
    mipmap: *mut CUmipmappedArray,
    extMem: CUexternalMemory,
    mipmapDesc: *const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuExternalMemoryGetMappedMipmappedArray {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(mipmap, extMem, mipmapDesc) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDestroyExternalMemory(extMem: CUexternalMemory) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDestroyExternalMemory {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(extMem) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuImportExternalSemaphore(
    extSem_out: *mut CUexternalSemaphore,
    semHandleDesc: *const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuImportExternalSemaphore {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(extSem_out, semHandleDesc) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuSignalExternalSemaphoresAsync(
    extSemArray: *const CUexternalSemaphore,
    paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
    numExtSems: ::std::os::raw::c_uint,
    stream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuSignalExternalSemaphoresAsync {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(extSemArray, paramsArray, numExtSems, stream) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuWaitExternalSemaphoresAsync(
    extSemArray: *const CUexternalSemaphore,
    paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
    numExtSems: ::std::os::raw::c_uint,
    stream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuWaitExternalSemaphoresAsync {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(extSemArray, paramsArray, numExtSems, stream) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDestroyExternalSemaphore(
    extSem: CUexternalSemaphore,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDestroyExternalSemaphore {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(extSem) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamWaitValue32_v2(
    stream: CUstream,
    addr: CUdeviceptr,
    value: cuuint32_t,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamWaitValue32_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(stream, addr, value, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamWaitValue64_v2(
    stream: CUstream,
    addr: CUdeviceptr,
    value: cuuint64_t,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamWaitValue64_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(stream, addr, value, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamWriteValue32_v2(
    stream: CUstream,
    addr: CUdeviceptr,
    value: cuuint32_t,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamWriteValue32_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(stream, addr, value, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamWriteValue64_v2(
    stream: CUstream,
    addr: CUdeviceptr,
    value: cuuint64_t,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamWriteValue64_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(stream, addr, value, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamBatchMemOp_v2(
    stream: CUstream,
    count: ::std::os::raw::c_uint,
    paramArray: *mut CUstreamBatchMemOpParams,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamBatchMemOp_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(stream, count, paramArray, flags) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuFuncGetAttribute(
    pi: *mut ::std::os::raw::c_int,
    attrib: CUfunction_attribute,
    hfunc: CUfunction,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuFuncGetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pi, attrib, hfunc) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuFuncSetAttribute(
    hfunc: CUfunction,
    attrib: CUfunction_attribute,
    value: ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuFuncSetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hfunc, attrib, value) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuFuncSetCacheConfig(
    hfunc: CUfunction,
    config: CUfunc_cache,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuFuncSetCacheConfig {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hfunc, config) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuFuncGetModule(
    hmod: *mut CUmodule,
    hfunc: CUfunction,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuFuncGetModule {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hmod, hfunc) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuFuncGetName(
    name: *mut *const ::std::os::raw::c_char,
    hfunc: CUfunction,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuFuncGetName {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(name, hfunc) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuFuncGetParamInfo(
    func: CUfunction,
    paramIndex: usize,
    paramOffset: *mut usize,
    paramSize: *mut usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuFuncGetParamInfo {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(func, paramIndex, paramOffset, paramSize) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuFuncIsLoaded(
    state: *mut CUfunctionLoadingState,
    function: CUfunction,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuFuncIsLoaded {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(state, function) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuFuncLoad(function: CUfunction) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuFuncLoad {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(function) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLaunchKernel(
    f: CUfunction,
    gridDimX: ::std::os::raw::c_uint,
    gridDimY: ::std::os::raw::c_uint,
    gridDimZ: ::std::os::raw::c_uint,
    blockDimX: ::std::os::raw::c_uint,
    blockDimY: ::std::os::raw::c_uint,
    blockDimZ: ::std::os::raw::c_uint,
    sharedMemBytes: ::std::os::raw::c_uint,
    hStream: CUstream,
    kernelParams: *mut *mut ::std::os::raw::c_void,
    extra: *mut *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLaunchKernel {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            f,
                            gridDimX,
                            gridDimY,
                            gridDimZ,
                            blockDimX,
                            blockDimY,
                            blockDimZ,
                            sharedMemBytes,
                            hStream,
                            kernelParams,
                            extra,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLaunchKernelEx(
    config: *const CUlaunchConfig,
    f: CUfunction,
    kernelParams: *mut *mut ::std::os::raw::c_void,
    extra: *mut *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLaunchKernelEx {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(config, f, kernelParams, extra) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLaunchCooperativeKernel(
    f: CUfunction,
    gridDimX: ::std::os::raw::c_uint,
    gridDimY: ::std::os::raw::c_uint,
    gridDimZ: ::std::os::raw::c_uint,
    blockDimX: ::std::os::raw::c_uint,
    blockDimY: ::std::os::raw::c_uint,
    blockDimZ: ::std::os::raw::c_uint,
    sharedMemBytes: ::std::os::raw::c_uint,
    hStream: CUstream,
    kernelParams: *mut *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLaunchCooperativeKernel {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            f,
                            gridDimX,
                            gridDimY,
                            gridDimZ,
                            blockDimX,
                            blockDimY,
                            blockDimZ,
                            sharedMemBytes,
                            hStream,
                            kernelParams,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLaunchCooperativeKernelMultiDevice(
    launchParamsList: *mut CUDA_LAUNCH_PARAMS,
    numDevices: ::std::os::raw::c_uint,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLaunchCooperativeKernelMultiDevice {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(launchParamsList, numDevices, flags) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLaunchHostFunc(
    hStream: CUstream,
    fn_: CUhostFn,
    userData: *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLaunchHostFunc {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, fn_, userData) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuFuncSetBlockShape(
    hfunc: CUfunction,
    x: ::std::os::raw::c_int,
    y: ::std::os::raw::c_int,
    z: ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuFuncSetBlockShape {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hfunc, x, y, z) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuFuncSetSharedSize(
    hfunc: CUfunction,
    bytes: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuFuncSetSharedSize {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hfunc, bytes) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuParamSetSize(
    hfunc: CUfunction,
    numbytes: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuParamSetSize {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hfunc, numbytes) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuParamSeti(
    hfunc: CUfunction,
    offset: ::std::os::raw::c_int,
    value: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuParamSeti {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hfunc, offset, value) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuParamSetf(
    hfunc: CUfunction,
    offset: ::std::os::raw::c_int,
    value: f32,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuParamSetf {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hfunc, offset, value) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuParamSetv(
    hfunc: CUfunction,
    offset: ::std::os::raw::c_int,
    ptr: *mut ::std::os::raw::c_void,
    numbytes: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuParamSetv {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hfunc, offset, ptr, numbytes) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLaunch(f: CUfunction) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLaunch {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(f) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLaunchGrid(
    f: CUfunction,
    grid_width: ::std::os::raw::c_int,
    grid_height: ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLaunchGrid {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(f, grid_width, grid_height) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLaunchGridAsync(
    f: CUfunction,
    grid_width: ::std::os::raw::c_int,
    grid_height: ::std::os::raw::c_int,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLaunchGridAsync {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(f, grid_width, grid_height, hStream) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuParamSetTexRef(
    hfunc: CUfunction,
    texunit: ::std::os::raw::c_int,
    hTexRef: CUtexref,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuParamSetTexRef {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hfunc, texunit, hTexRef) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuFuncSetSharedMemConfig(
    hfunc: CUfunction,
    config: CUsharedconfig,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuFuncSetSharedMemConfig {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hfunc, config) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphCreate(
    phGraph: *mut CUgraph,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphCreate {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(phGraph, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddKernelNode_v2(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddKernelNode_v2 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            phGraphNode,
                            hGraph,
                            dependencies,
                            numDependencies,
                            nodeParams,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphKernelNodeGetParams_v2(
    hNode: CUgraphNode,
    nodeParams: *mut CUDA_KERNEL_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphKernelNodeGetParams_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphKernelNodeSetParams_v2(
    hNode: CUgraphNode,
    nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphKernelNodeSetParams_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddMemcpyNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    copyParams: *const CUDA_MEMCPY3D,
    ctx: CUcontext,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddMemcpyNode {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            phGraphNode,
                            hGraph,
                            dependencies,
                            numDependencies,
                            copyParams,
                            ctx,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphMemcpyNodeGetParams(
    hNode: CUgraphNode,
    nodeParams: *mut CUDA_MEMCPY3D,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphMemcpyNodeGetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphMemcpyNodeSetParams(
    hNode: CUgraphNode,
    nodeParams: *const CUDA_MEMCPY3D,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphMemcpyNodeSetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddMemsetNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    memsetParams: *const CUDA_MEMSET_NODE_PARAMS,
    ctx: CUcontext,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddMemsetNode {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            phGraphNode,
                            hGraph,
                            dependencies,
                            numDependencies,
                            memsetParams,
                            ctx,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphMemsetNodeGetParams(
    hNode: CUgraphNode,
    nodeParams: *mut CUDA_MEMSET_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphMemsetNodeGetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphMemsetNodeSetParams(
    hNode: CUgraphNode,
    nodeParams: *const CUDA_MEMSET_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphMemsetNodeSetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddHostNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    nodeParams: *const CUDA_HOST_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddHostNode {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            phGraphNode,
                            hGraph,
                            dependencies,
                            numDependencies,
                            nodeParams,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphHostNodeGetParams(
    hNode: CUgraphNode,
    nodeParams: *mut CUDA_HOST_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphHostNodeGetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphHostNodeSetParams(
    hNode: CUgraphNode,
    nodeParams: *const CUDA_HOST_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphHostNodeSetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddChildGraphNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    childGraph: CUgraph,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddChildGraphNode {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            phGraphNode,
                            hGraph,
                            dependencies,
                            numDependencies,
                            childGraph,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphChildGraphNodeGetGraph(
    hNode: CUgraphNode,
    phGraph: *mut CUgraph,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphChildGraphNodeGetGraph {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, phGraph) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddEmptyNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddEmptyNode {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(phGraphNode, hGraph, dependencies, numDependencies)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddEventRecordNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    event: CUevent,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddEventRecordNode {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            phGraphNode,
                            hGraph,
                            dependencies,
                            numDependencies,
                            event,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphEventRecordNodeGetEvent(
    hNode: CUgraphNode,
    event_out: *mut CUevent,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphEventRecordNodeGetEvent {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, event_out) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphEventRecordNodeSetEvent(
    hNode: CUgraphNode,
    event: CUevent,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphEventRecordNodeSetEvent {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, event) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddEventWaitNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    event: CUevent,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddEventWaitNode {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            phGraphNode,
                            hGraph,
                            dependencies,
                            numDependencies,
                            event,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphEventWaitNodeGetEvent(
    hNode: CUgraphNode,
    event_out: *mut CUevent,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphEventWaitNodeGetEvent {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, event_out) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphEventWaitNodeSetEvent(
    hNode: CUgraphNode,
    event: CUevent,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphEventWaitNodeSetEvent {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, event) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddExternalSemaphoresSignalNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddExternalSemaphoresSignalNode {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            phGraphNode,
                            hGraph,
                            dependencies,
                            numDependencies,
                            nodeParams,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExternalSemaphoresSignalNodeGetParams(
    hNode: CUgraphNode,
    params_out: *mut CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExternalSemaphoresSignalNodeGetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, params_out) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExternalSemaphoresSignalNodeSetParams(
    hNode: CUgraphNode,
    nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExternalSemaphoresSignalNodeSetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddExternalSemaphoresWaitNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddExternalSemaphoresWaitNode {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            phGraphNode,
                            hGraph,
                            dependencies,
                            numDependencies,
                            nodeParams,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExternalSemaphoresWaitNodeGetParams(
    hNode: CUgraphNode,
    params_out: *mut CUDA_EXT_SEM_WAIT_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExternalSemaphoresWaitNodeGetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, params_out) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExternalSemaphoresWaitNodeSetParams(
    hNode: CUgraphNode,
    nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExternalSemaphoresWaitNodeSetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddBatchMemOpNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddBatchMemOpNode {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            phGraphNode,
                            hGraph,
                            dependencies,
                            numDependencies,
                            nodeParams,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphBatchMemOpNodeGetParams(
    hNode: CUgraphNode,
    nodeParams_out: *mut CUDA_BATCH_MEM_OP_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphBatchMemOpNodeGetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, nodeParams_out) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphBatchMemOpNodeSetParams(
    hNode: CUgraphNode,
    nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphBatchMemOpNodeSetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecBatchMemOpNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecBatchMemOpNodeSetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddMemAllocNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    nodeParams: *mut CUDA_MEM_ALLOC_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddMemAllocNode {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            phGraphNode,
                            hGraph,
                            dependencies,
                            numDependencies,
                            nodeParams,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphMemAllocNodeGetParams(
    hNode: CUgraphNode,
    params_out: *mut CUDA_MEM_ALLOC_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphMemAllocNodeGetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, params_out) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddMemFreeNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    dptr: CUdeviceptr,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddMemFreeNode {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            phGraphNode,
                            hGraph,
                            dependencies,
                            numDependencies,
                            dptr,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphMemFreeNodeGetParams(
    hNode: CUgraphNode,
    dptr_out: *mut CUdeviceptr,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphMemFreeNodeGetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, dptr_out) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGraphMemTrim(device: CUdevice) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGraphMemTrim {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(device) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetGraphMemAttribute(
    device: CUdevice,
    attr: CUgraphMem_attribute,
    value: *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetGraphMemAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(device, attr, value) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceSetGraphMemAttribute(
    device: CUdevice,
    attr: CUgraphMem_attribute,
    value: *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceSetGraphMemAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(device, attr, value) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphClone(
    phGraphClone: *mut CUgraph,
    originalGraph: CUgraph,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphClone {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(phGraphClone, originalGraph) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphNodeFindInClone(
    phNode: *mut CUgraphNode,
    hOriginalNode: CUgraphNode,
    hClonedGraph: CUgraph,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphNodeFindInClone {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(phNode, hOriginalNode, hClonedGraph) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphNodeGetType(
    hNode: CUgraphNode,
    type_: *mut CUgraphNodeType,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphNodeGetType {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, type_) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphNodeGetContainingGraph(
    hNode: CUgraphNode,
    phGraph: *mut CUgraph,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphNodeGetContainingGraph {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, phGraph) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphNodeGetLocalId(
    hNode: CUgraphNode,
    nodeId: *mut ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphNodeGetLocalId {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, nodeId) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphNodeGetToolsId(
    hNode: CUgraphNode,
    toolsNodeId: *mut ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphNodeGetToolsId {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, toolsNodeId) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphGetId(
    hGraph: CUgraph,
    graphId: *mut ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphGetId {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraph, graphId) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecGetId(
    hGraphExec: CUgraphExec,
    graphId: *mut ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecGetId {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, graphId) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphGetNodes(
    hGraph: CUgraph,
    nodes: *mut CUgraphNode,
    numNodes: *mut usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphGetNodes {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraph, nodes, numNodes) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphGetRootNodes(
    hGraph: CUgraph,
    rootNodes: *mut CUgraphNode,
    numRootNodes: *mut usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphGetRootNodes {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraph, rootNodes, numRootNodes) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphGetEdges_v2(
    hGraph: CUgraph,
    from: *mut CUgraphNode,
    to: *mut CUgraphNode,
    edgeData: *mut CUgraphEdgeData,
    numEdges: *mut usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphGetEdges_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(hGraph, from, to, edgeData, numEdges) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphNodeGetDependencies_v2(
    hNode: CUgraphNode,
    dependencies: *mut CUgraphNode,
    edgeData: *mut CUgraphEdgeData,
    numDependencies: *mut usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphNodeGetDependencies_v2 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(hNode, dependencies, edgeData, numDependencies)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphNodeGetDependentNodes_v2(
    hNode: CUgraphNode,
    dependentNodes: *mut CUgraphNode,
    edgeData: *mut CUgraphEdgeData,
    numDependentNodes: *mut usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphNodeGetDependentNodes_v2 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(hNode, dependentNodes, edgeData, numDependentNodes)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddDependencies_v2(
    hGraph: CUgraph,
    from: *const CUgraphNode,
    to: *const CUgraphNode,
    edgeData: *const CUgraphEdgeData,
    numDependencies: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddDependencies_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(hGraph, from, to, edgeData, numDependencies) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphRemoveDependencies_v2(
    hGraph: CUgraph,
    from: *const CUgraphNode,
    to: *const CUgraphNode,
    edgeData: *const CUgraphEdgeData,
    numDependencies: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphRemoveDependencies_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(hGraph, from, to, edgeData, numDependencies) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphDestroyNode(hNode: CUgraphNode) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphDestroyNode {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphInstantiateWithFlags(
    phGraphExec: *mut CUgraphExec,
    hGraph: CUgraph,
    flags: ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphInstantiateWithFlags {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(phGraphExec, hGraph, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphInstantiateWithParams(
    phGraphExec: *mut CUgraphExec,
    hGraph: CUgraph,
    instantiateParams: *mut CUDA_GRAPH_INSTANTIATE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphInstantiateWithParams {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(phGraphExec, hGraph, instantiateParams) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecGetFlags(
    hGraphExec: CUgraphExec,
    flags: *mut cuuint64_t,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecGetFlags {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecKernelNodeSetParams_v2(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecKernelNodeSetParams_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecMemcpyNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    copyParams: *const CUDA_MEMCPY3D,
    ctx: CUcontext,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecMemcpyNodeSetParams {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(hGraphExec, hNode, copyParams, ctx) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecMemsetNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    memsetParams: *const CUDA_MEMSET_NODE_PARAMS,
    ctx: CUcontext,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecMemsetNodeSetParams {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(hGraphExec, hNode, memsetParams, ctx) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecHostNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    nodeParams: *const CUDA_HOST_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecHostNodeSetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecChildGraphNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    childGraph: CUgraph,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecChildGraphNodeSetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, hNode, childGraph) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecEventRecordNodeSetEvent(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    event: CUevent,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecEventRecordNodeSetEvent {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, hNode, event) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecEventWaitNodeSetEvent(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    event: CUevent,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecEventWaitNodeSetEvent {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, hNode, event) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecExternalSemaphoresSignalNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecExternalSemaphoresSignalNodeSetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecExternalSemaphoresWaitNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecExternalSemaphoresWaitNodeSetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphNodeSetEnabled(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    isEnabled: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphNodeSetEnabled {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, hNode, isEnabled) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphNodeGetEnabled(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    isEnabled: *mut ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphNodeGetEnabled {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, hNode, isEnabled) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphUpload(
    hGraphExec: CUgraphExec,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphUpload {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphLaunch(
    hGraphExec: CUgraphExec,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphLaunch {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecDestroy(hGraphExec: CUgraphExec) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecDestroy {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphDestroy(hGraph: CUgraph) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphDestroy {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraph) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecUpdate_v2(
    hGraphExec: CUgraphExec,
    hGraph: CUgraph,
    resultInfo: *mut CUgraphExecUpdateResultInfo,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecUpdate_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, hGraph, resultInfo) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphKernelNodeCopyAttributes(
    dst: CUgraphNode,
    src: CUgraphNode,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphKernelNodeCopyAttributes {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(dst, src) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphKernelNodeGetAttribute(
    hNode: CUgraphNode,
    attr: CUkernelNodeAttrID,
    value_out: *mut CUkernelNodeAttrValue,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphKernelNodeGetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, attr, value_out) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphKernelNodeSetAttribute(
    hNode: CUgraphNode,
    attr: CUkernelNodeAttrID,
    value: *const CUkernelNodeAttrValue,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphKernelNodeSetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, attr, value) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphDebugDotPrint(
    hGraph: CUgraph,
    path: *const ::std::os::raw::c_char,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphDebugDotPrint {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraph, path, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuUserObjectCreate(
    object_out: *mut CUuserObject,
    ptr: *mut ::std::os::raw::c_void,
    destroy: CUhostFn,
    initialRefcount: ::std::os::raw::c_uint,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuUserObjectCreate {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(object_out, ptr, destroy, initialRefcount, flags)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuUserObjectRetain(
    object: CUuserObject,
    count: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuUserObjectRetain {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(object, count) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuUserObjectRelease(
    object: CUuserObject,
    count: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuUserObjectRelease {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(object, count) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphRetainUserObject(
    graph: CUgraph,
    object: CUuserObject,
    count: ::std::os::raw::c_uint,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphRetainUserObject {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(graph, object, count, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphReleaseUserObject(
    graph: CUgraph,
    object: CUuserObject,
    count: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphReleaseUserObject {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(graph, object, count) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphAddNode_v2(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    dependencyData: *const CUgraphEdgeData,
    numDependencies: usize,
    nodeParams: *mut CUgraphNodeParams,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphAddNode_v2 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            phGraphNode,
                            hGraph,
                            dependencies,
                            dependencyData,
                            numDependencies,
                            nodeParams,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphNodeSetParams(
    hNode: CUgraphNode,
    nodeParams: *mut CUgraphNodeParams,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphNodeSetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphExecNodeSetParams(
    hGraphExec: CUgraphExec,
    hNode: CUgraphNode,
    nodeParams: *mut CUgraphNodeParams,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphExecNodeSetParams {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hGraphExec, hNode, nodeParams) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphConditionalHandleCreate(
    pHandle_out: *mut CUgraphConditionalHandle,
    hGraph: CUgraph,
    ctx: CUcontext,
    defaultLaunchValue: ::std::os::raw::c_uint,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphConditionalHandleCreate {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(pHandle_out, hGraph, ctx, defaultLaunchValue, flags)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
    numBlocks: *mut ::std::os::raw::c_int,
    func: CUfunction,
    blockSize: ::std::os::raw::c_int,
    dynamicSMemSize: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuOccupancyMaxActiveBlocksPerMultiprocessor {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(numBlocks, func, blockSize, dynamicSMemSize) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    numBlocks: *mut ::std::os::raw::c_int,
    func: CUfunction,
    blockSize: ::std::os::raw::c_int,
    dynamicSMemSize: usize,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(numBlocks, func, blockSize, dynamicSMemSize, flags)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuOccupancyMaxPotentialBlockSize(
    minGridSize: *mut ::std::os::raw::c_int,
    blockSize: *mut ::std::os::raw::c_int,
    func: CUfunction,
    blockSizeToDynamicSMemSize: CUoccupancyB2DSize,
    dynamicSMemSize: usize,
    blockSizeLimit: ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuOccupancyMaxPotentialBlockSize {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            minGridSize,
                            blockSize,
                            func,
                            blockSizeToDynamicSMemSize,
                            dynamicSMemSize,
                            blockSizeLimit,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuOccupancyMaxPotentialBlockSizeWithFlags(
    minGridSize: *mut ::std::os::raw::c_int,
    blockSize: *mut ::std::os::raw::c_int,
    func: CUfunction,
    blockSizeToDynamicSMemSize: CUoccupancyB2DSize,
    dynamicSMemSize: usize,
    blockSizeLimit: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuOccupancyMaxPotentialBlockSizeWithFlags {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            minGridSize,
                            blockSize,
                            func,
                            blockSizeToDynamicSMemSize,
                            dynamicSMemSize,
                            blockSizeLimit,
                            flags,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuOccupancyAvailableDynamicSMemPerBlock(
    dynamicSmemSize: *mut usize,
    func: CUfunction,
    numBlocks: ::std::os::raw::c_int,
    blockSize: ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuOccupancyAvailableDynamicSMemPerBlock {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(dynamicSmemSize, func, numBlocks, blockSize) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuOccupancyMaxPotentialClusterSize(
    clusterSize: *mut ::std::os::raw::c_int,
    func: CUfunction,
    config: *const CUlaunchConfig,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuOccupancyMaxPotentialClusterSize {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(clusterSize, func, config) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuOccupancyMaxActiveClusters(
    numClusters: *mut ::std::os::raw::c_int,
    func: CUfunction,
    config: *const CUlaunchConfig,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuOccupancyMaxActiveClusters {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(numClusters, func, config) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefSetArray(
    hTexRef: CUtexref,
    hArray: CUarray,
    Flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefSetArray {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hTexRef, hArray, Flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefSetMipmappedArray(
    hTexRef: CUtexref,
    hMipmappedArray: CUmipmappedArray,
    Flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefSetMipmappedArray {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hTexRef, hMipmappedArray, Flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefSetAddress_v2(
    ByteOffset: *mut usize,
    hTexRef: CUtexref,
    dptr: CUdeviceptr,
    bytes: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefSetAddress_v2 {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(ByteOffset, hTexRef, dptr, bytes) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefSetAddress2D_v3(
    hTexRef: CUtexref,
    desc: *const CUDA_ARRAY_DESCRIPTOR,
    dptr: CUdeviceptr,
    Pitch: usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefSetAddress2D_v3 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hTexRef, desc, dptr, Pitch) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefSetFormat(
    hTexRef: CUtexref,
    fmt: CUarray_format,
    NumPackedComponents: ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefSetFormat {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(hTexRef, fmt, NumPackedComponents) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefSetAddressMode(
    hTexRef: CUtexref,
    dim: ::std::os::raw::c_int,
    am: CUaddress_mode,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefSetAddressMode {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hTexRef, dim, am) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefSetFilterMode(
    hTexRef: CUtexref,
    fm: CUfilter_mode,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefSetFilterMode {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hTexRef, fm) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefSetMipmapFilterMode(
    hTexRef: CUtexref,
    fm: CUfilter_mode,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefSetMipmapFilterMode {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hTexRef, fm) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefSetMipmapLevelBias(
    hTexRef: CUtexref,
    bias: f32,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefSetMipmapLevelBias {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hTexRef, bias) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefSetMipmapLevelClamp(
    hTexRef: CUtexref,
    minMipmapLevelClamp: f32,
    maxMipmapLevelClamp: f32,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefSetMipmapLevelClamp {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefSetMaxAnisotropy(
    hTexRef: CUtexref,
    maxAniso: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefSetMaxAnisotropy {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hTexRef, maxAniso) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefSetBorderColor(
    hTexRef: CUtexref,
    pBorderColor: *mut f32,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefSetBorderColor {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hTexRef, pBorderColor) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefSetFlags(
    hTexRef: CUtexref,
    Flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefSetFlags {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hTexRef, Flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefGetAddress_v2(
    pdptr: *mut CUdeviceptr,
    hTexRef: CUtexref,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefGetAddress_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pdptr, hTexRef) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefGetArray(
    phArray: *mut CUarray,
    hTexRef: CUtexref,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefGetArray {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(phArray, hTexRef) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefGetMipmappedArray(
    phMipmappedArray: *mut CUmipmappedArray,
    hTexRef: CUtexref,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefGetMipmappedArray {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(phMipmappedArray, hTexRef) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefGetAddressMode(
    pam: *mut CUaddress_mode,
    hTexRef: CUtexref,
    dim: ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefGetAddressMode {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pam, hTexRef, dim) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefGetFilterMode(
    pfm: *mut CUfilter_mode,
    hTexRef: CUtexref,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefGetFilterMode {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pfm, hTexRef) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefGetFormat(
    pFormat: *mut CUarray_format,
    pNumChannels: *mut ::std::os::raw::c_int,
    hTexRef: CUtexref,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefGetFormat {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pFormat, pNumChannels, hTexRef) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefGetMipmapFilterMode(
    pfm: *mut CUfilter_mode,
    hTexRef: CUtexref,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefGetMipmapFilterMode {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pfm, hTexRef) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefGetMipmapLevelBias(
    pbias: *mut f32,
    hTexRef: CUtexref,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefGetMipmapLevelBias {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pbias, hTexRef) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefGetMipmapLevelClamp(
    pminMipmapLevelClamp: *mut f32,
    pmaxMipmapLevelClamp: *mut f32,
    hTexRef: CUtexref,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefGetMipmapLevelClamp {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefGetMaxAnisotropy(
    pmaxAniso: *mut ::std::os::raw::c_int,
    hTexRef: CUtexref,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefGetMaxAnisotropy {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pmaxAniso, hTexRef) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefGetBorderColor(
    pBorderColor: *mut f32,
    hTexRef: CUtexref,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefGetBorderColor {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pBorderColor, hTexRef) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefGetFlags(
    pFlags: *mut ::std::os::raw::c_uint,
    hTexRef: CUtexref,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefGetFlags {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pFlags, hTexRef) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefCreate(pTexRef: *mut CUtexref) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefCreate {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pTexRef) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexRefDestroy(hTexRef: CUtexref) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexRefDestroy {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hTexRef) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuSurfRefSetArray(
    hSurfRef: CUsurfref,
    hArray: CUarray,
    Flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuSurfRefSetArray {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hSurfRef, hArray, Flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuSurfRefGetArray(
    phArray: *mut CUarray,
    hSurfRef: CUsurfref,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuSurfRefGetArray {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(phArray, hSurfRef) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexObjectCreate(
    pTexObject: *mut CUtexObject,
    pResDesc: *const CUDA_RESOURCE_DESC,
    pTexDesc: *const CUDA_TEXTURE_DESC,
    pResViewDesc: *const CUDA_RESOURCE_VIEW_DESC,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexObjectCreate {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(pTexObject, pResDesc, pTexDesc, pResViewDesc) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexObjectDestroy(texObject: CUtexObject) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexObjectDestroy {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(texObject) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexObjectGetResourceDesc(
    pResDesc: *mut CUDA_RESOURCE_DESC,
    texObject: CUtexObject,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexObjectGetResourceDesc {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pResDesc, texObject) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexObjectGetTextureDesc(
    pTexDesc: *mut CUDA_TEXTURE_DESC,
    texObject: CUtexObject,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexObjectGetTextureDesc {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pTexDesc, texObject) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTexObjectGetResourceViewDesc(
    pResViewDesc: *mut CUDA_RESOURCE_VIEW_DESC,
    texObject: CUtexObject,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTexObjectGetResourceViewDesc {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pResViewDesc, texObject) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuSurfObjectCreate(
    pSurfObject: *mut CUsurfObject,
    pResDesc: *const CUDA_RESOURCE_DESC,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuSurfObjectCreate {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pSurfObject, pResDesc) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuSurfObjectDestroy(surfObject: CUsurfObject) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuSurfObjectDestroy {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(surfObject) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuSurfObjectGetResourceDesc(
    pResDesc: *mut CUDA_RESOURCE_DESC,
    surfObject: CUsurfObject,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuSurfObjectGetResourceDesc {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pResDesc, surfObject) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTensorMapEncodeTiled(
    tensorMap: *mut CUtensorMap,
    tensorDataType: CUtensorMapDataType,
    tensorRank: cuuint32_t,
    globalAddress: *mut ::std::os::raw::c_void,
    globalDim: *const cuuint64_t,
    globalStrides: *const cuuint64_t,
    boxDim: *const cuuint32_t,
    elementStrides: *const cuuint32_t,
    interleave: CUtensorMapInterleave,
    swizzle: CUtensorMapSwizzle,
    l2Promotion: CUtensorMapL2promotion,
    oobFill: CUtensorMapFloatOOBfill,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTensorMapEncodeTiled {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            tensorMap,
                            tensorDataType,
                            tensorRank,
                            globalAddress,
                            globalDim,
                            globalStrides,
                            boxDim,
                            elementStrides,
                            interleave,
                            swizzle,
                            l2Promotion,
                            oobFill,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTensorMapEncodeIm2col(
    tensorMap: *mut CUtensorMap,
    tensorDataType: CUtensorMapDataType,
    tensorRank: cuuint32_t,
    globalAddress: *mut ::std::os::raw::c_void,
    globalDim: *const cuuint64_t,
    globalStrides: *const cuuint64_t,
    pixelBoxLowerCorner: *const ::std::os::raw::c_int,
    pixelBoxUpperCorner: *const ::std::os::raw::c_int,
    channelsPerPixel: cuuint32_t,
    pixelsPerColumn: cuuint32_t,
    elementStrides: *const cuuint32_t,
    interleave: CUtensorMapInterleave,
    swizzle: CUtensorMapSwizzle,
    l2Promotion: CUtensorMapL2promotion,
    oobFill: CUtensorMapFloatOOBfill,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTensorMapEncodeIm2col {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            tensorMap,
                            tensorDataType,
                            tensorRank,
                            globalAddress,
                            globalDim,
                            globalStrides,
                            pixelBoxLowerCorner,
                            pixelBoxUpperCorner,
                            channelsPerPixel,
                            pixelsPerColumn,
                            elementStrides,
                            interleave,
                            swizzle,
                            l2Promotion,
                            oobFill,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTensorMapEncodeIm2colWide(
    tensorMap: *mut CUtensorMap,
    tensorDataType: CUtensorMapDataType,
    tensorRank: cuuint32_t,
    globalAddress: *mut ::std::os::raw::c_void,
    globalDim: *const cuuint64_t,
    globalStrides: *const cuuint64_t,
    pixelBoxLowerCornerWidth: ::std::os::raw::c_int,
    pixelBoxUpperCornerWidth: ::std::os::raw::c_int,
    channelsPerPixel: cuuint32_t,
    pixelsPerColumn: cuuint32_t,
    elementStrides: *const cuuint32_t,
    interleave: CUtensorMapInterleave,
    mode: CUtensorMapIm2ColWideMode,
    swizzle: CUtensorMapSwizzle,
    l2Promotion: CUtensorMapL2promotion,
    oobFill: CUtensorMapFloatOOBfill,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTensorMapEncodeIm2colWide {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            tensorMap,
                            tensorDataType,
                            tensorRank,
                            globalAddress,
                            globalDim,
                            globalStrides,
                            pixelBoxLowerCornerWidth,
                            pixelBoxUpperCornerWidth,
                            channelsPerPixel,
                            pixelsPerColumn,
                            elementStrides,
                            interleave,
                            mode,
                            swizzle,
                            l2Promotion,
                            oobFill,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuTensorMapReplaceAddress(
    tensorMap: *mut CUtensorMap,
    globalAddress: *mut ::std::os::raw::c_void,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuTensorMapReplaceAddress {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(tensorMap, globalAddress) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceCanAccessPeer(
    canAccessPeer: *mut ::std::os::raw::c_int,
    dev: CUdevice,
    peerDev: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceCanAccessPeer {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(canAccessPeer, dev, peerDev) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxEnablePeerAccess(
    peerContext: CUcontext,
    Flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxEnablePeerAccess {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(peerContext, Flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxDisablePeerAccess(peerContext: CUcontext) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxDisablePeerAccess {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(peerContext) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetP2PAttribute(
    value: *mut ::std::os::raw::c_int,
    attrib: CUdevice_P2PAttribute,
    srcDevice: CUdevice,
    dstDevice: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetP2PAttribute {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(value, attrib, srcDevice, dstDevice) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetP2PAtomicCapabilities(
    capabilities: *mut ::std::os::raw::c_uint,
    operations: *const CUatomicOperation,
    count: ::std::os::raw::c_uint,
    srcDevice: CUdevice,
    dstDevice: CUdevice,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetP2PAtomicCapabilities {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            capabilities,
                            operations,
                            count,
                            srcDevice,
                            dstDevice,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphicsUnregisterResource(
    resource: CUgraphicsResource,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphicsUnregisterResource {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(resource) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphicsSubResourceGetMappedArray(
    pArray: *mut CUarray,
    resource: CUgraphicsResource,
    arrayIndex: ::std::os::raw::c_uint,
    mipLevel: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphicsSubResourceGetMappedArray {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(pArray, resource, arrayIndex, mipLevel) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphicsResourceGetMappedMipmappedArray(
    pMipmappedArray: *mut CUmipmappedArray,
    resource: CUgraphicsResource,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphicsResourceGetMappedMipmappedArray {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pMipmappedArray, resource) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphicsResourceGetMappedPointer_v2(
    pDevPtr: *mut CUdeviceptr,
    pSize: *mut usize,
    resource: CUgraphicsResource,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphicsResourceGetMappedPointer_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pDevPtr, pSize, resource) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphicsResourceSetMapFlags_v2(
    resource: CUgraphicsResource,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphicsResourceSetMapFlags_v2 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(resource, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphicsMapResources(
    count: ::std::os::raw::c_uint,
    resources: *mut CUgraphicsResource,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphicsMapResources {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(count, resources, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGraphicsUnmapResources(
    count: ::std::os::raw::c_uint,
    resources: *mut CUgraphicsResource,
    hStream: CUstream,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGraphicsUnmapResources {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(count, resources, hStream) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGetProcAddress_v2(
    symbol: *const ::std::os::raw::c_char,
    pfn: *mut *mut ::std::os::raw::c_void,
    cudaVersion: ::std::os::raw::c_int,
    flags: cuuint64_t,
    symbolStatus: *mut CUdriverProcAddressQueryResult,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGetProcAddress_v2 {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(symbol, pfn, cudaVersion, flags, symbolStatus)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCoredumpGetAttribute(
    attrib: CUcoredumpSettings,
    value: *mut ::std::os::raw::c_void,
    size: *mut usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCoredumpGetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(attrib, value, size) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCoredumpGetAttributeGlobal(
    attrib: CUcoredumpSettings,
    value: *mut ::std::os::raw::c_void,
    size: *mut usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCoredumpGetAttributeGlobal {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(attrib, value, size) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCoredumpSetAttribute(
    attrib: CUcoredumpSettings,
    value: *mut ::std::os::raw::c_void,
    size: *mut usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCoredumpSetAttribute {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(attrib, value, size) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCoredumpSetAttributeGlobal(
    attrib: CUcoredumpSettings,
    value: *mut ::std::os::raw::c_void,
    size: *mut usize,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCoredumpSetAttributeGlobal {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(attrib, value, size) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGetExportTable(
    ppExportTable: *mut *const ::std::os::raw::c_void,
    pExportTableId: *const CUuuid,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGetExportTable {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(ppExportTable, pExportTableId) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGreenCtxCreate(
    phCtx: *mut CUgreenCtx,
    desc: CUdevResourceDesc,
    dev: CUdevice,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGreenCtxCreate {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(phCtx, desc, dev, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGreenCtxDestroy(hCtx: CUgreenCtx) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGreenCtxDestroy {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hCtx) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxFromGreenCtx(
    pContext: *mut CUcontext,
    hCtx: CUgreenCtx,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxFromGreenCtx {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pContext, hCtx) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDeviceGetDevResource(
    device: CUdevice,
    resource: *mut CUdevResource,
    type_: CUdevResourceType,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDeviceGetDevResource {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(device, resource, type_) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCtxGetDevResource(
    hCtx: CUcontext,
    resource: *mut CUdevResource,
    type_: CUdevResourceType,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCtxGetDevResource {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hCtx, resource, type_) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGreenCtxGetDevResource(
    hCtx: CUgreenCtx,
    resource: *mut CUdevResource,
    type_: CUdevResourceType,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGreenCtxGetDevResource {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hCtx, resource, type_) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDevSmResourceSplitByCount(
    result: *mut CUdevResource,
    nbGroups: *mut ::std::os::raw::c_uint,
    input: *const CUdevResource,
    remainder: *mut CUdevResource,
    flags: ::std::os::raw::c_uint,
    minCount: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDevSmResourceSplitByCount {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(result, nbGroups, input, remainder, flags, minCount)
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDevSmResourceSplit(
    result: *mut CUdevResource,
    nbGroups: ::std::os::raw::c_uint,
    input: *const CUdevResource,
    remainder: *mut CUdevResource,
    flags: ::std::os::raw::c_uint,
    groupParams: *mut CU_DEV_SM_RESOURCE_GROUP_PARAMS,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDevSmResourceSplit {
                Ok(loaded_fn) => {
                    unsafe {
                        (*loaded_fn)(
                            result,
                            nbGroups,
                            input,
                            remainder,
                            flags,
                            groupParams,
                        )
                    }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuDevResourceGenerateDesc(
    phDesc: *mut CUdevResourceDesc,
    resources: *mut CUdevResource,
    nbResources: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuDevResourceGenerateDesc {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(phDesc, resources, nbResources) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGreenCtxRecordEvent(
    hCtx: CUgreenCtx,
    hEvent: CUevent,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGreenCtxRecordEvent {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hCtx, hEvent) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGreenCtxWaitEvent(
    hCtx: CUgreenCtx,
    hEvent: CUevent,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGreenCtxWaitEvent {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hCtx, hEvent) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamGetGreenCtx(
    hStream: CUstream,
    phCtx: *mut CUgreenCtx,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamGetGreenCtx {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, phCtx) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGreenCtxStreamCreate(
    phStream: *mut CUstream,
    greenCtx: CUgreenCtx,
    flags: ::std::os::raw::c_uint,
    priority: ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGreenCtxStreamCreate {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(phStream, greenCtx, flags, priority) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuGreenCtxGetId(
    greenCtx: CUgreenCtx,
    greenCtxId: *mut ::std::os::raw::c_ulonglong,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuGreenCtxGetId {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(greenCtx, greenCtxId) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuStreamGetDevResource(
    hStream: CUstream,
    resource: *mut CUdevResource,
    type_: CUdevResourceType,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuStreamGetDevResource {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(hStream, resource, type_) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLogsRegisterCallback(
    callbackFunc: CUlogsCallback,
    userData: *mut ::std::os::raw::c_void,
    callback_out: *mut CUlogsCallbackHandle,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLogsRegisterCallback {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(callbackFunc, userData, callback_out) }
                }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLogsUnregisterCallback(
    callback: CUlogsCallbackHandle,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLogsUnregisterCallback {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(callback) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLogsCurrent(
    iterator_out: *mut CUlogIterator,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLogsCurrent {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(iterator_out, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLogsDumpToFile(
    iterator: *mut CUlogIterator,
    pathToFile: *const ::std::os::raw::c_char,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLogsDumpToFile {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(iterator, pathToFile, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuLogsDumpToMemory(
    iterator: *mut CUlogIterator,
    buffer: *mut ::std::os::raw::c_char,
    size: *mut usize,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuLogsDumpToMemory {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(iterator, buffer, size, flags) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCheckpointProcessGetRestoreThreadId(
    pid: ::std::os::raw::c_int,
    tid: *mut ::std::os::raw::c_int,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCheckpointProcessGetRestoreThreadId {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pid, tid) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCheckpointProcessGetState(
    pid: ::std::os::raw::c_int,
    state: *mut CUprocessState,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCheckpointProcessGetState {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pid, state) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCheckpointProcessLock(
    pid: ::std::os::raw::c_int,
    args: *mut CUcheckpointLockArgs,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCheckpointProcessLock {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pid, args) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCheckpointProcessCheckpoint(
    pid: ::std::os::raw::c_int,
    args: *mut CUcheckpointCheckpointArgs,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCheckpointProcessCheckpoint {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pid, args) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCheckpointProcessRestore(
    pid: ::std::os::raw::c_int,
    args: *mut CUcheckpointRestoreArgs,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCheckpointProcessRestore {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pid, args) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn cuCheckpointProcessUnlock(
    pid: ::std::os::raw::c_int,
    args: *mut CUcheckpointUnlockArgs,
) -> CUresult {
    match cuda_driver() {
        Ok(api) => {
            match &api.cuCheckpointProcessUnlock {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(pid, args) }
                Err(_) => cudaError_enum_CUDA_ERROR_NOT_FOUND,
            }
        }
        Err(_) => cudaError_enum_CUDA_ERROR_NOT_INITIALIZED,
    }
}
