#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandCreateGenerator(
    generator: *mut curandGenerator_t,
    rng_type: curandRngType_t,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandCreateGenerator {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator, rng_type) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandCreateGeneratorHost(
    generator: *mut curandGenerator_t,
    rng_type: curandRngType_t,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandCreateGeneratorHost {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator, rng_type) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandDestroyGenerator(
    generator: curandGenerator_t,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandDestroyGenerator {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGetVersion(
    version: *mut ::std::os::raw::c_int,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGetVersion {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(version) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGetProperty(
    type_: libraryPropertyType,
    value: *mut ::std::os::raw::c_int,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGetProperty {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(type_, value) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandSetStream(
    generator: curandGenerator_t,
    stream: cudaStream_t,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandSetStream {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator, stream) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandSetPseudoRandomGeneratorSeed(
    generator: curandGenerator_t,
    seed: ::std::os::raw::c_ulonglong,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandSetPseudoRandomGeneratorSeed {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator, seed) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandSetGeneratorOffset(
    generator: curandGenerator_t,
    offset: ::std::os::raw::c_ulonglong,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandSetGeneratorOffset {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator, offset) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandSetGeneratorOrdering(
    generator: curandGenerator_t,
    order: curandOrdering_t,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandSetGeneratorOrdering {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator, order) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandSetQuasiRandomGeneratorDimensions(
    generator: curandGenerator_t,
    num_dimensions: ::std::os::raw::c_uint,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandSetQuasiRandomGeneratorDimensions {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator, num_dimensions) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGenerate(
    generator: curandGenerator_t,
    outputPtr: *mut ::std::os::raw::c_uint,
    num: usize,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGenerate {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator, outputPtr, num) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGenerateLongLong(
    generator: curandGenerator_t,
    outputPtr: *mut ::std::os::raw::c_ulonglong,
    num: usize,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGenerateLongLong {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator, outputPtr, num) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGenerateUniform(
    generator: curandGenerator_t,
    outputPtr: *mut f32,
    num: usize,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGenerateUniform {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator, outputPtr, num) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGenerateUniformDouble(
    generator: curandGenerator_t,
    outputPtr: *mut f64,
    num: usize,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGenerateUniformDouble {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator, outputPtr, num) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGenerateNormal(
    generator: curandGenerator_t,
    outputPtr: *mut f32,
    n: usize,
    mean: f32,
    stddev: f32,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGenerateNormal {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(generator, outputPtr, n, mean, stddev) }
                }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGenerateNormalDouble(
    generator: curandGenerator_t,
    outputPtr: *mut f64,
    n: usize,
    mean: f64,
    stddev: f64,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGenerateNormalDouble {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(generator, outputPtr, n, mean, stddev) }
                }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGenerateLogNormal(
    generator: curandGenerator_t,
    outputPtr: *mut f32,
    n: usize,
    mean: f32,
    stddev: f32,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGenerateLogNormal {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(generator, outputPtr, n, mean, stddev) }
                }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGenerateLogNormalDouble(
    generator: curandGenerator_t,
    outputPtr: *mut f64,
    n: usize,
    mean: f64,
    stddev: f64,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGenerateLogNormalDouble {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(generator, outputPtr, n, mean, stddev) }
                }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandCreatePoissonDistribution(
    lambda: f64,
    discrete_distribution: *mut curandDiscreteDistribution_t,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandCreatePoissonDistribution {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(lambda, discrete_distribution) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandDestroyDistribution(
    discrete_distribution: curandDiscreteDistribution_t,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandDestroyDistribution {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(discrete_distribution) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGeneratePoisson(
    generator: curandGenerator_t,
    outputPtr: *mut ::std::os::raw::c_uint,
    n: usize,
    lambda: f64,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGeneratePoisson {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator, outputPtr, n, lambda) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGeneratePoissonMethod(
    generator: curandGenerator_t,
    outputPtr: *mut ::std::os::raw::c_uint,
    n: usize,
    lambda: f64,
    method: curandMethod_t,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGeneratePoissonMethod {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(generator, outputPtr, n, lambda, method) }
                }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGenerateBinomial(
    generator: curandGenerator_t,
    outputPtr: *mut ::std::os::raw::c_uint,
    num: usize,
    n: ::std::os::raw::c_uint,
    p: f64,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGenerateBinomial {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator, outputPtr, num, n, p) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGenerateBinomialMethod(
    generator: curandGenerator_t,
    outputPtr: *mut ::std::os::raw::c_uint,
    num: usize,
    n: ::std::os::raw::c_uint,
    p: f64,
    method: curandMethod_t,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGenerateBinomialMethod {
                Ok(loaded_fn) => {
                    unsafe { (*loaded_fn)(generator, outputPtr, num, n, p, method) }
                }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGenerateSeeds(
    generator: curandGenerator_t,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGenerateSeeds {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(generator) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGetDirectionVectors32(
    vectors: *mut *mut curandDirectionVectors32_t,
    set: curandDirectionVectorSet_t,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGetDirectionVectors32 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(vectors, set) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGetScrambleConstants32(
    constants: *mut *mut ::std::os::raw::c_uint,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGetScrambleConstants32 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(constants) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGetDirectionVectors64(
    vectors: *mut *mut curandDirectionVectors64_t,
    set: curandDirectionVectorSet_t,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGetDirectionVectors64 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(vectors, set) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
#[allow(non_snake_case)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_inline_in_public_items)]
#[inline]
pub unsafe extern "C" fn curandGetScrambleConstants64(
    constants: *mut *mut ::std::os::raw::c_ulonglong,
) -> curandStatus_t {
    match curand_api() {
        Ok(api) => {
            match &api.curandGetScrambleConstants64 {
                Ok(loaded_fn) => unsafe { (*loaded_fn)(constants) }
                Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
            }
        }
        Err(_) => curandStatus_CURAND_STATUS_NOT_INITIALIZED,
    }
}
