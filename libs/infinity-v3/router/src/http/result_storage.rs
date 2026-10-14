//! Checked host-result allocations charged before constructing their payloads.
use infinity_v3_backend::{result_memory::reserve_for_request, BackendError};

pub(crate) fn vector<T>(capacity: usize) -> Result<Vec<T>, BackendError> {
    let bytes = capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| BackendError::Overloaded("Derived result vector size overflow".into()))?;
    reserve_for_request(bytes)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|_| BackendError::Overloaded("Derived result vector allocation failed".into()))?;
    Ok(result)
}

pub(crate) fn string(value: &str) -> Result<String, BackendError> {
    let mut result = string_capacity(value.len())?;
    result.push_str(value);
    Ok(result)
}

pub(crate) fn string_capacity(capacity: usize) -> Result<String, BackendError> {
    reserve_for_request(capacity)?;
    let mut result = String::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|_| BackendError::Overloaded("Derived result string allocation failed".into()))?;
    Ok(result)
}

pub(crate) fn index(value: usize) -> Result<String, BackendError> {
    use std::fmt::Write;
    let mut result = string_capacity(3 * std::mem::size_of::<usize>())?;
    write!(&mut result, "{value}").expect("integer fits admitted string");
    Ok(result)
}
