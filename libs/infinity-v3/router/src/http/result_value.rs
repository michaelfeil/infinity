//! Fallible response trees: requested vector and string capacities are admitted
//! before allocation and retained by the existing request result owner.
use super::result_storage;
use infinity_v3_backend::BackendError;
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Serialize, Serializer};
use serde_json::{Number, Value};

#[derive(Debug)]
pub(crate) enum AnswerError {
    Protocol(String),
    Memory(BackendError),
}
impl From<String> for AnswerError {
    fn from(v: String) -> Self {
        Self::Protocol(v)
    }
}
impl From<&str> for AnswerError {
    fn from(v: &str) -> Self {
        Self::Protocol(v.into())
    }
}
impl From<BackendError> for AnswerError {
    fn from(v: BackendError) -> Self {
        Self::Memory(v)
    }
}
impl std::fmt::Display for AnswerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Protocol(s) => s.fmt(f),
            Self::Memory(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for AnswerError {}

#[derive(Debug)]
pub(crate) enum ResultValue {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}
impl Serialize for ResultValue {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => s.serialize_unit(),
            Self::Bool(v) => s.serialize_bool(*v),
            Self::Number(v) => v.serialize(s),
            Self::String(v) => s.serialize_str(v),
            Self::Array(v) => {
                let mut out = s.serialize_seq(Some(v.len()))?;
                for item in v {
                    out.serialize_element(item)?;
                }
                out.end()
            }
            Self::Object(v) => {
                let mut out = s.serialize_map(Some(v.len()))?;
                for (k, item) in v {
                    out.serialize_entry(k, item)?;
                }
                out.end()
            }
        }
    }
}
impl ResultValue {
    pub fn number(v: f64) -> Self {
        Number::from_f64(v).map(Self::Number).unwrap_or(Self::Null)
    }
    pub fn integer(v: usize) -> Self {
        Self::Number(Number::from(v as u64))
    }
    pub fn string(v: &str) -> Result<Self, AnswerError> {
        Ok(Self::String(result_storage::string(v)?))
    }
    pub fn fields<const N: usize>(fields: [(&str, Self); N]) -> Result<Self, AnswerError> {
        let mut out = Self::object(N)?;
        for (k, v) in fields {
            out.put(k, v)?;
        }
        Ok(out)
    }
    pub fn object(capacity: usize) -> Result<Self, AnswerError> {
        Ok(Self::Object(result_storage::vector(capacity)?))
    }
    pub fn put(&mut self, key: &str, value: Self) -> Result<(), AnswerError> {
        let Self::Object(entries) = self else {
            return Err("Expected response object".into());
        };
        if let Some((_, stored)) = entries.iter_mut().find(|(k, _)| k == key) {
            *stored = value;
        } else {
            // Preserve serde_json's insertion order and last-value semantics.
            if entries.len() == entries.capacity() {
                return Err(BackendError::Overloaded(
                    "Derived response object capacity exceeded".into(),
                )
                .into());
            }
            entries.push((result_storage::string(key)?, value));
        }
        Ok(())
    }
    pub fn copy(v: &Value) -> Result<Self, AnswerError> {
        Ok(match v {
            Value::Null => Self::Null,
            Value::Bool(v) => Self::Bool(*v),
            Value::Number(v) => Self::Number(v.clone()),
            Value::String(v) => Self::string(v)?,
            Value::Array(values) => {
                let mut out = result_storage::vector(values.len())?;
                for v in values {
                    out.push(Self::copy(v)?);
                }
                Self::Array(out)
            }
            Value::Object(values) => {
                let mut out = Self::object(values.len())?;
                for (k, v) in values {
                    out.put(k, Self::copy(v)?)?;
                }
                out
            }
        })
    }
    pub fn probabilities<'a>(
        labels: impl ExactSizeIterator<Item = &'a str>,
        p: &[f64],
    ) -> Result<Self, AnswerError> {
        let mut out = Self::object(labels.len())?;
        for (label, p) in labels.zip(p) {
            out.put(label, Self::number(*p))?;
        }
        Ok(out)
    }
    pub fn legend(values: &[Value]) -> Result<Self, AnswerError> {
        let mut out = Self::object(values.len())?;
        for (i, v) in values.iter().enumerate() {
            let key = result_storage::index(i)?;
            out.put(&key, Self::copy(v)?)?;
        }
        Ok(out)
    }
    pub fn sort_objects(&mut self) {
        match self {
            Self::Object(entries) => {
                entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                for (_, v) in entries {
                    v.sort_objects();
                }
            }
            Self::Array(values) => {
                for v in values {
                    v.sort_objects();
                }
            }
            _ => {}
        }
    }
    pub fn json_text(&self) -> Result<String, AnswerError> {
        struct Writer(Vec<u8>);
        impl std::io::Write for Writer {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                let needed = self
                    .0
                    .len()
                    .checked_add(bytes.len())
                    .ok_or(std::io::ErrorKind::OutOfMemory)?;
                if needed > self.0.capacity() {
                    let target = needed.max(self.0.capacity().saturating_mul(2)).max(64);
                    // Reserve the full new target while the old allocation is live.
                    infinity_v3_backend::result_memory::reserve_for_request(target)
                        .map_err(|_| std::io::ErrorKind::OutOfMemory)?;
                    self.0
                        .try_reserve_exact(target - self.0.len())
                        .map_err(|_| std::io::ErrorKind::OutOfMemory)?;
                }
                self.0.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut writer = Writer(Vec::new());
        serde_json::to_writer(&mut writer, self)
            .map_err(|_| BackendError::Overloaded("Decision legend allocation failed".into()))?;
        Ok(String::from_utf8(writer.0).expect("JSON is UTF-8"))
    }
}
#[cfg(test)]
impl PartialEq<Value> for ResultValue {
    fn eq(&self, other: &Value) -> bool {
        serde_json::to_value(self).unwrap() == *other
    }
}

#[cfg(test)]
impl std::ops::Index<&str> for ResultValue {
    type Output = Self;
    fn index(&self, key: &str) -> &Self {
        static NULL: ResultValue = ResultValue::Null;
        match self {
            Self::Object(entries) => entries
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .unwrap_or(&NULL),
            _ => &NULL,
        }
    }
}
#[cfg(test)]
impl ResultValue {
    pub(super) fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => n.as_f64(),
            _ => None,
        }
    }
}
#[cfg(test)]
impl PartialEq<&str> for ResultValue {
    fn eq(&self, other: &&str) -> bool {
        matches!(self,Self::String(v)if v==other)
    }
}
#[cfg(test)]
impl PartialEq<f64> for ResultValue {
    fn eq(&self, other: &f64) -> bool {
        self.as_f64() == Some(*other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infinity_v3_backend::result_memory::{RequestResults, ResultPool, REQUEST_RESULTS};
    use std::sync::Arc;
    #[tokio::test]
    async fn derived_response_pressure_retains_credits_and_preserves_maps() {
        let pool = ResultPool::new(16_384).unwrap();
        let source = pool.reserve(8).unwrap();
        let owner = Arc::new(RequestResults::default());
        owner.retain(Some(source.clone()));
        let blocked = pool.reserve(16_376).unwrap();
        let expected = serde_json::json!({"z":[true,null,{"text":"é\n"}],"a":3});
        let rejected = REQUEST_RESULTS
            .scope(owner.clone(), async { ResultValue::copy(&expected) })
            .await;
        assert!(matches!(
            rejected,
            Err(AnswerError::Memory(BackendError::Overloaded(_)))
        ));
        assert_eq!(pool.used(), 16_384);
        drop(blocked);
        let response = REQUEST_RESULTS
            .scope(owner.clone(), async {
                let copy = ResultValue::copy(&expected).unwrap();
                let mut object = ResultValue::object(3).unwrap();
                object.put("z", ResultValue::number(1.)).unwrap();
                object.put("a", copy).unwrap();
                object.put("z", ResultValue::number(2.)).unwrap();
                object
            })
            .await;
        let expected = serde_json::json!({"z":2.,"a":expected});
        assert_eq!(serde_json::to_value(&response).unwrap(), expected);
        assert_eq!(
            serde_json::to_string(&response).unwrap(),
            serde_json::to_string(&expected).unwrap()
        );
        let packed = rmp_serde::to_vec_named(&response).unwrap();
        assert_eq!(rmp_serde::from_slice::<Value>(&packed).unwrap(), expected);
        let body_clone = owner.clone();
        drop(source);
        drop(owner);
        drop(response);
        assert!(pool.used() > 8);
        drop(body_clone);
        assert_eq!(pool.used(), 0);
        assert!(pool.reserve(16_384).is_ok());
    }
}
