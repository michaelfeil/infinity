//! Validated safetensors headers and the default native non-GDS disk reader.
use crate::{native_error, start, BackendError};
use safetensors::{tensor::Metadata, Dtype};
use std::{
    ffi::{c_char, c_void, CString},
    fs::File,
    io::Read,
    os::fd::AsRawFd,
    path::Path,
    ptr::NonNull,
};

const THREADS: &str = infinity_v3_config::infinity_env!("TORCH_WEIGHT_READER_THREADS");
const BUFFER: &str = infinity_v3_config::infinity_env!("TORCH_WEIGHT_READER_BUFFER_BYTES");
const DIRECT: &str = infinity_v3_config::infinity_env!("TORCH_WEIGHT_READER_DIRECT_IO");

#[derive(Debug, Clone, Copy)]
struct Options {
    threads: usize,
    buffer_bytes: usize,
    direct: bool,
    buffer_override: bool,
}
impl Options {
    fn parse(
        checkpoint_bytes: u64,
        threads: Option<&str>,
        buffer: Option<&str>,
        direct: Option<&str>,
    ) -> Result<Self, String> {
        let threads = threads
            .unwrap_or("16")
            .parse::<usize>()
            .map_err(|e| format!("Invalid {THREADS}: {e}"))?;
        let buffer_bytes = buffer
            .map(str::parse::<usize>)
            .transpose()
            .map_err(|e| format!("Invalid {BUFFER}: {e}"))?
            .unwrap_or_else(|| Self::default_buffer(checkpoint_bytes));
        let direct = match direct.unwrap_or("true").to_ascii_lowercase().as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => return Err(format!("{DIRECT} must be true, false, 1 or 0")),
        };
        if !(1..=64).contains(&threads) {
            return Err(format!("{THREADS} must be between 1 and 64"));
        }
        if !(65536..=1 << 30).contains(&buffer_bytes) {
            return Err(format!("{BUFFER} must be between 65536 and 1073741824"));
        }
        Ok(Self {
            threads,
            buffer_bytes,
            direct,
            buffer_override: buffer.is_some(),
        })
    }

    fn default_buffer(checkpoint_bytes: u64) -> usize {
        if checkpoint_bytes > 32 << 30 {
            256 << 20
        } else {
            32 << 20
        }
    }

    fn for_checkpoint(mut self, checkpoint_bytes: u64) -> Self {
        if !self.buffer_override {
            self.buffer_bytes = Self::default_buffer(checkpoint_bytes);
        }
        self
    }

    fn from_env(checkpoint_bytes: u64) -> Result<Self, String> {
        // Cache explicit overrides, not the first model's adaptive default.
        static ENV: std::sync::OnceLock<Result<Options, String>> = std::sync::OnceLock::new();
        ENV.get_or_init(|| {
            let read = |key: &str| match std::env::var(key) {
                Ok(value) => Ok(Some(value)),
                Err(std::env::VarError::NotPresent) => Ok(None),
                Err(error) => Err(format!("Invalid {key}: {error}")),
            };
            Self::parse(
                0,
                read(THREADS)?.as_deref(),
                read(BUFFER)?.as_deref(),
                read(DIRECT)?.as_deref(),
            )
        })
        .as_ref()
        .map(|&options| options.for_checkpoint(checkpoint_bytes))
        .map_err(Clone::clone)
    }
}

#[repr(C)]
struct NativeWeight {
    name: *const c_char,
    shape: *const i64,
    rank: usize,
    dtype: i32,
    offset: u64,
    length: u64,
}

extern "C" {
    fn tei_load_safetensors(
        handle: *mut c_void,
        fd: i32,
        start: u64,
        length: u64,
        weights: *const NativeWeight,
        count: usize,
        threads: usize,
        buffer_bytes: usize,
        direct_io: i32,
    ) -> i32;
}

fn read_header(file: &mut File) -> Result<(u64, Metadata), String> {
    let info = file.metadata().map_err(|e| e.to_string())?;
    if !info.is_file() {
        return Err("Safetensors input must be a regular file".into());
    }
    let mut size = [0u8; 8];
    file.read_exact(&mut size)
        .map_err(|e| format!("Invalid safetensors header: {e}"))?;
    let size = u64::from_le_bytes(size);
    // Same limit and complete-buffer check as SafeTensors::read_metadata. Only
    // the header is read here; tensor data is read by the native worker pool.
    if size > 100_000_000 {
        return Err("Safetensors header exceeds 100000000 bytes".into());
    }
    let data_start = size.checked_add(8).ok_or("Safetensors header overflow")?;
    if data_start > info.len() {
        return Err("Safetensors header extends past end of file".into());
    }
    let mut header = vec![0; usize::try_from(size).map_err(|e| e.to_string())?];
    file.read_exact(&mut header).map_err(|e| e.to_string())?;
    // Metadata's Deserialize validates contiguous offsets, shape products,
    // dtype bit widths, and each tensor's exact byte length.
    let metadata: Metadata = serde_json::from_slice(&header).map_err(|e| e.to_string())?;
    if data_start.checked_add(metadata.data_len() as u64) != Some(info.len()) {
        return Err("Safetensors data length does not match its header".into());
    }
    Ok((data_start, metadata))
}

pub(super) fn load(
    handle: NonNull<c_void>,
    path: &Path,
    prefix: &str,
    checkpoint_bytes: u64,
) -> Result<(), BackendError> {
    let load = || -> Result<(), String> {
        let started = std::time::Instant::now();
        let options = Options::from_env(checkpoint_bytes)?;
        let mut file = File::open(path).map_err(|e| e.to_string())?;
        let (start, metadata) = read_header(&mut file)?;
        let mut names = Vec::new();
        let mut shapes = Vec::new();
        let mut ranges = Vec::new();
        for name in metadata.offset_keys() {
            if name.ends_with("position_ids") || name.ends_with("token_type_ids") {
                continue;
            }
            let tensor = metadata.info(&name).expect("metadata key exists");
            let dtype = match tensor.dtype {
                Dtype::F32 => 0,
                Dtype::F16 => 1,
                Dtype::BF16 => 2,
                other => return Err(format!("Unsupported weight dtype {other:?}")),
            };
            names.push(CString::new(format!("{prefix}{name}")).map_err(|e| e.to_string())?);
            shapes.push(
                tensor
                    .shape
                    .iter()
                    .map(|&n| i64::try_from(n).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            ranges.push((dtype, tensor.data_offsets));
        }
        let weights = names
            .iter()
            .zip(&shapes)
            .zip(ranges)
            .map(|((name, shape), (dtype, (start, end)))| NativeWeight {
                name: name.as_ptr(),
                shape: shape.as_ptr(),
                rank: shape.len(),
                dtype,
                offset: start as u64,
                length: (end - start) as u64,
            })
            .collect::<Vec<_>>();
        // SAFETY: shapes/ranges were checked by safetensors and converted to the
        // native index type. Strings, descriptors and the opened file outlive the
        // call. Native code drains every read/DMA before returning, including on
        // errors, and stores tensors that own their final allocations.
        let status = unsafe {
            tei_load_safetensors(
                handle.as_ptr(),
                file.as_raw_fd(),
                start,
                metadata.data_len() as u64,
                weights.as_ptr(),
                weights.len(),
                options.threads,
                options.buffer_bytes,
                i32::from(options.direct),
            )
        };
        if status != 0 {
            return Err(native_error());
        }
        tracing::info!(path = %path.display(), tensors = weights.len(), bytes = metadata.data_len(),
                       threads_max = options.threads, buffer_bytes = options.buffer_bytes,
                       seconds = started.elapsed().as_secs_f64(),
                       direct_io_requested = options.direct, "Loaded weights with fastsafetensors non-GDS reader");
        Ok(())
    };
    load().map_err(|e| start(format!("Could not load {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, Write};

    fn fixture(data: &[u8]) -> File {
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "infinity-weight-header-{}-{time}",
            std::process::id()
        ));
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        std::fs::remove_file(path).unwrap();
        file.write_all(data).unwrap();
        file.rewind().unwrap();
        file
    }

    fn encoded(header: &str, bytes: usize) -> Vec<u8> {
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend_from_slice(header.as_bytes());
        data.resize(data.len() + bytes, 0);
        data
    }

    #[test]
    fn header_validation_matches_safetensors_without_reading_the_payload() {
        let valid = [
            encoded(r#"{}"#, 0),
            encoded(
                r#"{"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#,
                4,
            ),
            encoded(
                r#"{"empty":{"dtype":"F16","shape":[0],"data_offsets":[0,0]}}"#,
                0,
            ),
            // Offset 2 is valid safetensors even though an F32 view needs 4-byte alignment.
            encoded(
                r#"{"half":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"float":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}"#,
                6,
            ),
        ];
        for bytes in valid {
            assert!(safetensors::SafeTensors::deserialize(&bytes).is_ok());
            let mut file = fixture(&bytes);
            let (offset, _) = read_header(&mut file).unwrap();
            assert_eq!(file.stream_position().unwrap(), offset);
            assert_eq!(
                offset,
                8 + u64::from_le_bytes(bytes[..8].try_into().unwrap())
            );
        }
        let invalid = [
            vec![],
            vec![0; 7],
            u64::MAX.to_le_bytes().to_vec(),
            encoded("{", 0),
            encoded("{}", 1),
            encoded(
                r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
                4,
            ),
            encoded(
                r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
                8,
            ),
            encoded(
                r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                3,
            ),
            encoded(
                r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                5,
            ),
            encoded(
                r#"{"w":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,4]}}"#,
                4,
            ),
            encoded(
                r#"{"w":{"dtype":"UNKNOWN","shape":[1],"data_offsets":[0,4]}}"#,
                4,
            ),
            encoded(
                r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
                4,
            ),
        ];
        for bytes in invalid {
            assert!(safetensors::SafeTensors::deserialize(&bytes).is_err());
            assert!(read_header(&mut fixture(&bytes)).is_err());
        }
    }

    #[test]
    fn adaptive_default_uses_total_checkpoint_size_and_respects_overrides() {
        for (bytes, expected) in [
            (0, 32 << 20),
            (32 << 30, 32 << 20),
            ((32 << 30) + 1, 256 << 20),
        ] {
            assert_eq!(
                Options::parse(bytes, None, None, None)
                    .unwrap()
                    .buffer_bytes,
                expected
            );
            assert_eq!(
                Options::parse(bytes, None, Some("16777216"), None)
                    .unwrap()
                    .buffer_bytes,
                16 << 20
            );
        }
    }

    #[test]
    fn environment_is_read_once_without_freezing_adaptive_defaults() {
        const CHILD: &str = "INFINITY_TEST_WEIGHT_READER_ENV_CHILD";
        let Ok(mode) = std::env::var(CHILD) else {
            // Isolate env mutation and OnceLock initialization from other tests.
            for mode in ["adaptive", "override"] {
                let mut child = std::process::Command::new(std::env::current_exe().unwrap());
                child.args(["--exact", "weight_reader::tests::environment_is_read_once_without_freezing_adaptive_defaults"])
                    .env(CHILD, mode).env(THREADS, "4").env(DIRECT, "0").env_remove(BUFFER);
                if mode == "override" {
                    child.env(BUFFER, "33554432");
                }
                assert!(child.status().unwrap().success());
            }
            return;
        };
        assert_eq!(Options::from_env(0).unwrap().buffer_bytes, 32 << 20);
        for key in [THREADS, BUFFER, DIRECT] {
            std::env::set_var(key, "invalid");
        }
        let large = Options::from_env(u64::MAX).unwrap();
        assert_eq!(
            large.buffer_bytes,
            if mode == "adaptive" {
                256 << 20
            } else {
                32 << 20
            }
        );
        assert_eq!(large.threads, 4);
        assert!(!large.direct);
        assert_eq!(Options::from_env(0).unwrap().buffer_bytes, 32 << 20);
    }

    #[test]
    fn options_are_bounded_and_accept_buffered_mode() {
        let options = Options::parse(0, None, None, None).unwrap();
        assert_eq!(options.threads, 16);
        assert_eq!(options.buffer_bytes, 32 << 20);
        for bytes in [16 << 20, 64 << 20, 256 << 20] {
            assert_eq!(
                Options::parse(0, None, Some(&bytes.to_string()), None)
                    .unwrap()
                    .buffer_bytes,
                bytes
            );
        }
        assert!(options.direct);
        assert!(!Options::parse(0, None, None, Some("0")).unwrap().direct);
        assert!(!Options::parse(0, None, None, Some("false")).unwrap().direct);
        for value in ["0", "65", "-1", "nan"] {
            assert!(Options::parse(0, Some(value), None, None).is_err());
        }
        for value in ["0", "65535", "1073741825", "-1"] {
            assert!(Options::parse(0, None, Some(value), None).is_err());
        }
        assert!(Options::parse(0, None, None, Some("sometimes")).is_err());
    }
}
