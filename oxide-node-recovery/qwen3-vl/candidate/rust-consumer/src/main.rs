//! Real Qwen3-VL Native b66 CAPI consumer.
//!
//! Loads the exact b66 library, applies the Native production packet, loads all
//! 749 weights, and executes real processor fixtures through tei_forward and
//! tei_forward_submit. It records graph and prefix-cache observations; it never
//! claims a pass without the returned counters and output files.

use libloading::Library;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::fd::AsRawFd;
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};

#[repr(C)]
#[derive(Clone, Copy, Debug, Deserialize)]
struct TeiConfig {
    hidden: i64,
    heads: i64,
    layers: i64,
    intermediate: i64,
    vocab: i64,
    positions: i64,
    types: i64,
    epsilon: f64,
    activation: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct TeiImage {
    pixels: *const f32,
    rows: i64,
    patch_dim: i64,
    grid: [i64; 3],
    merge_size: i64,
    token_start: i64,
    token_count: i64,
    sequence_start: i64,
    feature_key: [u8; 32],
    has_feature_key: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct TeiWeight {
    name: *const c_char,
    shape: *const i64,
    rank: usize,
    dtype: i32,
    offset: u64,
    length: u64,
}

type TeiCreate = unsafe extern "C" fn(*const TeiConfig, *const c_char, i32) -> *mut c_void;
type TeiDevicePreflight = unsafe extern "C" fn(*const c_char) -> i32;
type TeiOption = unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> i32;
type TeiLoadSafetensors = unsafe extern "C" fn(
    *mut c_void,
    i32,
    u64,
    u64,
    *const TeiWeight,
    usize,
    usize,
    usize,
    i32,
) -> i32;
type TeiReady = unsafe extern "C" fn(*mut c_void) -> i32;
type TeiForward = unsafe extern "C" fn(
    *mut c_void,
    *const i64,
    *const i64,
    *const i64,
    *const i32,
    i64,
    i64,
    i32,
    *const i64,
    usize,
    *const i64,
    usize,
    *mut f32,
    usize,
    *const i64,
    *const TeiImage,
    usize,
    *const c_void,
    usize,
) -> i32;
type TeiForwardSubmit = unsafe extern "C" fn(
    *mut c_void,
    *const i64,
    *const i64,
    *const i64,
    *const i32,
    i64,
    i64,
    i32,
    *const i64,
    usize,
    *const i64,
    usize,
    usize,
    *const i64,
    *const TeiImage,
    usize,
    *const c_void,
    usize,
    *const c_void,
) -> *mut c_void;
type TeiOutputWait = unsafe extern "C" fn(*mut c_void) -> *const f32;
type TeiOutputDestroy = unsafe extern "C" fn(*mut c_void);
type TeiGraphCount = unsafe extern "C" fn(*mut c_void) -> i64;
type TeiGraphStats = unsafe extern "C" fn(*mut c_void, *mut u64, usize) -> i32;
type TeiPrefixSupported = unsafe extern "C" fn(*mut c_void, i32) -> i32;
type TeiPrefixStats = unsafe extern "C" fn(*mut c_void, *mut u64, usize) -> i32;
type TeiPrefixClear = unsafe extern "C" fn(*mut c_void) -> i32;
type TeiOutputWidth = unsafe extern "C" fn(*mut c_void) -> i64;
type TeiPooledWidth = unsafe extern "C" fn(*mut c_void) -> i64;
type TeiClassWidth = unsafe extern "C" fn(*mut c_void) -> i64;
type TeiDestroy = unsafe extern "C" fn(*mut c_void);
type TeiError = unsafe extern "C" fn() -> *const c_char;
type TeiLastErrorKind = unsafe extern "C" fn() -> i32;

struct Api {
    create: TeiCreate,
    device_preflight: TeiDevicePreflight,
    option: TeiOption,
    load_safetensors: TeiLoadSafetensors,
    ready: TeiReady,
    forward: TeiForward,
    forward_submit: TeiForwardSubmit,
    output_wait: TeiOutputWait,
    output_destroy: TeiOutputDestroy,
    graph_count: TeiGraphCount,
    graph_stats: TeiGraphStats,
    prefix_supported: TeiPrefixSupported,
    prefix_stats: TeiPrefixStats,
    prefix_clear: TeiPrefixClear,
    output_width: TeiOutputWidth,
    pooled_width: TeiPooledWidth,
    class_width: TeiClassWidth,
    destroy: TeiDestroy,
    error: TeiError,
    last_error_kind: TeiLastErrorKind,
}

impl Api {
    unsafe fn load(path: &Path) -> Result<(Library, Self), String> {
        let library = Library::new(path).map_err(|error| format!("load {}: {error}", path.display()))?;
        macro_rules! sym {
            ($name:literal) => {
                *library
                    .get(concat!($name, "\0").as_bytes())
                    .map_err(|error| format!("symbol {}: {error}", $name))?
            };
        }
        let api = Self {
            create: sym!("tei_create"),
            device_preflight: sym!("tei_device_preflight"),
            option: sym!("tei_option"),
            load_safetensors: sym!("tei_load_safetensors"),
            ready: sym!("tei_ready"),
            forward: sym!("tei_forward"),
            forward_submit: sym!("tei_forward_submit"),
            output_wait: sym!("tei_output_wait"),
            output_destroy: sym!("tei_output_destroy"),
            graph_count: sym!("tei_graph_count"),
            graph_stats: sym!("tei_graph_stats"),
            prefix_supported: sym!("tei_prefix_cache_supported"),
            prefix_stats: sym!("tei_prefix_cache_stats"),
            prefix_clear: sym!("tei_prefix_cache_clear"),
            output_width: sym!("tei_output_width"),
            pooled_width: sym!("tei_pooled_width"),
            class_width: sym!("tei_classification_width"),
            destroy: sym!("tei_destroy"),
            error: sym!("tei_error"),
            last_error_kind: sym!("tei_last_error_kind"),
        };
        Ok((library, api))
    }
}

#[derive(Deserialize)]
struct Packet {
    checkpoint: String,
    original_config: serde_json::Value,
    config_options: BTreeMap<String, String>,
    runtime_profile: String,
    runtime_options: BTreeMap<String, String>,
    tei_config: TeiConfig,
    inventory: Inventory,
    weights: Vec<Weight>,
    sdk: Sdk,
}

#[derive(Deserialize)]
struct Inventory {
    shards: Vec<Shard>,
}

#[derive(Deserialize)]
struct Shard {
    shard: String,
    data_start: u64,
    data_length: u64,
}

#[derive(Deserialize)]
struct Weight {
    original_name: String,
    shard: String,
    dtype: String,
    shape: Vec<i64>,
    source_offset: u64,
    source_length: u64,
}

#[derive(Deserialize)]
struct Sdk {
    expected_sha256_full: Option<String>,
}

#[derive(Deserialize)]
struct Fixtures {
    fixtures: Vec<Fixture>,
}

#[derive(Deserialize)]
struct Fixture {
    name: String,
    ids: Vec<u32>,
    types: Vec<u32>,
    positions: Vec<u32>,
    cumulative: Vec<u32>,
    media_positions: Option<[Vec<u32>; 3]>,
    #[serde(default)]
    images: Vec<FixtureImage>,
    pool: String,
}

#[derive(Deserialize)]
struct FixtureImage {
    pixels: String,
    rows: usize,
    patch_dim: usize,
    grid_thw: [usize; 3],
    merge_size: usize,
    token_start: usize,
    token_count: usize,
}

fn sha256(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

unsafe fn error_text(api: &Api) -> String {
    let kind = (api.last_error_kind)();
    let pointer = (api.error)();
    let text = if pointer.is_null() {
        "no error text".to_owned()
    } else {
        CStr::from_ptr(pointer).to_string_lossy().into_owned()
    };
    format!("kind={kind} {text}")
}

struct Handle {
    api: *const Api,
    handle: *mut c_void,
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { ((*self.api).destroy)(self.handle) };
    }
}

struct FixtureInputs {
    ids: Vec<i64>,
    types: Vec<i64>,
    positions: Vec<i64>,
    cumulative: Vec<i32>,
    media: Option<Vec<i64>>,
    pixels: Vec<Vec<f32>>,
    images: Vec<FixtureImage>,
}

impl FixtureInputs {
    fn load(fixture: &Fixture) -> Result<Self, String> {
        let mut pixels = Vec::with_capacity(fixture.images.len());
        for image in &fixture.images {
            let bytes = std::fs::read(&image.pixels)
                .map_err(|error| format!("{}: {error}", image.pixels))?;
            if bytes.len() % 4 != 0 {
                return Err(format!("{} is not F32", image.pixels));
            }
            let values = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            if values.len() != image.rows * image.patch_dim {
                return Err(format!("{} pixel length mismatch", image.pixels));
            }
            pixels.push(values);
        }
        let ids: Vec<i64> = fixture.ids.iter().map(|value| *value as i64).collect();
        let types: Vec<i64> = fixture.types.iter().map(|value| *value as i64).collect();
        let positions: Vec<i64> = fixture.positions.iter().map(|value| *value as i64).collect();
        let cumulative: Vec<i32> = fixture
            .cumulative
            .iter()
            .map(|value| i32::try_from(*value).map_err(|_| "cumulative overflow".to_owned()))
            .collect::<Result<Vec<_>, _>>()?;
        if cumulative.len() < 2
            || cumulative[0] != 0
            || cumulative.windows(2).any(|pair| pair[0] > pair[1])
            || cumulative.last().copied() != Some(ids.len() as i32)
            || types.len() != ids.len()
            || positions.len() != ids.len()
        {
            return Err(format!("{} has invalid packed token metadata", fixture.name));
        }
        let media = fixture.media_positions.as_ref().map(|axes| {
            axes.iter()
                .flat_map(|axis| axis.iter().map(|value| *value as i64))
                .collect::<Vec<_>>()
        });
        if let Some(media) = media.as_ref() {
            if media.len() != ids.len() * 3 {
                return Err(format!("{} media positions are not 3*tokens", fixture.name));
            }
        }
        let mut spans = Vec::new();
        for image in &fixture.images {
            let patches = image.grid_thw.iter().product::<usize>();
            let merge = image.merge_size * image.merge_size;
            if image.grid_thw.iter().any(|dim| *dim == 0)
                || image.merge_size == 0
                || image.grid_thw[1] % image.merge_size != 0
                || image.grid_thw[2] % image.merge_size != 0
                || patches != image.rows
                || patches / merge != image.token_count
                || image.token_start + image.token_count > ids.len()
            {
                return Err(format!("{} has invalid image geometry", fixture.name));
            }
            spans.push((image.token_start, image.token_start + image.token_count));
        }
        spans.sort_unstable();
        if spans.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(format!("{} has overlapping image spans", fixture.name));
        }
        Ok(Self {
            ids,
            types,
            positions,
            cumulative,
            media,
            pixels,
            images: fixture.images.iter().map(clone_image).collect(),
        })
    }

    fn tei_images(&self) -> Vec<TeiImage> {
        self.images
            .iter()
            .zip(self.pixels.iter())
            .map(|(image, pixels)| TeiImage {
                pixels: pixels.as_ptr(),
                rows: image.rows as i64,
                patch_dim: image.patch_dim as i64,
                grid: [
                    image.grid_thw[0] as i64,
                    image.grid_thw[1] as i64,
                    image.grid_thw[2] as i64,
                ],
                merge_size: image.merge_size as i64,
                token_start: image.token_start as i64,
                token_count: image.token_count as i64,
                sequence_start: sequence_start(&self.cumulative, image.token_start) as i64,
                feature_key: [0u8; 32],
                has_feature_key: 0,
            })
            .collect()
    }

    fn max_sequence(&self) -> i64 {
        self.cumulative
            .windows(2)
            .map(|pair| i64::from(pair[1] - pair[0]))
            .max()
            .unwrap_or(0)
    }
}

fn clone_image(image: &FixtureImage) -> FixtureImage {
    FixtureImage {
        pixels: image.pixels.clone(),
        rows: image.rows,
        patch_dim: image.patch_dim,
        grid_thw: image.grid_thw,
        merge_size: image.merge_size,
        token_start: image.token_start,
        token_count: image.token_count,
    }
}

fn sequence_start(cumulative: &[i32], token_start: usize) -> usize {
    cumulative
        .windows(2)
        .position(|pair| pair[0] as usize <= token_start && token_start < pair[1] as usize)
        .map(|index| cumulative[index] as usize)
        .unwrap_or(0)
}

fn pool_code(name: &str) -> Result<i32, String> {
    match name {
        "cls" => Ok(0),
        "mean" => Ok(1),
        "last_token" => Ok(2),
        _ => Err(format!("unsupported pool {name}")),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let packet_path = PathBuf::from(args.next().ok_or("usage: consumer PACKET FIXTURES LIBRARY OUT [--library-sha HEX] [--prefix-bytes N]")?);
    let fixtures_path = PathBuf::from(args.next().ok_or("missing fixtures")?);
    let library_path = PathBuf::from(args.next().ok_or("missing library")?);
    let output = PathBuf::from(args.next().ok_or("missing output")?);
    let mut expected = None;
    let mut prefix_bytes = 0usize;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--library-sha" => expected = args.next(),
            "--prefix-bytes" => {
                prefix_bytes = args
                    .next()
                    .ok_or("missing prefix bytes")?
                    .parse()
                    .map_err(|_| "invalid prefix bytes")?
            }
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    std::fs::create_dir_all(&output)?;
    let packet: Packet = serde_json::from_slice(&std::fs::read(&packet_path)?)?;
    let fixtures: Fixtures = serde_json::from_slice(&std::fs::read(&fixtures_path)?)?;
    let expected = expected
        .or_else(|| packet.sdk.expected_sha256_full.clone())
        .ok_or("packet has no full SDK SHA-256; pass --library-sha")?;
    let actual = sha256(&library_path)?;
    if actual != expected {
        return Err(format!(
            "library SHA-256 mismatch: expected {expected}, got {actual} (CPU or stale b66 builds are rejected)"
        )
        .into());
    }
    let (_library, api) = unsafe { Api::load(&library_path)? };
    let api_ref: &Api = &api;
    if unsafe { (api.device_preflight)(CString::new("cuda")?.as_ptr()) } != 0 {
        return Err(format!("device preflight failed: {}", unsafe { error_text(api_ref) }).into());
    }
    let handle = unsafe { (api.create)(&packet.tei_config, CString::new("cuda")?.as_ptr(), 2) };
    if handle.is_null() {
        return Err(format!("tei_create failed: {}", unsafe { error_text(api_ref) }).into());
    }
    let guard = Handle {
        api: api_ref,
        handle,
    };
    let mut options = packet.config_options.clone();
    options.extend(packet.runtime_options.clone());
    if prefix_bytes > 0 {
        options.insert("_prefix_cache_max_bytes".to_owned(), prefix_bytes.to_string());
    }
    for (key, value) in &options {
        let key = CString::new(key.as_str())?;
        let value = CString::new(value.as_str())?;
        if unsafe { (api.option)(guard.handle, key.as_ptr(), value.as_ptr()) } != 0 {
            return Err(format!("option {key:?} failed: {}", unsafe { error_text(api_ref) }).into());
        }
    }
    let root = Path::new(&packet.checkpoint);
    let config_file: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
    if packet.original_config != config_file {
        return Err("packet original_config differs from the checkpoint config.json".into());
    }
    for shard in &packet.inventory.shards {
        let mut names = Vec::new();
        let mut shapes = Vec::new();
        let mut weights = Vec::new();
        for weight in packet.weights.iter().filter(|weight| weight.shard == shard.shard) {
            names.push(CString::new(weight.original_name.as_str())?);
            shapes.push(weight.shape.clone());
            weights.push(TeiWeight {
                name: std::ptr::null(),
                shape: std::ptr::null(),
                rank: weight.shape.len(),
                dtype: if weight.dtype == "BF16" { 2 } else { 1 },
                offset: weight.source_offset,
                length: weight.source_length,
            });
        }
        for (index, weight) in weights.iter_mut().enumerate() {
            weight.name = names[index].as_ptr();
            weight.shape = shapes[index].as_ptr();
        }
        let file = std::fs::File::open(root.join(&shard.shard))?;
        if unsafe {
            (api.load_safetensors)(
                guard.handle,
                file.as_raw_fd(),
                shard.data_start,
                shard.data_length,
                weights.as_ptr(),
                weights.len(),
                4,
                64 << 20,
                0,
            )
        } != 0
        {
            return Err(format!("load {} failed: {}", shard.shard, unsafe { error_text(api_ref) }).into());
        }
    }
    if unsafe { (api.ready)(guard.handle) } != 0 {
        return Err(format!("tei_ready failed: {}", unsafe { error_text(api_ref) }).into());
    }
    let output_width = unsafe { (api.output_width)(guard.handle) };
    let pooled_width = unsafe { (api.pooled_width)(guard.handle) };
    let classification_width = unsafe { (api.class_width)(guard.handle) };
    if output_width != 4096 || pooled_width != 4096 || classification_width != 0 {
        return Err(format!(
            "unexpected widths output={output_width} pooled={pooled_width} class={classification_width}"
        )
        .into());
    }
    let mut observations = Vec::new();
    for fixture in &fixtures.fixtures {
        let inputs = FixtureInputs::load(fixture)?;
        let images = inputs.tei_images();
        let media = inputs.media.as_deref();
        let max_sequence = inputs.max_sequence();
        let pool = pool_code(&fixture.pool)?;
        let sequences = inputs.cumulative.len() - 1;
        let raw_indices: Vec<i64> = (0..sequences as i64).collect();
        let raw_rows: usize = inputs
            .cumulative
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as usize)
            .sum();
        let pooled_indices: Vec<i64> = (0..sequences as i64).collect();
        let capacity = pooled_indices.len() * pooled_width as usize + raw_rows * output_width as usize;
        let mut output_buffer = vec![0f32; capacity];
        let run = |output: &mut [f32]| unsafe {
            (api.forward)(
                guard.handle,
                inputs.ids.as_ptr(),
                inputs.types.as_ptr(),
                inputs.positions.as_ptr(),
                inputs.cumulative.as_ptr(),
                sequences as i64,
                max_sequence,
                pool,
                pooled_indices.as_ptr(),
                pooled_indices.len(),
                raw_indices.as_ptr(),
                raw_indices.len(),
                output.as_mut_ptr(),
                output.len(),
                media.map_or(std::ptr::null(), |values| values.as_ptr()),
                images.as_ptr(),
                images.len(),
                std::ptr::null(),
                0,
            )
        };
        let runs = if packet.runtime_profile == "native_graph_evidence" { 5 } else { 1 };
        for _ in 0..runs {
            if run(&mut output_buffer) != 0 {
                return Err(format!("fixture {} forward failed: {}", fixture.name, unsafe { error_text(api_ref) }).into());
            }
        }
        let mut graph_stats = [0u64; 9];
        if unsafe { (api.graph_stats)(guard.handle, graph_stats.as_mut_ptr(), graph_stats.len()) } != 0 {
            return Err(format!("graph stats failed: {}", unsafe { error_text(api_ref) }).into());
        }
        let graph_count = unsafe { (api.graph_count)(guard.handle) };
        // Submit path coverage.
        let submit_capacity = capacity;
        let lease = unsafe {
            (api.forward_submit)(
                guard.handle,
                inputs.ids.as_ptr(),
                inputs.types.as_ptr(),
                inputs.positions.as_ptr(),
                inputs.cumulative.as_ptr(),
                sequences as i64,
                max_sequence,
                pool,
                pooled_indices.as_ptr(),
                pooled_indices.len(),
                raw_indices.as_ptr(),
                raw_indices.len(),
                submit_capacity,
                media.map_or(std::ptr::null(), |values| values.as_ptr()),
                images.as_ptr(),
                images.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
            )
        };
        if lease.is_null() {
            return Err(format!("fixture {} submit failed: {}", fixture.name, unsafe { error_text(api_ref) }).into());
        }
        let pointer = unsafe { (api.output_wait)(lease) };
        if pointer.is_null() {
            unsafe { (api.output_destroy)(lease) };
            return Err(format!("fixture {} submit wait failed: {}", fixture.name, unsafe { error_text(api_ref) }).into());
        }
        let submitted = unsafe { std::slice::from_raw_parts(pointer, capacity) }.to_vec();
        unsafe { (api.output_destroy)(lease) };
        if submitted != output_buffer {
            return Err(format!("fixture {} submit output differs from synchronous forward", fixture.name).into());
        }
        let mut bytes = Vec::with_capacity(output_buffer.len() * 4);
        for value in &output_buffer {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let output_file = output.join(format!("{}.f32", fixture.name));
        std::fs::write(&output_file, bytes)?;
        // All three pools and repeated pooled indices on the same real fixture.
        let mut pool_outputs = Vec::new();
        for code in [0i32, 1, 2] {
            // Cover every packed sequence so the coupled all-pool comparison is
            // not limited to the first row.
            let pooled: Vec<i64> = (0..sequences as i64).collect();
            let mut buffer = vec![0f32; pooled.len() * pooled_width as usize];
            let status = unsafe {
                (api.forward)(
                    guard.handle,
                    inputs.ids.as_ptr(),
                    inputs.types.as_ptr(),
                    inputs.positions.as_ptr(),
                    inputs.cumulative.as_ptr(),
                    sequences as i64,
                    max_sequence,
                    code,
                    pooled.as_ptr(),
                    pooled.len(),
                    std::ptr::null(),
                    0,
                    buffer.as_mut_ptr(),
                    buffer.len(),
                    media.map_or(std::ptr::null(), |values| values.as_ptr()),
                    images.as_ptr(),
                    images.len(),
                    std::ptr::null(),
                    0,
                )
            };
            if status != 0 {
                return Err(format!("fixture {} pool {code} failed: {}", fixture.name, unsafe { error_text(api_ref) }).into());
            }
            let file = output.join(format!("{}.pool{code}.f32", fixture.name));
            let mut bytes = Vec::with_capacity(buffer.len() * 4);
            for value in &buffer {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            std::fs::write(&file, bytes)?;
            pool_outputs.push(file.display().to_string());
        }
        let repeats = vec![0i64, 0];
        let mut repeat_buffer = vec![0f32; repeats.len() * pooled_width as usize];
        if unsafe {
            (api.forward)(
                guard.handle,
                inputs.ids.as_ptr(),
                inputs.types.as_ptr(),
                inputs.positions.as_ptr(),
                inputs.cumulative.as_ptr(),
                sequences as i64,
                max_sequence,
                2,
                repeats.as_ptr(),
                repeats.len(),
                std::ptr::null(),
                0,
                repeat_buffer.as_mut_ptr(),
                repeat_buffer.len(),
                media.map_or(std::ptr::null(), |values| values.as_ptr()),
                images.as_ptr(),
                images.len(),
                std::ptr::null(),
                0,
            )
        } != 0
        {
            return Err(format!("fixture {} repeated pooled request failed: {}", fixture.name, unsafe { error_text(api_ref) }).into());
        }
        let repeat_file = output.join(format!("{}.repeat.f32", fixture.name));
        let mut repeat_bytes = Vec::with_capacity(repeat_buffer.len() * 4);
        for value in &repeat_buffer {
            repeat_bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&repeat_file, repeat_bytes)?;
        let mut prefix = [0u64; 7];
        let prefix_ok = unsafe { (api.prefix_stats)(guard.handle, prefix.as_mut_ptr(), prefix.len()) } == 0;
        observations.push(serde_json::json!({
            "fixture": fixture.name,
            "profile": packet.runtime_profile,
            "graph_count": graph_count,
            "graph_stats": graph_stats,
            "prefix_supported": unsafe { (api.prefix_supported)(guard.handle, 0) },
            "prefix_stats": prefix,
            "prefix_stats_ok": prefix_ok,
            "submit_matches_sync": true,
            "output": output_file.display().to_string(),
            "all_pool_outputs": pool_outputs,
            "repeat_output": repeat_file.display().to_string(),
            "pooled_rows": pooled_indices.len(),
            "raw_rows": raw_rows,
            "capacity": capacity,
        }));
        if prefix_bytes > 0 {
            unsafe { (api.prefix_clear)(guard.handle) };
            for _ in 0..3 {
                if run(&mut output_buffer) != 0 {
                    return Err(format!("prefix run failed: {}", unsafe { error_text(api_ref) }).into());
                }
            }
            let mut stats = [0u64; 7];
            unsafe { (api.prefix_stats)(guard.handle, stats.as_mut_ptr(), stats.len()) };
            observations.push(serde_json::json!({
                "fixture": format!("{}-prefix", fixture.name),
                "prefix_stats": stats,
            }));
        }
    }
    let report = serde_json::json!({
        "schema": "qwen3-vl-b66-consumer-results/v1",
        "packet": packet_path,
        "runtime_profile": packet.runtime_profile,
        "library_sha256": actual,
        "output_width": output_width,
        "pooled_width": pooled_width,
        "classification_width": classification_width,
        "observations": observations,
    });
    std::fs::write(output.join("results.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
