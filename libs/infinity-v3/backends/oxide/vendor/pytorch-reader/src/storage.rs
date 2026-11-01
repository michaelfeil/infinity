//! Where tensor bytes come from once a pickle has been parsed.
//!
//! Every PyTorch container keeps tensor metadata (a pickle) apart from the storages holding
//! the raw bytes (little-endian in every file this reader accepts). A [`StorageSource`] maps
//! a storage key from that pickle to its bytes on demand: the ZIP and legacy sources read
//! the file at that point, the TAR source slices an entry already in memory.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use zip::read::ZipFile;
use zip::{CompressionMethod, ZipArchive, ZipReadOptions};

use crate::tensor::{ChecksumLimitation, ChecksumStatus};
use crate::{MAX_PICKLE_SIZE, PytorchError};

/// Largest `version`, `byteorder` or similar text entry accepted.
const MAX_TEXT_ENTRY_SIZE: u64 = 4096;

/// Largest window any bounded range or checksum read may request at once.
pub(crate) const MAX_BOUNDED_WINDOW: usize = 64 * 1024;

/// Storage bytes for one container format.
pub(crate) enum StorageSource {
    Zip(ZipSource),
    Tar(TarSource),
    Legacy(LegacySource),
    #[cfg(test)]
    Mock(std::sync::Arc<MockSource>),
}

impl fmt::Debug for StorageSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Zip(_) => "StorageSource::Zip",
            Self::Tar(_) => "StorageSource::Tar",
            Self::Legacy(_) => "StorageSource::Legacy",
            #[cfg(test)]
            Self::Mock(_) => "StorageSource::Mock",
        })
    }
}

impl StorageSource {
    /// Read at most `max_len` bytes of the storage saved under `key`, starting at `start`.
    ///
    /// A caller knows the window its tensor can touch, and the rest of the storage is never
    /// held: bytes past the window are left unread, and bytes before it are dropped as they
    /// go by, so a ZIP entry that decompresses far beyond its archive size costs its window
    /// rather than its size. A storage shorter than the window yields what it has.
    ///
    /// Returns the window and the number of bytes that preceded it, which is below `start`
    /// only when the storage ends inside them. The two together say how far the storage
    /// reaches, which a short read is diagnosed with.
    pub fn read(&self, key: &str, start: usize, max_len: usize) -> io::Result<(Vec<u8>, usize)> {
        match self {
            Self::Zip(source) => source.read_storage(key, start, max_len),
            Self::Tar(source) => source.read_storage(key, start, max_len),
            Self::Legacy(source) => source.read_storage(key, start, max_len),
            #[cfg(test)]
            Self::Mock(source) => source.read_storage(key, start, max_len),
        }
    }

    /// Verify the whole entry saved under `key` without holding it.
    ///
    /// Stored ZIP entries are streamed in bounded chunks and checked against their declared
    /// CRC. A CRC of 0 means no checksum. Compressed ZIP, legacy and TAR entries are
    /// reported as unsupported rather than treated as verified.
    pub fn verify_storage(&self, key: &str) -> io::Result<ChecksumStatus> {
        match self {
            Self::Zip(source) => source.verify_storage(key),
            Self::Tar(_) => Ok(ChecksumStatus::Unsupported(ChecksumLimitation::TarFormat)),
            Self::Legacy(_) => Ok(ChecksumStatus::Unsupported(
                ChecksumLimitation::LegacyFormat,
            )),
            #[cfg(test)]
            Self::Mock(_) => Ok(ChecksumStatus::NoChecksum),
        }
    }

    /// Record the size a pickle declares for a storage.
    ///
    /// Only the legacy container needs this: its storages are concatenated without any
    /// index, so the pickle's declarations are the only way to find their boundaries.
    pub fn declare(&self, key: &str, byte_len: usize, element_size: usize) -> io::Result<()> {
        match self {
            Self::Legacy(source) => source.declare(key, byte_len, element_size),
            Self::Zip(_) | Self::Tar(_) => Ok(()),
            #[cfg(test)]
            Self::Mock(_) => Ok(()),
        }
    }
}

fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn lock_ignoring_poison<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Read exactly `len` bytes without trusting `len` for the up-front allocation.
///
/// A length read from an untrusted file could be anything, so at most `capacity_bound`
/// bytes are reserved ahead of the read (the caller's idea of the largest plausible size,
/// such as the file length); beyond that the buffer grows only with bytes that actually
/// arrive, and a short read is an error.
pub(crate) fn read_exact_len<R: Read>(
    reader: &mut R,
    len: u64,
    capacity_bound: usize,
) -> io::Result<Vec<u8>> {
    let capacity = usize::try_from(len).map_or(capacity_bound, |len| len.min(capacity_bound));
    let mut buffer = Vec::with_capacity(capacity);
    reader.by_ref().take(len).read_to_end(&mut buffer)?;
    if buffer.len() as u64 != len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("expected {len} bytes, found {}", buffer.len()),
        ));
    }
    Ok(buffer)
}

/// The modern ZIP container (PyTorch 1.6+).
///
/// `torch.save` writes every entry under a root directory named after the file
/// (`model/data.pkl`, `model/data/0`, `model/version`, ...). Files saved through a file
/// object or an in-memory buffer use `archive/`, and some tools write the entries at the
/// root. The directory holding `data.pkl` is the root for every other entry.
///
/// The file is held open from [`open`](Self::open) until drop and never reopened by path,
/// so a reader keeps reading the file it opened if that path is unlinked or replaced.
#[derive(Clone)]
struct CachedVerificationError {
    kind: io::ErrorKind,
    message: String,
}
impl CachedVerificationError {
    fn capture(error: io::Error) -> Self {
        Self {
            kind: error.kind(),
            message: error.to_string(),
        }
    }
    fn restore(&self) -> io::Error {
        io::Error::new(self.kind, self.message.clone())
    }
}

type Verification = Result<ChecksumStatus, CachedVerificationError>;
type VerificationCache = Mutex<HashMap<String, Arc<OnceLock<Verification>>>>;

pub(crate) struct ZipSource {
    /// The central directory, and the stream every entry but a stored storage is read
    /// through. Locked to look any entry up, and for the whole of a read through the
    /// stream: the pickle and text entries, and a deflated storage.
    archive: Mutex<ZipArchive<BufReader<File>>>,
    /// A second handle to the same file, read at explicit offsets and never through its
    /// cursor. A stored storage's bytes come through it once the lookup under `archive`
    /// has said where they are, so tensor reads from different threads run at once instead
    /// of queueing on `archive`. Unused past `open` on Windows and on a target without a
    /// positional read (see `read_exact_at`).
    file: File,
    /// Root directory including its trailing slash, or empty at the archive root.
    root: String,
    /// Size of the archive file. A stored storage is checked to lie within it, and it caps
    /// the up-front allocation for a deflated one, which grows as it is read.
    file_len: u64,
    // One immutable opened archive owns this cache. Independently named tensor
    // views share a storage key, while other archives never share verification.
    verification: VerificationCache,
    #[cfg(test)]
    crc_scans: std::sync::atomic::AtomicUsize,
}

impl ZipSource {
    pub fn open(path: &Path) -> Result<Self, PytorchError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        // The stream gets a duplicate of the handle rather than an open of its own, so a
        // replacement of `path` cannot slip in between two opens. The duplicate shares the
        // cursor, which is why `file` is only ever read at explicit offsets.
        let archive = ZipArchive::new(BufReader::new(file.try_clone()?))?;

        let root = archive
            .file_names()
            .filter_map(|name| name.strip_suffix("data.pkl"))
            .filter(|root| root.is_empty() || root.ends_with('/'))
            .min_by_key(|root| root.len())
            .map(str::to_string)
            .ok_or_else(|| {
                PytorchError::InvalidFormat(
                    "No data.pkl entry found in ZIP archive. Expected a PyTorch 1.6+ checkpoint"
                        .to_string(),
                )
            })?;

        Ok(Self {
            archive: Mutex::new(archive),
            file,
            root,
            file_len,
            verification: Mutex::new(HashMap::new()),
            #[cfg(test)]
            crc_scans: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Total uncompressed size of the storage entries.
    pub fn data_size(&self) -> io::Result<u64> {
        let data_prefix = format!("{}data/", self.root);
        let mut archive = lock_ignoring_poison(&self.archive);
        let indices: Vec<usize> = archive
            .file_names()
            .filter(|name| name.starts_with(&data_prefix) && !name.ends_with('/'))
            .filter_map(|name| archive.index_for_name(name))
            .collect();
        let mut data_size = 0u64;
        for index in indices {
            data_size = data_size.saturating_add(archive.by_index_raw(index)?.size());
        }
        Ok(data_size)
    }

    /// The `data.pkl` bytes. Refused beyond [`MAX_PICKLE_SIZE`], since a deflated entry can
    /// claim any decompressed size.
    pub fn pickle(&self) -> io::Result<Vec<u8>> {
        self.read_entry(&format!("{}data.pkl", self.root), MAX_PICKLE_SIZE)
    }

    /// A small text entry next to `data.pkl` (`version`, `byteorder`, ...), trimmed, if the
    /// archive has it.
    pub fn read_text(&self, name: &str) -> io::Result<Option<String>> {
        let full_name = format!("{}{name}", self.root);
        if !self.has_entry(name) {
            return Ok(None);
        }
        let bytes = self.read_entry(&full_name, MAX_TEXT_ENTRY_SIZE)?;
        let text = String::from_utf8(bytes)
            .map_err(|err| invalid_data(format!("ZIP entry '{full_name}': {err}")))?;
        Ok(Some(text.trim().to_string()))
    }

    /// Whether an entry next to `data.pkl` exists.
    pub fn has_entry(&self, name: &str) -> bool {
        let full_name = format!("{}{name}", self.root);
        lock_ignoring_poison(&self.archive)
            .index_for_name(&full_name)
            .is_some()
    }

    fn read_storage(
        &self,
        key: &str,
        start: usize,
        max_len: usize,
    ) -> io::Result<(Vec<u8>, usize)> {
        let name = format!("{}data/{key}", self.root);
        let mut archive = lock_ignoring_poison(&self.archive);
        let mut entry = open_entry(&mut archive, &name)?;
        let size = entry.size();
        let skipped = (start as u64).min(size);
        let rest = size - skipped;
        let len = rest.min(max_len as u64);

        if cfg!(unix) && entry.compression() == CompressionMethod::Stored {
            // Every entry `torch.save` writes is stored, so this is the path tensors take.
            if entry.compressed_size() != size {
                return Err(invalid_data(format!(
                    "ZIP entry '{name}' declares {size} bytes but stores {}",
                    entry.compressed_size()
                )));
            }
            let data_start = entry
                .data_start()
                .ok_or_else(|| invalid_data(format!("ZIP entry '{name}' has no data offset")))?;
            // A CRC of 0 was never computed (see `open_entry`).
            let crc32 = (len == size)
                .then_some(entry.crc32())
                .filter(|&crc32| crc32 != 0);
            drop(entry);
            drop(archive);
            let bytes = self.read_stored(&name, data_start, skipped, len, crc32)?;
            return Ok((bytes, skipped as usize));
        }

        // A compressed entry yields its bytes only in order, so those before the window are
        // decompressed and dropped instead of being held.
        io::copy(&mut (&mut entry).take(skipped), &mut io::sink())?;
        // Stopping short of the entry's end skips its CRC check; the bytes a tensor uses
        // are still validated against its declared extent.
        let bytes = read_zip_entry(&mut entry, &name, len, rest, self.file_len)?;
        Ok((bytes, skipped as usize))
    }

    /// Share whole-storage verification across independently named tensor views.
    /// Success and failure are retained for this opened archive's lifetime, just
    /// as tensor handles retain their validation state. The file must remain
    /// immutable during model loading; replacing its path never changes this owner.
    fn verify_storage(&self, key: &str) -> io::Result<ChecksumStatus> {
        let state = {
            let mut cache = lock_ignoring_poison(&self.verification);
            Arc::clone(
                cache
                    .entry(key.to_owned())
                    .or_insert_with(|| Arc::new(OnceLock::new())),
            )
        };
        match state.get_or_init(|| {
            self.verify_storage_uncached(key)
                .map_err(CachedVerificationError::capture)
        }) {
            Ok(status) => Ok(*status),
            Err(error) => Err(error.restore()),
        }
    }

    /// Validate an entry and stream its CRC in bounded chunks. The cache lock
    /// is released before I/O so different storages can be verified concurrently.
    fn verify_storage_uncached(&self, key: &str) -> io::Result<ChecksumStatus> {
        let name = format!("{}data/{key}", self.root);
        let mut archive = lock_ignoring_poison(&self.archive);
        let entry = open_entry(&mut archive, &name)?;
        let size = entry.size();
        let crc32 = entry.crc32();
        let compression = entry.compression();
        let compressed_size = entry.compressed_size();
        let data_start = entry
            .data_start()
            .ok_or_else(|| invalid_data(format!("ZIP entry '{name}' has no data offset")))?;
        drop(entry);
        // Validate the record's actual on-disk extent before any CRC-0 short circuit.
        data_start
            .checked_add(compressed_size)
            .filter(|&end| end <= self.file_len)
            .ok_or_else(|| {
                invalid_data(format!(
                    "ZIP entry '{name}' extends beyond the end of the file"
                ))
            })?;
        if compression != CompressionMethod::Stored {
            return Ok(ChecksumStatus::Unsupported(
                ChecksumLimitation::CompressedZipEntry,
            ));
        }
        if compressed_size != size {
            return Err(invalid_data(format!(
                "ZIP entry '{name}' declares {size} bytes but stores {compressed_size}"
            )));
        }
        if crc32 == 0 {
            return Ok(ChecksumStatus::NoChecksum);
        }
        // On Unix, positional reads do not touch the archive stream's cursor, so the lock can
        // be released. On Windows and other non-Unix targets the positional read shares the
        // cursor, so the archive guard is retained until verification finishes.
        #[cfg(unix)]
        drop(archive);
        #[cfg(not(unix))]
        let _archive_guard = archive;
        self.verify_stored(&name, data_start, size, crc32)
    }

    /// Stream `size` bytes from `data_start` through a fixed-size buffer and compare the
    /// running CRC with `crc32`. The whole entry is never held.
    fn verify_stored(
        &self,
        name: &str,
        data_start: u64,
        size: u64,
        crc32: u32,
    ) -> io::Result<ChecksumStatus> {
        data_start
            .checked_add(size)
            .filter(|&end| end <= self.file_len)
            .ok_or_else(|| {
                invalid_data(format!(
                    "ZIP entry '{name}' extends beyond the end of the file"
                ))
            })?;
        #[cfg(test)]
        self.crc_scans
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut hasher = crc32fast::Hasher::new();
        let mut buffer = vec![0u8; MAX_BOUNDED_WINDOW];
        let mut offset = data_start;
        let mut remaining = size;
        while remaining > 0 {
            let chunk = usize::try_from(remaining.min(MAX_BOUNDED_WINDOW as u64))
                .map_err(|_| invalid_data(format!("ZIP entry '{name}' chunk overflows usize")))?;
            let window = buffer.get_mut(..chunk).ok_or_else(|| {
                invalid_data(format!(
                    "ZIP entry '{name}' chunk exceeds the verify buffer"
                ))
            })?;
            read_exact_at(&self.file, window, offset).map_err(|err| match err.kind() {
                io::ErrorKind::UnexpectedEof => io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "ZIP entry '{name}' ends before its {size} bytes: the file shrank after it was opened"
                    ),
                ),
                _ => err,
            })?;
            hasher.update(window);
            offset = offset
                .checked_add(chunk as u64)
                .ok_or_else(|| invalid_data(format!("ZIP entry '{name}' offset overflows")))?;
            remaining -= chunk as u64;
        }
        if hasher.finalize() != crc32 {
            return Err(invalid_data(format!(
                "ZIP entry '{name}': Invalid checksum"
            )));
        }
        Ok(ChecksumStatus::Verified)
    }

    /// Read `len` bytes of a stored entry, `skipped` bytes past its data at `data_start`.
    /// They are checked against `crc32` when it is given, which the caller does only when
    /// they are the whole entry and the entry declares a checksum.
    fn read_stored(
        &self,
        name: &str,
        data_start: u64,
        skipped: u64,
        len: u64,
        crc32: Option<u32>,
    ) -> io::Result<Vec<u8>> {
        // The offsets and lengths come from the file's own headers, so they are checked
        // against the file before `len` sizes an allocation: a stored entry cannot reach
        // past the end of the file.
        let offset = data_start
            .checked_add(skipped)
            .filter(|offset| {
                offset
                    .checked_add(len)
                    .is_some_and(|end| end <= self.file_len)
            })
            .ok_or_else(|| {
                invalid_data(format!(
                    "ZIP entry '{name}' extends beyond the end of the file"
                ))
            })?;
        let mut bytes = vec![0u8; len as usize];
        read_exact_at(&self.file, &mut bytes, offset).map_err(|err| match err.kind() {
            // The range lay within the file when it was opened, so a short read means the
            // file has shrunk since. An operating system error passes through as it is.
            io::ErrorKind::UnexpectedEof => io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "ZIP entry '{name}' ends before its {len} bytes: the file shrank after it was opened"
                ),
            ),
            _ => err,
        })?;
        // The stream path checks the CRC whenever a read reaches the entry's end, since the
        // bytes it skips still pass through its checksum. Skipped bytes are never read
        // here, so only a read of the whole entry can be checked.
        if crc32.is_some_and(|crc32| crc32fast::hash(&bytes) != crc32) {
            return Err(invalid_data(format!(
                "ZIP entry '{name}': Invalid checksum"
            )));
        }
        Ok(bytes)
    }

    /// Read a whole entry that must not exceed `max_size` bytes.
    fn read_entry(&self, name: &str, max_size: u64) -> io::Result<Vec<u8>> {
        let mut archive = lock_ignoring_poison(&self.archive);
        let mut entry = open_entry(&mut archive, name)?;
        let size = entry.size();
        if size > max_size {
            return Err(invalid_data(format!(
                "ZIP entry '{name}' is {size} bytes, above the {max_size} byte limit"
            )));
        }
        read_zip_entry(&mut entry, name, size, size, self.file_len)
    }
}

/// Open `name` for a read through the archive's stream.
///
/// The `zip` crate checks an entry's CRC once a read reaches its end. This skips that check
/// when the central directory declares a CRC of 0: `torch.save` writes that for every
/// entry under `compute_crc32=False`, and `torch.load` accepts it, so it means no checksum
/// rather than one to fail. An empty entry's true CRC is 0 as well, and its check could
/// only pass; a non-empty entry whose true CRC happens to be 0 loses its check.
fn open_entry<'a>(
    archive: &'a mut ZipArchive<BufReader<File>>,
    name: &str,
) -> io::Result<ZipFile<'a, BufReader<File>>> {
    let index = archive
        .index_for_name(name)
        .ok_or_else(|| invalid_data(format!("ZIP entry '{name}': not found in archive")))?;
    // The raw lookup parses the local header at most once; the read below finds its offset
    // cached.
    let crc32 = archive
        .by_index_raw(index)
        .map_err(|err| invalid_data(format!("ZIP entry '{name}': {err}")))?
        .crc32();
    archive
        .by_index_with_options(index, ZipReadOptions::new().ignore_crc32(crc32 == 0))
        .map_err(|err| invalid_data(format!("ZIP entry '{name}': {err}")))
}

/// Read the first `len` bytes of a ZIP entry whose header declares `size` bytes.
///
/// The `zip` crate checks an entry's CRC only when a read reaches its end, and a bounded
/// read stops on its own limit instead. When the whole entry is wanted, one read past
/// `size` reaches that end: a well-formed entry has nothing there and pays only for the
/// checksum, and bytes beyond the declared size mean the header lied about it.
fn read_zip_entry<R: Read>(
    entry: &mut R,
    name: &str,
    len: u64,
    size: u64,
    capacity_bound: u64,
) -> io::Result<Vec<u8>> {
    let capacity_bound = usize::try_from(capacity_bound).unwrap_or(usize::MAX);
    let bytes = read_exact_len(entry, len, capacity_bound)?;
    if len == size {
        let mut probe = [0u8; 1];
        let trailing = entry
            .read(&mut probe)
            .map_err(|err| invalid_data(format!("ZIP entry '{name}': {err}")))?;
        if trailing != 0 {
            return Err(invalid_data(format!(
                "ZIP entry '{name}' holds more than the {size} bytes it declares"
            )));
        }
    }
    Ok(bytes)
}

/// Fill `buf` from `offset` without depending on the handle's cursor, so reads through one
/// handle from several threads do not interfere.
#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

/// As on Unix, except that `seek_read` also sets the cursor. `LegacySource` is its file
/// object's only user by the time it reads, so nothing depends on that cursor; `ZipSource`
/// shares its cursor with the archive's stream and stays on that stream on Windows instead.
/// Windows serializes I/O on a synchronous handle, so threads still queue in the kernel
/// here, though no longer behind a lock.
#[cfg(windows)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(read) => {
                buf = &mut buf[read..];
                offset += read as u64;
            }
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

// No stable positional read in std elsewhere. `ZipSource::read_storage` keeps those targets
// on the stream path; the legacy container has no stream and its storages cannot be read.
#[cfg(not(any(unix, windows)))]
fn read_exact_at(_file: &File, _buf: &mut [u8], _offset: u64) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "no positional file read on this target",
    ))
}

/// The TAR container written by PyTorch before 0.1.10.
///
/// The `storages` entry is a count pickle followed, per storage, by a `(key, location,
/// storage type)` pickle, an `i64` element count and the raw bytes, then a list of storage
/// views. The whole entry is kept in memory and sliced. `reader::parse_tar_storages` parses
/// the layout, since that module has the pickle parser the metadata needs.
pub(crate) struct TarSource {
    blob: Vec<u8>,
    /// Storage key to `(offset, byte length)` within `blob`.
    layout: HashMap<String, (usize, usize)>,
}

impl TarSource {
    /// Wrap a parsed `storages` entry. Every range in `layout` must lie within `blob`.
    pub fn new(blob: Vec<u8>, layout: HashMap<String, (usize, usize)>) -> Self {
        Self { blob, layout }
    }

    fn read_storage(
        &self,
        key: &str,
        start: usize,
        max_len: usize,
    ) -> io::Result<(Vec<u8>, usize)> {
        let &(offset, len) = self
            .layout
            .get(key)
            .ok_or_else(|| invalid_data(format!("storage '{key}' not found in TAR archive")))?;
        let start = start.min(len);
        let window = (len - start).min(max_len);
        offset
            .checked_add(start)
            .and_then(|from| Some(from..from.checked_add(window)?))
            .and_then(|range| self.blob.get(range))
            .map(|bytes| (bytes.to_vec(), start))
            .ok_or_else(|| invalid_data(format!("storage '{key}' lies outside the TAR data")))
    }
}

/// The legacy container (PyTorch 0.1.10 to 1.5, and `_use_new_zipfile_serialization=False`).
///
/// After the metadata pickles the file holds every storage back to back, in the order of a
/// key list that follows the main pickle. Each storage is an `i64` element count followed
/// by its bytes. The count prefix gives elements, not bytes, so the sizes declared by the
/// persistent ids in the main pickle are collected first and the layout is derived from
/// them once the key list is known.
///
/// The file is held open from [`new`](Self::new) until drop and never reopened by path,
/// so a reader keeps reading the file it opened if that path is unlinked or replaced.
pub(crate) struct LegacySource {
    /// Read at explicit offsets and never through its cursor, so tensor reads from
    /// different threads run at once.
    file: File,
    state: Mutex<LegacyState>,
}

enum LegacyState {
    /// Collecting declarations while the main pickle is parsed:
    /// storage key to `(byte length, element size)`.
    Declaring(HashMap<String, (usize, usize)>),
    /// Layout fixed by the key list:
    /// storage key to `(file offset of the element count, byte length, element size)`.
    Finished(HashMap<String, (u64, usize, usize)>),
}

impl LegacySource {
    /// Wrap the open checkpoint, whose storages are read at explicit offsets from here on.
    pub fn new(file: File) -> Self {
        Self {
            file,
            state: Mutex::new(LegacyState::Declaring(HashMap::new())),
        }
    }

    fn declare(&self, key: &str, byte_len: usize, element_size: usize) -> io::Result<()> {
        let mut state = lock_ignoring_poison(&self.state);
        let LegacyState::Declaring(declared) = &mut *state else {
            return Err(invalid_data(format!(
                "storage '{key}' declared after the legacy layout was finalized"
            )));
        };
        match declared.get(key) {
            Some(&(prior_len, prior_size))
                if (prior_len, prior_size) != (byte_len, element_size) =>
            {
                Err(invalid_data(format!(
                    "storage '{key}' is declared twice with different sizes ({prior_len} and {byte_len} bytes)"
                )))
            }
            Some(_) => Ok(()),
            None => {
                declared.insert(key.to_string(), (byte_len, element_size));
                Ok(())
            }
        }
    }

    /// Fix the storage layout from the ordered key list and the data section bounds.
    pub fn finish(&self, keys: &[String], data_start: u64, file_len: u64) -> io::Result<()> {
        let mut state = lock_ignoring_poison(&self.state);
        let LegacyState::Declaring(declared) = &*state else {
            return Err(invalid_data(
                "legacy storage layout finalized twice".to_string(),
            ));
        };
        let mut layout = HashMap::with_capacity(keys.len());
        let mut offset = data_start;

        for key in keys {
            let &(byte_len, element_size) = declared.get(key).ok_or_else(|| {
                invalid_data(format!(
                    "storage '{key}' is listed in the legacy key list but never referenced by the pickle"
                ))
            })?;
            let end = offset
                .checked_add(8)
                .and_then(|start| start.checked_add(byte_len as u64))
                .filter(|&end| end <= file_len)
                .ok_or_else(|| {
                    invalid_data(format!(
                        "storage '{key}' ({byte_len} bytes) extends beyond the end of the file"
                    ))
                })?;
            layout.insert(key.clone(), (offset, byte_len, element_size));
            offset = end;
        }

        *state = LegacyState::Finished(layout);
        Ok(())
    }

    fn read_storage(
        &self,
        key: &str,
        start: usize,
        max_len: usize,
    ) -> io::Result<(Vec<u8>, usize)> {
        let (offset, byte_len, element_size) = {
            let state = lock_ignoring_poison(&self.state);
            let LegacyState::Finished(layout) = &*state else {
                return Err(invalid_data(
                    "legacy storage layout was never finalized".to_string(),
                ));
            };
            *layout
                .get(key)
                .ok_or_else(|| invalid_data(format!("storage '{key}' not found in legacy file")))?
        };

        let mut count = [0u8; 8];
        read_exact_at(&self.file, &mut count, offset)?;
        let stored_numel = i64::from_le_bytes(count);
        let expected_numel = (byte_len / element_size) as i64;
        if stored_numel != expected_numel {
            return Err(invalid_data(format!(
                "storage '{key}' holds {stored_numel} elements but the pickle declares {expected_numel}"
            )));
        }

        // `finish` checked that the storage lies within the file, so the length is trusted.
        let start = start.min(byte_len);
        let len = (byte_len - start).min(max_len);
        let mut bytes = vec![0u8; len];
        read_exact_at(&self.file, &mut bytes, offset + 8 + start as u64)?;
        Ok((bytes, start))
    }
}

/// Test-only storage that records the largest window a bounded read requested.
#[cfg(test)]
pub(crate) struct MockSource {
    len: usize,
    pattern: bool,
    max_requested: Mutex<usize>,
}

#[cfg(test)]
impl MockSource {
    pub(crate) fn with_pattern(len: usize) -> Self {
        Self {
            len,
            pattern: true,
            max_requested: Mutex::new(0),
        }
    }

    pub(crate) fn max_requested(&self) -> usize {
        *lock_ignoring_poison(&self.max_requested)
    }

    fn read_storage(
        &self,
        _key: &str,
        start: usize,
        max_len: usize,
    ) -> io::Result<(Vec<u8>, usize)> {
        let mut peak = lock_ignoring_poison(&self.max_requested);
        *peak = (*peak).max(max_len);
        drop(peak);
        if max_len > MAX_BOUNDED_WINDOW {
            return Err(invalid_data(format!(
                "mock storage window {max_len} exceeds the {MAX_BOUNDED_WINDOW} byte bound"
            )));
        }
        let start = start.min(self.len);
        let len = (self.len - start).min(max_len);
        let mut data = vec![0u8; len];
        if self.pattern {
            for (offset, byte) in data.iter_mut().enumerate() {
                *byte = start.wrapping_add(offset).wrapping_rem(251) as u8;
            }
        }
        Ok((data, start))
    }
}

#[cfg(test)]
mod verification_cache_tests {
    use super::*;
    use crate::pickle_reader::{StorageRef, build_tensor};
    use crate::{DType, Tensor};
    use std::io::{Seek, SeekFrom, Write};
    use std::sync::atomic::Ordering;
    use zip::write::SimpleFileOptions;

    fn archive(compression: CompressionMethod) -> (tempfile::TempDir, std::path::PathBuf, Vec<u8>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.bin");
        let payload = (0..(MAX_BOUNDED_WINDOW * 3 + 32))
            .map(|i| (i % 251) as u8)
            .collect::<Vec<_>>();
        let mut writer = zip::ZipWriter::new(File::create(&path).unwrap());
        let opts = SimpleFileOptions::default().compression_method(compression);
        writer.start_file("data.pkl", opts).unwrap();
        writer
            .write_all(b"metadata not used by this storage test")
            .unwrap();
        for key in ["0", "1"] {
            writer.start_file(format!("data/{key}"), opts).unwrap();
            writer.write_all(&payload).unwrap();
        }
        writer.finish().unwrap();
        (dir, path, payload)
    }
    fn alias(source: &Arc<StorageSource>, key: &str, offset: usize, length: usize) -> Tensor {
        build_tensor(
            StorageRef {
                source: Arc::clone(source),
                key: key.into(),
                dtype: Some(DType::U8),
                reachable_bytes: MAX_BOUNDED_WINDOW * 3 + 32,
                view_offset: 0,
            },
            DType::U8,
            offset,
            vec![length],
            vec![1],
        )
        .unwrap()
    }
    fn scans(source: &StorageSource) -> usize {
        let StorageSource::Zip(zip) = source else {
            panic!("real ZIP required")
        };
        zip.crc_scans.load(Ordering::Relaxed)
    }
    fn corrupt(path: &Path) {
        let mut zip = ZipArchive::new(File::open(path).unwrap()).unwrap();
        let entry = zip.by_name("data/0").unwrap();
        let start = entry.data_start().unwrap();
        drop(entry);
        drop(zip);
        let mut file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        file.seek(SeekFrom::Start(start + MAX_BOUNDED_WINDOW as u64 + 3))
            .unwrap();
        file.write_all(&[255]).unwrap();
        file.flush().unwrap();
    }

    #[test]
    fn real_zip_aliases_and_concurrent_views_stream_one_crc_per_storage() {
        let (_dir, path, payload) = archive(CompressionMethod::Stored);
        let source = Arc::new(StorageSource::Zip(ZipSource::open(&path).unwrap()));
        let first = alias(&source, "0", 7, 19);
        let second = alias(&source, "0", MAX_BOUNDED_WINDOW + 5, 13);
        assert_eq!(first.verify_storage().unwrap(), ChecksumStatus::Verified);
        assert_eq!(second.verify_storage().unwrap(), ChecksumStatus::Verified);
        assert_eq!(first.read_range(0, 19).unwrap(), payload[7..26]);
        assert_eq!(
            second.read_range(0, 13).unwrap(),
            payload[MAX_BOUNDED_WINDOW + 5..MAX_BOUNDED_WINDOW + 18]
        );
        assert_eq!(
            scans(&source),
            1,
            "distinct Tensor handles must not rehash their common storage"
        );
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let threads = (0..8)
            .map(|offset| {
                let tensor = alias(&source, "1", offset, 3);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    assert_eq!(tensor.verify_storage().unwrap(), ChecksumStatus::Verified);
                    tensor.read_range(0, 3).unwrap()
                })
            })
            .collect::<Vec<_>>();
        for (offset, thread) in threads.into_iter().enumerate() {
            assert_eq!(thread.join().unwrap(), payload[offset..offset + 3]);
        }
        assert_eq!(
            scans(&source),
            2,
            "a second storage needs one separate CRC, including concurrent cold aliases"
        );
    }

    #[test]
    fn corrupt_storage_failure_is_shared_and_new_archive_owner_revalidates() {
        let (_dir, path, _) = archive(CompressionMethod::Stored);
        let source = Arc::new(StorageSource::Zip(ZipSource::open(&path).unwrap()));
        corrupt(&path);
        let first = alias(&source, "0", 0, 3);
        let second = alias(&source, "0", MAX_BOUNDED_WINDOW + 8, 3);
        let a = first.verify_storage().unwrap_err();
        let b = second.verify_storage().unwrap_err();
        assert_eq!(a.kind(), io::ErrorKind::InvalidData);
        assert_eq!(b.kind(), a.kind());
        assert!(a.to_string().contains("checksum"), "{a}");
        assert!(b.to_string().contains("checksum"), "{b}");
        assert_eq!(
            scans(&source),
            1,
            "cached corruption must stay failed without another full scan"
        );
        let independent = Arc::new(StorageSource::Zip(ZipSource::open(&path).unwrap()));
        assert_eq!(
            alias(&independent, "0", 0, 3)
                .verify_storage()
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            scans(&independent),
            1,
            "a new open never inherits another archive's status"
        );
    }

    #[test]
    fn compressed_entries_remain_unsupported_without_false_verification() {
        let (_dir, path, _) = archive(CompressionMethod::Deflated);
        let source = Arc::new(StorageSource::Zip(ZipSource::open(&path).unwrap()));
        for offset in [0, MAX_BOUNDED_WINDOW + 8] {
            assert_eq!(
                alias(&source, "0", offset, 3).verify_storage().unwrap(),
                ChecksumStatus::Unsupported(ChecksumLimitation::CompressedZipEntry)
            );
        }
        assert_eq!(scans(&source), 0);
    }
}
