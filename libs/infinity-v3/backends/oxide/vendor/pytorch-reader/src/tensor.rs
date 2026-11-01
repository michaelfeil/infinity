//! What the reader produces for each tensor in a checkpoint.

use std::fmt;
use std::io;
use std::sync::Arc;

/// Element type of a tensor.
///
/// The types `torch.save` writes as a plain storage. Complex, quantized and sparse tensors
/// are refused at parse time rather than given a variant here. Deliberately exhaustive: a
/// converter matching on it should fail to compile, not fall through, when a variant is
/// added, so adding one is a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    /// One byte per element, holding 0 or 1.
    Bool,
}

impl DType {
    /// Size of one element in bytes.
    pub const fn size(self) -> usize {
        match self {
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }
}

/// Outcome of verifying a tensor's whole storage entry.
///
/// [`Tensor::read_range`] is an explicitly unverified primitive. A loader that publishes
/// bytes read through ranges must call [`Tensor::verify_storage`] and reject a checksum
/// mismatch before publishing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChecksumStatus {
    /// The declared CRC was recomputed over the whole entry and matched.
    Verified,
    /// The entry declares no checksum (a stored-ZIP CRC of 0).
    NoChecksum,
    /// This bounded prototype cannot verify the container or entry. This is not proof of
    /// correctness, and is never reported as [`Verified`](Self::Verified).
    Unsupported(ChecksumLimitation),
}

/// Why a storage entry cannot be verified by this bounded prototype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChecksumLimitation {
    /// Verification currently covers stored ZIP entries, not deflated ones.
    CompressedZipEntry,
    /// Legacy checkpoint storages carry no CRC.
    LegacyFormat,
    /// Early TAR storages carry no CRC in this reader.
    TarFormat,
}

/// A tensor in a checkpoint, with its bytes not yet read.
///
/// Opening a checkpoint parses its metadata only. [`read`](Self::read) is what produces the
/// tensor's bytes, so a checkpoint can be inspected without loading it, or loaded one tensor
/// at a time. ZIP and legacy containers read the file at that point; a TAR container's
/// storages are held in memory from open.
///
/// The element type and shape are fixed at parse time, when the reader checks that the
/// tensor fits its storage, so they are read-only: [`byte_len`](Self::byte_len) is what
/// [`read`](Self::read) returns because nothing can move them apart.
#[derive(Clone)]
pub struct Tensor {
    /// Dotted path of the tensor through the checkpoint's dictionaries
    /// (`"encoder.0.weight"`). Free to rename; only error messages look at it.
    pub name: String,
    dtype: DType,
    shape: Vec<usize>,
    read: Arc<dyn Fn() -> io::Result<Vec<u8>> + Send + Sync>,
    read_range: Arc<dyn Fn(usize, usize) -> io::Result<Vec<u8>> + Send + Sync>,
    verify_storage: Arc<dyn Fn() -> io::Result<ChecksumStatus> + Send + Sync>,
}

impl Tensor {
    pub(crate) fn new(
        name: String,
        dtype: DType,
        shape: Vec<usize>,
        read: impl Fn() -> io::Result<Vec<u8>> + Send + Sync + 'static,
        read_range: impl Fn(usize, usize) -> io::Result<Vec<u8>> + Send + Sync + 'static,
        verify_storage: impl Fn() -> io::Result<ChecksumStatus> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name,
            dtype,
            shape,
            read: Arc::new(read),
            read_range: Arc::new(read_range),
            verify_storage: Arc::new(verify_storage),
        }
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Extents, outermost first. Empty for a scalar.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of bytes [`read`](Self::read) returns.
    pub fn byte_len(&self) -> usize {
        self.num_elements() * self.dtype.size()
    }

    /// Read the tensor's bytes from the checkpoint.
    ///
    /// The bytes are contiguous in row-major order and native-endian, whatever layout the
    /// file stored the tensor in: a strided or offset view is gathered, and a `bool` tensor
    /// holds only 0 and 1. Nothing is cached, so each call reads the storage again.
    ///
    /// Reads from different threads run at once: a ZIP read holds the archive lock only to
    /// locate its entry. A compressed entry, which `torch.save` does not write, streams
    /// under the lock instead, as does every entry on a target other than Unix.
    ///
    /// A full read of a strided view requests one storage window spanning every index the
    /// view touches, including holes, and is therefore not bounded for huge strides. Use
    /// [`read_range`](Self::read_range) plus [`verify_storage`](Self::verify_storage) for
    /// bounded loading of large or strided views.
    ///
    /// # Errors
    ///
    /// A storage that is missing, corrupt or unreadable by its container fails with
    /// [`io::ErrorKind::InvalidData`] (a ZIP storage's checksum is verified when the tensor
    /// spans all of it, unless the file was saved without checksums, as
    /// `compute_crc32=False` does), and one shorter than the pickle declared with
    /// [`io::ErrorKind::UnexpectedEof`]: the file disagrees with itself. Any other kind is
    /// the operating system's, and means the file could not be read at all. Every error
    /// names the tensor and its storage.
    pub fn read(&self) -> io::Result<Vec<u8>> {
        (self.read)().map_err(|err| named(&self.name, err))
    }

    /// Read a row-major slice of the tensor from the checkpoint.
    ///
    /// The result is exactly `read()[start_element * dtype().size()..
    /// (start_element + element_count) * dtype().size()]`: the same ordering,
    /// native endianness and `bool` normalization.
    ///
    /// Each storage read is capped at 64 KiB, independent of the storage span or stride, so
    /// working memory is bounded by the requested output plus one storage window. Contiguous
    /// storage runs are coalesced; holes between strided elements are never spanned as one
    /// unbounded read. A zero-stride dimension reuses the previous element.
    ///
    /// `start_element + element_count` must not exceed [`num_elements`](Self::num_elements);
    /// an out-of-bounds or overflowing range is [`io::ErrorKind::InvalidInput`]. A storage
    /// that ends inside a needed range is [`io::ErrorKind::UnexpectedEof`], as for
    /// [`read`](Self::read).
    ///
    /// # Checksums
    ///
    /// This is an explicitly unverified primitive: a successful range read does not claim
    /// that a ZIP entry's CRC was checked. A loader that publishes bytes gathered this way
    /// must call [`verify_storage`](Self::verify_storage) first and reject a mismatch. A
    /// partial range cannot prove a whole-entry checksum.
    ///
    /// The TAR container already holds its whole archive in memory at open; a range only
    /// slices that buffer and is not bounded by archive size. ZIP and legacy ranges read
    /// only bounded windows of the requested slice.
    pub fn read_range(&self, start_element: usize, element_count: usize) -> io::Result<Vec<u8>> {
        (self.read_range)(start_element, element_count).map_err(|err| named(&self.name, err))
    }

    /// Verify the whole storage entry backing this tensor.
    ///
    /// This is the checked-load companion to [`read_range`](Self::read_range). It streams
    /// the entire storage entry, not just the tensor's span, in chunks of at most 64 KiB and
    /// does not retain it. For a stored ZIP entry it validates the compression class,
    /// declared sizes and file extent before recomputing the declared CRC; a mismatch or an
    /// invalid record is [`io::ErrorKind::InvalidData`]. A stored entry whose declared CRC
    /// is 0 returns [`ChecksumStatus::NoChecksum`]. A deflated ZIP entry is always
    /// [`ChecksumStatus::Unsupported`], even with CRC 0, as are legacy and TAR entries; none
    /// is reported as a false success.
    ///
    /// Call this once for each distinct storage before publishing a model whose bytes were
    /// loaded through range reads. Full [`read`](Self::read) keeps its existing checksum
    /// behavior; this method does not change it.
    pub fn verify_storage(&self) -> io::Result<ChecksumStatus> {
        (self.verify_storage)().map_err(|err| named(&self.name, err))
    }

    /// Split the tensor into its metadata and its read handle.
    ///
    /// For a caller that keeps the metadata elsewhere and wants only the handle, without a
    /// second copy of the name and shape held alive with it. The handle reads and fails
    /// exactly as [`read`](Self::read) does.
    pub fn into_parts(
        self,
    ) -> (
        String,
        DType,
        Vec<usize>,
        impl Fn() -> io::Result<Vec<u8>> + Send + Sync + 'static,
    ) {
        let Self {
            name,
            dtype,
            shape,
            read,
            read_range: _,
            verify_storage: _,
        } = self;
        let context = name.clone();

        (name, dtype, shape, move || {
            read().map_err(|err| named(&context, err))
        })
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("name", &self.name)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

/// Put the tensor's name in front of a read error, keeping its kind.
///
/// The reader builds a tensor before it knows the name (that is assembled from the dict
/// path afterwards), so the name is added here rather than inside the handle.
fn named(name: &str, err: io::Error) -> io::Error {
    if name.is_empty() {
        return err;
    }
    io::Error::new(err.kind(), format!("tensor '{name}': {err}"))
}
