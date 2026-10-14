// SPDX-License-Identifier: Apache-2.0
//! Deterministic synthetic PyTorch ZIP/protocol-2 checkpoints for the existing
//! loader regression. No checked-in binary fixtures or Python are required.
use safetensors::{Dtype, SafeTensors};
use std::{fs, io::Write, path::Path};
use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

fn text(pickle: &mut Vec<u8>, value: &str) {
    pickle.push(b'X'); // BINUNICODE
    pickle.extend_from_slice(&(value.len() as u32).to_le_bytes());
    pickle.extend_from_slice(value.as_bytes());
}
fn integer(pickle: &mut Vec<u8>, value: usize) {
    pickle.push(b'J'); // BININT (all synthetic sizes fit i32)
    pickle.extend_from_slice(&i32::try_from(value).unwrap().to_le_bytes());
}
fn tuple(pickle: &mut Vec<u8>, values: &[usize]) {
    pickle.push(b'(');
    for &value in values {
        integer(pickle, value);
    }
    pickle.push(b't');
}
fn ordered_dict(pickle: &mut Vec<u8>) {
    pickle.extend_from_slice(b"ccollections\nOrderedDict\n)R");
}
fn tensor(
    pickle: &mut Vec<u8>,
    name: &str,
    storage: (&str, &str, usize, usize),
    shape: &[usize],
    strides: &[usize],
) {
    let (storage_key, storage_type, storage_elements, offset) = storage;
    text(pickle, name);
    pickle.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n(");
    pickle.push(b'(');
    text(pickle, "storage");
    pickle.extend_from_slice(format!("ctorch\n{storage_type}\n").as_bytes());
    text(pickle, storage_key);
    text(pickle, "cpu");
    integer(pickle, storage_elements);
    pickle.extend_from_slice(b"tQ"); // storage persistent ID
    integer(pickle, offset);
    tuple(pickle, shape);
    tuple(pickle, strides);
    pickle.push(0x89); // requires_grad=false
    ordered_dict(pickle); // backward hooks
    pickle.extend_from_slice(b"tR");
}

pub fn from_safetensors(source: &Path, target: &Path) {
    let bytes = fs::read(source).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let mut names = tensors.names();
    names.sort_unstable();
    let mut pickle = vec![0x80, 2];
    ordered_dict(&mut pickle);
    pickle.push(b'(');
    let mut storages = Vec::new();
    for name in names {
        let view = tensors.tensor(name).unwrap();
        assert_eq!(view.dtype(), Dtype::F32);
        let shape = view.shape();
        let logical = view.data();
        let (storage, offset, strides) = match *shape {
            [rows, cols] => {
                // A transposed contiguous backing, viewed with column-major strides.
                let mut data = vec![0; logical.len()];
                for row in 0..rows {
                    for col in 0..cols {
                        data[(col * rows + row) * 4..(col * rows + row + 1) * 4].copy_from_slice(
                            &logical[(row * cols + col) * 4..(row * cols + col + 1) * 4],
                        );
                    }
                }
                (data, 0, vec![1, rows])
            }
            [length] => {
                let mut data = vec![0; (length + 5) * 4];
                data[12..12 + logical.len()].copy_from_slice(logical);
                data[12 + logical.len()..]
                    .copy_from_slice(&[1f32.to_le_bytes(), 1f32.to_le_bytes()].concat());
                (data, 3, vec![1])
            }
            _ => panic!("unexpected synthetic fixture shape"),
        };
        let key = storages.len().to_string();
        tensor(
            &mut pickle,
            name,
            (&key, "FloatStorage", storage.len() / 4, offset),
            shape,
            &strides,
        );
        storages.push((key, storage));
    }
    // Non-floating buffers must be ignored by the inference loader.
    let key = storages.len().to_string();
    tensor(
        &mut pickle,
        "embeddings.position_ids",
        (&key, "LongStorage", 16, 0),
        &[1, 16],
        &[16, 1],
    );
    storages.push((key, (0i64..16).flat_map(i64::to_le_bytes).collect()));
    pickle.push(b'u'); // SETITEMS
    pickle.extend_from_slice(b"}");
    text(&mut pickle, "_metadata");
    ordered_dict(&mut pickle);
    text(&mut pickle, "");
    pickle.push(b'}');
    text(&mut pickle, "version");
    integer(&mut pickle, 1);
    pickle.extend_from_slice(b"sssb."); // metadata dictionaries + BUILD + STOP
    let mut zip = ZipWriter::new(fs::File::create(target).unwrap());
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    for (name, data) in [
        ("archive/data.pkl".to_owned(), pickle),
        ("archive/byteorder".to_owned(), b"little".to_vec()),
        ("archive/version".to_owned(), b"3\n".to_vec()),
    ]
    .into_iter()
    .chain(
        storages
            .into_iter()
            .map(|(key, data)| (format!("archive/data/{key}"), data)),
    ) {
        zip.start_file(name, options).unwrap();
        zip.write_all(&data).unwrap();
    }
    zip.finish().unwrap();
}
