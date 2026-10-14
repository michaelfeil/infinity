#![cfg(feature = "flash-attn")]

mod common;

use crate::common::{sort_embeddings, SnapshotEmbeddings};
use anyhow::Result;
use common::{batch, cosine_matcher, download_artifacts, load_tokenizer};
use infinity_v3_backend_candle::CandleBackend;
use infinity_v3_backend_core::{Backend, ModelType, Pool};

#[test]
#[ignore = "requires CUDA and downloads the EmbeddingGemma checkpoint"]
#[serial_test::serial]
fn test_gemma3() -> Result<()> {
    // Pinned ungated mirror: weights, tokenizer, and configs match the official
    // google/embeddinggemma-300m revision 57c266a740f537b4dc058e1b0cda161fd15afa75.
    let (model_root, dense_paths) = download_artifacts(
        "michaelfeil/embeddinggemma-300m",
        Some("759942eb3b857cf49e7b472b21b74e0a7a49418d"),
        None,
    )?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "bfloat16".to_string(),
        ModelType::Embedding(Pool::Mean),
        dense_paths,
        0,
    )?;

    let input_batch = batch(
        vec![
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
            tokenizer.encode("Deep Learning is...", true).unwrap(),
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
        ],
        [0, 1, 2].to_vec(),
        vec![],
    );

    let matcher = cosine_matcher();

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_batch)?);
    let embeddings_batch = SnapshotEmbeddings::from(pooled_embeddings);
    insta::assert_yaml_snapshot!("gemma3_cpu_batch", embeddings_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    insta::assert_yaml_snapshot!("gemma3_cpu_single", embeddings_single, &matcher);
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    Ok(())
}
