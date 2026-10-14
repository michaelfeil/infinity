use super::result_storage;
use infinity_v3_backend::BackendError;

// aggregation_strategy (str, optional, defaults to "none") — The strategy to fuse (or not) tokens based on the model prediction.
// "none" : Will simply not do any aggregation and simply return raw results from the model
// "simple" : Will attempt to group entities following the default schema. (A, B-TAG), (B, I-TAG), (C, I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being [{"word": ABC, "entity": "TAG"}, {"word": "D", "entity": "TAG2"}, {"word": "E", "entity": "TAG2"}] Notice that two consecutive B tags will end up as different entities. On word based languages, we might end up splitting words undesirably : Imagine Microsoft being tagged as [{"word": "Micro", "entity": "ENTERPRISE"}, {"word": "soft", "entity": "NAME"}]. Look for FIRST, MAX, AVERAGE for ways to mitigate that and disambiguate words (on languages that support that meaning, which is basically tokens separated by a space). These mitigations will only work on real words, "New york" might still be tagged with two different entities.
// "first" : (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with different tags. Words will simply use the tag of the first token of the word when there is ambiguity.
// "average" : (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with different tags. scores will be averaged first across tokens, and then the maximum label is applied.
// "max" : (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with different tags. Word entity will simply be the token with the maximum score.

#[derive(serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub(crate) enum AggregationStrategy {
    #[default]
    None,
    Simple,
    First,
    Average,
    Max,
}

pub fn apply_aggregation(
    sentence: &str,
    mut predictions: Vec<crate::http::types::TokenPrediction>,
    strategy: &AggregationStrategy,
    id2label: &std::collections::HashMap<String, String>,
    ignore_labels: &[String],
) -> Result<Vec<crate::http::types::TokenPrediction>, BackendError> {
    // First, filter out special tokens for ALL strategies (like HF does)
    for prediction in &mut predictions {
        prediction.results.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    }
    predictions.retain(|prediction| {
        !is_special_token(&prediction.token, prediction.start, prediction.end)
    });
    let mut filtered_predictions = predictions;

    let mut results = match strategy {
        AggregationStrategy::None => {
            // Preserve index ordering and last-maximum tie behavior while
            // borrowing checkpoint labels and reusing the original rows.
            let mut labels = result_storage::vector::<(usize, usize, &String)>(id2label.len())?;
            labels.extend(
                id2label.iter().enumerate().filter_map(|(ordinal, (k, v))| {
                    k.parse::<usize>().ok().map(|i| (i, ordinal, v))
                }),
            );
            // Explicit ordinals preserve stable index ordering without sort scratch allocation.
            labels.sort_unstable_by_key(|(index, ordinal, _)| (*index, *ordinal));
            for prediction in &mut filtered_predictions {
                let (entity_idx, score) = labels
                    .iter()
                    .map(|(_, _, label)| token_score(&prediction.results, label))
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .expect("id2label contains no score labels");
                let index = result_storage::index(entity_idx)?;
                let best_label = match id2label.get(&index) {
                    Some(label) => result_storage::string(label)?,
                    None => {
                        let mut label =
                            result_storage::string_capacity("UNKNOWN_".len() + index.len())?;
                        label.push_str("UNKNOWN_");
                        label.push_str(&index);
                        label
                    }
                };
                prediction.results.clear();
                // A valid score row already has room for one selected label.
                if prediction.results.capacity() == 0 {
                    prediction.results = result_storage::vector(1)?;
                }
                prediction.results.push((best_label, score));
            }
            filtered_predictions
        }
        AggregationStrategy::Simple => {
            simple_aggregation(sentence, filtered_predictions, id2label)?
        }
        AggregationStrategy::First => first_aggregation(sentence, filtered_predictions, id2label)?,
        AggregationStrategy::Average => {
            average_aggregation(sentence, filtered_predictions, id2label)?
        }
        AggregationStrategy::Max => max_aggregation(sentence, filtered_predictions, id2label)?,
    };

    // Filter out ignored labels (like Hugging Face does)
    if !ignore_labels.is_empty() {
        results.retain(|prediction| {
            if let Some((best_label, _)) = prediction
                .results
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                !ignore_labels.contains(best_label)
            } else {
                true
            }
        });
    }

    Ok(results)
}

fn token_score(scores: &[(String, f32)], label: &str) -> f32 {
    scores
        .binary_search_by(|(name, _)| name.as_str().cmp(label))
        .map(|index| scores[index].1)
        .unwrap_or(0.0)
}

/// Helper structure to represent pre-entities for aggregation
#[derive(Debug, Clone)]
struct PreEntity {
    word: String,
    scores: Vec<f32>,
    start: Option<usize>,
    end: Option<usize>,
    index: usize,
    is_subword: bool,
}

/// Safe detokenization from tokens (BERT-friendly version)
fn detok<'a>(tokens: impl Iterator<Item = &'a str> + Clone) -> Result<String, BackendError> {
    let size = tokens
        .clone()
        .enumerate()
        .try_fold(0usize, |size, (i, tok)| {
            let (text, space) = if let Some(rest) = tok.strip_prefix("##") {
                (rest, false)
            } else {
                (
                    tok.strip_prefix("Ġ")
                        .or_else(|| tok.strip_prefix("▁"))
                        .unwrap_or(tok),
                    i > 0,
                )
            };
            size.checked_add(text.len())
                .and_then(|s| s.checked_add(usize::from(space)))
                .ok_or_else(|| BackendError::Overloaded("NER joined word size overflow".into()))
        })?;
    let mut out = result_storage::string_capacity(size)?;
    for (i, tok) in tokens.enumerate() {
        if let Some(rest) = tok.strip_prefix("##") {
            out.push_str(rest);
            continue;
        }
        if i > 0 {
            out.push(' ');
        }
        out.push_str(
            tok.strip_prefix("Ġ")
                .or_else(|| tok.strip_prefix("▁"))
                .unwrap_or(tok),
        );
    }
    Ok(out)
}

fn label_for_index(
    id2label: &std::collections::HashMap<String, String>,
    index: usize,
) -> Result<String, BackendError> {
    let index = result_storage::index(index)?;
    if let Some(label) = id2label.get(&index) {
        return result_storage::string(label);
    }
    let mut label = result_storage::string_capacity("UNKNOWN_".len() + index.len())?;
    label.push_str("UNKNOWN_");
    label.push_str(&index);
    Ok(label)
}

/// Detect if a token is a special token that should be filtered out
fn is_special_token(token: &str, start: Option<usize>, end: Option<usize>) -> bool {
    // Hard-coded special token patterns
    if token == "[CLS]"
        || token == "[SEP]"
        || token == "[PAD]"
        || token == "[MASK]"
        || token == "[UNK]"
    {
        return true;
    }

    // Length-based detection: tokens with zero length are special tokens
    if let (Some(start), Some(end)) = (start, end) {
        if start >= end {
            return true;
        }
    }

    // Additional common special token patterns
    if token.starts_with("<") && token.ends_with(">") {
        // <s>, </s>, <pad>, etc.
        return true;
    }

    false
}

/// Detect if a token is a subword using marker-only approach (HF-compatible without sentence text)
fn is_subword_token(token: &str, _start: Option<usize>, _end: Option<usize>) -> bool {
    // BERT WordPiece: ## marks continuation
    if token.starts_with("##") {
        return true;
    }

    // RoBERTa/SentencePiece: Ġ/▁ marks word start, so subword = !starts_with("Ġ/▁")
    if token.starts_with('Ġ') || token.starts_with('▁') {
        return false;
    }

    // For all other tokens without markers, treat as word start (not subword)
    // This is the least harmful default without sentence text for whitespace checking
    false
}

/// Gather pre-entities from predictions
fn gather_pre_entities(
    predictions: Vec<crate::http::types::TokenPrediction>,
    id2label: &std::collections::HashMap<String, String>,
) -> Result<Vec<PreEntity>, BackendError> {
    let mut labels = result_storage::vector::<(usize, usize, &String)>(id2label.len())?;
    labels.extend(
        id2label
            .iter()
            .enumerate()
            .filter_map(|(ordinal, (key, label))| {
                key.parse::<usize>()
                    .ok()
                    .map(|index| (index, ordinal, label))
            }),
    );
    labels.sort_unstable_by_key(|(index, ordinal, _)| (*index, *ordinal));
    let mut result = result_storage::vector(predictions.len())?;
    for (index, prediction) in predictions.into_iter().enumerate() {
        let mut scores = result_storage::vector(labels.len())?;
        scores.extend(
            labels
                .iter()
                .map(|(_, _, label)| token_score(&prediction.results, label)),
        );
        let is_subword = is_subword_token(&prediction.token, prediction.start, prediction.end);
        result.push(PreEntity {
            word: prediction.token,
            scores,
            start: prediction.start,
            end: prediction.end,
            index,
            is_subword,
        });
    }
    Ok(result)
}

/// Get the BIO tag and entity type from a label
fn get_tag(entity_name: &str) -> (&str, &str) {
    if let Some(stripped) = entity_name.strip_prefix("B-") {
        ("B", stripped)
    } else if let Some(stripped) = entity_name.strip_prefix("I-") {
        ("I", stripped)
    } else {
        ("I", entity_name)
    }
}

/// Entity structure matching Hugging Face's format
#[derive(Debug, Clone)]
struct Entity {
    entity: String,
    score: f32,
    #[allow(dead_code)]
    index: usize,
    word: String,
    start: Option<usize>,
    end: Option<usize>,
}

/// EntityGroup structure matching Hugging Face's format
#[derive(Debug, Clone)]
struct EntityGroup {
    entity_group: String,
    score: f32,
    word: String,
    start: Option<usize>,
    end: Option<usize>,
}

/// Aggregate pre-entities based on the strategy
fn aggregate(
    sentence: &str,
    pre_entities: Vec<PreEntity>,
    aggregation_strategy: &AggregationStrategy,
    id2label: &std::collections::HashMap<String, String>,
) -> Result<Vec<EntityGroup>, BackendError> {
    if matches!(
        aggregation_strategy,
        AggregationStrategy::None | AggregationStrategy::Simple
    ) {
        let mut entities = result_storage::vector(pre_entities.len())?;
        for pre_entity in pre_entities {
            let entity_idx = pre_entity
                .scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let score = pre_entity.scores[entity_idx];

            let entity_label = label_for_index(id2label, entity_idx)?;
            let word = match (pre_entity.start, pre_entity.end) {
                (Some(start), Some(end)) if start <= end => match sentence.get(start..end) {
                    Some(text) => result_storage::string(text)?,
                    None => pre_entity.word,
                },
                _ => pre_entity.word,
            };

            let entity = Entity {
                entity: entity_label,
                score,
                index: pre_entity.index,
                word,
                start: pre_entity.start,
                end: pre_entity.end,
            };
            entities.push(entity);
        }

        if matches!(aggregation_strategy, AggregationStrategy::None) {
            let mut groups = result_storage::vector(entities.len())?;
            groups.extend(entities.into_iter().map(|e| EntityGroup {
                entity_group: e.entity,
                score: e.score,
                word: e.word,
                start: e.start,
                end: e.end,
            }));
            return Ok(groups);
        }

        group_entities(sentence, entities)
    } else {
        let word_entities =
            aggregate_words(sentence, pre_entities, aggregation_strategy, id2label)?;
        group_entities(sentence, word_entities)
    }
}

/// Aggregate a word using the specified strategy
fn aggregate_word(
    sentence: &str,
    entities: &[PreEntity],
    aggregation_strategy: &AggregationStrategy,
    id2label: &std::collections::HashMap<String, String>,
) -> Result<Entity, BackendError> {
    let word = match (
        entities.first().and_then(|e| e.start),
        entities.last().and_then(|e| e.end),
    ) {
        (Some(start), Some(end)) if start <= end => match sentence.get(start..end) {
            Some(text) => result_storage::string(text)?,
            None => detok(entities.iter().map(|e| e.word.as_str()))?,
        },
        _ => detok(entities.iter().map(|e| e.word.as_str()))?,
    };

    let (entity, score) = match aggregation_strategy {
        AggregationStrategy::First => {
            if entities.is_empty() {
                return Ok(Entity {
                    entity: result_storage::string("UNKNOWN")?,
                    score: 0.0,
                    word: String::new(),
                    start: None,
                    end: None,
                    index: 0,
                });
            }
            let scores = &entities[0].scores;
            let idx = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let score = scores[idx];
            let entity = label_for_index(id2label, idx)?;
            (entity, score)
        }
        AggregationStrategy::Max => {
            if entities.is_empty() {
                return Ok(Entity {
                    entity: result_storage::string("O")?,
                    score: 0.0,
                    word: String::new(),
                    start: None,
                    end: None,
                    index: 0,
                });
            }
            let max_entity = entities
                .iter()
                .max_by(|a, b| {
                    let a_max = a
                        .scores
                        .iter()
                        .filter(|score| score.is_finite())
                        .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(&0.0);
                    let b_max = b
                        .scores
                        .iter()
                        .filter(|score| score.is_finite())
                        .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(&0.0);
                    a_max
                        .partial_cmp(b_max)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(&entities[0]);

            let scores = &max_entity.scores;
            let idx = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let score = scores[idx];
            let entity = label_for_index(id2label, idx)?;
            (entity, score)
        }
        AggregationStrategy::Average => {
            if entities.is_empty() {
                return Ok(Entity {
                    entity: result_storage::string("O")?,
                    score: 0.0,
                    word: String::new(),
                    start: None,
                    end: None,
                    index: 0,
                });
            }
            let num_scores = entities[0].scores.len();
            let mut averaged_scores = result_storage::vector(num_scores)?;
            averaged_scores.resize(num_scores, 0.0);

            for entity in entities {
                for (i, score) in entity.scores.iter().enumerate() {
                    if i < averaged_scores.len() {
                        averaged_scores[i] += score;
                    }
                }
            }

            for score in &mut averaged_scores {
                if !entities.is_empty() {
                    *score /= entities.len() as f32;
                }
            }

            let entity_idx = averaged_scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let score = averaged_scores[entity_idx];
            let entity = label_for_index(id2label, entity_idx)?;
            (entity, score)
        }
        _ => (result_storage::string("O")?, 0.0),
    };

    Ok(Entity {
        entity,
        score,
        word,
        start: entities.first().and_then(|e| e.start),
        end: entities.last().and_then(|e| e.end),
        index: entities.first().map(|e| e.index).unwrap_or(0),
    })
}

/// Aggregate words for FIRST, AVERAGE, and MAX strategies using offset-based grouping
fn aggregate_words(
    sentence: &str,
    pre_entities: Vec<PreEntity>,
    aggregation_strategy: &AggregationStrategy,
    id2label: &std::collections::HashMap<String, String>,
) -> Result<Vec<Entity>, BackendError> {
    if matches!(
        aggregation_strategy,
        AggregationStrategy::None | AggregationStrategy::Simple
    ) {
        panic!("NONE and SIMPLE strategies are invalid for word aggregation");
    }

    let mut word_entities = result_storage::vector(pre_entities.len())?;
    let mut word_group = result_storage::vector(pre_entities.len())?;

    for entity in pre_entities {
        if word_group.is_empty() {
            word_group.push(entity);
        } else {
            let group = &mut word_group;
            let last_entity = group.last().unwrap();

            // Check if we should start a new word group based on offsets
            let should_split = match (last_entity.end, entity.start) {
                (Some(last_end), Some(curr_start)) => {
                    match curr_start.cmp(&last_end) {
                        std::cmp::Ordering::Less => {
                            // Overlapping tokens - split
                            true
                        }
                        std::cmp::Ordering::Greater => {
                            // Gap between tokens - check if it contains whitespace
                            let gap = sentence.get(last_end..curr_start).unwrap_or("");
                            gap.chars().any(|c| c.is_whitespace())
                        }
                        std::cmp::Ordering::Equal => {
                            // Adjacent tokens - check if this is a subword
                            entity.is_subword
                        }
                    }
                }
                _ => {
                    // Missing offsets - fall back to subword detection
                    !entity.is_subword
                }
            };

            if should_split {
                // Start a new word group
                word_entities.push(aggregate_word(
                    sentence,
                    group,
                    aggregation_strategy,
                    id2label,
                )?);
                word_group.clear();
                word_group.push(entity);
            } else {
                // Continue current word group
                group.push(entity);
            }
        }
    }

    if !word_group.is_empty() {
        let group = word_group;
        word_entities.push(aggregate_word(
            sentence,
            &group,
            aggregation_strategy,
            id2label,
        )?);
    }

    Ok(word_entities)
}

/// Group sub-entities together
fn group_sub_entities(sentence: &str, entities: &[Entity]) -> Result<EntityGroup, BackendError> {
    if entities.is_empty() {
        return Ok(EntityGroup {
            entity_group: result_storage::string("O")?,
            score: 0.0,
            word: String::new(),
            start: None,
            end: None,
        });
    }

    let entity = entities[0]
        .entity
        .split('-')
        .next_back()
        .unwrap_or(&entities[0].entity);

    let mut total_score = 0.0;
    let mut count = 0;
    for entity in entities {
        if entity.score.is_finite() {
            total_score += entity.score;
            count += 1;
        }
    }
    let score = if count > 0 {
        total_score / count as f32
    } else {
        0.0
    };

    let word = match (
        entities.first().and_then(|e| e.start),
        entities.last().and_then(|e| e.end),
    ) {
        (Some(start), Some(end)) if start <= end => match sentence.get(start..end) {
            Some(text) => result_storage::string(text)?,
            None => joined_words(entities)?,
        },
        _ => joined_words(entities)?,
    };
    Ok(EntityGroup {
        entity_group: result_storage::string(entity)?,
        score,
        word,
        start: entities[0].start,
        end: entities[entities.len() - 1].end,
    })
}

fn joined_words(entities: &[Entity]) -> Result<String, BackendError> {
    let bytes = entities
        .iter()
        .try_fold(entities.len().saturating_sub(1), |n, e| {
            n.checked_add(e.word.len())
        })
        .ok_or_else(|| BackendError::Overloaded("NER joined word size overflow".into()))?;
    let mut word = result_storage::string_capacity(bytes)?;
    for (index, entity) in entities.iter().enumerate() {
        if index > 0 {
            word.push(' ');
        }
        word.push_str(&entity.word);
    }
    Ok(word)
}

/// Group entities according to BIO tagging scheme
fn group_entities(sentence: &str, entities: Vec<Entity>) -> Result<Vec<EntityGroup>, BackendError> {
    let mut entity_groups = result_storage::vector(entities.len())?;
    let mut entity_group_disagg = result_storage::vector(entities.len())?;

    for entity in entities {
        if entity_group_disagg.is_empty() {
            entity_group_disagg.push(entity);
            continue;
        }

        let (bi, tag) = get_tag(&entity.entity);
        let (_last_bi, last_tag) = if let Some(last_entity) = entity_group_disagg.last() {
            get_tag(&last_entity.entity)
        } else {
            ("I", "")
        };

        if tag == last_tag && bi != "B" {
            entity_group_disagg.push(entity);
        } else {
            entity_groups.push(group_sub_entities(sentence, &entity_group_disagg)?);
            entity_group_disagg.clear();
            entity_group_disagg.push(entity);
        }
    }

    if !entity_group_disagg.is_empty() {
        entity_groups.push(group_sub_entities(sentence, &entity_group_disagg)?);
    }

    Ok(entity_groups)
}

/// Move entity groups into their wire-format rows after admitting storage.
fn groups_to_predictions(
    groups: Vec<EntityGroup>,
) -> Result<Vec<crate::http::types::TokenPrediction>, BackendError> {
    let mut result = result_storage::vector(groups.len())?;
    for group in groups {
        let mut scores = result_storage::vector(1)?;
        scores.push((group.entity_group, group.score));
        result.push(crate::http::types::TokenPrediction {
            token: group.word,
            token_id: 0,
            start: group.start,
            end: group.end,
            results: scores,
        });
    }
    Ok(result)
}

fn simple_aggregation(
    sentence: &str,
    predictions: Vec<crate::http::types::TokenPrediction>,
    id2label: &std::collections::HashMap<String, String>,
) -> Result<Vec<crate::http::types::TokenPrediction>, BackendError> {
    let pre_entities = gather_pre_entities(predictions, id2label)?;
    let entities = aggregate(
        sentence,
        pre_entities,
        &AggregationStrategy::Simple,
        id2label,
    )?;
    groups_to_predictions(entities)
}

fn first_aggregation(
    sentence: &str,
    predictions: Vec<crate::http::types::TokenPrediction>,
    id2label: &std::collections::HashMap<String, String>,
) -> Result<Vec<crate::http::types::TokenPrediction>, BackendError> {
    let pre_entities = gather_pre_entities(predictions, id2label)?;
    let entities = aggregate(
        sentence,
        pre_entities,
        &AggregationStrategy::First,
        id2label,
    )?;
    groups_to_predictions(entities)
}

fn average_aggregation(
    sentence: &str,
    predictions: Vec<crate::http::types::TokenPrediction>,
    id2label: &std::collections::HashMap<String, String>,
) -> Result<Vec<crate::http::types::TokenPrediction>, BackendError> {
    let pre_entities = gather_pre_entities(predictions, id2label)?;
    let entities = aggregate(
        sentence,
        pre_entities,
        &AggregationStrategy::Average,
        id2label,
    )?;
    groups_to_predictions(entities)
}

fn max_aggregation(
    sentence: &str,
    predictions: Vec<crate::http::types::TokenPrediction>,
    id2label: &std::collections::HashMap<String, String>,
) -> Result<Vec<crate::http::types::TokenPrediction>, BackendError> {
    let pre_entities = gather_pre_entities(predictions, id2label)?;
    let entities = aggregate(sentence, pre_entities, &AggregationStrategy::Max, id2label)?;
    groups_to_predictions(entities)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn grouped_results_reject_pressure_then_preserve_joined_words() {
        use infinity_v3_backend::result_memory::{RequestResults, ResultPool, REQUEST_RESULTS};
        use std::sync::Arc;
        let pool = ResultPool::new(16_384).unwrap();
        let source = pool.reserve(8).unwrap();
        let owner = Arc::new(RequestResults::default());
        owner.retain(Some(source));
        let labels = [("0".into(), "B-PER".into()), ("1".into(), "I-PER".into())]
            .into_iter()
            .collect();
        let rows = || {
            vec![
                crate::http::types::TokenPrediction {
                    token: "John".into(),
                    token_id: 1,
                    start: None,
                    end: None,
                    results: vec![("B-PER".into(), 0.9), ("I-PER".into(), 0.1)],
                },
                crate::http::types::TokenPrediction {
                    token: "Smith".into(),
                    token_id: 2,
                    start: None,
                    end: None,
                    results: vec![("B-PER".into(), 0.1), ("I-PER".into(), 0.9)],
                },
            ]
        };
        let held = pool.reserve(16_376).unwrap();
        assert!(matches!(
            REQUEST_RESULTS
                .scope(owner.clone(), async {
                    apply_aggregation("", rows(), &AggregationStrategy::Simple, &labels, &[])
                })
                .await,
            Err(BackendError::Overloaded(_))
        ));
        drop(held);
        let result = REQUEST_RESULTS
            .scope(owner.clone(), async {
                apply_aggregation("", rows(), &AggregationStrategy::Simple, &labels, &[])
            })
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].token, "John Smith");
        assert_eq!(result[0].results, vec![("PER".into(), 0.9)]);
        let response_owner = owner.clone();
        drop(owner);
        drop(result);
        assert!(pool.used() > 8);
        drop(response_owner);
        assert_eq!(pool.used(), 0);
        assert!(pool.reserve(16_384).is_ok());
    }

    #[test]
    fn test_is_subword_token_bert() {
        // BERT WordPiece tokens - marker-only approach
        assert!(!is_subword_token("microsoft", None, None)); // Regular word start
        assert!(is_subword_token("##soft", None, None)); // Continuation
        assert!(!is_subword_token("is", None, None)); // Regular word start
        assert!(!is_subword_token("great", None, None)); // Regular word start
        assert!(is_subword_token("##ing", None, None)); // Continuation
    }

    #[test]
    fn test_is_subword_token_roberta() {
        // RoBERTa BPE tokens - marker-only approach
        assert!(!is_subword_token("Ġmicrosoft", None, None)); // Word start with Ġ
        assert!(!is_subword_token("soft", None, None)); // No marker = word start
        assert!(!is_subword_token("Ġis", None, None)); // Word start with Ġ
    }

    #[test]
    fn test_is_subword_token_sentencepiece() {
        // SentencePiece tokens - marker-only approach
        assert!(!is_subword_token("▁microsoft", None, None)); // Word start with ▁
        assert!(!is_subword_token("soft", None, None)); // No marker = word start
        assert!(!is_subword_token("▁is", None, None)); // Word start with ▁
    }

    #[test]
    fn test_is_subword_token_edge_cases() {
        // Test edge cases with marker-only approach
        assert!(!is_subword_token("hello", None, None)); // Regular word
        assert!(!is_subword_token("world", None, None)); // Regular word
        assert!(!is_subword_token(".", None, None)); // Punctuation = word start
        assert!(!is_subword_token(",", None, None)); // Punctuation = word start
        assert!(is_subword_token("##ly", None, None)); // BERT continuation
        assert!(!is_subword_token("Ġtest", None, None)); // RoBERTa word start
        assert!(!is_subword_token("▁test", None, None)); // SentencePiece word start
    }

    #[test]
    fn test_detok_bert() {
        let tokens = [
            "micro".to_string(),
            "##soft".to_string(),
            "is".to_string(),
            "great".to_string(),
        ];
        let result = detok(tokens.iter().map(String::as_str)).unwrap();
        assert_eq!(result, "microsoft is great");
    }

    #[test]
    fn test_word_grouping() {
        // Test that BERT tokens are grouped correctly into words
        let tokens = vec![
            "micro".to_string(),
            "##soft".to_string(), // Should be one word: "microsoft"
            "is".to_string(),     // Should be one word: "is"
            "very".to_string(),
            "##great".to_string(), // Should be one word: "verygreat"
        ];

        let mut word_groups = Vec::new();
        let mut current_group: Option<Vec<String>> = None;

        for token in tokens {
            if is_subword_token(&token, None, None) {
                if let Some(group) = current_group.as_mut() {
                    group.push(token);
                } else {
                    current_group = Some(vec![token]);
                }
            } else {
                if let Some(group) = current_group {
                    word_groups.push(group);
                }
                current_group = Some(vec![token]);
            }
        }

        if let Some(group) = current_group {
            word_groups.push(group);
        }

        // Convert groups to words
        let words: Vec<String> = word_groups
            .iter()
            .map(|group| detok(group.iter().map(String::as_str)).unwrap())
            .collect();

        assert_eq!(words, vec!["microsoft", "is", "verygreat"]);
    }

    #[test]
    fn test_special_token_detection() {
        // Test hard-coded special tokens
        assert!(is_special_token("[CLS]", Some(0), Some(5)));
        assert!(is_special_token("[SEP]", Some(10), Some(15)));
        assert!(is_special_token("[PAD]", Some(20), Some(25)));
        assert!(is_special_token("[UNK]", Some(30), Some(35)));

        // Test length-based detection (zero-length tokens)
        assert!(is_special_token("something", Some(10), Some(10))); // start == end
        assert!(is_special_token("something", Some(15), Some(10))); // start > end

        // Test angle bracket tokens
        assert!(is_special_token("<s>", Some(0), Some(3)));
        assert!(is_special_token("</s>", Some(5), Some(9)));
        assert!(is_special_token("<pad>", Some(10), Some(15)));

        // Test normal tokens (should not be filtered)
        assert!(!is_special_token("hello", Some(0), Some(5)));
        assert!(!is_special_token("world", Some(6), Some(11)));
        assert!(!is_special_token("John", Some(12), Some(16)));

        // Test tokens without position info (should not be filtered unless hard-coded)
        assert!(!is_special_token("hello", None, None));
        assert!(is_special_token("[CLS]", None, None)); // Hard-coded still works
    }

    #[test]
    fn test_apply_aggregation_filters_special_tokens() {
        // Create mock predictions including special tokens
        let predictions = vec![
            crate::http::types::TokenPrediction {
                token: "[CLS]".to_string(),
                token_id: 0,
                start: Some(0),
                end: Some(5),
                results: {
                    let mut map = std::collections::HashMap::new();
                    map.insert("O".to_string(), 0.99);
                    map.into_iter().collect()
                },
            },
            crate::http::types::TokenPrediction {
                token: "John".to_string(),
                token_id: 1,
                start: Some(6),
                end: Some(10),
                results: {
                    let mut map = std::collections::HashMap::new();
                    map.insert("B-PER".to_string(), 0.95);
                    map.insert("O".to_string(), 0.05);
                    map.into_iter().collect()
                },
            },
            crate::http::types::TokenPrediction {
                token: "[SEP]".to_string(),
                token_id: 2,
                start: Some(11),
                end: Some(15),
                results: {
                    let mut map = std::collections::HashMap::new();
                    map.insert("O".to_string(), 0.98);
                    map.into_iter().collect()
                },
            },
        ];

        // Create mock id2label
        let mut id2label = std::collections::HashMap::new();
        id2label.insert("0".to_string(), "O".to_string());
        id2label.insert("1".to_string(), "B-PER".to_string());

        // Test None strategy (should filter special tokens and compute best label)
        let results = apply_aggregation(
            "Hello John",
            predictions.clone(),
            &AggregationStrategy::None,
            &id2label,
            &["O".to_string()],
        )
        .unwrap();

        // Should only contain "John" with B-PER, not "[CLS]" or "[SEP]"
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].token, "John");
        assert_eq!(results[0].results.len(), 1);
        assert_eq!(
            serde_json::to_value(&results[0]).unwrap()["results"],
            serde_json::json!({"B-PER":0.95f32})
        );
        assert!(results[0].results.iter().any(|(label, _)| label == "B-PER"));

        // Test Simple strategy (should also filter special tokens)
        let results_simple = apply_aggregation(
            "Hello John",
            predictions.clone(),
            &AggregationStrategy::Simple,
            &id2label,
            &["O".to_string()],
        )
        .unwrap();

        // Should also only contain "John" entity
        assert_eq!(results_simple.len(), 1);
        assert_eq!(results_simple[0].token, "John");
    }

    #[test]
    fn test_detok_sentencepiece() {
        let tokens = [
            "▁micro".to_string(),
            "soft".to_string(),
            "▁is".to_string(),
            "▁great".to_string(),
        ];
        let results = detok(tokens.iter().map(String::as_str)).unwrap();
        assert_eq!(results, "micro soft is great");
    }

    #[test]
    fn test_detok_mixed() {
        let tokens = ["i".to_string(), "like".to_string(), "pizza".to_string()];
        let result = detok(tokens.iter().map(String::as_str)).unwrap();
        assert_eq!(result, "i like pizza");
    }
}
