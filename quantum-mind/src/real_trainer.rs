//! Real training — actually trains a QuantumAI neural network on the
//! QuantumMind training corpus. Unlike the earlier simulated/printed
//! training numbers, this builds real feature vectors from the corpus,
//! runs real forward/backward passes via `quantumai::Sequential`, and
//! reports real loss/accuracy numbers.
//!
//! Task: intent/category classification. Given a user query (e.g. "how
//! do I train a classifier"), predict which corpus category it belongs
//! to (syntax, quantumai, cybersecurity, debugging, concept, codegen,
//! conversation). This is a genuinely useful component — it's the same
//! kind of routing QuantumMind's `detect_intent` does, but learned from
//! data instead of hand-written rules.

use crate::trainer::TrainingCorpus;
use quantumai::{Sequential, Tensor};
use std::collections::HashMap;

/// Fixed-size feature vector dimension. Each input string is hashed into
/// this many buckets (a simple hashed bag-of-words / "hashing trick"),
/// which avoids needing a full vocabulary while still giving the network
/// real signal to learn from.
const FEATURE_DIM: usize = 64;

/// Hash a single word into a bucket index in [0, FEATURE_DIM).
fn word_bucket(word: &str) -> usize {
    let mut hash: u64 = 5381;
    for b in word.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64);
    }
    (hash % FEATURE_DIM as u64) as usize
}

/// Convert an input string into a normalized hashed bag-of-words vector.
fn encode_text(text: &str) -> Vec<f64> {
    let mut v = vec![0.0f64; FEATURE_DIM];
    let mut n_words = 0usize;
    for word in text.to_lowercase().split_whitespace() {
        let w: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
        if w.len() < 2 { continue; }
        v[word_bucket(&w)] += 1.0;
        n_words += 1;
    }
    if n_words > 0 {
        for x in v.iter_mut() { *x /= n_words as f64; }
    }
    v
}

/// Result of a real training run.
pub struct RealTrainingResult {
    pub n_examples: usize,
    pub n_categories: usize,
    pub categories: Vec<String>,
    pub train_accuracy: f64,
    pub test_accuracy: f64,
    pub final_loss: f64,
    pub epoch_log: Vec<(usize, f64, f64)>, // (epoch, loss, train_acc)
    pub n_params: usize,
}

/// Actually train a small classifier on the corpus using QuantumAI.
/// Returns real metrics — no simulated/printed-only numbers.
pub fn train_intent_classifier(corpus: &TrainingCorpus, epochs: usize, verbose: bool) -> RealTrainingResult {
    // 1. Build category vocabulary (sorted for determinism)
    let mut categories: Vec<String> = corpus.examples.iter()
        .map(|e| e.category.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    categories.sort();
    let cat_index: HashMap<&str, usize> = categories.iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();
    let n_categories = categories.len();
    let n_examples = corpus.examples.len();

    // 2. Build feature matrix X and one-hot label matrix Y.
    // Shuffle example order first (deterministic seed) so the train/test
    // split below isn't dominated by whichever category happens to be
    // last in the JSON file — without this, test accuracy is meaningless.
    let mut indices: Vec<usize> = (0..n_examples).collect();
    let mut seed: u64 = 42;
    for i in (1..indices.len()).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (seed >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }

    let mut x_data = Vec::with_capacity(n_examples * FEATURE_DIM);
    let mut y_data = Vec::with_capacity(n_examples * n_categories);
    for &i in &indices {
        let ex = &corpus.examples[i];
        x_data.extend(encode_text(&ex.input));
        let idx = cat_index[ex.category.as_str()];
        for c in 0..n_categories {
            y_data.push(if c == idx { 1.0 } else { 0.0 });
        }
    }
    let x = Tensor::new(x_data, vec![n_examples, FEATURE_DIM]);
    let y = Tensor::new(y_data, vec![n_examples, n_categories]);

    // 3. Train/test split (80/20, deterministic — last 20% held out)
    let n_train = (n_examples as f64 * 0.8) as usize;
    let n_test = n_examples - n_train;

    let x_train = Tensor::new(x.data[..n_train * FEATURE_DIM].to_vec(), vec![n_train, FEATURE_DIM]);
    let y_train = Tensor::new(y.data[..n_train * n_categories].to_vec(), vec![n_train, n_categories]);
    let x_test = Tensor::new(x.data[n_train * FEATURE_DIM..].to_vec(), vec![n_test, FEATURE_DIM]);
    let y_test = Tensor::new(y.data[n_train * n_categories..].to_vec(), vec![n_test, n_categories]);

    // 4. Build a small network: FEATURE_DIM -> 32 -> n_categories
    let mut net = Sequential::new();
    net.dense(FEATURE_DIM, 32, "relu");
    net.dense(32, n_categories, "softmax");
    net.compile("adam", "cross_entropy", 0.01);
    let n_params = net.n_params;

    // 5. Real training loop, capturing per-epoch metrics at intervals
    let mut epoch_log = Vec::new();
    let log_every = (epochs / 10).max(1);
    for epoch in 0..epochs {
        // fit() runs one epoch internally when called with epochs=1;
        // call repeatedly so we can sample metrics between epochs.
        net.fit(&x_train, &y_train, 1, n_train.min(16).max(1), false);

        if epoch % log_every == 0 || epoch == epochs - 1 {
            let train_acc = net.compute_accuracy(&x_train, &y_train);
            let loss = net.history.losses.last().copied().unwrap_or(0.0);
            epoch_log.push((epoch + 1, loss, train_acc));
            if verbose {
                println!("  Epoch {:>4}/{} | loss: {:.4} | train_acc: {:.4}", epoch + 1, epochs, loss, train_acc);
            }
        }
    }

    let train_accuracy = net.compute_accuracy(&x_train, &y_train);
    let test_accuracy = if n_test > 0 { net.compute_accuracy(&x_test, &y_test) } else { f64::NAN };
    let final_loss = net.history.losses.last().copied().unwrap_or(0.0);

    RealTrainingResult {
        n_examples,
        n_categories,
        categories,
        train_accuracy,
        test_accuracy,
        final_loss,
        epoch_log,
        n_params,
    }
}
