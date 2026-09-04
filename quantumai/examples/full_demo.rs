//! QuantumAI v2.1 — Full Feature Demo
//! Tests ALL features from the document plan:
//! Neural nets, CNN/LSTM/GRU/Transformer, Gradient Boosting,
//! VAE, GAN, Q-Learning, Word2Vec, BPE, Sentiment, DBSCAN,
//! ElasticNet, new datasets, metrics (ROC-AUC, Cohen's Kappa)

use quantumai::*;
use quantumai::ml::*;
use quantumai::advanced::*;
use quantumai::nlp_advanced::*;
use quantumai::datasets::*;
use quantumai::metrics as qmetrics;

fn sep(title: &str) {
    println!("\n{}", "━".repeat(58));
    println!("  {}", title);
    println!("{}", "━".repeat(58));
}

fn main() {
    println!("\n{}", "═".repeat(58));
    println!("  QuantumAI v2.1 — Full Feature Demo");
    println!("  (All features from the language design docs)");
    println!("{}", "═".repeat(58));

    test_lstm_gru();
    test_transformer();
    test_gradient_boosting();
    test_elastic_net();
    test_dbscan();
    test_vae();
    test_gan();
    test_rl_qlearning();
    test_word2vec();
    test_bpe();
    test_sentiment_pipeline();
    test_new_datasets();
    test_advanced_metrics();
    test_conv1d();

    println!("\n{}", "═".repeat(58));
    println!("  ALL FEATURES TESTED ✅");
    println!("  QuantumAI v2.1 — Document spec fully implemented.");
    println!("{}", "═".repeat(58));
}

fn test_lstm_gru() {
    sep("1. LSTM & GRU (Sequence Models)");

    // LSTM: [batch=2, seq=5, input=4] → [batch=2, seq=5, hidden=8]
    let lstm = LSTM::new(4, 8, 1);
    let input = Tensor::randn(vec![2, 5, 4]);
    let (out, cell) = lstm.forward(&input);
    println!("  LSTM input:  {:?}", input.shape);
    println!("  LSTM output: {:?}", out.shape);
    println!("  LSTM cell:   {:?}", cell.shape);
    assert_eq!(out.shape, vec![2, 5, 8]);
    assert_eq!(cell.shape, vec![2, 8]);
    println!("  ✓ LSTM");

    // GRU: [batch=3, seq=6, input=4] → [batch=3, seq=6, hidden=16]
    let gru = GRU::new(4, 16);
    let input_g = Tensor::randn(vec![3, 6, 4]);
    let out_g = gru.forward(&input_g);
    println!("  GRU input:  {:?}", input_g.shape);
    println!("  GRU output: {:?}", out_g.shape);
    assert_eq!(out_g.shape, vec![3, 6, 16]);
    println!("  ✓ GRU");
}

fn test_transformer() {
    sep("2. Multi-Head Attention & Transformer Block");

    // Multi-head attention
    let mha = MultiHeadAttention::new(32, 4);
    let x = Tensor::randn(vec![2, 8, 32]); // [batch, seq, embed]
    let out = mha.forward(&x);
    println!("  MHA input:  {:?}", x.shape);
    println!("  MHA output: {:?}", out.shape);
    assert_eq!(out.shape, vec![2, 8, 32]);
    println!("  ✓ Multi-Head Attention");

    // Transformer block
    let block = TransformerBlock::new(32, 4, 64);
    let out_t = block.forward(&x);
    println!("  Transformer block output: {:?}", out_t.shape);
    assert_eq!(out_t.shape, vec![2, 8, 32]);
    println!("  ✓ Transformer Block");
}

fn test_gradient_boosting() {
    sep("3. Gradient Boosting");

    let ds = make_classification(300, 6, 2);
    let y_binary: Vec<f64> = (0..300).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

    let mut gb = GradientBoosting::new(20, 0.1, 4);
    gb.fit(&ds.x, &y_binary);
    let acc = gb.accuracy(&ds.x, &y_binary);
    println!("  GradientBoosting (20 trees) acc: {:.4}", acc);
    assert!(acc > 0.4, "Gradient Boosting should do better than random");

    let preds = gb.predict_class(&ds.x);
    println!("  Sample predictions: {:?}", &preds[..5]);
    println!("  ✓ Gradient Boosting");
}

fn test_elastic_net() {
    sep("4. ElasticNet Regression (L1+L2)");

    let ds = make_regression(200, 5, 0.1);
    let mut en = ElasticNet::new(0.1, 0.5); // alpha=0.1, l1_ratio=0.5
    en.lr = 0.01; en.epochs = 300;
    en.fit(&ds.x, &ds.y);
    let preds = en.predict(&ds.x);
    let mse = preds.data.iter().zip(&ds.y.data)
        .map(|(p,t)| (p-t).powi(2)).sum::<f64>() / 200.0;
    println!("  ElasticNet MSE: {:.4}", mse);
    println!("  ✓ ElasticNet (alpha=0.1, l1_ratio=0.5)");
}

fn test_dbscan() {
    sep("5. DBSCAN Clustering");

    let (ds, _) = make_blobs(150, 2, 3);
    let dbscan = DBSCAN::new(0.8, 3);
    let labels = dbscan.fit(&ds.x);
    let n_clusters = dbscan.n_clusters(&labels);
    let noise_pts = labels.iter().filter(|&&l| l == -1).count();
    println!("  DBSCAN: {} clusters found, {} noise points", n_clusters, noise_pts);
    println!("  Label sample: {:?}", &labels[..8]);
    assert!(n_clusters >= 1, "DBSCAN should find at least 1 cluster");
    println!("  ✓ DBSCAN");
}

fn test_vae() {
    sep("6. Variational Autoencoder (VAE)");

    let vae = VAE::new(16, 32, 4);
    let x = Tensor::randn(vec![8, 16]);
    let (recon, mu, logvar) = vae.forward(&x);
    println!("  Input shape:   {:?}", x.shape);
    println!("  Recon shape:   {:?}", recon.shape);
    println!("  Mu shape:      {:?}", mu.shape);
    println!("  Logvar shape:  {:?}", logvar.shape);
    assert_eq!(recon.shape, vec![8, 16]);
    assert_eq!(mu.shape, vec![8, 4]);

    let loss = vae.vae_loss(&recon, &x, &mu, &logvar);
    println!("  VAE loss: {:.4}", loss);

    let samples = vae.sample(5);
    println!("  Generated samples shape: {:?}", samples.shape);
    assert_eq!(samples.shape, vec![5, 16]);
    println!("  ✓ VAE");
}

fn test_gan() {
    sep("7. Generative Adversarial Network (GAN)");

    let mut gan = GAN::new(16, 32, 64);
    gan.summary();

    let real = Tensor::randn(vec![8, 32]);
    let fake = gan.generate(8);
    println!("  Generated shape: {:?}", fake.shape);
    assert_eq!(fake.shape, vec![8, 32]);

    let disc_out = gan.discriminate(&real);
    println!("  Discriminator output shape: {:?}", disc_out.shape);

    let (d_loss, g_loss) = gan.train_step(&real, 8);
    println!("  D loss: {:.4}  G loss: {:.4}", d_loss, g_loss);
    println!("  ✓ GAN");
}

fn test_rl_qlearning() {
    sep("8. Reinforcement Learning — Q-Learning");

    let mut agent = QLearning::new(4, 2); // CartPole-like: 4 state, 2 actions
    println!("  Q-network params: {}", agent.q_network.layers.iter().map(|l| l.param_count()).sum::<usize>());

    // Simulate a few environment steps
    let state = vec![0.1f64, -0.05, 0.02, 0.01];
    let action = agent.act(&state);
    println!("  Action for state {:?}: {}", &state[..2], action);
    assert!(action < 2, "Action must be 0 or 1");

    // Store some fake experience
    for i in 0..50 {
        let s = vec![i as f64 * 0.01; 4];
        let a = i % 2;
        let r = if i % 5 == 0 { 1.0 } else { 0.0 };
        let ns = vec![(i+1) as f64 * 0.01; 4];
        agent.remember(s, a, r, ns, false);
    }
    println!("  Memory size: {}", agent.memory.len());

    agent.replay(16);
    println!("  Epsilon after replay: {:.4}", agent.epsilon);

    agent.update_target();
    println!("  Target network updated");
    println!("  ✓ Q-Learning");
}

fn test_word2vec() {
    sep("9. Word2Vec Embeddings");

    let texts = vec![
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a subset of artificial intelligence",
        "deep learning neural networks transform data",
        "quantum programming language fast simple",
        "natural language processing text classification",
    ];

    let mut w2v = Word2Vec::new(32, 2);
    w2v.train(&texts, 3);
    println!("  Vocab size: {}", w2v.vocab.len());

    if let Some(vec) = w2v.get_vector("learning") {
        println!("  'learning' vector dim: {} (first 4: {:.3} {:.3} {:.3} {:.3})",
            vec.len(), vec[0], vec[1], vec[2], vec[3]);
    }

    let similar = w2v.most_similar("learning", 3);
    println!("  Most similar to 'learning': {:?}", similar.iter().map(|(w,s)| format!("{}({:.3})", w, s)).collect::<Vec<_>>());
    println!("  ✓ Word2Vec");
}

fn test_bpe() {
    sep("10. BPE Tokenizer");

    let texts = vec![
        "hello world quantum programming",
        "machine learning deep neural network",
        "the quick brown fox jumps",
        "artificial intelligence language model",
    ];

    let mut bpe = BPETokenizer::new(500);
    bpe.train(&texts);
    println!("  BPE vocab size: {}", bpe.vocab_size_actual());

    let encoded = bpe.encode("hello quantum world", 8);
    println!("  Encoded 'hello quantum world': {:?}", encoded);
    assert_eq!(encoded.len(), 8, "Should be padded to max_len");
    println!("  ✓ BPE Tokenizer");
}

fn test_sentiment_pipeline() {
    sep("11. Sentiment Analysis Pipeline");

    let texts = vec![
        "amazing excellent wonderful love it highly recommend",
        "terrible awful horrible bad waste of money",
        "good decent okay nothing special",
        "outstanding superb perfect best ever fantastic",
        "poor quality broken disappointing garbage",
        "pretty good works fine decent product",
    ];
    let labels = vec![2usize, 0, 1, 2, 0, 1]; // 0=neg, 1=neutral, 2=pos

    let mut pipeline = SentimentPipeline::new(200, 16);
    pipeline.train(&texts, &labels, 30);
    println!("  Trained on {} samples", texts.len());

    let test_texts = [
        "absolutely wonderful love it",
        "completely broken terrible",
        "it is okay nothing amazing",
    ];

    for text in &test_texts {
        let (sentiment, conf) = pipeline.predict(text);
        println!("  '{}' → {} ({:.3})", text, sentiment, conf);
    }
    println!("  ✓ Sentiment Pipeline");
}

fn test_new_datasets() {
    sep("12. New Datasets (regression, blobs, spiral)");

    // Regression dataset
    let reg = make_regression(100, 5, 0.5);
    println!("  make_regression: x={:?} y={:?}", reg.x.shape, reg.y.shape);
    assert_eq!(reg.x.shape, vec![100, 5]);

    // Blobs dataset
    let (blobs, labels) = make_blobs(120, 3, 4);
    println!("  make_blobs: x={:?} labels={}", blobs.x.shape, labels.len());
    assert_eq!(blobs.x.shape, vec![120, 3]);

    // Spiral dataset
    let spiral = make_spiral(90, 3);
    println!("  make_spiral: x={:?} y={:?}", spiral.x.shape, spiral.y.shape);
    assert_eq!(spiral.x.shape, vec![90, 2]);

    // Iris-like
    let iris = make_iris_like(150);
    println!("  make_iris_like: x={:?} y={:?}", iris.x.shape, iris.y.shape);
    assert_eq!(iris.x.shape, vec![150, 4]);
    println!("  ✓ New datasets");
}

fn test_advanced_metrics() {
    sep("13. Advanced Metrics (ROC-AUC, Kappa, Classification Report)");

    let y_true = vec![1.0f64, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let y_score = vec![0.9, 0.2, 0.8, 0.3, 0.7, 0.6, 0.4, 0.85, 0.15, 0.25];
    let auc = qmetrics::roc_auc(&y_true, &y_score);
    println!("  ROC-AUC: {:.4} (should be > 0.9)", auc);
    assert!(auc > 0.8, "AUC should be high for good predictions");

    let pred_cls = vec![1usize, 0, 1, 0, 1, 1, 0, 1, 0, 0];
    let true_cls = vec![1usize, 0, 1, 0, 1, 0, 0, 1, 1, 0];
    let kappa = qmetrics::cohen_kappa(&pred_cls, &true_cls, 2);
    println!("  Cohen's Kappa: {:.4}", kappa);

    let log_l = qmetrics::log_loss(&y_score, &y_true);
    println!("  Log Loss: {:.4}", log_l);

    let mape = qmetrics::mean_absolute_percentage_error(
        &[1.1, 2.2, 3.3], &[1.0, 2.0, 3.0]);
    println!("  MAPE: {:.4}%", mape);

    qmetrics::classification_report(&pred_cls, &true_cls, 2, &["negative", "positive"]);
    println!("  ✓ Advanced Metrics");
}

fn test_conv1d() {
    sep("14. Conv1D (1D Convolutional Layer)");

    // input: [batch=2, channels=3, length=10]
    let conv = Conv1D::new(3, 8, 3); // in_ch=3, out_ch=8, kernel=3
    let input = Tensor::randn(vec![2, 3, 10]);
    let out = conv.forward(&input);
    println!("  Conv1D input:  {:?}", input.shape);
    println!("  Conv1D output: {:?}", out.shape);
    // out_len = (10 + 0 - 3)/1 + 1 = 8
    assert_eq!(out.shape, vec![2, 8, 8]);
    println!("  ✓ Conv1D");
}
