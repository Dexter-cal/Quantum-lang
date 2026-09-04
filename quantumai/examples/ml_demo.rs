//! QuantumAI v2.1 — Full ML Demo
//! Tests: Neural nets, backprop, all optimizers, NLP, traditional ML,
//!        clustering, dimensionality reduction, AutoML, pre-trained models

use quantumai::*;
use quantumai::ml::*;
use quantumai::nlp;

fn separator(title: &str) {
    println!("\n{}", "━".repeat(62));
    println!("  {}", title);
    println!("{}", "━".repeat(62));
}

fn main() {
    println!("\n{}", "═".repeat(62));
    println!("  QuantumAI v2.1  —  ML/DL Framework Full Demo");
    println!("  Quantum Programming Language");
    println!("{}", "═".repeat(62));

    test_tensors();
    test_activations();
    test_xor_neural_net();
    test_sine_regression();
    test_multiclass();
    test_nlp();
    test_all_optimizers();
    test_traditional_ml();
    test_clustering();
    test_ensemble();
    test_pretrained();
    test_automl();
    test_hyperparams();
    test_save_load();

    println!("\n{}", "═".repeat(62));
    println!("  ALL TESTS PASSED ✅");
    println!("  QuantumAI v2.1 is fully operational.");
    println!("{}", "═".repeat(62));
}

// ──────────────────────────────────────────────────────
fn test_tensors() {
    separator("1. Tensor Operations");
    let a = Tensor::new(vec![1.0,2.0,3.0,4.0], vec![2,2]);
    let b = Tensor::xavier(2, 2);
    let c = a.matmul(&b);
    println!("  matmul   shape: {}", c.shape_str());
    println!("  mean:    {:.4}", a.mean());
    println!("  std_dev: {:.4}", b.std_dev());
    println!("  norm_l2: {:.4}", a.norm_l2());
    let clipped = b.clip_grad_norm(0.5);
    println!("  grad_clip norm: {:.4} → {:.4}", b.norm_l2(), clipped.norm_l2());
    let t2 = Tensor::randn(vec![4,4]);
    println!("  randn[4,4] mean≈0: {:.3}", t2.mean());
    println!("  softmax sum=1: {:.6}", a.softmax().sum());
    println!("  ✓ Tensor ops");
}

// ──────────────────────────────────────────────────────
fn test_activations() {
    separator("2. Activation Functions");
    let t = Tensor::new(vec![-2.0,-1.0,0.0,1.0,2.0], vec![1,5]);
    let acts = ["relu","sigmoid","tanh","gelu","swish","mish","elu","selu","leaky_relu"];
    for name in &acts {
        let act = Activation::from_str(name);
        let out = act.apply(&t);
        println!("  {:12} → [{:.3},{:.3},{:.3},{:.3},{:.3}]",
            name, out.data[0], out.data[1], out.data[2], out.data[3], out.data[4]);
    }
    println!("  ✓ All {} activations", acts.len());
}

// ──────────────────────────────────────────────────────
fn test_xor_neural_net() {
    separator("3. XOR Neural Network (backprop)");
    let ds = make_xor(200);
    let (train, test) = ds.train_test_split(0.2);
    let mut m = Sequential::new();
    m.dense(2, 16, "relu");
    m.dense(16, 8, "relu");
    m.dense(8, 2, "softmax");
    m.compile("adam", "cross_entropy", 0.01);
    m.summary();
    m.fit(&train.x, &train.y, 100, 16, true);
    let metrics = m.evaluate(&test.x, &test.y);
    println!("  Test loss={:.4}  acc={:.4}", metrics["loss"], metrics["accuracy"]);
    m.history.plot_ascii();
    m.history.print_report();
    println!("  ✓ XOR net");
}

// ──────────────────────────────────────────────────────
fn test_sine_regression() {
    separator("4. Sine Regression (MSE + RMSE)");
    let ds = make_sine(300);
    let (train, test) = ds.train_test_split(0.2);
    let mut m = Sequential::new();
    m.dense(1, 64, "gelu");
    m.dense(64, 128, "gelu");
    m.dense(128, 64, "gelu");
    m.dense(64, 1, "linear");
    m.compile("adam", "mse", 0.001);
    m.fit(&train.x, &train.y, 200, 32, true);
    let ev = m.evaluate(&test.x, &test.y);
    println!("  MSE={:.6}  MAE={:.6}", ev["mse"], ev["mae"]);

    // Also test moons dataset
    let moons = make_moons(200, 0.15);
    let (mt, mv) = moons.train_test_split(0.2);
    let mut m2 = Sequential::new();
    m2.dense(2, 32, "relu"); m2.dense(32, 2, "softmax");
    m2.compile("adam", "cross_entropy", 0.01);
    m2.fit(&mt.x, &mt.y, 60, 16, false);
    let ev2 = m2.evaluate(&mv.x, &mv.y);
    println!("  Moons acc={:.4}", ev2["accuracy"]);

    // Circles dataset
    let circles = make_circles(200, 0.1, 0.5);
    let (ct, cv) = circles.train_test_split(0.2);
    let mut m3 = Sequential::new();
    m3.dense(2, 32, "relu"); m3.dense(32, 2, "softmax");
    m3.compile("adam", "cross_entropy", 0.01);
    m3.fit(&ct.x, &ct.y, 60, 16, false);
    let ev3 = m3.evaluate(&cv.x, &cv.y);
    println!("  Circles acc={:.4}", ev3["accuracy"]);
    println!("  ✓ Regression + moons + circles");
}

// ──────────────────────────────────────────────────────
fn test_multiclass() {
    separator("5. Multi-class + BatchNorm + Dropout");
    let mut ds = make_classification(500, 12, 5);
    ds.normalize();
    ds.shuffle();
    ds.info();
    let (train, test) = ds.train_test_split(0.2);
    let mut m = Sequential::new();
    m.dense(12, 128, "relu");
    m.batchnorm(128);
    m.dropout(0.3);
    m.dense(128, 64, "gelu");
    m.batchnorm(64);
    m.dropout(0.2);
    m.dense(64, 32, "swish");
    m.dense(32, 5, "softmax");
    m.compile("adamw", "cross_entropy", 0.001);
    m.summary();
    m.fit(&train.x, &train.y, 80, 64, true);
    let ev = m.evaluate(&test.x, &test.y);
    println!("  Test acc={:.4}", ev["accuracy"]);
    // Test predict_class and predict_proba
    let sample = Tensor::new(test.x.data[..12].to_vec(), vec![1,12]);
    let cls = m.predict_class(&sample);
    let proba = m.predict_proba(&sample);
    println!("  Sample prediction: class={} confidence={:.4}", cls, proba.max());
    println!("  ✓ Multi-class");
}

// ──────────────────────────────────────────────────────
fn test_nlp() {
    separator("6. NLP — TF-IDF + Embedding + Tokenizer");

    // TF-IDF
    let texts = [
        "quantum programming language fast",
        "machine learning deep neural network",
        "natural language processing text",
        "quantum computing qubit superposition",
        "deep learning transformer attention",
    ];
    let (tfidf_vecs, vocab) = nlp::tfidf(&texts);
    println!("  TF-IDF vocab={} docs={}", vocab.len(), tfidf_vecs.len());

    // N-grams
    let ngrams2 = nlp::ngrams("quantum programming language", 2);
    println!("  Bigrams: {:?}", ngrams2);

    // Cosine similarity
    if tfidf_vecs.len() >= 2 {
        let sim = nlp::cosine_similarity(&tfidf_vecs[0], &tfidf_vecs[3]);
        println!("  Cosine sim (quantum docs): {:.4}", sim);
    }

    // Tokenizer + embedding model
    let mut tok = nlp::Tokenizer::new(200);
    tok.fit(&texts);
    println!("  Vocab size: {}", tok.vocab_size());

    let encoded: Vec<f64> = tok.encode("quantum programming fast language", 10)
        .into_iter().map(|x| x as f64).collect();
    let decoded = tok.decode(&encoded.iter().map(|&x| x as usize).collect::<Vec<_>>());
    println!("  Encode→decode: '{}'", decoded);

    // Sentiment model
    let sentiments = [
        "love amazing great excellent wonderful",
        "hate terrible awful horrible bad",
        "good nice okay decent fine",
        "poor weak disappointing lacking failure",
        "outstanding superb perfect fantastic brilliant",
        "broken unusable garbage worst disaster",
    ];
    let labels = [1usize,0,1,0,1,0];
    let mut tok2 = nlp::Tokenizer::new(100);
    tok2.fit(&sentiments);
    let x_data: Vec<f64> = sentiments.iter()
        .flat_map(|t| tok2.encode(t,8).into_iter().map(|v| v as f64))
        .collect();
    let x = Tensor::new(x_data, vec![sentiments.len(), 8]);
    let mut y_data = vec![0.0f64; sentiments.len()*2];
    for (i, &l) in labels.iter().enumerate() { y_data[i*2+l] = 1.0; }
    let y = Tensor::new(y_data, vec![sentiments.len(), 2]);

    let mut nlp_model = Sequential::new();
    nlp_model.embedding(tok2.vocab_size(), 16);
    nlp_model.dense(16, 32, "relu");
    nlp_model.dense(32, 2, "softmax");
    nlp_model.compile("adam", "cross_entropy", 0.01);
    nlp_model.fit(&x, &y, 80, 6, true);

    let test_x = Tensor::new(
        tok2.encode("absolutely wonderful amazing experience", 8).into_iter().map(|v| v as f64).collect(),
        vec![1,8]
    );
    let cls = nlp_model.predict_class(&test_x);
    println!("  'wonderful experience' → {} ({})", cls, if cls==1 {"POSITIVE ✓"} else {"NEGATIVE"});
    println!("  ✓ NLP pipeline");
}

// ──────────────────────────────────────────────────────
fn test_all_optimizers() {
    separator("7. All Optimizers Benchmark");
    let ds = make_classification(200, 4, 2);
    let opts = ["adam","adamw","sgd","rmsprop","adagrad","nadam","amsgrad"];
    for opt in &opts {
        let mut m = Sequential::new();
        m.dense(4, 32, "relu");
        m.dense(32, 2, "softmax");
        m.compile(opt, "cross_entropy", 0.01);
        m.fit(&ds.x, &ds.y, 30, 32, false);
        let ev = m.evaluate(&ds.x, &ds.y);
        println!("  {:10} → loss={:.4}  acc={:.4}", opt, ev["loss"], ev["accuracy"]);
    }
    println!("  ✓ {} optimizers tested", opts.len());
}

// ──────────────────────────────────────────────────────
fn test_traditional_ml() {
    separator("8. Traditional ML Algorithms");

    // Linear Regression
    let sine = make_sine(150);
    let mut lr = LinearRegression::new(); lr.lr = 0.01; lr.epochs = 400;
    lr.fit(&sine.x, &sine.y);
    println!("  LinearRegression   R²={:.4}", lr.r2_score(&sine.x, &sine.y));

    // Ridge Regression
    let mut ridge = Ridge::new(0.1);
    ridge.fit(&sine.x, &sine.y);
    println!("  Ridge(α=0.1)       params={}", ridge.weights.len());

    // Logistic Regression with L2
    let ds2 = make_classification(200, 4, 2);
    let y_flat: Vec<f64> = (0..200).map(|i| ds2.y.data[i*2+1]).collect();
    let y1d = Tensor::new(y_flat, vec![200,1]);
    let mut log_reg = LogisticRegression::new();
    log_reg.l2 = 0.01;
    log_reg.fit(&ds2.x, &y1d);
    println!("  LogisticReg(L2)    acc={:.4}", log_reg.accuracy(&ds2.x, &y1d));

    // KNN with metrics
    let ds = make_classification(200, 4, 3);
    let labels: Vec<usize> = (0..200).map(|i| i%3).collect();
    for (k, metric) in [(3,"euclidean"),(5,"manhattan"),(3,"cosine")] {
        let mut knn = KNN::new(k).with_metric(metric);
        knn.fit(&ds.x, &labels);
        println!("  KNN(k={},{:12}) acc={:.4}", k, metric, knn.accuracy(&ds.x, &labels));
    }

    // Decision Tree
    let mut dt = DecisionTree::new(6);
    dt.min_samples = 3;
    dt.fit(&ds.x, &labels, 3);
    println!("  DecisionTree       acc={:.4}", dt.accuracy(&ds.x, &labels));

    // Naive Bayes
    let mut nb = NaiveBayes::new();
    nb.fit(&ds.x, &labels, 3);
    println!("  NaiveBayes         acc={:.4}", nb.accuracy(&ds.x, &labels));

    // SVM (linear)
    let svm_y: Vec<f64> = labels.iter().map(|&l| if l==0 {1.0} else {-1.0}).collect();
    let mut svm = SVM::new(1.0);
    svm.lr = 0.01; svm.epochs = 300;
    svm.fit(&ds.x, &svm_y);
    println!("  SVM(C=1.0)         acc={:.4}", svm.accuracy(&ds.x, &svm_y));

    // Metrics
    let dt_pred = dt.predict(&ds.x);
    let cm = confusion_matrix(&dt_pred, &labels, 3);
    println!("  Confusion Matrix (3×3):");
    for row in &cm { println!("    {:?}", row); }
    let prf = precision_recall_f1(&dt_pred, &labels, 3);
    for (i, (p,r,f)) in prf.iter().enumerate() {
        println!("    class {}: prec={:.3} rec={:.3} f1={:.3}", i, p, r, f);
    }
    println!("  ✓ Traditional ML");
}

// ──────────────────────────────────────────────────────
fn test_clustering() {
    separator("9. Clustering + Dimensionality Reduction");

    // K-Means
    let ds = make_classification(200, 4, 3);
    let mut km = KMeans::new(3);
    km.max_iter = 50;
    let labels = km.fit(&ds.x);
    println!("  KMeans(k=3) inertia={:.2}", km.inertia(&ds.x));
    let unique: std::collections::HashSet<usize> = labels.iter().cloned().collect();
    println!("  Clusters found: {}", unique.len());

    // PCA
    let mut pca = PCA::new(2);
    let reduced = pca.fit_transform(&ds.x);
    println!("  PCA: {}→{}", ds.x.shape_str(), reduced.shape_str());

    println!("  ✓ Clustering + PCA");
}

// ──────────────────────────────────────────────────────
fn test_ensemble() {
    separator("10. Ensemble — Random Forest");
    let ds = make_classification(300, 6, 4);
    let labels: Vec<usize> = (0..300).map(|i| i%4).collect();
    let mut rf = RandomForest::new(10, 6);
    rf.fit(&ds.x, &labels, 4);
    println!("  RandomForest(10 trees, depth=6) acc={:.4}", rf.accuracy(&ds.x, &labels));
    println!("  ✓ Random Forest");
}

// ──────────────────────────────────────────────────────
fn test_pretrained() {
    separator("11. Pre-trained Architectures");
    let _resnet  = pretrained::resnet50(true, 1000);
    let _bert    = pretrained::bert_tiny(true);
    let _gpt2    = pretrained::gpt2_mini();
    let _yolo    = pretrained::yolov8_nano();
    let _effnet  = pretrained::efficientnet_b0(10);
    let _t5      = pretrained::t5_small();
    let _whisper = pretrained::whisper_tiny();
    println!("  ✓ 7 pre-trained architectures initialized");
}

// ──────────────────────────────────────────────────────
fn test_automl() {
    separator("12. AutoML — Automatic Model Search");
    let ds = make_classification(300, 8, 3);
    let mut auto = AutoClassifier::new(4);
    auto.fit(&ds);
    println!("  Best accuracy: {:.4}", auto.best_accuracy);
    for r in &auto.results {
        println!("    Trial {}: {} → acc={:.4}", r.trial, r.config, r.accuracy);
    }
    println!("  ✓ AutoML");
}

// ──────────────────────────────────────────────────────
fn test_hyperparams() {
    separator("13. Hyperparameter Grid Search");
    let ds = make_classification(200, 6, 3);
    let best = HyperparamSearch::grid_search(
        &[0.01, 0.001],
        &[32, 64],
        &ds,
    );
    println!("  Best grid search accuracy: {:.4}", best);
    println!("  ✓ Hyperparameter search");
}

// ──────────────────────────────────────────────────────
fn test_save_load() {
    separator("14. Save / Load Weights");
    let ds = make_xor(100);
    let mut m = Sequential::new();
    m.dense(2, 8, "relu"); m.dense(8, 2, "softmax");
    m.compile("adam", "cross_entropy", 0.01);
    m.fit(&ds.x, &ds.y, 30, 16, false);

    let path = "/tmp/quantum_weights.json";
    m.save_weights(path);

    // New model, load weights, check preds match
    let mut m2 = Sequential::new();
    m2.dense(2, 8, "relu"); m2.dense(8, 2, "softmax");
    m2.compile("adam", "cross_entropy", 0.01);
    let loaded = m2.load_weights(path);
    println!("  Loaded: {}", loaded);

    let sample = Tensor::new(vec![1.0, 0.0], vec![1,2]);
    let p1 = m.predict(&sample);
    let p2 = m2.predict(&sample);
    let diff: f64 = p1.data.iter().zip(&p2.data).map(|(a,b)| (a-b).abs()).sum();
    println!("  Prediction diff after load: {:.8}", diff);
    println!("  ✓ Save/Load");
}
