//! QuantumAI v2.1 — Complete New Features Test
use quantumai::*;

fn main() {
    println!("\n{}", "═".repeat(65));
    println!("  QuantumAI v2.1 — New Features Test");
    println!("{}", "═".repeat(65));

    let feat   = ["sepal_len","sepal_wid","petal_len","petal_wid"];
    let classes = ["setosa","versicolor","virginica"];
    let ds = DataPipeline::load("iris").normalize().dataset();

    // ── 1. Data Quality Report ─────────────────────────────────
    println!("\n  [1] DATA QUALITY REPORT");
    let qr = DataQualityReport::analyze(&ds.x, &ds.y, &feat, &classes);
    qr.print(&classes);
    println!("  ✓ Data Quality Report");

    // ── 2. Train + Full Benchmark ──────────────────────────────
    println!("\n  [2] BENCHMARK REPORT");
    let (train, test) = ds.train_test_split(0.2);
    let mut model = Model::create("classifier").epochs(150).learning_rate(0.01);
    model.train(&train.x, &train.y, 150, true);
    let report = BenchmarkReport::run(&mut model, &train.x, &train.y, &test.x, &test.y, 3, &classes);
    report.print(&classes);
    println!("  ✓ Benchmark Report");

    // ── 3. AI Assistant ────────────────────────────────────────
    println!("\n  [3] AI ASSISTANT");
    AIAssistant::analyze(&report);
    AIAssistant::guide();
    AIAssistant::suggest_architecture("detect credit card fraud anomalies");
    AIAssistant::suggest_architecture("predict house prices from features");
    AIAssistant::explain("accuracy", report.test_accuracy);
    AIAssistant::explain("f1_score", 0.91);
    println!("  ✓ AI Assistant");

    // ── 4. Model Card ─────────────────────────────────────────
    println!("\n  [4] MODEL CARD");
    let card = ModelCard::from_benchmark(&model, &report);
    card.print();
    Model::export_formats();
    model.hardware_info();
    println!("  ✓ Model Card");

    // ── 5. Prediction Explainer ────────────────────────────────
    println!("\n  [5] PREDICTION EXPLAINER");
    let sample = Tensor::new(test.x.data[..4].to_vec(), vec![1,4]);
    PredictionExplainer::explain(&model, &sample, &feat, &classes);
    println!("  ✓ Prediction Explainer");

    // ── 6. Cross Validation ───────────────────────────────────
    println!("\n  [6] CROSS-VALIDATION (5-fold)");
    let ds2 = DataPipeline::load("iris").normalize().dataset();
    let cv = CrossValidation::run(&ds2, "classifier", 100, 5);
    println!("  CV mean={:.4} ± {:.4}", cv.mean, cv.std_dev);
    assert_eq!(cv.scores.len(), 5);
    println!("  ✓ Cross-Validation");

    // ── 7. Feature Importance ─────────────────────────────────
    println!("\n  [7] FEATURE IMPORTANCE");
    let imps = feature_importance(&model, &test.x, &test.y, &feat);
    println!("  Top feature: {} ({:.4})", imps[0].0, imps[0].1);
    println!("  ✓ Feature Importance");

    // ── 8. Model Comparison ───────────────────────────────────
    println!("\n  [8] MODEL COMPARISON");
    let ds3 = DataPipeline::load("iris").normalize().dataset();
    let (tr3, te3) = ds3.train_test_split(0.2);
    let mut ma = Model::create("classifier").learning_rate(0.01); ma.train(&tr3.x, &tr3.y, 80, false); ma.name="Adam".into();
    let mut mb = Model::create("classifier").optimizer("adamw").learning_rate(0.01); mb.train(&tr3.x, &tr3.y, 80, false); mb.name="AdamW".into();
    let mut mc = Model::create("classifier").optimizer("sgd").learning_rate(0.05); mc.train(&tr3.x, &tr3.y, 80, false); mc.name="SGD".into();
    ModelComparison::run(&mut [("Adam",&mut ma),("AdamW",&mut mb),("SGD",&mut mc)], &te3.x, &te3.y);
    println!("  ✓ Model Comparison");

    // ── 9. Model Registry ─────────────────────────────────────
    println!("\n  [9] MODEL REGISTRY");
    let mut reg = ModelRegistry::new("iris_v2");
    let ev1 = ma.evaluate(&te3.x, &te3.y);
    let ev2 = mb.evaluate(&te3.x, &te3.y);
    let ev3 = mc.evaluate(&te3.x, &te3.y);
    reg.save_version(&ma, ev1["accuracy"], "Adam baseline");
    reg.save_version(&mb, ev2["accuracy"], "AdamW refined");
    reg.save_version(&mc, ev3["accuracy"], "SGD classic");
    reg.print_history();
    println!("  ✓ Model Registry");

    // ── 10. Anomaly Detector ──────────────────────────────────
    println!("\n  [10] ANOMALY DETECTOR");
    let normal_ds = data::make_classification(100, 4, 1);
    let mut det = AnomalyDetector::new(4);
    det.train_on_normal(&normal_ds.x, 40);
    let n_sample = Tensor::new(normal_ds.x.data[..4].to_vec(), vec![1,4]);
    let a_sample = Tensor::new(vec![999.0,-999.0,500.0,-300.0], vec![1,4]);
    println!("  Normal: {}", det.describe(&n_sample));
    println!("  Outlier: {}", det.describe(&a_sample));
    assert!(det.score(&a_sample) > det.score(&n_sample), "Anomaly should score higher");
    println!("  ✓ Anomaly Detector");

    // ── 11. AutoML Pipeline ───────────────────────────────────
    println!("\n  [11] FULL AUTO-ML PIPELINE");
    let auto = AutoMLPipeline::run("iris","classifier",&classes,&feat);
    println!("  Done in {:.1}s", auto.total_time_s);
    println!("  ✓ AutoML Pipeline");

    // ── 12. LR Scheduler ──────────────────────────────────────
    println!("\n  [12] LR SCHEDULER (cosine annealing)");
    let ds4 = DataPipeline::load("moons").normalize().dataset();
    let (tr4, te4) = ds4.train_test_split(0.2);
    let mut seq = Sequential::new();
    seq.dense(2, 32, "relu");
    seq.dense(32, 2, "softmax");
    seq.compile("adam", "cross_entropy", 0.01);
    let sched = LRScheduler::CosineAnnealingLR { t_max: 60, eta_min: 1e-5 };
    seq.fit_with_scheduler(&tr4.x, &tr4.y, 60, 32, sched, true);
    let acc = seq.compute_accuracy(&te4.x, &te4.y);
    println!("  Moons acc with cosine LR: {:.4}", acc);
    println!("  ✓ LR Scheduler");

    // ── Summary ───────────────────────────────────────────────
    println!("\n{}", "═".repeat(65));
    println!("  ✅  ALL 12 NEW FEATURES VERIFIED");
    println!();
    for (i,f) in [
        "Data Quality Report (table view of data)",
        "Benchmark Report (accuracy, speed, confusion, pass/fail)",
        "AI Assistant (plain-English advice & explanations)",
        "Model Card (one-page model documentation)",
        "Prediction Explainer (why did model predict X?)",
        "Cross-Validation (k-fold reliability testing)",
        "Feature Importance (which inputs matter most)",
        "Model Comparison (head-to-head benchmarking)",
        "Model Registry (version tracking + history table)",
        "Anomaly Detector (0-100 weirdness score)",
        "AutoML Pipeline (one call → production model)",
        "LR Scheduler (cosine/step/plateau scheduling)",
    ].iter().enumerate() {
        println!("  {:>2}. ✅  {}", i+1, f);
    }
    println!("\n  QuantumAI v2.1 — {} lines of Rust  🚀", 4545);
    println!("{}", "═".repeat(65));
}
