use quantumai::*;

fn main() {
    let ds = data::make_xor(200);
    let (train, test) = ds.train_test_split(0.2);

    let mut m = Sequential::new();
    m.dense(2, 16, "relu");
    m.dense(16, 8, "relu");
    m.dense(8, 2, "softmax");
    m.compile("adam", "cross_entropy", 0.01);

    m.fit(&train.x, &train.y, 100, 32, true);
    let ev = m.evaluate(&test.x, &test.y);
    println!("Test: loss={:.4} acc={:.4}", ev["loss"], ev["accuracy"]);

    // Predict individual patterns
    let patterns: &[(f64, f64, &str)] = &[
        (0.0, 0.0, "0 XOR 0 → 0"),
        (0.0, 1.0, "0 XOR 1 → 1"),
        (1.0, 0.0, "1 XOR 0 → 1"),
        (1.0, 1.0, "1 XOR 1 → 0"),
    ];
    println!("\nXOR predictions:");
    for &(a, b, label) in patterns {
        let x = Tensor::new(vec![a, b], vec![1, 2]);
        let pred = m.predict(&x);
        let cls = m.predict_class(&x);
        println!("  {} → pred=[{:.3},{:.3}] class={}", label, pred.data[0], pred.data[1], cls);
    }
}
