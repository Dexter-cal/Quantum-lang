
use quantumai::*;

fn main() {
    // Reproduce EXACTLY what complete_test does
    let ds = DataPipeline::load("iris").normalize().dataset();
    let (train, test) = ds.train_test_split(0.2);

    println!("train.x.shape = {:?}", train.x.shape);
    println!("train.y.shape = {:?}", train.y.shape);
    println!("test.x.shape  = {:?}", test.x.shape);
    println!("test.y.shape  = {:?}", test.y.shape);
    println!("First train y (should be one-hot): {:?}", &train.y.data[..6]);

    let mut m = Model::create("classifier")
        .epochs(150)
        .learning_rate(0.01);
    m.train(&train.x, &train.y, 150, false);

    // Check final loss and acc separately
    let ev = m.evaluate(&train.x, &train.y);
    println!("train acc={:.4} loss={:.6}", ev["accuracy"], ev["loss"]);
    let ev2 = m.evaluate(&test.x, &test.y);
    println!("test  acc={:.4} loss={:.6}", ev2["accuracy"], ev2["loss"]);

    // Manual check on first 3 test samples
    let nf = test.x.shape[1];
    let nc = test.y.shape[1];
    println!("nf={} nc={}", nf, nc);
    for i in 0..3 {
        let xi = Tensor::new(test.x.data[i*nf..(i+1)*nf].to_vec(), vec![1, nf]);
        let pred = m.predict(&xi);
        println!("  pred={:?} true={:?}",
            pred.data.iter().map(|x| format!("{:.3}",x)).collect::<Vec<_>>(),
            &test.y.data[i*nc..(i+1)*nc]);
    }
}
