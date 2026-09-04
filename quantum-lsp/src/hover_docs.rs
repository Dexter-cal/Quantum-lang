//! Small, self-contained hover documentation table for Quantum keywords
//! and common QuantumAI API names. Intentionally minimal — this is a
//! starting point, not a full reference (see quantum-mind's KnowledgeBase
//! for the larger corpus used by the chat assistant).

/// Look up a short Markdown doc string for `word`, if known.
pub fn lookup(word: &str) -> Option<&'static str> {
    Some(match word {
        // ── Keywords ─────────────────────────────────────────────
        "fn" => "**fn** — declares a function.\n```quantum\nfn add(a: int, b: int) -> int {\n    return a + b\n}\n```",
        "let" => "**let** — declares a variable. Add `mut` for a mutable binding.\n```quantum\nlet x = 42\nlet mut count = 0\n```",
        "mut" => "**mut** — marks a `let` binding as mutable, allowing reassignment.",
        "if" => "**if** — conditional expression/statement.\n```quantum\nif x > 0 {\n    println(\"positive\")\n} else {\n    println(\"non-positive\")\n}\n```",
        "else" => "**else** — alternative branch of an `if`.",
        "for" => "**for** — iterates over a range.\n```quantum\nfor i in 0..10 { println(i) }\nfor i in 1..=10 { println(i) } // inclusive\n```",
        "while" => "**while** — loops while a condition is true.\n```quantum\nlet mut i = 0\nwhile i < 10 { i = i + 1 }\n```",
        "return" => "**return** — returns a value from the enclosing function.",
        "struct" => "**struct** — defines a custom data type with named fields.\n```quantum\nstruct Point { x: float, y: float }\n```",
        "enum" => "**enum** — defines a type with multiple named variants.\n```quantum\nenum Color { Red, Green, Blue }\n```",
        "match" => "**match** — pattern matching, similar to a switch statement.\n```quantum\nmatch x {\n    0 => println(\"zero\"),\n    _ => println(\"other\")\n}\n```",
        "import" => "**import** — brings a module or its items into scope.\n```quantum\nimport quantumai as qai\n```",
        "true" | "false" => "**bool literal** — `true` or `false`.",
        "int" => "**int** — default integer type (maps to a 32-bit signed integer in generated code).",
        "float" => "**float** — default floating-point type (double precision).",
        "string" | "String" => "**string** — text type.",
        "bool" => "**bool** — boolean type, `true` or `false`.",

        // ── QuantumAI: data ──────────────────────────────────────
        "DataPipeline" => "**DataPipeline** — loads and prepares datasets.\n```quantum\nlet ds = DataPipeline::load(\"iris\")\n    .normalize()\n    .dataset();\n```\nBuilt-in datasets: `iris`, `xor`, `moons`, `circles`, `blobs`, `spiral`.",
        "Tensor" => "**Tensor** — a multi-dimensional array, the core data type for ML.\n```quantum\nlet t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);\n```",
        "train_test_split" => "**train_test_split(ratio)** — splits a dataset into train/test sets.\n```quantum\nlet (train, test) = ds.train_test_split(0.2);\n```",

        // ── QuantumAI: models ────────────────────────────────────
        "Model" => "**Model** — high-level model wrapper.\n```quantum\nlet mut model = Model::create(\"classifier\")\n    .epochs(150)\n    .learning_rate(0.01);\nmodel.train(&train.x, &train.y, 150, true);\n```",
        "Sequential" => "**Sequential** — build a neural network layer by layer.\n```quantum\nlet mut net = Sequential::new();\nnet.dense(4, 64, \"relu\");\nnet.dense(64, 3, \"softmax\");\nnet.compile(\"adam\", \"cross_entropy\", 0.01);\n```",
        "dense" => "**.dense(in, out, activation)** — adds a fully-connected layer to a `Sequential` network.\nActivations: `relu`, `gelu`, `softmax`, `sigmoid`, `tanh`.",
        "dropout" => "**.dropout(rate)** — adds a dropout layer for regularisation (helps prevent overfitting).",
        "compile" => "**.compile(optimizer, loss, lr)** — configures a `Sequential` network before training.\nOptimizers: `adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`, `nadam`.\nLosses: `cross_entropy`, `mse`, `mae`, `binary_crossentropy`, `huber`, `focal`.",
        "fit" => "**.fit(x, y, epochs, batch_size, verbose)** — trains a `Sequential` network.",
        "train" => "**.train(x, y, epochs, verbose)** — trains a `Model`.",
        "evaluate" => "**.evaluate(x, y)** — returns a map with `\"accuracy\"` and `\"loss\"` keys.",
        "predict" => "**.predict(x)** — returns the model's raw output tensor for input `x`.",
        "predict_class" => "**.predict_class(x)** — returns the predicted class index (argmax of output).",
        "save" => "**.save(path)** — saves model weights to a JSON file.",

        // ── QuantumAI: evaluation/diagnostics ────────────────────
        "BenchmarkReport" => "**BenchmarkReport::run(...)** — runs a full benchmark: accuracy, speed, confusion matrix, and a pass/fail verdict.\n```quantum\nlet report = BenchmarkReport::run(&mut model, &train.x, &train.y, &test.x, &test.y, n_classes, &class_names);\nreport.print(&class_names);\n```",
        "AIAssistant" => "**AIAssistant** — gives plain-English advice based on a `BenchmarkReport`.\n```quantum\nAIAssistant::analyze(&report);\n```",
        "ModelCard" => "**ModelCard::from_benchmark(model, report)** — generates a one-page model summary (strengths, weaknesses, hardware compatibility).",
        "CrossValidation" => "**CrossValidation::run(dataset, task, epochs, k)** — k-fold cross-validation for a reliable accuracy estimate.",
        "AnomalyDetector" => "**AnomalyDetector::new(n_features)** — an autoencoder-based anomaly detector.\n```quantum\nlet mut det = AnomalyDetector::new(n_features);\ndet.train_on_normal(&normal_data, 100);\nprintln(det.describe(&new_sample));\n```",
        "AutoMLPipeline" => "**AutoMLPipeline::run(dataset, task, classes, features)** — one call that loads data, tunes hyperparameters, trains, and benchmarks a model.",
        "feature_importance" => "**feature_importance(model, x, y, feature_names)** — returns features ranked by how much shuffling them hurts accuracy.",
        "PredictionExplainer" => "**PredictionExplainer::explain(model, sample, features, classes)** — explains why a model made a particular prediction.",

        _ => return None,
    })
}
