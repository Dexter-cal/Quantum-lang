# Quantum Programming Language v2.1

A compiled, statically-typed, systems programming language with **first-class AI/ML capabilities**.

```
██████  ██    ██  █████  ███    ██ ████████ ██    ██ ███    ███
██    ██ ██    ██ ██   ██ ████   ██    ██    ██    ██ ████  ████
██    ██ ██    ██ ███████ ██ ██  ██    ██    ██    ██ ██ ████ ██
██    ██ ██    ██ ██   ██ ██  ██ ██    ██    ██    ██ ██  ██  ██
██████   ██████  ██   ██ ██   ████    ██     ██████  ██      ██
                                    v2.1
```

## Features

### Language
- **Compiled to native code** via C backend — fast, no VM, no GC pauses
- **Static typing** with inference — catch bugs at compile time
- **Rust-inspired syntax** — safe, expressive, familiar
- Generics, traits, enums, pattern matching, closures
- Async/await, channels, threads
- Module system with imports
- Full standard library

### QuantumAI Framework (built-in ML/AI)
- **Neural Networks** — Sequential model, real backpropagation
- **9 Activation functions** — ReLU, GELU, Swish, Mish, SELU, ELU, LeakyReLU, Sigmoid, Tanh
- **10 Loss functions** — MSE, MAE, CrossEntropy, Focal, Huber, KL-Divergence, LogCosh...
- **9 Optimizers** — Adam, AdamW, Nadam, AMSGrad, SGD+momentum, RMSProp, Adagrad, Adadelta
- **LR Schedulers** — StepLR, ExponentialLR, CosineAnnealing, ReduceOnPlateau, Warmup+Cosine
- **Layers** — Dense, BatchNorm, LayerNorm, Dropout, Embedding
- **Traditional ML** — LinearRegression, Ridge, LogisticRegression, SVM, KNN, DecisionTree, RandomForest, KMeans, PCA, NaiveBayes
- **NLP** — Tokenizer, TF-IDF, N-grams, cosine similarity
- **Pre-trained models** — ResNet50, BERT-tiny, GPT-2 mini, YOLOv8, EfficientNet-B0, T5-small, Whisper-tiny
- **AutoML** — automatic model search + hyperparameter grid search
- **Datasets** — XOR, sine, classification, moons, circles, CSV loading

---

## Quick Start

### Install
```bash
# Build from source (requires Rust + cargo + gcc)
git clone https://github.com/quantum-lang/quantum
cd quantum
cargo build --release -p quantum-compiler
# Binary: ./target/release/quantumc
```

### Hello World
```quantum
fn main() {
    println("Hello, Quantum World!")
}
```
```bash
quantumc run hello.qtm
# Hello, Quantum World!
```

### Neural Network (3 lines)
```quantum
import quantumai as qai
import quantumai.data as data

fn main() {
    let model = qai.create("classifier")
    let ds    = data.make_classification(n: 500, features: 10, classes: 4)
    model.fit(ds.x, ds.y, epochs: 100)

    let ev = model.evaluate(ds.x, ds.y)
    println("Accuracy: " + ev["accuracy"])
}
```

### Custom Deep Network
```quantum
import quantumai as qai

fn main() {
    let model = qai.Sequential::new()
    model.dense(64, 256, "relu")
    model.batchnorm(256)
    model.dropout(0.3)
    model.dense(256, 128, "gelu")
    model.dense(128, 10, "softmax")
    model.compile("adamw", "cross_entropy", lr: 0.001)
    model.summary()
    println("Model ready!")
}
```

### Traditional ML
```quantum
import quantumai.ml as ml
import quantumai.data as data

fn main() {
    let ds     = data.make_classification(300, 6, 3)
    let labels = [0, 1, 2]  // class labels

    let rf = ml.RandomForest::new(n_trees: 100, max_depth: 8)
    rf.fit(ds.x, labels, n_classes: 3)
    println("RandomForest acc: " + rf.accuracy(ds.x, labels))

    let km = ml.KMeans::new(k: 3)
    let clusters = km.fit(ds.x)
    println("KMeans inertia: " + km.inertia(ds.x))
}
```

---

## Compiler Commands

```
quantumc run <file>          Compile and execute
quantumc compile <file>      Compile to binary
quantumc check <file>        Type-check only (no compile)
quantumc new <name>          Create new project
quantumc new <name> --ai     Create AI project template
quantumc new <name> --lib    Create library project
quantumc build               Build current project
quantumc build --release     Optimized release build
quantumc repl                Start interactive REPL

Options:
  -o <file>     Output filename
  -O0 / -O3     Optimization level (0=debug, 3=max)
  --debug       Include debug symbols
  --emit-c      Keep generated C file
```

---

## Project Structure

```
quantum-lang/
├── compiler/           # quantumc compiler
│   └── src/
│       ├── main.rs     # CLI entry point
│       ├── lexer.rs    # tokenizer
│       ├── parser.rs   # AST parser
│       ├── ast.rs      # AST nodes
│       ├── typechecker.rs  # type system
│       ├── mir.rs      # MIR lowering
│       ├── codegen.rs  # C code generation
│       └── error.rs    # error reporting
├── quantumai/          # ML/AI framework
│   └── src/
│       └── lib.rs      # 2200+ line framework
├── stdlib/             # standard library
├── examples/           # .qtm example files
│   ├── hello.qtm
│   ├── features.qtm
│   └── ai_example.qtm
├── docs/
│   ├── LANGUAGE_REFERENCE.md
│   └── QUANTUMAI_REFERENCE.md
└── README.md
```

---

## Compilation Pipeline

```
source.qtm
    │
    ▼ Lexer       — tokenize source → token stream
    │
    ▼ Parser      — token stream → AST
    │
    ▼ TypeChecker — AST → typed AST (type inference + checking)
    │
    ▼ MIR         — typed AST → MIR (basic blocks, explicit control flow)
    │
    ▼ Optimizer   — dead code elimination, constant folding (O0–O3)
    │
    ▼ CodeGen     — MIR → C source code
    │
    ▼ GCC/Clang   — C → native binary
    │
  binary
```

---

## Documentation
- [Language Reference](docs/LANGUAGE_REFERENCE.md) — full syntax guide
- [QuantumAI Reference](docs/QUANTUMAI_REFERENCE.md) — ML/AI API

---

## License
MIT © Quantum Contributors
