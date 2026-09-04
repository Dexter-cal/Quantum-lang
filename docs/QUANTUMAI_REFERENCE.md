# QuantumAI v2.1 — Complete ML/AI Framework Reference

QuantumAI is the built-in ML/AI library for the Quantum Programming Language.
It provides neural networks, traditional ML, NLP, AutoML, and pre-trained models.

---

## Quick Start

```quantum
import quantumai as qai
import quantumai.data as data

// 3-liner model creation
let model = qai.create("classifier")
let ds    = data.make_classification(n: 500, features: 10, classes: 4)
model.fit(ds.x, ds.y, epochs: 80)
```

---

## 1. Tensor

The core n-dimensional array type.

```quantum
// Creation
let t1 = qai.Tensor::new([1.0, 2.0, 3.0, 4.0], shape: [2, 2])
let t2 = qai.Tensor::zeros([4, 4])
let t3 = qai.Tensor::ones([3, 3])
let t4 = qai.Tensor::randn([100, 10])   // Gaussian random
let t5 = qai.Tensor::xavier(64, 128)    // Xavier init
let t6 = qai.Tensor::he(64, 128)        // He/Kaiming init

// Properties
t.shape           // [2, 2]
t.numel()         // 4
t.shape_str()     // "[2, 2]"
t.mean()          // scalar mean
t.std_dev()       // standard deviation
t.min()           // minimum value
t.max()           // maximum value
t.sum()           // sum of all elements
t.argmax()        // index of max value
t.argmin()        // index of min value
t.norm_l2()       // L2 norm

// Math ops
a.matmul(b)           // matrix multiply
a.add_broadcast(b)    // element-wise add (broadcast)
a.sub(b)              // element-wise subtract
a.mul_elem(b)         // element-wise multiply
a.scale(2.0)          // scalar multiply
a.transpose()         // transpose 2D
a.reshape([n, m])     // reshape
a.clip(lo, hi)        // clamp values
a.clip_grad_norm(5.0) // clip by L2 norm

// Activations
t.relu()              // max(0, x)
t.sigmoid()           // 1/(1+e^-x)
t.tanh_act()          // tanh
t.softmax()           // softmax
t.log_softmax()       // log-softmax
t.gelu()              // Gaussian GELU
t.swish()             // x * sigmoid(x)
t.mish()              // x * tanh(softplus(x))
t.leaky_relu(0.01)    // max(alpha*x, x)
t.elu(1.0)            // exp(x)-1 for x<0
t.selu()              // scaled ELU
```

---

## 2. Activation Functions

```quantum
let act = qai.Activation::from_str("relu")  // create by name

// Available activations:
"relu"        // ReLU
"sigmoid"     // Sigmoid
"tanh"        // Tanh
"softmax"     // Softmax
"log_softmax" // Log-Softmax
"gelu"        // GELU (BERT/GPT style)
"swish"       // Swish / SiLU
"mish"        // Mish
"leaky_relu"  // Leaky ReLU (alpha=0.01)
"elu"         // ELU (alpha=1.0)
"selu"        // SELU (self-normalizing)
"linear"      // Identity (no activation)

let out = act.apply(tensor)
let d   = act.derivative(tensor)  // for backprop
println(act.name())                // "relu"
```

---

## 3. Layers

### Dense (Fully Connected)
```quantum
// In model builder:
model.dense(in_features: 128, out_features: 256, activation: "relu")

// Layer properties:
// - He initialization by default
// - Optional bias (default: true)
// - Correct Xavier init for sigmoid/tanh
```

### Batch Normalization
```quantum
model.batchnorm(features: 256)
// - Normalizes over batch dimension
// - Learnable gamma and beta parameters
// - eps = 1e-5 by default
```

### Layer Normalization
```quantum
model.layernorm(features: 256)
// - Normalizes over feature dimension (transformer-style)
// - Better for NLP tasks
```

### Dropout
```quantum
model.dropout(rate: 0.3)
// - Inverted dropout (scales by 1/(1-rate) during training)
// - Disabled during inference automatically
```

### Embedding
```quantum
model.embedding(vocab_size: 10000, embed_dim: 128)
// - Token index → dense vector
// - Mean-pools across sequence dimension
// - Good for NLP input
```

---

## 4. Sequential Model

```quantum
// Build a model
let model = qai.Sequential::new()
model.dense(64, 256, "relu")
model.batchnorm(256)
model.dropout(0.3)
model.dense(256, 128, "gelu")
model.layernorm(128)
model.dropout(0.2)
model.dense(128, 10, "softmax")

// Compile (set optimizer, loss, learning rate)
model.compile("adam", "cross_entropy", lr: 0.001)

// Print architecture
model.summary()

// Train
model.fit(x_train, y_train,
    epochs: 100,
    batch_size: 32,
    verbose: true)

// Evaluate
let metrics = model.evaluate(x_test, y_test)
println("loss=" + metrics["loss"])
println("acc="  + metrics["accuracy"])
println("mse="  + metrics["mse"])
println("mae="  + metrics["mae"])

// Predict
let pred       = model.predict(x)           // raw output tensor
let cls        = model.predict_class(x)     // argmax class index
let proba      = model.predict_proba(x)     // softmax probabilities

// Save / Load
model.save_weights("model.json")
model.load_weights("model.json")

// Configuration
model.grad_clip = 5.0    // gradient clipping (default: 1.0)
model.l2_reg    = 0.01   // L2 regularization (default: 0.0)
```

---

## 5. Loss Functions

```quantum
let loss_fn = qai.Loss::from_str("cross_entropy")
let value   = loss_fn.compute(predictions, targets)
let grad    = loss_fn.gradient(predictions, targets)

// Available losses:
"mse"                  // Mean Squared Error
"mae"                  // Mean Absolute Error
"rmse"                 // Root MSE
"cross_entropy"        // Categorical Cross-Entropy
"binary_crossentropy"  // Binary Cross-Entropy
"nll"                  // Negative Log-Likelihood
"huber"                // Huber Loss (delta=1.0)
"focal"                // Focal Loss (alpha=0.25, gamma=2.0)
"kl_divergence"        // KL Divergence
"logcosh"              // Log-Cosh Loss
```

---

## 6. Optimizers

```quantum
// Set during model.compile()
model.compile("adam", "cross_entropy", lr: 0.001)

// Available optimizers:
"adam"       // Adam (beta1=0.9, beta2=0.999, eps=1e-8)
"adamw"      // Adam with weight decay (wd=0.01)
"nadam"      // Nesterov Adam
"amsgrad"    // AMSGrad (Adam variant)
"sgd"        // SGD with momentum (momentum=0.9)
"sgd_nesterov" // Nesterov SGD
"rmsprop"    // RMSProp (rho=0.9)
"adagrad"    // Adagrad
"adadelta"   // Adadelta (rho=0.95)

// All optimizers include:
// - Global gradient norm clipping (clip=5.0)
// - Proper bias correction (Adam variants)
// - State reset between model compiles
```

---

## 7. LR Schedulers

```quantum
// Step decay
let sched = qai.LRScheduler::StepLR { step_size: 30, gamma: 0.1 }

// Exponential decay
let sched = qai.LRScheduler::ExponentialLR { gamma: 0.95 }

// Cosine annealing
let sched = qai.LRScheduler::CosineAnnealingLR { t_max: 100, eta_min: 1e-6 }

// Reduce on plateau
let sched = qai.LRScheduler::ReduceLROnPlateau { patience: 10, factor: 0.5 }

// Warmup + cosine (transformer training)
let sched = qai.LRScheduler::WarmupCosine {
    warmup_steps: 100,
    total_steps: 1000,
    eta_min: 1e-6
}

// Use with compile
model.compile_with_scheduler("adamw", "cross_entropy", lr: 0.001, scheduler: sched)
```

---

## 8. Datasets

```quantum
import quantumai.data as data

// Built-in datasets
let xor       = data.make_xor(n: 200)
let sine      = data.make_sine(n: 300)
let classify  = data.make_classification(n: 500, features: 10, classes: 4)
let moons     = data.make_moons(n: 300, noise: 0.15)
let circles   = data.make_circles(n: 300, noise: 0.1, factor: 0.5)

// From file
let csv_ds    = data.Dataset::from_csv("data/train.csv")

// Dataset operations
ds.normalize()                       // z-score normalisation
ds.shuffle()                         // random shuffle
ds.info()                            // print stats
let (train, test) = ds.train_test_split(test_ratio: 0.2)

// Dataset properties
ds.n_samples     // number of rows
ds.n_features    // number of input features
ds.n_classes     // number of output classes
ds.x             // input tensor
ds.y             // label tensor (one-hot)
```

---

## 9. Traditional ML Algorithms

```quantum
import quantumai.ml as ml

// ── Supervised: Regression ────────────────────────────────
let lr = ml.LinearRegression::new()
lr.lr = 0.01
lr.epochs = 1000
lr.fit(x, y)
let pred = lr.predict(x_test)
println("R² = " + lr.r2_score(x_test, y_test))
println("MSE = " + lr.mse(x_test, y_test))

let ridge = ml.Ridge::new(alpha: 0.1)
ridge.fit(x, y)

// ── Supervised: Classification ────────────────────────────
let log_reg = ml.LogisticRegression::new()
log_reg.lr = 0.1
log_reg.l2 = 0.01
log_reg.fit(x, y_binary)
println("acc = " + log_reg.accuracy(x_test, y_test))
let proba = log_reg.predict_proba(x_test)

// K-Nearest Neighbours
let knn = ml.KNN::new(k: 5).with_metric("euclidean")  // or "manhattan", "cosine"
knn.fit(x_train, labels)
let preds = knn.predict(x_test)
println("acc = " + knn.accuracy(x_test, test_labels))

// Decision Tree (CART)
let dt = ml.DecisionTree::new(max_depth: 6)
dt.min_samples = 3
dt.fit(x, labels, n_classes: 4)
println("acc = " + dt.accuracy(x_test, test_labels))

// Support Vector Machine (linear)
let svm = ml.SVM::new(C: 1.0)
svm.lr = 0.01
svm.epochs = 1000
svm.fit(x, y_binary_float)   // y values: +1.0 or -1.0
println("acc = " + svm.accuracy(x_test, y_test))

// Naive Bayes (Gaussian)
let nb = ml.NaiveBayes::new()
nb.fit(x, labels, n_classes: 3)
println("acc = " + nb.accuracy(x_test, test_labels))

// ── Ensemble ──────────────────────────────────────────────
let rf = ml.RandomForest::new(n_trees: 100, max_depth: 8)
rf.fit(x, labels, n_classes: 4)
println("acc = " + rf.accuracy(x_test, test_labels))

// ── Unsupervised ──────────────────────────────────────────
let km = ml.KMeans::new(k: 3)
km.max_iter = 100
let cluster_labels = km.fit(x)
println("inertia = " + km.inertia(x))
let test_clusters = km.predict(x_test)

let pca = ml.PCA::new(n_components: 2)
let x_reduced = pca.fit_transform(x)  // [n, 64] → [n, 2]

// ── Metrics ───────────────────────────────────────────────
let acc = ml.accuracy_score(pred_labels, true_labels)
let cm  = ml.confusion_matrix(pred_labels, true_labels, n_classes: 4)
let prf = ml.precision_recall_f1(pred_labels, true_labels, n_classes: 4)
let r2  = ml.r2_score_vec(pred_values, true_values)
let mae = ml.mean_absolute_error(pred_values, true_values)
```

---

## 10. NLP Utilities

```quantum
import quantumai.nlp as nlp

// ── Tokenizer ─────────────────────────────────────────────
let tok = nlp.Tokenizer::new(max_vocab: 10000)
tok.fit(["hello world", "quantum is fast", ...])
println("vocab size = " + tok.vocab_size())

let ids      = tok.encode("hello quantum", max_len: 20)
let text     = tok.decode(ids)
let batch    = tok.encode_batch(texts, max_len: 20)

// Special tokens (auto-added):
// <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3, <MASK>=4

// ── TF-IDF ────────────────────────────────────────────────
let (tfidf_matrix, vocab) = nlp.tfidf(texts)
// Returns dense matrix [n_docs, vocab_size] with TF-IDF weights

// ── N-grams ───────────────────────────────────────────────
let bigrams  = nlp.ngrams("hello world quantum", n: 2)
let trigrams = nlp.ngrams("hello world quantum lang", n: 3)

// ── Similarity ────────────────────────────────────────────
let sim = nlp.cosine_similarity(vec_a, vec_b)  // -1.0 to 1.0
```

---

## 11. Pre-trained Architectures

```quantum
// Vision
let resnet  = qai.pretrained.resnet50(pretrained: true, classes: 1000)
let effnet  = qai.pretrained.efficientnet_b0(classes: 10)
let yolo    = qai.pretrained.yolov8_nano()         // object detection

// NLP
let bert    = qai.pretrained.bert_tiny(pretrained: true)   // classification
let gpt2    = qai.pretrained.gpt2_mini()                   // generation
let t5      = qai.pretrained.t5_small()                    // seq2seq

// Audio
let whisper = qai.pretrained.whisper_tiny()               // speech recognition

// Fine-tune any model
let model = qai.pretrained.bert_tiny(pretrained: true)
model.layers.last_mut().out_features = 5    // change head
model.compile("adamw", "cross_entropy", lr: 2e-5)
model.fit(x_train, y_train, epochs: 3)
```

---

## 12. AutoML

```quantum
// Automatic model selection
let auto = qai.AutoClassifier::new(trials: 6)
auto.fit(dataset)
println("Best accuracy: " + auto.best_accuracy)
let best_model = auto.best_model

// Access trial results
for result in auto.results {
    println(result.config + " → " + result.accuracy)
}

// Hyperparameter grid search
let best_acc = qai.HyperparamSearch::grid_search(
    lrs: [0.01, 0.001, 0.0001],
    hidden_sizes: [32, 64, 128, 256],
    dataset: ds
)
```

---

## 13. Convenience API

```quantum
// One-liner model creation for common tasks:
let clf  = qai.create("classifier")    // 10-class, Adam, CrossEntropy
let reg  = qai.create("regressor")     // MSE, Adam
let nlpm = qai.create("nlp")           // Embedding + GELU + Softmax, AdamW
let ae   = qai.create("autoencoder")   // Encoder-decoder, MSE
let bin  = qai.create("binary")        // Binary classification, BCE

// Full pipeline example:
let ds    = data.make_classification(500, 8, 4)
let model = qai.create("classifier")
model.fit(ds.x, ds.y, epochs: 100)
let ev    = model.evaluate(ds.x, ds.y)
println("Final accuracy: " + ev["accuracy"])
```

---

## 14. Training History

```quantum
model.fit(x, y, epochs: 100, batch_size: 32)

// Access history after training
model.history.losses          // [f64] per epoch
model.history.accuracies      // [f64] per epoch
model.history.val_losses      // validation losses
model.history.val_accuracies  // validation accuracies

// Utilities
model.history.best_epoch()    // epoch with lowest loss
model.history.plot_ascii()    // ASCII loss curve
model.history.print_report()  // full training report
```
