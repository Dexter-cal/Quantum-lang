#!/bin/bash
# quantum-pkg -- The Quantum Language Package Manager
# Usage:
#   quantum-pkg install ai          # install QuantumAI
#   quantum-pkg install hashmap_str # install string hashmap
#   quantum-pkg list                # list installed packages
#   quantum-pkg search <query>      # search available packages
#   quantum-pkg new classifier      # create boilerplate project
#   quantum-pkg docs <topic>        # show documentation

QUANTUM_HOME="${QUANTUM_HOME:-$HOME/.quantum}"
PKG_DIR="$QUANTUM_HOME/packages"
STD_DIR="$(dirname "$0")/../qtm-std"
SCRIPT_DIR="$(dirname "$0")"

mkdir -p "$PKG_DIR"

# Package registry (built-in)
declare -A PACKAGES
PACKAGES["ai"]="QuantumAI ML/DL framework | qtm-std/ai.qtm"
PACKAGES["hashmap"]="Integer key hash map | qtm-std/hashmap.qtm"
PACKAGES["hashmap_str"]="String key hash map | qtm-std/hashmap_str.qtm"
PACKAGES["vec"]="Growable dynamic array | qtm-std/vec.qtm"
PACKAGES["math"]="Math functions | qtm-std/math.qtm"
PACKAGES["strings"]="String utilities | qtm-std/strings.qtm"
PACKAGES["collections"]="Array collections | qtm-std/collections.qtm"

cmd="$1"
shift

case "$cmd" in
  install)
    pkg="$1"
    if [ -z "$pkg" ]; then
      echo "Usage: quantum-pkg install <package>"
      exit 1
    fi
    if [ -n "${PACKAGES[$pkg]}" ]; then
      src="$STD_DIR/$pkg.qtm"
      if [ -f "$src" ]; then
        cp "$src" "$PKG_DIR/$pkg.qtm"
        echo "✓ Installed $pkg"
        echo "  Use: import $pkg"
      else
        echo "✗ Package file not found: $src"
        exit 1
      fi
    else
      echo "✗ Unknown package: $pkg"
      echo "  Run 'quantum-pkg search' to see available packages"
      exit 1
    fi
    ;;

  list)
    echo "Installed packages in $PKG_DIR:"
    if ls "$PKG_DIR"/*.qtm 2>/dev/null | head -20; then
      :
    else
      echo "  (none)"
    fi
    ;;

  search)
    query="${1:-}"
    echo "Available packages:"
    echo ""
    for name in "${!PACKAGES[@]}"; do
      info="${PACKAGES[$name]}"
      if [ -z "$query" ] || echo "$name $info" | grep -qi "$query"; then
        printf "  %-20s %s\n" "$name" "${info%%|*}"
      fi
    done
    ;;

  new)
    template="$1"
    project="${2:-my_quantum_project}"
    mkdir -p "$project"
    case "$template" in
      classifier)
        cat > "$project/main.qtm" << 'QTMEOF'
#link "quantumai_sys"
#lib-path "/usr/local/lib/quantum"
import ai as qai

fn main() {
    println("=== Quantum Classifier ===")
    let hw = qai.hw_detect()
    qai.hw_print(hw)

    // Load or create dataset
    let ds = qai.make_classification(500, 8, 4)
    qai.dataset_normalize(ds)
    qai.dataset_shuffle(ds)
    let train = qai.dataset_train_split(ds, 0.2)
    let test  = qai.dataset_test_split(ds, 0.2)

    // Build model
    let model = qai.seq_new()
    qai.seq_dense(model, 8, 64, activation: "relu")
    qai.seq_batchnorm(model, 64)
    qai.seq_dropout(model, 0.2)
    qai.seq_dense(model, 64, 32, activation: "relu")
    qai.seq_dense(model, 32, 4, activation: "softmax")
    qai.seq_compile(model, optimizer: "adam", loss: "cross_entropy", lr: 0.001)
    qai.seq_summary(model)

    // Train
    let pipe = qai.training_pipeline_new(model, "/tmp/training.json")
    qai.training_pipeline_early_stop(pipe, 10)
    qai.training_pipeline_fit(pipe, train, test, epochs: 100, batch: 32, lr: 0.001)

    // Evaluate
    println(qai.seq_accuracy(model, test))

    // Chat with your model
    let chat = qai.chat_new(model)
    qai.chat_set_system(chat, "I am a trained 4-class classifier.")
    qai.chat_respond(chat, "What did you learn?")

    // Cleanup
    qai.chat_free(chat)
    qai.dataset_free(ds)
    qai.dataset_free(train)
    qai.dataset_free(test)
    qai.training_pipeline_free(pipe)
    qai.hw_free(hw)
}
QTMEOF
        ;;
      chatbot)
        cat > "$project/main.qtm" << 'QTMEOF'
#link "quantumai_sys"
#lib-path "/usr/local/lib/quantum"
import ai as qai

fn main() {
    println("=== Quantum Chatbot ===")
    let model = qai.seq_new()
    qai.seq_dense(model, 64, 128, activation: "relu")
    qai.seq_dense(model, 128, 64, activation: "relu")
    qai.seq_dense(model, 64, 10, activation: "softmax")
    qai.seq_compile(model, optimizer: "adam", loss: "cross_entropy", lr: 0.001)

    let ds = qai.make_classification(200, 64, 10)
    let pipe = qai.training_pipeline_new(model, "/tmp/training.json")
    qai.training_pipeline_fit(pipe, ds, 0, epochs: 50, batch: 32, lr: 0.001)
    qai.training_pipeline_free(pipe)

    // Start interactive chat
    let chat = qai.chat_new(model)
    qai.chat_set_system(chat, "I am a trained conversational AI.")
    qai.chat_set_temp(chat, 0.7)
    qai.chat_loop(chat)  // reads from stdin until 'exit'
    qai.chat_free(chat)
    qai.dataset_free(ds)
}
QTMEOF
        ;;
      regression)
        cat > "$project/main.qtm" << 'QTMEOF'
#link "quantumai_sys"
#lib-path "/usr/local/lib/quantum"
import ai as qai

fn main() {
    println("=== Quantum Regression ===")
    let ds = qai.make_regression(500, 4, 0.1)
    qai.dataset_normalize(ds)
    let train = qai.dataset_train_split(ds, 0.2)
    let test  = qai.dataset_test_split(ds, 0.2)

    // Linear regression
    let lr = qai.linreg_new()
    qai.linreg_fit(lr, train)
    println(qai.linreg_r2(lr, test))

    // Neural regression
    let model = qai.seq_new()
    qai.seq_dense(model, 4, 32, activation: "relu")
    qai.seq_dense(model, 32, 1, activation: "linear")
    qai.seq_compile(model, optimizer: "adam", loss: "mse", lr: 0.01)
    let pipe = qai.training_pipeline_new(model, "/tmp/training.json")
    qai.training_pipeline_fit(pipe, train, test, epochs: 100, batch: 32, lr: 0.01)

    qai.linreg_free(lr)
    qai.dataset_free(ds)
    qai.dataset_free(train)
    qai.dataset_free(test)
    qai.training_pipeline_free(pipe)
}
QTMEOF
        ;;
      rnn)
        cat > "$project/main.qtm" << 'QTMEOF'
#link "quantumai_sys"
#lib-path "/usr/local/lib/quantum"
import ai as qai

fn main() {
    println("=== Quantum RNN / LSTM / GRU ===")
    // Download real sequence dataset
    let ok = qai.download_dataset("diabetes", "/tmp/data.csv")
    if ok {
        let ds = qai.load_csv("/tmp/data.csv", 8, 8, 0)
        qai.dataset_normalize(ds)
        let an = qai.analyze(ds)
        let model = qai.analysis_create_model(an)
        let lr = qai.analysis_lr(an)
        let ep = qai.analysis_epochs(an)
        qai.analysis_free(an)
        let trainer = qai.auto_train(model)
        qai.auto_train_target_acc(trainer, 0.70)
        qai.auto_train_max_epochs(trainer, 100)
        qai.auto_train_fit(trainer, ds)
        let acc = qai.auto_train_best_acc(trainer)
        println(acc)
        qai.auto_train_free(trainer)
        qai.dataset_free(ds)
    }
    // Also show LSTM directly
    let lstm = qai.lstm_new(8, 64, 2, 2)
    qai.lstm_summary(lstm)
    let gru = qai.gru_new(8, 32, 2)
    qai.gru_summary(gru)
    qai.lstm_free(lstm)
    qai.gru_free(gru)
}
QTMEOF
        ;;
      multimodal)
        cat > "$project/main.qtm" << 'QTMEOF'
#link "quantumai_sys"
#lib-path "/usr/local/lib/quantum"
import ai as qai

fn main() {
    println("=== Multi-Modal AI ===")
    let hw = qai.hw_detect()
    qai.hw_print(hw)

    let model = qai.multimodal_new("MyMultiModal", 256, 512, "text,audio,image")
    qai.multimodal_summary(model)

    let ds = qai.make_classification(100, 256, 5)
    qai.multimodal_train(model, ds, epochs: 20, lr: 0.001)

    println(qai.multimodal_gen_text(model, ds, 10))
    let audio = qai.multimodal_gen_audio(model, ds, 50)
    println(qai.tensor_len(audio))
    qai.tensor_free(audio)

    qai.multimodal_free(model)
    qai.dataset_free(ds)
    qai.hw_free(hw)
}
QTMEOF
        ;;
      *)
        cat > "$project/main.qtm" << 'QTMEOF'
fn main() {
    println("Hello from Quantum!")
    let x = 42
    let y = 3.14
    println(x)
    println(y)
}
QTMEOF
        ;;
    esac
    cat > "$project/README.md" << MDEOF
# $project

A Quantum language project.

## Run
\`\`\`bash
quantumc run main.qtm
\`\`\`

## Build
\`\`\`bash
quantumc compile main.qtm -o $project
\`\`\`
MDEOF
    echo "✓ Created project: $project/"
    echo "  Template: $template"
    echo "  Run: quantumc run $project/main.qtm"
    ;;

  docs)
    topic="${1:-intro}"
    case "$topic" in
      intro|overview)
        cat << 'DOCEOF'
╔══════════════════════════════════════════════════════════════╗
║              Quantum Language — Quick Reference              ║
╚══════════════════════════════════════════════════════════════╝

BASIC SYNTAX
  fn main() {                    // main function
      let x = 42                 // integer
      let y = 3.14               // float
      let mut z = "hello"        // mutable string
      println(x)                 // print
  }

CONTROL FLOW
  if x > 0 { ... } else { ... }
  while i < 10 { i += 1 }
  for item in collection { ... }

FUNCTIONS
  fn add(a: int, b: int) -> int { return a + b }
  fn greet(name: string, times: int) { ... }
  greet("Alice", times: 3)       // named args

STRUCTS (including generic)
  struct Point { x: float, y: float }
  struct Pair<A, B> { first: A, second: B }
  let p = Pair { first: 42, second: 3.14 }

ERROR HANDLING
  let r: Result<int, string> = Ok(42)
  let v = r.unwrap()             // panics if Err
  let v = r.unwrap_or(0)        // safe default
  let v = r?                     // propagate error

COLLECTIONS
  import vec
  let mut v = vec_new()
  v = vec_push(v, 10)

  import hashmap
  let mut m = map_new()
  m = map_insert(m, 1, 100)

  import hashmap_str
  let sm = strmaps_new()
  strmaps_insert(sm, "key", 42)

AI (import ai as qai)
  let model = qai.create("classifier")
  let ds = qai.make_classification(1000, 8, 4)
  qai.model_train(model, ds, epochs: 100)
  println(qai.seq_accuracy(model, ds))

Run 'quantum-pkg docs <topic>' for:
  types, functions, structs, errors, ai, fileio, generics
DOCEOF
        ;;
      ai)
        cat << 'DOCEOF'
╔══════════════════════════════════════════════════════════════╗
║                  QuantumAI — Quick Reference                 ║
╚══════════════════════════════════════════════════════════════╝

GLOSSARY (beginner-friendly)
  Token       - A single word/piece the model reads at once
  Embedding   - A word stored as a list of numbers (its meaning as coordinates)
  Epoch       - One full pass through all training data
  Batch size  - How many samples the model sees before updating weights
  Loss        - How wrong the model is (lower = better)
  Accuracy    - How often the model is right (higher = better)
  Gradient    - Direction to adjust weights to reduce loss
  Overfitting - Model memorizes training data, fails on new data
  Dropout     - Randomly turn off neurons to prevent overfitting
  LoRA        - Fine-tune only small adapter layers, not the whole model
  RAG         - Retrieve relevant documents before generating a response
  MoE         - Mixture of Experts: route each token to specialized sub-models

AI WORKFLOW
  1. Get data:    qai.make_classification() or qai.hf_stream()
  2. Clean data:  qai.cleaner_new() → qai.cleaner_run()
  3. Build model: qai.seq_new() → qai.seq_dense() → qai.seq_compile()
  4. Train:       qai.training_pipeline_fit()
  5. Evaluate:    qai.seq_accuracy()
  6. Chat:        qai.chat_new() → qai.chat_respond()
  7. Deploy:      qai.seq_save() → load on any machine

MODEL TYPES
  "classifier"  - Predict which category (e.g. spam/not spam)
  "regressor"   - Predict a number (e.g. house price)
  "binary"      - Two-class prediction (yes/no)
  "autoencoder" - Compress and reconstruct data
  "nlp"         - Text understanding

ACTIVATIONS
  "relu"    - ReLU: max(0, x). Most common hidden layer activation
  "sigmoid" - Squashes to 0-1. Good for binary output
  "softmax" - Squashes to probabilities summing to 1. Multi-class output
  "tanh"    - Squashes to -1 to 1. Good for audio/image

OPTIMIZERS
  "adam"    - Adaptive learning rate. Best default choice
  "sgd"     - Stochastic Gradient Descent. Simple, requires tuning
  "rmsprop" - Good for recurrent networks

LOSSES
  "mse"            - Mean Squared Error. Regression problems
  "cross_entropy"  - Classification problems

2026 TECHNIQUES
  qai.lora_new()       - Fine-tune with <1% of parameters
  qai.rag_new()        - Add a knowledge base to any model
  qai.moe_new()        - Scale to billions of parameters efficiently
  qai.diffusion_new()  - Generate images/audio/video
  qai.mamba_new()      - Alternative to transformers for long sequences
DOCEOF
        ;;
      token|embedding|gradient)
        cat << 'DOCEOF'
BEGINNER GLOSSARY — Machine Learning Terms

TOKEN
  A token is the smallest unit a language model reads.
  "Hello world" might be 2 tokens: ["Hello", "world"]
  or 3 tokens: ["Hel", "lo", "world"] depending on the tokenizer.
  Your model sees numbers, not words — each token is converted
  to a number, then to an embedding vector.

EMBEDDING
  An embedding is how a word (or token) is stored as a list of numbers.
  "king" might be [0.72, -0.45, 0.18, 0.91, -0.33, ...]
  The amazing part: king - man + woman ≈ queen in vector space.
  Words with similar meanings cluster near each other.
  In Quantum: qai.show_vector("king", vec_buf, 16)

GRADIENT
  A gradient tells the model WHICH WAY to adjust its weights to reduce error.
  Imagine you're blindfolded on a hill trying to walk downhill —
  the gradient points in the steepest downhill direction.
  Training = repeatedly following gradients to reduce loss.

WEIGHT
  A weight is a number in the model that gets adjusted during training.
  A model with 1M parameters has 1M weights.
  Training adjusts all these weights to minimize the loss.

LOSS
  Loss = how wrong the model is on average.
  Loss 0.0 = perfect predictions (rare/impossible on real data)
  Loss 1.0 = very wrong
  Your goal: minimize loss on the test set (not just training set)

OVERFITTING
  When a model gets too good at training data but fails on new data.
  Like a student who memorized past exams but can't solve new problems.
  Fix: dropout, more data, regularization, early stopping.
DOCEOF
        ;;
      *)
        echo "Available topics: intro, ai, token, embedding, gradient"
        echo "Run: quantum-pkg docs <topic>"
        ;;
    esac
    ;;

  *)
    echo "Quantum Package Manager"
    echo ""
    echo "Commands:"
    echo "  quantum-pkg install <pkg>        Install a package"
    echo "  quantum-pkg list                 List installed packages"
    echo "  quantum-pkg search [query]       Search available packages"
    echo "  quantum-pkg new <template> [dir] Create project from template"
    echo "  quantum-pkg docs <topic>         Show documentation"
    echo ""
    echo "Templates: classifier, chatbot, regression, multimodal"
    echo "Topics:    intro, ai, token, embedding, gradient"
    ;;
esac
