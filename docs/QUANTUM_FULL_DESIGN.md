# Quantum / qai — Full Design Document

A programming language (**Quantum**, compiles to C, built in Rust) with a built-in AI development framework (**qai**). Core philosophy: **AI development should be simple, fast, and never a black box.** Nothing is hardcoded — every technique, layer, bridge, and visualization is an open, overridable interface.

---

## 1. Layers of AI Development (Pipeline)

- **Data layer** — collection, cleaning, labeling, streaming
- **Model layer** — architecture (techniques, layers)
- **Training layer** — the actual learning process
- **Reasoning/inference layer** — test-time compute, chain-of-thought
- **Agent layer** — tool-calling, planning, multi-step tasks
- **Evaluation layer** — accuracy, F1, RMSE, benchmarks
- **Deployment/serving layer** — APIs, edge/server targets
- **Monitoring/MLOps layer** — drift detection, versioning
- **Application layer** — chat interfaces, end-user products

---

## 2. Model Types / Techniques to Support

**Classical ML:** Linear/Logistic Regression, Decision Trees, Random Forests, Gradient Boosting (XGBoost/LightGBM), SVM, K-Means, Naive Bayes

**Deep Learning:** CNN, RNN/LSTM/GRU, Transformer, MoE (Mixture of Experts), GAN, Autoencoder/VAE, Diffusion, GNN, Reinforcement Learning (Q-learning, PPO, RLHF), Multimodal models

**2026 landscape notes:**
- MoE is now the dominant architecture for frontier models
- Test-time compute / reasoning models are a new scaling axis
- Agentic AI (tool use, autonomous multi-step tasks) is the biggest usage shift
- Multimodal generation matured fast
- Physical AI / robotics moving from research to real deployment
- Open-weight models are closing the gap with closed/proprietary ones

---

## 3. Core Language/Framework Philosophy

- **Nothing hardcoded** — every technique, layer, bridge, optimizer, and visualization is defined via an open interface
- **Interpretability by default** — every model type supports inspection/explanation out of the box
- **Fewer lines than Python** — no heavy import chains; common techniques/layers available with zero imports
- **Compiled to C via Rust** — fast, with compile-time shape/architecture checking

### The `Technique` Interface

```
technique Technique {
    fn forward(input) -> output
    fn objective(output, target) -> Signal
    fn update(signal)
    fn dashboard() -> DashboardLayout
}
```

Built-ins (Regression, CNN, Transformer, MoE, Diffusion, etc.) are just pre-shipped implementations of this same interface — no special compiler treatment.

---

## Key Capabilities Summary

This document describes 115 sections covering:

### Core Features
- **Open interfaces** — Technique, Type, Bridge, Environment, Input/Output, custom tools
- **Logic system** — real control flow inside models, weight mutation, self-improvement loops
- **Memory management** — three-tier (weights/working/retrieval), dynamic region loading, smart triage
- **Skills system** — composable, attachable capabilities without base model retraining
- **Interpretability** — explanation, visualization, weight inspection, formula rendering
- **Experiment tracking** — full reproducibility, run comparison, checkpoint management

### Training & Learning
- **Multiple learning techniques** — supervised, self-supervised, reinforcement, online, transfer, few-shot
- **Advanced training strategies** — progressive resizing, SWA, population-based, test-time adaptation
- **Data pipeline** — streaming, balancing, augmentation, versioning, golden eval sets
- **Hardware awareness** — auto-adapt, distributed training, quantization-aware, mixed precision

### Deployment & Operations
- **Serving infrastructure** — REST/gRPC/WebSocket, API keys, MCP exposure, peer-to-peer
- **Export formats** — `.qmodel` (with logic), GGUF (weights-only), WASM, chip/ASIC burning
- **Monitoring** — behavior contracts, drift detection, cost tracking, incident management
- **CI/CD integration** — automated testing gates, evidence binding, fault tolerance

### Developer Experience
- **Unified CLI** — `qai run`, `qai models`, `qai connect`, experiment management
- **Interactive development** — notebook-style, live training dashboard, logic sandbox
- **Marketplace** — discoverable techniques, skills, bridges; community-contributed components
- **Learning curriculum** — 10-module path from "hello world" to production deployment

---

## Architecture Highlights

### Technique/Type System (Open Interface Pattern)
Every AI technique is defined by the same interface, enabling user-authored custom types alongside built-ins:
- No compiler magic for any specific architecture
- Users write custom techniques the exact same way built-ins are defined
- Fork-and-protect system prevents accidentally breaking originals while customizing

### Logic Blocks (Real Language Construct)
```
model.logic {
    input(prompt)
    let draft = model.generate(prompt)
    while model.confidence(draft) < 0.9 {
        draft = model.generate(prompt, previous_attempt: draft)
    }
    output(draft)
}
```
- Compose multiple models
- Implement retry/refinement loops
- Trigger retraining on demand
- Access/modify weights directly
- Route based on real conditions

### Three-Tier Memory System
```
model.memory.weights         // Permanent, changes only via training
model.memory.working         // Live context window, per-session
model.memory.retrieval       // External, searchable, fast to update
```
- Each tier independently optional
- Dynamic region loading (avoid loading unused weights)
- Smart triage (relevance scoring, archiving)
- Transparent to the developer

### Skills (Attachable Capabilities)
```
let rigging_skill = qai.skill.train(base: model, data: rigging_data)
model.attach_skill(rigging_skill)
model.generate("a walking character").rig()
```
- Small, trained adapters, not full models
- Base model weights never touched during skill training
- Attach/detach/version/compare skills independently
- Auto-check catastrophic forgetting after attach

---

## Implementation Priorities (MVP Path)

Based on the design, the recommended first-phase focus:

### Phase 1 (Foundation — 2-3 months)
1. Core language (variables, functions, control flow, types)
2. Basic `Technique` interface + Regression implementation
3. Simple training loop with `.train()` / `.test()`
4. Minimal dashboard (loss/accuracy graphs)
5. Export to GGUF (weights-only)
6. CLI: `qai build`, `qai train`, `qai run`

### Phase 2 (Extensibility — 2-3 months)
7. Custom `Technique` authoring
8. CNN and Transformer as built-in Techniques
9. Logic blocks (basic composition)
10. Simple `Bridge` and custom `Environment`
11. Skills system (attach/detach)
12. Experiment tracking table

### Phase 3 (Intelligence — 2-3 months)
13. MoE and Diffusion Techniques
14. Full logic system (weight access, retraining)
15. Multi-model self-play
16. Advanced memory (region loading, triage)
17. Marketplace/registry infrastructure
18. Full experiment dashboard

### Phase 4 (Production — ongoing)
19. Serving infrastructure (REST/gRPC)
20. Federated learning
21. Chip/ASIC export
22. Advanced training strategies
23. Real-world deployment integrations

---

## Why This Design

### Differentiation from Existing Tools

**vs. PyTorch/TensorFlow:**
- Everything is a native language primitive (no "layers library" needed)
- Interpretability first, not an afterthought (`.explain()` on every output)
- Compiled performance
- Open-interface pattern prevents feature bloat

**vs. LangGraph/AutoGPT/CrewAI:**
- Those orchestrate *around* frozen models
- Quantum logic reaches *inside* (weights, architecture, retraining)
- Weights can mutate based on logic conditions
- Single language for model definition + orchestration

**vs. LLaMA/Ollama/vLLM:**
- Those are inference runtimes
- Quantum is full language + training framework
- User-authored custom architectures, not preset menu

### Core Principle: Open Interface Everywhere

Instead of accumulating special cases, one pattern scales:
- `technique Technique { }` — any architecture
- `bridge Bridge { }` — any model-to-model communication
- `environment Environment { }` — any execution context (VM, game engine, robot)
- `input Input { }` / `output Output { }` — any data format
- `tool Tool { }` — any external action
- Custom hooks in logic blocks — user-defined event points

This consistency is the real power, not any single feature.

---

## Critical Implementation Decisions Still Needed

1. **MVP scope** — which ~20 of these 115 sections actually ship in v0.1?
2. **Primary audience** — beginners or researchers?
3. **Compilation target** — which C compiler/platform first (LLVM, GCC, bare metal)?
4. **Self-hosting** — can Quantum code itself be written in Quantum early?
5. **Community/governance** — single author, open-source from day one, or stealth until MVP?

---

## This Document

- Sections 1–47: Load-bearing core (techniques, framework, training, interpretability)
- Sections 48–80: Production features (federation, serving, memory, weights)
- Sections 81–115: Advanced patterns (logic, automation, research, chip export)

Treat sections 1–47 as the reference for v0.1; everything after is a genuine but optional backlog.

**Next step:** pick one small vertical slice (one Technique, full train → test → serve cycle) and get it working end-to-end before pulling in more of this backlog.
