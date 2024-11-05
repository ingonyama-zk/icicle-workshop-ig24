

# 🌲 ICICLE Workshop at Invisible Garden 2024 🌲

Welcome to the ICICLE Workshop! In this session, we'll dive into hands-on Rust examples provided in this repository. Get ready to explore, learn, and have fun with ICICLE. Here’s how to get started:

---

## 🚀 Getting Started

### Step 1: Clone the Repository
Start by cloning this repository to your local machine:

```bash
git clone https://github.com/ingonyama-zk/icicle-workshop-ig24.git
```

---

### Step 2: Install the CUDA Backend
Make sure to set up the CUDA backend to run the examples efficiently. Here’s how:

1. **Create a directory for the CUDA backend and navigate to it**:
   ```shell
   ~/home/icicle-workshop-ig24$ mkdir cuda_backend && cd cuda_backend
   ```
2. **Download the CUDA package:**
   ```bash
   ~/home/icicle-workshop-ig24/cuda_backend$ curl -L -O https://github.com/ingonyama-zk/icicle/releases/download/v3.0.0/icicle30-ubuntu22-cuda122.tar.gz
3. **Extract the package:**
   ```bash
   ~/home/icicle-workshop-ig24/cuda_backend$ tar -xzvf icicle30-ubuntu22-cuda122.tar.gz

---

### Step 3: Run Rust Examples
Let’s bring the ICICLE examples to life! Run these commands:

1. **Simple Arithmetic Example**:
   ```bash
   ~/home/icicle-workshop-ig24/rust/polynomials$ cargo run --package polynomial-icicle --example simple_arithmetic
2. **Complex Arithmetic Example (with specific configurations):**
   ```bash
   ~/home/icicle-workshop-ig24/rust/polynomials$ cargo run --package polynomial-icicle --example complex_arithmetic -- --max-ntt-log-size 22 --poly-log-size 18 --device-type "CUDA"





   
