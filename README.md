# Buckshot Roulette Value Learning

Learns the current player's win probability under the default rules of Buckshot Roulette PVP for VRChat v1.5. Game rules and search use C++20; the value network uses PyTorch.

## Search

Search enumerates all legal actions and chance outcomes until termination, replenishment on a turn change, or reload. Replenishment and reload outcomes are resolved before a frozen EMA network evaluates the resulting boundary states. Values are backed up using minimax decisions and probability-weighted expectations. Internal states provide training targets.

Unobserved ammunition is represented by a belief distribution over 18 hypotheses. Caching is local to each root search, and frontier inference is batched. Beyond the search boundary, values are approximated by the network.

## Network

The 43 features become 17 tokens: one global, two player, and 14 inventory tokens. C++ and Python share the input layout defined in [feature_schema.hpp](cpp/include/roulette/feature_schema.hpp).

The default Pre-LN Transformer has 8 layers, width 256, 8 attention heads, and FFN width 1024. Inventory tokens have no slot-position embeddings, making the value invariant to each player's inventory order. The global token produces 101 logits on the support $z_k=k/100$. With softmax probabilities $p_k$, the predicted value is $V(s)=\sum_{k=0}^{100}z_kp_k(s)$.

## Training objective

Each searched value becomes a two-hot target distribution $q$ on adjacent support points. For prediction $p$, the squared Cramér loss is:

```math
L=\frac{0.01}{B}\sum_{s=1}^{B}\sum_{k=0}^{99}
\left(F_{p_s}[k]-F_{q_s}[k]\right)^2
```

$F$ denotes the binwise CDF and $B$ the batch size. The final bin is excluded; bins are summed and states are averaged. Softmax, CDF, and loss use FP32. Targets are detached. AdamW updates the learner, followed by an EMA update.

## Run

Requires Python 3.11+ and a C++20 toolchain. On Windows, use Visual Studio's developer PowerShell. Resume uses the current Cramér checkpoint format.

```powershell
python -m pip install -e ".[test]"
python -m roulette train --config configs/cpu_pilot.json
python -m roulette train --config configs/cpu_pilot.json --resume
```

TensorBoard reports training loss and bootstrap residuals (MAE, MSE, P99, and maximum error). Only `latest.pt` is retained, with periodic and exit saves.

Training orchestration is in `train.py`; weight updates, run control, and checkpoint I/O are in `learner.py`, `run_control.py`, and `checkpoint.py`.
