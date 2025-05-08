# CoherentBEVGen: Temporally Consistent Multi-View BEV Generation from Single Front View using Conditional Diffusion

This repository contains the codebase for our AAAI 2025 project: generating temporally coherent multi-view BEV representations from monocular front-view images via memory-augmented conditional diffusion. Our goal is to simulate rare long-tail driving scenarios from limited input to support robust motion planning in simulators like `highway-env`.

---

## 🧠 Project Highlights

- **Single-View to BEV**: Lift single front-view image to partial BEV using LSS-style depth lifting.
- **Memory-Augmented Temporal Fusion**: Fuse BEV from past frames using ego-motion and self-attention.
- **Conditional Masked Diffusion**: Complete BEV in occluded regions via lightweight DDIM in BEV space.
- **Cycle-Consistency Supervision**: Enforce geometric consistency by re-rendering the front view.
- **Downstream Simulation**: Output multiple plausible full BEVs per frame to generate highway-env training scenarios.


