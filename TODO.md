# Data
- [ ] Color image $(RGB)$
- [ ] Depth image $(RGBA \to (D \in [0,1]))$
- [ ] Motion vectors (relative pixel velocity, EXR-2
  - Note: these store relative pixel velocity relative to the image bounds, i.e [-1,1]. 
- [ ] Jitter (sample position)
- [ ] 5th level 2d halton

# Model
- [ ] Jitter compensation
- [ ] Depth-informed dilation
- [x] Depth-to-space and space-to-depth
- [ ] Jitter-conditioned convolutions
- [ ] Core network 
- [x] Blending module
