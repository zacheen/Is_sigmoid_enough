# SESSIONS

- **2026-03-27: explore-sigmoid-saturation-cases** — Designed 4 new test cases (XOR, Binary Encoding, Staircase, 2D Checkerboard) to explore sigmoid saturation in different problem geometries. Key insight: Scale factor must be proportional to problem boundary count to show measurable differences in weight behavior.

# LESSONS

## Sigmoid Saturation & ScaledSigmoid Trade-offs

**Problem:** Standard sigmoid forces weights to grow very large when the network must output near 0 or 1, causing vanishing gradients. ScaledSigmoid extends the output range to reduce this pressure, but the benefit depends on problem structure.

**Mitigation:** Test with multiple scale factors and measure both **convergence speed** (early epochs) and **final weight magnitude**. Larger boundary counts require larger scale factors to see measurable differences.

**Lesson learned:** Not all saturation problems are equal — a `scale=1.05` improvement that's visible on 2-boundary problems may be invisible on 3-boundary problems. Match scale factor to problem complexity: more boundaries = larger scale needed.

## Task Complexity Matters More Than Architecture Depth

**Problem:** MNIST (LeNet-5) showed ~98.8% accuracy for all activation variants despite simpler architecture. CIFAR-10 (VGG) showed fluctuating results even with deeper network, suggesting task difficulty (not just depth) drives activation differences.

**Mitigation:** Choose test tasks that actually force saturation — binary outputs, hard boundaries, multi-output patterns. Simple tasks (MNIST) won't reveal activation differences; harder tasks (CIFAR-10 or synthetic boundary problems) show clearer separation.

**Lesson learned:** Architecture depth alone doesn't guarantee sigmoid problems emerge; task complexity is equally important. A shallow network on a hard task can show bigger differences than a deep network on an easy task.
