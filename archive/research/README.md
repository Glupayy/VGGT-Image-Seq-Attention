# Position encoding research notes

The original research manuscripts and figures are collected here without changing their claims:

- [VGGT position embedding research report](VGGT_POSITION_EMBEDDING_RESEARCH_REPORT.md)
- [Detailed 2D RoPE analysis](2D_ROPE_DETAILED_ANALYSIS.md)
- [Architecture and information flow](vggt_position_architecture.png)
- [RoPE mechanism](vggt_rope_mechanism.png)
- [Attention flow](vggt_attention_flow.png)

The two analysis scripts, [Chinese](position_analysis.py) and [English](position_analysis_en.py), generate explanatory figures using fixed example dimensions and **simulated attention**. Their outputs are illustrations, not measurements of a pretrained VGGT model, and do not establish accuracy of the attention visualizer. The historical manuscripts may contain assumptions or interpretations that need independent verification against the current code and model.

To regenerate figures, install the scripts' optional dependencies (`torch`, `numpy`, `matplotlib`, `seaborn`, and `pandas`), then run from this folder so generated files remain together:

```bash
cd archive/research
python position_analysis_en.py
```

These scripts are separate from the attention visualizer and its automated tests. See the project [README](../../README.md) for the supported workflow.
