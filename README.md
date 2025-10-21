# 🌍 WoW: World-Omniscient World Model

> Towards an Embodied, Physically-Consistent Generative World Model

[![arXiv](https://img.shields.io/badge/arXiv-2509.22642v1-b31b1b.svg)](https://arxiv.org/abs/2509.22642)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Website](https://img.shields.io/badge/demo-wow--world--model.github.io-green.svg)](https://wow-world-model.github.io)

WoW (World-Omniscient World Model) is a 14B-parameter generative world model trained on 2 million real-world robot interaction trajectories. It is designed for physically consistent imagination, reasoning, and action in robotics.

## 🔥 News!!
- We've updated the WoW-WAN2.1 Gradio ‘demo’ in the demo folder. A more user-friendly inference interface is now available. Just download the checkpoint and run the code to try it out!
- We release the DiT postraining checkpoints of WoW，includes DiT-2B based on Cosmos-Predict2, DiT-7B based on Cosmos-Predict1, and DiT-14B based on the Wan2.1

## 🧰 Quick Start

### 1. Install Dependencies

For Cosmos based DiT models:

```bash
pip install -r dit_models/wow-dit-2b/requires.txt
```

For Wan based models, follow the demo/README.md

### 2. Run Demo Scripts

#### A. DiT Scripts

- Example: Inference with 2B DiT model

```bash
python scripts/infer_wow_dit_2b.py --help
```

- For 7B model:

```bash
python scripts/infer_wow_dit_7b.py --help
```

#### B. Wan Scripts

- Run the Wan demo:

```bash
python demo/wan_infer_demo.py 
```

- For custom input or parameters, please refer to comments in the corresponding demo scripts.

---

## 🧠 Open-Source Weights & Datasets

We have released the following models and datasets on [Hugging Face](https://huggingface.co/WoW-world-model):

| Model Name | Parameters | Training Steps | Link | 
|------------|------------|----------------|------|
| WoW-1-DiT-2B-600k | 2B | 600k | [🔗 Link](https://huggingface.co/WoW-world-model/WoW-1-DiT-2B-600k) |
| WoW-1-DiT-7B-600k | 7B | 600k | [🔗 Link](https://huggingface.co/WoW-world-model/WoW-1-DiT-7B-600k) |
| WoW-1-Wan-14B-600k | 14B | 600k | [🔗 Link](https://huggingface.co/WoW-world-model/WoW-1-Wan-14B-600k) |
| WoW-1-Wan-14B-2M | 14B | 2M | [🔗 Coming Soon](https://huggingface.co/WoW-world-model/WoW-1-Wan-14B-2M) |
| Wan-1-Wan-1.3B-2M | 14B | 2M | [🔗 Coming Soon](https://huggingface.co/WoW-world-model/Wan-1-Wan-14B-600k) |

### 📊 Benchmark Dataset

| Dataset Name | Description | Link |
|--------------|-------------|------|
| WoW-1-Benchmark-Samples | Evaluation set for physical consistency and causal reasoning (WoWBench). | [📄 Link](https://huggingface.co/datasets/WoW-world-model/WoW-1-Benchmark-Samples) |

---

## 🚀 Open-Source Roadmap

> WoW is being released in phases to promote transparency and collaboration. Below is the current open-source progress:

### ✅ Phase 1 – Published
- [x] Paper released on [arXiv:2509.22642](https://arxiv.org/abs/2509.22642)
- [x] Project website launched: [wow-world-model.github.io](https://wow-world-model.github.io)

### 🚧 Phase 2 – Ongoing (Oct. 2025)
- [x] Model Weights (2B, 7B, 14B WoW-DiT)
- [x] Inference Scripts & Colab Demo
- [x] Baseline Inverse Dynamics Model
- [ ] Baseline Model Weights (SVD, CogVideoX, Cosmos1&2)

### 🚀 Phase 3 – Planned (Dec. 2025)
- [ ] 3D-Flow-Mask Inverse Dynamics Model
- [ ] Training Pipeline
- [ ] SOPHIA Framework Code
- [ ] WoWBench benchmark design & evaluation metrics

### 🌐 Phase 4 – 2026 Onward
- [ ] Continuous release of real/simulated trajectory data
- [ ] Expansion to multimodal inputs (audio, tactile, etc.)
- [ ] Universal fine-tuning API for downstream tasks
- [ ] Community challenges and leaderboard

---

## 🤝 Contributing
- Submit issues or feature requests
- Improve code or documentation
- Run experiments and submit results
- Contribute real-world robot data

---

## 📬 Contact
- Email: litwellchi@gmail.com
- Project website: [wow-world-model.github.io](https://wow-world-model.github.io)

---

## 📖 Citation

If you use WoW in your research, please cite:

```bibtex
@article{chi2025wow,
  title={WoW: Towards a World omniscient World model Through Embodied Interaction},
  author={Chi, Xiaowei and Jia, Peidong and Fan, Chun-Kai and Ju, Xiaozhu and Mi, Weishi and Qin, Zhiyuan and Zhang, Kevin  and Tian, Wanxin and Ge, Kuangzhi and Li, Hao and others},
  journal={arXiv preprint arXiv:2509.22642},
  year={2025}
}
```
