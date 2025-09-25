<p align="center">
    <img src="https://github.com/LengSicong/MMR1/blob/main/assets/logo.png?raw=true" width="150" style="margin-bottom: 0.2;"/>
<p>

<h3 align="center"><a href="https://arxiv.org/" style="color:#9C276A">
MMR1: Enhancing Multimodal Reasoning with Variance-Aware Sampling and Open Resources</a></h3>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>


<h5 align="center">

[![hf_data](https://img.shields.io/badge/🤗-Dataset-9C276A.svg)](https://huggingface.co/MMR1/datasets)
[![hf_checkpoint](https://img.shields.io/badge/🤗-Checkpoints-9C276A.svg)](https://huggingface.co/MMR1/models) 
[![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/)
[![arXiv](https://img.shields.io/badge/Arxiv-xxx-AD1C18.svg?logo=arXiv)](https://arxiv.org/)
<br>
</h5> 

## 📰 News
* **[2025.09.25]**  🔥🔥 Release [technical report](assets/report.pdf)!
* **[2025.09.25]**  🚀🚀 Release MMR1-SFT (~16M) and MMR1-RL (15k) datasets!
* **[2025.09.25]**  🚀🚀 Release MMR1-3B and MMR1-7B, 32B checkpoint are on the way!
* **[2025.03.11]**  🔥🔥 Release MMR1-Math-v0-7B, achieving SOTA with only **6k public training data**!

<!--## 🌟 Introduction-->
<h2><img src="https://github.com/LengSicong/MMR1/blob/main/assets/logo.png?raw=true" width="25"> Introduction</h2>

This repo introduces our work on enhancing multimodal reasoning models. Current progress is limited by:  

- ❌ **Lack of open, large-scale, high-quality long chain-of-thought (CoT) data**  
- ❌ **Instability of RL fine-tuning**, where standard GRPO often suffers from *gradient vanishing* under low reward variance  

### 🔑 Our Contributions  
- **Variance-Aware Sampling (VAS):**  
  A new data selection strategy guided by the *Variance Promotion Score (VPS)*. VAS combines outcome variance and trajectory diversity to promote reward variance, stabilize policy optimization, and improve convergence.  

- **Large-scale curated resources:**  
  - ~1.6M long CoT cold-start trajectories with verified short answer
  - ~15k RL QA pairs  
  - Designed for **quality, difficulty, and diversity**  

- **Open-source codebase & models:**  
  - Fully reproducible end-to-end training pipeline  
  - Released models at multiple scales as standardized baselines for multimodal reasoning  

Please refer to our [TRAIN.md](TRAIN.md) for detailed instructions on training with VAS.

## 💡 Methodology Overview
Our method introduces **Variance-Aware Sampling (VAS)** to address the *gradient vanishing problem* in reinforcement learning with Group Relative Policy Optimization (GRPO).  

<p align="center">
<img src="assets/fig1.png" alt="Overview of the VAS framework" width="700"/>
</p>

### 🔹 Framework  
As illustrated in **Figure 1**, training begins with a pool of prompts from the dataset:  
1. A **random sampler** provides uniform coverage of data.  
2. A **weighted sampler**, guided by Variance Promotion Score (VPS), prioritizes prompts with higher reward variance and trajectory diversity.  
3. These two sources are combined to form training batches, balancing exploration and coverage.  
4. The policy model generates rollouts, which are evaluated with rewards and used to update the policy. VPS scores are periodically re-estimated as the model improves, ensuring dynamic adaptation.  

This design ensures that training consistently focuses on prompts that provide strong learning signals, while still maintaining sufficient randomness for coverage.  

<p align="center">
<img src="assets/algo1.png" alt="algo" width="700"/>
</p>

### 🔹 Algorithm  
**Algorithm 1** provides a step-by-step description of VAS within the GRPO framework:  
- **Initialization:** For each prompt, multiple rollouts are sampled to estimate pass rate, outcome variance (OVS), trajectory diversity (TDS), and VPS.  
- **Periodic VPS update:** At specified intervals, these statistics are refreshed to reflect the evolving policy.  
- **Batch construction:** A mixture of prompts is drawn—some uniformly at random, others proportionally to VPS—controlled by the mixture ratio λ.  
- **Policy optimization:** Rollouts are generated for the selected prompts, GRPO loss is computed, and the policy parameters are updated accordingly.  

By adaptively steering training toward prompts with higher reward variance, VAS effectively stabilizes optimization and amplifies gradient signals, enabling more efficient and robust learning.  

## 📦 Open Resources

We release the following resources for the community:
- **[MMR1-SFT](https://huggingface.co/datasets/MMR1/MMR1-SFT) (~16M):**  Supervised fine-tuning dataset with 16M long CoT cold-start trajectories (Gemini2.5 Pro/Flash) with verified short answer (GPT-4o) 
- **[MMR1-RL](https://huggingface.co/datasets/MMR1/MMR1-RL) (15k):** RL dataset with 15k question-answer pairs (GPT-4o)
- **[MMR1-3B-SFT](https://huggingface.co/MMR1/MMR1-3B-SFT):** 3B checkpoint trained with MMR1-SFT
- **[MMR1-3B-RL](https://huggingface.co/MMR1/MMR1-3B-RL):** 3B checkpoint trained with MMR1-SFT and MMR1-RL
- **[MMR1-7B-SFT](https://huggingface.co/MMR1/MMR1-7B-SFT):** 7B checkpoint trained with MMR1-SFT
- **[MMR1-7B-RL](https://huggingface.co/MMR1/MMR1-7B-RL):** 7B checkpoint trained with MMR1-SFT and MMR1-RL
- **[MMR1-32B-SFT](https://huggingface.co/MMR1/MMR1-32B-SFT):** 32B checkpoint trained with MMR1-SFT
- **[MMR1-32B-RL](https://huggingface.co/MMR1/MMR1-32B-RL):** 32B checkpoint trained with MMR1-SFT and MMR1-RL (On the way!)


<p align="center">
<img src="assets/data.png" alt="data" width="700"/>
</p>

The dataset spans diverse domains—including mathematics, science, charts/figures, document tables, and general understanding—covering ~1.6M math samples and an additional ~37K samples across other domains. It integrates existing public resources (e.g., MathVerse, ScienceQA, ChartQA, DocVQA, GQA) together with newly curated and self-collected data, ensuring quality, difficulty, and diversity. This collection establishes one of the most comprehensive open resources for multimodal reasoning models.
We hope these resources can serve as a benchmark for the community and facilitate the research of multimodal reasoning.

## 📊 Evaluation Results  

We evaluate our models on a suite of **mathematics-related multimodal reasoning benchmarks** (MathVerse, MathVista, MathVision, LogicVista, and ChartQA).  

<p align="center">
<img src="assets/result.png" alt="result" width="700"/>
</p>

- **MMR1-7B-RL** achieves an average score of **58.4**, establishing new state-of-the-art performance among 7B-scale reasoning models.  
- **MMR1-3B-RL** performs competitively with **52.7**, showing strong reasoning ability even at smaller scale.  
- Our models consistently outperform or match larger baselines, demonstrating the effectiveness of **Variance-Aware Sampling (VAS)** and our curated **long CoT training data**.  

## 🔍 Analysis of VAS Training Dynamics  

We further analyze the effectiveness of **Variance-Aware Sampling (VAS)** through training efficiency and the evolution of **Variance Promotion Score (VPS)**.  

<p align="center">
<img src="assets/anal1.png" alt="anal1" width="700"/>
</p>    

**Training Efficiency (Fig. 2).**  
- **Gradient norm**: VAS substantially amplifies gradient magnitudes compared to the vanilla baseline, mitigating the gradient vanishing issue. This indicates that VAS consistently provides stronger optimization signals.  
- **Clip fraction**: Higher clipping fractions in VAS runs suggest that policy updates are closer to the trust-region boundary, enabling more effective utilization of the learning signal without destabilizing training.  
- **Validation accuracy**: Both full VAS (λ = 1.0) and mixed VAS–random sampling (λ = 0.5) converge faster and achieve higher final accuracy than the baseline, demonstrating that VAS improves both efficiency and performance. Notably, the mixed strategy achieves competitive results while maintaining broader data coverage.  

<p align="center">
<img src="assets/anal2.png" alt="anal2" width="700"/>
</p>

**VPS Dynamics (Fig. 3).**  
- **Score distribution**: VPS distributions evolve from relatively uniform at the beginning of training to more concentrated in the middle bins, suggesting convergence in identifying consistently informative prompts.  
- **Weight transitions**: Transition matrices show that many prompts shift across bins over time, with both upward and downward movements, reflecting the dynamic nature of reward variance as the policy evolves. Early transitions are more widespread, while later updates become more stable, consistent with convergence.  
- **Interpretation**: This dynamic reweighting ensures that the model continually prioritizes prompts with higher variance while still allowing redistribution as learning progresses, preventing overfitting to a static subset of data.  

👉 Together, these analyses highlight how **VAS effectively mitigates gradient vanishing, improves sample efficiency, and adapts dynamically to the evolving training landscape.**

## 🎨 Qualitative Demo

To illustrate the reasoning capability of our models, we provide qualitative examples from **MathVerse**.  
The demo showcases how the model carefully analyzes the problem, plans a structured solution, executes step-by-step reasoning, verifies results, and even provides alternative solution paths.  

<p align="center">
<img src="assets/demo.png" alt="demo" width="700"/>
</p>

This demonstrates the model’s ability to maintain logical consistency, perform reflective verification, and present human-readable reasoning traces.

## 🤝 Contribution and Contact
This project is still under active development. Community feedback and contributions are highly appreciated. If you want to contribute, please feel free to make a pull request or create an issue.

If you have any questions or would like to engage with our community, feel free to scan the QR code below to join our WeChat group.


## 👍 Acknowledgement
Our MMR1 is build on top of [Qwen2.5VL](https://github.com/QwenLM/Qwen2.5-VL), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [EasyR1](https://github.com/hiyouga/EasyR1/tree/main).
Besides, our MMR1 benefits from tons of open-source efforts. We sincerely appreciate these efforts and compile a list in [ACKNOWLEDGEMENT.md](https://github.com/LengSicong/MMR1/blob/main/ACKNOWLEDGEMENT.md) to express our gratitude. If your work is used in MMR1 but not mentioned in either this repo or the technical report, feel free to let us know :heart:.

<details open><summary>💡 Some other multimodal-LLM projects from our team may interest you ✨. </summary><p>

> [**VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding**](https://github.com/DAMO-NLP-SG/VideoLLaMA3) <br>
> Boqiang Zhang<sup>* </sup>, Kehan Li<sup>* </sup>, Zesen Cheng<sup>* </sup>, Zhiqiang Hu<sup>* </sup>, Yuqian Yuan<sup>* </sup>, Guanzheng Chen<sup>* </sup>, Sicong Leng<sup>* </sup>, Yuming Jiang<sup>* </sup>, Hang Zhang<sup>* </sup>, Xin Li<sup>* </sup>, Peng Jin, Wenqi Zhang, Fan Wang, Lidong Bing, Deli Zhao <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/DAMO-NLP-SG/VideoLLaMA3)  [![github](https://img.shields.io/github/stars/DAMO-NLP-SG/VideoLLaMA3.svg?style=social)](https://github.com/DAMO-NLP-SG/VideoLLaMA3) [![arXiv](https://img.shields.io/badge/Arxiv-2501.13106-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.13106) <br>

> [**VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs**](https://github.com/DAMO-NLP-SG/VideoLLaMA2) <br>
> Zesen Cheng*, Sicong Leng*, Hang Zhang*, Yifei Xin*, Xin Li*, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, Lidong Bing <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/DAMO-NLP-SG/VideoLLaMA2)  [![github](https://img.shields.io/github/stars/DAMO-NLP-SG/VideoLLaMA2.svg?style=social)](https://github.com/DAMO-NLP-SG/VideoLLaMA2) [![arXiv](https://img.shields.io/badge/Arxiv-2406.07476-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.07476) <be> 

> [**VCD: Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding**](https://arxiv.org/abs/2311.16922) <br>
> Sicong Leng*, Hang Zhang*, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, Lidong Bing <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/DAMO-NLP-SG/VCD)  [![github](https://img.shields.io/github/stars/DAMO-NLP-SG/VCD.svg?style=social)](https://github.com/DAMO-NLP-SG/VCD)  [![arXiv](https://img.shields.io/badge/Arxiv-2311.16922-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2311.16922) <br>

> [**The Curse of Multi-Modalities: Evaluating Hallucinations of Large Multimodal Models across Language, Visual, and Audio**](https://arxiv.org/abs/2410.12787) <br>
> Sicong Leng*, Yun Xing*, Zesen Cheng*, Yang Zhou, Hang Zhang, Xin Li, Deli Zhao, Shijian Lu, Chunyan Miao, Lidong Bing <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/DAMO-NLP-SG/CMM)  [![github](https://img.shields.io/github/stars/DAMO-NLP-SG/CMM.svg?style=social)](https://github.com/DAMO-NLP-SG/CMM)  [![arXiv](https://img.shields.io/badge/Arxiv-2410.12787-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.12787) <br>

> [**Breaking the Memory Barrier: Near Infinite Batch Size Scaling for Contrastive Loss**](https://arxiv.org/abs/2410.17243) <br>
> Zesen Cheng*, Hang Zhang*, Kehan Li*, Sicong Leng, Zhiqiang Hu, Fei Wu, Deli Zhao, Xin Li, Lidong Bing <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/DAMO-NLP-SG/Inf-CLIP)  [![github](https://img.shields.io/github/stars/DAMO-NLP-SG/Inf-CLIP.svg?style=social)](https://github.com/DAMO-NLP-SG/Inf-CLIP)  [![arXiv](https://img.shields.io/badge/Arxiv-2410.17243-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.17243) <br>

> [**VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM**](https://arxiv.org/abs/2501.00599) <br>
> Yuqian Yuan, Hang Zhang, Wentong Li, Zesen Cheng, Boqiang Zhang, Long Li, Xin Li, Deli Zhao, Wenqiao Zhang, Yueting Zhuang, Jianke Zhu, Lidong Bing <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/DAMO-NLP-SG/VideoRefer)  [![github](https://img.shields.io/github/stars/DAMO-NLP-SG/VideoRefer.svg?style=social)](https://github.com/DAMO-NLP-SG/VideoRefer)  [![arXiv](https://img.shields.io/badge/Arxiv-2501.00599-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.00599) <br>

</p></details>

## 📑 Citation

If you find MMR1 useful for your research and applications, please cite using this BibTeX:

```bibtex
@misc{MMR1-Math2025,
  title={MMR1: Advancing the Frontiers of Multimodal Reasoning},
  author={Sicong Leng*, Jing Wang*, Jiaxi Li*, Hao Zhang*, Zhiqiang Hu, Boqiang Zhang, Hang Zhang, Yuming Jiang, Xin Li, Deli Zhao, Fan Wang, Yu Rong, Aixin Sun†, Shijian Lu†},
  year={2025},
  howpublished={\url{https://github.com/LengSicong/MMR1}},
}
```

## 🔒 License

This project is released under the Apache 2.0 license as found in the LICENSE file.
The service is a research preview intended for **non-commercial use ONLY**, subject to the model Licenses of Qwen, Terms of Use of the data generated by OpenAI and Gemini, and Privacy Practices of ShareGPT. Please get in touch with us if you find any potential violations.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=LengSicong/MMR1&type=Date)](https://star-history.com/#LengSicong/MMR1&Date) 
