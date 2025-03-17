<p align="center">
    <img src="https://github.com/LengSicong/MMR1/blob/main/assets/logo.png?raw=true" width="150" style="margin-bottom: 0.2;"/>
<p>

<h3 align="center"><a href="https://arxiv.org/" style="color:#9C276A">
MMR1: Advancing the Frontiers of Multimodal Reasoning</a></h3>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>


<h5 align="center">

[![hf_data](https://img.shields.io/badge/🤗-Dataset-9C276A.svg)](https://huggingface.co/datasets/MMR1/MMR1-Math-RL-Data-v0)
[![hf_checkpoint](https://img.shields.io/badge/🤗-Checkpoints-9C276A.svg)](https://huggingface.co/MMR1/MMR1-Math-v0-7B) 
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/LengSicong/MMR1/blob/main/LICENSE) <br>
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLengSicong%2FMMR1&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitor&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub issues](https://img.shields.io/github/issues/LengSicong/MMR1?color=critical&label=Issues)](https://github.com/LengSicong/MMR1/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/LengSicong/MMR1?color=success&label=Issues)](https://github.com/LengSicong/MMR1/issues?q=is%3Aissue+is%3Aclosed)  <br>
<!-- [![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/)
[![arXiv](https://img.shields.io/badge/Arxiv-xxx-AD1C18.svg?logo=arXiv)](https://arxiv.org/)  -->
</h5>


## 📰 News

* **[2025.03.11]**  🔥🔥 Release MMR1-Math-v0-7B, achieving SOTA with only **6k public training data**!


<!--## 🌟 Introduction-->
<h2><img src="https://github.com/LengSicong/MMR1/blob/main/assets/logo.png?raw=true" width="25"> Introduction</h2>
Introducing MMR1-Math-v0, a Large Multimodal Model specialized in mathematical tasks. Remarkably, MMR1-Math-v0 achieves state-of-the-art performance among open-source 7B multimodal models, competing effectively even against proprietary models with significantly larger parameter sizes—all trained using only 6k carefully curated data instances.

### 💡 Key Highlights:

- **SOTA Performance**: Sets a new **state-of-the-art** benchmark on math-related multimodal tasks among open-source 7B models.

- **Minimal Training Data**: Remarkably achieves top-tier performance with just **6k** high-quality samples from **public training datasets**.

- **Efficient Training with GRPO**: 6 hours of RL training with 64 H100s for 15 epochs.

- **Public and High-Quality Data**: Publicly sourced datasets, rigorously filtered and balanced across both difficulty and mathematical problem types.

- **Balanced Data Strategy**: Uniform sampling of data based on both task difficulty (filtering out overly simple problems) and mathematical reasoning diversity.


## ✅ Evaluation Results

We evaluated our model using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit/tree/main) on four mathematical reasoning benchmarks: MathVista_MINI, MathVision, LogicVista, and MathVerse_MINI.

We also include results on the MathVerse_MINI_Vision_Only_cot (MathVerse_V) subset to maintain consistency with the VLMEvalKit leaderboard. The table below compares our model's performance against various open-source and proprietary models.

| Model | size | MathVista | MathVision | LogicVista | MathVerse_V | MathVerse |
|-------|:----:|:--------------:|:----------:|:----------:|:--------------:|:-------------------:|
| **Close-sourced** | | | | | | |
| [GPT-4o 1120](https://openai.com/index/gpt-4o-system-card/)  | - | 60.0 | 31.2 | 52.8 | 40.6 | - |
| [Gemini-2.0-flash](https://deepmind.google/technologies/gemini/flash/) | - | 70.4 | 43.6 | 52.3 | 47.8 | - |
| [Claude3.7-Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) | - | 66.8 | 41.9 | 58.2 | 46.7 | - |
| **R1-related** | | | | | | |
| [LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT) | 11B | 52.5 | 19.9 | 39.6 | 22.6 | - |
| [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) | 7B | 60.6 | - | - | - | - |
| [Mulberry](https://github.com/HJYao00/Mulberry) | 7B | 63.1 | - | - | - | - |
| [LMM-R1](https://arxiv.org/abs/2503.07536) | 3B | 63.2 | 26.4 | - | - | 41.6 |
| [R1-Onevision](https://github.com/Fancy-MLLM/R1-Onevision?tab=readme-ov-file) | 7B | - | 26.2 | - | - | 44.1 |
| [MM-Eureka](https://github.com/ModalMinds/MM-EUREKA) | 8B | 67.1 | 22.2 | - | - | 40.4 |
| [MM-Eureka](https://github.com/ModalMinds/MM-EUREKA) | 38B | 64.2 | 26.6 | - | - | 48.9 |
| **Open-sourced** | | | | | | |
| [Ovis2-8b](https://github.com/AIDC-AI/Ovis) | 8B | 71.8 | 25.9 | 39.4 | 42.3 | - |
| [MiniCPM-o-2.6](https://github.com/OpenBMB/MiniCPM-o) | 8B | **71.9** | 21.7 | 36.0 | 35.0 | - |
| [VITA-1.5](https://github.com/VITA-MLLM/VITA) | 7B | 66.2 | 19.5 | 38.9 | - | 23.4 |
| [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) (official) | 7B | 68.2 | 25.4 | 47.9 | 41.1 | - |
| [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) (reproduced) | 7B | 67.5 | 25.6 | 46.8 | 42.5 | 46.9 |
| **Ours** | | | | | | |
| **MMR1-math-v0** | 7B | 71.0 | **30.2** | **50.8** | **45.1** | **49.8** |

### Ablation Studies
To further examine the effectiveness of GRPO, we perform ablation experiments by comparing our model with two SFT-based variants. Specifically, we fine-tune Qwen2.5-VL-7B on the 6k dataset using direct answer supervision (Qwen2.5-VL-sft) and chain-of-thought supervision (Qwen2.5-VL-sft-cot). 

| Model | size | MathVista | MathVision | LogicVista | MathVerse | MathVerse_V |
|-------|:----:|:--------------:|:----------:|:----------:|:--------------:|:-------------------:|
| Qwen2.5-VL-sft | 7B | 52.2 | 27.0 | 31.8 | 20.7 | 24.7 |
| Qwen2.5-VL-sft-cot | 7B | 54.7 | 23.4 | 33.8 | 23.7 | 25.7 |
| **MMR1-math-v0** | 7B | 71.0 | **30.2** | **50.8** | **45.1** | **49.8** |


## 🏫 Project Zoo
| Project | Latest Model | Checkpoints | Data | Link |
|-----------|--------------|-------|-----|-----------|
| MMR1-Math | MMR1-Math-v0 | [![hf_space](https://img.shields.io/badge/🤗-7B-9C276A.svg)](https://huggingface.co/MMR1/MMR1-Math-v0-7B) | [![hf_space](https://img.shields.io/badge/🤗-data-9C276A.svg)](https://huggingface.co/datasets/MMR1/MMR1-Math-RL-Data-v0) | [🔗](https://github.com/LengSicong/MMR1/tree/main/MMR1-Math) |
| MMR1-Science (coming soon!)| | | | |


## 💪 TODO
This project is under active development. Stay tuned for our upcoming updates! 
- [ ] Release data composition and preprocessing scripts.
- [ ] Release GRPO training scripts.
- [ ] Cold-start before RL training. Both dataset and checkpoint for cold-start will be released soon.
- [ ] More efficient GRPO training recipes. (Coming soon)
- [ ] More model sizes and variants.




## 🛠️ Requirements and Installation

Basic Dependencies:

* Python >= 3.10
* transformers>=4.49.0
* flash-attn>=2.4.3
* vllm>=0.7.3

Install required packages:

```bash
pip install -r requirements.txt
```

## 🤖 Inference
Here we show a code snippet to show you how to use MMR1-Math with `transformers` and `qwen_vl_utils`:

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "MMR1/MMR1-Math-v0-7B", 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
# default processer
processor = AutoProcessor.from_pretrained("MMR1/MMR1-Math-v0-7B")
# Example input
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "path/to/image.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
<details>
<summary>Batch inference</summary>

```python
# Sample messages for batch inference
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "What are the common elements in these pictures?"},
        ],
    }
]
messages2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]
# Combine messages for batch processing
messages = [messages1, messages2]
# Preparation for batch inference
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")
# Batch Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_texts)
```
</details>


## 🗝️ Training
Coming soon!


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
