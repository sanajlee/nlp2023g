---
layout: default
---

# 👾108.536A Studies on Computational Linguistics II💻
## Transformer-based Pre-Trained Models and Prompt Tuning


### Course Information

* Instructor: Sangah Lee (sanalee@snu.ac.kr)

This course deals with the Transformer-based pre-trained language models and application fields focused on Prompt Tuning. Students should present some of the selected papers on relevant topics of the course, and finally, implement a system or write a conference-level paper based on the course content. Students should have taken the course [Computational Linguistics I] before or should be familiar with the relevant content, including Python and PyTorch.

* Paper list for presentation
  * https://docs.google.com/spreadsheets/d/1D0_wUdGtKQsha5TZqU58BWh5vqYF5LuEt7L-iFmOa3k/edit#gid=0


### Resources
* [Jurafsky and Martin (2023 draft), “Speech and Language Processing”](https://web.stanford.edu/~jurafsky/slp3/)
* [Practical Deep Learning with PyTorch](https://www.deeplearningwizard.com/deep_learning/course_progression/)
* [Jupyter Notebook](https://jupyter.org/)
  * [Jupyter notebook for beginners-A tutorial](https://towardsdatascience.com/jupyter-notebook-for-beginners-a-tutorial-f55b57c23ada)
* [Google Colabatory](https://colab.research.google.com/notebooks/welcome.ipynb)
  * [Primer for Learning Google CoLab](https://medium.com/dair-ai/primer-for-learning-google-colab-bb4cabca5dd6)
  * [Google Colab - Quick Guide](https://www.tutorialspoint.com/google_colab/google_colab_quick_guide.htm)


### Syllabus

* **Week 0 (3/2 Thu)** Course Introduction
  * [slide](https://github.com/sanajlee/nlp2023g/raw/master/nlp0_courseintro.pdf)

* **Week 1 (3/7, 3/9)** Paradigms of NLP
  * [Pre-Trained Models: Past, Present and Future](https://arxiv.org/pdf/2106.07139.pdf)
  * [Natural Language Processing: the Age of Transformers](https://towardsdatascience.com/natural-language-processing-the-age-of-transformers-a36c0265937d)
  * [Paradigm Shift in Natural Language Processing](https://arxiv.org/pdf/2109.12575.pdf)

* **Week 2 (3/14, 3/16)** Attention
  * [Introduction to Attention Mechanism](https://ai.plainenglish.io/introduction-to-attention-mechanism-bahdanau-and-luong-attention-e2efd6ce22da)
  * [Attn: Illustrated Attention](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)
  * [Attention and Memory in Deep Learning and NLP](https://dennybritz.com/posts/wildml/attention-and-memory-in-deep-learning-and-nlp/)
  * [PyTorch: Translation with Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

* **Week 3 (3/21, 3/23)** Transformer
  * [Vaswani et al. (2017), Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
  * [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  * [Transformers Illustrated!](https://tamoghnasaha-22.medium.com/transformers-illustrated-5c9205a6c70f)
  * Seq2seq pay Attention to Self Attention: [Part 1](https://bgg.medium.com/seq2seq-pay-attention-to-self-attention-part-1-d332e85e9aad) [Part 2](https://bgg.medium.com/seq2seq-pay-attention-to-self-attention-part-2-cf81bf32c73d)
  * [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

* **Week 4 (3/28, 3/30)** Transformer-based Language Models
  * [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)
  * [FROM Pre-trained Word Embeddings TO Pre-trained Language Models — Focus on BERT](https://towardsdatascience.com/from-pre-trained-word-embeddings-to-pre-trained-language-models-focus-on-bert-343815627598)
  * Dissecting BERT [Part 1](https://medium.com/@mromerocalvo/dissecting-bert-part1-6dcf5360b07f) [Part 2](https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73) [Part 3](https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f)
  * [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
  * [BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)
  * [The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2/)
  * [How GPT3 Works - Visualizations and Animations](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)
  * [Using BERT for the First Time](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

* **Week 5 (4/4, 4/6)** Transformer-based Language Models (Improvements)
  * Unified sequence modeling
    * XLNet: [paper](https://arxiv.org/abs/1906.08237), [code](processing/2019/09/11/xlnet/)
    * UniLM: [paper](https://proceedings.neurips.cc/paper/2019/file/c20bb2d9a50d5ac1f713f8b34d9aac5a-Paper.pdf), [code](https://github.com/microsoft/unilm)
    * GLM: [paper](https://aclanthology.org/2022.acl-long.26/), [code](https://github.com/THUDM/GLM)
  * Applying generalized encoder-decoder
    * MASS: [paper](http://proceedings.mlr.press/v97/song19d/song19d.pdf), [code](https://github.com/microsoft/MASS)
    * T5: [paper](https://arxiv.org/abs/1910.10683), [code](https://github.com/google-research/text-to-text-transfer-transformer)
    * BART: [paper](https://aclanthology.org/2020.acl-main.703/), [code](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.md)
    * PEGASUS: [paper](https://proceedings.mlr.press/v119/zhang20ae.html), [code](https://github.com/google-research/pegasus)
    * PALM: [paper](https://aclanthology.org/2020.emnlp-main.700/), [code](https://github.com/overwindows/PALM)
  * Dealing with longer texts
    * Transformer-XL: [paper](https://aclanthology.org/P19-1285), [code](https://github.com/kimiyoung/transformer-xl)
  * More variants
    * SpanBERT: [paper](https://aclanthology.org/2020.tacl-1.5/), [code](https://github.com/facebookresearch/SpanBERT)
    * ELECTRA: [paper](https://openreview.net/pdf?id=r1xMH1BtvB), [code](https://github.com/google-research/electra)


* **Week 6 (4/11, 4/13)** Transformer-based Language Models (Applications: tasks)
  * Dealing with sentence pair
    * Sentence Transformers: [paper](https://aclanthology.org/D19-1410), [code](https://github.com/UKPLab/sentence-transformers)
  * Summarization
    * ASPECTNEWS: Aspect-Oriented Summarization of News Documents: [paper](https://aclanthology.org/2022.acl-long.449/), [code](https://github.com/oja/aosumm)
    * Summarization of Podcast Transcripts: [paper](https://aclanthology.org/2022.acl-long.302/), [code](https://github.com/tencent-ailab/GrndPodcastSum)
    * Long document Summarization <https://aclanthology.org/2022.emnlp-main.692/>
    * Discourse-Aware Neural Extractive Text Summarization <https://aclanthology.org/2020.acl-main.451/>
  * Classification
    * Automatic Identification and Classification of Bragging in Social Media: [paper](https://aclanthology.org/2022.acl-long.273/), [data](https://archive.org/details/bragging_data)
    * Aspect Sentiment Classification <https://aclanthology.org/2020.acl-main.338/>
    * Clinical Document Classification <https://aclanthology.org/2021.emnlp-main.361/>
  * Named Entity Recognition
    * Leveraging Type Descriptions for Zero-shot Named Entity Recognition and Classification <https://aclanthology.org/2021.acl-long.120/>
  * Question-Answering
    * Dual Reader-Parser on Hybrid Textual and Tabular Evidence for Open Domain Question Answering <https://aclanthology.org/2021.acl-long.315/>
    * Few-Shot Question Answering by Pretraining Span Selection <https://aclanthology.org/2021.acl-long.239/>
    * DeFormer <https://aclanthology.org/2020.acl-main.411/>
  * Generation
    * Generating Scientific Claims for Zero-Shot Scientific Fact Checking: [paper](https://aclanthology.org/2022.acl-long.175/), [code](https://github.com/allenai/scientific-claim-generation)
    * Polyjuice: Generating Counterfactuals for Explaining, Evaluating, and Improving Models <https://aclanthology.org/2021.acl-long.523/>
    * Long Text Generation by Modeling Sentence-Level and Discourse-Level Coherence <https://aclanthology.org/2021.acl-long.499/>
  * Others
    * Coreferential Reasoning Learning for Language Representation <https://aclanthology.org/2020.emnlp-main.582/>
    * Implicit discourse relation recognition <https://aclanthology.org/2021.emnlp-main.187/>
    * Text Detoxification, Style Transfer <https://aclanthology.org/2021.emnlp-main.629/>
    * Syntactically-Informed Unsupervised Paraphrasing with Non-Parallel Data <https://aclanthology.org/2021.emnlp-main.203/>
    * Semantic Role Labeling <https://aclanthology.org/2020.emnlp-main.319/>
    * Generating Derivational Morphology <https://aclanthology.org/2020.emnlp-main.316/>
  * And any other papers you are interested!


* **Week 7 (4/18, 4/20)** Transformer-based Language Models (Applications: multilingual and multimodal)
  * Multilingual models
    * XLM-R: [paper](https://aclanthology.org/2020.acl-main.747/), [code](https://github.com/facebookresearch/fairseq/blob/main/examples/xlmr/README.md)
    * Cross-lingual analysis of multilingual BERT <https://aclanthology.org/2022.emnlp-main.552/>
  * Vision-language models
    * ViLBERT: [paper](https://proceedings.neurips.cc/paper/2019/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf), [code](https://github.com/facebookresearch/vilbert-multi-task)
    * Unicoder-VL <https://aclanthology.org/2022.emnlp-main.552/>
    * Zero-Shot Text-to-Image Generation: [paper](https://arxiv.org/abs/2102.12092), [code](https://github.com/lucidrains/DALLE-pytorch), [demo](https://openai.com/dall-e-2/)
    * X-LXMERT <https://aclanthology.org/2020.emnlp-main.707/>
  * Knowledge-enhanced pre-training
    * KEPLER: [paper](https://aclanthology.org/2021.tacl-1.11/), [code](https://github.com/THU-KEG/KEPLER)
    * ERNIE <https://aclanthology.org/P19-1139/>
    * KnowBERT <https://aclanthology.org/D19-1005/>
    * KGLM <https://aclanthology.org/P19-1598/>
  * And any other papers you are interested!


* **Week 8 (4/25, 4/27)** Fine-Tuning and Prompt Tuning
  * Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing <https://arxiv.org/pdf/2107.13586.pdf>
  * OpenPrompt: An Open-source Framework for Prompt-learning <https://arxiv.org/pdf/2111.01998.pdf>
  * Finetuned Language Models Are Zero-Shot Learners <https://arxiv.org/abs/2109.01652>
  * PromptBERT <https://aclanthology.org/2022.emnlp-main.603/>
  * Calibrate Before Use: Improving Few-Shot Performance of Language Models <https://arxiv.org/pdf/2102.09690.pdf>
  * PromptPapers <https://github.com/thunlp/PromptPapers>

* **Week 9 (5/2, 5/4)** Prompt Tuning (basics)
  * Discrete prompts
    * PET: [paper](https://aclanthology.org/2021.eacl-main.20/), [code](https://github.com/timoschick/pet)
    * ADAPET <https://aclanthology.org/2021.emnlp-main.407/>
    * AutoPrompt <https://arxiv.org/pdf/2010.15980>
    * Small-scale model, few shot <https://aclanthology.org/2021.naacl-main.185>
    * Automatically Identifying Words That Can Serve as Labels for Few-Shot Text Classification <https://aclanthology.org/2020.coling-main.488/>
  * Soft prompts
    * Prefix-Tuning <https://aclanthology.org/2021.acl-long.353>
    * P-Tuning v1 <https://arxiv.org/abs/2103.10385>
    * P-Tuning v2 <https://aclanthology.org/2022.acl-short.8/>
    * LoRA <https://arxiv.org/abs/2106.09685>


* **Week 10 (5/9, 5/11)** Prompt Tuning (basics)
  * Discrete prompts
    * LM-BFF <https://aclanthology.org/2021.acl-long.295/>
    * NSP-BERT <https://aclanthology.org/2022.coling-1.286/>
  * Soft prompts
    * Soft prompts are better <https://aclanthology.org/2021.emnlp-main.243/>
    * PPT: pre-train prompts by adding soft prompts into the pre-training stage to obtain a better initialization <https://aclanthology.org/2022.acl-long.576>

* **Week 11 (5/16, 5/18)** Prompt Tuning (analysis) 
  * Quantifying benefits of prompts <https://aclanthology.org/2021.naacl-main.208/>
  * Prompt order sensitivity <https://aclanthology.org/2022.acl-long.556/>
  * Analysis of prompts and templates <https://aclanthology.org/2022.naacl-main.167/>
  * Better sentence-pair classification than fine-tuning (zero-shot is good) <https://aclanthology.org/2021.emnlp-main.713>
  * Analysis pipeline (intrinsic prompt tuning, IPT) <https://arxiv.org/abs/2110.07867>
  * Effect of aspects of demonstration <https://aclanthology.org/2022.emnlp-main.759/>
  * Vulnerability
    * Exploring the Universal Vulnerability of Prompt-based Learning Paradigm <https://aclanthology.org/2022.findings-naacl.137/>
    * Ignore Previous Prompt: Attack Techniques For Language Models <https://arxiv.org/abs/2211.09527>

* **Week 12 (5/23, 5/25)** Prompt Tuning (improvements)
  * Better prompts
    * DART <https://arxiv.org/pdf/2108.13161>
    * Optimizing discrete prompts by RL <https://aclanthology.org/2022.emnlp-main.222>
  * Better verbalizer
    * Knowledgeable Prompt-tuning <https://aclanthology.org/2022.acl-long.158/>
    * Prototypical verbalizer <https://aclanthology.org/2022.acl-long.483/>
  * Better prompt tuning
    * Parameter-efficient, generalized prompt tuning <https://arxiv.org/abs/2207.07087>
    * Noisy channel approach <https://aclanthology.org/2022.acl-long.365/>
    * A variant of soft prompt tuning to MLM models (adversarial) <https://aclanthology.org/2021.acl-long.381/>
    * Optimizing continuous embedding space <https://aclanthology.org/2021.naacl-main.398/>

 
* **Week 13 (5/30, 6/1)** Prompt Tuning (applications)
  & Week 14 (6/8 Thu, 6/13 Tue)** Prompt Tuning (applications)
  * Domain adaptation
    * Multitask Prompted Training Enables Zero-Shot Task Generalization <https://arxiv.org/abs/2110.08207>
    * PADA <https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00468/110538/PADA-Example-based-Prompt-Learning-for-on-the-fly>
  * Semantics
    * Semantic parsing as paraphrasing <https://aclanthology.org/2021.emnlp-main.608>
    * relation extraction, NLI, verbalizer <https://aclanthology.org/2021.emnlp-main.92>
  * Entity
    * Entity typing <https://arxiv.org/abs/2108.10604>
    * Named Entity Recognition <https://aclanthology.org/2022.coling-1.209/>
  * Question Answering
    * Ask Me Anything: A simple strategy for prompting language models <https://arxiv.org/abs/2210.02441>
    * Compositionally gap <https://arxiv.org/abs/2210.03350>
  * Generation
    * Few-shot generation <https://aclanthology.org/2021.emnlp-main.32/>
    * Dialog generation <https://arxiv.org/abs/2109.06513>
    * Prefix-tuning <https://arxiv.org/abs/2110.08329>
    * Translation <https://aclanthology.org/2022.acl-long.424/>
  * Reasoning
    * Chain-of-thought prompting <https://arxiv.org/abs/2201.11903>
    * Chain-of-thought, self-consistency <https://arxiv.org/abs/2203.11171>
    * Reasoning, consistency, tree <https://aclanthology.org/2022.emnlp-main.82/>
    * Reasoning paths, verifying answers <https://arxiv.org/abs/2206.02336>
  * Vision-language models
    * Visual-enhanced entity, relation extraction <https://aclanthology.org/2022.findings-naacl.121/>
    * Vision-language, low resource <https://aclanthology.org/2022.acl-long.197/>
    * Novel form of soft prompting for vision-language models <https://arxiv.org/abs/2204.03574>
    * Vision-language models <https://link.springer.com/article/10.1007/s11263-022-01653-1>
  * Knowledge-enhanced models
    * Relation extraction <https://dl.acm.org/doi/pdf/10.1145/3485447.3511998>
    * Knowledge injection, ontology <https://dl.acm.org/doi/pdf/10.1145/3485447.3511921>
    * Sentiment knowledge <https://arxiv.org/abs/2109.08306>
  * And any other papers you are interested!

* **Week 15 (6/15 Thu)** Final Project Presentations


