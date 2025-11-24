# ✅ KUNDELU AI FUNDAMENTALS (COMPLETE ROADMAP)

## TRACK 1 — Foundational Fundamentals (Prerequisites)

*(Best for “Fundamentals Friday”)*

### 1. Math for ML

* Why math matters for ML
* Linear Algebra

  * Vectors, matrices, norms
  * Matrix multiplication intuition
  * Eigenvalues/eigenvectors
  * SVD & PCA math
* Calculus

  * Derivatives, gradients
  * Chain rule (with backprop intuition)
  * Optimization landscape
* Probability & Statistics

  * Random variables, distributions
  * Expectation, variance, covariance
  * Bayes theorem
  * KL divergence, entropy, cross-entropy
* Multivariate statistics (for generative models)

### 2. Python + ML Engineering Prerequisites

* Python essentials
* NumPy for ML
* Pandas for ML
* Matplotlib/Seaborn
* Scikit-Learn basics
* Writing clean ML code
* Reproducibility and random seeds
* Using GitHub for ML projects
* Virtual environments (conda/venv)

### 3. Data Fundamentals

* Data types & formats (CSV, Parquet, JSONL, images, audio)
* Data collection
* Data cleaning
* Feature extraction
* Exploratory Data Analysis
* Train/validation/test splits
* Data leakage
* Pipelines & preprocessing
* Feature engineering
* Label engineering

---

##  TRACK 2 — Machine Learning (Complete Coverage)

### 1. Foundations of ML

* What is ML?
* Types: Supervised / Unsupervised / Semi / Self-supervised / RL
* Bias-variance tradeoff
* Underfitting vs overfitting
* Regularization: L1, L2, dropout (for DL)
* Loss functions
* Optimization: Gradient Descent variants
* Cross-validation
* Evaluation metrics (classification, regression, ranking)

### 2. Classical ML Algorithms

#### Regression

* Linear Regression
* Polynomial Regression
* Ridge, Lasso, ElasticNet
* Gradient Descent vs Normal Equation
* Evaluation: R², MSE, MAE

#### Classification

* Logistic Regression
* Naïve Bayes
* kNN
* Decision Trees
* SVM (linear, RBF kernel)
* Metrics: Precision/Recall/F1/AUC/PR-AUC

#### Ensemble Learning

* Bagging
* Random Forest
* Boosting: AdaBoost, GradientBoosting
* XGBoost / LightGBM / CatBoost
* Stacking & blending

#### Unsupervised Learning

* Clustering: KMeans, GMM, DBSCAN, Hierarchical
* Dimensionality reduction: PCA, t-SNE, UMAP
* Density estimation
* Anomaly detection

#### Time Series ML

* Stationarity, ACF/PACF
* ARIMA/SARIMA
* Prophet
* Feature-based ML forecasting
* Sliding window forecasting

#### Recommendation Systems

* Content-based
* Collaborative filtering
* Matrix factorization
* Hybrid recommenders
* Ranking models

---

##  TRACK 3 — Deep Learning (Full Spectrum)

### 1. Foundations

* What is a neural network?
* Perceptron
* Activation functions
* Backpropagation (complete math + intuition)
* Weight initialization
* Vanishing/exploding gradients
* Batch norm, layer norm
* Optimization for DL: Adam, RMSProp, SGD, Warmups, Schedulers
* Regularization: dropout, early stopping
* Hardware & GPU basics

### 2. Feedforward Neural Networks

* MLPs
* Autoencoders
* Variational Autoencoders

### 3. Computer Vision

#### Convolutional Neural Networks

* CNN basics
* Padding, stride, filters
* Max pooling, average pooling
* Batch norm vs layer norm
* ResNet
* Inception
* EfficientNet
* MobileNet
* DenseNet
* Vision normalization layers

#### Modern CV

* Vision Transformers overview
* ConvNext
* Self-supervised CV: SimCLR, BYOL, Dino
* Image generation foundations

### 4. Sequence Modelling (Pre-Transformer)

* RNN
* Vanishing gradient problem
* LSTM
* GRU
* Seq2Seq
* Attention (original idea)

### 5. NLP (Pre-Transformer + Modern)

* Tokenization basics
* Bag of Words
* TF-IDF
* Word2Vec
* GloVe
* FastText
* Recurrent models for NLP
* Encoder-decoder architectures

---

##  TRACK 4 — Transformers & LLMs (Deep Dive)

### 1. Core Transformer Architecture

* Why attention is all you need
* Multi-Head Attention

  * Queries, Keys, Values
* Scaled dot-product
* Self-attention vs cross-attention
* Positional encoding
* Feedforward network
* Layer norm
* Residual connections
* Encoder vs decoder vs encoder-decoder

### 2. Training Transformers

* Pretraining objectives

  * Causal LM
  * Masked LM
  * Denoising objective
* Tokenization:

  * BPE
  * WordPiece
  * SentencePiece
* Data curation for LLMs
* Distributed training
* Mixed precision
* Optimizing training (AdamW, warmups)

### 3. Large Language Models (LLMs)

* GPT models
* BERT & variants
* T5
* LLaMA
* Mistral
* Phi
* Qwen
* Model architectures & differences

### 4. Fine-Tuning LLMs

* Full fine-tuning
* LoRA & PEFT
* QLoRA
* Adapter layers
* Instruction tuning
* Preference optimization

  * RLHF
  * RLAIF
  * DPO
  * ORPO

### 5. LLM Evaluation

* Perplexity
* Win-rate
* MT-bench
* BEIR
* TruthfulQA
* Safety testing
* Hallucination testing

### 6. LLM Inference

* KV cache
* Speculative decoding
* Multi-token prediction
* Flash attention
* Quantization
* Pruning
* Distillation

### 7. Retrieval-Augmented Generation

* Vector databases
* Embeddings
* RAG with cross-encoder re-ranking
* RAG with Agents
* RAG with retriever fine-tuning
* Structured RAG
* Graph RAG
* Document chunking strategies

### 8. Agents & Tools

* LangChain
* Agent types
* Tool-calling
* Memory
* Multi-agent systems
* LangGraph
* Function calling

### 9. Advanced (Modern Research Topics)

* Mixture of Experts (MoE)
* State Space Models (SSMs), Mamba
* Linear attention models
* Structured sparsity
* Retrieval-augmented training
* Long-context models
* Vision-Language Models
* Audio + multimodal transformers
* Diffusion models
* Generative video models

---

##  TRACK 5 — MLOps for ML/DL/LLMs

*(Important for your technical audience)*

### 1. ML Deployment Essentials

* Model packaging
* Docker
* REST APIs with FastAPI
* gRPC

### 2. MLOps Lifecycle

* Experiment tracking (MLflow, Weights & Biases)
* Model registry
* Continuous training
* Monitoring
* Drift detection
* Logging & Alerting

### 3. LLMOps

* Prompt management
* Logging user interactions
* Feedback loops
* Safety monitoring
* RAG pipelines
* Evaluating LLM changes

### 4. Scaling ML

* Distributed training
* Sharding
* Parallelism
* Serverless inference
* Caching strategies

---

##  TRACK 6 — Real-World Case Studies (Your New Category!)

*(Perfect for advanced content)*

### 1. Finance

* Fraud detection
* Time series forecasting
* LLMs for banking
* Portfolio optimization models

### 2. Healthcare

* Diagnosis prediction
* Clinical LLMs
* Medical imaging

### 3. Gaming

* AI for game difficulty balancing
* NPC behavior modelling
* Unity + ML Agents

### 4. Recommendation Systems

* ML + LLM hybrid recommenders

### 5. E-commerce

* Ranking systems
* Search systems
* LLMs for product data enrichment

---

##  TRACK 7 — Interview Prep & Theory

* ML theory
* DL theory
* Transformer theory
* Whiteboard problem solving
* System design for ML
* LLM architecture design
* Coding interview prep
* Debugging ML models
