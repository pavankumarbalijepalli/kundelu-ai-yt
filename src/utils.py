from datetime import datetime as dt
import os 

file = dt.now().strftime("%Y-%m-%d") + ".log"

def log(log: str):
    log = f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} - {log}\n"
    print(log)
    if os.path.exists('logs/') == False:
        os.makedirs('logs/')
    open(f'logs/{file}', "a").write(log)
    
email_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>YT Automation - ML Learning Path</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&display=swap');

    body {{
    margin: 0;
    padding: 0;
    background-color: #faf7ff; /* soft lavender background */
    font-family: 'Merriweather', 'Segoe UI', Arial, sans-serif;
    }}

    .container {{
    max-width: 700px;
    margin: 20px auto;
    background-color: #ffffff;
    border-radius: 16px;
    box-shadow: 0 6px 25px rgba(155, 123, 255, 0.15); /* subtle purple glow */
    overflow: hidden;
    border: 1px solid #f2e6ff;
    }}

    /* Header */
    .header {{
    background: linear-gradient(135deg, #dba7ff, #9b7bff);
    color: #fff;
    text-align: center;
    padding: 35px 15px;
    border-bottom: 4px solid #f9b9e1;
    }}

    .header img {{
    width: 80px;
    height: 80px;
    border-radius: 50%;
    border: 3px solid #fff;
    margin-bottom: 12px;
    box-shadow: 0 0 12px rgba(255, 255, 255, 0.4);
    }}

    .header h1 {{
    font-size: 26px;
    margin: 10px 0 4px;
    font-weight: 700;
    color: #ffffff;
    }}

    .header p {{
    font-size: 14px;
    margin: 0;
    color: #f9eaff;
    }}

    /* Section */
    .section {{
    padding: 28px;
    }}

    .section h2 {{
    font-size: 20px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    }}

    .section p {{
    font-size: 15px;
    line-height: 1.6;
    color: #444;
    margin-bottom: 15px;
    }}

    /* Buttons */
    .btn {{
    display: inline-block;
    padding: 10px 22px;
    color: #fff;
    text-decoration: none;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.3s ease;
    }}

    .linkedin {{
    background: linear-gradient(135deg, #8cdbff, #0077b5);
    }}

    .medium {{
    background: linear-gradient(135deg, #8affc1, #02b875);
    }}

    .youtube {{
    background: linear-gradient(135deg, #ff7b7b, #ff0000);
    }}

    .btn:hover {{
    opacity: 0.9;
    transform: translateY(-1px);
    }}

    /* Divider */
    .divider {{
    border-top: 1px solid #eee;
    }}

    /* Footer */
    .footer {{
    background-color: #1d093c;
    text-align: center;
    padding: 25px;
    color: #c7b6ff;
    font-size: 12px;
    }}

    .footer a {{
    color: #f9b9e1;
    text-decoration: none;
    font-weight: 600;
    }}

    .footer a:hover {{
    text-decoration: underline;
    }}

  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <img src="{kundelu_ai}" alt="HappyDev Logo"/>
      <h1>Daily Fundamentals</h1>
      <p>Curated by Y.T.A Agent | Kundelu AI</p>
    </div>

    <div class="section">
      <h2 style="color:#ff0000;">YouTube Content</h2>
      <p>
        {youtube_post}
      </p>
    </div>

    <div class="footer">
      <p>© 2025 Kundelu AI | Created by Pavan Kumar Balijepalli</p>
      <p>
        <a href="https://www.linkedin.com/in/pavan-kumar-balijepalli/">LinkedIn</a> •
        <a href="https://medium.com/@pavanbalijepalli.bits">Medium</a> •
        <a href="https://youtube.com/@kundelu-ai">YouTube</a>
      </p>
    </div>
  </div>
</body>
</html>
"""

content_map = {
    "2025-11-25": "Foundational Fundamentals (Prerequisites) > Math for ML > Why math matters for ML",
    "2025-11-26": "Foundational Fundamentals (Prerequisites) > Math for ML > Linear Algebra > Vectors, matrices, norms",
    "2025-11-27": "Foundational Fundamentals (Prerequisites) > Math for ML > Linear Algebra > Matrix multiplication intuition",
    "2025-11-28": "Foundational Fundamentals (Prerequisites) > Math for ML > Linear Algebra > Eigenvalues/eigenvectors",
    "2025-11-29": "Foundational Fundamentals (Prerequisites) > Math for ML > Linear Algebra > SVD & PCA math",
    "2025-11-30": "Foundational Fundamentals (Prerequisites) > Math for ML > Calculus > Derivatives, gradients",
    "2025-12-01": "Foundational Fundamentals (Prerequisites) > Math for ML > Calculus > Chain rule (with backprop intuition)",
    "2025-12-02": "Foundational Fundamentals (Prerequisites) > Math for ML > Calculus > Optimization landscape",
    "2025-12-03": "Foundational Fundamentals (Prerequisites) > Math for ML > Probability & Statistics > Random variables, distributions",
    "2025-12-04": "Foundational Fundamentals (Prerequisites) > Math for ML > Probability & Statistics > Expectation, variance, covariance",
    "2025-12-05": "Foundational Fundamentals (Prerequisites) > Math for ML > Probability & Statistics > Bayes theorem",
    "2025-12-06": "Foundational Fundamentals (Prerequisites) > Math for ML > Probability & Statistics > KL divergence, entropy, cross-entropy",
    "2025-12-07": "Foundational Fundamentals (Prerequisites) > Math for ML > Probability & Statistics > Multivariate statistics (for generative models)",
    "2025-12-08": "Foundational Fundamentals (Prerequisites) > Python + ML Engineering Prerequisites > Python essentials",
    "2025-12-09": "Foundational Fundamentals (Prerequisites) > Python + ML Engineering Prerequisites > NumPy for ML",
    "2025-12-10": "Foundational Fundamentals (Prerequisites) > Python + ML Engineering Prerequisites > Pandas for ML",
    "2025-12-11": "Foundational Fundamentals (Prerequisites) > Python + ML Engineering Prerequisites > Matplotlib/Seaborn",
    "2025-12-12": "Foundational Fundamentals (Prerequisites) > Python + ML Engineering Prerequisites > Scikit-Learn basics",
    "2025-12-13": "Foundational Fundamentals (Prerequisites) > Python + ML Engineering Prerequisites > Writing clean ML code",
    "2025-12-14": "Foundational Fundamentals (Prerequisites) > Python + ML Engineering Prerequisites > Reproducibility and random seeds",
    "2025-12-15": "Foundational Fundamentals (Prerequisites) > Python + ML Engineering Prerequisites > Using GitHub for ML projects",
    "2025-12-16": "Foundational Fundamentals (Prerequisites) > Python + ML Engineering Prerequisites > Virtual environments (conda/venv)",
    "2025-12-17": "Foundational Fundamentals (Prerequisites) > Data Fundamentals > Data types & formats (CSV, Parquet, JSONL, images, audio)",
    "2025-12-18": "Foundational Fundamentals (Prerequisites) > Data Fundamentals > Data collection",
    "2025-12-19": "Foundational Fundamentals (Prerequisites) > Data Fundamentals > Data cleaning",
    "2025-12-20": "Foundational Fundamentals (Prerequisites) > Data Fundamentals > Feature extraction",
    "2025-12-21": "Foundational Fundamentals (Prerequisites) > Data Fundamentals > Exploratory Data Analysis",
    "2025-12-22": "Foundational Fundamentals (Prerequisites) > Data Fundamentals > Train/validation/test splits",
    "2025-12-23": "Foundational Fundamentals (Prerequisites) > Data Fundamentals > Data leakage",
    "2025-12-24": "Foundational Fundamentals (Prerequisites) > Data Fundamentals > Pipelines & preprocessing",
    "2025-12-25": "Foundational Fundamentals (Prerequisites) > Data Fundamentals > Feature engineering",
    "2025-12-26": "Foundational Fundamentals (Prerequisites) > Data Fundamentals > Label engineering",
    "2025-12-27": "Machine Learning (Complete Coverage) > Foundations of ML > What is ML?",
    "2025-12-28": "Machine Learning (Complete Coverage) > Foundations of ML > Types: Supervised / Unsupervised / Semi / Self-supervised / RL",
    "2025-12-29": "Machine Learning (Complete Coverage) > Foundations of ML > Bias-variance tradeoff",
    "2025-12-30": "Machine Learning (Complete Coverage) > Foundations of ML > Underfitting vs overfitting",
    "2025-12-31": "Machine Learning (Complete Coverage) > Foundations of ML > Regularization: L1, L2, dropout",
    "2026-01-01": "Machine Learning (Complete Coverage) > Foundations of ML > Loss functions",
    "2026-01-02": "Machine Learning (Complete Coverage) > Foundations of ML > Optimization: Gradient Descent variants",
    "2026-01-03": "Machine Learning (Complete Coverage) > Foundations of ML > Cross-validation",
    "2026-01-04": "Machine Learning (Complete Coverage) > Foundations of ML > Evaluation metrics",
    "2026-01-05": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Regression > Linear Regression",
    "2026-01-06": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Regression > Polynomial Regression",
    "2026-01-07": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Regression > Ridge, Lasso, ElasticNet",
    "2026-01-08": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Regression > Gradient Descent vs Normal Equation",
    "2026-01-09": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Regression > Evaluation: R\u00b2, MSE, MAE",
    "2026-01-10": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Classification > Logistic Regression",
    "2026-01-11": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Classification > Na\u00efve Bayes",
    "2026-01-12": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Classification > kNN",
    "2026-01-13": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Classification > Decision Trees",
    "2026-01-14": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Classification > SVM (linear, RBF)",
    "2026-01-15": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Ensemble Learning > Bagging",
    "2026-01-16": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Ensemble Learning > Random Forest",
    "2026-01-17": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Ensemble Learning > AdaBoost",
    "2026-01-18": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Ensemble Learning > GradientBoosting",
    "2026-01-19": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Ensemble Learning > XGBoost",
    "2026-01-20": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Ensemble Learning > LightGBM",
    "2026-01-21": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Ensemble Learning > CatBoost",
    "2026-01-22": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Ensemble Learning > Stacking & blending",
    "2026-01-23": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Unsupervised Learning > KMeans",
    "2026-01-24": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Unsupervised Learning > GMM",
    "2026-01-25": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Unsupervised Learning > DBSCAN",
    "2026-01-26": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Unsupervised Learning > Hierarchical Clustering",
    "2026-01-27": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Unsupervised Learning > PCA",
    "2026-01-28": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Unsupervised Learning > t-SNE",
    "2026-01-29": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Unsupervised Learning > UMAP",
    "2026-01-30": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Unsupervised Learning > Density estimation",
    "2026-01-31": "Machine Learning (Complete Coverage) > Classical ML Algorithms > Unsupervised Learning > Anomaly detection",
    "2026-02-01": "Machine Learning (Complete Coverage) > Time Series ML > Stationarity, ACF/PACF",
    "2026-02-02": "Machine Learning (Complete Coverage) > Time Series ML > ARIMA/SARIMA",
    "2026-02-03": "Machine Learning (Complete Coverage) > Time Series ML > Prophet",
    "2026-02-04": "Machine Learning (Complete Coverage) > Time Series ML > Feature-based ML forecasting",
    "2026-02-05": "Machine Learning (Complete Coverage) > Time Series ML > Sliding window forecasting",
    "2026-02-06": "Machine Learning (Complete Coverage) > Recommendation Systems > Content-based",
    "2026-02-07": "Machine Learning (Complete Coverage) > Recommendation Systems > Collaborative filtering",
    "2026-02-08": "Machine Learning (Complete Coverage) > Recommendation Systems > Matrix factorization",
    "2026-02-09": "Machine Learning (Complete Coverage) > Recommendation Systems > Hybrid recommenders",
    "2026-02-10": "Machine Learning (Complete Coverage) > Recommendation Systems > Ranking models",
    "2026-02-11": "Deep Learning (Full Spectrum) > Foundations > What is a neural network?",
    "2026-02-12": "Deep Learning (Full Spectrum) > Foundations > Perceptron",
    "2026-02-13": "Deep Learning (Full Spectrum) > Foundations > Activation functions",
    "2026-02-14": "Deep Learning (Full Spectrum) > Foundations > Backpropagation",
    "2026-02-15": "Deep Learning (Full Spectrum) > Foundations > Weight initialization",
    "2026-02-16": "Deep Learning (Full Spectrum) > Foundations > Vanishing/exploding gradients",
    "2026-02-17": "Deep Learning (Full Spectrum) > Foundations > Batch norm",
    "2026-02-18": "Deep Learning (Full Spectrum) > Foundations > Layer norm",
    "2026-02-19": "Deep Learning (Full Spectrum) > Foundations > Optimization: Adam, RMSProp, SGD",
    "2026-02-20": "Deep Learning (Full Spectrum) > Foundations > Warmups, Schedulers",
    "2026-02-21": "Deep Learning (Full Spectrum) > Foundations > Regularization: dropout, early stopping",
    "2026-02-22": "Deep Learning (Full Spectrum) > Foundations > Hardware & GPU basics",
    "2026-02-23": "Deep Learning (Full Spectrum) > Feedforward Neural Networks > MLPs",
    "2026-02-24": "Deep Learning (Full Spectrum) > Feedforward Neural Networks > Autoencoders",
    "2026-02-25": "Deep Learning (Full Spectrum) > Feedforward Neural Networks > Variational Autoencoders",
    "2026-02-26": "Deep Learning (Full Spectrum) > Computer Vision > CNN basics",
    "2026-02-27": "Deep Learning (Full Spectrum) > Computer Vision > Padding, stride, filters",
    "2026-02-28": "Deep Learning (Full Spectrum) > Computer Vision > Max pooling",
    "2026-03-01": "Deep Learning (Full Spectrum) > Computer Vision > Average pooling",
    "2026-03-02": "Deep Learning (Full Spectrum) > Computer Vision > Batch norm vs layer norm",
    "2026-03-03": "Deep Learning (Full Spectrum) > Computer Vision > ResNet",
    "2026-03-04": "Deep Learning (Full Spectrum) > Computer Vision > Inception",
    "2026-03-05": "Deep Learning (Full Spectrum) > Computer Vision > EfficientNet",
    "2026-03-06": "Deep Learning (Full Spectrum) > Computer Vision > MobileNet",
    "2026-03-07": "Deep Learning (Full Spectrum) > Computer Vision > DenseNet",
    "2026-03-08": "Deep Learning (Full Spectrum) > Computer Vision > Vision normalization layers",
    "2026-03-09": "Deep Learning (Full Spectrum) > Modern CV > Vision Transformers overview",
    "2026-03-10": "Deep Learning (Full Spectrum) > Modern CV > ConvNext",
    "2026-03-11": "Deep Learning (Full Spectrum) > Modern CV > SimCLR",
    "2026-03-12": "Deep Learning (Full Spectrum) > Modern CV > BYOL",
    "2026-03-13": "Deep Learning (Full Spectrum) > Modern CV > Dino",
    "2026-03-14": "Deep Learning (Full Spectrum) > Modern CV > Image generation foundations",
    "2026-03-15": "Deep Learning (Full Spectrum) > Sequence Modelling > RNN",
    "2026-03-16": "Deep Learning (Full Spectrum) > Sequence Modelling > Vanishing gradient problem",
    "2026-03-17": "Deep Learning (Full Spectrum) > Sequence Modelling > LSTM",
    "2026-03-18": "Deep Learning (Full Spectrum) > Sequence Modelling > GRU",
    "2026-03-19": "Deep Learning (Full Spectrum) > Sequence Modelling > Seq2Seq",
    "2026-03-20": "Deep Learning (Full Spectrum) > Sequence Modelling > Attention (original)",
    "2026-03-21": "Deep Learning (Full Spectrum) > NLP > Tokenization basics",
    "2026-03-22": "Deep Learning (Full Spectrum) > NLP > Bag of Words",
    "2026-03-23": "Deep Learning (Full Spectrum) > NLP > TF-IDF",
    "2026-03-24": "Deep Learning (Full Spectrum) > NLP > Word2Vec",
    "2026-03-25": "Deep Learning (Full Spectrum) > NLP > GloVe",
    "2026-03-26": "Deep Learning (Full Spectrum) > NLP > FastText",
    "2026-03-27": "Deep Learning (Full Spectrum) > NLP > Recurrent models for NLP",
    "2026-03-28": "Deep Learning (Full Spectrum) > NLP > Encoder-decoder architectures",
    "2026-03-29": "Transformers & LLMs (Deep Dive) > Core Transformer Architecture > Why attention is all you need",
    "2026-03-30": "Transformers & LLMs (Deep Dive) > Core Transformer Architecture > Multi-Head Attention",
    "2026-03-31": "Transformers & LLMs (Deep Dive) > Core Transformer Architecture > Queries, Keys, Values",
    "2026-04-01": "Transformers & LLMs (Deep Dive) > Core Transformer Architecture > Scaled dot-product",
    "2026-04-02": "Transformers & LLMs (Deep Dive) > Core Transformer Architecture > Self-attention vs cross-attention",
    "2026-04-03": "Transformers & LLMs (Deep Dive) > Core Transformer Architecture > Positional encoding",
    "2026-04-04": "Transformers & LLMs (Deep Dive) > Core Transformer Architecture > Feedforward network",
    "2026-04-05": "Transformers & LLMs (Deep Dive) > Core Transformer Architecture > Layer norm",
    "2026-04-06": "Transformers & LLMs (Deep Dive) > Core Transformer Architecture > Residual connections",
    "2026-04-07": "Transformers & LLMs (Deep Dive) > Core Transformer Architecture > Encoder vs decoder vs encoder-decoder",
    "2026-04-08": "Transformers & LLMs (Deep Dive) > Training Transformers > Causal LM",
    "2026-04-09": "Transformers & LLMs (Deep Dive) > Training Transformers > Masked LM",
    "2026-04-10": "Transformers & LLMs (Deep Dive) > Training Transformers > Denoising objective",
    "2026-04-11": "Transformers & LLMs (Deep Dive) > Training Transformers > Tokenization: BPE",
    "2026-04-12": "Transformers & LLMs (Deep Dive) > Training Transformers > Tokenization: WordPiece",
    "2026-04-13": "Transformers & LLMs (Deep Dive) > Training Transformers > Tokenization: SentencePiece",
    "2026-04-14": "Transformers & LLMs (Deep Dive) > Training Transformers > Data curation for LLMs",
    "2026-04-15": "Transformers & LLMs (Deep Dive) > Training Transformers > Distributed training",
    "2026-04-16": "Transformers & LLMs (Deep Dive) > Training Transformers > Mixed precision",
    "2026-04-17": "Transformers & LLMs (Deep Dive) > Training Transformers > AdamW, warmups",
    "2026-04-18": "Transformers & LLMs (Deep Dive) > LLMs > GPT models",
    "2026-04-19": "Transformers & LLMs (Deep Dive) > LLMs > BERT & variants",
    "2026-04-20": "Transformers & LLMs (Deep Dive) > LLMs > T5",
    "2026-04-21": "Transformers & LLMs (Deep Dive) > LLMs > LLaMA",
    "2026-04-22": "Transformers & LLMs (Deep Dive) > LLMs > Mistral",
    "2026-04-23": "Transformers & LLMs (Deep Dive) > LLMs > Phi",
    "2026-04-24": "Transformers & LLMs (Deep Dive) > LLMs > Qwen",
    "2026-04-25": "Transformers & LLMs (Deep Dive) > LLMs > Model architectures & differences",
    "2026-04-26": "Transformers & LLMs (Deep Dive) > Fine-Tuning LLMs > Full fine-tuning",
    "2026-04-27": "Transformers & LLMs (Deep Dive) > Fine-Tuning LLMs > LoRA & PEFT",
    "2026-04-28": "Transformers & LLMs (Deep Dive) > Fine-Tuning LLMs > QLoRA",
    "2026-04-29": "Transformers & LLMs (Deep Dive) > Fine-Tuning LLMs > Adapter layers",
    "2026-04-30": "Transformers & LLMs (Deep Dive) > Fine-Tuning LLMs > Instruction tuning",
    "2026-05-01": "Transformers & LLMs (Deep Dive) > Fine-Tuning LLMs > RLHF",
    "2026-05-02": "Transformers & LLMs (Deep Dive) > Fine-Tuning LLMs > RLAIF",
    "2026-05-03": "Transformers & LLMs (Deep Dive) > Fine-Tuning LLMs > DPO",
    "2026-05-04": "Transformers & LLMs (Deep Dive) > Fine-Tuning LLMs > ORPO",
    "2026-05-05": "Transformers & LLMs (Deep Dive) > LLM Evaluation > Perplexity",
    "2026-05-06": "Transformers & LLMs (Deep Dive) > LLM Evaluation > Win-rate",
    "2026-05-07": "Transformers & LLMs (Deep Dive) > LLM Evaluation > MT-bench",
    "2026-05-08": "Transformers & LLMs (Deep Dive) > LLM Evaluation > BEIR",
    "2026-05-09": "Transformers & LLMs (Deep Dive) > LLM Evaluation > TruthfulQA",
    "2026-05-10": "Transformers & LLMs (Deep Dive) > LLM Evaluation > Safety testing",
    "2026-05-11": "Transformers & LLMs (Deep Dive) > LLM Evaluation > Hallucination testing",
    "2026-05-12": "Transformers & LLMs (Deep Dive) > LLM Inference > KV cache",
    "2026-05-13": "Transformers & LLMs (Deep Dive) > LLM Inference > Speculative decoding",
    "2026-05-14": "Transformers & LLMs (Deep Dive) > LLM Inference > Multi-token prediction",
    "2026-05-15": "Transformers & LLMs (Deep Dive) > LLM Inference > Flash attention",
    "2026-05-16": "Transformers & LLMs (Deep Dive) > LLM Inference > Quantization",
    "2026-05-17": "Transformers & LLMs (Deep Dive) > LLM Inference > Pruning",
    "2026-05-18": "Transformers & LLMs (Deep Dive) > LLM Inference > Distillation",
    "2026-05-19": "Transformers & LLMs (Deep Dive) > RAG > Vector databases",
    "2026-05-20": "Transformers & LLMs (Deep Dive) > RAG > Embeddings",
    "2026-05-21": "Transformers & LLMs (Deep Dive) > RAG > RAG with cross-encoder re-ranking",
    "2026-05-22": "Transformers & LLMs (Deep Dive) > RAG > RAG with Agents",
    "2026-05-23": "Transformers & LLMs (Deep Dive) > RAG > RAG with retriever fine-tuning",
    "2026-05-24": "Transformers & LLMs (Deep Dive) > RAG > Structured RAG",
    "2026-05-25": "Transformers & LLMs (Deep Dive) > RAG > Graph RAG",
    "2026-05-26": "Transformers & LLMs (Deep Dive) > RAG > Document chunking strategies",
    "2026-05-27": "Transformers & LLMs (Deep Dive) > Agents & Tools > LangChain",
    "2026-05-28": "Transformers & LLMs (Deep Dive) > Agents & Tools > Agent types",
    "2026-05-29": "Transformers & LLMs (Deep Dive) > Agents & Tools > Tool-calling",
    "2026-05-30": "Transformers & LLMs (Deep Dive) > Agents & Tools > Memory",
    "2026-05-31": "Transformers & LLMs (Deep Dive) > Agents & Tools > Multi-agent systems",
    "2026-06-01": "Transformers & LLMs (Deep Dive) > Agents & Tools > LangGraph",
    "2026-06-02": "Transformers & LLMs (Deep Dive) > Agents & Tools > Function calling",
    "2026-06-03": "Transformers & LLMs (Deep Dive) > Advanced > Mixture of Experts (MoE)",
    "2026-06-04": "Transformers & LLMs (Deep Dive) > Advanced > State Space Models (SSMs), Mamba",
    "2026-06-05": "Transformers & LLMs (Deep Dive) > Advanced > Linear attention models",
    "2026-06-06": "Transformers & LLMs (Deep Dive) > Advanced > Structured sparsity",
    "2026-06-07": "Transformers & LLMs (Deep Dive) > Advanced > Retrieval-augmented training",
    "2026-06-08": "Transformers & LLMs (Deep Dive) > Advanced > Long-context models",
    "2026-06-09": "Transformers & LLMs (Deep Dive) > Advanced > Vision-Language Models",
    "2026-06-10": "Transformers & LLMs (Deep Dive) > Advanced > Audio + multimodal transformers",
    "2026-06-11": "Transformers & LLMs (Deep Dive) > Advanced > Diffusion models",
    "2026-06-12": "Transformers & LLMs (Deep Dive) > Advanced > Generative video models",
    "2026-06-13": "MLOps for ML/DL/LLMs > Deployment > Model packaging",
    "2026-06-14": "MLOps for ML/DL/LLMs > Deployment > Docker",
    "2026-06-15": "MLOps for ML/DL/LLMs > Deployment > FastAPI",
    "2026-06-16": "MLOps for ML/DL/LLMs > Deployment > gRPC",
    "2026-06-17": "MLOps for ML/DL/LLMs > MLOps Lifecycle > MLflow",
    "2026-06-18": "MLOps for ML/DL/LLMs > MLOps Lifecycle > Weights & Biases",
    "2026-06-19": "MLOps for ML/DL/LLMs > MLOps Lifecycle > Model registry",
    "2026-06-20": "MLOps for ML/DL/LLMs > MLOps Lifecycle > Continuous training",
    "2026-06-21": "MLOps for ML/DL/LLMs > MLOps Lifecycle > Monitoring",
    "2026-06-22": "MLOps for ML/DL/LLMs > MLOps Lifecycle > Drift detection",
    "2026-06-23": "MLOps for ML/DL/LLMs > MLOps Lifecycle > Logging & alerting",
    "2026-06-24": "MLOps for ML/DL/LLMs > LLMOps > Prompt management",
    "2026-06-25": "MLOps for ML/DL/LLMs > LLMOps > Logging user interactions",
    "2026-06-26": "MLOps for ML/DL/LLMs > LLMOps > Feedback loops",
    "2026-06-27": "MLOps for ML/DL/LLMs > LLMOps > Safety monitoring",
    "2026-06-28": "MLOps for ML/DL/LLMs > LLMOps > RAG pipelines",
    "2026-06-29": "MLOps for ML/DL/LLMs > LLMOps > Evaluating LLM changes",
    "2026-06-30": "MLOps for ML/DL/LLMs > Scaling ML > Distributed training",
    "2026-07-01": "MLOps for ML/DL/LLMs > Scaling ML > Sharding",
    "2026-07-02": "MLOps for ML/DL/LLMs > Scaling ML > Parallelism",
    "2026-07-03": "MLOps for ML/DL/LLMs > Scaling ML > Serverless inference",
    "2026-07-04": "MLOps for ML/DL/LLMs > Scaling ML > Caching strategies",
    "2026-07-05": "Real-World Case Studies (Your New Category!) > Finance > Fraud detection",
    "2026-07-06": "Real-World Case Studies (Your New Category!) > Finance > Time series forecasting",
    "2026-07-07": "Real-World Case Studies (Your New Category!) > Finance > LLMs for banking",
    "2026-07-08": "Real-World Case Studies (Your New Category!) > Finance > Portfolio optimization",
    "2026-07-09": "Real-World Case Studies (Your New Category!) > Healthcare > Diagnosis prediction",
    "2026-07-10": "Real-World Case Studies (Your New Category!) > Healthcare > Clinical LLMs",
    "2026-07-11": "Real-World Case Studies (Your New Category!) > Healthcare > Medical imaging",
    "2026-07-12": "Real-World Case Studies (Your New Category!) > Gaming > AI for game difficulty balancing",
    "2026-07-13": "Real-World Case Studies (Your New Category!) > Gaming > NPC behavior modelling",
    "2026-07-14": "Real-World Case Studies (Your New Category!) > Gaming > Unity + ML Agents",
    "2026-07-15": "Real-World Case Studies (Your New Category!) > Recommendation Systems > ML + LLM hybrid recommenders",
    "2026-07-16": "Real-World Case Studies (Your New Category!) > E-commerce > Ranking systems",
    "2026-07-17": "Real-World Case Studies (Your New Category!) > E-commerce > Search systems",
    "2026-07-18": "Real-World Case Studies (Your New Category!) > E-commerce > LLMs for product data enrichment",
    "2026-07-19": "Interview Prep & Theory > Interview Prep > ML theory",
    "2026-07-20": "Interview Prep & Theory > Interview Prep > DL theory",
    "2026-07-21": "Interview Prep & Theory > Interview Prep > Transformer theory",
    "2026-07-22": "Interview Prep & Theory > Interview Prep > Whiteboard problem solving",
    "2026-07-23": "Interview Prep & Theory > Interview Prep > System design for ML",
    "2026-07-24": "Interview Prep & Theory > Interview Prep > LLM architecture design",
    "2026-07-25": "Interview Prep & Theory > Interview Prep > Coding interview prep",
    "2026-07-26": "Interview Prep & Theory > Interview Prep > Debugging ML models"
}