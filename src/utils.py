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

content_map = {'2025-12-10': ['Foundational Fundamentals (Prerequisites) > Math for ML > Why math matters for ML',
  'Foundational Fundamentals (Prerequisites) > Math for ML > Linear Algebra > Vectors, matrices, norms'],
 '2025-12-11': ['Foundational Fundamentals (Prerequisites) > Math for ML > Linear Algebra > Matrix multiplication intuition',
  'Foundational Fundamentals (Prerequisites) > Math for ML > Linear Algebra > Eigenvalues/eigenvectors'],
 '2025-12-12': ['Foundational Fundamentals (Prerequisites) > Math for ML > Linear Algebra > SVD & PCA math',
  'Foundational Fundamentals (Prerequisites) > Math for ML > Calculus > Derivatives, gradients'],
 '2025-12-13': ['Foundational Fundamentals (Prerequisites) > Math for ML > Calculus > Chain rule (with backprop intuition)',
  'Foundational Fundamentals (Prerequisites) > Math for ML > Calculus > Optimization landscape'],
 '2025-12-14': ['Foundational Fundamentals (Prerequisites) > Math for ML > Probability & Statistics > Random variables, distributions',
  'Foundational Fundamentals (Prerequisites) > Math for ML > Probability & Statistics > Expectation, variance, covariance'],
 '2025-12-15': ['Foundational Fundamentals (Prerequisites) > Math for ML > Probability & Statistics > Bayes theorem',
  'Foundational Fundamentals (Prerequisites) > Math for ML > Probability & Statistics > KL divergence, entropy, cross-entropy'],
 '2025-12-16': ['Foundational Fundamentals (Prerequisites) > Math for ML > Probability & Statistics > Multivariate statistics',
  'Foundational Fundamentals (Prerequisites) > Python + ML Engineering > Python essentials'],
 '2025-12-17': ['Foundational Fundamentals (Prerequisites) > Python + ML Engineering > NumPy for ML',
  'Foundational Fundamentals (Prerequisites) > Python + ML Engineering > Pandas for ML'],
 '2025-12-18': ['Foundational Fundamentals (Prerequisites) > Python + ML Engineering > Matplotlib/Seaborn',
  'Foundational Fundamentals (Prerequisites) > Python + ML Engineering > Scikit-Learn basics'],
 '2025-12-19': ['Foundational Fundamentals (Prerequisites) > Python + ML Engineering > Writing clean ML code',
  'Foundational Fundamentals (Prerequisites) > Python + ML Engineering > Reproducibility & seeds'],
 '2025-12-20': ['Foundational Fundamentals (Prerequisites) > Python + ML Engineering > GitHub for ML',
  'Foundational Fundamentals (Prerequisites) > Python + ML Engineering > Virtual environments'],
 '2025-12-21': ['Foundational Fundamentals (Prerequisites) > Data Fundamentals > Data types & formats',
  'Foundational Fundamentals (Prerequisites) > Data Fundamentals > Data collection'],
 '2025-12-22': ['Foundational Fundamentals (Prerequisites) > Data Fundamentals > Data cleaning',
  'Foundational Fundamentals (Prerequisites) > Data Fundamentals > Feature extraction'],
 '2025-12-23': ['Foundational Fundamentals (Prerequisites) > Data Fundamentals > Exploratory Data Analysis',
  'Foundational Fundamentals (Prerequisites) > Data Fundamentals > Train/validation/test splits'],
 '2025-12-24': ['Foundational Fundamentals (Prerequisites) > Data Fundamentals > Data leakage',
  'Foundational Fundamentals (Prerequisites) > Data Fundamentals > Pipelines & preprocessing'],
 '2025-12-25': ['Foundational Fundamentals (Prerequisites) > Data Fundamentals > Feature engineering',
  'Foundational Fundamentals (Prerequisites) > Data Fundamentals > Label engineering'],
 '2025-12-26': ['Machine Learning (Complete Coverage) > Foundations of ML > What is ML?',
  'Machine Learning (Complete Coverage) > Foundations of ML > Types of ML'],
 '2025-12-27': ['Machine Learning (Complete Coverage) > Foundations of ML > Bias-variance tradeoff',
  'Machine Learning (Complete Coverage) > Foundations of ML > Underfitting vs overfitting'],
 '2025-12-28': ['Machine Learning (Complete Coverage) > Foundations of ML > Regularization',
  'Machine Learning (Complete Coverage) > Foundations of ML > Loss functions'],
 '2025-12-29': ['Machine Learning (Complete Coverage) > Foundations of ML > Gradient Descent variants',
  'Machine Learning (Complete Coverage) > Foundations of ML > Cross-validation'],
 '2025-12-30': ['Machine Learning (Complete Coverage) > Foundations of ML > Evaluation metrics',
  'Machine Learning (Complete Coverage) > Regression > Linear Regression'],
 '2025-12-31': ['Machine Learning (Complete Coverage) > Regression > Polynomial Regression',
  'Machine Learning (Complete Coverage) > Regression > Ridge Regression'],
 '2026-01-01': ['Machine Learning (Complete Coverage) > Regression > Lasso',
  'Machine Learning (Complete Coverage) > Regression > ElasticNet'],
 '2026-01-02': ['Machine Learning (Complete Coverage) > Regression > Normal Equation vs GD',
  'Machine Learning (Complete Coverage) > Regression > R², MSE, MAE'],
 '2026-01-03': ['Machine Learning (Complete Coverage) > Classification > Logistic Regression',
  'Machine Learning (Complete Coverage) > Classification > Naive Bayes'],
 '2026-01-04': ['Machine Learning (Complete Coverage) > Classification > kNN',
  'Machine Learning (Complete Coverage) > Classification > Decision Trees'],
 '2026-01-05': ['Machine Learning (Complete Coverage) > Classification > SVM (linear, RBF)',
  'Machine Learning (Complete Coverage) > Classification > PR/F1/AUC metrics'],
 '2026-01-06': ['Machine Learning (Complete Coverage) > Ensemble Learning > Bagging',
  'Machine Learning (Complete Coverage) > Ensemble Learning > Random Forest'],
 '2026-01-07': ['Machine Learning (Complete Coverage) > Ensemble Learning > AdaBoost',
  'Machine Learning (Complete Coverage) > Ensemble Learning > GradientBoosting'],
 '2026-01-08': ['Machine Learning (Complete Coverage) > Ensemble Learning > XGBoost',
  'Machine Learning (Complete Coverage) > Ensemble Learning > LightGBM'],
 '2026-01-09': ['Machine Learning (Complete Coverage) > Ensemble Learning > CatBoost',
  'Machine Learning (Complete Coverage) > Ensemble Learning > Stacking & blending'],
 '2026-01-10': ['Machine Learning (Complete Coverage) > Unsupervised Learning > KMeans',
  'Machine Learning (Complete Coverage) > Unsupervised Learning > GMM'],
 '2026-01-11': ['Machine Learning (Complete Coverage) > Unsupervised Learning > DBSCAN',
  'Machine Learning (Complete Coverage) > Unsupervised Learning > Hierarchical Clustering'],
 '2026-01-12': ['Machine Learning (Complete Coverage) > Unsupervised Learning > PCA',
  'Machine Learning (Complete Coverage) > Unsupervised Learning > t-SNE'],
 '2026-01-13': ['Machine Learning (Complete Coverage) > Unsupervised Learning > UMAP',
  'Machine Learning (Complete Coverage) > Unsupervised Learning > Density estimation'],
 '2026-01-14': ['Machine Learning (Complete Coverage) > Unsupervised Learning > Anomaly Detection',
  'Machine Learning (Complete Coverage) > Time Series ML > Stationarity, ACF/PACF'],
 '2026-01-15': ['Machine Learning (Complete Coverage) > Time Series ML > ARIMA/SARIMA',
  'Machine Learning (Complete Coverage) > Time Series ML > Prophet'],
 '2026-01-16': ['Machine Learning (Complete Coverage) > Time Series ML > Feature-based forecasting',
  'Machine Learning (Complete Coverage) > Time Series ML > Sliding window forecasting'],
 '2026-01-17': ['Machine Learning (Complete Coverage) > Recommendation Systems > Content-based',
  'Machine Learning (Complete Coverage) > Recommendation Systems > Collaborative filtering'],
 '2026-01-18': ['Machine Learning (Complete Coverage) > Recommendation Systems > Matrix factorization',
  'Machine Learning (Complete Coverage) > Recommendation Systems > Hybrid'],
 '2026-01-19': ['Machine Learning (Complete Coverage) > Recommendation Systems > Ranking models',
  'Deep Learning (Full Spectrum) > Foundations > What is a neural network?'],
 '2026-01-20': ['Deep Learning (Full Spectrum) > Foundations > Perceptron',
  'Deep Learning (Full Spectrum) > Foundations > Activation functions'],
 '2026-01-21': ['Deep Learning (Full Spectrum) > Foundations > Backpropagation',
  'Deep Learning (Full Spectrum) > Foundations > Weight initialization'],
 '2026-01-22': ['Deep Learning (Full Spectrum) > Foundations > Vanishing/exploding gradients',
  'Deep Learning (Full Spectrum) > Foundations > Batch norm'],
 '2026-01-23': ['Deep Learning (Full Spectrum) > Foundations > Layer norm',
  'Deep Learning (Full Spectrum) > Foundations > Optimizers'],
 '2026-01-24': ['Deep Learning (Full Spectrum) > Foundations > Regularization',
  'Deep Learning (Full Spectrum) > Foundations > GPUs & hardware'],
 '2026-01-25': ['Deep Learning (Full Spectrum) > Feedforward Neural Networks > MLPs',
  'Deep Learning (Full Spectrum) > Feedforward Neural Networks > Autoencoders'],
 '2026-01-26': ['Deep Learning (Full Spectrum) > Feedforward Neural Networks > Variational Autoencoders',
  'Deep Learning (Full Spectrum) > Computer Vision > CNN basics'],
 '2026-01-27': ['Deep Learning (Full Spectrum) > Computer Vision > Padding/stride/filters',
  'Deep Learning (Full Spectrum) > Computer Vision > Max pooling'],
 '2026-01-28': ['Deep Learning (Full Spectrum) > Computer Vision > Average pooling',
  'Deep Learning (Full Spectrum) > Computer Vision > ResNet'],
 '2026-01-29': ['Deep Learning (Full Spectrum) > Computer Vision > Inception',
  'Deep Learning (Full Spectrum) > Computer Vision > EfficientNet'],
 '2026-01-30': ['Deep Learning (Full Spectrum) > Computer Vision > MobileNet',
  'Deep Learning (Full Spectrum) > Computer Vision > DenseNet'],
 '2026-01-31': ['Deep Learning (Full Spectrum) > Computer Vision > Vision normalization layers',
  'Deep Learning (Full Spectrum) > Modern CV > Vision Transformers overview'],
 '2026-02-01': ['Deep Learning (Full Spectrum) > Modern CV > ConvNeXt',
  'Deep Learning (Full Spectrum) > Modern CV > SimCLR'],
 '2026-02-02': ['Deep Learning (Full Spectrum) > Modern CV > BYOL',
  'Deep Learning (Full Spectrum) > Modern CV > Dino'],
 '2026-02-03': ['Deep Learning (Full Spectrum) > Modern CV > Diffusion basics',
  'Deep Learning (Full Spectrum) > Sequence Models > RNN'],
 '2026-02-04': ['Deep Learning (Full Spectrum) > Sequence Models > Vanishing gradient problem',
  'Deep Learning (Full Spectrum) > Sequence Models > LSTM'],
 '2026-02-05': ['Deep Learning (Full Spectrum) > Sequence Models > GRU',
  'Deep Learning (Full Spectrum) > Sequence Models > Seq2Seq'],
 '2026-02-06': ['Deep Learning (Full Spectrum) > Sequence Models > Attention (original)',
  'Deep Learning (Full Spectrum) > NLP > Tokenization'],
 '2026-02-07': ['Deep Learning (Full Spectrum) > NLP > Bag of Words',
  'Deep Learning (Full Spectrum) > NLP > TF-IDF'],
 '2026-02-08': ['Deep Learning (Full Spectrum) > NLP > Word2Vec',
  'Deep Learning (Full Spectrum) > NLP > GloVe'],
 '2026-02-09': ['Deep Learning (Full Spectrum) > NLP > FastText',
  'Deep Learning (Full Spectrum) > NLP > Encoder-decoder architectures'],
 '2026-02-10': ['Transformers & LLMs (Deep Dive) > Transformers > Why attention is all you need',
  'Transformers & LLMs (Deep Dive) > Transformers > Multi-Head Attention'],
 '2026-02-11': ['Transformers & LLMs (Deep Dive) > Transformers > QKV',
  'Transformers & LLMs (Deep Dive) > Transformers > Scaled dot-product'],
 '2026-02-12': ['Transformers & LLMs (Deep Dive) > Transformers > Self vs cross attention',
  'Transformers & LLMs (Deep Dive) > Transformers > Positional encoding'],
 '2026-02-13': ['Transformers & LLMs (Deep Dive) > Transformers > Feedforward network',
  'Transformers & LLMs (Deep Dive) > Transformers > Layer norm'],
 '2026-02-14': ['Transformers & LLMs (Deep Dive) > Transformers > Residual connections',
  'Transformers & LLMs (Deep Dive) > Transformers > Encoder/decoder architecture'],
 '2026-02-15': ['Transformers & LLMs (Deep Dive) > Training Transformers > Causal LM',
  'Transformers & LLMs (Deep Dive) > Training Transformers > Masked LM'],
 '2026-02-16': ['Transformers & LLMs (Deep Dive) > Training Transformers > Denoising objective',
  'Transformers & LLMs (Deep Dive) > Training Transformers > Tokenization (BPE)'],
 '2026-02-17': ['Transformers & LLMs (Deep Dive) > Training Transformers > Tokenization (WordPiece)',
  'Transformers & LLMs (Deep Dive) > Training Transformers > Tokenization (SentencePiece)'],
 '2026-02-18': ['Transformers & LLMs (Deep Dive) > Training Transformers > Data curation',
  'Transformers & LLMs (Deep Dive) > Training Transformers > Distributed training'],
 '2026-02-19': ['Transformers & LLMs (Deep Dive) > Training Transformers > Mixed precision',
  'Transformers & LLMs (Deep Dive) > Training Transformers > Optimizers (AdamW)'],
 '2026-02-20': ['Transformers & LLMs (Deep Dive) > LLMs > GPT models',
  'Transformers & LLMs (Deep Dive) > LLMs > BERT'],
 '2026-02-21': ['Transformers & LLMs (Deep Dive) > LLMs > T5',
  'Transformers & LLMs (Deep Dive) > LLMs > LLaMA'],
 '2026-02-22': ['Transformers & LLMs (Deep Dive) > LLMs > Mistral',
  'Transformers & LLMs (Deep Dive) > LLMs > Phi'],
 '2026-02-23': ['Transformers & LLMs (Deep Dive) > LLMs > Qwen',
  'Transformers & LLMs (Deep Dive) > LLMs > Architecture differences'],
 '2026-02-24': ['Transformers & LLMs (Deep Dive) > LLM Fine Tuning > Full FT',
  'Transformers & LLMs (Deep Dive) > LLM Fine Tuning > LoRA'],
 '2026-02-25': ['Transformers & LLMs (Deep Dive) > LLM Fine Tuning > PEFT',
  'Transformers & LLMs (Deep Dive) > LLM Fine Tuning > QLoRA'],
 '2026-02-26': ['Transformers & LLMs (Deep Dive) > LLM Fine Tuning > Adapters',
  'Transformers & LLMs (Deep Dive) > LLM Fine Tuning > Instruction tuning'],
 '2026-02-27': ['Transformers & LLMs (Deep Dive) > LLM Fine Tuning > RLHF',
  'Transformers & LLMs (Deep Dive) > LLM Fine Tuning > RLAIF'],
 '2026-02-28': ['Transformers & LLMs (Deep Dive) > LLM Fine Tuning > DPO',
  'Transformers & LLMs (Deep Dive) > LLM Fine Tuning > ORPO'],
 '2026-03-01': ['Transformers & LLMs (Deep Dive) > LLM Evaluation > Perplexity',
  'Transformers & LLMs (Deep Dive) > LLM Evaluation > Win-rate'],
 '2026-03-02': ['Transformers & LLMs (Deep Dive) > LLM Evaluation > MT-bench',
  'Transformers & LLMs (Deep Dive) > LLM Evaluation > BEIR'],
 '2026-03-03': ['Transformers & LLMs (Deep Dive) > LLM Evaluation > TruthfulQA',
  'Transformers & LLMs (Deep Dive) > LLM Evaluation > Safety testing'],
 '2026-03-04': ['Transformers & LLMs (Deep Dive) > LLM Evaluation > Hallucination testing',
  'Transformers & LLMs (Deep Dive) > Inference > KV cache'],
 '2026-03-05': ['Transformers & LLMs (Deep Dive) > Inference > Speculative decoding',
  'Transformers & LLMs (Deep Dive) > Inference > Multi-token prediction'],
 '2026-03-06': ['Transformers & LLMs (Deep Dive) > Inference > Flash attention',
  'Transformers & LLMs (Deep Dive) > Inference > Quantization'],
 '2026-03-07': ['Transformers & LLMs (Deep Dive) > Inference > Pruning',
  'Transformers & LLMs (Deep Dive) > Inference > Distillation'],
 '2026-03-08': ['Transformers & LLMs (Deep Dive) > RAG > Vector databases',
  'Transformers & LLMs (Deep Dive) > RAG > Embeddings'],
 '2026-03-09': ['Transformers & LLMs (Deep Dive) > RAG > Cross-encoder reranking',
  'Transformers & LLMs (Deep Dive) > RAG > Agents in RAG'],
 '2026-03-10': ['Transformers & LLMs (Deep Dive) > RAG > Retriever fine-tuning',
  'Transformers & LLMs (Deep Dive) > RAG > Structured RAG'],
 '2026-03-11': ['Transformers & LLMs (Deep Dive) > RAG > Graph RAG',
  'Transformers & LLMs (Deep Dive) > RAG > Chunking strategies'],
 '2026-03-12': ['Transformers & LLMs (Deep Dive) > Agents > LangChain',
  'Transformers & LLMs (Deep Dive) > Agents > Tool calling'],
 '2026-03-13': ['Transformers & LLMs (Deep Dive) > Agents > Memory',
  'Transformers & LLMs (Deep Dive) > Agents > Multi-agent systems'],
 '2026-03-14': ['Transformers & LLMs (Deep Dive) > Agents > LangGraph',
  'Transformers & LLMs (Deep Dive) > Agents > Function calling'],
 '2026-03-15': ['Transformers & LLMs (Deep Dive) > Advanced Research > MoE',
  'Transformers & LLMs (Deep Dive) > Advanced Research > SSMs (Mamba)'],
 '2026-03-16': ['Transformers & LLMs (Deep Dive) > Advanced Research > Linear attention models',
  'Transformers & LLMs (Deep Dive) > Advanced Research > Structured sparsity'],
 '2026-03-17': ['Transformers & LLMs (Deep Dive) > Advanced Research > Retrieval-augmented training',
  'Transformers & LLMs (Deep Dive) > Advanced Research > Long-context models'],
 '2026-03-18': ['Transformers & LLMs (Deep Dive) > Advanced Research > VLMs',
  'Transformers & LLMs (Deep Dive) > Advanced Research > Multimodal transformers'],
 '2026-03-19': ['Transformers & LLMs (Deep Dive) > Advanced Research > Diffusion models',
  'Transformers & LLMs (Deep Dive) > Advanced Research > Generative video models'],
 '2026-03-20': ['MLOps for ML/DL/LLMs > Deployment > Model packaging',
  'MLOps for ML/DL/LLMs > Deployment > Docker'],
 '2026-03-21': ['MLOps for ML/DL/LLMs > Deployment > FastAPI',
  'MLOps for ML/DL/LLMs > Deployment > gRPC'],
 '2026-03-22': ['MLOps for ML/DL/LLMs > MLOps Lifecycle > Experiment tracking',
  'MLOps for ML/DL/LLMs > MLOps Lifecycle > Model registry'],
 '2026-03-23': ['MLOps for ML/DL/LLMs > MLOps Lifecycle > Continuous training',
  'MLOps for ML/DL/LLMs > MLOps Lifecycle > Monitoring'],
 '2026-03-24': ['MLOps for ML/DL/LLMs > MLOps Lifecycle > Drift detection',
  'MLOps for ML/DL/LLMs > MLOps Lifecycle > Logging & alerting'],
 '2026-03-25': ['MLOps for ML/DL/LLMs > LLMOps > Prompt management',
  'MLOps for ML/DL/LLMs > LLMOps > Logging interactions'],
 '2026-03-26': ['MLOps for ML/DL/LLMs > LLMOps > Feedback loops',
  'MLOps for ML/DL/LLMs > LLMOps > Safety monitoring'],
 '2026-03-27': ['MLOps for ML/DL/LLMs > LLMOps > RAG pipelines',
  'MLOps for ML/DL/LLMs > LLMOps > LLM eval changes'],
 '2026-03-28': ['MLOps for ML/DL/LLMs > Scaling ML > Distributed training',
  'MLOps for ML/DL/LLMs > Scaling ML > Sharding'],
 '2026-03-29': ['MLOps for ML/DL/LLMs > Scaling ML > Parallelism',
  'MLOps for ML/DL/LLMs > Scaling ML > Serverless inference'],
 '2026-03-30': ['MLOps for ML/DL/LLMs > Scaling ML > Caching strategies',
  'Real-World Case Studies (Your New Category!) > Finance > Fraud detection'],
 '2026-03-31': ['Real-World Case Studies (Your New Category!) > Finance > Time series forecasting',
  'Real-World Case Studies (Your New Category!) > Finance > LLMs for banking'],
 '2026-04-01': ['Real-World Case Studies (Your New Category!) > Finance > Portfolio optimization',
  'Real-World Case Studies (Your New Category!) > Healthcare > Diagnosis prediction'],
 '2026-04-02': ['Real-World Case Studies (Your New Category!) > Healthcare > Clinical LLMs',
  'Real-World Case Studies (Your New Category!) > Healthcare > Medical imaging'],
 '2026-04-03': ['Real-World Case Studies (Your New Category!) > Gaming > Difficulty balancing',
  'Real-World Case Studies (Your New Category!) > Gaming > NPC behavior modelling'],
 '2026-04-04': ['Real-World Case Studies (Your New Category!) > Gaming > Unity + ML Agents',
  'Real-World Case Studies (Your New Category!) > Recommendation Systems > ML + LLM recommenders'],
 '2026-04-05': ['Real-World Case Studies (Your New Category!) > E-commerce > Ranking systems',
  'Real-World Case Studies (Your New Category!) > E-commerce > Search systems'],
 '2026-04-06': ['Real-World Case Studies (Your New Category!) > E-commerce > Product enrichment with LLMs',
  'Interview Prep & Theory > Interview Prep > ML theory'],
 '2026-04-07': ['Interview Prep & Theory > Interview Prep > DL theory',
  'Interview Prep & Theory > Interview Prep > Transformer theory'],
 '2026-04-08': ['Interview Prep & Theory > Interview Prep > Whiteboard problem solving',
  'Interview Prep & Theory > Interview Prep > ML system design'],
 '2026-04-09': ['Interview Prep & Theory > Interview Prep > LLM architecture design',
  'Interview Prep & Theory > Interview Prep > Coding interview prep'],
 '2026-04-10': ['Interview Prep & Theory > Interview Prep > Debugging ML models',
  'Design Patterns > ML System Design > What is ML system design?'],
 '2026-04-11': ['Design Patterns > ML System Design > Online vs batch systems',
  'Design Patterns > ML System Design > SLA/SLO'],
 '2026-04-12': ['Design Patterns > ML System Design > Real-time pipelines',
  'Design Patterns > ML System Design > Feature stores'],
 '2026-04-13': ['Design Patterns > ML System Design > Real-time feature pipelines',
  'Design Patterns > ML System Design > Feature versioning'],
 '2026-04-14': ['Design Patterns > ML System Design > Distributed training',
  'Design Patterns > ML System Design > Model serving'],
 '2026-04-15': ['Design Patterns > ML System Design > Low-latency inference',
  'Design Patterns > ML System Design > Autoscaling'],
 '2026-04-16': ['Design Patterns > ML System Design > Model caching',
  'Design Patterns > ML System Design > Vector search systems'],
 '2026-04-17': ['Design Patterns > ML System Design > Hybrid search',
  'Design Patterns > ML System Design > Recommendation system design'],
 '2026-04-18': ['Design Patterns > ML System Design > Monitoring & drift',
  'Design Patterns > ML System Design > Explainability systems'],
 '2026-04-19': ['Design Patterns > ML System Design > Failure modes',
  'Design Patterns > ML System Design > Safe fallback'],
 '2026-04-20': ['Design Patterns > ML System Design > A/B testing architecture',
  'More than text > Multimodal AI > What is multimodal learning?'],
 '2026-04-21': ['More than text > Multimodal AI > Modality alignment',
  'More than text > Multimodal AI > CLIP'],
 '2026-04-22': ['More than text > Multimodal AI > BLIP',
  'More than text > Multimodal AI > BLIP-2'],
 '2026-04-23': ['More than text > Multimodal AI > Flamingo',
  'More than text > Multimodal AI > LLaVA'],
 '2026-04-24': ['More than text > Multimodal AI > Qwen-VL',
  'More than text > Multimodal AI > Kosmos'],
 '2026-04-25': ['More than text > Multimodal AI > Whisper',
  'More than text > Multimodal AI > AudioLM'],
 '2026-04-26': ['More than text > Multimodal AI > MusicLM',
  'More than text > Multimodal AI > Video Diffusion'],
 '2026-04-27': ['More than text > Multimodal AI > Sora',
  'More than text > Multimodal AI > Runway'],
 '2026-04-28': ['More than text > Multimodal AI > Pika',
  'More than text > Multimodal AI > Multimodal RAG'],
 '2026-04-29': ['More than text > Multimodal AI > Screenshot agents',
  'More than text > Multimodal AI > Voice agents'],
 '2026-04-30': ['Real-World Challenges > Product Development > What is an AI product?',
  'Real-World Challenges > Product Development > AI-first vs AI-augmented'],
 '2026-05-01': ['Real-World Challenges > Product Development > Use-case selection',
  'Real-World Challenges > Product Development > AI UX'],
 '2026-05-02': ['Real-World Challenges > Product Development > Prompt UX',
  'Real-World Challenges > Product Development > Failure UX'],
 '2026-05-03': ['Real-World Challenges > Product Development > Human-in-the-loop UX',
  'Real-World Challenges > Product Development > Data strategy'],
 '2026-05-04': ['Real-World Challenges > Product Development > Labeling strategy',
  'Real-World Challenges > Product Development > Synthetic data'],
 '2026-05-05': ['Real-World Challenges > Product Development > Model selection',
  'Real-World Challenges > Product Development > RAG vs fine-tuning'],
 '2026-05-06': ['Real-World Challenges > Product Development > Success metrics',
  'Real-World Challenges > Product Development > Red teaming'],
 '2026-05-07': ['Real-World Challenges > Product Development > A/B testing',
  'Real-World Challenges > Product Development > Shipping AI products'],
 '2026-05-08': ['Real-World Challenges > Product Development > Latency budgets',
  'Real-World Challenges > Product Development > Monetization models'],
 '2026-05-09': ['Real-World Challenges > Product Development > Pricing strategies',
  'Real-World Challenges > Product Development > Enterprise AI'],
 '2026-05-10': ['Real-World Challenges > Product Development > Ethical AI',
  'Real-World Challenges > Product Development > Governance & compliance']}