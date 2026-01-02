
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/P4pal2004/sentiment-analysis-lstm-ulmfit-bert/blob/main/Sentiment_Analysis_LSTM_ULMFiT_BERT.ipynb)



# Sentiment Analysis using LSTM, ULMFiT, and BERT

## 📌 Project Overview
This project implements and compares three deep learning models for sentiment analysis:
- Custom LSTM (trained from scratch)
- ULMFiT (AWD-LSTM with transfer learning)
- BERT (Transformer-based model)

The goal is to analyze how transfer learning and transformer architectures improve performance over traditional recurrent models.

---

## 🧠 Skills Gained
- Sequence modeling with LSTM
- Transfer learning in NLP (ULMFiT & BERT)
- PyTorch & fastai
- HuggingFace Transformers
- Model evaluation and comparison

---

## 📂 Dataset
- IMDb Movie Reviews Dataset
- 50,000 labeled reviews (Positive / Negative)
- Source: Stanford AI Lab / HuggingFace Datasets

---

## 🏗 Project Structure
├── sentiment_analysis_lstm_ulmfit_bert.ipynb
├── README.md

---

## 🔍 Project Phases

### Phase 1: Custom LSTM vs ULMFiT
- Text preprocessing
- LSTM model training from scratch
- Fine-tuning pretrained AWD-LSTM
- Performance comparison

### Phase 2: Custom LSTM vs BERT
- WordPiece tokenization
- Fine-tuning BERT-base-uncased
- Evaluation against LSTM models

---

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Training & Validation Loss
- Epochs to convergence

---

## 📈 Results Summary
- Custom LSTM converges slowly
- ULMFiT achieves higher accuracy with fewer epochs
- BERT provides the best performance but at higher computational cost

---

## 🛠 Tech Stack
- Python
- PyTorch
- fastai
- HuggingFace Transformers
- Google Colab

---

## ▶️ How to Run
1. Open the notebook in Google Colab
2. Enable GPU (Runtime → Change runtime type)
3. Run all cells sequentially

---

## 👨‍💻 Author
Mahendra Pal







