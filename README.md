# Question Answering with WikiQA using Transformer and RNN-based Models

This repository contains implementations of **Machine Reading Comprehension (MRC)** on the **WikiQA** dataset using **PyTorch** and GPU acceleration. The models implemented include:

- **Transformer-based Models:**
  - DistilBERT
  - RoBERTa
  - XLNet
- **Recurrent-based Models:**
  - GRU
  - LSTM
  - RNN

**Installation**

Clone the Repository  

git clone https://github.com/MuzamilAhmed007/Distil-BERT-based-MRC.git

cd WikiQA-MRC

Install Dependencies
All required packages are listed in requirements.txt. Install them using:
pip install -r requirements.txt

training & Evaluation
Each script can be executed independently.
For example, to train RoBERTa:

python train_roberta.py

