# Xenodet: Detecting Xenophobic Content in Social Media Posts

Welcome to **Xenodet**, a project focused on identifying and classifying xenophobic content in social media posts. This repository showcases various Natural Language Processing (NLP) techniques, including frequency-based embeddings, word embeddings, and sentence embeddings, combined with multiple classification models to detect xenophobic language effectively.

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Project Overview](#project-overview)  
3. [Dataset](#dataset)  
4. [Data Preprocessing](#data-preprocessing)  
5. [Approaches](#approaches)  
6. [Models](#models)  
7. [Metrics](#metrics)  
8. [Results and Observations](#results-and-observations)  
9. [Usage](#usage)  
10. [Future Work](#future-work)  
11. [References](#references)  

---

## Introduction

In the face of rising xenophobia on social media, **Xenodet** aims to provide a robust system for detecting xenophobic content automatically. By leveraging NLP and machine learning techniques, we hope to help social media platforms, researchers, and policymakers identify and mitigate hostile content aimed at ethnic or foreign communities.

---

## Project Overview

- **Objective**: Classify social media posts into two categories:  
  1. **Xenophobic**  
  2. **Non-xenophobic**

- **Key Steps**:
  1. **Data Collection & Curation**: Leveraging an existing hate speech dataset with a focus on ethnicity-related content.  
  2. **Preprocessing & Feature Engineering**: Cleaning text, removing stopwords, and creating linguistic features (POS tags, NER, sentiment scores).  
  3. **Model Training**: Using multiple embedding strategies—TF-IDF, Word2Vec, GloVe, and Sentence-BERT (S-BERT)—with three main classifiers: SVM, XGBoost, and Logistic Regression.  
  4. **Evaluation**: Comparing performance using accuracy, precision, recall, and F1-score.  

---

## Dataset

We used a subset of the **HateXplain** dataset, filtering for xenophobic content by focusing on posts that target ethnic groups. To maintain balance, non-hate (or non-targeted) posts were also included:

- **Initial Size**: 15,905 posts (original hate speech dataset).  
- **Filtered for Ethnicity**: 5,801 posts targeting ethnic groups.  
- **Additional Non-target Posts**: 4,850 rows with no specific target.  
- **Final Dataset**: 10,651 posts (near-equal distribution of xenophobic vs. non-xenophobic).  

*Note*: The dataset retains emojis and removes other potential noise (links, pictures, duplicates) to preserve relevant textual information.

---

## Data Preprocessing

1. **Dropped Unwanted Columns**  
   - Kept only the essential fields: `text`, `label`, and `target`.

2. **Handling Missing Values**  
   - The filtered dataset contained no null values, so no imputation was necessary.

3. **Removal of Unnecessary Characters**  
   - Removed extraneous symbols but preserved emojis, as they can convey sentiment.

4. **Stopword Removal**  
   - Used SpaCy’s default English stopwords, plus a custom list for domain-specific terms.

5. **Label Encoding**  
   - Converted labels into binary format (0 for non-xenophobic, 1 for xenophobic).

6. **Resulting Structure**  
   - **`text`**: cleaned social media post  
   - **`label`**: binary-encoded indicator of xenophobic or non-xenophobic  

---

## Approaches

### 1. Frequency-Based Embedding
- **TF-IDF** (Term Frequency–Inverse Document Frequency)  
  - Good at capturing high-frequency “key” terms used in xenophobic discourse.  

- **Sentiment Analysis (VADER)**  
  - Computes a sentiment score, which can help identify strongly negative or positive language.  

- **POS Tagging & NER Features**  
  - Part-of-Speech tags (e.g., nouns, verbs, adjectives) to highlight critical roles in xenophobic sentences.  
  - Named Entity Recognition (NER) to identify group references (e.g., immigrants, foreigners).  

### 2. Word Embedding
- **GloVe** and **Word2Vec**  
  - Capture semantic relationships among words but can miss deeper context at the sentence level.

- **Word Embedding + POS Tagging**  
  - Combining syntactic information (POS) with semantic vectors (GloVe/Word2Vec) improved accuracy for xenophobic detection.  

### 3. Sentence Embedding
- **S-BERT (Sentence-BERT)**  
  - Generates 384-dimensional embeddings that incorporate the sentence’s full context.  
  - Achieved the highest recall, critical for identifying all xenophobic posts.  

- **Universal Sentence Encoder (USE)**  
  - Outputs 512-dimensional embeddings.  
  - Designed for large-scale usage, performed slightly lower than S-BERT on the smaller dataset.

---

## Models

We employed three main classifiers to test the effectiveness of different embeddings:

1. **Support Vector Machine (SVM)**  
   - Particularly effective with high-dimensional data.  
   - RBF kernel captured non-linear decision boundaries well.

2. **XGBoost**  
   - Powerful tree-based algorithm.  
   - Good at handling complex feature interactions and large feature spaces.  

3. **Logistic Regression**  
   - Simple and interpretable baseline.  
   - Provides a good comparison point for more advanced models.  

---

## Metrics

To evaluate performance, we used:

- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-score**

**Recall** (sensitivity) is crucial for flagging xenophobic posts, ensuring fewer instances of hateful content go undetected.

---

## Results and Observations

1. **TF-IDF**  
   - Achieved up to **86% accuracy** with XGBoost.  
   - Highlights repetitive or “key” terms often found in xenophobic language.

2. **Word Embeddings (GloVe / Word2Vec)**  
   - Moderate performance (60–68% accuracy).  
   - Improved significantly when combined with POS tags, reaching up to **80% accuracy** (Word2Vec + SVM).

3. **Sentence Embeddings**  
   - **S-BERT** gave the best overall performance, reaching **86% accuracy** and notably the highest recall (~83%).  
   - **USE** also performed well (~82% accuracy), but slightly behind S-BERT for this dataset.

The **best trade-off** between precision and recall emerged with **S-BERT** embeddings fed into SVM or XGBoost, confirming that sentence-level context is crucial for detecting subtle or implicit xenophobic content.

---

## Usage

Below is a general outline for using this repository. (Adjust as needed for your environment.)

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/xenodet.git
   cd xenodet
   ```

2. **Create and Activate a Virtual Environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   # or
   venv\Scripts\activate      # For Windows
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   *(Make sure to include packages such as SpaCy, scikit-learn, XGBoost, Transformers, etc.)*

4. **Download and Prepare the Dataset**  
   - Obtain the HateXplain dataset (or your custom dataset) and place it in a designated `data/` directory.
   - Follow the data preprocessing steps in the notebook or script provided.

5. **Run the Preprocessing and Training Scripts**  
   ```bash
   python preprocess.py   # Example script for data cleaning
   python train.py        # Example script for model training
   ```

6. **Evaluate the Models**  
   - Check `results/` or console logs for accuracy, precision, recall, and F1-scores.

---

## Future Work

1. **Advanced Transformer Models**  
   - Fine-tune larger models like **RoBERTa** or **BERT-large** for improved context capture.

2. **Multilingual Support**  
   - Expand beyond English to detect xenophobia across languages.

3. **Multimodal Analysis**  
   - Incorporate images or memes, since hateful content often appears in mixed media formats.

4. **Real-Time Implementation**  
   - Develop a streaming pipeline for live xenophobia monitoring on social media platforms.

---

## References

- Binny Mathew, Punyajoy Saha, Seid Muhie Yimam, Chris Biemann, Pawan Goyal, and Animesh Mukherjee. 2020. **HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection**. Proceedings of the Eighteenth International AAAI Conference on Web and Social Media.

- Khonzoda Umarova, Oluchi Okorafor, Pinxian Lu, Sophia Shan, Alex Xu, Ray Zhou, Jennifer Otiono, Beth Lyon, Gilly Leshed. 2024. **Xenophobia Meter: Defining and Measuring Online Sentiment toward Foreigners on Twitter**. Proceedings of the Eighteenth International AAAI Conference on Web and Social Media (ICWSM2024).

- HateXplain Dataset:  
  [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KYR4IY](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KYR4IY)

- Jackie Swift. 2024. **Xenophobia Meter Aims to Track Anti-Immigrant Hate Speech**.  
  [https://cis.cornell.edu/xenophobia-meter-aims-track-anti-immigrant-hate-speech](https://cis.cornell.edu/xenophobia-meter-aims-track-anti-immigrant-hate-speech)

---

**Thank you for your interest in Xenodet!**  
Feel free to open an issue or submit a pull request if you have any questions or suggestions. Let’s work together to foster safer, more inclusive online spaces.
