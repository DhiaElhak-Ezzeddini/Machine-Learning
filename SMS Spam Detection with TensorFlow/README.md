# Spam Message Classification

This project demonstrates the development and evaluation of different machine learning models to classify text messages as either "ham" (non-spam) or "spam." The task utilizes various techniques including traditional machine learning methods and deep learning architectures.

### Project Overview
The dataset consists of a collection of text messages labeled as either "ham" or "spam." The goal of this project is to build multiple classification models and compare their performance. The models implemented include:

- **Multinomial Naive Bayes (Baseline Model):** A traditional machine learning algorithm using Term Frequency-Inverse Document Frequency (TF-IDF) for text vectorization.
- **Custom Text Vectorization and Embedding Model:** A neural network model using custom text vectorization and an embedding layer to represent words.
- **Bidirectional LSTM Model:** A deep learning model using Bidirectional Long Short-Term Memory (LSTM) layers to capture long-term dependencies in the text.
- **Universal Sentence Encoder (USE) Transfer Learning Model:** A transfer learning approach using the Universal Sentence Encoder to encode sentences into fixed-size vectors, followed by a dense layer for classification.

### Key Features
- **Text Vectorization:** Used TF-IDF and custom Keras `TextVectorization` for transforming text data into numerical representations.
- **Model Building:** Implemented models using scikit-learn (MultinomialNB) and TensorFlow/Keras for deep learning models.
- **Model Evaluation:** Evaluated the models using accuracy, precision, recall, and F1-score metrics. Visualized model performance using confusion matrices.

### Models Comparison
The models are evaluated and compared based on the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Results show that the deep learning models (Bidirectional LSTM and USE) outperformed the Multinomial Naive Bayes model, with the USE-based model showing the best performance.

### Requirements
To run this project, you will need the following libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `tensorflow_hub`
- `sklearn`
