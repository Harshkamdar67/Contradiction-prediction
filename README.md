# Contradiction Prediction Project

## Overview
This project is dedicated to predicting relationships between pairs of sentences, specifically focusing on identifying whether sentences contradict each other, are neutral, or entail each other. It utilizes deep learning and natural language processing techniques to analyze sentence pairs and classify their relationships. The model is trained on a combination of a Kaggle dataset and the Multi_NLI dataset to ensure diversity and robustness.

## Datasets
The project leverages two datasets:

1. **Kaggle Contradiction Prediction Dataset**: A collection of sentence pairs with annotations for contradiction, neutrality, or entailment. This dataset forms the basis of the training and testing examples for the project.

2. **Multi_NLI Dataset**: An extensive dataset used to augment the training data, providing additional examples across a variety of genres, styles, and contexts, thereby enhancing the model's generalization capabilities.

## Installation and Requirements
The project is developed using Python and requires the following libraries:
- NumPy
- Pandas
- TensorFlow
- Hugging Face 'nlp' and 'transformers'

You can install the necessary packages using the following command:

```bash
pip install numpy pandas tensorflow nlp transformers
```

## Data Preparation and Preprocessing
The data preparation involves loading the datasets, cleaning, and tokenizing the text. This process includes concatenating the Kaggle and Multi_NLI datasets to form a comprehensive training set. The scripts handle null values and split the data into training, validation, and testing sets to ensure a thorough evaluation.

## Exploratory Data Analysis (EDA) Insights
The EDA process reveals important insights into the data:
- Distribution of languages shows a predominance of English, which could affect the model's performance in other languages.
- 
  <img width="1058" alt="Screenshot 2024-03-10 at 21 49 16" src="https://github.com/Harshkamdar67/Contradiction-prediction/assets/63010578/3996427c-4951-40c1-baf8-6c09e9352664">
  

- Label distribution indicates a balanced dataset in terms of contradiction, neutrality, and entailment, which is beneficial for training unbiased models.

<img width="681" alt="Screenshot 2024-03-10 at 21 49 41" src="https://github.com/Harshkamdar67/Contradiction-prediction/assets/63010578/eba8db7b-58ea-43aa-91f9-ea2e9c4d694c">

  
- Analysis of sentence length for both premises and hypotheses shows varying lengths, with most sentences being of moderate length, suggesting that the model needs to handle varying textual lengths effectively.
  
<img width="1060" alt="Screenshot 2024-03-10 at 21 49 59" src="https://github.com/Harshkamdar67/Contradiction-prediction/assets/63010578/dbd5719f-797c-434c-83a0-041a1056f68d">

  
- The class distribution by language offers insights into the dataset's composition, which can inform strategies for internationalization and localization of the model.

These insights inform data preprocessing and model training, ensuring the model is robust and performs well across different datasets and sentence structures.

## Model Architecture and Training
This project employs a pretrained model from the Hugging Face library, specifically `symanto/xlm-roberta-base-snli-mnli-anli-xnli`. This choice leverages transfer learning to improve the prediction accuracy without extensive computational resources. The model architecture includes:
- **XLM-RoBERTa Base**: A powerful language model capable of understanding multiple languages, making it ideal for our diverse dataset.
- **Fine-tuning**: The pretrained model is fine-tuned on our specific task of contradiction prediction.

Training involves adjusting the model to our specific dataset and monitoring performance on the validation set to prevent overfitting.

## Evaluation and Results
The model's performance is evaluated using Kaggle's test cases and metrics. For detailed results and performance metrics, please refer to the evaluation section in the notebook.

![Model Evaluation](https://github.com/Harshkamdar67/Contradiction-prediction/assets/63010578/d77bad90-a73d-4f3d-b7b6-94bf1dd9a271)

## Usage
To use this project:
1. Clone the repository to your local environment.
2. Ensure you have all required libraries installed.
3. Run the Jupyter notebook from start to finish to train the model and evaluate its performance.

## Contributions
Contributions to this project are welcome. If you have suggestions for improvements or encounter any issues, please open a pull request or an issue in the repository.
