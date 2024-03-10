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

## Model Architecture and Training
This project employs a pretrained model from the Hugging Face library, specifically `symanto/xlm-roberta-base-snli-mnli-anli-xnli`. This choice leverages transfer learning to improve the prediction accuracy without extensive computational resources. The model architecture includes:
- **XLM-RoBERTa Base**: A powerful language model capable of understanding multiple languages, making it ideal for our diverse dataset.
- **Fine-tuning**: The pretrained model is fine-tuned on our specific task of contradiction prediction.

Training involves adjusting the model to our specific dataset and monitoring performance on the validation set to prevent overfitting.

## Evaluation and Results
Evaulation done on kaggle using their provided test cases

<img width="1146" alt="Screenshot 2024-03-10 at 21 45 42" src="https://github.com/Harshkamdar67/Contradiction-prediction/assets/63010578/d77bad90-a73d-4f3d-b7b6-94bf1dd9a271">


## Usage
To use this project:
1. Clone the repository to your local environment.
2. Ensure you have all required libraries installed.
3. Run the Jupyter notebook from start to finish to train the model and evaluate its performance.

## Contributions
Contributions to this project are welcome. If you have suggestions for improvements or encounter any issues, please open a pull request or an issue in the repository.
