AY23S1 SC4002 NLP Assignment - G06

# Installation
To set up the required environment, use the provided environment file with Conda:
`conda env create -f environment.yml`
Alternatively, ensure you have the following prerequisites installed for inference:
- Python 3.x
- PyTorch
- Gensim

# Quick inference
## Part 1
- For **one-shot** mode, pass a sentence as an argument:
  ```
  cd NER
  python inference.py "OpenAI is a research laboratory in San Francisco, California."
  ```
  - The output will display NER tags for each token in the input sentence.
- For **interactive** mode, invoke the script with the --interactive flag:
  ```
  cd NER
  python inference.py --interactive
  ```
  - The script will prompt you to enter sentences one by one and will output the predicted tags for each, until you type `exit`.
- Notes:
  - Make sure best_model.pth is the correct name and path of your saved model.
  - Ensure the local word2vec model file is correctly placed and accessible by the script.

## Part 2

## Submission Files
The directory structure is outlined below, with individual files for different parts of the assignment:
```
NER
│   Part1_1.ipynb                                 // Explore word embedding
│   Part1_2.ipynb                                 // Analyze dataset                 
│   Part1_3_Models.ipynb                          // Initial model performance benchmarking
│   Part1_3_Tuning.ipynb                          // Optimal BiLSTM setting discovery
│   Part1_3_Final.ipynb                           // Final model training and testing
│   hyperparameter_tuning_results.json            // Results of hyper-parameter tuning
│   inference.py                                  // Inference code
|
Question_Classification
|   VGG16-Adience.ipynb										
|   ResNet101-Adience.ipynb								      
|   InceptionV3-Adience.ipynb									
|   EfficientNetB5-Adience.ipynb								 
|   
└───Explore_EfficientNet                              // explore EfficientNet variations 
      ├─ EfficientNetB5-Hyperpara_Tuning.ipynb
      ├─ EfficientNetB5-CelebA.ipynb
      └─ EfficientNetB5-Gender-Age.ipynb              // consider age and gender simultaneously
│
pretrained_models                                                
│   ViT-Adience.ipynb
│   CLIP_Zero-shot-Adience.ipynb
│   

```