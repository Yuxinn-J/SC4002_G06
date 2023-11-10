<div align="center">
  <h1>AY23S1 SC4002 NLP Assignment - G06</h1>
</div>


# Installation
To set up the required environment, use the provided environment file with Conda:
`conda env create -f environment.yml`
Alternatively, ensure you have the following prerequisites installed for inference:
- Python 3.x
- PyTorch
- Gensim

# Quick inference
## Part 1
- Download pre-trained models: `best_model_bilstm.pth` 
  ```
  wget https://github.com/Yuxinn-J/SC4002_G06/releases/download/v1/best_model_bilstm.pth -P pretrained_models
  ```
- For **one-shot** mode, pass a sentence as an argument:
  ```
  python NER/inference.py "European Union rejects German call to boycott British lamb."
  ```
  - The output will display NER tags for each token in the input sentence.
- For **interactive** mode, invoke the script with the --interactive flag:
  ```
  python NER/inference.py --interactive
  ```
  - The script will prompt you to enter sentences one by one and will output the predicted tags for each, until you type `exit`.
  - ![image](https://github.com/Yuxinn-J/SC4002_G06/assets/73170270/5869c062-987e-43c3-8d28-663fc71137e7)
- Notes:
  - Make sure `./pretrained_models/best_model_bilstm.pth` is the correct name and path of your saved model.
  - Save the Word2Vec model locally  to speed up future usage.

## Part 2

# Submission Files
The directory structure is outlined below, with individual files for different parts of the assignment:

```
NER
│   Part1_1.ipynb                                 // Explore word embedding
│   Part1_2.ipynb                                 // Analyze dataset                 
│   Part1_3_Models.ipynb                          // Initial model performance benchmarking
│   Part1_3_Tuning.ipynb                          // Optimal BiLSTM setting discovery
│   Part1_3_Final.ipynb                           // Final model training and testing
│   hyperparameter_tuning_results.json            // Results of hyper-parameter tuning
│   train.py                                      // Train code (quite messy...)
│   inference.py                                  // Inference code
|   best_model.pth                                // best model weight
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

# Final Model
