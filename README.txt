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

  - Make sure `./NER/best_model_bilstm.pth` is the correct name and path of your saved model.
  - Save the Word2Vec model locally  to speed up future usage.

## Part 2

- To train and test seven different models on Question_Classification task.

> \* part2.ipynb

- To fine-tune the transformers under different settings.

```
chmod a+x start_transformers.sh
./start_transformers.sh
```

\* You may need to activate a Python virtual environment and adjust the path to align with the relevant files as necessary.

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
│   train.py                                      // Train code 
│   inference.py                                  // Inference code
|   best_model_bilstm.pth                         // Best model weight (N.A. unfeasible to submit)
|
Question_Classification
|   part2.ipynb																		// Train and test models
|   part2_fine_tune.py								      			// Fine-tune transformers under different settings
|   start_transformers.sh													// Script to run the transformers training tasks
|   Transformers_log.out								 					// Transformers training log file
|   
```