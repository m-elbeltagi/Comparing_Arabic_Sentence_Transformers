# Comparing Arabic Sentence Transformers
There is a scarcity of evidence and benchmarks indicating the performance of sentence embedding transformers on Arabic language text, **_especially_** for smaller models that can be trained on a single "garden-variety" GPU (such as those used for gaming). Here I test a few different sentence embedding models on a multi-class classification problem to see which performs better. 

##  Dataset
The dataset used consists of 25,000 labeled examples, with arabic poems (representing the text to be embedded), and labels (indicating the historical era the poem comes from). This dataset was obtained from [Kaggle](https://www.kaggle.com/competitions/arabic-poem-classification/overview). The data set was split into a 80%-20% for tain-eval. The 5 historical eras are mapped to numerical values (0,1,2,3,4) before training. 
As some of the poems exceeded the context length of the transformers used, 512 tokens ~350 words, the longer poems were _chunked_ into smaller strings, each with the same label as the original poem.

## Training Procedures
1) The first training procedure used a transformer model to produce the embeddings or _hidden states_, this involves passing the dataset through the transformer once, then using the produced hidden states to train a NN classifier. These hidden states encode the meaning of the sentence accounting for context. The NN classifier is fully connected with 2 hidden layers, a 768 dim input layer (the dim of the embeddings output by the transformer), and a 5 dim output layer (the log odds of each label, which are later turned into probabilities via softmax).

2) The second training procedure (facilitated by the Hugging Face Trainer), trained both transformer and classifier (the gradients flow all the way back), to optimize the produced embeddings specifically for this classification task.

This is the 
