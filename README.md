# Comparing Arabic Sentence Transformers
There is a scarcity of evidence and benchmarks indicating the performance of sentence embedding transformers on Arabic language text, **_especially_** for smaller models that can be trained on "garden-variety" GPUs (such as those used for gaming). Here I test a few different sentence embedding models on a multi-class classification problem to see which performs better. 

##  Dataset
The dataset used consists of 25,000 labeled examples, with arabic poems (representing the text to be embedded), and labels (indicating the historical era the poem comes from). This dataset was obtained from [Kaggle](https://www.kaggle.com/competitions/arabic-poem-classification/overview). The 5 historical eras are mapped to numerical values (0,1,2,3,4) before training. The data set was split into a 
As some of the poems exceeded the context length of the transformers used, 512 tokens ~350 words, the longer poems were _chunked_ into smaller strings, each with the same label as the original poem.

## Training Procedures
The first training procedure used a transformer model to 
