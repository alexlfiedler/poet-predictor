# Poet Predictor

In this project, I create a model that, given a poem, will attempt to classify the poem by its author. The idea for this project came from a lifetime of creative writing and a concern, shared by many in the creative community, that AI-generated writing could hurt the livelihoods of career writers. I thought that going through the process of creating the Poet Predictor might allow me to better understand how generative AI models process text.

## Data

My dataset contains the complete archives of PoetryFoundation.org, uploaded to kaggle.com by user John Hallman. It contains features including the author of each poem, the title of each poem, the poem’s “PoetryFoundation ID,” and the complete content of each poem. The original dataset included about 15,600 unique poems from about 3,300 unique authors. Upon loading my data, I filtered the dataset so that only poems by authors with five or more poems in the dataset were represented. After filtering, my dataset contained about 11,000 poems from about 1,000 unique authors.

Loading the data (only the poems and their corresponding authors were needing for this project) tensorizes it, encoding the words as integers. Each poem is truncated at 512 characters. The data is split into a train set, a validation set, and a test set. The train set represents 60% of the data, the validation set represents 20% of the data, and the test set represents 20% of the data.

## Model & Training

I built my model off of a pretrained BERT uncased Transformer model, courtesy of Hugging Face, which I chose because it appeared to be a relatively simple model while still being able to handle the complexity of text-based data with sequential features. My model has one hidden layer with a dimension of 768 and incorporates dropout and attention masking. It trains over the course of 10 epochs, with a batch size of 32 and a learning rate of 0.001. I use cross entropy loss and the Adam optimizer to tune my parameters. The train_model function in in train_model.py will run the training loop and return a fully trained model.

## Results

![image](https://github.com/user-attachments/assets/1eed97a3-ec03-4412-93b6-2e3f1102ff5a)

In my evaluation, I looked at the model’s final accuracy, F1 score, and loss for the training and test sets. Clearly, it did not perform well:
- Loss fell between 6-7 for both training and test sets.
- Accuracy was about 0.7% for the training set and 0.5% for the test set.
- The F1 Score was 0.0001 for the training set and under 0.00006 for the test set.

While these numbers still indicate that the model performed significantly better than random chance (with 1,000 authors represented in the complete filtered dataset), it would not be useful for any real-life applications. The difference in accuracy and F1 score between the training and test sets may indicate that the model is slightly overfit, but for both cases, performance is extremely poor.

## Looking Ahead

My model runs and is functional but not very useful in its present form. The fact that it consistently performs better than random chance suggests that it is at least picking up on some relevant features in the data. If I wanted to continue with this project in the future, there are multiple fronts on which I could improve, including:
- I could experiment with my choice of model, potentially including various transformer models or maybe an LSTM.
- I could adjust the architecture of my model, adding more hidden layers and/or playing with the dimensionality.
- I could adjust my hyperparameters, including the maximum truncation of the poems, dropout, batch size, number of epochs, learning rate, loss function, or optimizer.
- Perhaps most importantly, using a different dataset in the first place could significantly benefit my performance. With over 1,000 unique authors represented in my dataset, my model was built using a rather excessive number of classes, especially considering the ratio of labels to poems. In future iterations of this project, I could scale back the number of authors significantly and possibly increase the number of poems, working up from there. Out of all of the factors I have thus far identified, this likely has the most potential to result in a more accurate model.

Overall, while my model does not work well, I am happy that it runs, and some of my faith in the independence and humanity of the creative writing process has been restored.

## Acknowledgements

I would like to thank PoetryFoundation.org for their extensive archives of poetry, Kaggle user John Hallman for uploading the dataset, Professor Jake Searcy for his assistance in narrowing down my idea, Hugging Face for their BERT transformer model, ChatGPT, and the hundreds of poets whose work made this project possible.
