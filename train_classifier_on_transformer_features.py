import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import random_split
import torch.optim.lr_scheduler as lr_scheduler
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModel
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import datetime
import seaborn as sns


"""the first part of the file, along with the check function is for 
exploring the dataset, for training just call the preprocess_dataset 
function to tokenize the training dataset which acts as a pipeline, 
then prepare_torch_trainset, then start_train_loop
"""


## setting device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device in use: {} \n".format(device))

# setting some parameters
torch.manual_seed(666)
batch_size = 100
n_epochs = 10
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999

## using 'engine = python' for the second file because it's large
poem_train_df = pd.read_csv(r"./poems.csv", engine="python")


# extracting unique values in Category column
unique_values = poem_train_df["Category"].unique()

# map to numerical labels for each category (using index in unique_values)
class_mapping = {value: index for index, value in enumerate(unique_values)}

# applying class_mapping to Category column
poem_train_df["Category"] = poem_train_df["Category"].map(class_mapping)

# shuffling dataset
poem_train_df = poem_train_df.sample(frac=1, random_state=666).reset_index(drop=True)

# dropping the other 2 colums with strings for now (trying Poem only)
poem_train_df = poem_train_df.drop(["Title", "Author"], axis=1)


def plot_texts_lengths(train_dataset):
    """PLots lengths of strings used for training, to check if any of the string
    sizes exceeds the maximum context size of the  model we choose
    (which we can't have), notice we're only looking at length in words here,
     but model context size is in tokens, so if it's close, truncation needs to
     be activated below, if it's much less than it's fine
    """

    train_dataset = Dataset.from_pandas(train_dataset)
    train_dataset.set_format(type="pandas")
    train_dataframe = train_dataset[:]
    train_dataframe["Words per String (per Class)"] = (
        train_dataframe["Poem"].str.split().apply(len)
    )
    train_dataframe.boxplot(
        "Words per String (per Class)",
        by="Category",
        grid=False,
        showfliers=False,
        color="black",
    )
    plt.suptitle("")
    plt.xlabel("Class (0: Non-Troll. 1:Troll)")
    # plt.savefig(save_path + '/text_lengths.png', bbox_inches='tight', dpi=1000)


pretrained_model_name = "distilbert/distilbert-base-multilingual-cased"

# loading pretrained model
transformer = AutoModel.from_pretrained(pretrained_model_name).to(device)

# tokenizer used needs to match pretrained model
# max context size 512 tokens
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name, model_max_length=transformer.config.max_position_embeddings
)


def check_model(test_string):
    """checks if forward pass of imported model works.
    encode() method only applies the tokenizer to get the input_ids,
    not the attention masks as well"""

    transformer = AutoModel.from_pretrained(pretrained_model_name).to(device)

    # shape = (batch_size, n_tokens)
    text_tensor = tokenizer.encode(test_string, return_tensors="pt").to(device)
    print("input text tensor size {} \n".format(text_tensor.shape))

    with torch.no_grad():
        outputs = transformer(text_tensor)

    # shape = (batch_size, n_tokens, hidden_dim)
    print(
        "output hidden state tensor size {} \n".format(outputs.last_hidden_state.shape)
    )

    # print (outputs.last_hidden_state[:,0])
    print(outputs)

    # remove from GPU memory
    del transformer


def tokenize(batch):
    """padding fills all strings, to match largest string size in the batch.
    truncation removes anything longer than context size, just in case.
    """

    return tokenizer(batch["Poem"], padding=True, truncation=True)


def extract_hidden_state(batch, transformer=transformer):
    """extracts hidden states to be applied to whole dataset"""

    input_ids = batch["input_ids"].detach().clone().to(device)
    attention_mask = batch["attention_mask"].detach().clone().to(device)

    with torch.no_grad():
        """this ectracts the [CLS] token at the start of each string, which
        captures information about the whole string, and thus can be used for
        classification, without the need to used all the hidden states for
        all the tokens, which would require more compute"""
        last_hidden_state = transformer(input_ids, attention_mask).last_hidden_state[
            :, 0
        ]

        # needs to be on cpu to use numpy, so we can use map() method
        last_hidden_state = last_hidden_state.cpu().numpy()

    # free up memory
    del input_ids
    del attention_mask

    return {"hidden_state": last_hidden_state}


def preprocess_dataset(dataset):
    arrow_dataset = Dataset.from_pandas(dataset)
    print("---ENCODING---")

    """tokenizing the dataset. bacth_size = None applies it to the dataset as 
    a whole (as a single batch), if not set to None, then needs to match the 
    training batch size (which is set later on)"""
    arrow_dataset = arrow_dataset.map(tokenize, batched=True, batch_size=batch_size)

    arrow_dataset.set_format(type="torch")
    print("---PASSING THROUGH MODEL---")

    """the training dataset with strings, classes, input, ids, atternion_mask, 
    and hidden_states (obtained by extract_hidden_state function) output by 
    the model. the loading this does when called shows it going 
    through the dataset once"""
    dataset_hidden = arrow_dataset.map(
        extract_hidden_state, batched=True, batch_size=batch_size
    )

    return dataset_hidden


# apply preprocess_dataset to the training dataset
train_dataset_hidden = preprocess_dataset(poem_train_df)


def dimReduction(train_dataset_hidden):
    """project the classes hidden states down to 2D (using UMAP, see paper)
    to visualise separability, note if separability not visible by eye,
    still doesn't mean they're not separable for sure"""

    print("---Creating UMAP Projection---")

    X_train = np.array(train_dataset_hidden["hidden_state"])
    y_train = np.array(train_dataset_hidden["Category"])

    ## UMAP works best with features scaled to [0,1]
    X_scaled = MinMaxScaler().fit_transform(X_train)

    mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)

    reduced_df = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    reduced_df["Category"] = y_train
    print(reduced_df.head())

    sns.set_style("whitegrid", {"axes.grid": False})
    sns.scatterplot(
        x=reduced_df["X"],
        y=reduced_df["Y"],
        hue=reduced_df["Category"],
        palette="tab10",
        legend=True,
    ).set(title="Class Separability")

    plt.xlabel("Reduced dim. 1")
    plt.ylabel("Reduced dim. 2")

    # plt.savefig('./umap_high_res.png', bbox_inches='tight', dpi=300)


# remove from GPU memory
del transformer


#%%


def prepare_torch_trainset(train_dataset_hidden):
    """create the arrays that will be transformed into
    torch tensors (from the HF Dataset)"""

    print("---Creating Torch Train Dataset---")

    train_hidden = np.array(train_dataset_hidden["hidden_state"])
    labels = np.array(train_dataset_hidden["Category"])

    hidden_states_tensor = torch.from_numpy(train_hidden).float()

    """this only if using the full hidden state, if only CLS token 
    hidden state (or avg of hidden states), then don't need this line, 
    becasue its already in the appropriate shape of batch size and the 
    number of input to the classifier network"""
    # hidden_states_tensor = hidden_states_tensor.view(A, B)

    hidden_states_tensor.requires_grad = True

    labels = torch.from_numpy(labels).float()

    classifier_train_dataset = TensorDataset(hidden_states_tensor, labels)

    ## splitting into train and validation datasets
    train_size = int(0.8 * len(classifier_train_dataset))
    val_size = len(classifier_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        classifier_train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    return train_loader, val_loader


train_loader, val_loader = prepare_torch_trainset(train_dataset_hidden)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.lin1 = torch.nn.Linear(in_features=768, out_features=512, bias=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lin2 = torch.nn.Linear(in_features=512, out_features=256, bias=True)
        self.dropout2 = nn.Dropout(0.1)
        self.lin3 = torch.nn.Linear(in_features=256, out_features=5)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.dropout1(x)
        x = F.relu(self.lin2(x))
        x = self.dropout2(x)

        """softmax NOT applied here! as the nn.CrossEntropyLoss() handles
        that internally"""
        x = self.lin3(x)
        return x


def make_train_step(model, loss_func, optimizer):
    def train_step(x, y):
        optimizer.zero_grad()  # reset gradients (because they accumulate)

        # put model in train mode
        model.train()

        # log odds outputs of the model
        y_hat = model(x)

        """model outputs y_hat shape: (batch_size,1), squeeze gets rid of the 
        1, to match shape of y: (batch_size)"""
        y_hat = y_hat.squeeze()

        """NO one hot encoding of true labels! nn.CrossEntropyLoss() also 
        handles this internally"""
        loss = loss_func(y_hat, y)
        loss.backward()  # calculate gradients
        optimizer.step()  # update parameters

        # checking training is on the correct device
        # print(next(model.parameters()).device)

        return loss.item()

    return train_step


model = Classifier()
model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=learning_rate, betas=(beta1, beta2)
)

"""learning rate scheduler, step_size determines no. of time the schduler is 
called before it does an update, so it should be balaced againsts no.epochs"""
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

train_step = make_train_step(model, loss_func, optimizer)


# defning the training loop
losses = []
val_losses = []
val_accuracy = []


def start_train_loop():
    print("---Training start time is: {} --- \n".format(datetime.datetime.now()))
    for epoch in range(n_epochs):
        for counter, data in enumerate(train_loader, start=0):

            x_batch = data[0].to(device)

            # casting to long, the CE loss requires this
            y_batch = data[1].type(torch.LongTensor)
            y_batch = y_batch.to(device)

            loss = train_step(x_batch, y_batch)
            losses.append(loss)

            # turn off gradient computation
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss = 0

                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)

                    # same as above
                    y_val = y_val.type(torch.LongTensor)
                    y_val = y_val.to(device)

                    # put model in evaluation mode
                    model.eval()

                    y_hat = model(x_val)
                    y_hat = y_hat.squeeze()
                    val_loss += loss_func(y_hat, y_val)

                    # apply softmax to log odds model output
                    softmax_outputs = torch.softmax(y_hat, dim=1)

                    # collapse probabilities to 1 predicted label
                    _, predicted_labels = torch.max(softmax_outputs, 1)

                    correct_predictions = predicted_labels == y_val
                    correct += correct_predictions.sum().item()
                    total += len(y_val)

                accuracy = 100 * correct / total
                avg_val_loss = val_loss / total

                val_accuracy.append(accuracy)
                val_losses.append(avg_val_loss.item())

                print(
                    "[epoch: {}][batch: {}]  train_loss: {}, val_accuracy: {}, avg_val_loss: {} \n".format(
                        epoch + 1, counter + 1, loss, accuracy, avg_val_loss
                    )
                )
                if (counter + 1) % 100 == 0:
                    torch.save(
                        model.state_dict(),
                        r"./temp_files/Classifier_checkpoint_weights_epoch{}_batch{}_accuracy{}.pt".format(
                            epoch + 1, counter + 1, accuracy
                        ),
                    )
                    print("-----CHECKPOINT----- \n")

        scheduler.step()

    torch.save(model.state_dict(), r"./pretrainedLLM_classifier_weights.pt")
    print("-----model is saved----- \n")
    print("-----Finsih time is: {} ----- \n".format(datetime.datetime.now()))


# call to start training loop
start_train_loop()



