---
title: LLM Chat Agents - Lightweight Tool Selection (with BERT)
date: "2024-01-02T19:15:00.121Z"
template: "post"
draft: false
slug: "llm-agents-lightweight-tool-selection-with-bert"
category: "Machine Learning"
tags:
  - "LLM"
  - "Agents"
description: "This article explores the concept of tools selection--a process that involves understanding a user's intent and subsequently choosing the tool most suitable to handle their request. While an LLM stands as an ideal candidate for this selection process, the inherent costs and high latencies associated with LLM interactions, prompt an exploration of alternatives."
socialImage: "./media/bert-classifier.jpeg"
---

In the realm of conversational systems, Agent architecture serves as the blueprint for crafting intelligent bots, weaving together various elements to shape a human-like conversational experience. Agents typically have access to:

1. **Knowledge Base** - A set of external datasources, distinct from the Large Language Model (LLM), the knowledge base provides additional information, supplementing the LLMs understanding through in-context learning.
2. **Memory** - An agent capability that allows it to persist and recollect information from past interactions, enhancing its contextual awareness.
3. **Tools** - A toolkit of skills tailored to specialized tasks, broadening the agent's capabilities. Tools provide versatility, enabling the agent to, for example, execute an internet search or solve advanced math problems.

Collectively, these components influence agent responses, fostering engaging interactions with users. In contemporary Agent architecture, Large Language Models (LLMs) play a key role, introducing elements of reasoning and understanding that were previously unattainable.


![llm agent tool selection](/media/bert-classifier.jpeg)

### Tool Selection

This article explores the concept of tools selection, a process that entails understanding a user's intent and choosing the tool most suitable to address their request. While LLMs are an ideal choice tool selection, the inherent costs and considerable latencies (currently) tied to LLM interactions, prompt an exploration of alternatives.

This article suggests an approach that restricts expensive LLM interactions to high-stakes scenaios, reserving them for final response generation. It does this through the use of simpler classification-based techniques for tool selection. Although an LLM will perform the task more reliably amongst a broad set of tools, a classifier-based approach can also prove highly effective, presenting a pragrmatic balance between accuracy and efficiency.

## The Scenario

Consider the scenario where we manage a online bank. Integral to our experience is a user-friendly chatbot equipped with access to our customer-help knowledge base and a suite of our read-only Banking APIs. The knowledge base encompasses a variety of documents, including FAQs, customer help documents, and other informative content. Additionally, the Banking API provides a read-only interface capable of listing balances, transactions, and statemants for the account holder. Each of these capabilities is available to the chatbot as a tool, the KnowledgeBaseTool and ListBankingDetailsTool respectively.

Our objective is to ensure the agent makes effective use of its tools, using the KnowledgeBaseTool for all requests, except those corresponding to the user's account details. To achieve this, we will train a binary classifier. However it is important to note that scenarios with multiple tools can be accommodated through techniques like multi-class classificaiton, such as a one-vs-all classifier or multiple independent binary classifiers.

## The Classifier - BERT

We will use pre-trained BERT embeddings--Bidirectional Encoder Representations, a groundbreaking natural language processing (NLP) technique introduced by Google in 2018. BERT belongs to the Transformer architecture family and can understand and encode contextual nuances of language into its embeddings.

The bidirectional nature of BERT allows it to capture relationships and meanings in language by considering both left and right contexts of a text at all layers of its neural network. This makes BERT a formiddable choice for our classifier, as it is capable of understanding linguistic nuances that help distinguish intent, ultimately, guiding tool selection within our scenario.

## Building the classifier

The goal of the classifier is to ensure that the best tool is selected to handle the user's request. The tool, ListBankingDetailsTools, is responsible for retrieving banking details by interacting with the Banking API and providing those details as output. The Agent will present the tool's output as context when prompting the LLM, providing the LLM the information needed to generate its final answer.

The following sections focus on the creation, training, and evaluation of the tool classifier.

### Training data

Training a classifier involves training data, a set of (text, label) pairs, used to help our classifier learn utterances that express the intent to list banking details. We'll capture these examples in a comma-separated file, `data.csv`.

The file will include positive examples, such as:

```csv
text,label
show me my most recent bank statements,show_bank_details
what's my balance,show_bank_details
how much money to i have in my account?,show_bank_details
show recent transactions,show_bank_details
```

And negative examples, like:

```csv
text,label
how do i login?,other
my atm card isn't working?,other
where is the closest bank?,other
```

When constructing our training data, we observe that many of our positive examples convey a "show me" intent and include terms like transactions, balance, statements. Additionally, we will also expect uttarances that lack these terms yet convey a similar intent, like "How much did i spend last month?". Our classifier must be robust to these utterances. Similar to "Show my statements", this utterance can be satisifed by invoking the ListBankingDetaisTool to retrieve "recent monthly statements". The Agent will gather the tool response and include it as context when constructing its prompt to the LLM. The LLM uses the context to generate its final answer. For example, "Based on your last statement, you spent $1037.36 last month".

BERT is a fantastic fit for this classification case, given its ablity to understand context and meaning. Ultimately, the main goal of the classifier is to choose the best tool.

You might be thinking, sure, but how does the tool know which API to call. For now, let's keep it simple and assume the ListBankingDetaisTool invokes all three Banking APIs returning all results as output. While the LLM is capable of interpretting all three results, it doesn't scale well. We can leverage additional techniques to ensure that only the most appropriate API, statement details in this case, is invoked. Such techiques are beyond the scope of this article.

### Tip

- Use an LLM to generate additional positive and negative training examples

## Training the classifier

### Train / Test splits

With our training data constructed, let's create train test splits. Train-test splits are an important component in the process of training and evaluating machine learning models. The main idea behind these splits is to divide the available dataset into two subsets: one for training the model and the other for evaluating its performance. These subsets are commonly referred to as the training set and the test set. We will hold out `20%` of our data for test, allowing our ability to test the model on these unseen examples.

We'll use sklearn

```python
from sklearn.model_selection import train_test_split

# 1. Load the data
data = pd.read_csv("/path/to/data.csv")
texts = data["text"]
labels = data["label"]

# Create train and test splits
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42
)
```

### Vectorize the texts using BERT embeddings

We use the `bert-base-uncased` pre-trained model to created vectorized embeddings for our `text` examples. Recall that BERTembeddings encode contextual information within its embedding vectors, enabling it to better understand context and the meaning a given text.

**Note** that the same tokenizer is used to create embeddings for texts when traiing our classifier (below) as when tokenizing a user input at inference time.

Below is the code used to create BERT embeddings for our training texts:

```python
from transformers import BertTokenizer

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_inputs = tokenizer(
    train_texts,
    padding=True,
    truncation=True,
    return_tensors='pt')
```

### Train the BERT classifier

Next, we'll train our binary classifier to detect whether or not to use the ListBankingDetailsTools. The classifier uses the pre-trained BERT model to create token embeddings. Note that we must the same pre-trained BERT model, `bert-base-uncased` as used to encode our training data above. Additionally, we will use the `BertForSequenceClassification` class with `2` labels to represent our binary classification model.

See the bullets below for an explanation of code blocks, 1-5.

```python
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset

## 1. Model Initialization
model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )  # Binary classification, so num_labels=2

## 2. Dataset Preparation
train_dataset = TensorDataset(
    train_inputs['input_ids'],
    train_inputs['attention_mask'],
    train_labels)

## 3. Data Loader Setup
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True)

## 4. Optimizer definition
optimizer = AdamW(
    model.parameters(),
    lr=1e-5)

## 5. Model Training
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

1. Initializes a pre-trained BERT model for sequence classification with two output labels (binary classification).
2. Creates a PyTorch TensorDataset from the training inputs (input IDs and attention mask) and corresponding labels.
3. Creates a PyTorch DataLoader to handle batches of training data. Batches of size `8` are used, and the data is shuffled during each epoch.
4. Defines the optimizer (AdamW) with a fairly high learning rate of `1e-5` due to our small number of epochs, for updating the model parameters during training.
5. Loops over the specified number of epochs, setting the model to training mode.
   Iterates through batches of training data, computes the model's output, calculates the loss, and performs backpropagation with gradient descent to update the model parameters.

## Evaluate the classifier

At this point in our classifier development, we shift focus to assessing the model's performance. The evaluation process involves testing the classifier on previously held out test data--examples not seen during training. The unseen examples are input into are model to determine the accuracy of its predictions. Accuracy is used to meansure well our classifier correctly predicts the label for each unseen example.

```python
import torch
from sklearn.metrics import accuracy_score

# 1. Tokenize the test texts
test_inputs = tokenizer(
    test_texts,
    padding=True,
    truncation=True,
    return_tensors='pt')

# 2. Evaluate the model on the test set
model.eval()
with torch.no_grad():
    logits = model(
        test_inputs['input_ids'],
        attention_mask=test_inputs['attention_mask']
    ).logits

# 3. Compte predictions and output accuracy
predictions = torch.argmax(logits, dim=1)
accuracy = accuracy_score(test_labels, predictions.numpy())
print(f"Accuracy on the test set: {accuracy:.2f}")
```

1. Converts each text in the test set into a vector embedding
2. Sets the model to evaluation mode and disable gradients (since we will not be updating the model's weights), and computes the raw (un-normalized) scores for row (and class).
3. Converts logits to the predicted class, then outputs accuracy. Logits will have form

```python
tensor([
  [ 0.6418,  0.0498],
  [-0.1198,  0.9538],
  [-0.4276,  0.6776],
  [ 0.7586, -0.3332],
  ...
  [-0.0584,  0.8782],
  [-0.1820,  0.4000],
  [-0.1534,  0.9564]
])
```

and prediction is converted to

```
tensor([0, 1, 1, 0, ... 1, 1, 1])
```

## Use the classifier

Now that we have trained and evaluated our tools classifier, let's use it to make predictions on the best tool to handle a user inputs.

```python
# 1. Tokenize the user input
text = "show me my account balance"
input_ids = tokenizer.encode(
    text,
    add_special_tokens=True,
    padding=True,
    truncation=True,
    return_tensors='pt')

# 2. Make the prediction
model.eval()
with torch.no_grad():
    logits = model(input_ids).logits

# Get the predicted class (1 or 0)
prediction = torch.argmax(logits, dim=1).item()

print("prediction")
```

In this example, we take a user input text, tokenize it using the same tokenizer used during training, and then pass it through the trained model.

The model predicts a 1 indicating the Agent should use the ListBankingDetailsTool!

## Conclusion

In summary, an LLMs ability to reason and understand text, undoubtedly makes them an ideal candidate for tool selection, however their inherent costs and high latency can pose challenges in real-world conversational systems. We have presented an alternative approach, leveraging BERT embeddings and a lightweight classifier for efficient low-latency tool selection. While an LLM may perform that task more reliably, classifiers can prove highly effective given the right circumstances.

```

```
