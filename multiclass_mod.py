#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import datetime
import csv
import re
import os
from random import shuffle
from collections import defaultdict


# settings

minimum_evaluation_items = 1200 # annotate this many randomly sampled items first for evaluation data before creating training data
minimum_training_items = 400 # minimum number of training items before we first train a model

epochs = 10 # number of epochs per training session
select_per_epoch = 200  # number to select per epoch per label


data = []
test_data = []

# directories with data
unlabeled_data = "unlabeled_data/unlabeled_data.csv"

evaluation_negative = "evaluation_data/negative.csv"
evaluation_neutral = "evaluation_data/neutral.csv"
evaluation_positive = "evaluation_data/positive.csv"



#validation__data # not used in this example

training_negative = "training_data/negative.csv"
training_neutral = "training_data/neutral.csv"
training_positive = "training_data/positive.csv"



already_labeled = {} # tracking what is already labeled
feature_index = {} # feature mapping for one-hot encoding


def load_data(filepath, skip_already_labeled=False):
    # csv format: [ID, TEXT, LABEL, SAMPLING_STRATEGY, CONFIDENCE]
    with open(filepath, 'r') as csvfile:
        data = []
        reader = csv.reader(csvfile)
        for row in reader:
            if skip_already_labeled and row[0] in already_labeled:
        	    continue

            if len(row) < 3:
                row.append("") # add empty col for LABEL to add later
            if len(row) < 4:
                row.append("") # add empty col for SAMPLING_STRATEGY to add later
            if len(row) < 5:
                row.append(0) # add empty col for CONFIDENCE to add later
            data.append(row)

            label = str(row[2])
            if row[2] != "":
                textid = row[0]
                already_labeled[textid] = label

    csvfile.close()
    return data

def append_data(filepath, data):
    with open(filepath, 'a', errors='replace') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    csvfile.close()

def write_data(filepath, data):
    with open(filepath, 'w', errors='replace') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    csvfile.close()


# LOAD ALL UNLABELED, TRAINING, VALIDATION, AND EVALUATION DATA
training_data = load_data(training_negative) + load_data(training_neutral) + load_data(training_positive)
training_count = len(training_data)

evaluation_data = load_data(evaluation_negative) + load_data(evaluation_neutral) + load_data(evaluation_positive)
evaluation_count = len(evaluation_data)

data = load_data(unlabeled_data, skip_already_labeled=True)

annotation_instructions = "Please type 1 if this message's sentiment is negative, "
annotation_instructions += "type 2 if it's neutral,\n"
annotation_instructions += "or 3 if the message is positive, "
annotation_instructions += "type d to see detailed definitions, "
annotation_instructions += "or type s to save your annotations.\n"

last_instruction = "All done!\n"
last_instruction += "Type b to go back to change any labels,\n"
last_instruction += "or Enter to save your annotations."

detailed_instructions = "A message with negative sentiment indicates that the customer is upset.\n"
detailed_instructions += "It includes:\n"
detailed_instructions += " \n"
detailed_instructions += "  \n"
detailed_instructions += "  \n"
detailed_instructions += "\n"
detailed_instructions += " \n"
detailed_instructions += " \n\n"


def get_annotations(data, default_sampling_strategy="random"):
    """Prompts annotator for label from command line and adds annotations to data

    Keyword arguments:
        data -- an list of unlabeled items where each item is
                [ID, TEXT, LABEL, SAMPLING_STRATEGY, CONFIDENCE]
        default_sampling_strategy -- strategy to use for each item if not already specified
    """

    ind = 0
    while ind <= len(data):
        if ind < 0:
            ind = 0 # in case you've gone back before the first
        if ind < len(data):
            textid = data[ind][0]
            text = data[ind][1]
            label = data[ind][2]
            strategy =  data[ind][3]

            if textid in already_labeled:
                print("Skipping seen "+label)
                ind+=1
            else:
                print(annotation_instructions)
                label = str(input(text+"\n\n> "))

                if label == "b":
                    ind-=1  # go back # go back
                elif label == "d":
                    print(detailed_instructions) # print detailed instructions
                elif label == "s":
                    break  # save and exit
                else:
                    pass
                    data[ind][2] = label # add label to our data

                    if data[ind][3] is None or data[ind][3] == "":
                        data[ind][3] = default_sampling_strategy # add default if none given
                    ind+=1

        else:
            #last one - give annotator a chance to go back
            print(last_instruction)
            label = str(input("\n\n> "))
            if label == "b":
                ind-=1
            else:
                ind+=1

    return data


def create_features(minword = 3):
    """Create indexes for one-hot encoding of words in files

    """

    total_training_words = {}
    for item in data + training_data:
        text = item[1]
        for word in text.split():
            if word not in total_training_words:
                total_training_words[word] = 1
            else:
                total_training_words[word] += 1

    for item in data + training_data:
        text = item[1]
        for word in text.split():
            if word not in feature_index and total_training_words[word] >= minword:
                feature_index[word] = len(feature_index)

    return len(feature_index)


class SimpleTextClassifier(nn.Module):  # inherit pytorch's nn.Module
    """Text Classifier with 1 hidden layer

    """

    def __init__(self, num_labels, vocab_size):
        super(SimpleTextClassifier, self).__init__() # call parent init

        # Define model with one hidden layer with 128 neurons
        self.linear1 = nn.Linear(vocab_size, 128)
        self.linear2 = nn.Linear(128, num_labels)

    def forward(self, feature_vec):
        # Define how data is passed through the model

        hidden1 = self.linear1(feature_vec).clamp(min=0) # ReLU
        output = self.linear2(hidden1)
        return F.log_softmax(output, dim=1)


def make_feature_vector(features, feature_index):
    vec = torch.zeros(len(feature_index))
    for feature in features:
        if feature in feature_index:
            vec[feature_index[feature]] += 1
    return vec.view(1, -1)


def train_model(training_data, validation_data = "", evaluation_data = "", num_labels=3, vocab_size=0):
    """Train model on the given training_data

    Tune with the validation_data
    Evaluate accuracy with the evaluation_data
    """

    model = SimpleTextClassifier(num_labels, vocab_size)
    # let's hard-code our labels for this example code
    # and map to the same meaningful numbers in our data,
    # so we don't mix anything up when inspecting our data
    label_to_ix = {"negative": 1, "neutral": 2, "positive": 3}

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # epochs training
    for epoch in range(epochs):
        print("Epoch: "+str(epoch))
        current = 0

        # make a subset of data to use in this epoch
        # with an equal number of items from each label

        shuffle(training_data) #randomize the order of the training data
        negative = [row for row in training_data if '1' in row[2]]
        neutral = [row for row in training_data if '2' in row[2]]
        positive = [row for row in training_data if '3' in row[2]]

        epoch_data = negative[:select_per_epoch]
        epoch_data += neutral[:select_per_epoch]
        epoch_data += positive[:select_per_epoch]
        shuffle(epoch_data)

        # train our model
        for item in epoch_data:
            training_idx = random.randint(0,len(data)-1)
            features = item[1].split()
            label = int(item[2])

            model.zero_grad()

            feature_vec = make_feature_vector(features, feature_index)
            target = torch.LongTensor([int(label)])

            log_probs = model(feature_vec)

			# compute loss function, do backward pass, and update the gradient
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()

    fscore, auc = evaluate_model(model, evaluation_data)
    fscore = round(fscore,3)
    auc = round(auc,3)

    # save model to path that is alphanumeric and includes number of items and accuracies in filename
    timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
    training_size = "_"+str(len(training_data))
    accuracies = str(fscore)+"_"+str(auc)

    model_path = "models/"+timestamp+accuracies+training_size+".params"

    torch.save(model.state_dict(), model_path)
    return model_path


def get_low_conf_unlabeled(model, unlabeled_data, number=80, limit=10000):
    confidences = []
    if limit == -1: # we're predicting confidence on *everything* this will take a while
    	print("Get confidences for unlabeled data (this might take a while)")
    else:
    	# only apply the model to a limited number of items
    	shuffle(unlabeled_data)
    	unlabeled_data = unlabeled_data[:limit]

    with torch.no_grad():
        for item in unlabeled_data:
            textid = item[0]
            if textid in already_labeled:
                continue

            text = item[1]

            feature_vector = make_feature_vector(text.split(), feature_index)
            log_probs = model(feature_vector)

            # get confidence that it is related
            prob_related = math.exp(log_probs.data.tolist()[0][1])
            #The threshold set below is probably ultimately too low.
            if prob_related < 0.5:
                confidence = 1 - prob_related
            else:
                confidence = prob_related

            item[3] = "low confidence"
            item[4] = confidence
            confidences.append(item)

    confidences.sort(key=lambda x: x[4])
    return confidences[:number:]


def get_random_items(unlabeled_data, number = 10):
    shuffle(unlabeled_data)

    random_items = []
    for item in unlabeled_data:
        textid = item[0]
        if textid in already_labeled:
            continue
        random_items.append(item)
        if len(random_items) >= number:
            break

    return random_items


def get_outliers(training_data, unlabeled_data, number=10):
    """Get outliers from unlabeled data in training data

    Returns number outliers

    An outlier is defined as the percent of words in an item in
    unlabeled_data that do not exist in training_data
    """
    outliers = []

    total_feature_counts = defaultdict(lambda: 0)

    for item in training_data:
        text = item[1]
        features = text.split()

        for feature in features:
            total_feature_counts[feature] += 1

    while(len(outliers) < number):
        top_outlier = []
        top_match = float("inf")

        for item in unlabeled_data:
            textid = item[0]
            if textid in already_labeled:
                continue

            text = item[1]
            features = text.split()
            total_matches = 1 # start at 1 for slight smoothing
            for feature in features:
                if feature in total_feature_counts:
                    total_matches += total_feature_counts[feature]

            ave_matches = total_matches / len(features)
            if ave_matches < top_match:
                top_match = ave_matches
                top_outlier = item

        # add this outlier to list and update what is 'labeled',
        # assuming this new outlier will get a label
        top_outlier[3] = "outlier"
        outliers.append(top_outlier)
        text = top_outlier[1]
        features = text.split()
        for feature in features:
            total_feature_counts[feature] += 1

    return outliers


#MODEL EVALUATION SECTION NEEDS TO BE MODDED FOR MULTICLASS. F1 AND ROC ARE VALID METRICS, BUT EQUATIONS/LOGIC WOULD NEED TO BE CHANGED.

# def evaluate_model(model, evaluation_data):
#     """Evaluate the model on the held-out evaluation data

#     Return the f-value and the AUC for sentiment classification
#     """

#     negative_confs = [] # items and their confidence of being positive
#     neutral_confs=[] #  items and their confidence of being neutral
#     positive_confs = [] #  items and their confidence of negative

#     true_pos = 0.0 # true positives, etc
#     false_pos = 0.0
#     false_neg = 0.0

#     with torch.no_grad():
#         for item in evaluation_data:
#             _, text, label, _, _, = item

#             feature_vector = make_feature_vector(text.split(), feature_index)
#             log_probs = model(feature_vector)

#             # get confidence that item is disaster-related
#             prob_related = math.exp(log_probs.data.tolist()[0][1])

#             if(label == "1"):
#                 # positive
#                 negative_confs.append(negative_confs)
#                 if prob_related > 0.5:
#                     true_pos += 1.0
#                 else:
#                     false_neg += 1.0

#             elif(label == "2"):
#                 # neutral
#                 neutral_confs.append(neutral_confs)
#                 if prob_related > 0.5:
#                     true_pos += 1.0
#                 else:
#                     false_neg += 1.0

#             elif(label == "3"):
#                 # negative
#                 positive_confs.append(positive_confs)
#                 if prob_related > 0.5:
#                     true_pos += 1.0
#                 else:
#                     false_neg += 1.0

#     # Get FScore
#     if true_pos == 0.0:
#         fscore = 0.0
#     else:
#         precision = true_pos / (true_pos + false_pos)
#         recall = true_pos / (true_pos + false_neg)
#         fscore = (2 * precision * recall) / (precision + recall)

#     # GET AUC
#     negative_confs.sort()
#     total_greater = 0 # count of how many total have higher confidence
#     for conf in negative_confs:
#         for conf2 in neutral_confs:
#             if conf < conf2:
#                 break
#             else:
#                 total_greater += 1


#     denom = len(negative_confs) * len(neutral_confs) * len(positive_confs)
#     auc = total_greater / denom

#     return[fscore, auc]



# if evaluation_count <  minimum_evaluation_items:
#     #Keep adding to evaluation data first
#     print("Creating evaluation data:\n")

#     shuffle(data)
#     needed = minimum_evaluation_items - evaluation_count
#     data = data[:needed]
#     print(str(needed)+" more annotations needed")

#     data = get_annotations(data)

#     negative = []
#     neutral=[]
#     positive = []

#     for item in data:
#         label = item[2]
#         if label == "1":
#             negative.append(item)
#         elif label =="2":
#             neutral.append(item)
#         elif label == "3":
#             positive.append(item)

#     # append evaluation data
#     append_data(evaluation_negative, negative)
#     append_data(evaluation_neutral, neutral)
#     append_data(evaluation_positive, positive)

# elif training_count < minimum_training_items:
#     # lets create our first training data!
#     print("Creating initial training data:\n")

#     shuffle(data)
#     needed = minimum_training_items - training_count
#     data = data[:needed]
#     print(str(needed)+" more annotations needed")

#     data = get_annotations(data)

#     negative = []
#     neutral=[]
#     positive= []

#     for item in data:
#         label = item[2]
#         if label == "1":
#             negative.append(item)
#         elif label == "2":
#             neutral.append(item)
#         elif label == "3":
#             positive.append(item)

#     # append training data
#     append_data(training_negative, negative)
#     append_data(training_neutral, neutral)
#     append_data(training_positive, positive)

# else:
#     # lets start Active Learning!!

# 	# Train new model with current training data
#     vocab_size = create_features()
#     model_path = train_model(training_data, evaluation_data=evaluation_data, vocab_size=vocab_size)

#     print("Sampling via Active Learning:\n")

#     model = SimpleTextClassifier(3, vocab_size)
#     model.load_state_dict(torch.load(model_path))

# 	# get 100 items per iteration with the following breakdown of strategies:
#     random_items = get_random_items(data, number=10)
#     print("About to get low confidences.")
#     low_confidences = get_low_conf_unlabeled(model, data, number=80)
#     outliers = get_outliers(training_data+random_items+low_confidences, data, number=10)

#     sampled_data = random_items + low_confidences + outliers
#     shuffle(sampled_data)

#     sampled_data = get_annotations(sampled_data)
#     negative = []
#     neutral = []
#     positive = []
#     for item in sampled_data:
#         label = item[2]
#         if label == "1":
#             negative.append(item)
#         elif label == "2":
#             neutral.append(item)
#         elif label == "3":
#             positive.append(item)

#     # append training data
#     append_data(training_negative, negative)
#     append_data(training_neutral, neutral)
#     append_data(training_positive, positive)


# if training_count > minimum_training_items:
#     print("\nRetraining model with new data")

# 	# UPDATE OUR DATA AND (RE)TRAIN MODEL WITH NEWLY ANNOTATED DATA
#     training_data = load_data(training_negative) + load_data(training_neutral) + load_data(training_positive)
#     training_count = len(training_data)

#     evaluation_data = load_data(evaluation_negative) + load_data(evaluation_neutral) + load_data(evaluation_positive)
#     evaluation_count = len(evaluation_data)

#     vocab_size = create_features()
#     model_path = train_model(training_data, evaluation_data=evaluation_data, vocab_size=vocab_size)
#     model = SimpleTextClassifier(3, vocab_size)
#     model.load_state_dict(torch.load(model_path))

#     accuracies = evaluate_model(model, evaluation_data)
#     print("[fscore, auc] =")
#     print(accuracies)
#     print("Model saved to: "+model_path)
