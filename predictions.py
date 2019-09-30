"""
Script to generate a CSV file of predictions on the test data.
"""

import tensorflow as tf
import os
import importlib
import random
from util import logger
import util.parameters as params
from util.data_processing import *
from util.evaluate import *
import pickle
import numpy as np
import re

FIXED_PARAMETERS = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]
logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)

model = FIXED_PARAMETERS["model_type"]

module = importlib.import_module(".".join(['models', model])) 
MyModel = getattr(module, 'MyModel')

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistenyl use the same hyperparameter settings. 
logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)


######################### LOAD DATA #############################

logger.Log("Loading data")

dev_snli = load_nli_data(FIXED_PARAMETERS["dev_snli"], snli=True)
test_snli = load_nli_data(FIXED_PARAMETERS["test_snli"], snli=True)
dev_mismatched = load_nli_data(FIXED_PARAMETERS["dev_mismatched"])
"""
>>>> Original
training_snli = load_nli_data(FIXED_PARAMETERS["training_snli"], snli=True)

training_mnli = load_nli_data(FIXED_PARAMETERS["training_mnli"])
dev_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"])
dev_mismatched = load_nli_data(FIXED_PARAMETERS["dev_mismatched"])

test_matched = load_nli_data(FIXED_PARAMETERS["test_matched"])
test_mismatched = load_nli_data(FIXED_PARAMETERS["test_mismatched"])
"""

dictpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".p"

if not os.path.isfile(dictpath): 
    print "No dictionary found!"
    exit(1)

else:
    logger.Log("Loading dictionary from %s" % (dictpath))
    word_indices = pickle.load(open(dictpath, "rb"))
    logger.Log("Padding and indexifying sentences")
    """
    >>>>>>>>>Original
    sentences_to_padded_index_sequences(word_indices, [training_mnli, training_snli, dev_matched, dev_mismatched, dev_snli, test_snli, test_matched, test_mismatched])
    """
    sentences_to_padded_index_sequences(word_indices, [dev_snli, test_snli, dev_mismatched])
loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)

class modelClassifier:
    def __init__(self, seq_length):
        ## Define hyperparameters
        self.learning_rate =  FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.emb_train = FIXED_PARAMETERS["emb_train"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"] 
        self.alpha = FIXED_PARAMETERS["alpha"]

        logger.Log("Building model from %s.py" %(model))
        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=loaded_embeddings, emb_train=self.emb_train)

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.model.total_cost)

        # tf things: initialize variables and create placeholder for session
        logger.Log("Initializing variables")
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()

    def get_minibatch(self, dataset, start_index, end_index):
        indices = range(start_index, end_index)
        premise_vectors = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence'] for i in indices])
        hypothesis_vectors = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence'] for i in indices])
        labels = [dataset[i]['label'] for i in indices]
        return premise_vectors, hypothesis_vectors, labels

    def classify(self, examples, return_alphas=False):
        # This classifies a list of examples
        best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver.restore(self.sess, best_path)
        logger.Log("Model restored from file: %s" % best_path)

        result = []
        all_alphas_pre = []
        all_alphas_hyp = []

        for i in range(19):
            logits = np.empty(3)
        #minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels = self.get_minibatch(examples, 0, len(examples))
        ############ NEED CHANGE ############
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels = self.get_minibatch(examples, i * 500, i * 500 + 500)


            feed_dict = {self.model.premise_x: minibatch_premise_vectors, 
                            self.model.hypothesis_x: minibatch_hypothesis_vectors, 
                            self.model.keep_rate_ph: 1.0}
            logit, alphas_pre, alphas_hyp = self.sess.run([self.model.logits, self.model.alphas_pre_hex, self.model.alphas_hyp_hex], feed_dict)
            logits = np.vstack([logits, logit])

            temp = np.argmax(logits[1:], axis=1)
            result.extend(temp)
            all_alphas_pre.extend(alphas_pre)
            all_alphas_hyp.extend(alphas_hyp)
        if return_alphas:
            return result, all_alphas_pre, all_alphas_hyp
        return result
        #####################################


def html_render(x_orig, alphas, max_len=50):
    epsilon = 1e-10
    k = 80
    b = 600
    #color_vals = (100 + np.log(alphas)) * 40
    color_vals = k * np.log(alphas + epsilon) + b
    ############################### Need Changing ###################################
    #x_orig_words = x_orig.split(' ')[:max_len]
    
    ############################### Need Changing ###################################
    words = []
    parts = x_orig.split(',')
    for part in parts:
        part_words = part.split()
        for word in part_words:
            words.append(word)
        words.append(',')

    x_orig_words = words[:max_len]
    #################################################################################

    orig_html = []
    for i in range(len(x_orig_words)):
        color_val = color_vals[i]
        colors = [0, 0, 0]
        if color_val >= 510:
            colors = [255, 0, 0]
        elif color_val >= 255:
            colors = [int(color_val - 255), 0, 0]
        else:
            colors = [0, 255 - int(color_val), 0]

        orig_html.append(format("<b style='color:rgb(%d,%d,%d)'>%s</b>" %(colors[0], colors[1], colors[2], x_orig_words[i])))
    
    orig_html = ' '.join(orig_html)
    return orig_html



classifier = modelClassifier(FIXED_PARAMETERS["seq_length"])

"""
Get CSVs of predictions.
"""

"""
>>>>>Original
logger.Log("Creating CSV of predicitons on matched test set: %s" %(modname+"_matched_predictions.csv"))
predictions_kaggle(classifier.classify, test_matched, FIXED_PARAMETERS["batch_size"], modname+"_dev_matched")

logger.Log("Creating CSV of predicitons on mismatched test set: %s" %(modname+"_mismatched_predictions.csv"))
predictions_kaggle(classifier.classify, test_mismatched, FIXED_PARAMETERS["batch_size"], modname+"_dev_mismatched")
"""

############ NEED CHANGE ############
eval_set = dev_mismatched
#####################################

INVERSE_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
    }

MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

print "Classification started"
hypotheses, all_alphas_pre, all_alphas_hyp = classifier.classify(eval_set, True)
print "Classification finished"
predictions = []
html_content = []

############ NEED CHANGE ############
for i in range(1000):
    hypothesis = hypotheses[i]
    #prediction = INVERSE_MAP[hypothesis]
    prediction = hypothesis
    target = MAP[eval_set[i]["gold_label"]]
    pairID = eval_set[i]["pairID"]
    premise = eval_set[i]["sentence1"]
    hypothesis = eval_set[i]["sentence2"]
    predictions.append((prediction, target))

    line = "<p>"
    line += "Sample %d ###################" %i
    line += "<p>"
    line += "<p>"
    line += INVERSE_MAP[prediction]
    line += " <-prediction...|...target-> "
    line += INVERSE_MAP[target]
    line += "<p>"

    line += html_render(premise, all_alphas_pre[i])
    line += "</p>"
    line += str(all_alphas_pre[i])
    line += "<p>"
    line += html_render(hypothesis, all_alphas_hyp[i])
    line += "</p>"
    line += str(all_alphas_hyp[i])
    line += "<p>"
    line += "###############################"
    line += "</p>"
    html_content.append(line)


content = "\n".join(html_content)

#predictions = sorted(predictions, key=lambda x: int(x[0]))
logger.Log("Creating CSV of predicitons on dev set: %s" %(modname+"_matched_predictions.csv"))


############ NEED CHANGE ############
name = 'bilstm_attention_dev_mismatched_hexpart'
f = open( name + '_predictions.csv', 'wb')
w = csv.writer(f, delimiter = '\t')
#w.writerow(['predicted label', 'target label', 'premise','hypothesis'])
for example in predictions:
    w.writerow(example)
f.close()

f2 = open( name + '_vis.html', 'wb')
f2.write(content)
f.close()

