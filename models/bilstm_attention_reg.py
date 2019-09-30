import tensorflow as tf
from util import blocks

class MyModel(object):
    def __init__(self, seq_length, emb_dim, hidden_dim, embeddings, emb_train):
        ## Define hyperparameters
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.attention_size = 128
        self.mlp_size = self.dim
        self.sequence_length = seq_length 
        self.lam = 0.01
        self.epsilon = 1e-10

        ## Define the placeholders
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_rate_ph = tf.placeholder(tf.float32, [])

        ## Define parameters
        self.E = tf.Variable(embeddings, trainable=emb_train)
        
        self.W_mlp = tf.Variable(tf.random_normal([self.dim * 4, self.mlp_size], stddev=0.1))
        self.b_mlp = tf.Variable(tf.random_normal([self.mlp_size], stddev=0.1))

        self.W_cl = tf.Variable(tf.random_normal([self.mlp_size, 3], stddev=0.1))
        self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1))
        
        ## Function for embedding lookup and dropout at embedding layer
        def emb_drop(x):
            emb = tf.nn.embedding_lookup(self.E, x)
            emb_drop = tf.nn.dropout(emb, self.keep_rate_ph)
            return emb_drop

        # Get lengths of unpadded sentences
        prem_seq_lengths, prem_mask = blocks.length(self.premise_x)
        hyp_seq_lengths, hyp_mask = blocks.length(self.hypothesis_x)


        ### BiLSTM layer ###
        premise_in = emb_drop(self.premise_x)
        hypothesis_in = emb_drop(self.hypothesis_x)


        ############# MY CODE STARTS ########


        premise_outs, premise_final = blocks.biLSTM(premise_in, dim=self.dim, seq_len=prem_seq_lengths, name='premise')
        attention_outs_pre, self.alphas_pre = blocks.attention(premise_outs, self.attention_size, return_alphas=True)
        drop_pre = tf.nn.dropout(attention_outs_pre, self.keep_rate_ph)
        #drop_pre = attention_outs_pre

        hypothesis_outs, hypothesis_final = blocks.biLSTM(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths, name='hypothesis')
        attention_outs_hyp, self.alphas_hyp = blocks.attention(hypothesis_outs, self.attention_size, return_alphas=True)
        drop_hyp = tf.nn.dropout(attention_outs_hyp, self.keep_rate_ph)
        #drop_hyp = attention_outs_hyp
        
        # Concat output of pre and hyp outpuratet
        drop = tf.concat([drop_pre, drop_hyp], axis=1)

        # Add a small constant
        alphas_pre_loss = self.alphas_pre * tf.squeeze(prem_mask) + self.epsilon
        alphas_hyp_loss = self.alphas_hyp * tf.squeeze(hyp_mask) + self.epsilon

        # Calculate entropy
        reg1 = tf.reduce_mean(-tf.reduce_sum(alphas_pre_loss * tf.log(alphas_pre_loss), axis=1))
        reg2 = tf.reduce_mean(-tf.reduce_sum(alphas_hyp_loss * tf.log(alphas_hyp_loss), axis=1))
        reg = reg1 + reg2


        # MLP layer
        h_mlp = tf.nn.relu(tf.matmul(drop, self.W_mlp) + self.b_mlp)

        ############# MY CODE ENDS ########

        # Get prediction
        self.logits = tf.matmul(h_mlp, self.W_cl) + self.b_cl

        # Define the cost function
        self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits) + self.lam * reg) 

