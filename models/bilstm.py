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
        attention_outs_pre, self.alphas_pre = blocks.attention(premise_outs, self.attention_size, return_alphas=True, mask=tf.squeeze(prem_mask))
        drop_pre = tf.nn.dropout(attention_outs_pre, self.keep_rate_ph)
        #drop_pre = attention_outs_pre

        hypothesis_outs, hypothesis_final = blocks.biLSTM(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths, name='hypothesis')
        attention_outs_hyp, self.alphas_hyp = blocks.attention(hypothesis_outs, self.attention_size, return_alphas=True, mask=tf.squeeze(hyp_mask))
        drop_hyp = tf.nn.dropout(attention_outs_hyp, self.keep_rate_ph)
        #drop_hyp = attention_outs_hyp
        
        # Concat output of pre and hyp outpuratet
        drop = tf.concat([drop_pre, drop_hyp], axis=1)
        h_mlp = tf.nn.relu(tf.matmul(drop, self.W_mlp) + self.b_mlp)

        ############# MY CODE ENDS ########


        ############# Hex Part #########
        ############  MY CODE STARTS #########


        attention_outs_pre_hex, self.alphas_pre_hex = blocks.attention(premise_outs, self.attention_size, return_alphas=True, mask=tf.squeeze(prem_mask))
        drop_pre_hex = tf.nn.dropout(attention_outs_pre_hex, self.keep_rate_ph)
        #drop_pre = attention_outs_pre

        attention_outs_hyp_hex, self.alphas_hyp_hex = blocks.attention(hypothesis_outs, self.attention_size, return_alphas=True, mask=tf.squeeze(hyp_mask))
        drop_hyp_hex = tf.nn.dropout(attention_outs_hyp_hex, self.keep_rate_ph)
        #drop_hyp = attention_outs_hyp
        
        # Concat output of pre and hyp outpuratet
        bag_of_word_in = tf.concat([drop_pre_hex, drop_hyp_hex], axis=1)


        # Hex component inputs

        h_fc1 = h_mlp # (?, 300)

        h_fc2 = bag_of_word_in # (?, 1200)

        
        # Hex layer definition
	self.W_cl_1 = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1))
        self.W_cl_2 = tf.Variable(tf.random_normal([1200, 3]), trainable=True)
        self.b_cl = tf.Variable(tf.random_normal((3,)), trainable=True)
        self.W_cl = tf.concat([self.W_cl_1, self.W_cl_2], 0)

	# Compute prediction using  [h_fc1, 0(pad)]
        pad=tf.zeros_like(h_fc2, tf.float32)
            # print(pad.shape) -> (?, 600)

        yconv_contact_pred = tf.nn.dropout(tf.concat([h_fc1, pad],1), self.keep_rate_ph)
        y_conv_pred = tf.matmul(yconv_contact_pred, self.W_cl) + self.b_cl

        self.logits = y_conv_pred # Prediction

	# Compute loss using [h_fc1, h_fc2] and [0(pad2), h_fc2]
        pad2 = tf.zeros_like(h_fc1, tf.float32)

        yconv_contact_H = tf.nn.dropout(tf.concat([pad2, h_fc2],1), self.keep_rate_ph)
        y_conv_H = tf.matmul(yconv_contact_H, self.W_cl) + self.b_cl # get Fg


        yconv_contact_loss = tf.nn.dropout(tf.concat([h_fc1, h_fc2],1), self.keep_rate_ph)
        y_conv_loss = tf.matmul(yconv_contact_loss, self.W_cl) + self.b_cl # get Fb


        self.temp = y_conv_H
        temp = tf.matmul(y_conv_H, y_conv_H, transpose_a=True)

        y_conv_loss = y_conv_loss - tf.matmul(tf.matmul(tf.matmul(y_conv_H, tf.matrix_inverse(temp)), y_conv_H, transpose_b=True), y_conv_loss) # get loss

	cost_logits = y_conv_loss

        
        # Regularize hex attention
        alphas_pre_loss_hex = self.alphas_pre_hex + self.epsilon
        alphas_hyp_loss_hex = self.alphas_hyp_hex + self.epsilon
        reg1 = tf.reduce_mean(-tf.reduce_sum(alphas_pre_loss_hex * tf.log(alphas_pre_loss_hex), axis=1))
        reg2 = tf.reduce_mean(-tf.reduce_sum(alphas_hyp_loss_hex * tf.log(alphas_hyp_loss_hex), axis=1))
        reg = reg1 + reg2


        self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss))

        ############# MY CODE ENDS ########
