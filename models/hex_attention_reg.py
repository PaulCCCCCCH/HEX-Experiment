import tensorflow as tf
from util import blocks
import numpy as np

class MyModel(object):
    def __init__(self, seq_length, emb_dim, hidden_dim, embeddings, emb_train):
        ## Define hyperparameters
	## note: embedding_dim and hidden_dim are both 300, used interchangeably	
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.sequence_length = seq_length 

        ## Define the placeholders
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_rate_ph = tf.placeholder(tf.float32, [])

        ## Define parameters
        self.E = tf.Variable(embeddings, trainable=emb_train)
        
        self.W_mlp = tf.Variable(tf.random_normal([self.dim * 8, self.dim], stddev=0.1))
        self.b_mlp = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

        self.W_cl = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1))
        self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1))
        
        ## Function for embedding lookup and dropout at embedding layer
        def emb_drop(x):
            emb = tf.nn.embedding_lookup(self.E, x)
            emb_drop = tf.nn.dropout(emb, self.keep_rate_ph)
            return emb_drop

        # Get lengths of unpadded sentences
        prem_seq_lengths, mask_prem = blocks.length(self.premise_x)
        hyp_seq_lengths, mask_hyp = blocks.length(self.hypothesis_x)


        ### First biLSTM layer ###

        premise_in = emb_drop(self.premise_x)
        hypothesis_in = emb_drop(self.hypothesis_x)

        premise_outs, c1 = blocks.biLSTM(premise_in, dim=self.dim, seq_len=prem_seq_lengths, name='premise')
        hypothesis_outs, c2 = blocks.biLSTM(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths, name='hypothesis')

        premise_bi = tf.concat(premise_outs, axis=2)
        hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

        premise_list = tf.unstack(premise_bi, axis=1)
        hypothesis_list = tf.unstack(hypothesis_bi, axis=1)


        ### Attention ###

        scores_all = []
        premise_attn = []
        alphas = []

        for i in range(self.sequence_length):

            scores_i_list = []
            for j in range(self.sequence_length):
                score_ij = tf.reduce_sum(tf.multiply(premise_list[i], hypothesis_list[j]), 1, keep_dims=True)
                scores_i_list.append(score_ij)
            
            scores_i = tf.stack(scores_i_list, axis=1)
            alpha_i = blocks.masked_softmax(scores_i, mask_hyp)
            a_tilde_i = tf.reduce_sum(tf.multiply(alpha_i, hypothesis_bi), 1)
            premise_attn.append(a_tilde_i)
            
            scores_all.append(scores_i)
            alphas.append(alpha_i)

        scores_stack = tf.stack(scores_all, axis=2)
        scores_list = tf.unstack(scores_stack, axis=1)

        hypothesis_attn = []
        betas = []
        for j in range(self.sequence_length):
            scores_j = scores_list[j]
            beta_j = blocks.masked_softmax(scores_j, mask_prem)
            b_tilde_j = tf.reduce_sum(tf.multiply(beta_j, premise_bi), 1)
            hypothesis_attn.append(b_tilde_j)

            betas.append(beta_j)

        # Make attention-weighted sentence representations into one tensor,
        premise_attns = tf.stack(premise_attn, axis=1)
        hypothesis_attns = tf.stack(hypothesis_attn, axis=1)

        # For making attention plots, 
        self.alpha_s = tf.stack(alphas, axis=2)
        self.beta_s = tf.stack(betas, axis=2) 


        ### Subcomponent Inference ###

        prem_diff = tf.subtract(premise_bi, premise_attns)
        prem_mul = tf.multiply(premise_bi, premise_attns)
        hyp_diff = tf.subtract(hypothesis_bi, hypothesis_attns)
        hyp_mul = tf.multiply(hypothesis_bi, hypothesis_attns)

        m_a = tf.concat([premise_bi, premise_attns, prem_diff, prem_mul], 2)
        m_b = tf.concat([hypothesis_bi, hypothesis_attns, hyp_diff, hyp_mul], 2)


        ### Inference Composition ###

        v1_outs, c3 = blocks.biLSTM(m_a, dim=self.dim, seq_len=prem_seq_lengths, name='v1')
        v2_outs, c4 = blocks.biLSTM(m_b, dim=self.dim, seq_len=hyp_seq_lengths, name='v2')

        v1_bi = tf.concat(v1_outs, axis=2)
        v2_bi = tf.concat(v2_outs, axis=2)


        ### Pooling Layer ###

        v_1_sum = tf.reduce_sum(v1_bi, 1)
        v_1_ave = tf.div(v_1_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1))

        v_2_sum = tf.reduce_sum(v2_bi, 1)
        v_2_ave = tf.div(v_2_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1))

        v_1_max = tf.reduce_max(v1_bi, 1)
        v_2_max = tf.reduce_max(v2_bi, 1)

        v = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1)
        

        # MLP layer
        h_mlp = tf.nn.tanh(tf.matmul(v, self.W_mlp) + self.b_mlp)

        ############### MY CODE STARTS #####

	# Define layer size
        self.attention_size = 128

        # LSTM layer (final layer of the original esim model)
        h_fc1 = h_mlp

	# Bag-of-word input (averaging word embeddings)
	bow_pre = premise_in
	bow_hyp = hypothesis_in
            # print(bow_pre.shape) -> (?, 50, 300)


        w_pre_omega = tf.Variable(tf.random_normal([self.dim, self.attention_size], stddev=0.1))
        b_pre_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
        u_pre_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
        v_pre = tf.tanh(tf.tensordot(premise_in, w_pre_omega, axes=1) + b_pre_omega)
        vu_pre = tf.tensordot(v_pre, u_pre_omega, axes=1)
        #(Batch, dim)
        self.alphas_pre = tf.nn.softmax(vu_pre)
        weighted_pre = tf.reduce_sum(premise_in * tf.expand_dims(self.alphas_pre, -1), 1)


        w_hyp_omega = tf.Variable(tf.random_normal([self.dim, self.attention_size], stddev=0.1))
        b_hyp_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
        u_hyp_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
        v_hyp = tf.tanh(tf.tensordot(hypothesis_in, w_hyp_omega, axes=1) + b_hyp_omega)
        vu_hyp = tf.tensordot(v_hyp, u_hyp_omega, axes=1)
        #(Batch, dim)
        self.alphas_hyp = tf.nn.softmax(vu_hyp)
        weighted_hyp = tf.reduce_sum(hypothesis_in * tf.expand_dims(self.alphas_hyp, -1), 1)
        
        #(Batch, dim*2)
        self.alphas = tf.concat([self.alphas_pre, self.alphas_hyp], 1)


	bag_of_word_pre = tf.reduce_mean(bow_pre, 1)
        bag_of_word_hyp = tf.reduce_mean(bow_hyp, 1)
            # print(bag_of_word_pre.shape) -> (?, 300)
        attention_in = tf.concat([weighted_pre, weighted_hyp], 1)
            # print(bag_of_word_in.shape) -> (?, 600)
        
        # Bag-of-word input layer params
        h_fc2 = attention_in
            # print( h_fc2.shape) -> (?, 600)

	# Bag-of-word output layer params
        # weights_from_split = np.load("../../rearrangingDS/rearranged_even_seqlen50/weights.npy")
            # (600, 3)
        # bias_from_split = np.load("../../rearrangingDS/rearranged_even_seqlen50/bias.npy")
            # (3,)

	self.W_cl_1 = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1))
        self.W_cl_2 = tf.Variable(tf.random_normal([600, 3]), trainable=True)
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

        y_conv_loss = y_conv_loss - tf.matmul(tf.matmul(tf.matmul(y_conv_H, tf.matrix_inverse(tf.matmul(y_conv_H, y_conv_H, transpose_a=True))), y_conv_H, transpose_b=True), y_conv_loss) # get loss

	cost_logits = y_conv_loss

        reg1 = tf.reduce_mean(-tf.reduce_sum(self.alphas_pre * tf.log(self.alphas_pre), axis=1))
        reg2 = tf.reduce_mean(-tf.reduce_sum(self.alphas_hyp * tf.log(self.alphas_hyp), axis=1))
        reg = reg1 + reg2

        self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss)) + reg

	############### MY CODE ENDS #####
