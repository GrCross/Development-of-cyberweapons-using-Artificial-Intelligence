def read_data(file_path):
    tokens = []
    tags = []
    
    tweet_tokens = []
    tweet_tags = []
    for line in open(file_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            if tweet_tokens:
                tokens.append(tweet_tokens)
                tags.append(tweet_tags)
            tweet_tokens = []
            tweet_tags = []
        else:
            token, tag = line.split()
            # Replace all urls with <URL> token
            # Replace all users with <USR> token

            
            if(token.startswith('@')):
                token = '<USR>'
            elif(token.startswith('http://') or token.startswith('https://')):
                token = '<URL>'
            tweet_tokens.append(token)
            tweet_tags.append(tag)
            
    return tokens, tags




train_tokens, train_tags = read_data('data/train.txt')
validation_tokens, validation_tags = read_data('data/validation.txt')
test_tokens, test_tags = read_data('data/test.txt')






from collections import defaultdict




def build_dict(tokens_or_tags, special_tokens):
    """
        tokens_or_tags: a list of lists of tokens or tags
        special_tokens: some special tokens
    """
    # Create a dictionary with default value 0
    tok2idx = defaultdict(lambda: 0)
    idx2tok = []
    
    # Create mappings from tokens to indices and vice versa
    # Add special tokens to dictionaries
    # The first special token must have index 0
    
    
    idx = 0
    for token in special_tokens:
        idx2tok.append(token)
        tok2idx[token] = idx
        idx += 1

    for token_list in tokens_or_tags:
        for token in token_list:
            if token not in tok2idx:
                idx2tok.append(token)
                tok2idx[token] = idx
                idx += 1
    
    return tok2idx, idx2tok



special_tokens = ['<UNK>', '<PAD>']
special_tags = ['O']


token2idx, idx2token = build_dict(train_tokens + validation_tokens, special_tokens)
tag2idx, idx2tag = build_dict(train_tags, special_tags)





print(tag2idx)


# The next additional functions will help you to create the mapping between tokens and ids for a sentence. 



print(idx2tag)



print(len(idx2tag))



def words2idxs(tokens_list):
    return [token2idx[word] for word in tokens_list]

def tags2idxs(tags_list):
    return [tag2idx[tag] for tag in tags_list]

def idxs2words(idxs):
    return [idx2token[idx] for idx in idxs]

def idxs2tags(idxs):
    return [idx2tag[idx] for idx in idxs]



def batches_generator(batch_size, tokens, tags,
                      shuffle=True, allow_smaller_last_batch=True):
    """Generates padded batches of tokens and tags."""
    
    n_samples = len(tokens)
    if shuffle:
        order = np.random.permutation(n_samples)
    else:
        order = np.arange(n_samples)

    n_batches = n_samples // batch_size
    if allow_smaller_last_batch and n_samples % batch_size:
        n_batches += 1

    for k in range(n_batches):
        batch_start = k * batch_size
        batch_end = min((k + 1) * batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        x_list = []
        y_list = []
        max_len_token = 0
        for idx in order[batch_start: batch_end]:
            x_list.append(words2idxs(tokens[idx]))
            y_list.append(tags2idxs(tags[idx]))
            max_len_token = max(max_len_token, len(tags[idx]))
            
        # Fill in the data into numpy nd-arrays filled with padding indices.
        x = np.ones([current_batch_size, max_len_token], dtype=np.int32) * token2idx['<PAD>']
        y = np.ones([current_batch_size, max_len_token], dtype=np.int32) * tag2idx['O']
        lengths = np.zeros(current_batch_size, dtype=np.int32)
        for n in range(current_batch_size):
            utt_len = len(x_list[n])
            x[n, :utt_len] = x_list[n]
            lengths[n] = utt_len
            y[n, :utt_len] = y_list[n]
        yield x, y, lengths


import tensorflow as tf
import numpy as np



class BiLSTMModel():
    pass



def declare_placeholders(self):
    """Specifies placeholders for the model."""

    # Placeholders for input and ground truth output.
    self.input_batch = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_batch') 
    self.ground_truth_tags = tf.placeholder(dtype=tf.int32, shape=[None, None], name='ground_truth_tags')
  
    # Placeholder for lengths of the sequences.
    self.lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='lengths')
    
    # Placeholder for a dropout keep probability. If we don't feed
    # a value for this placeholder, it will be equal to 1.0.
    self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])
    
    # Placeholder for a learning rate (tf.float32).
    self.learning_rate_ph = tf.placeholder_with_default(1e4, shape=[])



BiLSTMModel.__declare_placeholders = classmethod(declare_placeholders)



def build_layers(self, vocabulary_size, embedding_dim, n_hidden_rnn, n_tags):
    """Specifies bi-LSTM architecture and computes logits for inputs."""
    
    # Create embedding variable (tf.Variable) with dtype tf.float32
    initial_embedding_matrix = np.random.randn(vocabulary_size, embedding_dim) / np.sqrt(embedding_dim)
    embedding_matrix_variable = tf.Variable(initial_embedding_matrix, name='embeddings_matrix', dtype=tf.float32)
    
    # Create RNN cells (for example, tf.nn.rnn_cell.BasicLSTMCell) with n_hidden_rnn number of units 
    # and dropout (tf.nn.rnn_cell.DropoutWrapper), initializing all *_keep_prob with dropout placeholder.
    forward_cell = tf.nn.rnn_cell.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden_rnn, forget_bias=3.0),
        input_keep_prob=self.dropout_ph,
        output_keep_prob=self.dropout_ph,
        state_keep_prob=self.dropout_ph
    )
    backward_cell = tf.nn.rnn_cell.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden_rnn, forget_bias=3.0),
        input_keep_prob=self.dropout_ph,
        output_keep_prob=self.dropout_ph,
        state_keep_prob=self.dropout_ph
    )

    # Look up embeddings for self.input_batch (tf.nn.embedding_lookup).
    # Shape: [batch_size, sequence_len, embedding_dim].
    embeddings = tf.nn.embedding_lookup(embedding_matrix_variable, self.input_batch)
    
    # Pass them through Bidirectional Dynamic RNN (tf.nn.bidirectional_dynamic_rnn).
    # Shape: [batch_size, sequence_len, 2 * n_hidden_rnn]. 
    # Also don't forget to initialize sequence_length as self.lengths and dtype as tf.float32.
    (rnn_output_fw, rnn_output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw= forward_cell, cell_bw= backward_cell,
        dtype=tf.float32,
        inputs=embeddings,
        sequence_length=self.lengths
    )
    rnn_output = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)

    # Dense layer on top.
    # Shape: [batch_size, sequence_len, n_tags].   
    self.logits = tf.layers.dense(rnn_output, n_tags, activation=None)


BiLSTMModel.__build_layers = classmethod(build_layers)



def compute_predictions(self):
    """Transforms logits to probabilities and finds the most probable tags."""
    
    # Create softmax (tf.nn.softmax) function
    softmax_output = tf.nn.softmax(self.logits)
    
    # Use argmax (tf.argmax) to get the most probable tags
    # Don't forget to set axis=-1
    # otherwise argmax will be calculated in a wrong way
    self.predictions = tf.argmax(softmax_output, axis=-1)



BiLSTMModel.__compute_predictions = classmethod(compute_predictions)



def compute_loss(self, n_tags, PAD_index):
    """Computes masked cross-entopy loss with logits."""
    
    # Create cross entropy function function (tf.nn.softmax_cross_entropy_with_logits)
    ground_truth_tags_one_hot = tf.one_hot(self.ground_truth_tags, n_tags)
    loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_tags_one_hot, logits=self.logits)
    
    # Create loss function which doesn't operate with <PAD> tokens (tf.reduce_mean)
    mask = tf.cast(tf.not_equal(loss_tensor, PAD_index), tf.float32)
    self.loss =  tf.reduce_mean(tf.reduce_sum(tf.multiply(loss_tensor, mask), axis=-1) / tf.reduce_sum(mask, axis=-1))




BiLSTMModel.__compute_loss = classmethod(compute_loss)




def perform_optimization(self):
    """Specifies the optimizer and train_op for the model."""
    
    # Create an optimizer (tf.train.AdamOptimizer)
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
    self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
    
    # Gradient clipping (tf.clip_by_norm) for self.grads_and_vars
    # Pay attention that you need to apply this operation only for gradients 
    # because self.grads_and_vars contains also variables.
    # list comprehension might be useful in this case.
    clip_norm = tf.cast(1.0, tf.float32)
    self.grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in self.grads_and_vars]
    
    self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)



BiLSTMModel.__perform_optimization = classmethod(perform_optimization)




def init_model(self, vocabulary_size, n_tags, embedding_dim, n_hidden_rnn, PAD_index):
    self.__declare_placeholders()
    self.__build_layers(vocabulary_size, embedding_dim, n_hidden_rnn, n_tags)
    self.__compute_predictions()
    self.__compute_loss(n_tags, PAD_index)
    self.__perform_optimization()




BiLSTMModel.__init__ = classmethod(init_model)


# ## Train the network and predict tags


def train_on_batch(self, session, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability):
    feed_dict = {self.input_batch: x_batch,
                 self.ground_truth_tags: y_batch,
                 self.learning_rate_ph: learning_rate,
                 self.dropout_ph: dropout_keep_probability,
                 self.lengths: lengths}
    
    session.run(self.train_op, feed_dict=feed_dict)





BiLSTMModel.train_on_batch = classmethod(train_on_batch)


def predict_for_batch(self, session, x_batch, lengths):
    
    predictions = session.run(self.predictions, feed_dict={self.input_batch:x_batch, self.lengths:lengths})
    return predictions

BiLSTMModel.predict_for_batch = classmethod(predict_for_batch)




from evaluation import precision_recall_f1


# In[199]:


def predict_tags(model, session, token_idxs_batch, lengths):
    """Performs predictions and transforms indices to tokens and tags."""
    
    tag_idxs_batch = model.predict_for_batch(session, token_idxs_batch, lengths)
    
    tags_batch, tokens_batch = [], []
    for tag_idxs, token_idxs in zip(tag_idxs_batch, token_idxs_batch):
        tags, tokens = [], []
        for tag_idx, token_idx in zip(tag_idxs, token_idxs):
            tags.append(idx2tag[tag_idx])
            tokens.append(idx2token[token_idx])
        tags_batch.append(tags)
        tokens_batch.append(tokens)
    return tags_batch, tokens_batch
    
    
def eval_conll(model, session, tokens, tags, short_report=True):
    """Computes NER quality measures using CONLL shared task script."""
    
    y_true, y_pred = [], []
    for x_batch, y_batch, lengths in batches_generator(1, tokens, tags):
        tags_batch, tokens_batch = predict_tags(model, session, x_batch, lengths)
        if len(x_batch[0]) != len(tags_batch[0]):
            raise Exception("Incorrect length of prediction for the input, "
                            "expected length: %i, got: %i" % (len(x_batch[0]), len(tags_batch[0])))
        predicted_tags = []
        ground_truth_tags = []
        for gt_tag_idx, pred_tag, token in zip(y_batch[0], tags_batch[0], tokens_batch[0]): 
            if token != '<PAD>':
                ground_truth_tags.append(idx2tag[gt_tag_idx])
                predicted_tags.append(pred_tag)

        # We extend every prediction and ground truth sequence with 'O' tag
        # to indicate a possible end of entity.
        y_true.extend(ground_truth_tags + ['O'])
        y_pred.extend(predicted_tags + ['O'])
        
    results = precision_recall_f1(y_true, y_pred, print_results=True, short_report=short_report)
    return results



from tensorflow.python.framework import ops
ops.reset_default_graph()


#tf.reset_default_graph()

model = BiLSTMModel(20505, 21, 200, 200, token2idx['<PAD>'])

batch_size = 32
n_epochs = 10
learning_rate = 0.02
learning_rate_decay = 1.4
dropout_keep_probability = 0.5


# In[224]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Start training... \n')
for epoch in range(n_epochs):
    # For each epoch evaluate the model on train and validation data
    print('-' * 20 + ' Epoch {} '.format(epoch+1) + 'of {} '.format(n_epochs) + '-' * 20)
    print('Train data evaluation:')
    eval_conll(model, sess, train_tokens, train_tags, short_report=True)
    print('Validation data evaluation:')
    eval_conll(model, sess, validation_tokens, validation_tags, short_report=True)
    
    # Train the model
    for x_batch, y_batch, lengths in batches_generator(batch_size, train_tokens, train_tags):
        model.train_on_batch(sess, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability)
        
    # Decaying the learning rate
    learning_rate = learning_rate / learning_rate_decay
    
print('...training finished.')




print('-' * 20 + ' Train set quality: ' + '-' * 20)
train_results = eval_conll(model, sess, train_tokens, train_tags, short_report=False)

print('-' * 20 + ' Validation set quality: ' + '-' * 20)
validation_results = eval_conll(model, sess, validation_tokens, validation_tags, short_report=False)

print('-' * 20 + ' Test set quality: ' + '-' * 20)
test_results = eval_conll(model, sess, test_tokens, test_tags, short_report=False)


