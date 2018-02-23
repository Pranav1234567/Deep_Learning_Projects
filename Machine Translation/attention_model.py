import tensorflow as tf
import numpy as np
import sys
import time

start_time = time.time()

# Input files
french_train_file = sys.argv[1]
english_train_file = sys.argv[2]
french_dev_file = sys.argv[3]
english_dev_file = sys.argv[4]

# Parameters
rnn_size = 256
embedSz = 64

# Cannot change the below
batchSz = 20
windowSz = 13 #size of our sentences with padding

# Preprocess Train and Development Data, building vocabulary, tokenizing, padding

with open(french_train_file) as f:
	french_train_data = f.read().split()
with open(french_train_file) as f:
	french_train_sentences = f.read().split("\n")

french_train_vocab = set(french_train_data)
# Need to add stop to vocab
french_vocabSz = len(french_train_vocab) + 1

with open(english_train_file) as f:
	english_train_data = f.read().split()
with open(english_train_file) as f:
	english_train_sentences = f.read().split("\n")

english_train_vocab = set(english_train_data)
# Need to add stop to vocab
english_vocabSz = len(english_train_vocab) + 1

with open(french_dev_file) as f:
	french_dev_data = f.read().split()
with open(french_dev_file) as f:
	french_dev_sentences = f.read().split("\n")

with open(english_dev_file) as f:
	english_dev_data = f.read().split()
with open(english_dev_file) as f:
	english_dev_sentences = f.read().split("\n")


# Build the English and French dictionaries
english_dict = {}
english_dict["STOP"] = 0
id = 1

for word in english_train_vocab:
	    if word not in english_dict:
	            english_dict[word] = id
	            id += 1

french_dict = {}
french_dict["STOP"] = 0
id = 1

for word in french_train_vocab:
	    if word not in french_dict:
	            french_dict[word] = id
	            id += 1

# helper function to convert a vector of words to their indices in dict
def words_to_ids(word_lst, dictionary):
    ids = np.zeros(len(word_lst), dtype = np.int)
    for i in range(0, len(word_lst)):
        ids[i] = dictionary[word_lst[i]]
    return ids

# helper function for masking
def find_lengths(f_batch):
	lengths = []
	for i in range(batchSz):
		num_non_zeros = 0
		for j in range(windowSz):
			if f_batch[i][j] != 0:
				num_non_zeros += 1
		lengths.append(num_non_zeros+1)
	return lengths

def find_length_english(e_batch):
	lengths = []
	for i in range(batchSz):
		num_non_zeros = 0
		for j in range(1, windowSz):
			if e_batch[i][j] != 0:
				num_non_zeros += 1
		lengths.append(num_non_zeros)
	return lengths
# helper function: compute accuracy
def compute_accuracy(logits, answers):
	num_words = 0
	sum_acc = 0
	for i in range(batchSz):
		for j in range(windowSz):
			if (np.argmax(logits[i][j]) == answers[i][j]):
				sum_acc += 1
			num_words += 1
			if (answers[i][j] == 0):
				break
	return sum_acc * 1.0 / num_words


# Split corpi into sentences and do padding - train
french_id_sentences = []

for sentence in french_train_sentences:
	sentence_lst = sentence.split()
	while len(sentence_lst) < 13:
		sentence_lst.append("STOP")
	french_id_sentences.append(words_to_ids(sentence_lst, french_dict).tolist())

english_id_sentences = []
english_id_answers = []

for sentence in english_train_sentences:
	sentence_lst = sentence.split()
	sentence_lst.insert(0, "STOP")
	while len(sentence_lst) < 13:
		sentence_lst.append("STOP")

	answer = sentence_lst[1:]
	answer.append("STOP")

	english_id_answers.append(words_to_ids(answer, english_dict).tolist())
	english_id_sentences.append(words_to_ids(sentence_lst, english_dict).tolist())

# Split corpi into sentences and do padding - dev
french_id_test_sentences = []

for sentence in french_dev_sentences:
	sentence_lst = sentence.split()
	while len(sentence_lst) < 13:
		sentence_lst.append("STOP")
	french_id_test_sentences.append(words_to_ids(sentence_lst, french_dict).tolist())

english_id_test_sentences = []
english_id_test_answers = []

for sentence in english_dev_sentences:
	sentence_lst = sentence.split()
	sentence_lst.insert(0, "STOP")
	while len(sentence_lst) < 13:
		sentence_lst.append("STOP")

	answer = sentence_lst[1:]
	answer.append("STOP")

	english_id_test_answers.append(words_to_ids(answer, english_dict).tolist())
	english_id_test_sentences.append(words_to_ids(sentence_lst, english_dict).tolist())

# Collect sentences into batches - train

french_batches = []

num_french_batches = len(french_id_sentences) / batchSz

for i in range(0, num_french_batches):
	f_batch = [french_id_sentences[j*num_french_batches + i] for j in range(batchSz)]
	french_batches.append(f_batch)

english_batches = []
english_batches_labels = []
num_english_batches = len(english_id_sentences) / batchSz

for i in range(0, num_english_batches):
	e_batch_raw = [english_id_sentences[j*num_english_batches + i] for j in range(batchSz)]
	e_batch = np.reshape(np.asarray(e_batch_raw).flatten(), [batchSz, windowSz])
	e_answers_raw = [english_id_answers[j*num_english_batches + i] for j in range(batchSz)]
	e_answers = np.reshape(np.asarray(e_answers_raw).flatten(), [batchSz, windowSz])
	english_batches.append(e_batch)
	english_batches_labels.append(e_answers)

french_batches_dev = []
num_french_batches_dev = len(french_id_test_sentences) / batchSz

for i in range(0, num_french_batches_dev):
	f_batch = [french_id_test_sentences[j*num_french_batches_dev + i] for j in range(batchSz)]
	french_batches_dev.append(f_batch)

english_batches_dev = []
english_batches_labels_dev = []
num_english_batches_dev = len(english_id_test_sentences) / batchSz

for i in range(0, num_english_batches_dev):
	e_batch_raw = [english_id_test_sentences[j*num_english_batches_dev + i] for j in range(batchSz)]
	e_batch = np.reshape(np.asarray(e_batch_raw).flatten(), [batchSz, windowSz])
	e_answers_raw = [english_id_test_answers[j*num_english_batches_dev + i] for j in range(batchSz)]
	e_answers = np.reshape(np.asarray(e_answers_raw).flatten(), [batchSz, windowSz])
	english_batches_dev.append(e_batch)
	english_batches_labels_dev.append(e_answers)

# Placeholders
encode_in = tf.placeholder(tf.int32, shape = [batchSz, windowSz])
decode_in = tf.placeholder(tf.int32, shape = [batchSz, windowSz])
answers = tf.placeholder(tf.int32, shape = [batchSz, windowSz])
lengths = tf.placeholder(tf.int32, shape = [batchSz])

# french sentence lengths
french_lengths = find_lengths(encode_in)
french_lengths[:] = [x - 1 for x in french_lengths]

# english sentence lengths
english_lengths = find_length_english(decode_in)

# scopes
with tf.variable_scope("enc"):
	F = tf.Variable(tf.random_normal([french_vocabSz, embedSz], stddev = 0.1))
	embeddings = tf.nn.embedding_lookup(F, encode_in)
	gru_cell = tf.contrib.rnn.GRUCell(rnn_size)
	initState = gru_cell.zero_state([batchSz], tf.float32)
	encOut, encState = tf.nn.dynamic_rnn(gru_cell, embeddings, initial_state = initState, sequence_length = french_lengths)

attention_weights = tf.Variable(tf.random_normal([windowSz, windowSz], stddev = 0.1))
encOT = tf.transpose(encOut, [0,2,1])
decIT = tf.tensordot(encOT, attention_weights, [[2],[0]])
decI = tf.transpose(decIT, [0,2,1])

with tf.variable_scope("dec"):
	E = tf.Variable(tf.random_normal([english_vocabSz, embedSz], stddev = 0.1))
	embeddings = tf.nn.embedding_lookup(E, decode_in)
	gru_cell = tf.contrib.rnn.GRUCell(rnn_size)
	decOut, _ = tf.nn.dynamic_rnn(gru_cell, tf.concat([embeddings, decI], 2), initial_state = encState, sequence_length = english_lengths)

# Weights and Biases
W = tf.Variable(tf.random_normal([rnn_size, english_vocabSz], stddev = 0.1))
b = tf.Variable(tf.random_normal([english_vocabSz], stddev = 0.1))

# Rest of the Forward pass
L1Output = tf.tensordot(decOut,W,axes=[[2],[0]]) + b

# Seq2Seq Loss
Loss = tf.contrib.seq2seq.sequence_loss(L1Output, answers, tf.sequence_mask(lengths, 13, dtype = tf.float32)) #tf.ones([batchSz, windowSz])

# Training Setup
Learning_Rate = 1E-3
train = tf.train.AdamOptimizer(Learning_Rate).minimize(Loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Encoding and Decoding - train
for i in range(num_english_batches):
 	# f_batch is still a list of lists, good for masking
	f_batch = french_batches[i]
	e_batch = english_batches[i]
	e_answers = english_batches_labels[i]

	f_batch_shaped = np.reshape(np.asarray(f_batch).flatten(), [batchSz, windowSz])
	f_batch_lengths = find_lengths(f_batch)
	feed_dict = {encode_in: f_batch_shaped, decode_in: e_batch, answers: e_answers, lengths: f_batch_lengths}
	sess.run(train, feed_dict)

sum_a = 0

# Encoding and Decoding - dev
for i in range(num_english_batches_dev):
 	# f_batch is still a list of lists, good for masking
	f_batch_dev = french_batches_dev[i]
	e_batch_dev = english_batches_dev[i]
	e_answers_dev = english_batches_labels_dev[i]

	f_batch_shaped_dev = np.reshape(np.asarray(f_batch_dev).flatten(), [batchSz, windowSz])
	f_batch_lengths_dev = find_lengths(f_batch_dev)

	feed_dict = {encode_in: f_batch_shaped_dev, decode_in: e_batch_dev, answers: e_answers_dev, lengths: f_batch_lengths_dev}

	log = sess.run(L1Output, feed_dict)
	sum_a += compute_accuracy(log, e_answers_dev)
print("Time taken: " + str(time.time() - start_time))
print("Test Accuracy: " + str(sum_a / num_french_batches_dev))
