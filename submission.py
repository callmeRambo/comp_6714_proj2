## Submission.py for COMP6714-Project2
###################################################################################################################
import os
import math
import random
import zipfile
import collections
import numpy as np
import tensorflow as tf
import spacy
import gensim
import re
nlp = spacy.load('en')
adj_store = []  #store all adj
data_index = 0
vocabulary_size = 20000
#there build and train the model 
def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    global vocabulary_size
    with open(data_file) as f:
        data_tmp = f.read().split()
        data, count, dictionary, reverse_dictionary, vocabulary_size = build_dataset(data_tmp, vocabulary_size)
    batch_size = 128      # Size of mini-batch for skip-gram model.
    embedding_size = embedding_dim  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right of the target word.
    num_samples = 2         # How many times to reuse an input to generate a label.
    num_sampled = 64      # Sample size for negative examples.
    logs_path = './log/'    
    # Specification of test Sample:
    sample_size = 20       # Random sample of words to evaluate similarity.
    sample_window = 100    # Only pick samples in the head of the distribution.
    sample_examples = np.random.choice(sample_window, sample_size, replace=False) # Randomly pick a sample of size 16

    # Constructing the graph...
    graph = tf.Graph()

    with graph.as_default():
        
        # with tf.device('/gpu:0'):
            # Placeholders to read input data.
            with tf.name_scope('Inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
                
            # Look up embeddings for inputs.
            with tf.name_scope('Embeddings'):            
                sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
                
                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                        stddev=1.0 / math.sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
            
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            with tf.name_scope('Loss'):
                loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_biases, 
                                                labels=train_labels, inputs=embed, 
                                                num_sampled=num_sampled, num_classes=vocabulary_size))
            
            # Construct the Adam optimizer using a learning rate of 0.001.
            with tf.name_scope('Adam_Optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

            # Normalize the embeddings to avoid overfitting.
            with tf.name_scope('Normalization'):
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                normalized_embeddings = embeddings / norm
                
            sample_embeddings = tf.nn.embedding_lookup(normalized_embeddings, sample_dataset)
            similarity = tf.matmul(sample_embeddings, normalized_embeddings, transpose_b=True)
            
            # Add variable initializer.
            init = tf.global_variables_initializer()
            
            
            # Create a summary to monitor cost tensor
            tf.summary.scalar("cost", loss)
            # Merge all summary variables.
            merged_summary_op = tf.summary.merge_all()
    

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        session.run(init)
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
        print('Initializing the model')
        
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_samples, skip_window,data,reverse_dictionary)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            
            # We perform one update step by evaluating the optimizer op using session.run()
            _, loss_val, summary = session.run([optimizer, loss, merged_summary_op], feed_dict=feed_dict)
            
            summary_writer.add_summary(summary, step )
            average_loss += loss_val

            if step % 5000 == 0:
                if step > 0:
                    average_loss /= 5000
                
                    # The average loss is an estimate of the loss over the last 5000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

            # Evaluate similarity after every 10000 iterations.
            if step % 10000 == 0:
                sim = similarity.eval() #
                for i in range(sample_size):
                    sample_word = reverse_dictionary[sample_examples[i]]
                    top_k = 10  # Look for top-10 neighbours for words in sample set.
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % sample_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
                print()

        
                
        final_embeddings = normalized_embeddings.eval()

        with open(embeddings_file_name,'a',encoding='utf-8') as out_put:
            adj_list = []
            embedding_list = []
            
            for i in range(len(count)):
                if count[i][0] in adj_store:
                    #  or doc_word.text == 'chief'
                    adj_list.append(count[i][0])
                    embedding_list.append([str(j)  for j in np.around(final_embeddings[i],6)])

            out_put.write(str(len(adj_list)) + ' ' + str(embedding_dim) + '\n')
            
            for j in range(len(adj_list)):
                tmp_data = adj_list[j] + ' ' + (' ').join(embedding_list[j]) + '\n'
                out_put.write(tmp_data)

# there transfer the data to skip-gram model
def generate_batch(batch_size, num_samples, skip_window,data,reverse_dictionary):
    global data_index   
    
    assert batch_size % num_samples == 0
    assert num_samples <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # span is the width of the sliding window
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span]) # initial buffer content = first sliding window
    
    # print('data_index = {}, buffer = {}'.format(data_index, [reverse_dictionary[w] for w in buffer]))

    data_index += span
    for i in range(batch_size // num_samples):
        context_words = [w for w in range(span) if w != skip_window]
        random.shuffle(context_words)
        words_to_use = collections.deque(context_words) # now we obtain a random list of context words
        for j in range(num_samples): # generate the training pairs
            batch[i * num_samples + j] = buffer[skip_window]
            context_word = words_to_use.pop()
            labels[i * num_samples + j, 0] = buffer[context_word] # buffer[context_word] is a random context word
        
        # slide the window to the next position    
        if data_index == len(data):
            buffer = data[:span]
            data_index = span
        else: 
            buffer.append(data[data_index]) # note that due to the size limit, the left most word is automatically removed from the buffer.
            data_index += 1
        
        # print('data_index = {}, buffer = {}'.format(data_index, [reverse_dictionary[w] for w in buffer]))
        
    # end-of-for
    data_index = (data_index + len(data) - span) % len(data) # move data_index back by `span`
    return batch, labels

################################################---------Build dataset-----------#####################################################  
def build_dataset(words, n_words):
    """Process raw inputs into a dataset. 
       words: a list of words, i.e., the input data
       n_words: Vocab_size to limit the size of the vocabulary. Other words will be mapped to 'UNK'
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common())
    count_tmp = count[n_words+1:]
    count = count[:n_words+1]
    for i in count_tmp:
        if i[0] in adj_store:
            count.append(i)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # i.e., one of the 'UNK' words
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print(len(dictionary))
    return data, count, dictionary, reversed_dictionary,len(count)



################################################---------Process data-----------#####################################################  
def process_data(input_data):
    with zipfile.ZipFile(input_data) as f:
        with open('temp','a',encoding='utf-8') as out:
            for i in f.namelist():
                data = tf.compat.as_str(f.read(i))
                ls = process_data2(data)
                tmp = (' ').join(ls) + ' '
                out.write(tmp)
    return 'temp'

# there data is processed
def process_data2(data):
    global adj_store
    #transfer money
    data = re.sub(r'(\$|£)([0-9]|\.)*(m|b|n|;)*','money',data)
    #transfer percentage
    data = re.sub(r'([0-9]|\.)*\%','percent',data)
    entity_list = ['PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY']
    doc_list = []
    pattern = re.compile(r'(.+\d+)')
    new_doc = nlp(data)
    for token in new_doc:
        if re.match(pattern, str(token)):
            continue
        if not str(token).isalpha():
            continue
        if token.is_punct or token.is_space:
            continue
        if token.pos_ == 'ADJ':
            doc_list.append(token.lemma_.lower().strip())
            adj_store.append(token.text.lower())
            continue
        if token.ent_type_ in entity_list:
            doc_list.append(token.ent_type_)
            continue
        elif token.pos_ == 'VERB':
            doc_list.append(token.lemma_.lower())
            continue
        doc_list.append(token.text.lower())
    return doc_list
# get topk adj there
def Compute_topk(model_file, input_adjective, top_k):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    list = model.most_similar(positive=input_adjective,topn = vocabulary_size-1)
    nlp = spacy.load('en')
    adj_synonyms = []
    for i in range(len(list)):
        word = nlp(list[i][0])
        if word[0].pos_ == 'ADJ':
            adj_synonyms.append(list[i][0])
        if len(adj_synonyms) == top_k:
            break
    return adj_synonyms

