import tensorflow as tf
from tensorflow import data
import datetime
import multiprocessing
import shutil
import numpy as np
import datetime
import os, sys
import csv
from gensim.models import Word2Vec

# Usage
# $: python3 lstm-estimator-dyn-glove.py 'model-name'

tf.logging.set_verbosity(tf.logging.INFO)

# ---------------------------------------------------
# ------------- PIPELINE INPUT FUNCTION -------------
# ---------------------------------------------------

num_words_in_sentence = lambda x: str(x).count(' ') + 1

def parse_tsv_row(tsv_row):
    columns = tf.decode_csv(tsv_row, record_defaults=HEADER_DEFAULTS, field_delim='\t')
    features = {HEADER[0]: columns[0], HEADER[1]: columns[1]}
    #features['length'] = tf.cast(num_words_in_sentence(columns[0]),tf.int64)
    target = features.pop(TARGET_NAME)
    # Uncomment if dataset not already balanced
    #features[WEIGHT_COLUNM_NAME] =  tf.cond( tf.equal(target,'spam'), lambda: 6.6, lambda: 1.0 )
    return features, target


def parse_label_column(label_string_tensor):
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(TARGET_LABELS))
    return table.lookup(label_string_tensor)

def input_fn(files_name_pattern, mode=tf.estimator.ModeKeys.EVAL,
                 skip_header_lines=0,
                 num_epochs=1,
                 batch_size=200):

    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    num_threads = multiprocessing.cpu_count() if MULTI_THREADING else 1

    # representing the number of elements from this dataset from which the new dataset will sample.
    buffer_size = 2 * batch_size + 1

    file_names = tf.matching_files(files_name_pattern)
    dataset = data.TextLineDataset(filenames=file_names)
    dataset = dataset.skip(skip_header_lines)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.map(lambda tsv_row: parse_tsv_row(tsv_row), num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size)
    iterator = dataset.make_one_shot_iterator()
    features, target = iterator.get_next()
    return features, parse_label_column(target)

def process_text(text_feature):

    # Load vocabolary lookup table to map word => word_id
    vocab_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=VOCAB_LIST_FILE,
                                                          num_oov_buckets=1, default_value=-1)
    # Get text feature
    smss = text_feature
    # Split text to words -> this will produce sparse tensor with variable-lengthes (word count) entries
    words = tf.string_split(smss)
    # Convert sparse tensor to dense tensor by padding each entry to match the longest in the batch
    dense_words = tf.sparse_tensor_to_dense(words, default_value=PAD_WORD)
    # Convert word to word_ids via the vocab lookup table
    word_ids = vocab_table.lookup(dense_words)
    # Create a word_ids padding
    padding = tf.constant([[0,0],[0,MAX_DOCUMENT_LENGTH]])
    # Pad all the word_ids entries to the maximum document length
    word_ids_padded = tf.pad(word_ids, padding)
    word_id_vector = tf.slice(word_ids_padded, [0,0], [-1, MAX_DOCUMENT_LENGTH])

    # Return the final word_id_vector
    return word_id_vector


def get_word_index_dict():
    words_index = {}
    with open(VOCAB_LIST_FILE) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        cnt = 1
        next(tsvfile)
        for row in reader:
            words_index[row[0]] = cnt
            cnt += 1
    return words_index


def load_glove_embeddings(word_index, model):
    embeddings = {}
    for word in model.wv.vocab:
        vectors = model.wv[str(word)]
        #print(vectors)
        embeddings[word] = vectors
    embedding_matrix = np.random.uniform(-1, 1, size=(N_WORDS, 300))
    num_loaded = 0
    for w, i in word_index.items():
        #print('-------- w, i', w, i, type(w))
        vv = embeddings.get(str(w))
        #print('---------  vvv  vvv   ----------', vv)
        if vv is not None and i < N_WORDS:
            embedding_matrix[i] = vv
            num_loaded += 1
    print('Successfully loaded pretrained embeddings for '
          f'{num_loaded}/{N_WORDS} words.')
    embedding_matrix = embedding_matrix.astype(np.float32)
    return embedding_matrix


def model_fn(features, labels, mode, params):

    hidden_units = params.hidden_units
    output_layer_size = len(TARGET_LABELS)
    embedding_size = params.embedding_size
    embedding_initializer = params.embedding_initializer
    window_size = params.window_size
    stride = int(window_size/2)
    filters = params.filters
    #forget_bias = params.forget_bias
    #keep_prob = params.keep_prob

    # word_id_vector
    word_id_vector = process_text(features[TEXT_FEATURE_NAME])
    # print(' - - - - - - -  - - - - - - -  - - - - -', word_id_vector[0])
    # feature_length_array = [len(np.nonzero(word_id_vector[i])[0]) for i in range(BATCH_SIZE)]
    feature_length_array = tf.count_nonzero(word_id_vector, 1)

    # layer to take each word_id and convert it into vector (embeddings)
    word_embeddings = tf.contrib.layers.embed_sequence(word_id_vector,
                                                 vocab_size=N_WORDS,
                                                 embed_dim=embedding_size,
                                                 initializer=embedding_initializer,
                                                 trainable=False)


    training = (mode == tf.estimator.ModeKeys.TRAIN)
    dropout_emb = tf.layers.dropout(inputs=word_embeddings, rate=0.2, training=training)
    # convolution: a sentence can be seen like an image with dimansion length x 1 (that's why conv1d)
    words_conv= tf.layers.conv1d(dropout_emb, filters=filters, kernel_size=window_size,
                                  strides=stride, padding='SAME', activation=tf.nn.relu)
    #print('----------------------------- 1', words_conv.get_shape())

    # apply pooling
    words_conv = tf.reduce_max(input_tensor=words_conv, axis=1)
    print('----------------------------- 2', words_conv.get_shape())

    # words_conv_shape = words_conv.get_shape()
    # dim = words_conv_shape[1] * words_conv_shape[2]
    # input_layer = tf.reshape(words_conv,[-1, dim])
    input_layer = words_conv

    #print('----------------------------- 3', input_layer.get_shape())
    if hidden_units is not None:

        # Create a fully-connected layer-stack based on the hidden_units in the params
        hidden_layers = tf.contrib.layers.stack(inputs=input_layer,
                                                layer=tf.contrib.layers.fully_connected,
                                                stack_args= hidden_units,
                                                activation_fn=tf.nn.relu)

        hidden_layers = tf.layers.dropout(inputs=hidden_layers, rate=0.2, training=training)
        # print("hidden_layers: {}".format(hidden_layers)) # (?, last-hidden-layer-size)

    else:
        hidden_layers = input_layer

    # Connect the output layer (logits) to the hidden layer (no activation fn)
    logits = tf.layers.dense(inputs=hidden_layers,
                             units=output_layer_size,
                             activation=None)
    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        #print("-------------------- Predicting ---------------------")
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Convert predicted_indices back into strings
        predictions = {
            'class': tf.gather(TARGET_LABELS, predicted_indices),
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }

        # Provide an estimator spec for `ModeKeys.PREDICT` modes.
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)

    # weights
    #weights = features[WEIGHT_COLUNM_NAME]

    # Calculate loss using softmax cross entropy
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

    #loss = tf.losses.sparse_softmax_cross_entropy(
    #    logits=logits, labels=labels,
    #    weights=weights)

    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Create Optimiser
        optimizer = tf.train.AdamOptimizer(params.learning_rate)

        # Create training operation
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        # Provide an estimator spec for `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Return accuracy and area under ROC curve metrics
        labels_one_hot = tf.one_hot(
            labels,
            depth=len(TARGET_LABELS),
            on_value=True,
            off_value=False,
            dtype=tf.bool
        )

        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predicted_indices) ,#, weights=weights),
            'auroc': tf.metrics.auc(labels_one_hot, probabilities) # , weights=weights)
        }

        # Provide an estimator spec for `ModeKeys.EVAL` modes.
        return tf.estimator.EstimatorSpec(mode, loss=loss,eval_metric_ops=eval_metric_ops)


def create_estimator(run_config, hparams):
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                  params=hparams,
                                  config=run_config)

    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")
    return estimator



def serving_input_fn():
    # At serving time, it accepts inference requests and prepares them for the model.
    receiver_tensor = {
      'sentence': tf.placeholder(tf.string, [None]),
      #'length': tf.placeholder(tf.int64, [None]),
    }

    features = {
      key: tensor
      for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(
        features, receiver_tensor)


def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
    assert dtype is tf.float32
    return embedding_matrix


if __name__ == "__main__":

    MODEL_NAME = str(sys.argv[1])
    model_dir = '/home/asr/Data/classif_task/trained_models/{}'.format(MODEL_NAME)

    TRAIN_DATA_FILES_PATTERN = '/home/asr/Data/classif_task/dev_data/train-data-maxlength16.tsv'
    VALID_DATA_FILES_PATTERN = '/home/asr/Data/classif_task/dev_data/valid-data-maxlength16.tsv'

    VOCAB_LIST_FILE = '/home/asr/Data/classif_task/dev_data/vocab_list_13k_2k_mystop_nodgts.tsv'
    N_WORDS_FILE = '/home/asr/Data/classif_task/dev_data/n_words_13k_2k_mystop_nodgts.tsv'
    DEV_DATA_PATH="/home/asr/Data/classif_task/dev_data/"

    RESUME_TRAINING = False
    MULTI_THREADING = True

    # ------------------------------------
    # ------------- TRAINING & EVAL SETUP -------------
    # ------------------------------------
    TRAIN_SIZE = 317890 #159598 #1429485
    NUM_EPOCHS = 5
    BATCH_SIZE = 128
    EVAL_AFTER_SEC = 120
    TOTAL_STEPS = int((TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS)

    # ------------------------------------
    # ------------- METADATA -------------
    # ------------------------------------
    MAX_DOCUMENT_LENGTH = 10
    PAD_WORD = '#=KS=#'
    HEADER = ['sentence', 'class']
    HEADER_DEFAULTS = [['NA'], ['NA']]
    TEXT_FEATURE_NAME = 'sentence'
    TARGET_NAME = 'class'
    TARGET_LABELS = ['0', '1']
    #WEIGHT_COLUNM_NAME = 'weight'

    with open(N_WORDS_FILE) as file:
        N_WORDS = int(file.read())+2

    word_index = get_word_index_dict()
    # link: http://hlt.isti.cnr.it/wordembeddings/
    model = Word2Vec.load('/home/asr/Data/classif_task/glove-emb/italian-glove/glove_WIKI')

    #glove_embedding_matrix = load_glove_embeddings(word_index, model)

    def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
        assert dtype is tf.float32
        embedding_matrix = load_glove_embeddings(word_index, model)
        return embedding_matrix

    hparams  = tf.contrib.training.HParams(
        num_epochs = NUM_EPOCHS,
        batch_size = BATCH_SIZE,
        embedding_size = 300,
        hidden_units= [64, 32],  #None
        window_size = 3,
        filters = 32,
        max_steps = TOTAL_STEPS,
        learning_rate = 0.01,
        embedding_initializer = my_initializer)

    run_config = tf.estimator.RunConfig(
        log_step_count_steps=5000,
        tf_random_seed=19830610,
        model_dir=model_dir)

    train_spec = tf.estimator.TrainSpec(
        input_fn = lambda: input_fn(
            TRAIN_DATA_FILES_PATTERN,
            mode = tf.estimator.ModeKeys.TRAIN,
            num_epochs=hparams.num_epochs,
            batch_size=hparams.batch_size),
        max_steps=hparams.max_steps,
        hooks=None)

    eval_spec = tf.estimator.EvalSpec(
        input_fn = lambda: input_fn(
            VALID_DATA_FILES_PATTERN,
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=hparams.batch_size),
        exporters=[tf.estimator.LatestExporter(  # this class regularly exports the serving graph and checkpoints.
            name="predict", # the name of the folder in which the model will be exported to under export
            serving_input_receiver_fn=serving_input_fn,
            exports_to_keep=1,
            as_text=True)],
        steps=None,
        throttle_secs = EVAL_AFTER_SEC)



    if not RESUME_TRAINING:
        print("Removing previous artifacts...")
        shutil.rmtree(model_dir, ignore_errors=True)
    else:
        print("Resuming training...")

    time_start = datetime.datetime.utcnow()
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................")

    estimator = create_estimator(run_config, hparams)

    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec)

    time_end = datetime.datetime.utcnow()
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
