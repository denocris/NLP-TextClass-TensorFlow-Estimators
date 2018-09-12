
#from preprocessing import process_text
import preprocessing
import params as pm
import tensorflow as tf

def lstm_model_fn(features, labels, mode, params):

    hidden_units = params.hidden_units
    output_layer_size = len(pm.TARGET_LABELS)
    embedding_size = params.embedding_size
    embedding_initializer = params.embedding_initializer
    forget_bias = params.forget_bias
    keep_prob = params.keep_prob
    learning_rate = params.learning_rate

    # word_id_vector
    word_id_vector = preprocessing.process_text(features[pm.TEXT_FEATURE_NAME])
    # print(' - - - - - - -  - - - - - - -  - - - - -', word_id_vector[0])
    # feature_length_array = [len(np.nonzero(word_id_vector[i])[0]) for i in range(BATCH_SIZE)]
    feature_length_array = tf.count_nonzero(word_id_vector, 1)

    # layer to take each word_id and convert it into vector (embeddings)
    word_embeddings = tf.contrib.layers.embed_sequence(word_id_vector,
                                                 vocab_size=pm.N_WORDS,
                                                 embed_dim=embedding_size,
                                                 initializer=embedding_initializer,
                                                 trainable=False)


    # configure the RNN
    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100)

    # configure the RNN
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(
            num_units=size,
            forget_bias=forget_bias,
            activation=tf.nn.tanh) for size in hidden_units]

        # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    input_layer = word_embeddings
    #input_layer = tf.unstack(word_embeddings, axis=1)


    _, final_states = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=input_layer,
                                   sequence_length=feature_length_array,
                                   dtype=tf.float32)


    # slice to keep only the last cell of the RNN
    rnn_output = final_states[-1].h

    # Connect the output layer (logits) to the hidden layer (no activation fn)
    logits = tf.layers.dense(inputs=rnn_output,
                             units=output_layer_size,
                             activation=None)
    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        #print("-------------------- Predicting ---------------------")
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Convert predicted_indices back into strings
        predictions = {
            'class': tf.gather(pm.TARGET_LABELS, predicted_indices),
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
        optimizer = tf.train.AdamOptimizer(learning_rate)

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
            depth=len(pm.TARGET_LABELS),
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


def cnn_model_fn(features, labels, mode, params):

    hidden_units = params.hidden_units
    output_layer_size = len(pm.TARGET_LABELS)
    embedding_size = params.embedding_size
    embedding_initializer = params.embedding_initializer
    window_size = params.window_size
    stride = int(window_size/2)
    filters = params.filters
    #forget_bias = params.forget_bias
    #keep_prob = params.keep_prob

    print('xxxxxxxxxxxxxxxxxxxxxxx ------------------------- vvvvvvvvvvvvvvvvvvvvv', window_size)

    # word_id_vector
    word_id_vector = preprocessing.process_text(features[pm.TEXT_FEATURE_NAME])
    # print(' - - - - - - -  - - - - - - -  - - - - -', word_id_vector[0])
    # feature_length_array = [len(np.nonzero(word_id_vector[i])[0]) for i in range(BATCH_SIZE)]
    feature_length_array = tf.count_nonzero(word_id_vector, 1)

    # layer to take each word_id and convert it into vector (embeddings)
    word_embeddings = tf.contrib.layers.embed_sequence(word_id_vector,
                                                 vocab_size=pm.N_WORDS,
                                                 embed_dim=embedding_size,
                                                 initializer=embedding_initializer,
                                                 trainable=False)


    training = (mode == tf.estimator.ModeKeys.TRAIN)
    dropout_emb = tf.layers.dropout(inputs=word_embeddings, rate=0.2, training=training)
    # convolution: a sentence can be seen like an image with dimansion length x 1 (that's why conv1d)
    words_conv_1 = tf.layers.conv1d(dropout_emb, filters=filters, kernel_size=4,
                                  strides=stride, padding='SAME', activation=tf.nn.relu)

    max_conv_1 = tf.reduce_max(input_tensor=words_conv_1, axis=1)

    words_conv_2 = tf.layers.conv1d(dropout_emb, filters=filters, kernel_size=3,
                                      strides=stride, padding='SAME', activation=tf.nn.relu)

    max_conv_2 = tf.reduce_max(input_tensor=words_conv_2, axis=1)

    words_conv_3 = tf.layers.conv1d(dropout_emb, filters=filters, kernel_size=2,
                                          strides=stride, padding='SAME', activation=tf.nn.relu)

    max_conv_3 = tf.reduce_max(input_tensor=words_conv_3, axis=1)

    print('----------------------------- 1', words_conv_1.get_shape())
    print('----------------------------- max', max_conv_1.get_shape())
    print('----------------------------- 1', words_conv_2.get_shape())
    print('----------------------------- 1', words_conv_3.get_shape())
    # apply pooling
    #words_conv = tf.reduce_max(input_tensor=words_conv, axis=1)
    #print('----------------------------- 2', words_conv.get_shape())

    words_conv = tf.concat([max_conv_1, max_conv_2,max_conv_3], 1)
    print('----------------------------- 2', words_conv.get_shape())

    # words_conv_shape = words_conv.get_shape()
    # dim = words_conv_shape[1] * words_conv_shape[2]
    # input_layer = tf.reshape(words_conv,[-1, dim])
    input_layer = words_conv

    print('----------------------------- 3', input_layer.get_shape())
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
            'class': tf.gather(pm.TARGET_LABELS, predicted_indices),
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
            depth=len(pm.TARGET_LABELS),
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
