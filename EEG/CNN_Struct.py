import tensorflow as tf

def cnn_model(features, labels, mode):
    inputLayer = tf.reshape(features["x"], [-1, 250, 1])

    conv1 = tf.layers.conv1d(
        inputs = inputLayer,
        filters = 10,
        kernel_size = [10],
        padding = "same",
        activation = tf.nn.relu
    )

    pool1 = tf.layers.max_pooling1d(
        inputs = conv1,
        pool_size=[2],
        strides = 2
    )

    conv2 = tf.layers.conv1d(
        inputs = pool1,
        filters = 10,
        kernel_size = [10],
        padding = "same",
        activation = tf.nn.relu
    )

    pool2 = tf.layers.max_pooling1d(
        inputs = conv2,
        pool_size = [5],
        strides = 5
    )

    pool2Flat = tf.reshape(pool2, [-1, 25 * 10])

    dense = tf.layers.dense(
        inputs = pool2Flat,
        units = 200,
        activation = tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs = dense,
        rate = 0.4,
        training = mode == tf.estimator.ModeKeys.TRAIN
    )
    logits = tf.layers.dense(
        inputs = dropout,
        units = 6
    )

    predictions = {
        "classes": tf.argmax(input = logits, axis = 1),
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    eval_metrics_ops = {
        "accuracy": tf.metrics.accuracy(labels = labels, predictions = predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode = mode,
        loss = loss,
        eval_metric_ops = eval_metric_ops
    )
