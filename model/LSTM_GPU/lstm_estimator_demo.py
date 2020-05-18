from model.get_data_path import get_train_data_path
import numpy as np
import tensorflow as tf
import pandas as pd
from util.data_normalized import data_normalized

HIDDEN_SIZE = 500
NUM_LAYERS = 2
TIMESTEPS = 20
TRAINING_STEPS = 10000
BATCH_SIZE = 1000
INPUT_SIZE = 9
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.9
NUM_EXAMPLES = 50000
datafile=get_train_data_path('wfzc','A2','cap_temp_1','model_params_v3')
f=open(datafile)
df=pd.read_csv(f,index_col=0)

MODEL_SAVE_PATH = "model_saved/"
MODEL_NAME = "model.ckpt"
LSTM_KEEP_PROB = 0.9

def lstm_model(x,dropout_keep_prob):

    lstm_cells = [
        tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
            output_keep_prob=dropout_keep_prob)
        for _ in range(NUM_LAYERS)]
    cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
    print(np.shape(x))
    print("cell_created")
    x=tf.reshape(x,[-1,TIMESTEPS,INPUT_SIZE])
    print(np.shape(x))
    outputs,_=tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
    outputs=outputs[:,-1,:]
    predictions01 = tf.contrib.layers.fully_connected(outputs, 100,activation_fn= tf.nn.relu)
    predictions = tf.contrib.layers.fully_connected(predictions01, 1, activation_fn=None)
    return predictions


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev/'+name, stddev)

def get_train_data(time_step):
    features=df.columns.values
    data = df.iloc[:, :].values
    data_temp = data[:,:-1]
    normalized_data_feature=data_normalized(data_temp)
    data_labels = data[:,-1]
    print(np.shape(data_labels))
    train_x=[]
    train_y=[]
    for i in range(len(normalized_data_feature)-time_step):
        train_x.append([normalized_data_feature[i:i + time_step, :]])
        train_y.append([data_labels[i+time_step]])
    print("get_train_data_finished")
    train_x=np.array(train_x).reshape(-1,TIMESTEPS,INPUT_SIZE)
    print(np.shape(train_x[0]))
    train_y=np.array(train_y).squeeze()
    train_y=np.array(train_y).reshape(-1,1)
    print(np.array(train_x, dtype=np.float32).shape)
    print(np.array(train_y, dtype=np.float32).shape)
    return np.array(train_x, dtype=np.float32), np.array(train_y, dtype=np.float32)


def input_fn():
    train_x, train_y=get_train_data(TIMESTEPS)

    ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    # ds = ds.repeat().shuffle(100000).batch(BATCH_SIZE)
    x, y = ds.make_one_shot_iterator().get_next()
    return x,y

def model_fn(x,y,mode,params):
    pred=lstm_model(x,dropout_keep_prob=LSTM_KEEP_PROB)
    if mode==tf.estimator.ModeKeys.PREDICT:
        spec=tf.estimator.EstimatorSpec(mode=mode,predictions=pred)
    else:
        loss_mse=tf.losses.mean_squared_error(labels=y,predictions=pred)
        optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_BASE)
        train_op=optimizer.minimize(loss_mse,global_step=tf.train.get_global_step())
        metrics={'error':tf.metrics.mean_squared_error(labels=y,predictions=pred)}
        spec=tf.estimator.EstimatorSpec(mode=mode,loss=loss_mse,train_op=train_op,eval_metric_ops=metrics)
    return spec



def train(train_x, train_y):
    global_step = tf.Variable(0, trainable=False)
    ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    #ds = ds.repeat().shuffle(100000).batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()
    print(np.shape(X[0]))
    predictions = lstm_model(X,LSTM_KEEP_PROB)
    print(predictions)
    loss = tf.losses.mean_squared_error(labels=tf.reshape(y,[-1]),predictions=tf.reshape(predictions,[-1]))
    tf.summary.scalar('loss',loss)
    print(loss)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        NUM_EXAMPLES / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    print("All paras are setted")
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("log/log", tf.get_default_graph())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            if i % 100 == 0:
                run_options = tf.RunOptions(
                    trace_level = tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _, l, step = sess.run([merged, train_op, loss, global_step],options=run_options,run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata,'step%02d' % i)
                train_writer.add_summary(summary, i)
                print("train step is  %s loss is  %s " % (str(step), str(l)))
                saver.save(sess, "model_saved/model.ckpt")
                print("model has been saved")

            else:
                summary, _, l, step = sess.run([merged, train_op, loss, global_step])
                train_writer.add_summary(summary, i)

    train_writer.close()

def main(argv=None):
    # train_x, train_y = get_train_data(TIMESTEPS, 0, NUM_EXAMPLES)
    # print(np.shape(train_x[0]))
    # train(train_x, train_y)

    #params={'feature_colums':}
    model=tf.estimator.Estimator(model_fn=model_fn,model_dir='./lstm_estmator')
    model.train(input_fn=input_fn,steps=10000)

if __name__ == '__main__':
    tf.app.run()