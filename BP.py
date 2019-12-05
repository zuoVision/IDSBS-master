import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split



class BP():
    def __init__(self):
        super(BP, self).__init__()

    def add_layer(self,x, in_size, out_size, activation=None):
        with tf.name_scope('W') as scope:
            W = tf.Variable(tf.truncated_normal([in_size, out_size]))
        with tf.name_scope('b') as scope:
            b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('A') as scope:
            A = tf.matmul(x, W) + b

        if activation == 0:
            output = tf.nn.sigmoid(A)
        elif activation == 1:
            output = tf.nn.relu(A)
        elif activation == 2:
            output = tf.nn.tanh(A)
        else:
            output = A
        return output

    def normalization(self,x):
        for i in range(x.shape[0]):
            max_value = max(x[i])
            x[i] /= max_value
        return x





    def data_split(self,data,test_size=0.1,isScale=True):
        m = len(data[:,0])
        n_test = int(m*test_size)
        x_train = data[:-n_test,:-1]
        x_test = data[-n_test:,:-1]
        y_train = data[:-n_test,-1]
        y_test = data[-n_test:,-1]
        if isScale:
            x_train = scale(x_train)
            x_test = scale(x_test)
            y_train = scale(y_train.reshape(-1,1))
            y_test = scale(y_test.reshape(-1,1))
        return (x_train,x_test,y_train,y_test)



    def scale(X, axis=0, with_mean=True, with_std=True, copy=True):
        """Standardize a dataset along any axis

        Center to the mean and component wise scale to unit variance.

        Read more in the :ref:`User Guide <preprocessing_scaler>`.

        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data to center and scale.

        axis : int (0 by default)
            axis used to compute the means and standard deviations along. If 0,
            independently standardize each feature, otherwise (if 1) standardize
            each sample.

        with_mean : boolean, True by default
            If True, center the data before scaling.

        with_std : boolean, True by default
            If True, scale the data to unit variance (or equivalently,
            unit standard deviation).

        copy : boolean, optional, default True
            set to False to perform inplace row normalization and avoid a
            copy (if the input is already a numpy array or a scipy.sparse
            CSC matrix and if axis is 1).

        Notes
        -----
        This implementation will refuse to center scipy.sparse matrices
        since it would make them non-sparse and would potentially crash the
        program with memory exhaustion problems.

        Instead the caller is expected to either set explicitly
        `with_mean=False` (in that case, only variance scaling will be
        performed on the features of the CSC matrix) or to call `X.toarray()`
        if he/she expects the materialized dense array to fit in memory.

        To avoid memory copy the caller should pass a CSC matrix.

        NaNs are treated as missing values: disregarded to compute the statistics,
        and maintained during the data transformation.

        We use a biased estimator for the standard deviation, equivalent to
        `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
        affect model performance.

        For a comparison of the different scalers, transformers, and normalizers,
        see :ref:`examples/preprocessing/plot_all_scaling.py
        <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

        See also
        --------
        StandardScaler: Performs scaling to unit variance using the``Transformer`` API
            (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).

        """  # noqa
        X = check_array(X, accept_sparse='csc', copy=copy, ensure_2d=False,
                        warn_on_dtype=True, estimator='the scale function',
                        dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
        if sparse.issparse(X):
            if with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` instead"
                    " See docstring for motivation and alternatives.")
            if axis != 0:
                raise ValueError("Can only scale sparse matrix on axis=0, "
                                 " got axis=%d" % axis)
            if with_std:
                _, var = mean_variance_axis(X, axis=0)
                var = _handle_zeros_in_scale(var, copy=False)
                inplace_column_scale(X, 1 / np.sqrt(var))
        else:
            X = np.asarray(X)
            if with_mean:
                mean_ = np.nanmean(X, axis)
            if with_std:
                scale_ = np.nanstd(X, axis)
            # Xr is a view on the original array that enables easy use of
            # broadcasting on the axis in which we are interested in
            Xr = np.rollaxis(X, axis)
            if with_mean:
                Xr -= mean_
                mean_1 = np.nanmean(Xr, axis=0)
                # Verify that mean_1 is 'close to zero'. If X contains very
                # large values, mean_1 can also be very large, due to a lack of
                # precision of mean_. In this case, a pre-scaling of the
                # concerned feature is efficient, for instance by its mean or
                # maximum.
                if not np.allclose(mean_1, 0):
                    warnings.warn("Numerical issues were encountered "
                                  "when centering the data "
                                  "and might not be solved. Dataset may "
                                  "contain too large values. You may need "
                                  "to prescale your features.")
                    Xr -= mean_1
            if with_std:
                scale_ = _handle_zeros_in_scale(scale_, copy=False)
                Xr /= scale_
                if with_mean:
                    mean_2 = np.nanmean(Xr, axis=0)
                    # If mean_2 is not 'close to zero', it comes from the fact that
                    # scale_ is very small so that mean_2 = mean_1/scale_ > 0, even
                    # if mean_1 was close to zero. The problem is thus essentially
                    # due to the lack of precision of mean_. A solution is then to
                    # subtract the mean again:
                    if not np.allclose(mean_2, 0):
                        warnings.warn("Numerical issues were encountered "
                                      "when scaling the data "
                                      "and might not be solved. The standard "
                                      "deviation of the data is probably "
                                      "very close to 0. ")
                        Xr -= mean_2
        return X

    def run(self,net,lr=0.01,epoch=1000):
        # 数据


        boston = load_boston()
        x_data,x_test,y_data,y_test = train_test_split(boston.data,boston.target,test_size=0.1,random_state=0)
        x_data = scale(x_data)
        x_test = scale(x_test)
        y_data = scale(y_data.reshape((-1,1)))
        y_test = scale(y_test.reshape((-1,1)))
        print("x_data", x_data)
        print("x_test", x_test)
        print("y_data", y_data)
        print("y_test", y_test)
        # print(x_data.shape,x_test.shape)

        # 网络参数
        # net = [13, 10, 1]
        # epoch = 10000
        # lr = 0.01


        # x_data = normalization(x_data)

        with tf.name_scope('input') as scope:
            x = tf.placeholder(tf.float32, [None, net[0]], name='x_input')
            y = tf.placeholder(tf.float32, [None, net[-1]], name='y_input')


        with tf.name_scope('hide_layer') as scope:
            h1 = self.add_layer(x, net[0],net[1], activation=tf.nn.relu)
            # h2 = add_layer(h1,net[1],net[2],activation=tf.nn.relu)

        with tf.name_scope('output') as scope:
            output = self.add_layer(h1, net[-2], net[-1])

        with tf.name_scope('loss') as scope:
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - output), reduction_indices=[1]))
            tf.summary.scalar('loss', loss)

        with tf.name_scope('train') as scope:
            # train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
            train_step = tf.train.AdamOptimizer(lr).minimize(loss)

        merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            # writer = tf.train.SummaryWriter('logs/',sess.graph)
            # writer = tf.summary.FileWriter('logs/', sess.graph)
            sess.run(init)

            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)
            # ax.scatter(x_data,y_data)
            # plt.ion() #开启互动模式

            loss_ = []
            for i in range(epoch):
                loss_value,_ = sess.run([loss,train_step], feed_dict={x: x_data, y: y_data})
                if i % 100 == 0:
                    loss_.append(loss_value)
                    status = 'Epoch: [%.4d/%d], Loss:%g' % (i, epoch, loss_value)
                    print(status)

                    # plt.pause(0.1)
                    # try:
                    #     ax.lines.remove(lines[0]) #插除之前的轨迹
                    # except Exception:
                    #     pass

            prediction_value = sess.run(output, feed_dict={x: x_test})
            print(prediction_value)
            # plt.figure()
            # plt.plot(loss_[4:],'r')
            # plt.title('Epoch:%d, Learning Rate:%g'%(epoch,lr))
            plt.figure()
            l1, = plt.plot(y_test[:50], 'r--', marker='o')
            # l2, = plt.plot(y_data[:50], 'g--', marker='*')
            l3, = plt.plot(prediction_value[:50], 'b', marker='o')
            # l4, = plt.plot(prediction_value[:50], 'g', marker='*')
            plt.title('Epoch:%d, Learning Rate:%g, Loss:%g' % (epoch, lr,loss_[-1]))
            plt.legend([l1,l3], ['house_price','prediction_price'], loc='best')
            plt.savefig('house price.jpg')
            plt.show()
        # return prediction_value,loss_value




if __name__ == "__main__":
    bp = BP()
    net = [13, 10, 1]
    epoch = 5000
    lr = 0.01
    bp.run(net,epoch=epoch)