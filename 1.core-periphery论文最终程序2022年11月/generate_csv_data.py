import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class img_to_data():
    def __init__(self):
        self.normal = []
        self.covid_19 = []
        self.min_value, self.max_value, self.steps = 100, 160, 10

        self.Threshold = [x for x in range(self.min_value, self.max_value, self.steps)]
        print(self.Threshold)
    def convertjpg(self, pngfile, data, label):
        """

        :param pngfile: orginal img
        :param data: a list for saveing the fractral dimension of the img
        :param label:label of the class
        :return:
        """
        img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE) #读取为灰度图
        # print('img:', img)
        #因为opencv读取图片方式和matplot不一样，所以显示结果不同，所以不再显示原图，直接显示灰度图和二值图像
        #img = cv2.imread(pngfile) #读取为灰度图,
        #data.append(img) #添加原图
        #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #data.append(img)
        eac_d = []
        for i in self.Threshold:
            ret, b_img = cv2.threshold(img, i, 255, cv2.THRESH_BINARY) #二值图像，返回 i:是阈值
            d = self.fractal_dimension(b_img, i)
            eac_d.append(d)
        eac_d.append(label)
        print("eac_d:", eac_d)
        data.append(eac_d)
        #new_data = binary_image(img, data)

        return data

    def fractal_dimension(self, Z, threshold):
        """

        :param Z：img
        :param threshold:threshold
        :return: fractral dimension
        """
        # Only for 2d image
        assert(len(Z.shape) == 2) #assert ： 判断语句，如果是二维图像就执行，不是的话，直接报错，不再执行

        # From https://github.com/rougier/numpy-100 (#87)
        def boxcount(Z, k):
            #axis=0, 按列计算， 1：按行计算
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                   np.arange(0, Z.shape[1], k), axis=1)
            # We count non-empty (0) and non-full boxes (k*k)
            return len(np.where((S > 0) & (S < k*k))[0])

        # Transform Z into a binary array
        Z = (Z < threshold)
        # Minimal dimension of image
        p = min(Z.shape)
        # Greatest power of 2 less than or equal to p
        n = 2**np.floor(np.log(p)/np.log(2))
        # Extract the exponent
        n = int(np.log(n)/np.log(2))
        # Build successive box sizes (from 2**n down to 2**1)
        sizes = 2**np.arange(n, 1, -1)
        # Actual box counting with decreasing size
        counts = []
        for size in sizes:
            counts.append(boxcount(Z, size))
        # Fit the successive log(sizes) with log (counts)
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    def get_data(self):
        """
        read the orginal img and convert them to data which format is .csv
        :return:
        """
        #读取图片数据，转化为矩阵
        count0 = 0
        NORMAL_label = 0
        for pngfile in glob.glob("E:/PycharmProjects/1.Python Fundamental Programme/datasets/COVID-19-c/NORMAL/*.png"):
            self.normal = self.convertjpg(pngfile, self.normal, NORMAL_label)
            count0 += 1
            if count0 == 150:
                break
        print(len(self.normal))

        """
        count1 = 0
        for pngfile in glob.glob("E:\PycharmProjects\1.Python Fundamental Programme\datasets\COVID-19-c/Viral Pneumonia/*.png"):
            viral_pneumonia = convertjpg(pngfile, viral_pneumonia)
            count1 += 1
            if count1 == 6:
                break
        print(len(viral_pneumonia))
        """

        count2 = 0
        COVID_label = 1
        for pngfile in glob.glob("E:/PycharmProjects/1.Python Fundamental Programme/datasets/COVID-19-c/COVID-19/*.png"):
            self.covid_19 = self.convertjpg(pngfile, self.covid_19, COVID_label)
            count2 += 1
            if count2 == 150:
                break
        print(len(self.covid_19))

        data = np.vstack((np.array(self.normal), np.array(self.covid_19)))
        print(len(data))

        save = pd.DataFrame(data, columns=['100', '110', '120', '130', '140', '150', 'target'])
        ##保存数据与之范围是100-150，阈值间隔是10
        save.to_csv("C:/Users/Administrator/Desktop/dimension_100_150_10.csv", index=False)

        # end_time = time.time()
        # time = end_time-start_time
        print("time:", time)

# def load_csv_data(
#     data_file_name,
#     *,
#     data_module=DATA_MODULE,
#     descr_file_name=None,
#     descr_module=DESCR_MODULE,
# ):
#     """Loads `data_file_name` from `data_module with `importlib.resources`.
#
#     Parameters
#     ----------
#     data_file_name : str
#         Name of csv file to be loaded from `data_module/data_file_name`.
#         For example `'wine_data.csv'`.
#
#     data_module : str or module, default='sklearn.datasets.data'
#         Module where data lives. The default is `'sklearn.datasets.data'`.
#
#     descr_file_name : str, default=None
#         Name of rst file to be loaded from `descr_module/descr_file_name`.
#         For example `'wine_data.rst'`. See also :func:`load_descr`.
#         If not None, also returns the corresponding description of
#         the dataset.
#
#     descr_module : str or module, default='sklearn.datasets.descr'
#         Module where `descr_file_name` lives. See also :func:`load_descr`.
#         The default is `'sklearn.datasets.descr'`.
#
#     Returns
#     -------
#     data : ndarray of shape (n_samples, n_features)
#         A 2D array with each row representing one sample and each column
#         representing the features of a given sample.
#
#     target : ndarry of shape (n_samples,)
#         A 1D array holding target variables for all the samples in `data`.
#         For example target[0] is the target variable for data[0].
#
#     target_names : ndarry of shape (n_samples,)
#         A 1D array containing the names of the classifications. For example
#         target_names[0] is the name of the target[0] class.
#
#     descr : str, optional
#         Description of the dataset (the content of `descr_file_name`).
#         Only returned if `descr_file_name` is not None.
#     """
#     with resources.open_text(data_module, data_file_name) as csv_file:
#         data_file = csv.reader(csv_file)
#         temp = next(data_file)
#         n_samples = int(temp[0])
#         n_features = int(temp[1])
#         target_names = np.array(temp[2:])
#         data = np.empty((n_samples, n_features))
#         target = np.empty((n_samples,), dtype=int)
#
#         for i, ir in enumerate(data_file):
#             data[i] = np.asarray(ir[:-1], dtype=np.float64)
#             target[i] = np.asarray(ir[-1], dtype=int)
#
#     if descr_file_name is None:
#         return data, target, target_names
#     else:
#         assert descr_module is not None
#         descr = load_descr(descr_module=descr_module, descr_file_name=descr_file_name)
#         return data, target, target_names, descr
#
# def load_iris(*, return_X_y=False, as_frame=False):
#     """Load and return the iris dataset (classification).
#
#     The iris dataset is a classic and very easy multi-class classification
#     dataset.
#
#     =================   ==============
#     Classes                          3
#     Samples per class               50
#     Samples total                  150
#     Dimensionality                   4
#     Features            real, positive
#     =================   ==============
#
#     Read more in the :ref:`User Guide <iris_dataset>`.
#
#     Parameters
#     ----------
#     return_X_y : bool, default=False
#         If True, returns ``(data, target)`` instead of a Bunch object. See
#         below for more information about the `data` and `target` object.
#
#         .. versionadded:: 0.18
#
#     as_frame : bool, default=False
#         If True, the data is a pandas DataFrame including columns with
#         appropriate dtypes (numeric). The target is
#         a pandas DataFrame or Series depending on the number of target columns.
#         If `return_X_y` is True, then (`data`, `target`) will be pandas
#         DataFrames or Series as described below.
#
#         .. versionadded:: 0.23
#
#     Returns
#     -------
#     data : :class:`~sklearn.utils.Bunch`
#         Dictionary-like object, with the following attributes.
#
#         data : {ndarray, dataframe} of shape (150, 4)
#             The data matrix. If `as_frame=True`, `data` will be a pandas
#             DataFrame.
#         target: {ndarray, Series} of shape (150,)
#             The classification target. If `as_frame=True`, `target` will be
#             a pandas Series.
#         feature_names: list
#             The names of the dataset columns.
#         target_names: list
#             The names of target classes.
#         frame: DataFrame of shape (150, 5)
#             Only present when `as_frame=True`. DataFrame with `data` and
#             `target`.
#
#             .. versionadded:: 0.23
#         DESCR: str
#             The full description of the dataset.
#         filename: str
#             The path to the location of the data.
#
#             .. versionadded:: 0.20
#
#     (data, target) : tuple if ``return_X_y`` is True
#
#         .. versionadded:: 0.18
#
#     Notes
#     -----
#         .. versionchanged:: 0.20
#             Fixed two wrong data points according to Fisher's paper.
#             The new version is the same as in R, but not as in the UCI
#             Machine Learning Repository.
#
#     Examples
#     --------
#     Let's say you are interested in the samples 10, 25, and 50, and want to
#     know their class name.
#
#     >>> from sklearn.datasets import load_iris
#     >>> data = load_iris()
#     >>> data.target[[10, 25, 50]]
#     array([0, 0, 1])
#     >>> list(data.target_names)
#     ['setosa', 'versicolor', 'virginica']
#     """
#     data_file_name = "iris.csv"
#     data, target, target_names, fdescr = load_csv_data(
#         data_file_name=data_file_name, descr_file_name="iris.rst"
#     )
#
#     feature_names = [
#         "sepal length (cm)",
#         "sepal width (cm)",
#         "petal length (cm)",
#         "petal width (cm)",
#     ]
#
#     frame = None
#     target_columns = [
#         "target",
#     ]
#     if as_frame:
#         frame, data, target = _convert_data_dataframe(
#             "load_iris", data, target, feature_names, target_columns
#         )
#
#     if return_X_y:
#         return data, target
#
#     return Bunch(
#         data=data,
#         target=target,
#         frame=frame,
#         target_names=target_names,
#         DESCR=fdescr,
#         feature_names=feature_names,
#         filename=data_file_name,
#         data_module=DATA_MODULE,
#     )

if __name__ == "__main__":
    img = img_to_data()
    img.get_data()
