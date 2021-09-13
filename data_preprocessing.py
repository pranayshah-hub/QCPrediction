# coding=utf-8

# Copyright [2019] [Pranay Shah].

# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Python 3

"""Data Preprocessing functions."""

from utils import *
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


def csv_to_df(list_of_csvs):
    """
    Converts CSV files to pandas DataFrames.

        Args:
          lists_of_csvs: A list of the csv filenames.

        Returns:
          list_of_csvs_as_dfs: returns the list as a list of dataframes.

    """

    list_of_csvs_as_dfs = []
    for csv in list_of_csvs:
        dataframe = pd.read_csv('./Original Data/' + csv + '.csv', sep = '|', encoding = 'latin-1')
        list_of_csvs_as_dfs.append(dataframe)

    return list_of_csvs_as_dfs


def oversample(qc_values, column_names):
    """
    Implements oversampling to balance the dataset.

        Args:
          qc_values: Data frame containing the QC values.
          column_names: List of names of columns in the Data frame, renamed for ease.

        Returns:
          oversampled_data: Balanced data frame.

    """
    item_counts = qc_values['Control Status'].value_counts()
    max_val = item_counts.max()
    temp_data_balanced = pd.DataFrame(data = None, columns = column_names)
    control_status = [0.0, 1.0]
    filtered_len = len(qc_values.loc[qc_values['Control Status'] == a])
    for a in control_status:
        if filtered_len == item_counts.max():
            df_majority = qc_values[qc_values['Control Status']==a]
            temp_data_balanced = temp_data_balanced.append(df_majority, ignore_index = True)
        elif filtered_len > 0 and filtered_len != item_counts.max():
            df_upsample = qc_values[qc_values['Control Status']==a]
            df_upsample = resample(df_upsample,replace=True,n_samples=max_val,random_state=1)
            temp_data_balanced = temp_data_balanced.append(df_upsample, ignore_index = True)
        oversampled_data = temp_data_balanced

    return oversampled_data


def pop_target(unpopped_data):
    """
    Removed target column from the data frame.

        Args:
          unpopped_data: Data frame with target column.

        Returns:
          data: Data frame without target column.
          target: Target column.

    """
    data = unpopped_data.copy()
    target = data.pop('CONTROL_STATUS')

    return data, target


def encode(example, unencoded_data, categoric_variables, encode_only=False):
    """
    Implements encoding of categorical variables.

        Returns:
          encoded_data: Data frame with categorical variable values as encodings.

    """
    if encode_only == False:
        encoding = pd.DataFrame(data=None, columns=['LabelEncoder', 'OneHotEncoder'], index=categoric_variables)
        for category in categoric_variables:
            le = preprocessing.LabelEncoder()
            enc = preprocessing.OneHotEncoder()
            encoded_concat = le.fit_transform(unencoded_data[category]).reshape(-1,1)
            encoded_concat = pd.DataFrame(data = enc.fit_transform(encoded_concat).toarray(), columns = le.classes_)
            unencoded_data = pd.concat([unencoded_data, encoded_concat], axis = 1)
            encoding.loc[category, 'LabelEncoder'] = le
            encoding.loc[category, 'OneHotEncoder'] = enc
        encoded_data = unencoded_data.drop(categoric_variables, axis=1)
        example.encoder = encoding
        return encoded_data
    else:
        for category in categoric_variables:
            le = example.encoding.loc[category, 'LabelEncoder']
            enc = example.encoding.loc[category, 'OneHotEncoder']
            encoded_concat = le.transform(unencoded_data[category]).reshape(-1,1)
            encoded_concat = pd.DataFrame(data = enc.transform(encoded_concat).toarray(), columns = le.classes_)
            encoded_concat.index = list(unencoded_data.index)
            unencoded_data = pd.concat([unencoded_data, encoded_concat], axis = 1)
        encoded_data = unencoded_data.drop(categoric_variables, axis=1)

        return encoded_data


def normalise(unnormalised_data, norms, normalise_only=False):
    """
    Normalises the features.

        Args:
          unnormalised_data: Data frame with unnormalised features.
          norms: List of existing values of norms.
          normalise_only: Boolean. Whether or not to return both normalised data and the norm values.

        Returns:
          normalised_data: Data frame with normalised features.

    """
    if normalise_only == False:
        for i in range(len(unnormalised_data.columns)):
            vector = unnormalised_data.iloc[:, i].values.reshape(1, len(unnormalised_data.iloc[:, i]))
            [normalized_vector, n] = preprocessing.normalize(vector, norm="l2", return_norm=True)
            norms.append(n[0])
            unnormalised_data.iloc[:, i] = normalized_vector.reshape(len(unnormalised_data.iloc[:, i]))
        normalised_data = unnormalised_data
        return normalised_data, norms
    else:
        for i in range(len(unnormalised_data.columns)):
            unnormalised_data.iloc[:, i] = unnormalised_data.iloc[:, i] / norms[i]
        normalised_data = unnormalised_data

        return normalised_data


def rescale(dataframe, norms):
    """
    Rescales the feature values using norms.

        Args:
          dataframe: The Data frame to rescale.
          norms: List of norm values.

        Returns:
          dataframe: Data frame with rescaled values.

    """
    for i in range(len(norms)):
        dataframe.iloc[:, i] = dataframe.iloc[:, i] * norms[i]

    return dataframe


def sort_by_datetime(unsorted_df):
    """
    Sorts the Data frame by datetime.

        Args:
          unsorted_df: Unsorted data frame.

        Returns:
          unsorted_dataframe: Unsorted data frame.
          df_sorted: Sorted data frame.

    """
    unsorted_dataframe = unsorted_df.copy()
    unsorted_df['CRDT'] = pd.to_datetime(unsorted_df.CRDT)
    unsorted_df = unsorted_df.sort_values(by='CRDT')
    df_sorted = unsorted_df
    print('Dataframe has been sorted')

    return unsorted_dataframe, df_sorted


def scores_kfold(encoded_data, target):
    """
    Implements K-fold Cross-Validation using Scikit-learn's KFold function.

        Args:
          encoded_data: Data frame with categorical variable values as encodings.
          target: The target column of the data.

    """
    kf = KFold(n_splits=4, shuffle=True)

    accuracy = []
    precision = []
    recall = []
    f_score = []
    confusion = []

    for train_indexes, test_indexes in kf.split(encoded_data, y=target):
        train_dataset = pd.DataFrame(encoded_data.iloc[train_indexes])
        test_dataset = pd.DataFrame(encoded_data.iloc[test_indexes])
        train_target = pd.DataFrame(target.iloc[train_indexes])
        test_target = pd.DataFrame(target.iloc[test_indexes])

        train_target_vector = train_target.values.reshape(-1)
        test_target_vector = test_target.values.reshape(-1)

        example.model.fit(train_dataset, train_target_vector.tolist())
        predicted_y = example.model.predict(test_dataset)
        print(np.unique(train_target_vector, return_counts=True))

        acc = sm.accuracy_score(test_target_vector.tolist(), predicted_y)
        prec = sm.precision_score(test_target_vector.tolist(), predicted_y, average='weighted')
        rec = sm.recall_score(test_target_vector.tolist(), predicted_y, average='weighted')
        f = sm.f1_score(test_target_vector.tolist(), predicted_y, average='weighted')
        conf = confusion_matrix(test_target_vector.tolist(), predicted_y, labels=[1, 2, 3])

        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f_score.append(f)
        confusion.append(conf)


def time_series_dataprep(encoded_data, ti, encoding_size):
    """
    This function prepares data such that each row
    in the data contains the 4 previous time steps,
    with the 5th time step being the target
    encoding_size = 8 (hidden layer size in the AENN)

        Args:
          encoded_data: Data frame with categorical variable values as encodings.
          ti: Integer time interval index.
          encoding_size: Integer size of the latent vector.

        Returns:
          time_series_df: The time-delayed data frame.

    """
    time_int = [3, 4, 5, 6, 7, 8, 9, 10]  # can be extended to more time intervals
    ts_columns = []
    ts_length = []
    counter = 1
    counter_lower = 0
    counter_upper = counter_lower + time_int[ti]  # counter_upper = [4, 5, 6] can be extended as above

    for i in range(encoding_size * (time_int[ti] + 1)):
        key = 'A' + str(i + 1)
        ts_columns.append(key)
        counter = counter + 1

    for i in range(len(time_int)):
        time_series_length = len(encoded_data) - time_int[i]  # ts length = [596, 595, 594], can be extended as above
        ts_length.append(time_series_length)

    ts_shape = (ts_length[ti], encoding_size * (time_int[ti] + 1))
    time_series_df = pd.DataFrame(data=np.zeros(ts_shape).astype(float), columns=ts_columns)

    for i in range(ts_length[ti]):
        idx1 = 0
        if i == 0:
            for j in range(counter_lower, (counter_upper + 1)):
                idx2 = (j + 1) * encoding_size
                if j < counter_upper:
                    # assignment statements
                    time_series_df.iloc[i, idx1:idx2] = encoded_data.iloc[j, :].values
                    idx1 += encoding_size
                else:
                    # assignment statements
                    time_series_df.iloc[i, idx1:idx2] = encoded_data.iloc[j, :].values
                    counter_lower += 1
                    counter_upper += 1
        else:
            for j in range(counter_lower, (counter_upper + 1)):
                idx2 = (j - i + 1) * encoding_size
                if j < counter_upper:
                    # assignment statements
                    time_series_df.iloc[i, idx1:idx2] = encoded_data.iloc[j, :].values
                    idx1 += encoding_size
                else:
                    # assignment statements
                    time_series_df.iloc[i, idx1:idx2] = encoded_data.iloc[j, :].values
                    counter_lower += 1
                    counter_upper += 1

    return time_series_df


def outliers(trained_balanced, LSL, USL):
    """
    Deals with outliers by assigning a lower or upper bound to them.

        Args:
          trained_balanced: The balanced data frame.
          LSL: Lower Specification Limit.
          USL: Upper Specification Limit.

        Returns:
          trained_balanced: The balance data frame with no outliers.

    """
    for i in range(trained_balanced.shape[0]):
        for j in range(5):
            if trained_balanced.iloc[i, j] < LSL:
                trained_balanced.iloc[i, j] = LSL
            if trained_balanced.iloc[i, j] > USL:
                trained_balanced.iloc[i, j] = USL

    return trained_balanced


def preprocessing_aenn(data_dict, qc):
    """
    Implements the data pre-processing pipeline in preparation for AENN training.

        Args:
          data_dict: Dictionary of the file names of each data set.
          qc: String value of the desired quality characteristic.

        Returns:
          x_ae_train: AENN training data
          x_ae_cv: AENN validation data
          list_of_vals_and_feats: List of strings of feature names.

    """

    # QC balanced AE training data:
    qc_trained_balanced = pd.read_csv('./' + data_dict[qc][0] + '.csv')

    # Sort Outliers:
    qc_trained_balanced = outliers(qc_trained_balanced, 13.05, 13.19)

    # AENN Training Data
    train_target = qc_trained_balanced.pop('Control Status')

    # Normalise
    scaler = MinMaxScaler()
    qc_bal_norm = scaler.fit_transform(qc_trained_balanced)
    qc_bal_norm = pd.DataFrame(data=qc_bal_norm, columns=['V1', 'V2', 'V3', 'V4', 'V5'])

    # Feature Engineering
    qc_bal_norm, list_of_vals_and_feats = feature_engineer(qc_bal_norm)

    # Create training data:
    x_ae_train = qc_bal_norm.copy()

    # Create a validation set for Hyperparameter Tuning
    [x_ae_train, x_ae_cv] = train_cv_ae(x_ae_train)

    return x_ae_train, x_ae_cv, list_of_vals_and_feats


def preprocessing_tdnn(data_dict, qc):
    """
    Implements the data pre-processing pipeline in preparation for TDNN training.

        Args:
          data_dict: Dictionary of the file names of each data set.
          qc: String value of the desired quality characteristic.

        Returns:
          qc_bal_norm_ts: Normalised and Balanced training data for the TDNN.

    """

    # qc Time Series data:
    qc_time_series = pd.read_csv('./' + data_dict[qc][1] + '.csv')

    # Sort Outliers:
    qc_time_series = outliers(qc_time_series, 13.05, 13.19)

    # Normalise
    scaler = MinMaxScaler()
    qc_bal_norm_ts = scaler.fit_transform(qc_time_series)
    qc_bal_norm_ts = pd.DataFrame(data=qc_bal_norm_ts, columns=['V1', 'V2', 'V3', 'V4', 'V5'])

    # Feature Engineering
    qc_bal_norm_ts, _ = feature_engineer(qc_bal_norm_ts)

    return qc_bal_norm_ts


def preprocessing_ts(data_dict, encoder, time_int, encoding_size):
    """
    Implements the data pre-processing pipeline in preparation for TDNN training.

        Args:
          data_dict: Dictionary of the file names of each data set.
          encoder: The trained Keras AENN encoder Model.
          time_int: List of Integer values of the time interval.
          encoding_size: Integer value of the latent vector size.

        Returns:
          x_ts_train: Training data for the TDNN.
          x_ts_test: Test data for the TDNN.
          target_ts_train: Target column of the training data for TDNN.
          target_ts_test: Target column of the test data for TDNN.
          encoded_columns: List of column names for the encoded data.

    """

    qc_bal_norm_ts = preprocessing_tdnn(data_dict=data_dict, qc='21')

    encoded_data = encoder.predict(qc_bal_norm_ts)

    encoded_columns = []
    counter = 1

    for i in range(encoded_data.shape[1]):
        key = 'A' + str(counter)
        encoded_columns.append(key)
        counter = counter + 1

    encoded_data = pd.DataFrame(data=encoded_data, columns=encoded_columns)
    time_series_df = time_series_dataprep(encoded_data, 1, encoding_size)

    # Create data and target x_ts, target_ts
    list_of_target_columns = time_series_df.columns[
                             (time_int[1] * encoding_size):((time_int[1] + 1) * encoding_size)].tolist()
    list_of_input_columns = time_series_df.columns[0:(time_int[1] * encoding_size)].tolist()

    target_ts = time_series_df[list_of_target_columns].copy()
    x_ts = time_series_df[list_of_input_columns].copy()

    # Split into train-test
    [x_ts_train, x_ts_test, target_ts_train, target_ts_test] = train_cv_ts(x_ts, target_ts)

    return x_ts_train, x_ts_test, target_ts_train, target_ts_test, encoded_columns

























