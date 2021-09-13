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

"""Cross Validation, Feature Engineering, and Plotting Functions."""

import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Train-CV Split: AE
def train_cv_ae(normalised_data):
    """
    Splits the data for AENN validation.

        Args:
          normalised_data: Data frame with normalised values.

        Returns:
          x_ae_train: Training set.
          x_ae_cv: Validation set.

    """
    x_ae_train, x_ae_cv= train_test_split(normalised_data, test_size = (1/14), shuffle = True)
    return x_ae_train, x_ae_cv


# Train-Test Split: TS
def train_cv_ts(normalised_data, target):
    """
    Splits the data for TDNN testing.

        Args:
          normalised_data: Data frame with normalised values.

        Returns:
          x_ae_train: Training set.
          x_ae_test: Test set.
          target_ts_train: Target column of training data.
          target_ts_test: Target column of test data.

    """
    x_ts_train, x_ts_test, target_ts_train, target_ts_test = train_test_split(normalised_data, target,
                                                                  test_size = (179/596), shuffle = False)
    return x_ts_train, x_ts_test, target_ts_train, target_ts_test


def feature_engineer(balanced_data):
    """
    Implements the features in the list, 'list_of_vals_and_feats'.

        Args:
          balanced_data: Balanced data frame.

        Returns:
          balanced_data: Balanced data frame with additional feature columns.
          list_of_vals_and_feats: A List of strings of the names of features.

    """

    # Feature Engineering
    list_of_vals_and_feats = ['V1', 'V2', 'V3', 'V4', 'V5',
                              'V1V1', 'V1V2', 'V1V3', 'V1V4', 'V1V5',
                              'V2V2', 'V2V3', 'V2V4', 'V2V5',
                              'V3V3', 'V3V4', 'V3V5',
                              'V4V4', 'V4V5',
                              'V5V5']

    for column in balanced_data.columns:
        if column == 'V1':
            balanced_data['V1V1'] = balanced_data.V1*balanced_data.V1
            balanced_data['V1V2'] = balanced_data.V1*balanced_data.V2
            balanced_data['V1V3'] = balanced_data.V1*balanced_data.V3
            balanced_data['V1V4'] = balanced_data.V1*balanced_data.V4
            balanced_data['V1V5'] = balanced_data.V1*balanced_data.V5
        elif column == 'V2':
            balanced_data['V2V2'] = balanced_data.V2*balanced_data.V2
            balanced_data['V2V3'] = balanced_data.V2*balanced_data.V3
            balanced_data['V2V4'] = balanced_data.V2*balanced_data.V4
            balanced_data['V2V5'] = balanced_data.V2*balanced_data.V5
        elif column == 'V3':
            balanced_data['V3V3'] = balanced_data.V3*balanced_data.V3
            balanced_data['V3V4'] = balanced_data.V3*balanced_data.V4
            balanced_data['V3V5'] = balanced_data.V3*balanced_data.V5
        elif column == 'V4':
            balanced_data['V4V4'] = balanced_data.V4*balanced_data.V4
            balanced_data['V4V5'] = balanced_data.V4*balanced_data.V5
        elif column == 'V5':
            balanced_data['V5V5'] = balanced_data.V5*balanced_data.V5
    return balanced_data, list_of_vals_and_feats


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def control_chart(real_time_series, predicted_time_series, LCL, UCL, qc = '', actual = True):
    """

    Plots the control charts of the real and predicted time series for comparison.

        Args:
          real_time_series: Time-delayed data frame with real target values.
          predicted_time_series: Time-delayed data frame with predicted target values.
          LCL: Lower Control Limit.
          UCL: Upper Control Limit.
          qc: String value of the desired quality characteristic.
          actual: Boolean.

    """
    if actual == True:
        real_time_series = real_time_series.iloc[0:(predicted_time_series.shape[0]), 0:5].copy()
        prediction_tsp = predicted_time_series.iloc[:, 0:5].copy()
        real_time_series.index = range(predicted_time_series.shape[0])
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 12
        fig_size[1] = 6
        plt.rcParams["figure.figsize"] = fig_size
        plt.plot(real_time_series, 'ro', prediction_tsp, 'go')
        plt.axhline(y=UCL)
        plt.axhline(y=LCL)
        plt.axhline(y=(((UCL-LCL)/2)+LCL))
        plt.title('Control Chart F B1060, QC{}, Actual vs Predicted'.format(qc))
        plt.xlabel('Time Series (Each sample taken approximately 2-3 hours apart)')
        plt.ylabel('Quality Characteristic {}'.format(qc))
        plt.legend(('Real measurements',
        'Predicted measurements'), loc='lower right')
    if actual == False:
        real_time_series = real_time_series.iloc[(600-(predicted_time_series.shape[0])):600, :].copy()
        prediction_tsp = predicted_time_series.iloc[:, 0:5].copy()
        real_time_series.index = range(predicted_time_series.shape[0])
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 12
        fig_size[1] = 6
        plt.rcParams["figure.figsize"] = fig_size
        plt.plot(real_time_series, 'ro', prediction_tsp, 'go')
        plt.axhline(y=UCL)
        plt.axhline(y=LCL)
        plt.axhline(y=(((UCL-LCL)/2)+LCL))
        plt.title('Control Chart F B1060, QC{}, Real vs Predicted'.format(qc))
        plt.xlabel('Time Series (Each sample taken approximately 2-3 hours apart)')
        plt.ylabel('Quality Characteristic {}'.format(qc))
        plt.legend(('Real measurements', 'Predicted measurements'),
        loc='lower right')
