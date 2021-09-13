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

"""Predicting the QC in the time-series."""

from ae_tdnn import *
import pandas as pd
from absl import app


def run_training():
    """
    This function trains both the AENN and the TDNN.

        Returns:
          decoder: The trained decoder.
          time_delay_nn: The trained TDNN.
          x_ts_test: The test data for the TDNN.
          target_ts_test: The target column of the test data for the TDNN.
          list_of_vals_and_feats: A List of strings of the names of features.
          data_dict: Dictionary of the file names of each data set.
          encoded_columns: List of column names for the encoded data.

    """

    # Encode data
    encoder, decoder, data_dict, list_of_vals_and_feats = training_aenn(training_data_dividing_constant=15)

    # Train the TDNN
    time_delay_nn, x_ts_test, target_ts_test, encoded_columns = training_tdnn(data_dict=data_dict,
                                                                             bs_scaling_constant=4,
                                                                             encoder=encoder)

    return (decoder, time_delay_nn, x_ts_test,
            target_ts_test, list_of_vals_and_feats, data_dict, encoded_columns)


def predict(reconstructions, LCL, UCL):
    """
    This function makes predictions for the time-series.

        Args:
          reconstructions: Reconstructed data at the output of the decoder.
          LCL: Float Lower Control Limit.
          UCL: Float Upper Control Limit.

        Returns:
          prediction: Float prediction of the value.
          predicted_labels: Predicted label.

    """
    predicted_outcome = []
    for i in range(reconstructions.shape[0]):
        temp = reconstructions.iloc[i, :].tolist()  # temp is a list of values in the first measurement
        temp_bool = [0, 0, 0, 0, 0]
        temp_bool = [1 if temp[j] > UCL or temp[j] < LCL else 0 for j in range(5)]
        if any(temp_bool) == True:
            predicted_outcome.append(1)
        else:
            predicted_outcome.append(0)

    print(len(predicted_outcome))
    predicted_outcome = np.array([predicted_outcome])
    predicted_outcome = np.reshape(np.ravel(predicted_outcome), (reconstructions.shape[0], 1))
    predicted_labels = pd.DataFrame(data=predicted_outcome, columns=['Predicted Control Status'], dtype=int)
    prediction = pd.concat([reconstructions, predicted_labels], axis=1)
    prediction.index = range(reconstructions.shape[0])

    return prediction, predicted_labels


def run_predict(decoder,
                time_delay_nn,
                x_ts_test,
                target_ts_test,
                list_of_vals_and_feats,
                encoded_columns,
                data_dict,
                qc):
    """
    This function pre-processes the test data
        and generates predictions of the QCs in the time series.

        Args:
          decoder: The trained decoder.
          time_delay_nn: The trained TDNN.
          x_ts_test: The test data for the TDNN.
          target_ts_test: The target column of the test data for the TDNN.
          list_of_vals_and_feats: A List of strings of the names of features.
          encoded_columns: List of column names for the encoded data.
          data_dict: Dictionary of the file names of each data set.
          qc: String value of the desired quality characteristic.

        Returns:
          target_labels: Real test-data labels.
          predicted_labels: Predicted labels.
          test_target: Target column of the test data.

    """

    # Predict (Time Series): x_ts_test
    predicted_ts_test = time_delay_nn.predict(x_ts_test)
    predicted_ts_test = pd.DataFrame(data=predicted_ts_test, columns=encoded_columns)

    # Decode (AutoEncoder Predict)
    decoded_data = pd.DataFrame(data=decoder.predict(predicted_ts_test),
                                columns=list_of_vals_and_feats)

    list_of_vs = ['V1', 'V2', 'V3', 'V4', 'V5']
    predicted_values = decoded_data[list_of_vs].copy()

    # Rescale
    scaler = MinMaxScaler()
    pred_rescaled_vals = scaler.inverse_transform(predicted_values)
    pred_rescaled_vals = pd.DataFrame(data=pred_rescaled_vals, columns=list_of_vs)

    # Determine Predicted Classification
    [prediction, predicted_labels] = predict(pred_rescaled_vals, 13.0675, 13.1725)

    # Decode actual targets
    decoded_targets = pd.DataFrame(data=decoder.predict(target_ts_test), columns=list_of_vals_and_feats)
    target_values = decoded_targets[list_of_vs].copy()

    # Rescale
    target_rescaled_vals = scaler.inverse_transform(target_values)
    target_rescaled_vals = pd.DataFrame(data=target_rescaled_vals, columns=list_of_vs)

    # Determine Actual Classification:
    [actual, target_labels] = predict(target_rescaled_vals, 13.0675, 13.1725)

    # Access real test data with Real Control Status
    QC_time_series = pd.read_csv('./' + data_dict[qc][1] + '.csv')
    test_target = QC_time_series.pop('Control Status')

    test_target = pd.DataFrame(data=test_target.iloc[(600 - (target_labels.shape[0])):601].values,
                               columns=['Real Control Status'])

    return target_labels, predicted_labels, test_target


def plot_results(target_labels, predicted_labels, test_target):
    """
    This function plots the confusion matrix.

        Args:
          target_labels: Real test-data labels.
          predicted_labels: Predicted labels.
          test_target: Target column of the test data.

    """

    # Compute confusion matrix
    c_mat_ae = confusion_matrix(target_labels, predicted_labels, labels=[0, 1])
    c_mat_real = confusion_matrix(test_target, predicted_labels, labels=[0, 1])

    # Plot normalised AE confusion matrix
    plt.figure()
    plot_confusion_matrix(c_mat_ae, classes=[0, 1], normalize=False,
                          title='Normalized AE Confusion Matrix')
    plt.show()

    # Plot normalised true confusion matrix
    plt.figure()
    plot_confusion_matrix(c_mat_real, classes=[0, 1], normalize=False,
                          title='Normalized Real Confusion Matrix')
    plt.show()


def main():
    (decoder, time_delay_nn, x_ts_test,
     target_ts_test, list_of_vals_and_feats, data_dict, encoded_columns) = run_training()
    target_labels, predicted_labels, test_target = run_predict(decoder,
                                                                time_delay_nn,
                                                                x_ts_test,
                                                                target_ts_test,
                                                                list_of_vals_and_feats,
                                                                encoded_columns,
                                                                data_dict,
                                                                qc='21')
    plot_results(target_labels, predicted_labels, test_target)


if __name__ == '__main__':
    app.run(main)