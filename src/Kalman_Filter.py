import pandas as pd
import numpy as np
import matplotlib as mlp
from matplotlib import pyplot as plt

mlp.use('TkAgg')

if __name__ == "__main__":

    """
    load in the data from the data set,
    this data comes from a machine learning dataset
    used to determine if someone has entered the room
    and therefore the data has spikes when someone does
    indeed enter the room.
    """
    temperature_dataframe = pd.read_csv("..\\data\\datatraining.txt", sep=',')

    """
    make initial estimates for the kalman filter.
    due to the length of the dataset, the initial 
    guess will be set to 0 so that the filter will
    have time to get a fix on the 'true' value.
    """

    subplot_index = 221             #number for the index of the subplot

    variance_process = [0.0001, 0.00001, 0.000001, 0.0000001]
                                        # q, system noise (constant), how close to follow the data

    for sys_err in variance_process:

        estimate_state_prev = temperature_dataframe.iloc[0]['Temperature']
        # x_t-1
        variance_state_prev = 1000      # p_t-1

        variance_measurement = 0.01     # r, measurement noise (constant)

        kalman_values = np.zeros(shape=len(temperature_dataframe.index), dtype=np.double)
                                        # array holding the 'true' values

        kalman_gain = 0                 #how much to change the data to obtain a true value

        for index, row in temperature_dataframe.iterrows():
            kalman_values[index - 1] = estimate_state_prev
            """
            the prediction stage make predictions about
            where the state should be
            """

            estimate_state = estimate_state_prev
                                        #x_t | x_t-1
            variance_state = variance_state_prev + sys_err
                                        #p_t | p_t-1

            """
            the update stage calculates the kalman gain
            and updates the estimates into the system
            """

            kalman_gain = variance_state * ( ( variance_state + variance_measurement ) ** -1 )
                                        #k represents how much change we should implement
            estimate_state_prev = estimate_state + ( kalman_gain * ( row['Temperature'] - estimate_state ) )
                                        #updates the estimate
            variance_state_prev = ( 1 - kalman_gain ) * ( variance_state )
                                        #updates the variance of the system
        plt.subplot(subplot_index)
        plt.title("Q = %.7f" % sys_err)
        plt.plot(temperature_dataframe['Temperature'].index,
                 temperature_dataframe['Temperature'])
        plt.plot(temperature_dataframe.index, kalman_values, color='green', marker='+')
        subplot_index = subplot_index + 1

    """
    finally show the plot
    """
    plt.show()
