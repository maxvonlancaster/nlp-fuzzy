import skfuzzy
import numpy as np
import matplotlib.pyplot as plt

def calculate_fuzzy_param(params, max):
    # Define the universe of values for parameters
    range = np.arange(0,max,max/16)

    # Define trapezoidal membership functions for parameters
    low = skfuzzy.trapmf(range, [0, 0, max/8, max/4])
    middle = skfuzzy.trapmf(range, [max/8, max/4, 3*max/8, 5*max/8])
    high = skfuzzy.trapmf(range, [3*max/8, 5*max/8, max, max])

    # fig, ax = plt.subplots()

    # ax.plot(range, low, 'g', linewidth=1.5, label='Low')
    # ax.plot(range, middle, 'r', linewidth=1.5, label='Middle')
    # ax.plot(range, high, 'c', linewidth=1.5, label='High')

    # ax.set_title('Angry Mood Membership Functions (Trapezoidal)')
    # ax.legend()

    # # Turn off top/right axes
    # for ax in (ax,):
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.get_xaxis().tick_bottom()
    #     ax.get_yaxis().tick_left()

    # plt.tight_layout()

    # plt.show()

    print(skfuzzy.interp_membership(range, high, 0.06))