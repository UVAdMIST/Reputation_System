import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh import palettes

def robust_avg(df_input):

    df_input = df_input.reset_index(drop=True)

    # robust average
    avg = []
    avg_weight = []
    temp = []

    #initial robust weight
    newp = [1.0 / len(df_input.columns)] * len(df_input.columns)

    df_weight = pd.DataFrame(columns=df_input.columns)

    for i in range(len(df_input)):
        # missing observations will not be included in the robust average computation
        test = df_input.loc[i,:][df_input.loc[i,:].notnull()]

        # initial weight
        avg_weight.append(1.0 / len(test))


        newp = [1.0 / len(test)] * len(test)
        p = [1.0 / len(test)] * len(test)
        x = 0
        while True:
            x = 0

            # compute deviation of each value from the robust average
            dev = (test - (test * newp).sum()) ** 2

            # apply robust average equation
            temp = 1.0 / ((dev / dev.sum()) + 0.1)

            newp = temp / temp.sum()
            for j in (newp - p):
                if j < 0.0001:
                    x = x + 1
            # temporary robust weight under iteration
            p = newp
            # print 'x = %s' % x
            if np.all(np.isnan(newp)) == True:
                # print(i)
                # print('all records are 0')
                avg.append(test.sum())
                newp = [1.0 / len(df_input.columns)] * len(df_input.columns)
                for k in range(len(df_input.columns)):
                    df_weight.at[i, df_weight.columns[k]] = newp[k]

                avg_weight[i] = 1.0 / len(df_input.columns)
                break
            if x == len(test):
                for k in range(len(p)):
                    temp_columns = df_weight[df_input.loc[i,:][df_input.loc[i,:].notnull()].index].columns

                    df_weight.at[i, temp_columns[k]] = p[k]
                # assign initial weight to missing observations so that the score won't change
                null_columns = df_weight[df_input.loc[i, :][df_input.loc[i, :].isnull()].index].columns
                for h in null_columns:
                    df_weight.at[i, h] = avg_weight[i]
                avg.append((test * p).sum())
                break
    return df_weight


def beta_repu(df_input, df_weight, ff = 1.0): #ff is forgetting factor

    df_input = df_input.reset_index(drop=True)
    df_mean = df_input.mean(axis = 1)


    # n = weight of reward
    n = 1.0
    # m = weight of penalty
    m = 1.0
    # R = rescale factor
    R = 10

    # set alpha and beta initial value
    initial = 0

    # set alpha and beta dataframe
    df_alpha = pd.DataFrame(columns=df_input.columns)
    df_alpha.at[0] = initial

    df_beta = pd.DataFrame(columns=df_input.columns)
    df_beta.at[0] = initial

    score1 = df_input.copy()

    for i in range(len(df_input)):

        for j in df_input:
            std_weight = np.std(df_weight.loc[i][df_input.loc[i].notnull()])
            dev = df_weight[j][i] - df_weight.mean(axis = 1)[i]
            T = dev / std_weight

            # if std = 0, it means all observations are the same (most likely to be 0) --> do not change the score
            if std_weight < 0.000000001:
                df_alpha.at[i + 1, j] = df_alpha[j][i]
                df_beta.at[i + 1, j] = df_beta[j][i]

            elif np.isinf(T) == True:
                df_alpha.at[i + 1, j] = df_alpha[j][i]
                df_beta.at[i + 1, j] = df_beta[j][i]

            elif T < 0.0:
                if df_mean[i] < 5.0:
                    # T = 0
                    if T < 0:
                        T = -1.0
                    else:
                        T = 1.0
                else:
                    if T < 0:
                        T = -1.0
                    else:
                        T = 1.0
                df_alpha.at[i + 1, j] = df_alpha[j][i] * ff
                df_beta.at[i + 1, j] = df_beta[j][i] * ff + n * abs(T)

            elif T == 1.0 / len(df_input.columns):
                df_alpha.at[i + 1, j] = df_alpha[j][i]
                df_beta.at[i + 1, j] = df_beta[j][i]

            # if std = 0, T will be nan, do not change the score
            elif np.isnan(T) == True:
                df_alpha.at[i + 1, j] = df_alpha[j][i]
                df_beta.at[i + 1, j] = df_beta[j][i]
                # print("Fuck! it's NaN")
                # print("alpha fuck = %s" %df_alpha[j][i])
                # print("beta fuck = %s" %df_beta[j][i])
            else:
                if df_mean[i] < 5.0:
                    # T = 0
                    if T < 0:
                        T = -1.0
                    else:
                        T = 1.0
                else:
                    if T < 0:
                        T = -1.0
                    else:
                        T = 1.0
                df_alpha.at[i + 1, j] = df_alpha[j][i] * ff + m * abs(T)
                df_beta.at[i + 1, j] = df_beta[j][i] * ff

            alpha = float(df_alpha[j][i + 1])
            beta = float(df_beta[j][i + 1])
            score = R * ((alpha + 1) / (alpha + beta + 2))
            #         print 'alpha = %s' %alpha
            #         print 'beta = %s' %beta
            #         print 'score is %s' %score
            #         print "--------------------"
            score1.at[i, j] = score

    in_score = 0 * df_input.iloc[0,:]
    #set values to 5
    for i in in_score.index:
        in_score[i] = 5

    #append the initial score to dataframe
    score1 = pd.DataFrame(columns=list(df_input.columns.values)).append(in_score).append(score1, ignore_index = True)

    return score1, df_alpha, df_beta

def plot_score(df_score, fig_name = ''):
    output = df_score.copy()

    fs = 9
    fig1, ax1 = plt.subplots(figsize=(3.5, 5.0))
    ax1.set_prop_cycle(color=palettes.Category20[20])
    # ms = ["o", "*", "s", "v", "D", "X", "^", ">", "<", "+"]
    ms = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    temp = 0
    for i in output.columns:
        # ax1.plot(output[i], label=i, marker = ms[temp], markersize = 4, alpha = 0.5)
        ax1.plot(output[i], label=i, alpha=1, marker = ms[temp])
        temp = temp + 1

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 - box.height * 0.50,
                     box.width, box.height * 0.50])

    ax1.set_xlabel('Time Epoch (day)', fontsize=fs)
    ax1.set_xlim(0, len(df_score))
    ax1.set_ylabel('Trust Score', fontsize=fs)
    ax1.set_yticks(np.arange(0, 11, 1))
    ax1.legend(fontsize=7, loc='lower center', ncol = 3, bbox_to_anchor = (0.5, -0.50))
    ax1.axhline(y = 5, linestyle = '--', alpha = 0.3)
    plt.tick_params(axis = 'both', labelsize = fs)
    plt.tight_layout()
    plt.savefig('score_ff' + fig_name + '.png', dpi=300)
    plt.show()