import itertools
import numpy as np
import pandas as pd


def BT(df, epsilon=1e-9):
    n_competitors = df.shape[1]
    competitors = df.columns

    win_matrix = np.zeros((n_competitors, n_competitors))

    for pair in itertools.combinations(range(n_competitors), 2):
        idx_a, idx_b = pair
        competitor_a = competitors[idx_a]
        competitor_b = competitors[idx_b]

        win_ab = np.sum([int(score_a > score_b) for score_a, score_b in zip(df[competitor_a], df[competitor_b])])
        win_ba = df.shape[0] - win_ab

        win_matrix[idx_a][idx_b] = win_ab
        win_matrix[idx_b][idx_a] = win_ba

    W = np.sum(win_matrix, axis=1)
    p = [0.5] * n_competitors

    while True:
        new_p = [0.5] * n_competitors
        for i in range(n_competitors):
            summing_term = 0
            for j in range(n_competitors):
                if i == j:
                    continue

                summing_term += (win_matrix[i][j] + win_matrix[j][i]) / (p[i] + p[j])

            new_p[i] = W[i] / summing_term

        new_p /= np.sum(new_p)
        diff = np.sum([(x - y) ** 2 for x, y in zip(p, new_p)])
        if diff < epsilon:
            return new_p
        p = new_p


if __name__ == '__main__':
    player_a = [2, 5, 2, 3, 4]
    player_b = [1, 2, 3, 4, 1]
    player_c = [2, 4, 5, 2, 2]
    df = pd.DataFrame.from_dict({'player_a': player_a, 'player_b': player_b, 'player_c': player_c})
    res = BT(df)
    print(res)
