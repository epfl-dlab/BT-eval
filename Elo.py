import itertools


class Elo:
    def __init__(self, k, g=1, homefield=0):
        self.ratingDict = {}
        self.k = k
        self.g = g
        self.homefield = homefield

    def add_player(self, name, rating=1500):
        self.ratingDict[name] = rating

    def game_over(self, winner, loser, winnerHome=False):
        if winnerHome:
            result = self.expectResult(self.ratingDict[winner] + self.homefield, self.ratingDict[loser])
        else:
            result = self.expectResult(self.ratingDict[winner], self.ratingDict[loser] + self.homefield)

        self.ratingDict[winner] = self.ratingDict[winner] + (self.k * self.g) * (1 - result)
        self.ratingDict[loser] = self.ratingDict[loser] + (self.k * self.g) * (0 - (1 - result))

    def expectResult(self, p1, p2):
        exp = (p2 - p1) / 400.0
        return 1 / ((10.0**(exp)) + 1)


def ELO(df, k=20, g=1, homefield=0):
    # n_competitors = df.shape[1]
    competitors = df.columns

    elo_eval = Elo(k, g, homefield)
    for x in competitors:
        elo_eval.add_player(x)

    for (sys_a, sys_b) in itertools.combinations(competitors, 2):
        scores_a, scores_b = df[sys_a], df[sys_b]
        for Xs_a, Xs_b in zip(scores_a, scores_b):
            if Xs_a > Xs_b:
                elo_eval.game_over(sys_a, sys_b)
            else:
                elo_eval.game_over(sys_b, sys_a)

    return [elo_eval.ratingDict[sys] for sys in df.columns]
