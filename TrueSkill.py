import itertools
import trueskill as ts
from trueskill import rate_1vs1


def TrueSkill(df, mu=ts.MU, sigma=ts.SIGMA,
              beta=ts.BETA, tau=ts.TAU,
              draw_prob=ts.DRAW_PROBABILITY):

    trueskill_env = ts.TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau, draw_probability=draw_prob)
    competitors = df.columns

    system_ratings = {}
    for x in competitors:
        system_ratings[x] = trueskill_env.create_rating()

    for (sys_a, sys_b) in itertools.combinations(competitors, 2):
        scores_a, scores_b = df[sys_a], df[sys_b]
        for Xs_a, Xs_b in zip(scores_a, scores_b):
            if Xs_a > Xs_b:
                sys_a_rating, sys_b_rating = rate_1vs1(system_ratings[sys_a], system_ratings[sys_b])
            elif Xs_a < Xs_b:
                sys_b_rating, sys_a_rating = rate_1vs1(system_ratings[sys_b], system_ratings[sys_a])
            else:
                sys_b_rating, sys_a_rating = rate_1vs1(system_ratings[sys_b], system_ratings[sys_a], drawn=True)

            system_ratings[sys_a] = sys_a_rating
            system_ratings[sys_b] = sys_b_rating

    return [system_ratings[sys].mu for sys in df.columns]
