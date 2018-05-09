# coding: utf-8
# viterbi algorithm
# https://ko.wikipedia.org/wiki/%EB%B9%84%ED%84%B0%EB%B9%84_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98
# https://ratsgo.github.io/data%20structure&algorithm/2017/11/14/viterbi/
"""
    Viterbi Algorithm finds the most likely sequence of hidden states by dynamic way.
        like Assembly-Line Scheduling.

"""
def dptable(V):
    yield ' '.join(('%12d' % i) for i in range(len(V)))
    for state in V[0]:
        yield '%.7s: ' % state + ' '.join('%.7s' % ('%f' % v[state]['prob']) for v in V)

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    # initial all states
    for st in states:
        V[0][st] = {'prob': start_p[st] * emit_p[st][obs[0]], 'prev': None}
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = max(V[t-1][prev_st]['prob'] *
                              trans_p[prev_st][st] for prev_st in states
                              )
            for prev_st in states:
                # max 라면 다음 상태를 max의 상태로 정의하고 break
                if V[t-1][prev_st]['prob'] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    V[t][st] = {'prob': max_prob, 'prev': prev_st}
                    break
    # print out dptable line
    for line in dptable(V):
        print(line)

    # The highest probability
    max_prob = max(value['prob'] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    opt = []
    for st, data in V[-1].items():
        if data['prob'] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t+1][previous]['prev'])
        previous = V[t+1][previous]['prev']
    print('The steps of states are ' + '-'.join(opt) + ' with highest probability of %s' % max_prob)

if __name__ == '__main__':
    # observations
    obs = ('normal', 'cold', 'dizzy')
    # states will be hidden
    states = ('Healthy', 'Fever')

    start_p = {'Healthy': .6, 'Fever': .4}
    # state transition
    trans_p = {
        'Healthy': {'Healthy': .7, 'Fever': .3},
        'Fever': {'Healthy': .4, 'Fever': .6}
    }
    # emit probability
    emit_p = {
        'Healthy': {'normal': .5, 'cold': .4, 'dizzy': .1},
        'Fever': {'normal': .1, 'cold': .3, 'dizzy': .6}
    }
    # only we observe emitted probability

    viterbi(obs=obs,
            states=states,
            start_p=start_p,
            trans_p=trans_p,
            emit_p=emit_p
            )
