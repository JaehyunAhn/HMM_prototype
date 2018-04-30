# coding: utf-8
# 비터비, CRF & HMM difference
# * https://ko.wikipedia.org/wiki/%EC%A1%B0%EA%B1%B4%EB%B6%80_%EB%AC%B4%EC%9E%91%EC%9C%84%EC%9E%A5
# url: https://ratsgo.github.io/machine%20learning/2017/10/14/computeHMMs/
# url: https://ratsgo.github.io/machine%20learning/2017/11/10/CRF/

class Model(object):
    def __init__(self, states, symbols, start_prob=None, trans_prob=None, emit_prob=None):
        # Q: 상태(states): hot, cold
        self._states = set(states)
        # observation: 아이스 소비 개수 1, 2, 3
        self._symbols = set(symbols)
        # start prob: p(hot | start), p(cold | start)
        self._start_prob = None
        self._trans_prob = None
        self._emit__prob = None

    def _forward(self, sequence):
        # sequence : 0 (start)
        # 아이스크림 소비 시퀀스 [1, 2, 3, 2 ... ]
        sequence_length = len(sequence)
        if sequence_length == 0:
            return []
        # DP 를 사용하여 alpha 라는 변수에 저장
        alpha = [{}]
        # alpha[0] = {
        #   'hot': p(hot | start) * p(1| hot),
        #   'cold': p(cold | start) * p(1| cold)
        # 0번째 sequence 에서 1번째 sequence 로 갈때 likelihood value
        for state in self._states:
            alpha[0][state] = self._start_prob(state) * self._emit__prob(state, sequence[0])
        # sequence 2번째 부터 마지막까지 likelihood value, j 번째 상태에서 o_t 가 나타날 확률을 구하는 법
        for index in range(1, sequence_length):
            alpha.append({})
            for state_to in self._states:
                # 여러 시점의 states 로부터 특정 미래 시점의 probability sum 을 모두 계산
                prob = 0
                # DP probability
                for state_from in self.states:
                    # index-1 : [바로 이전의 모든 시점 | states 모든 경우의 수] * [전환 확률]
                    prob += alpha[index - 1][state_from] * self._trans_prob(state_from, state_to)
                alpha[index][state_to] = prob * self._emit__prob(state_to, sequence[index])
        return alpha

