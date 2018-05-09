# coding: utf-8
# 비터비, CRF & HMM difference
# * https://ko.wikipedia.org/wiki/%EC%A1%B0%EA%B1%B4%EB%B6%80_%EB%AC%B4%EC%9E%91%EC%9C%84%EC%9E%A5
# url: https://ratsgo.github.io/machine%20learning/2017/10/14/computeHMMs/
# url: https://ratsgo.github.io/machine%20learning/2017/11/10/CRF/

def _normalize_prob(prob, item_set):
    result = {}
    if prob is None:
        number = len(item_set)
        for item in item_set:
            result[item] = 1.0 /number
    else:
        prob_sum = 0.0
        for item in item_set:
            prob_sum += prob.get(item, 0)
        if prob_sum > 0:
            for item in item_set:
                result[item] = prob.get(item, 0) / prob_sum
        else:
            for item in item_set:
                result[item] = 0
    return result

def _normalize_prob_two_dim(prob, item_set1, item_set2):
    result = {}
    if prob is None:
        for item in item_set1:
            result[item] = _normalize_prob(None, item_set2)
    else:
        for item in item_set1:
            result[item] = _normalize_prob(prob.get(item), item_set2)
    return result

def train():
    # learn with symbol & smoothing params
    '''
    + model.learn
    + return model
    :return:
    '''
    pass

class Model(object):
    def __init__(self, states, symbols, start_prob=None, trans_prob=None, emit_prob=None):
        # Q: 상태(states): hot, cold
        self._states = set(states)
        # observation: 아이스 소비 개수 1, 2, 3
        self._symbols = set(symbols)
        # start prob: p(hot | start), p(cold | start)
        self._start_prob = _normalize_prob(start_prob, self._states)
        self._trans_prob = _normalize_prob_two_dim(trans_prob, self._states, self._states)
        self._emit__prob = _normalize_prob_two_dim(emit_prob, self._states, self._symbols)

    def start_prob(self, state):
        if state not in self._states:
            return 0
        return self._start_prob[state]

    def emit__prob(self, state, symbol):
        if state not in self._states or symbol not in self._symbols:
            return 0
        return self._emit__prob[state][symbol]

    def trans_prob(self, state_from, state_to):
        if state_from not in self._states or state_to not in self._states:
            return 0
        return self._trans_prob[state_from][state_to]

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
            alpha[0][state] = self.start_prob(state) * self.emit__prob(state, sequence[0])
        # sequence 2번째 부터 마지막까지 likelihood value, j 번째 상태에서 o_t 가 나타날 확률을 구하는 법
        for index in range(1, sequence_length):
            alpha.append({})
            for state_to in self._states:
                # 여러 시점의 states 로부터 특정 미래 시점의 probability sum 을 모두 계산
                prob = 0
                # DP probability
                for state_from in self._states:
                    # index-1 : [바로 이전의 모든 시점 | states 모든 경우의 수] * [전환 확률]
                    prob += alpha[index - 1][state_from] * self.trans_prob(state_from, state_to)
                alpha[index][state_to] = prob * self.emit__prob(state_to, sequence[index])
        return alpha

    def _backward(self, sequence):
        # sequence : 0 (start)
        # 아이스크림 소비 시퀀스
        sequence_length = len(sequence)
        if sequence_length == 0:
            return []
        beta = [{}]
        for state in self._states:
            beta[0][state] = 1

        # beta_t of i = sum of j(transProb_ij * emitProb(seq t + 1) * beta_t+1_j
        for index in range(sequence_length -1, 0, -1):
            beta.insert(0, {})
            for state_from in self._states:
                prob = 0
                for state_to in self._states:
                    prob += self.trans_prob(state_from, state_to) * \
                        self.emit__prob(state_to, sequence[index]) * \
                        beta[1][state_to]
                beta[0][state_from] = prob
        return beta

    def evaluate(self):
        '''
        Likelihood calculation
        get sum of probabilities > and _forward algorithm
        :return:
        '''
        pass


    def decode(self):
        '''
        decode >> the backtrace map
        optimize and estimate map
        :return:
        '''
        pass


    def learn(self):
        '''
        find best state transition / emission prob
        _forward & _backward >> make a map of all possible way
        :return:
        '''
        pass
