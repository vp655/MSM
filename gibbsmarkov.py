import numpy as np
from numpy.random import normal, uniform, beta, binomial, randint
from scipy.stats import norm

class GibbsSampler:
    def __init__(self, observed_data, numstates=2, var=0.1, ar=0):
        """
        Args: 
            initial_state (list or np.array): Initial values for all variables.
            conditional_distributions (list of callables): Each function takes the current state and returns a sample for one variable.
            num_samples (int): Number of samples to generate.
        """
        self.observed_data = observed_data
        self.numstates = numstates #markov states
        #self.ar = ar     #autoregressive components   
        self.var = var    #noise param

        self.hiddenestimates = np.zeros_like(observed_data, dtype=int)
        self.hidden_post = np.zeros((len(observed_data), numstates)) #probability distribution for each hidden

        self.transitionestimates = np.zeros((numstates, numstates))
        self.probs_post = np.zeros((numstates, numstates, 2)) #posterior alpha/beta for each state

        #self.arestimates = np.zeros((numstates, ar))
        #self.ar_post = np.zeros((numstates, ar, 2)) # posterior mean and variance for each ar coef
        self.meansestimates = np.zeros(numstates)
        self.means_post = np.zeros((numstates, 2)) # posterior mean, variance for each mean
        
    def initialize(self):

        for i in range(self.numstates):
            # Expected means: 0.75 for diagonal, rest split equally
            expected = np.full(self.numstates, 0.25 / (self.numstates - 1))
            expected[i] = 0.75

            # Convert expected means into alpha/beta parameters
            for j in range(self.numstates):
                mean_ij = expected[j]
                alpha = mean_ij * 6
                beta = (1 - mean_ij) * 6
                self.probs_post[i, j] = [alpha, beta]
                self.transitionestimates[i, j] = alpha / (alpha + beta)
                #print(i, j, "initializing transition estimates", self.transitionestimates[i, j])

        self.meansestimates[0] = 0
        self.meansestimates[1] = 5
        self.means_post[:, 0] = self.meansestimates
        self.means_post[:, 1] = 3 #first prior should have variance .5

    def forwardBackward(self):

        #calculate alpha_1(k)= p(s_t = k | z1:t) for all k (and then normalize)
        numstates = self.numstates
        alpha = np.zeros((len(self.observed_data),numstates))

        T = len(self.observed_data)

        p00 = self.transitionestimates[0, 0]
        p11 = self.transitionestimates[1, 1]
        prior = np.array([(1-p11)/(2-p00-p11), 0])
        prior[1] = 1 - prior[0] #we are assuming k = 2 

        for k in range(0, numstates):
            alpha[0][k] = prior[k] * norm.pdf(self.observed_data[0], self.meansestimates[k], self.var)

        alpha[0] = alpha[0] / np.sum(alpha[0])

        #alpha_t_k represents p(s_t = k | z1:t)
        for i in range(1, T):
            for k in range(0,numstates):
                #print("alpha[i-1]", alpha[i-1])
                #print("transition estimates", self.transitionestimates[:, k])
                inner_sum = np.dot(alpha[i-1], self.transitionestimates[:, k])
                #print(i, k, "inner sum", inner_sum)
                alpha[i][k] = inner_sum * norm.pdf(self.observed_data[i], self.meansestimates[k], self.var)
            alpha[i] = alpha[i] / np.sum(alpha[i]) #normalize
        #print("alpha", alpha)
        #backward
        self.hiddenestimates[-1] = np.random.choice(numstates, p=alpha[-1])
        for i in range(T-2, -1, -1):
            #P(s_t = i | s_t+1 = j, z1:T) is prop to alpha_t(i)P_ij
            #P_ij = p(st+1 = j | st = i)
            distr = alpha[i] * self.transitionestimates[:, self.hiddenestimates[i+1]]
            distr /= sum(distr)
            self.hiddenestimates[i] = np.random.choice(numstates, p=distr)
        #print("New hidden estimates", self.hiddenestimates)
    def updateprobabilities(self):
        counts = np.zeros((self.numstates, self.numstates), dtype=int)

        for (from_state, to_state) in zip(self.hiddenestimates[:-1], self.hiddenestimates[1:]):
            counts[from_state, to_state] += 1
        
        self.probs_post[0, 0, 0] += counts[0, 0]
        self.probs_post[0, 0, 1] += counts[0, 1]
        self.probs_post[1, 1, 0] += counts[1, 1]
        self.probs_post[1, 1, 1] += counts[1, 0]
        
        self.transitionestimates[0, 0] = np.random.beta(self.probs_post[0, 0, 0], self.probs_post[0, 0, 1])
        self.transitionestimates[1, 1] = np.random.beta(self.probs_post[1, 1, 0], self.probs_post[1, 1, 1])
        self.transitionestimates[0, 1] = 1 - self.transitionestimates[0, 0]
        self.transitionestimates[1, 0] = 1 - self.transitionestimates[1, 1]

        #print("New transition estimates", self.transitionestimates)
    def updatemeans(self):

        for k in range(self.numstates):

            m_k = self.means_post[k][0] #prior mean for MEAN
            v_k = self.means_post[k][1] #prior variance for MEAN

            zk = self.observed_data[self.hiddenestimates == k]
            nk = len(zk)
            

            # Compute posterior parameters
            if nk > 0:
                bar_zk = np.mean(zk)
                v_post = 1 / (nk / self.var + 1 / v_k)
                m_post = v_post * ((nk * bar_zk) / self.var + m_k / v_k)
            else:
                v_post = v_k
                m_post = m_k
            
            self.meansestimates[k] = normal(loc=m_post, scale=np.sqrt(v_post))
            self.means_post[k][0] = m_post
            self.means_post[k][1] = v_post
        #print("New mean estimates", self.meansestimates)
        
    def sample(self, numsteps=1, burn_in = 100):
        for i in range(numsteps):
            self.forwardBackward()
            self.updateprobabilities()
            self.updatemeans()
        print("Mean estimates:", self.meansestimates)
        print("Transition prob estimates:", self.transitionestimates)
        print("Hidden state estimates:", self.hiddenestimates)


def generateData(n=250):
    
    observedstates = np.zeros(n+1)
    hiddenstates = np.zeros(n+1, dtype=int)
    
    hiddenmarkov = np.array([[0.75, 0.25],[0.3, 0.7]])

    observedstates[0] = np.random.normal(1, .25)

    for i in range(1,n+1):
        curr_state = hiddenstates[i-1]
        hiddenstates[i] = np.random.choice([0, 1], p=hiddenmarkov[curr_state])
        observedstates[i] = np.random.normal(2, 0.1) if hiddenstates[i] else np.random.normal(1, 0.1)

    #samples = np.random.randn(1000)
    return hiddenstates, observedstates

hidden, observed = generateData()

print("true hidden", hidden)
print("true observed", observed)

sampler = GibbsSampler(observed_data=observed)
sampler.initialize()

sampler.sample(numsteps=10)

print("Hidden state accuracy:", 1 - sum(abs(sampler.hiddenestimates - hidden)) / len(hidden))