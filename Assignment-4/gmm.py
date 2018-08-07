import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None
        
    def gaussian_prob(self,input_x,mu,Sigma):
        if np.linalg.matrix_rank(Sigma) < Sigma.shape[0]:
            Sigma = Sigma + 1e-3*np.eye(Sigma.shape[0])
            if np.linalg.matrix_rank(Sigma) < Sigma.shape[0]:
                Sigma = Sigma + 1e-3*np.eye(Sigma.shape[0])
  
        K = input_x.shape[0]
        temp = np.reshape((input_x - mu),(K,1))
        part1 = np.exp(-1/2*(np.dot(np.dot(np.transpose(temp),np.linalg.inv(Sigma)),temp)))
        prob = np.power(np.linalg.det(2*np.pi*Sigma),-1/2)*part1
        return prob    
        
    
    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape
        
        gaussian_prob = self.gaussian_prob
        compute_log_likelihood = self.compute_log_likelihood
        
        n_cluster = self.n_cluster
        e = self.e
        max_iter = self.max_iter

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            ini_k_means = KMeans(n_cluster=n_cluster, max_iter = max_iter, e = e)
            mu_k, membership, num_updates = ini_k_means.fit(x)
            
            variance_k = []
            pi_k = []
            for v in range(n_cluster):
                x_temp = x[membership == v,:]
                mu_k_temp = mu_k[v]
                subtract = x_temp - mu_k_temp
                var_cov = np.zeros((D,D))
                for i in range(x_temp.shape[0]):
                    subtract = np.reshape((x_temp[i] - mu_k_temp),(D,1))
                    var_cov = var_cov + np.dot(subtract,np.transpose(subtract))
                variance_k.append(var_cov/x_temp.shape[0])
                pi_k.append(x_temp.shape[0]/N)
                
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            mu_k = np.random.uniform(low=0.0, high=1.0, size=(n_cluster,D))
            variance_k  = [np.identity((D))]*n_cluster
            pi_k = [1/n_cluster]*n_cluster
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        #--- COMPUTE THE log-likelihood
        
        self.means, self.variances,self.pi_k = np.array(mu_k),np.array(variance_k),np.array(pi_k)
            
        l = compute_log_likelihood(x)  
        
        iteration = 0
        while iteration < max_iter: 
            ##--- E step: compute responsibilities
            gamma_ik_temp = np.zeros((N,n_cluster))
            for p in range(N):
                for j in range(n_cluster):
                    gamma_ik_temp[p,j] = pi_k[j]*gaussian_prob(x[p],mu_k[j],variance_k[j]) 
            gamma_ik = np.divide(gamma_ik_temp,np.transpose(np.tile(np.sum(gamma_ik_temp,axis = 1),(n_cluster,1))))
            
            
            N_k = np.sum(gamma_ik,axis = 0)

            ##--- M step
            #------ Estimate means
            mu_k = []
            for s in range(n_cluster):
                mu_k.append(np.sum(np.multiply(np.transpose(np.tile(gamma_ik[:,s],(D,1))),x),axis = 0)/N_k[s])
            variance_k = []
            pi_k = []
            for m in range(n_cluster):
                var = np.zeros((D,D))
                for t in range(N):
                    minus = np.reshape((x[t] - mu_k[m]),(D,1))
                    var = var + gamma_ik[t,m]*np.dot(minus,np.transpose(minus))
                variance_k.append(var/N_k[m])
                pi_k.append(N_k[m]/N)
            
            self.means, self.variances,self.pi_k = np.array(mu_k),np.array(variance_k),np.array(pi_k)
            #---- converged? loglikelihood
            l_new = compute_log_likelihood(x)  
            
            if np.abs(l - l_new) <= e:
                break
            iteration = iteration + 1
            l = l_new
        return iteration
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        mu_k,variance_k,pi_k = self.means, self.variances,self.pi_k
        draw_sample = []
        for a in range(N):
            k = np.argmax(np.random.multinomial(1, pi_k, size=1))
            draw_sample.append(np.random.multivariate_normal(mu_k[k], variance_k[k]))
        return np.array(draw_sample)
        
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        N,D= x.shape
        gaussian_prob = self.gaussian_prob
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        mu_k = self.means
        variance_k = self.variances
        pi_k = self.pi_k
        n_cluster = self.n_cluster
        px = []
        for n in range(N):
            pxi = 0
            for u in range(n_cluster):
                pxi = pxi + pi_k[u]*gaussian_prob(x[n,:],mu_k[u],variance_k[u])
            px.append(pxi)
        loglike = np.sum(np.log(px)).astype(float)
        return np.float64(loglike).item()
        # DONOT MODIFY CODE BELOW THIS LINE
