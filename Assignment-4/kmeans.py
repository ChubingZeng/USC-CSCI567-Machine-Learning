import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        
    def distance(self,vec1,vec2):
        sub = np.subtract(vec1,vec2)
        return np.inner(sub,sub)
    
    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape
        distance = self.distance
        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        #- Initialize
        mu_k = x[np.random.choice(N, self.n_cluster, replace=False),:]
        iteration = 1
        J = np.inf
        converged = False
        number_of_updates = np.int(0)
        
        #- repeat
        while iteration < self.max_iter:
            R = np.zeros((N,self.n_cluster))
            cluster_assignment = []
            distance_matrix = []
            for i in range(N):
                distance_k = []
                for j in range(self.n_cluster):
                    distance_k.append(distance(x[i],mu_k[j])) 
                distance_matrix.append(distance_k)
                cluster = np.argmin(distance_k)
                cluster_assignment.append(cluster)
                R[i,cluster] = 1
            
            distance_ik = np.array(distance_matrix)
            cluster_assignment = np.array(cluster_assignment)
            J_new = np.sum(np.multiply(R,distance_ik))/N
            
            if np.abs(J - J_new) < self.e:
                break
                
            
            J = J_new 
            
            mu_k = np.zeros((self.n_cluster,D))
            for l in range(self.n_cluster):
                mu_k[l,:] = np.sum(x[cluster_assignment == l,],axis = 0)/np.sum(cluster_assignment == l)
            
            iteration = iteration +1 
            number_of_updates = number_of_updates + 1
        
        return mu_k, cluster_assignment, number_of_updates
        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        
        
    def distance(self,vec1,vec2):
        sub = np.subtract(vec1,vec2)
        return np.inner(sub,sub)
    
    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        
        distance = self.distance
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, i = k_means.fit(x)
        centroid_labels = []
        
        for cluster in range(self.n_cluster):
            # its member
            sub_y = y[membership == cluster]
            (_, idx, counts) = np.unique(sub_y, return_index=True, return_counts=True)
            index = idx[np.argmax(counts)]
            mode = sub_y[index]
            centroid_labels.append(mode)

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = np.array(centroid_labels)
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        distance = self.distance
        pred = []
        for m in range(N):
            distance_obs = []
            for n in range(self.n_cluster):
                distance_obs.append(distance(x[m,:],self.centroids[n,:]))
            pred.append(self.centroid_labels[np.argmin(distance_obs)])
            
        return np.array(pred)
        # DONOT CHANGE CODE BELOW THIS LINE


