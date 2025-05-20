import numpy as np  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
class RTLearner(object):
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def __init__(self, verbose = False,leaf_size=1):
        self.rt = None
        self.leaf_size = leaf_size
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def author(self):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        return 'dh'

    def build_tree(self, data):
        #Sourced from lecture pseudocode

        # for regression, use the mean of the response vector at the leaf size
        leaf = np.array([-1, np.mean(data[:, -1]), np.nan, np.nan])
        if data.shape[0] <= self.leaf_size:
            return leaf
        if len(np.unique(data[:, -1])) == 1:
            return leaf
        #shuffle data index for feature selection at random
        random_values = np.arange(data.shape[1]-1)
        np.random.shuffle(random_values)
        i = 0
        for split in random_values:
            SplitVal = np.median(data[:, split])
            left = data[data[:, split] <= SplitVal]
            right = data[data[:, split] > SplitVal]
            #check case where split does not divide data into pieces
            if np.size(left) == np.size(data) or np.size(right) == np.size(data):
                i += 1
                continue
            if len(np.unique(left)) and len(np.unique(right)) > 1:
                break
            i += 1
        if i == len(random_values):
            return leaf

        lefttree = self.build_tree(left)
        righttree = self.build_tree(right)
        if lefttree.ndim > 1:
            root = np.array([split, SplitVal, 1, np.shape(lefttree)[0] + 1])
        elif lefttree.ndim == 1:
            root = np.array([split, SplitVal, 1, 2])
        return np.vstack((root, lefttree, righttree))

    def search(self,point,entry=0):
        # search helper function for query, searches nodes until leaf is reached
        # node pointer 'entry' increments until leaf is reached
        feature,SplitVal = self.rt[entry,0:2]
        if feature ==-1:
            return SplitVal
        elif point[int(feature)]<=SplitVal:
            pred = self.search(point,entry+int(self.rt[entry,2])) #go left
        else:
            pred = self.search(point,entry+int(self.rt[entry,3])) #go right
        return pred
    def add_evidence(self,dataX,dataY):
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @summary: Add training data to learner  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param dataX: X values of data to add  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param dataY: the Y training values  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        data = np.concatenate((dataX,dataY[:,None]),axis=1)
        self.rt = self.build_tree(data)
        return
    def query(self,points):
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @summary: Estimate a set of test points given the model we built.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param points: should be a numpy array with each row corresponding to a specific query.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @returns the estimated values according to the saved model.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """
        #use vectorized np apply function to apply helper method to the vector
        return np.apply_along_axis(self.search,1,points)
