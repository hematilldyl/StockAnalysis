import numpy as np
import LinRegLearner,DTLearner,RTLearner
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
class BagLearner(object):
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def __init__(self, learner = DTLearner.DTLearner,bags=20, verbose = False,kwargs={},boost=False):
        #sourced from project documentation, default bag learner makes a random forest
        self.bags = bags
        self.boost = False
        learners = []
        for i in range(0,bags):
            learners.append(learner(**kwargs))
        self.learners = learners
        self.kwargs =kwargs
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def author(self):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        return 'dh'

    def add_evidence(self,dataX,dataY):
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @summary: Add training data to learner  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param dataX: X values of data to add  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param dataY: the Y training values  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        for learner in self.learners:
            idx = np.random.choice(dataX.shape[0],dataX.shape[0])
            X_bag = dataX[idx]
            y_bag = dataY[idx]
            learner.add_evidence(X_bag,y_bag)
        return
    def query(self,points):
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @summary: Estimate a set of test points given the model we built.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param points: should be a numpy array with each row corresponding to a specific query.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @returns the estimated values according to the saved model.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """
        predicted = np.array([learner.query(points) for learner in self.learners])
        return np.mean(predicted,axis=0)

