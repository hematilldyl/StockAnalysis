import numpy as np
import LinRegLearner,DTLearner,RTLearner,BagLearner
class InsaneLearner(object):
    def __init__(self, bag_learner = BagLearner.BagLearner, learner=LinRegLearner.LinRegLearner,bags=20,num_bag_learners=20,verbose = False,kwargs={}):
        self.bags,self.kwargs,self.num_bag_learners= bags,kwargs,num_bag_learners
        bag_learners = []
        for i in range(0,num_bag_learners):
            bag_learners.append(bag_learner(learner=learner,**kwargs))
        self.bag_learners = bag_learners
    def author(self):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        return 'dhematillake3'
    def add_evidence(self,dataX,dataY):
        for bag_learner in self.bag_learners:
            bag_learner.add_evidence(dataX,dataY)
        return
    def query(self,points):
        predicted = np.array([learner.query(points) for learner in self.bag_learners])
        return np.mean(predicted,axis=0)