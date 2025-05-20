"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Atlanta, Georgia 30332  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
All Rights Reserved  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template code for CS 4646/7646  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
and other users of this template code are advised not to share it with others  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or to make it available on publicly viewable websites including repositories  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or edited.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
We do grant permission to share solutions privately with non-students such  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
as potential employers. However, sharing with other current or future  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT honor code violation.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
-----do not edit anything above this line---  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import math
import matplotlib.pyplot as plt
import BagLearner as bl
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import sys
import time
import RTLearner

if __name__=="__main__":  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    if len(sys.argv) != 2:  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        print("Usage: python testlearner.py <filename>")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        sys.exit(1)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    inf = open(sys.argv[1])
    if inf.name == "Data/Istanbul.csv":
        data = np.array([list(map(float,s.strip().split(',')[1:])) for s in inf.readlines()[1:]])
    else:
        data = np.array([list(map(float, s.strip().split(','))) for s in inf.readlines()])
    np.random.seed(903741146)
    # compute how much of the data is training and testing  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    #EXPERIMENT1
    n_experiments = 50
    max_leaf = 50
    in_sample_rmse = np.zeros([max_leaf,n_experiments])
    out_sample_rmse = np.zeros([max_leaf,n_experiments])
    for leaf in range(1,max_leaf):
        data_experiment = np.copy(data)
        for i in range(0,n_experiments):
            dt_learner = dt.DTLearner(verbose=False, leaf_size=int(leaf))
            np.random.shuffle(data_experiment)
            trainX = data_experiment[:train_rows, 0:-1]
            trainY = data_experiment[:train_rows, -1]
            testX = data_experiment[train_rows:, 0:-1]
            testY = data_experiment[train_rows:, -1]
            dt_learner.add_evidence(trainX, trainY)
            predY = dt_learner.query(trainX)  # get the predictions
            in_sample_rmse[leaf,i]=math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
            predY = dt_learner.query(testX)  # get the predictions
            out_sample_rmse[leaf,i]=math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
    fig,ax=plt.subplots()
    plt.plot(np.mean(in_sample_rmse[1:,:],axis=1),label="In Sample RMSE")
    plt.plot(np.mean(out_sample_rmse[1:,:],axis=1),label="Out of Sample RMSE")
    plt.legend()
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Error Plot of Decision Tree')
    #ax.text(0.6,0.6,'dhematillake3',transform=ax.transAxes,fontsize=50,color='gray',alpha=0.5,ha='center',va='center',rotation='30')
    plt.savefig("Experiment1.png")
    
    
    # EXPERIMENT2
    n_experiments = 10
    max_leaf = 50
    in_sample_rmse = np.zeros([max_leaf,n_experiments])
    out_sample_rmse = np.zeros([max_leaf,n_experiments])
    for leaf in range(1,max_leaf):
        data_experiment = np.copy(data)
        for i in range(0,n_experiments):
            bag_learner = bl.BagLearner(verbose=False,kwargs={'leaf_size':int(leaf)})
            np.random.shuffle(data_experiment)
            trainX = data_experiment[:train_rows, 0:-1]
            trainY = data_experiment[:train_rows, -1]
            testX = data_experiment[train_rows:, 0:-1]
            testY = data_experiment[train_rows:, -1]
            bag_learner.add_evidence(trainX, trainY)
            predY = bag_learner.query(trainX)  # get the predictions
            in_sample_rmse[leaf,i]=math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
            predY = bag_learner.query(testX)  # get the predictions
            out_sample_rmse[leaf,i]=math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
    fig,ax=plt.subplots()
    plt.plot(np.mean(in_sample_rmse[1:,:],axis=1),label="In Sample RMSE")
    plt.plot(np.mean(out_sample_rmse[1:,:],axis=1),label="Out of Sample RMSE")
    plt.legend()
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Error Plot of Random Forest')
    #ax.text(0.6,0.6,'dhematillake3',transform=ax.transAxes,fontsize=50,color='gray',alpha=0.5,ha='center',va='center',rotation='30')
    plt.savefig("Experiment2.png")

    # EXPERIMENT3
    n_experiments = 50
    max_leaf = 50
    in_sample_mae_dt = np.zeros([max_leaf, n_experiments])
    out_sample_mae_dt = np.zeros([max_leaf, n_experiments])
    in_sample_mae_rt = np.zeros([max_leaf, n_experiments])
    out_sample_mae_rt = np.zeros([max_leaf, n_experiments])
    timetotrain_dt = np.zeros([max_leaf, n_experiments])
    timetotrain_rt= np.zeros([max_leaf, n_experiments])


    for leaf in range(1, max_leaf):
        data_experiment = np.copy(data)
        for i in range(0, n_experiments):

            dt_learner = dt.DTLearner(verbose=False, leaf_size=int(leaf))
            rt_learner = rt.RTLearner(verbose=False, leaf_size=int(leaf))
            np.random.shuffle(data_experiment)
            trainX = data_experiment[:train_rows, 0:-1]
            trainY = data_experiment[:train_rows, -1]
            testX = data_experiment[train_rows:, 0:-1]
            testY = data_experiment[train_rows:, -1]
            start = time.time()
            dt_learner.add_evidence(trainX, trainY)
            end = time.time()
            timetotrain_dt[leaf,i]=end-start
            start = time.time()
            rt_learner.add_evidence(trainX, trainY)
            end = time.time()
            timetotrain_rt[leaf, i] = end - start
            predY_dt = dt_learner.query(trainX)  # get the predictions
            predY_rt = rt_learner.query(trainX)  # get the predictions
            in_sample_mae_dt[leaf, i] = ((abs(predY_dt-trainY))).sum() / trainY.shape[0]
            in_sample_mae_rt[leaf, i] = ((abs(predY_rt - trainY))).sum() / trainY.shape[0]
            predY_dt = dt_learner.query(testX)  # get the predictions
            predY_rt = rt_learner.query(testX)  # get the predictions
            out_sample_mae_dt[leaf, i] = ((abs(predY_dt-testY))).sum() / testY.shape[0]
            out_sample_mae_rt[leaf, i] = ((abs(predY_rt-testY))).sum() / testY.shape[0]
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))
    ax1.plot(np.mean(in_sample_mae_dt[1:, :], axis=1),"b--",label="In Sample Decision Tree MAE")
    ax1.plot(np.mean(out_sample_mae_dt[1:, :], axis=1),"b",label="Out of Sample Decision Tree MAE")
    ax1.plot(np.mean(in_sample_mae_rt[1:, :], axis=1),"r--",label="In Sample Random Tree MAE")
    ax1.plot(np.mean(out_sample_mae_rt[1:, :], axis=1),"r",label="Out of Sample Random Tree MAE")
    ax1.legend()
    ax1.set_xlabel('Leaf Size')
    ax1.set_ylabel('MAE')
    ax1.set_title('Error Plot of Random vs Decision Tree')

    ax2.plot(np.mean(timetotrain_dt[1:, :], axis=1), label="Decision Tree")
    ax2.plot(np.mean(timetotrain_rt[1:, :], axis=1), label="Random Tree")
    ax2.legend()
    ax2.set_xlabel('Leaf Size')
    ax2.set_ylabel('Time to Train (s)')
    ax2.set_title('Time to Train Random vs Decision Tree')


    # ax.text(0.6,0.6,'dhematillake3',transform=ax.transAxes,fontsize=50,color='gray',alpha=0.5,ha='center',va='center',rotation='30')
    plt.savefig("Experiment3.png",dpi=1000)
