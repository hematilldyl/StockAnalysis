import numpy as np  		  	   		  	  			  		 			     			  	 
import matplotlib.pyplot as plt
  		  	   		  	  			  		 			     			  	 
def author():  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    :return: The GT username of the student  		  	   		  	  			  		 			     			  	 
    :rtype: str  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    return "dh
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def gtid():  		  	   		  	  			  		 			     			  	 		  	   		  	  			  		 			     			  	 
    return `123
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def get_spin_result(win_prob):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    :param win_prob: The probability of winning  		  	   		  	  			  		 			     			  	 
    :type win_prob: float  		  	   		  	  			  		 			     			  	 
    :return: The result of the spin.  		  	   		  	  			  		 			     			  	 
    :rtype: bool  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    result = False  		  	   		  	  			  		 			     			  	 
    if np.random.random() <= win_prob:  		  	   		  	  			  		 			     			  	 
        result = True  		  	   		  	  			  		 			     			  	 
    return result  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def test_code():  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    Method to test your code  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    win_prob = 18/38  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once  		  	   		  	  			  		 			     			  	 
    print(get_spin_result(win_prob))  # test the roulette spin  		  	   		  	  			  		 			     			  	 
    # add your code here to implement the experiments

    episode_values = np.zeros([1000,10])
    expected_values = np.zeros(4)
    #Question 1
    for i in range(0, 10):
        episode_winnings = 0
        j = 0
        while episode_winnings < 80 and j<1000:
            won = False
            bet_amount = 1
            while not won:
                j=j+1
                won = get_spin_result(win_prob)
                if won:
                    episode_winnings = episode_winnings+bet_amount
                else:
                    episode_winnings = episode_winnings-bet_amount
                    bet_amount = bet_amount*2
                episode_values[j, i] = episode_winnings
        if j <=1000:
            episode_values[j:,i]=episode_winnings

    plt.plot(episode_values)
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings ($)")
    plt.title("Episode Plot for 10 Episodes")
    plt.legend(['E1','E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10'])
    plt.savefig('images/Figure_1.png')

    # Question 2
    episode_values = np.zeros([1000,1000])

    for i in range(0, 1000):
        episode_winnings = 0
        j = 0
        while episode_winnings < 80 and j<1000:
            won = False
            bet_amount = 1
            while not won:
                j=j+1
                won = get_spin_result(win_prob)
                if won:
                    episode_winnings = episode_winnings+bet_amount
                else:
                    episode_winnings = episode_winnings-bet_amount
                    bet_amount = bet_amount*2
                episode_values[j, i] = episode_winnings
        if j <=1000:
            episode_values[j:,i]=episode_winnings
    expected_values[0] = np.mean(episode_values, axis=1)[-1]
    expected_values[2]=np.count_nonzero(episode_values[-1, :] == 80)/1000
    plt.figure()
    plt.plot(np.mean(episode_values, axis=1),label='Mean Win Per Spin')
    plt.plot(np.mean(episode_values, axis=1)+np.std(episode_values, axis=1),label='Mean Win + Std Dev.')
    plt.plot(np.mean(episode_values, axis=1)-np.std(episode_values, axis=1), label='Mean Win - Std Dev')
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.title("Episode Plot Average Spin Win")
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings at Spin Number for All Episodes ($)")
    plt.legend()
    plt.savefig('images/Figure_2.png')


    plt.figure()
    plt.plot(np.median(episode_values, axis=1),label='Median Win Per Spin')
    plt.plot(np.median(episode_values, axis=1)+np.std(episode_values, axis=1),label='Median Win + Std Dev.')
    plt.plot(np.median(episode_values, axis=1)-np.std(episode_values, axis=1), label='Median Win - Std Dev')
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.title("Episode Plot Median Spin Win")
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings at Spin Number for All Episodes ($)")
    plt.legend()
    plt.savefig('images/Figure_3.png')


    episode_values = np.zeros([1000,1000])

    for i in range(0, 1000):
        episode_winnings = 0
        j = 0
        while -256<episode_winnings < 80 and j<999:
            won = False
            bet_amount = 1
            while not won and episode_winnings>-256 and j<999:
                j=j+1
                won = get_spin_result(win_prob)
                if won:
                    episode_winnings = episode_winnings+bet_amount
                else:
                    episode_winnings = episode_winnings-bet_amount
                    if episode_winnings ==-256:
                        break
                    elif episode_winnings<0:
                        if 256 - abs(episode_winnings)<bet_amount*2:
                            bet_amount = 256-abs(episode_winnings)
                    else:
                        bet_amount = bet_amount*2
                episode_values[j, i] = episode_winnings
        if j <=1000:
            episode_values[j:,i]=episode_winnings
    expected_values[1] = np.mean(episode_values, axis=1)[-1]
    expected_values[3] = np.count_nonzero(episode_values[-1, :] == 80) / 1000
    np.savetxt('p1_results.txt',expected_values)
    plt.figure()
    plt.plot(np.mean(episode_values, axis=1),label='Mean Win Per Spin')
    plt.plot(np.mean(episode_values, axis=1)+np.std(episode_values, axis=1),label='Mean Win + Std Dev.')
    plt.plot(np.mean(episode_values, axis=1)-np.std(episode_values, axis=1), label='Mean Win - Std Dev')
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.title("Bankroll Episode Plot Average Spin Win")
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings at Spin Number for All Episodes ($)")
    plt.legend()
    plt.savefig('images/Figure_4.png')


    plt.figure()
    plt.plot(np.median(episode_values, axis=1),label='Median Win Per Spin')
    plt.plot(np.median(episode_values, axis=1)+np.std(episode_values, axis=1),label='Median Win + Std Dev.')
    plt.plot(np.median(episode_values, axis=1)-np.std(episode_values, axis=1), label='Median Win - Std Dev')
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.title("Bankroll Episode Plot Median Spin Win")
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings at Spin Number for All Episodes ($)")
    plt.legend()
    plt.savefig('images/Figure_5.png')

  

if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    test_code()  		  	   		  	  			  		 			     			  	 
