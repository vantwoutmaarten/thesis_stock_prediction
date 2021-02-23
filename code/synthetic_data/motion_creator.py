import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import pandas as pd

class Stochastic():
    """
    A Stochastic motion class constructor
    """
    def __init__(self,x0=0):
        """
        Init class
        """
        assert (type(x0)==float or type(x0)==int or x0 is None), "Expect a float or None for the initial value"
        
        self.x0 = float(x0)

    def gen_normal(self,n_step=100):
        """
        Generate motion by drawing from the Normal distribution
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process
            w[i] = yi
        
        return w

class Brownian():
    """
    A Brownian motion class constructor
    """
    def __init__(self,x0=0):
        """
        Init class
        """
        assert (type(x0)==float or type(x0)==int or x0 is None), "Expect a float or None for the initial value"
        
        self.x0 = float(x0)
    
    def gen_random_walk(self,n_step=100):
        """
        Generate motion by random walk
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        # Warning about the small number of steps
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1,-1])
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
     
    def gen_normal(self,n_step=100):
        """
        Generate motion by drawing from the Normal distribution
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
    
    def stock_price(
                    self,
                    s0=100,
                    mu=0.2,
                    sigma=0.68,
                    deltaT=146,
                    dt=0.1
                    ):
        """
        Models a stock price S(t) using the Weiner process W(t) as
        `S(t) = S(0).exp{(mu-(sigma^2/2).t)+sigma.W(t)}`
        

        Arguments:
            s0: Iniital stock price, default 100
            mu: 'Drift' of the stock (upwards or downwards), default 1
            sigma: 'Volatility' of the stock, default 1
            deltaT: The time period for which the future prices are computed, default 52 (as in 52 weeks)
            dt (optional): The granularity of the time-period, default 0.1
        
        After testing it turns out that mu/sigma are not very related to the drift/volatility
        
        Returns:
            s: A NumPy array with the simulated stock prices over the time-period deltaT
        """
        n_step = int(deltaT/dt)
        time_vector = np.linspace(0,deltaT,num=n_step)
        # Stock variation
        stock_var = (mu-(sigma**2/2))*time_vector
        # Forcefully set the initial value to zero for the stock price simulation
        self.x0=0
        # Weiner process (calls the `gen_normal` method)
        weiner_process = sigma*self.gen_normal(n_step)
        # Add two time series, take exponent, and multiply by the initial stock price
        s = s0*(np.exp(stock_var+weiner_process))
        
        return s

b = Brownian()
s = Stochastic()

def plot_stock_price(mu,sigma):
    """
    Plots stock price for multiple scenarios
    """
    plt.figure(figsize=(9,4))
    for i in range(8):
        plt.plot(b.stock_price(mu=mu,
                               sigma=sigma,
                               dt=0.1))
    plt.legend(['Scenario-'+str(i) for i in range(1,6)],
               loc='upper left')
    plt.hlines(y=100,xmin=0,xmax=1460,
               linestyle='--',color='k')
    plt.show()

def create_stock_price_scenario(mu, sigma, scenario_name):
    """
    create a scenario of a stock price. 
    Default is three years, 1460 days, this can be changed in the stockprice function.
    """
    deltaT = 146
    dt = 0.1
    # The stockprice is a brownian motion 7 days a week all year. 
    stockprice = b.stock_price(mu=mu,sigma=sigma,dt=0.1)
    

    # The sp_no_missing_values will take the allyear stockprice and remove the weekends + 7 days a year random holidays.
    sp_no_missing_values = copy.deepcopy(stockprice)

    sp_missing_closed_days = copy.deepcopy(stockprice)
    for i in range(len(sp_no_missing_values)):
        if(i%7==6 or i%7==5):
            sp_missing_closed_days[i] = np.nan
    # the holidaycounter sets the number of holidays. 
    holidaycounter = 0
    total_days = deltaT/dt

    random.randint(0, total_days-1)
    while(holidaycounter < (7/365)*(total_days)):
        randomday = random.randint(0, total_days)
        if(sp_missing_closed_days[randomday] != np.nan):
            sp_missing_closed_days[randomday] = np.nan
            holidaycounter = holidaycounter + 1

    plt.figure(figsize=(9,4))
    plt.plot(sp_missing_closed_days)
    plt.legend([scenario_name],
               loc='upper left')
    plt.hlines(y=100,xmin=0,xmax=1460,
               linestyle='--',color='k')
    plt.show()

            
    dict = {'stockprice': stockprice, 'stockprice_missing_closed_days': sp_missing_closed_days} 
    df = pd.DataFrame(dict)
    output_loc = 'synthetic_data/brownian_scenarios/' + scenario_name
    df.to_csv(output_loc)
    return stockprice



def create_sinus_plus_brownian_noise_scenario(missing_percentage, periodparameter, scenario_name):
    """
    create a scenario of sinus + a brownian noise 
    Default is three years, 1460 days, this can be changed in the stockprice function.
    """
    # The stockprice is a brownian motion 7 days a week all year. 
    total_days = 1460
    noise = b.gen_normal(1460)
    days = np.arange(total_days)
    sinus = np.sin(days/periodparameter)
    noisy_sin = sinus+1.2*noise

    

    # The sp_no_missing_values will take the allyear stockprice and remove the weekends + 7 days a year random holidays.
    sin_missing_values = copy.deepcopy(noisy_sin)


    missing_counter = 0

    while(missing_counter < missing_percentage*(total_days)):
        randomday = random.randint(0, total_days)
        if(sin_missing_values[randomday] != np.nan):
            sin_missing_values[randomday] = np.nan
            missing_counter = missing_counter + 1

    plt.figure(figsize=(9,4))
    plt.plot(sin_missing_values)
    plt.legend([scenario_name],
               loc='upper left')
    plt.show()

            
    dict = {'noisy_sin': noisy_sin, 'noisy_sin_missing_values': sin_missing_values} 
    df = pd.DataFrame(dict)
    output_loc = 'synthetic_data/sinus_scenarios/' + scenario_name
    df.to_csv(output_loc)
    return noisy_sin

def create_sinus_scenario(missing_percentage, periodparameter, scenario_name):
    """
    create a scenario of sinus + a brownian noise 
    Default is three years, 1460 days, this can be changed in the stockprice function.git 
    """
    # The stockprice is a brownian motion 7 days a week all year. 
    total_days = 1460
    days = np.arange(total_days)
    sinus = np.sin(days/periodparameter)

    # The sp_no_missing_values will take the allyear stockprice and remove the weekends + 7 days a year random holidays.
    sin_missing_values = copy.deepcopy(sinus)


    missing_counter = 0

    while(missing_counter < missing_percentage*(total_days)):
        randomday = random.randint(0, total_days)
        if(sin_missing_values[randomday] != np.nan):
            sin_missing_values[randomday] = np.nan
            missing_counter = missing_counter + 1

    plt.figure(figsize=(9,4))
    plt.plot(sin_missing_values)
    plt.legend([scenario_name],
               loc='upper left')
    plt.show()

            
    dict = {'sinus': sinus, 'noisy_sin_missing_values': sin_missing_values} 
    df = pd.DataFrame(dict)
    output_loc = 'synthetic_data/sinus_scenarios/' + scenario_name
    df.to_csv(output_loc)
    return noisy_sin

def create_sinus_plus_stochastic_noise_scenario(missing_percentage, periodparameter, scenario_name):
    """
    create a scenario of sinus + a brownian noise 
    Default is three years, 1460 days, this can be changed in the stockprice function.
    """
    # The stockprice is a brownian motion 7 days a week all year. 
    total_days = 1460
    noise = s.gen_normal(1460)
    days = np.arange(total_days)
    sinus = np.sin(days/periodparameter)
    noisy_sin = sinus+0.15*noise

    

    # The sp_no_missing_values will take the allyear stockprice and remove the weekends + 7 days a year random holidays.
    sin_missing_values = copy.deepcopy(noisy_sin)


    missing_counter = 0

    while(missing_counter < missing_percentage*(total_days)):
        randomday = random.randint(0, total_days)
        if(sin_missing_values[randomday] != np.nan):
            sin_missing_values[randomday] = np.nan
            missing_counter = missing_counter + 1

    plt.figure(figsize=(9,4))
    plt.plot(sin_missing_values)
    plt.legend([scenario_name],
               loc='upper left')
    plt.show()

            
    dict = {'noisy_sin': noisy_sin, 'noisy_sin_missing_values': sin_missing_values} 
    df = pd.DataFrame(dict)
    output_loc = 'synthetic_data/sinus_scenarios/' + scenario_name
    df.to_csv(output_loc)
    return noisy_sin

# Scenario upward, sideways and downward, all three are made with the same settings and a few experiments. mu_021_sig_065
# create_stock_price_scenario(mu=0.21,sigma=0.65, scenario_name= 'noisy_sin_missing_10_065.csv')

# create_sinus_plus_brownian_noise_scenario(missing_percentage= 0.20,period = 40, scenario_name= 'noisy_sin_period40_missing20.csv')

# create_sinus_scenario(missing_percentage= 0.20,periodparameter = 2, scenario_name= 'small_sin_period2_missing20.csv')

create_sinus_plus_stochastic_noise_scenario(missing_percentage= 0.20,periodparameter = 10, scenario_name= 'stochastic015_sin_period63_missing20.csv')






