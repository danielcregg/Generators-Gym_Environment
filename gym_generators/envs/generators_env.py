import gym
from gym import spaces
import numpy as np
import pandas as pd  # import pandas to use pandas DataFrame
from math import sin, exp


class GeneratorsEnv(gym.Env):
    """Class for Generators environment"""
    metadata = {'render.modes': ['human']}

    # Set constants
    M = 24      # M = number of hours in day = Number of States
    N = 10      # N = number of generators (1 slack, 9 agent controlled)
    E = 10      # Emissions scaling factor
    Wc = 0.225  # Cost weight used for linear scalarisation
    We = 0.275  # Emissions weight used for linear scalarisation
    Wp = 0.5    # Power weight used for linear scalarisation
    C = 10E6    # C is the violation constant
    states2 = np.zeros([240,2])

    # DataFrame created with generator data in the form of list of tuples
    gen_chars = pd.DataFrame(
        [(150, 470, 786.7988, 38.5397, 0.1524, 450, 0.041, 103.3908, -2.4444,   0.0312, 0.5035, 0.0207, 80, 80),
         (135, 470, 451.3251, 46.1591, 0.1058, 600, 0.036, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80, 80),
         (73, 340, 1049.9977, 40.3965, 0.0280, 320, 0.028, 300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 80, 80),
         (60, 300, 1243.5311, 38.3055, 0.0354, 260, 0.052, 300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 50, 50),
         (73, 243, 1658.5696, 36.3278, 0.0211, 280, 0.063, 320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50, 50),
         (57, 160, 1356.6592, 38.2704, 0.0179, 310, 0.048, 320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50, 50),
         (20, 130, 1450.7045, 36.5104, 0.0121, 300, 0.086, 330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30, 30),
         (47, 120, 1450.7045, 36.5104, 0.0121, 340, 0.082, 330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30, 30),
         (20, 80, 1455.6056, 39.5804, 0.1090, 270, 0.098, 350.0056, -3.9524, 0.0465, 0.5475, 0.0234, 30, 30),
         (10, 55, 1469.4026, 40.5407, 0.1295, 380, 0.094, 360.0012, -3.9864, 0.0470, 0.5475, 0.0234, 30, 30)],
        columns=["p_min_i", "p_max_i", "a_i", "b_i", "c_i", "d_i", "e_i", "alpha_i", "beta_i", "gamma_i", "eta_i", "delta_i", "ur_i", "dr_i"],
        index=["unit1", "unit2", "unit3", "unit4", "unit5", "unit6", "unit7", "unit8", "unit9", "unit10"])

    # B is the matrix of transmission line loss coefficients
    # data in the form of list of tuples
    B = pd.DataFrame(
        [(0.000049, 0.000014, 0.000015, 0.000015, 0.000016, 0.000017, 0.000017,  0.000018, 0.000019, 0.000020),
         (0.000014, 0.000045, 0.000016, 0.000016, 0.000017, 0.000015, 0.000015, 0.000016, 0.000018, 0.000018),
         (0.000015, 0.000016, 0.000039, 0.000010, 0.000012, 0.000012, 0.000014, 0.000014, 0.000016, 0.000016),
         (0.000015, 0.000016, 0.000010, 0.000040, 0.000014, 0.000010, 0.000011, 0.000012, 0.000014, 0.000015),
         (0.000016, 0.000017, 0.000012, 0.000014, 0.000035, 0.000011, 0.000013, 0.000013, 0.000015, 0.000016),
         (0.000017, 0.000015, 0.000012, 0.000010, 0.000011, 0.000036, 0.000012, 0.000012, 0.000014, 0.000015),
         (0.000017, 0.000015, 0.000014, 0.000011, 0.000013, 0.000012, 0.000038, 0.000016, 0.000016, 0.000018),
         (0.000018, 0.000016, 0.000014, 0.000012, 0.000013, 0.000012, 0.000016, 0.000040, 0.000015, 0.000016),
         (0.000019, 0.000018, 0.000016, 0.000014, 0.000015, 0.000014, 0.000016, 0.000015, 0.000042, 0.000019),
         (0.000020, 0.000018, 0.000016, 0.000015, 0.000016, 0.000015, 0.000018, 0.000016, 0.000019, 0.000044)])

    # self.state = np.array([0.0,74.0,148.0,148.0,74.0,148.0,74.0,74.0,148.0,98.0,84.0,44.0,-78.0,-148.0,-148.0,-222.0,
    # -74.0,148.0,148.0,196.0,-48.0,-296.0,-296.0,-148.0])

    # data in the form of list of tuples
    hour_power_demand = pd.DataFrame([1036, 1110, 1258, 1406, 1480, 1628, 1702,    1776, 1924, 2022, 2106, 2150, 2072, 1924, 1776, 1554, 1480, 1628,          1776, 1972, 1924, 1628, 1332, 1184],
        columns=["p_d"],
        index=["hour1", "hour2", "hour3", "hour4", "hour5", "hour6", "hour7",         "hour8", "hour9", "hour10", "hour11", "hour12", "hour13",              "hour14", "hour15", "hour16", "hour17", "hour18", "hour19",            "hour20", "hour21","hour22", "hour23", "hour24"])

    hour_power_demand_diff = hour_power_demand.diff().fillna(0.)  # Get diff values and fill NaN with 0.0

    hour_power_demand_diff.rename(columns={'p_d':'delta_p_d'}, inplace=True)

    states = hour_power_demand_diff.assign(p_n_m_prev=0.)  # Add new column and set all rows to 0.0_

    # data in the form of list of tuples
    p_n_m = pd.DataFrame(0., columns=["hour1", "hour2", "hour3", "hour4", "hour5", "hour6", "hour7", "hour8", "hour9", "hour10", "hour11", "hour12", "hour13", "hour14", "hour15", "hour16", "hour17", "hour18", "hour19", "hour20", "hour21","hour22", "hour23", "hour24"], index=["unit1", "unit2", "unit3", "unit4", "unit5", "unit6", "unit7", "unit8", "unit9", "unit10"])
        
    def __init__(self):
        print("Generators Environment Initialising...")
        self.states2.fill(0.)
        state = 0
        for n in self.gen_chars.index:
            self.states2[state] = [0. , self.unit_get_random_power(n)]
            state += self.M
        #print(self.states2)
        
        self.m = self.states.index.get_loc("hour1") # m = current Hour = 0
        self.state = self.states2[self.m]
        self.active_unit = "unit1"
        self.action_space = spaces.Discrete(101)  # Set with 101 elements {0, 1, 2 ... 100}
        self.states.iloc[self.m, self.states.columns.get_loc("p_n_m_prev")] = self.gen_chars.loc[self.active_unit, "p_min_i"]

        self.state = self.states.iloc[self.m, : ]
        #self.state = self.states.loc["hour1", : ]
        # for unit in self.p_n_m.index:
        #     # Start all generators at min power value
        #     #self.p_n_m["hour1"][unit] = self.gen_chars["p_min_i"][unit]
        #     #Start all generators at random power value within discrete range
        #     self.p_n_m["hour1"][unit] = self.gen_chars["p_min_i"][unit] + ((self.gen_chars["p_max_i"][unit] - self.gen_chars["p_min_i"][unit]) * (np.random.randint(0, 101)/100))
        # print(self.p_n_m)
        self.reward = 0
        self.done = 0
        self.add = [0, 0]
        print("Generators Environment Initialised...")

    def show_unit_characteristics(self, unit):
        print("Unit Name:", unit)
        print(self.gen_chars.loc[unit, :])

    def unit_get_random_power(self, n):
        if type(n) == str:
            assert n in self.p_n_m.index, "%r (%s) invalid unit" % (n,type(n))
            p_n_m = self.gen_chars["p_min_i"][n] + ((self.gen_chars["p_max_i"][n] - self.gen_chars["p_min_i"][n]) * (np.random.randint(0, 101)/100))
        elif type(n) == int:
            assert  type(n) == int and n > 0 and n <= self.N, "%r (%s) invalid unit" % (n,type(n))
            n = n - 1 # indexing start at 0 so pushing it up for humans
            p_min_i_col_loc = self.gen_chars.columns.get_loc("p_min_i") # = 0
            p_max_i_col_loc = self.gen_chars.columns.get_loc("p_max_i") # = 1
            p_n_m = self.gen_chars.iloc[n, p_min_i_col_loc] + ((self.gen_chars.iloc[n, p_max_i_col_loc] - self.gen_chars.iloc[n, p_min_i_col_loc]) * (np.random.randint(0, 101)/100))
        return p_n_m


    def unit_set_power(self, n, m, action):
        #assert n in self.p_n_m.index, "%r (%s) invalid unit" % (n,type(n))
        #assert m in self.hour_power_demand.index, "%r (%s) invalid hour" % (m,type(m))
        #assert self.action_space.contains(action), "%r (%s) invalid" % (action,type(action))
        self.p_n_m[m][n] = self.gen_chars["p_min_i"][n] + ((self.gen_chars["p_max_i"][n] - self.gen_chars["p_min_i"][n]) * (action/100))
        return self.p_n_m[m][n]

    def find_p_d_m(self, m):  # Input m = the hour
        if type(m) == str:
            assert m in self.hour_power_demand.index, "%r (%s) invalid hour" % (m,type(m))
            p_d_m = self.hour_power_demand.loc[m, "p_d"]
        elif type(m) == int:
            assert  type(m) == int and m > 0 and m < self.M, "%r (%s) invalid hour" % (m,type(m))
            p_d_m = self.hour_power_demand.iloc[m - 1, 0]
        return p_d_m
    
    # Find the power loss of a generator unit 
    # Need to work on this after i figure out rewards system
    def find_p_l_m(self, n, m):  # Input m = the hour
        if type(m) == str:
            assert m in self.hour_power_demand.index, "%r (%s) invalid hour" % (m,type(m))
            assert n in self.gen_chars.index, "%r (%s) invalid hour" % (n,type(n))
            p_l_m = 0 # 
        elif type(m) == int:
            assert  type(m) == int and m > 0 and m < self.M, "%r (%s) invalid hour" % (m,type(m))
            assert  type(n) == int and n > 0 and n < self.N, "%r (%s) invalid hour" % (n,type(n))
            p_l_m = 0 # 
        return p_l_m
    
    # Find the power output of a generator unit - Adgent chooes this ... 
    # Need to work on this after i figure out rewards system
    def find_p_n_m(self, n, m):  # Input n = generator unit number
        if type(n) == str:
            assert m in self.hour_power_demand.index, "%r (%s) invalid hour" % (m,type(m))
            assert n in self.gen_chars.index, "%r (%s) invalid hour" % (n,type(n))
            p_n_m = 0 # agent to determine
        elif type(m) == int:
            assert  type(m) == int and m > 0 and m < self.M, "%r (%s) invalid hour" % (m,type(m))
            assert  type(n) == int and n > 0 and n < self.N, "%r (%s) invalid hour" % (n,type(n))
            p_n_m = 0
        return p_n_m

    def f_c_l(self, n, m):
        # Get required variables
        a_n = self.gen_chars.loc[n, "a_i"]
        b_n = self.gen_chars.loc[n, "b_i"]
        c_n = self.gen_chars.loc[n, "c_i"]
        d_n = self.gen_chars.loc[n, "d_i"]
        e_n = self.gen_chars.loc[n, "e_i"]
        p_n_m = self.p_n_m.loc[n][m]
        p_min_n = self.gen_chars.loc[n, "p_min_i"]
        return round(a_n + (b_n * p_n_m) + c_n * (p_n_m ** 2) + abs(d_n * sin(e_n * (p_min_n - p_n_m))), 2)

    def f_c_g(self, m):
        global_cost = 0
        for n in self.gen_chars.index:
            global_cost += self.f_c_l(n, m)
        return round(global_cost, 2)

    def f_e_l(self, n, m):
        alpha_n = self.gen_chars.loc[n, "alpha_i"]
        beta_n = self.gen_chars.loc[n, "beta_i"]
        gamma_n = self.gen_chars.loc[n, "gamma_i"]
        eta_n = self.gen_chars.loc[n, "eta_i"]
        delta_n = self.gen_chars.loc[n, "delta_i"]
        p_n_m = self.p_n_m.loc[n][m]
        return self.E * (alpha_n + (beta_n * p_n_m) + gamma_n * (p_n_m ** 2) + eta_n * exp(delta_n * p_n_m))

    def f_e_g(self, m):
        global_emissions = 0
        for n in self.gen_chars.index:
            global_emissions += self.f_e_l(n, m)
        return global_emissions

    # Find the Power output of the slack generator at a given hour m
    def find_p_1_m(self, m):
        print("This is the Slack power ouput at given hour")
        p_d_m = find_p_d_m(m)
        p_l_m = find_p_l_m(m)
        for n in range (2..self.N):
            p_n_m += find_p_d_m(m)
        #p_m = p_d_m + p_l_m - (p_n_m_non_slack_total)
        #return p_m

    def f_p_g(self, m):
        print("This is the Global Penalty Function")
        # Slack generator is unit 1
        p1m = self.find_p_n_m("unit1", m)
        p1m_prev = self.states2[0][1] 
        p1min = self.gen_chars.loc["unit1", "p_min_i"]
        p1max = self.gen_chars.loc["unit1", "p_max_i"]
        ur1 = self.gen_chars.loc["unit1", "ur_i"]
        dr1 = self.gen_chars.loc["unit1", "dr_i"]
        delta1 = 0 # delta = 0 if no violation in constraint else 1 if the constraint is violated
        delta2 = 0

        if p1m > p1max:
            h1 = p1m - p1max
            delta1 = 1 # delta = 0 if no violation in constraint else 1 if the constraint is violated
        elif p1m < p1min: 
            h1 = p1min - p1m
            delta1 = 1 # delta = 0 if no violation in constraint else 1 if the constraint is violated
        else:
            h1 = 0
            delta1 = 0 # delta = 0 if no violation in constraint else 1 if the constraint is violated

        if p1m - p1m_prev > ur1:
            h2 = (p1m - p1m_prev) - ur1
            delta2 = 1
        elif p1m - p1m_prev < -dr1: 
            h2 = (p1m - p1m_prev) + dr1
            delta2 = 1
        else:
            h2 = 0
            delta2 = 0
        
        print(delta1,delta2)

        return self.C * (abs(h1 + 1) * delta1) + self.C * (abs(h2 + 1) * delta2)

    def find_constrained_action_space(self, unit, p_n_m_prev):
        print("This is the restrain action space Function")
        p_min_n = self.gen_chars.loc[unit, "p_min_i"]
        p_max_n = self.gen_chars.loc[unit, "p_max_i"]
        dr_n = self.gen_chars.loc[unit, "dr_i"]
        ur_n = self.gen_chars.loc[unit, "ur_i"]
        
        #unit_p_range = p_max_n - p_min_n
        # 101 possible actions but 100 power segments in power range
        num_action_p_segments = self.action_space.n - 1
        action_p_per_segment = (p_max_n - p_min_n)/(num_action_p_segments)
        current_action_number = int(round((p_n_m_prev - p_min_n)/(action_p_per_segment)))

        print("curren action:",current_action_number)
        
        max_actions_down = int(round(dr_n/action_p_per_segment))
        max_actions_up = int(round(ur_n/action_p_per_segment))

        if max_actions_down > current_action_number:
            max_actions_down = current_action_number
        
        if max_actions_up + current_action_number > num_action_p_segments:
            max_actions_up = num_action_p_segments - current_action_number
        
        print(max_actions_down, max_actions_up, 1)
        
        
        possible_actions = np.arange(current_action_number - max_actions_down, current_action_number + max_actions_up + 1)
        print(possible_actions)
        return possible_actions


    def step(self, action):
        print("action",action)
        print("hour:", self.hour_power_demand.index[self.m])
        print("p_n",self.unit_set_power("unit1", self.hour_power_demand.index[self.m], action))
        #print(self.p_n_m.iloc[0,self.m])
        #self.states2[self.m]
        #ur_i = self.gen_chars.loc[self.active_unit, "ur_i"]
        #dr_i = self.gen_chars.loc[self.active_unit, "dr_i"]
        # p_min_n = self.gen_chars.loc[self.active_unit, "p_min_i"]
        # p_max_n = self.gen_chars.loc[self.active_unit, "p_max_i"]
        # p_n = p_min_n + action * ((p_max_n - p_min_n) / self.action_space.n)
    

        rc=self.f_c_l("unit1", self.hour_power_demand.index[self.m])  # to be calculated
        re=self.f_e_l("unit1", self.hour_power_demand.index[self.m])  # to be calculated
        #rp=  # to be calculated
        self.reward = -((self.Wc*rc)*(self.We*re))#*(self.Wp*rp)
        
        if self.m == 23:
            self.done = 1
            print("Episode Complete.")
            return [self.state, self.reward, self.done, self.add]
            
        # Move to next hour
        self.m += 1
        # Update state
        delta_p_d_n = self.hour_power_demand_diff.iloc[self.m , 0]
        print("sdfsfsdf", delta_p_d_n)
       
        print(self.states2[self.m - 1][1])
        self.states2[self.m] = [delta_p_d_n, self.p_n_m.iloc[0,self.m - 2]]
        
        self.state = self.states2[self.m]

        # self.states.iloc[self.m, self.states.columns.get_loc('p_n_m_prev')] = p_n

        #print("Episode in progress...")
        return [self.state, self.reward, self.done, self.add]

    def render(self):
        print("This is a render of the Generators Environment")

    def reset(self):
        print("Generators Environment Reset")
        # self.state = hour_power_demand.diff()
        self.state = []
        #self.counter = 0
        self.done = 0
        self.add = [0, 0]
        self.reward = 0
        # space = spaces.Discrete(24) # Set with 8 elements {0, 1, 2, ..., 23}
        # self.action_space = spaces.Tuple((spaces.Discrete(101)))

#gens1 = GeneratorsEnv()
#for x in range(0, gens1.M):
#    print(gens1.step(gens1.action_space.sample()))  # Take random action




#print(gens1.states)
#print(gens1.unit_get_random_power(1))
#print(gens1.gen_chars.columns.get_loc("p_min_i"))
#gens1.show_unit_characteristics("unit1")

#for hour in range(1, gens1.M):
 #   print(gens1.find_p_d_m(hour))

#unit = "unit1"
#p_n_m = np.random.uniform(low=gens1.gen_chars.loc[unit, "p_min_i"],high=gens1.gen_chars.loc[unit, "p_max_i"])

#print(gens1.f_c_l("unit1", "hour1"))
#print(gens1.f_c_g("hour1"))
#print(gens1.f_e_l("unit1", "hour1"))
#print(gens1.f_e_g("hour1"))


#gens1.find_constrained_action_space("unit1", 470)

#print(gens1.f_p_g("hour1"))


