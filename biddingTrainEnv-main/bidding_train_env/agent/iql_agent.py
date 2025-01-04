import numpy as np
import torch
import pickle

from bidding_train_env.agent.base_agent import BaseAgent
from bidding_train_env.baseline.iql.iql import IQL


class IqlAgent(BaseAgent):
    """
    IQL方法训练的出价智能体
    """

    def __init__(self, budget=100, name="Iql-PlayerAgent", cpa=2,category=0):
        super().__init__(budget, name, cpa,category)

        # 2024_2_9 16*3维度  45000stepnum bs5000  向量型
        path_643_16_3="IQLtest_643_1-3dim"
        # 2024_2_9 16维度  50000stepnum bs2000
        path_643_16="IQLtest_643"

        path_003="IQLtest_640_1-3dim"
        path_004="IQLtest_641_1-3dim"
        path_005="IQLtest_642_1-3dim"
        path_006="IQLtest_001"

        self.model_643_16_3 = IQL(dim_obs=48)
        self.model_643_16_3.load_net("./saved_model/{}".format(path_643_16_3))

        self.model_643_16 = IQL(dim_obs=16)
        self.model_643_16.load_net("./saved_model/{}".format(path_643_16))

        self.model_003 = IQL(dim_obs=48)
        self.model_003.load_net("./saved_model/{}".format(path_003))

        self.model_004 = IQL(dim_obs=48)
        self.model_004.load_net("./saved_model/{}".format(path_004))

        self.model_005 = IQL(dim_obs=48)
        self.model_005.load_net("./saved_model/{}".format(path_005))

        self.model_006 = IQL(dim_obs=48)
        self.model_006.load_net("./saved_model/{}".format(path_006))

        # Load and apply normalization to test_state
        with open('./saved_model/{}/normalize_dict.pkl'.format(path_643_16_3), 'rb') as file:
            self.normalize_dict_643_16_3 = pickle.load(file)
        with open('./saved_model/{}/normalize_dict.pkl'.format(path_643_16), 'rb') as file:
            self.normalize_dict_643_16 = pickle.load(file)
        #
        with open('./saved_model/{}/normalize_dict.pkl'.format(path_003), 'rb') as file:
            self.normalize_dict_003 = pickle.load(file)

        with open('./saved_model/{}/normalize_dict.pkl'.format(path_004), 'rb') as file:
            self.normalize_dict_004 = pickle.load(file)
        with open('./saved_model/{}/normalize_dict.pkl'.format(path_005), 'rb') as file:
            self.normalize_dict_005 = pickle.load(file)
        with open('./saved_model/{}/normalize_dict.pkl'.format(path_006), 'rb') as file:
            self.normalize_dict_006 = pickle.load(file)

    def reset(self):
        self.remaining_budget = self.budget

    def action(self, tick_index, budget, remaining_budget, pv_values, history_pv_values, history_bid,
               history_status, history_reward, history_market_price):

        time_left = (24 - tick_index) / 24
        budget_left = remaining_budget / budget if budget > 0 else 0

        # 计算历史状态的均值
        historical_status_mean = np.mean([np.mean(status) for status in history_status]) if history_status else 0
        # 计算历史回报的均值
        historical_reward_mean = np.mean([np.mean(reward) for reward in history_reward]) if history_reward else 0
        # 计算历史市场价格的均值
        historical_market_price_mean = np.mean(
            [np.mean(price) for price in history_market_price]) if history_market_price else 0
        # 计算历史pvValue的均值
        historical_pv_values_mean = np.mean([np.mean(value) for value in history_pv_values]) if history_pv_values else 0
        # 历史调控单元的出价均值
        historical_bid_mean = np.mean([np.mean(bid) for bid in history_bid]) if history_bid else 0

        # remainingBudget_c_state_features=remaining_budget/historical_bid_mean if historical_pv_values_mean else 0

        # Calculate mean of the last three ticks for different history data
        def mean_of_last_n_elements(history, n, last_number=3):
            last_three_data = history[max(0, n - last_number):n]
            if len(last_three_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_three_data])
        def max_of_last_n_elements(history, n, last_number=3):
            last_three_data = history[max(0, n - last_number):n]
            if len(last_three_data) == 0:
                return 0
            else:
                return np.max([np.mean(data) for data in last_three_data])

        last_3_status_mean = mean_of_last_n_elements(history_status, tick_index)
        last_3_reward_mean = mean_of_last_n_elements(history_reward, tick_index)
        last_3_market_price_mean = mean_of_last_n_elements(history_market_price, tick_index)
        last_3_pv_values_mean = mean_of_last_n_elements(history_pv_values, tick_index)
        last_3_bid_mean = mean_of_last_n_elements(history_bid, tick_index)
        # # 7 hours

        current_pv_values_mean = np.mean(pv_values)
        current_pv_num = len(pv_values)

        historical_pv_num_total = sum(len(bids) for bids in history_bid) if history_bid else 0
        last_3_pv_num_total = sum(
            len(history_bid[i]) for i in range(max(0, tick_index - 3), tick_index)) if history_bid else 0
        last_7_pv_num_total = sum(
            len(history_bid[i]) for i in range(max(0, tick_index - 7), tick_index)) if history_bid else 0
        # last_12_pv_num_total = sum(
        #     len(history_bid[i]) for i in range(max(0, tick_index - 12), tick_index)) if history_bid else 0
        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_3_bid_mean,
            historical_market_price_mean, historical_pv_values_mean, historical_reward_mean,
            historical_status_mean, last_3_market_price_mean, last_3_pv_values_mean,
            last_3_reward_mean, last_3_status_mean, current_pv_values_mean,
            current_pv_num, last_3_pv_num_total, historical_pv_num_total,
            # remainingBudget_c_state_features
            # last_7_bid_mean,last_7_market_price_mean,last_7_pv_values_mean,last_7_reward_mean,last_7_status_mean,last_7_pv_num_total,
            # last_12_bid_mean, last_12_market_price_mean, last_12_pv_values_mean, last_12_reward_mean, last_12_status_mean,last_12_pv_num_total,
        ])
        # test_state1=test_state
        # test_state2=test_state
        # test_state3=test_state
        # test_state4=test_state
        # test_state5=test_state
        # test_state6=test_state

        test_state1 = np.array([
            time_left, budget_left, historical_bid_mean, last_3_bid_mean,
            historical_market_price_mean, historical_pv_values_mean, historical_reward_mean,
            historical_status_mean, last_3_market_price_mean, last_3_pv_values_mean,
            last_3_reward_mean, last_3_status_mean, current_pv_values_mean,
            current_pv_num, last_3_pv_num_total, historical_pv_num_total,
        ])
        test_state2 = np.array([
            time_left, budget_left, historical_bid_mean, last_3_bid_mean,
            historical_market_price_mean, historical_pv_values_mean, historical_reward_mean,
            historical_status_mean, last_3_market_price_mean, last_3_pv_values_mean,
            last_3_reward_mean, last_3_status_mean, current_pv_values_mean,
            current_pv_num, last_3_pv_num_total, historical_pv_num_total,
        ])
        test_state3 = np.array([
            time_left, budget_left, historical_bid_mean, last_3_bid_mean,
            historical_market_price_mean, historical_pv_values_mean, historical_reward_mean,
            historical_status_mean, last_3_market_price_mean, last_3_pv_values_mean,
            last_3_reward_mean, last_3_status_mean, current_pv_values_mean,
            current_pv_num, last_3_pv_num_total, historical_pv_num_total,
        ])
        test_state4 = np.array([
            time_left, budget_left, historical_bid_mean, last_3_bid_mean,
            historical_market_price_mean, historical_pv_values_mean, historical_reward_mean,
            historical_status_mean, last_3_market_price_mean, last_3_pv_values_mean,
            last_3_reward_mean, last_3_status_mean, current_pv_values_mean,
            current_pv_num, last_3_pv_num_total, historical_pv_num_total,
        ])
        test_state5 = np.array([
            time_left, budget_left, historical_bid_mean, last_3_bid_mean,
            historical_market_price_mean, historical_pv_values_mean, historical_reward_mean,
            historical_status_mean, last_3_market_price_mean, last_3_pv_values_mean,
            last_3_reward_mean, last_3_status_mean, current_pv_values_mean,
            current_pv_num, last_3_pv_num_total, historical_pv_num_total,
        ])
        test_state6 = np.array([
            time_left, budget_left, historical_bid_mean, last_3_bid_mean,
            historical_market_price_mean, historical_pv_values_mean, historical_reward_mean,
            historical_status_mean, last_3_market_price_mean, last_3_pv_values_mean,
            last_3_reward_mean, last_3_status_mean, current_pv_values_mean,
            current_pv_num, last_3_pv_num_total, historical_pv_num_total,
        ])



        def normalize(value, min_value, max_value):
            return (value - min_value) / (max_value - min_value) if max_value > min_value else 0

        #开始融合技！！！
        for key, value in self.normalize_dict_643_16.items():
            test_state1[key] = normalize(test_state1[key], value["min"], value["max"])
        test_state1 = torch.tensor(test_state1, dtype=torch.float)
        alpha1 = self.model_643_16.take_actions(test_state1)
        #2
        test_state2 = np.array([element for element in test_state2 for _ in range(3)])
        for key, value in self.normalize_dict_643_16_3.items():
            test_state2[key] = normalize(test_state2[key], value["min"], value["max"])
        test_state2 = torch.tensor(test_state2, dtype=torch.float)
        alpha2 = self.model_643_16_3.take_actions(test_state2)
        #3
        test_state3 = np.array([element for element in test_state3 for _ in range(3)])
        for key, value in self.normalize_dict_003.items():
            test_state3[key] = normalize(test_state3[key], value["min"], value["max"])
        test_state3 = torch.tensor(test_state3, dtype=torch.float)
        alpha3 = self.model_003.take_actions(test_state3)

        test_state4 = np.array([element for element in test_state4 for _ in range(3)])
        for key, value in self.normalize_dict_004.items():
            test_state4[key] = normalize(test_state4[key], value["min"], value["max"])
        test_state4 = torch.tensor(test_state4, dtype=torch.float)
        alpha4 = self.model_004.take_actions(test_state4)

        test_state5 = np.array([element for element in test_state5 for _ in range(3)])
        for key, value in self.normalize_dict_005.items():
            test_state5[key] = normalize(test_state5[key], value["min"], value["max"])
        test_state5 = torch.tensor(test_state5, dtype=torch.float)
        alpha5 = self.model_005.take_actions(test_state5)

        test_state6 = np.array([element for element in test_state6 for _ in range(3)])
        for key, value in self.normalize_dict_006.items():
            test_state6[key] = normalize(test_state6[key], value["min"], value["max"])
        test_state6 = torch.tensor(test_state6, dtype=torch.float)
        alpha6 = self.model_006.take_actions(test_state6)

        # 策略2：去除最大最小之后 生成一个
        combined_array = np.concatenate((alpha1, alpha2, alpha3,alpha4,alpha5,alpha6))
        new_alpha=np.mean(combined_array)

        bids = new_alpha * pv_values
        # print(bids)

        return bids
