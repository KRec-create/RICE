import numpy as np
from scipy import stats
from collections import defaultdict


# 读取txt文件并统计每个用户的交互数
def read_data(file_path):
    user_interactions = defaultdict(int)
    with open(file_path, 'r') as file:
        data = file.readlines()
    for line in data:
        user_id = line.strip().split()[0]
        user_interactions[user_id] += 1
    return user_interactions


# 计算统计信息
def calculate_statistics(interactions_dict):
    interactions = list(interactions_dict.values())
    min_interactions = np.min(interactions)
    max_interactions = np.max(interactions)
    mean_interactions = np.mean(interactions)
    median_interactions = np.median(interactions)

    # 找到交互数最少和最多的用户
    min_users = [user for user, count in interactions_dict.items() if count == min_interactions]
    max_users = [user for user, count in interactions_dict.items() if count == max_interactions]

    return {
        "min": min_interactions,
        "max": max_interactions,
        "mean": mean_interactions,
        "median": median_interactions,
        "min_users": min_users,
        "max_users": max_users
    }


# 主函数
def main():
    file_path = 'train.txt'  # 替换为你的txt文件路径
    interactions_dict = read_data(file_path)
    statistics = calculate_statistics(interactions_dict)

    print("用户交互数最少的值:", statistics["min"])
    print("交互数最少的用户:", statistics["min_users"])
    print("用户交互数最多的值:", statistics["max"])
    print("交互数最多的用户:", statistics["max_users"])
    print("用户交互平均数:", statistics["mean"])
    print("用户交互中位数:", statistics["median"])


if __name__ == "__main__":
    main()
