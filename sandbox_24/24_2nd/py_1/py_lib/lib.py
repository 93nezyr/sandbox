import abc
import yaml

class RewardFunction(metaclass = abc.ABCMeta):
    """
    # About

    報酬関数のインターフェース.

    ## Methods

    - `reward(reward: list) -> list` - 報酬関数の計算を行う.
    """
    @abc.abstractmethod
    def reward(reward: list) -> list:
        raise NotImplementedError("Method 'reward' must be implemented.")

def build_reward_function(path: str) -> RewardFunction:
    """
    # About

    報酬関数の設定ファイルを読み込み、報酬関数を生成する.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    reward_setting = data["reward_setting"]
    if reward_setting["id"] == "LinearAnnealing":
        config = reward_setting["config"]
        return LinearAnnealingRewardFunction(config["start_step"], config["end_step"], config["start_p"], config["end_p"])
    else:
        raise NotImplementedError("Reward function is not implemented.")

class LinearAnnealingRewardFunction(RewardFunction):
    """
    # About

    線形アニーリングを行う報酬関数の設定.

    ## Fields

    - `start_step: int` - アニーリング開始ステップ.
    - `end_step: int` - アニーリング終了ステップ.
    - `start_p: float` - 開始時の報酬係数.
    - `end_p: float` - 終了時の報酬係数.

    """

    def __init__(self, start_step: int, end_step: int, start_p: float, end_p: float):
        """
        # About

        コンストラクタ.
        """
        # 設定値の受け取り.
        self.__start_step = start_step
        self.__end_step = end_step
        self.__start_p = start_p
        self.__end_p = end_p

        # 内部情報の初期化.
        self.__count_opt = 0

    def reward(self, reward: list) -> list:
        print(reward)
        pass

if __name__ == '__main__':
    path = R"py_1\py_lib\test\test.yaml"
    print(path)

    reward_function = build_reward_function(path)

    reward_function.reward([0.7, 0.7, 0.7])

    print("Goodnight, World!")