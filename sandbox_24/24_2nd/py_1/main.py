import sys
import py_lib

if __name__ == '__main__':
    print(sys.path)

    # 報酬関数の設定ファイルの読み込み.
    path = R"py_1\py_lib\test\test.yaml"
    reward_function = py_lib.build_reward_function(path)
    reward_function.reward([1, 2, 3])
