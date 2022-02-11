import pandas as pd

def my_decorater(func, sep_word: str = "|"):
    def wrapper(*args, **kwargs):
        print(sep_word, end="")
        func(*args, **kwargs, sep="|", end="|\n")
    return wrapper

@my_decorater
def my_print(*args, **kwargs):
    print(*args, **kwargs)

###

df = pd.read_csv("result.csv", encoding="utf-8")

datasets = df["ds_type"].unique()
epsilons = sorted(list(df["eps"].unique()))
attack_types = list(df["attack_type"].unique())

for ds_type in datasets:
    my_print(*(["epsilon"] + attack_types))
    my_print(*([":---:"] * (len(attack_types) + 1)))

    for eps in epsilons:
        foo = df.loc[(df["ds_type"] == ds_type) & (df["eps"] == eps), "acc_adv"].values
        my_print(*([eps] + [f"{i:.2f}" for i in foo]))

