import pandas as pd


def filter_kinetics_class(data: pd.DataFrame, class_name: str, num_samples: int = None) -> pd.DataFrame:
    df = data[data.label.str.contains(class_name)]
    new_df = pd.DataFrame()
    label_values = df['label'].unique()
    if num_samples is None:
        df.to_csv(f"../data/kinetics_700/{class_name}.csv")
        return df

    for label in label_values:
        label_data = df[df['label'] == label].sample(num_samples)
        new_df = pd.concat([new_df, label_data])

    new_df.to_csv(f"../data/kinetics_700/{class_name}.csv")
    return new_df


def main():
    data = pd.read_csv("../data/kinetics_700/train.csv")
    new_df = filter_kinetics_class(data, 'dancing', 200)
    print(new_df)


main()
