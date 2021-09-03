import os
import pandas as pd


def label_prefog(dataset, window_length=1):
    dataset.drop(index=list(dataset[dataset['Action'] == 0].index),
                 inplace=True)
    window_length = 64 * window_length

    fog_index = []
    for i in dataset.index:
        if dataset.loc[i, 'Action'] == 2:
            fog_index.append(i)


    start_indices = []
    for i in fog_index:
        if (dataset.loc[i - 1, 'Action'] != dataset.loc[i, 'Action']):
            start_indices.append(i)

    prefog = []
    for start in start_indices:
        prefog_start = [x for x in range(start - window_length, start)]
        prefog.append(prefog_start)

    prefog = [item for sublist in prefog for item in sublist]

    for i in prefog:
        dataset.loc[i, 'Action'] = 3
    dataset['Action'] = dataset['Action'] - 1
    return dataset


data_path = "./dataset_fog_release/dataset"
people = []
for person in os.listdir(data_path):
    if '.txt' in person:
        people.append(person)
for window_length in range(1, 5):
    dataset = pd.DataFrame()
    for person in people:
        name = person.split('R')[0]
        print(name)
        file = data_path + "\\" + person
        temp = pd.read_csv(file, delimiter=" ", header=None)
        print(person, ' is read', end='\t')
        if 2 in temp[max(temp.columns)].unique():
            print('Adding {} to dataset'.format(person), end='\t')  # 将人加入到dataset中
            temp.columns = ['time', 'A_F', 'A_V', 'A_L', 'L_F', 'L_V', 'L_L', 'T_F', 'T_V', 'T_L', 'Action']
            temp = label_prefog(temp, window_length).reset_index(drop=True)
            temp['name'] = name
            print('{} is labelled'.format(person))
            dataset = pd.concat([dataset, temp], axis=0)

        print('')
    dataset.reset_index(drop=True, inplace=True)
    to_path = "./TGNDA_for_FOG"
    to_name = to_path + str(window_length) + ".csv"
    dataset.to_csv(to_name, index=False)
