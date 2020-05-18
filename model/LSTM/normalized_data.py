import numpy as np

#feature_normalize=[{'mean': 25.76035066064466, 'std': 7.3569232191957585}, {'mean': 29.837132156983756, 'std': 7.072682043660889}, {'mean': 45.577636266726515, 'std': 48.89829557932252}, {'mean': 3.5434743456672364, 'std': 6.371631391585683}, {'mean': 5.7921342328947745, 'std': 3.6605893378270493}, {'mean': 12.560072937414084, 'std': 3.2526404585075688}]


def data_normalized(dataset,feature_normalize):
    n_features=np.shape(dataset)[1]
    print(n_features)
    tmp_list=[]
    for i in range(n_features):
        data=dataset[:,i]
        feature_norm=feature_normalize[i]
        mean=feature_norm['mean']
        std=feature_norm['std']
        normalized_data=(data-mean)/std
        tmp_list.append(np.array(normalized_data).reshape(-1,1))
    res=np.concatenate((np.array(tmp_list)),axis=1)
    return res