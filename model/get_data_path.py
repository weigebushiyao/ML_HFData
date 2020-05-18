import os

cur_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(cur_path, '..'))


def get_train_data_path(fc,fj,model_kind,params_kind):
    data_path = parent_path + '/data/mergedData_new/'+fc+'_'+fj+'_'+model_kind+'_'+params_kind+'/'
    datafile = os.listdir(data_path)[0]
    return data_path + datafile


def get_test_data_path(fc,fj,model_kind,params_kind):
    data_path = parent_path + '/data/mergedFaultData_new/'+fc+'_'+fj+'_'+model_kind+'_'+params_kind+'/'
    datafile = os.listdir(data_path)[0]
    return data_path + datafile

def get_filtered_data_path(fc,fj,model_kind,params_kind):
    data_path = parent_path + '/data/mergedData_filtered/'+fc+'_'+fj+'_'+model_kind+'_'+params_kind+'/'
    datafile = os.listdir(data_path)[0]
    return data_path + datafile