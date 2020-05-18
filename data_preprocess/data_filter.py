import pandas as pd
import csv
import os
import time
import sys
import re
import datetime
from shutil import copyfile
from model.params_dict import ParamsDict

def time_split(time_str):
    time_s = str.split(time_str, '|')[1]
    # 2018-01-0100:00:10
    # ts=time_s[:10]+' '+time_s[10:]
    ts = time.strptime(time_s, '%Y-%m-%d%H:%M:%S')
    ts = int(time.mktime(ts))
# time_split()
def get_time():
    f = open('./time_range.csv', 'r')
    csvfile = csv.reader(f)
    time_list = []
    for cf in csvfile:
        time_start = cf[0]
        time_end = cf[1]
        time_start = time_split(time_start)
        time_end = time_split(time_end)

        print(cf)
    f.close()

# get_time()
# 获取电厂特定风机的故障时间段。供后续的训练数据根据时间段进行抽取数据用。返回一个list：[['2016/4/16 8:30' '2016/4/16 16:00']
# ['2016/5/1 8:10' '2016/5/1 15:00']
# ['2017/9/11 16:00' '2017/9/11 19:30']
# ['2017/9/12 1:20' '2017/9/13 0:00']]，此为各个时间段的起止时间。
class GetGuZhangData:
    def __init__(self, fc,fj,model_kind,feature_number=5,params_kind=None):
        self.fc = fc  # 风电场
        self._time_list=None#故障时间
        self.fj_YearMonth=set()
        self.fj=fj
        self.feature_number=feature_number
        cur_path=os.path.abspath(os.path.dirname(__file__))
        self.data_path= os.path.abspath(os.path.join(cur_path, '..')) + '/data/'
        self.faultRecordData_path=self.data_path+'faultRecordData/'
        self.originalData_path= self.data_path + 'originalData/'
        self.file_list=[]
        self.historyDataPath = []
        self.guzhangFilePath=set()
        self.folder_name=None
        self.model_number=str.split(model_kind,'_')[-1]
        self.fj_model_data_path=fc+'_'+fj+'_'+model_kind+'_'+params_kind+'/'
        self.fj_model_data_file_name=fc+'_'+fj+'_'+model_kind+'.csv'
        # self.fault_path= self.data_path +'f'+str(self.feature_number)+ '_faultTestData/'
        # self.train_path= self.data_path +'f'+str(self.feature_number)+ '_trainTestData/'
        # self.train_data_merged_path=self.data_path+'f'+str(self.feature_number)+'_mergedData_new/'
        # self.fault_data_merged_path=self.data_path+'f'+str(self.feature_number)+'_mergedFaultData_new/'
        self.fault_path = self.data_path +  'faultTestData/'
        self.train_path = self.data_path +  'trainTestData/'
        self.train_data_merged_path = self.data_path + 'mergedData_new/'
        self.fault_data_merged_path = self.data_path + 'mergedFaultData_new/'
        self.single_merged_fj_train_data_path=self.train_data_merged_path+self.fj_model_data_path
        self.single_merged_fj_fault_data_path=self.fault_data_merged_path+self.fj_model_data_path
        self.single_fj_fault_data_path=self.fault_path+self.fj_model_data_path
        self.single_fj_train_data_path=self.train_path+self.fj_model_data_path
        self.model_kind=model_kind

        #              'cap_temp_3':['pitch_Atech_hub_temp_3', 'pitch_Atech_cabinet_temp_3',
        # 'pitch_Atech_motor_current_3', 'pitch_position_3', 'wind_speed', 'rotor_speed',
        # 'pitch_Atech_capacitor_temp_3'],
        model_params = ParamsDict()
        #print(model_params.model_params)
        print(model_kind)
        self.params = model_params.model_params[params_kind][model_kind]

    def _process_time(self,time_str,time_end):
        time_str+=':00'
        time_end+=':00'
        start_timearray=time.strptime(time_str,'%Y/%m/%d %H:%M:%S')
        end_timearray=time.strptime(time_end,'%Y/%m/%d %H:%M:%S')
        #当风机发生故障时，应该将发生故障维修时间段前后的总共两天纳为故障时间。故应该将故障的开始时间往前移一天，修好的时间往后延一天。
        laststart_timearray=datetime.datetime(start_timearray.tm_year,start_timearray.tm_mon,start_timearray.tm_mday-1,0)
        latestend_timearray=datetime.datetime(end_timearray.tm_year,end_timearray.tm_mon,end_timearray.tm_mday+1,end_timearray.tm_hour)
        start_timearray=time.strptime(str(laststart_timearray),"%Y-%m-%d %H:%M:%S")
        end_timearray=time.strptime(str(latestend_timearray),"%Y-%m-%d %H:%M:%S")
        # print(timearray.tm_year)
        #将故障发生的年月存入set，供后续查找文件用。
        #ret=None
        if start_timearray.tm_mon==end_timearray.tm_mon:
            str_time=time.strftime("%Y-%m-%d %H:%M:%S",start_timearray)
            mon_str=str.split(str_time,'-')[1]
            self.fj_YearMonth.add(str(start_timearray.tm_year) +mon_str)
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", end_timearray)
            ret=[{str(start_timearray.tm_year)+mon_str+self.fj:[str_time,end_time]}]
        else:
            future_mouth_first = datetime.datetime(start_timearray.tm_year, start_timearray.tm_mon + 1,1, 23, 59, 59)
            # 当月最后一天最后一秒时间
            str_time=time.strftime("%Y-%m-%d %H:%M:%S",start_timearray)
            mon_str_start=str.split(str_time,'-')[1]
            start_timelastday = future_mouth_first - datetime.timedelta(days=1)
            end_timefirstday=datetime.datetime(end_timearray.tm_year,end_timearray.tm_mon,1)
            mon_str_end=str.split(end_timefirstday,'-')
            end_time=time.strftime("%Y-%m-%d %H:%M:%S", end_timearray)
            self.fj_YearMonth.add(str(start_timearray.tm_year)+str(start_timearray.tm_mon))
            self.fj_YearMonth.add(str(end_timearray.tm_year)+str(end_timearray.tm_mon))
            ret=[{str(start_timearray.tm_year)+mon_str_start+self.fj:[str_time,start_timelastday]}]
            ret+=[{str(end_timearray.tm_year)+mon_str_end+self.fj:[end_timefirstday,end_time]}]

        return ret


    def _tran_time(self,timelist):
        time_list=[]
        for e in timelist:
            if len(e)!=2:
                pass
            #将故障时间发生时往前移一天，将故障修复时间往后延一天。
            start_t,start_mon=self._process_time(e[0], -1)
            end_t,end_mon=self._process_time(e[1], 1)
            time_list.append()
        self._time_list=time_list
        #print(self._time_list)

    def fault_time(self):
        guzhangYearMonth=[]
        guzhangdir= self.faultRecordData_path
        if not os.path.exists(guzhangdir):
            os.makedirs(guzhangdir)
            print("NO guzhangdata file.")
        filename = os.listdir(guzhangdir)
        if len(filename)!=1:
            print("The number of guzhangdata is wrong. ")
            sys.exit(1)
        df = pd.read_csv(guzhangdir + filename[0], encoding='gbk')
        new_df = df[(df['FC'].isin([self.fc])) & (df['FJ'] == self.fj)]
        _time_list = new_df[['KSSJ', 'JSSJ']].values  # 435...17
        for t in _time_list:
            ret=self._process_time(t[0],t[1])
            if ret:
                guzhangYearMonth+=ret
        return guzhangYearMonth#某风机的故障时间段

    #folder_name是包含各个数据文件的文件夹。
    def _get_file_name(self):
        self.folder_name=os.listdir(self.originalData_path)
        for fn in self.folder_name:
            tmp_list=[]
            file_name=os.listdir(self.originalData_path + '/' + fn)
            for fi in file_name:
                tmp_dict={fi:fn}
                tmp_list.append(tmp_dict)
                self.historyDataPath.append(fn + '/' + fi)
            self.file_list+=tmp_list#由文件名，上层文件夹名构成kv对。


    #提供已经知道的风场和机组，根据获取到的故障时间，然后在根据时间的年月去找到对应的文件，并从文件中获取对应时间段的数据，并另存为故障数据文件
    def get_guzhangdata_file(self):
        guzhang_file_dict=dict()
        for gz_time in list(self.fj_YearMonth): #单台风机的故障年月。
            #gz_time='201806'
            guzhang_file_list = []
            pattern=self.fc+'\w\w\w\d.\d\w'+gz_time+self.fj
            for fl in self.file_list:
                #fn='rzjx_UP2.0_201806w002.csv'
                fn=list(fl)[0]
                res=re.match(pattern,fn)
                if res:
                    guzhang_file_list.append(list(fl.values())[0]+'/'+fn)
            if gz_time+self.fj not in guzhang_file_dict:
                guzhang_file_dict[gz_time+self.fj]=guzhang_file_list
            else:
                guzhang_file_dict[gz_time+self.fj]+=guzhang_file_list
        return guzhang_file_dict

    def tran_dict(self,d):
        k=list(d)[0]
        v=list(d.values())[0]
        return k,v

    def _process_file_path_list(self, list1, list2):
        if list1 and list2:
            tmp_list=list(list1-list2)
            for e in tmp_list:
                copyfile(self.originalData_path + e, self.train_path + self.fj_model_data_path + str.split(e, '/')[1])
                #df=pd.read_csv()


    def open_guzhangdata_file(self,guzhangfile):#guzhangfile:{201808w002:[starttime,enttime]}
        #目录/文件
        #time_list=['2018-02-01 00:00:00','2018-02-02 00:00:00']
        #tr=df[df['fjtime'].between('2018-04-01 00:05:30','2018-04-01 0:05:40')]
        if not os.path.exists(self.single_fj_train_data_path):
            os.makedirs(self.single_fj_train_data_path)
        if not os.path.exists(self.single_fj_fault_data_path):
            os.makedirs(self.single_fj_fault_data_path)
            # f=open(faultDataName,mode='a',encoding='utf-8',newline='')
            # f.close()
        gztime_fj,time_starend=self.tran_dict(guzhangfile)
        fileTimeRange=time_starend[0][5:7]+time_starend[0][8:10]+'-'+time_starend[1][5:7]+time_starend[1][8:10]
        res = self.get_guzhangdata_file()#guzhangfj:[path]
        for v in res.values():
            self.guzhangFilePath.add(v[0])
        guzhangfile_pathlist=res[gztime_fj]
        def processTime(t):
            new_t = str.split(t, '|')[1]
            new_t = new_t[:10] + ' ' + new_t[10:]
            return new_t
        for gzf in guzhangfile_pathlist:#某风机故障文件列表
            data_path= self.originalData_path  + gzf
            df=pd.read_csv(data_path,encoding='utf-8')
            print(df.shape)
            # df=df[df['ControllerState']==1]
            t_name = df.columns.values[0]
            df[t_name] = df.apply(lambda x: processTime(x[t_name]), axis=1)
            data_timerange= df[df[t_name].between(time_starend[0], time_starend[1])]
            faultDataFile= 'gz' + fileTimeRange + '_' + str.split(gzf, '/')[1]
            faultDataName= self.single_fj_fault_data_path + faultDataFile
            print(data_timerange.shape)
            data_timerange.to_csv(faultDataName,index=False)
            indext_list = data_timerange.index.values
            start_idx, end_idx = indext_list[0], indext_list[-1]
            new_df = df.drop(labels=range(start_idx, end_idx), axis=0, inplace=False,index=None)
            if new_df.shape[0]:
                tmp_path=self.single_fj_train_data_path+str.split(gzf,'/')[1]
                if os.path.exists(tmp_path):
                    print('remove file:'+tmp_path)
                    os.remove(tmp_path)
                new_df.to_csv(tmp_path, mode='a',index=None)
                print(new_df.shape)


    def save_data(self):
        self._get_file_name()
        #从文件中获取故障时间
        tmp_guzhangYearMonth=self.fault_time()
        guzhangYearMonth=[]
        guzhangTime_dict=dict()
        for e in tmp_guzhangYearMonth:
            k,v=self.tran_dict(e)
            if k in guzhangTime_dict:
                guzhangTime_dict[k].append(v)
            else:
                guzhangTime_dict[k]=[v]
        for k,v in guzhangTime_dict.items():
            guzhangYearMonth.append({k:[v[0][0],v[-1][-1]]})
        print(guzhangYearMonth)
        #将故障之时间转换为csv文件原始数据可以比较接近的数据。
        # self._tran_time(time_list)#故障时间范围集合，由起始时间组成
        for gzym_fj in guzhangYearMonth:
            self.open_guzhangdata_file(gzym_fj)
        self._process_file_path_list(set(self.historyDataPath), self.guzhangFilePath)
        #获取所有历史数据
        #使用正则表达式，找出故障发生时间段内的文件，供后续抽取故障数据用。

    def merge_history_data(self):
        if not os.path.exists(self.single_merged_fj_train_data_path):
            os.makedirs(self.single_merged_fj_train_data_path)
        fileList=os.listdir(self.single_fj_train_data_path)
        if os.path.exists(self.single_merged_fj_train_data_path+self.fj_model_data_file_name):
            os.remove(self.single_merged_fj_train_data_path+self.fj_model_data_file_name)
            print('remove file:'+self.train_data_merged_path+self.fj_model_data_file_name)
        #fileList=['wfzc_UP1.5_201808A2.csv','wfzc_UP1.5_201907A2.csv','wfzc_UP1.5_201908A2.csv','wfzc_UP1.5_201911A2.csv','wfzc_UP1.5_201903A2.csv']
        flag=True
        for fl in fileList:
            print(self.single_fj_train_data_path+fl)
            df=pd.read_csv(self.single_fj_train_data_path+fl,encoding='utf-8',low_memory=False)
            #print(df.columns)
            print(df.shape)
            #获取并网的数据
            posi_df=df[df['pitch_position_'+self.model_number]>0.1]
            print(posi_df.shape)
            conn_df=posi_df[posi_df['ControllerState']==1]
            if re.match('pitch_Atech_motor_temp_'+'\d',self.params[-1]):
                print('true')
                conn_df=conn_df[conn_df[self.params[-1]]>0]
                conn_df=conn_df[conn_df['pitch_Atech_motor_current_'+str(self.model_number)]>0]
                #conn_df=conn_df[conn_df['pitch_position_'+self.model_number]<30]
                #conn_df=conn_df[conn_df['rotor_speed']>8]
            print(conn_df.shape)
            new_df=conn_df[self.params]
            print(new_df.shape)
            if flag:
                h=self.params
            else:
                h=None
            new_df.to_csv(self.single_merged_fj_train_data_path+self.fj_model_data_file_name,encoding='utf-8',mode='a',header=h)
            print(h)
            flag=False

    def merge_fault_data(self):
        if not os.path.exists(self.single_merged_fj_fault_data_path):
            os.makedirs(self.single_merged_fj_fault_data_path)
        file_list=os.listdir(self.single_fj_fault_data_path)
        merged_fault_data_file='gz_'+self.fj_model_data_file_name
        if os.path.exists(self.single_fj_fault_data_path+merged_fault_data_file):
            os.remove(self.single_merged_fj_fault_data_path)
            print('remove file:'+self.single_fj_fault_data_path+merged_fault_data_file)
        flag = True
        for fl in file_list:
            print(fl)
            df = pd.read_csv(self.single_fj_fault_data_path  + fl, encoding='utf-8')
            #print(df.columns)
            print(df.shape)
            # 获取并网的数据
            conn_df = df[df['ControllerState'] == 1]
            print(conn_df.shape)
            conn_df=conn_df[conn_df['converter_power']>0]
            new_df = conn_df[self.params]
            print(new_df.shape)
            if flag:
                h = self.params
            else:
                h = None
            new_df.to_csv(self.single_merged_fj_fault_data_path + merged_fault_data_file, encoding='utf-8', mode='a', header=h)
            print(h)
            flag = False

    def process_data(self):
        #self.save_data()
        #self.merge_history_data()
        self.merge_fault_data()

gz = GetGuZhangData('wfzc','A2',model_kind='cap_temp_3',params_kind='model_params_v3')
gz.process_data()

