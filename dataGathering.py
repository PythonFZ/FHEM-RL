'''
 Coding in UTF-8
 This program is used to gather data from a mysql-server and preprocess it for further usage in a
 reinforcment learning algorithm
'''
#IDEEN:
# Später soll das ganze in echtzeit laufen, es soll immer die Zeit zum vorrangegangen Event (und ev. die aktuelle Zeit)
# in State zur Verfügung gestellt werden. Die Aktions sind dann Tu nichts oder schalte etwas.
# Wenn Dinge gleichzeitig vorkommen, dann soll das auch möglich sein! -> 2^devices

import numpy as np
import pandas as pd
import sqlalchemy as db


class getData():
    '''
    Get Data from the database and restructure it for further usage
    '''

    def __init__(self):
        self.initVars()

    def initVars(self):
        '''
        Parameters
        ----------

        myIP : str
            This should be the ip Adress of the Server ( e.g. '192.168.4.1')
        myUser: str
            This should be the user of the database. Only reading is needed and there should be no write
            access to prevent mistakes (e. g. 'fhemuser')
        myPassword: str
            This should be the password for database access (e.g. 'fhempassword')
        device_list: list
            This should be the list of devices which are actively monitored. Examples are Motionsensors or Lightswitches.
            The Structure of the list should be as followd: [['DEV1', 'state'], ['DEV2', 'luminance'], ['DEV2', 'state']]
        identical_readings: list
            This should be a list of readings that can be handled identically (e.g. [['set_on', 'on'],
             ['set_off', 'off']])
            This is only needed to prevent Errors! If you do not want this, just leave the list empty like []

        Returns
        -------
        df : pd.DataFrame
            This contains the data from the server
        device_reading_token : dict
            This contains a dictionary of the devices with the readings and an integer token for usage in the program.
            This is very helpful and necessary to gain the device names, readings and states back!

        '''
        self.myIP = '192.168.37.33'
        self.myUser = 'fhemuser'
        self.myPassword = 'fhempassword'
        self.df = None
        self.device_list = [['ZWave_SENSOR_NOTIFICATION_7', 'state'],
                            ['ZWave_SENSOR_NOTIFICATION_11', 'state'],
                            ['ZWave_SENSOR_NOTIFICATION_9', 'state']]
        self.device_list += [['ZWave_SWITCH_BINARY_5.01', 'state'],
                             ['ZWave_SWITCH_BINARY_5.02', 'state'],
                             ['Fabian.Relais', 'state'],
                             ['Kueche.LED', 'state'],
                             ['Kueche.Relais', 'state']]
        self.identical_readings = [['set_on', 'on'], ['set_off', 'off'], ['set_toggle', 'off']]
        self.device_reading_token = {}

    def remoteConnect(self):
        '''

        Notes
        -----
        Connects to the database, queries the predefined devices and removes all unnecessary readings.

        '''
        engine = db.create_engine(f'mysql+pymysql://{self.myUser}:{self.myPassword}@{self.myIP}:3306/fhem')
        queryString = f'SELECT TIMESTAMP, DEVICE, VALUE, READING FROM {"history"} '
        queryString += f' WHERE DEVICE IN {str(tuple([row[0] for row in self.device_list]))} ORDER BY TIMESTAMP DESC'
        self.df = pd.read_sql_query(queryString, engine, index_col='TIMESTAMP')

    def clear_reading(self):
        '''

        Returns
        -------
        df: pd.DataFrame
            Dataframe containing only Device + Reading combinations allowed in device_list

        '''
        elements = [tuple(element) for element in self.device_list]
        df = self.df[self.df[['DEVICE', 'READING']].apply(tuple, axis=1).isin(elements)]
        self.df = df

    def sort_df(self):
        '''
        Notes
        ------
        To prevent problems with events occurring at the same timestamp this sorts everything depending on
        time and alphanumeric!

        Returns
        -------
        df : pd.DataFrame
            sorted Dataframe

        '''
        data = self.df
        data['join'] = data['DEVICE'] + '.' + data['READING']

        for idx, element in enumerate(self.device_list):
            data['join'] = data['join'].replace(f'{element[0]}.{element[1]}', idx)

        data = data.sort_values(['TIMESTAMP', 'join'])
        data.drop(columns='join', inplace=True)

        self.df = data

    def pivot_to_readings(self):
        '''
        Notes
        ------
        Converts the dataframe to the new format of columns by devices + readings and rows by timestamp.
        Before this pivoting the dataframe has only been ordered by timestamp!
        Returns
        -------

        '''
        #TODO set_off besser ganz rauß als ersetzten?
        # Bei gleichem Zeitpunkt kombiniert pivot die Werte!
        data = self.df
        data = data.pivot_table(index='TIMESTAMP', columns=['DEVICE', 'READING'],
                                values='VALUE', dropna=True, aggfunc='first')
        for col in data.columns:
            if col not in self.device_reading_token:
                self.device_reading_token[len(self.device_reading_token)] = col
            else:
                raise Exception(f'{col} is already in list but should not!')

        #         data.columns = ['.-.'.join(col).strip() for col in data.columns]  # Kombiniert zu single column df
        data.columns = [idx for idx in range(len(self.device_reading_token))]
        data.dropna(how='all', axis=1, inplace=True)

        for identical in self.identical_readings:
            data = data.replace(identical[0], identical[1])

        self.df = data

class prepareData:
    '''
    This class prepares data for usage with a reinforcment network. It applies techniques like one_hot_encoding for
    easier machine processing.
    '''

    def __init__(self, df, device_reading_token):
        '''

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe from the class getData
        device_reading_token: dict
            Dictionary containing the column names and tokens!
        '''
        self.df = df.copy()
        self.device_reading_token = device_reading_token
        self.device_reading_state_token = {}

    def to_one_hot(self):
        '''

        Returns
        -------
        Creates a one_hot_encoding and device_reading_state_token dictionary for easier processing.
        '''
        data = self.df
        one_hot_df = pd.DataFrame(None)
        data.ffill(inplace=True)
        for col in list(data):
            for val in data[col].dropna().unique():
                if (col, val) not in self.device_reading_state_token:
                    self.device_reading_state_token[len(self.device_reading_state_token)] = (
                    self.device_reading_token[col][0],
                    self.device_reading_token[col][1],
                    val)
                    idx = len(self.device_reading_state_token) - 1
                else:
                    for key, value in self.device_reading_state_token:
                        if value == (self.device_reading_token[col][0], self.device_reading_token[col][1], val):
                            idx = key
                one_hot_df[idx] = (data[col] == val) * 1
        #             data.drop(columns=col, inplace=True)
        self.df = one_hot_df
        self.tok_lst = data.columns.values.tolist()

    def to_human_readable(self):
        '''

        Returns
        -------
        df : pd.DataFrame
            Dataframe with human readable not tokenized but long column names

        Additionally prints out the column names.

        '''
        self.df.rename(index=str, columns=self.device_reading_state_token, inplace=True)
        print(list(self.df))

    def get_all_actions(self):
        '''

        Returns
        -------
        actionspace_dict: dict
            This dictionaray contains all the possible unique actions that have been observed. It is unlikely to add
            more actions because this is the sum of single actions and actions observed combined at the same timestep!
            The np.array has been converted to a tuple to avoid Errors!
        '''
        my_array = self.df.drop_duplicates().to_numpy()
        self.actionspace_dict = dict(enumerate(my_array))
        self.actionspace_dict = dict([[tuple(v), k] for k,v in self.actionspace_dict.items()])

    def get_all_changes(self):
        '''

        Returns
        -------
        actionspace_dict: dict
            This dictionary contains all changes in device states. Only changes from 0 -> 1 are registred. Changes from
            1 -> 0 are ignored!
        '''
        my_array = self.df.to_numpy()[1:] - self.df.to_numpy()[:-1]
        # Find all the changes in between timesteps
        my_array[my_array < 0] = 0
        my_array = np.unique(my_array, axis=0)
        self.actionspace_dict = dict(enumerate(my_array))
        # Reverse dictionaray keys and values!
        self.actionspace_dict = dict([[tuple(v), k] for k, v in self.actionspace_dict.items()])

    def normalize_data(self):
        '''

        Returns
        -------
        df: pd.DataFrame
            T-Transformed dataframe
        '''
        self.df = self.df.ffill().bfill().dropna()
        self.df = self.df.apply(pd.to_numeric)
        self.df = (self.df - self.df.mean(axis=0)) / self.df.std(axis=0)  # Z-Transformation/Normalisierung

    def fit_data_to_df(self, new_df):
        '''

        Parameters
        ----------
        new_df: pd.DataFrame
            self.df will be fitted to the index of new_df. Therefore ffill and bfill will be applied!

        Returns
        -------

        '''
        new_df = new_df.copy()
        new_df['isDev'] = True
        new_df = pd.DataFrame(new_df['isDev'])
        df = pd.concat([self.df, new_df], sort=True)
        df['isDev'].fillna(value=False, inplace=True)
        df = df.ffill().bfill()

        self.df = df[df['isDev']].drop(columns=['isDev']).apply(pd.to_numeric)



if __name__ == '__main__':
    myData = getData()
    myData.remoteConnect()
    myData.clear_reading()
    myData.sort_df()
    myData.pivot_to_readings()

    myPrep = prepareData(myData.df, myData.device_reading_token)
    myPrep.to_one_hot()
    myPrep.get_all_actions()
    print(myPrep.actionspace_dict)