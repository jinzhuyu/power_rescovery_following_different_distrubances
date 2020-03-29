# -*- coding: utf-8 -*-
"""
1. read power outage data from Sandia;
2. detect power outages, return outage count and the corresponding time points
3. data visualization.
"""
#def main():
#    # Todo: Add your code here
#    pass
#
#
#if __name__ == '__main__':
#    main()

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import (
        YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange)
from matplotlib import cm # rainbow color scheme
from datetime import datetime, timedelta
import time
import os
import re # to extract substring from string delimited by a combination of ":", ",", or ";"
import copy
#import dask.dataframe

plt.style.use('classic')
plt.rc('lines', linewidth=1.5)
plt.rc('lines', markersize=8)
plt.rc('font', family='Times New Roman')            # controls default text sizes
plt.rc('axes', titlesize=14, titleweight='bold')    # fontsize of the axes title
plt.rc('axes', labelsize=12, labelweight='bold')    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
#ref.:https://www.programcreek.com/python/example/94515/matplotlib.rc
def plot_trajectories(x, fx):  
#    font = {'family' : 'Times New Roman',
#            'size'   : 12}
#    mpl.rc('font', **font)
    plt.figure(figsize=(8,6))
    line=plt.plot(x,fx) 
    plt.setp(line)
    plt.title('A looong title looks like this')
    plt.xlabel('Time after disruption (h)')
    plt.ylabel('Outage count')
    plt.xlim(left=0)
    axe = plt.gca()
    axe.yaxis.grid(True)
    axe.tick_params(top=False, right=False)
#    plt.legend(loc='best')
#    plt.savefig('Power outage k=%s.pdf' % ', dpi=600, bbox_inches='tight')
    plt.show()
#plot_trajectories(x, np.sin(x))


#import seaborn as sns
#import plotly.express as px
#import pickle


# set direc
os.chdir('C:\GitHub\Power_grid_resilience')

# import data
outage_attr = ['utility_id','utility_name','fips_code','county','state','outage_count','run_start_time']
outage_data_df = pd.read_csv('outage_summary.csv', header=None, names=outage_attr)
#outage_data_df_orig = pd.read_csv('outage_summary.csv', header=None, names=outage_attr)
#outage_data_df = pickle.load('power_outage_summary_py.pickle')


# county lat and long and fips
county_coor_fips_df = pd.read_csv('county fips and coordinates.csv')
states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}
for ii in range(len(county_coor_fips_df)):
    abbr_temp = county_coor_fips_df['state'].iloc[ii]
    county_coor_fips_df['state'].iloc[ii]=states.get(abbr_temp)
#state_df = pd.DataFrame.from_dict(states)

# population
county_popul_df = pd.read_csv('county_population.csv', header=0)
county_popul_df = county_popul_df.dropna()

# weather data
weather_data_df = pd.read_csv('weather data.csv', header=0)

# calculate county fips code in the county impacted by a weather event
weather_data_df['fips_code'] = weather_data_df['STATE_FIPS']*1000 + weather_data_df['CZ_FIPS']
weather_data_df['BEGIN_DATE_TIME'] = pd.to_datetime(weather_data_df['BEGIN_DATE_TIME'])
weather_data_df['END_DATE_TIME'] = pd.to_datetime(weather_data_df['END_DATE_TIME'])

# simplify event type names
weather_data_df.replace('Marine Hurricane/Typhoon','Hurricane')
weather_data_df.replace('Marine Tropical Storm','Tropical Storm')
weather_data_df.replace('Marine Tropical Depression','Tropical Depression')

# fill population
weather_data_df['population'] = np.nan
for ii in np.arange(0, weather_data_df.shape[0]):
    fips_code_temp = weather_data_df.loc[ii,'fips_code']
    if np.any(county_popul_df['fips_code']==fips_code_temp):
        weather_data_df.loc[ii,'population'] = county_popul_df.loc[
                county_popul_df['fips_code']==fips_code_temp,'population'].values

# time range
time_first = '2014-11-01 04:00:00'   #  min(outage_data_df['run_start_time'])
time_last = '2019-10-29 15:45:00'    #  max(outage_data_df['run_start_time'])


#######################################################
# electric disturbance events for Department of Engergy
dist_event_orig_df = pd.read_csv('EIA major disturbances_201411-201910.csv')
dist_event_OE_df = pd.read_csv('2014_to_2019_Annual_Summary.csv') # same data from OE without utility information

# remove empty space at two ends
dist_event_orig_df['Type of Disturbance'] = dist_event_orig_df['Type of Disturbance'].str.strip()
dist_event_orig_df['Number of Customers Affected'] = dist_event_orig_df['Number of Customers Affected'].str.strip()

# 
# dist_event_orig_df['dist_type'] = dist_event_orig_df['Type of Disturbance'].str.rsplit('-').str[0] 

# dist_event_orig_df.head()
dist_event_orig_df['Number of Customers Affected'] = dist_event_orig_df['Number of Customers Affected'].replace('Unknown', np.nan)
dist_event_orig_df['Number of Customers Affected'] = pd.to_numeric(dist_event_orig_df['Number of Customers Affected'])
dist_event_orig_df['Loss (megawatts)'] = dist_event_orig_df['Loss (megawatts)'].replace('Unknown', np.nan)
dist_event_orig_df['Loss (megawatts)'] = pd.to_numeric(dist_event_orig_df['Loss (megawatts)'])

# select data: # of customers affected > 500 and demand loss > 3.
#dist_event_df = dist_event_orig_df[(dist_event_orig_df['Number of Customers Affected'] >= 500)|
#                                   (dist_event_orig_df['Number of Customers Affected'].isna())]
#dist_event_df = dist_event_df[(dist_event_df['Loss (megawatts)'] >= 3)|
#                              (dist_event_df['Loss (megawatts)'].isna())]
dist_event_df = dist_event_orig_df

# change time format
dist_event_df['dist_start_time'] = pd.to_datetime(dist_event_df['Event Date and Time']).dt.strftime("%Y-%m-%d %H:%M:%S")

dist_event_OE_df['dist_start_time'] = dist_event_OE_df['Date Event Began'] + ' ' + dist_event_OE_df['Time Event Began']
dist_event_OE_df['dist_start_time'] = pd.to_datetime(dist_event_OE_df['dist_start_time']).dt.strftime("%Y-%m-%d %H:%M:%S")

# find the disturbance type
dist_event_df['dist_type'] = ''
for ii in np.arange(dist_event_df.shape[0]):
#    print(ii)
    dist_type_temp = dist_event_OE_df.loc[
            (dist_event_OE_df['dist_start_time']==dist_event_df['dist_start_time'].iloc[ii]), 'Event Type']
    if dist_type_temp.shape[0]==1:
        dist_event_df['dist_type'].iloc[ii] = dist_type_temp.iloc[0]
    elif dist_type_temp.shape[0]>1: 
        dist_event_df['dist_type'].iloc[ii] = dist_type_temp.iloc[0]
    else:
        dist_event_df['dist_type'].iloc[ii] = dist_event_df['Type of Disturbance'].iloc[ii]

# demand loss >=10 or number of customers >=2000
# dist_event_df.loc[(dist_event_df['Number of Customers Affected'] >= 1000)|
#                    (dist_event_df['Loss (megawatts)'] >= 10), 'dist_type'].unique()    
# dist_event_df['dist_type'] = dist_event_df['dist_type'].str.replace('\d+', '').str.strip()
# dist_event_df['dist_type'].unique()
# disturbance can occur at the same date and time!!!!!!!!!!

# dist_event_df['dist_end_time'] = dist_event_df['Restoration Date and Time'].replace( '.        .', None)

dist_event_df.loc[dist_event_df['Duration'] =='. Hours,  . Minutes', 'Restoration Date and Time'] = None
dist_event_df.loc[dist_event_df['Duration'] =='ongoing', 'Restoration Date and Time'] = None

dist_event_df['restoration_time'] = pd.to_datetime(dist_event_df['Restoration Date and Time']).dt.strftime("%Y-%m-%d %H:%M:%S")

# extract state and county
dist_event_with_county_df = dist_event_df[dist_event_df['Area Affected'].str.contains('County')]
dist_event_with_county_df['Area Affected'] = dist_event_with_county_df['Area Affected'].str.replace('County','')

# extract entries where at least 2 counties are impacted

dist_event_over_two_county_df = dist_event_df[(dist_event_df['Area Affected'].str.count('County')>=2)&
                                              (dist_event_df['dist_type']!='Severe Weather')&
                                              (dist_event_df['dist_type']!='Other')&
                                              (dist_event_df['Number of Customers Affected']>0)&
                                              (dist_event_df['restoration_time']!='NaT')]
dist_event_over_two_county_df['Area Affected'] = dist_event_over_two_county_df['Area Affected'].str.replace('County','')
#dist_event_with_two_county_df_add = pd.read_csv('dist_event_with_two_county_df_add.csv')
#dist_event_with_two_county_df = pd.concat([dist_event_with_two_county_df_sub, dist_event_with_two_county_df_add])
#dist_event_with_two_county_df.to_csv(r'C:\GitHub\Power_grid_resilience\dist_event_with_two_county_df.csv', index = False, header=True)
#outage_data_df_dist = outage_data_df.loc[outage_data_df['run_start_time']>=min(dist_event_over_two_county_df['dist_start_time'])]
#outage_data_df_dist.to_csv(r'C:\GitHub\Power_grid_resilience\outage_data_df_dist.csv', index = False, header=True)
outage_data_df_dist = pd.read_csv('outage_data_df_dist.csv')

for ii in np.arange(len(dist_event_over_two_county_df)):
 
    try:
#        print(ii)    
        if ii in [1, 5, 7, 8, 10]:
            continue
    # ii=1 too few data for one of the imperial county in Califonia
    # ii=5 Data about Dallas and Tarrant County are missing data.before 2016-02-26 17:28:00
    # ii=7, 8, no data after '2019-10-29 15:45:00'
    # ii=10, no data about Kane, Garfiled, and Coconino; loc_str = 'Utah: Garfield; Arizona: Mohave'      
        print(ii)    
        if ii==2:
            select_start_time = '2017-02-02 08:00:00'
            select_end_time = '2017-02-02 16:30:00'
            delay_end_boo = 0
        elif ii==6:
            select_start_time = dist_event_over_two_county_df['dist_start_time'].iloc[ii]
            select_end_time = '2016-09-13 01:00:00'
            delay_end_boo = 0
        elif ii==9:
            select_start_time = '2018-01-12 00:45:00'
            select_end_time = '2018-01-12 20:15:00'
            delay_end_boo = 0
        else:
            select_start_time =  dist_event_over_two_county_df['dist_start_time'].iloc[ii]            
            select_end_time =  dist_event_over_two_county_df['restoration_time'].iloc[ii]
            delay_end_boo = 1
            
        loc_str = dist_event_over_two_county_df['Area Affected'].iloc[ii]
        loc_dict = extract_county_state(loc_str)  
        dist_type = dist_event_over_two_county_df['dist_type'].iloc[ii]
        
        plot_all_county_dist(outage_data_df, loc_dict, select_start_time, dist_type,
                             select_end_time, plot_all = 1, delay_end_boo = delay_end_boo)

    except:
        print('error in loop {}'.format(ii))  
        pass

#select_start_time = '2016-05-10'
#select_end_time = '2016-12-10'
#dddf = outage_data_df.loc[(outage_data_df['county']=='Garfield') & (outage_data_df['state']=='Utah')&
#                   (outage_data_df['run_start_time']>=select_start_time)&
#                   (outage_data_df['run_start_time']<=select_end_time)]

# restoration_duration = pd.to_datetime(dist_event_df.loc[dist_event_df['restoration_time']!='NaT','restoration_time']) - 
#    pd.to_datetime(dist_event_df.loc[dist_event_df['restoration_time']!='NaT','dist_start_time'])
# max_duration = max(restoration_duration) # 40 days
max_after_unknown_duration = 50
max_after_known_duration = 30
unknonw_restoration_time_estimate = pd.to_datetime(
        dist_event_with_county_df.loc[dist_event_with_county_df['restoration_time']=='NaT', 'dist_start_time']) + timedelta(days = max_after_unknown_duration)
dist_event_with_county_df.loc[dist_event_with_county_df['restoration_time']=='NaT', 'restoration_time'] = unknonw_restoration_time_estimate.dt.strftime("%Y-%m-%d %H:%M:%S")
knonw_restoration_time_estimate = pd.to_datetime(
        dist_event_with_county_df.loc[dist_event_with_county_df['restoration_time']!='NaT', 'restoration_time']) + timedelta(days = max_after_known_duration)
dist_event_with_county_df.loc[dist_event_with_county_df['restoration_time']!='NaT', 'restoration_time'] = knonw_restoration_time_estimate.dt.strftime("%Y-%m-%d %H:%M:%S")
# the real restoration time is later than the given restoration time

for ii in np.arange(dist_event_with_county_df.shape[0]):
    dist_start_time_curr = dist_event_with_county_df['dist_start_time'].iloc[ii]
    time_to_recovery_curr = dist_event_with_county_df['restoration_time'].iloc[ii]
    state_county_str = dist_event_with_county_df['Area Affected'].iloc[ii]
    state_county_split = re.split(':|;|,', state_county_str)
    state_county_split = ' '.join(state_county_split).split()
    state_impacted = state_county_split[0]
    county_impacted_list = state_county_split[1:]
#    county_impacted_list = ['Black Hawk']
    for jj in np.arange(len(county_impacted_list)):
        county_temp = county_impacted_list[jj]
        fips_code_curr = county_coor_fips_df.loc[
                (county_coor_fips_df['state']==state_impacted) &
                (county_coor_fips_df['county']==county_temp), 'fips_code'].values.item()
        # fips_code and dist_start_time --> select county outage data
        # identify time to recovery 
        county_outage_data_curr = select_county_data(
                fips_code= fips_code_curr, start_time=dist_start_time_curr, end_time=time_to_recovery_curr, LB_count=0)
        if county_outage_data_curr.empty == False:
            plot_outage_hurr(fips_code = fips_code_curr, county_outage_data=county_outage_data_curr, 
                             event_name='Severe weather', event_type='dist', event_start_time = dist_start_time_curr,
                             event_end_time = time_to_recovery_curr, y_LB=0.955, LB_count=0)
#        time.sleep(0.01)
#        print('Next iter #: {}'.format(jj+1))


# ii =7, Maricopa

def debug_main():
    fips_code = fips_code_curr
    county_outage_data=county_outage_data_curr
    event_name='Islanding'
    event_type='dist'
    event_start_time = dist_start_time_curr
    event_end_time = time_to_recovery_curr
    LB_count = 0
    globals().update(locals())
    

#sss1 = dist_event_with_county_df['Area Affected'].str.split(r'[;:|]').apply(pd.Series, 1).stack()
#fragment.split(',') for fragment in dist_event_with_county_df['Area Affected'].iloc[30].split(':')

## utility name
#utility_name_parts = 'ISO New England'
#utility_name_bool = outage_data_df['utility_name'].apply(lambda sentence: any(word in sentence for word in utility_name_parts))
#utility_state_bool = outage_data_df.loc[outage_data_df['state']=='Massachusetts:', 'state']
#utility_name = outage_data_df.loc[(utility_name_bool) & (utility_state_bool),
#                                  'utility_name']
##str.contains(pat='Entergy')]



#dist_event_type = dist_event_df['Type of Disturbance'].unique() 
#vandalism = dist_event_orig_df[dist_event_orig_df['Type of Disturbance'] =='Vandalism']
#sabotage = dist_event_orig_df[dist_event_orig_df['Type of Disturbance'] =='Sabotage']

#n_county = len(outage_data_df['fips_code'].unique())
#print('Total # of counties and county-equivalents in the data:', n_county)  # 2990
#print('Total # of counties and county-equivalents in the US (Wiki):', 3142)

# find the first 3 counties that are servied by more than one utilities
# note that county name can be the same but fips_code is unique
#fips_two_utilities =[]
#for i_county in np.arange(n_county):
#    fips_temp = outage_data_df['fips_code'].iloc[i_county]
#    utility_id = outage_data_df['utility_id'].loc[outage_data_df['fips_code']==fips_temp]
#    if len(utility_id.unique())>=2:
#        fips_two_utilities.append(fips_temp)
#    if len(fips_two_utilities)>=3:
#        break
#print('The fips code of the first 3 counties served by more than two utilities:\n', fips_two_utilities)
## 

#oo_df = outage_data_df.loc[outage_data_df['fips_code'] == 53011, ['fips_code','outage_count']]
# select the power outage data of a county      
# county and fips_code are unique while utility_id is not,
# because a utility company can serve more than one county.
def select_county_data(fips_code, start_time=time_first, end_time=time_last, LB_count=0):
    '''
    args:
        fips code is unique for each county and can be found in original data: outage_data_df
    '''
    outage_data_county_df = outage_data_df[
                            (outage_data_df['run_start_time']>=start_time) &
                            (outage_data_df['run_start_time']<=end_time) &
                            (outage_data_df['fips_code']==fips_code) & 
                            (outage_data_df['outage_count'] >= LB_count)]
    outage_data_county_df = outage_data_county_df.sort_values(by='run_start_time')
    
# def aggregate_outage_count(outage_data_county_df):
    '''
    Add the outage counts of a county if it is served by more than one utilities
    args: 
        a pandas DataFrame that contains utility_id, utility_name, fips_code, 
        county, state, outage_count, run_start_time.
    out:
        county outage data DataFrame with utility_name (named as comb_utility), 
        fips_code, county, state, outage_count, run_start_time.
    '''
    if outage_data_county_df.empty==True:
        print('Data about county with fips code {} are missing'.format(fips_code))
        return pd.DataFrame()
    else:
        n_utility = len(outage_data_county_df['utility_id'].unique())
        if n_utility==1:
            return outage_data_county_df
        else:
            time_len = len(outage_data_county_df['run_start_time'])
            outage_time = np.asarray(outage_data_county_df['run_start_time'])
            outage_count = np.asarray(outage_data_county_df['outage_count'])
            
            outage_count_new = [outage_count[0]]
            outage_time_new = [outage_time[0]]
            i_time = 1
            index_keep_temp = 0  # index of time points that will be kept
            index_comb_boo = [0]
            
    #       records of different utility companies at the same time will be added
            while i_time < time_len:
                if outage_time[i_time]==outage_time[i_time-1]:
                    if outage_time[i_time]!=np.nan:
                        outage_count_new[index_keep_temp] += outage_count[i_time]
                    index_comb_boo[index_keep_temp] = 1
                    i_time += 1
                else:
                    outage_count_new.append(outage_count[i_time])
                    outage_time_new.append(outage_time[i_time])
                    index_comb_boo.append(0)
                    index_keep_temp += 1
                    i_time += 1
            # store the data in the dataframe        
            outage_count_comb_df = pd.DataFrame('comb_utility', columns=['utility_name'],
                                               index = np.arange(0,len(outage_time_new)))
            assign_col = {'fips_code': outage_data_county_df['fips_code'].iloc[0],
                          'county': outage_data_county_df['county'].iloc[0],
                          'state': outage_data_county_df['state'].iloc[0]}
            outage_count_comb_df = outage_count_comb_df.assign(**assign_col)
            outage_count_comb_df['run_start_time'] = np.array(outage_time_new)
            outage_count_comb_df['outage_count'] = np.array(outage_count_new)
            outage_count_comb_df['index_comb_boo'] = np.array(index_comb_boo)
            
            # add the original outage count data of each utility company
            
            # first create a df with time and count of each utility company and merge into the df with combined count
            utility_id_unique = outage_data_county_df['utility_id'].unique()
            for ii in np.arange(len(utility_id_unique)):
                outage_data_utility_df = outage_data_county_df.loc[
                        outage_data_county_df['utility_id']==utility_id_unique[ii],
                        ['run_start_time', 'outage_count']]
                
                count_name_temp = 'outage_count_{}'.format(ii+1)
                outage_data_utility_df = outage_data_utility_df.rename(
                        columns={'outage_count': count_name_temp})
                
                # merge into the combined dataset
                outage_count_comb_df = outage_count_comb_df.merge(outage_data_utility_df, 
                                              how='left', left_on='run_start_time', right_on='run_start_time')
                
            # add outage causes to be filled later
            outage_count_comb_df['n_utility'] = n_utility
            outage_count_comb_df['event_type'] = ''
            print('# of utility: {}'.format(n_utility))
            return outage_count_comb_df

# select power outage data using state and county
def select_county_data_dist(outage_data_df, county, state, start_time, end_time):
    '''
    args: fips code is unique for each county and can be found in original data: outage_data_df  
    out:  county outage data DataFrame with utility_name (named as comb_utility), 
          fips_code, county, state, outage_count, run_start_time.
    '''
    
    out_data_county_df = outage_data_df[
        (outage_data_df['run_start_time']>=start_time) & (outage_data_df['run_start_time']<=end_time) &
        (outage_data_df['county']==county) & (outage_data_df['state']==state)]
    out_data_county_df = out_data_county_df.sort_values(by='run_start_time') 
    
    if out_data_county_df.empty==True:
        print('Data about {} County, {} are missing'.format(county, state))
        return pd.DataFrame()
    else:
        n_util = len(out_data_county_df['utility_id'].unique())
        if n_util==1:
            return out_data_county_df
        else:
            time_len = len(out_data_county_df['run_start_time'])
            outage_time = np.asarray(out_data_county_df['run_start_time'])
            out_count = np.asarray(out_data_county_df['outage_count'])
            
            out_count_new = [out_count[0]]
            outage_time_new = [outage_time[0]]
            i_time = 1
            index_keep_temp = 0  # index of time points that will be kept
            index_comb_boo = [0]
            
    #        records of different utility companies at the same time will be added
            while i_time < time_len:
                if outage_time[i_time]==outage_time[i_time-1]:
                    if outage_time[i_time]!=np.nan:
                        out_count_new[index_keep_temp] += out_count[i_time]
                    index_comb_boo[index_keep_temp] = 1
                    i_time += 1
                else:
                    out_count_new.append(out_count[i_time])
                    outage_time_new.append(outage_time[i_time])
                    index_comb_boo.append(0)
                    index_keep_temp += 1
                    i_time += 1
            # store the data in the dataframe        
            out_count_comb_df = pd.DataFrame('comb_utility', columns=['utility_name'],
                                             index = np.arange(0,len(outage_time_new)))
            asgn_col = {'fips_code': out_data_county_df['fips_code'].iloc[0],
                          'county': out_data_county_df['county'].iloc[0],
                          'state': out_data_county_df['state'].iloc[0]}
            out_count_comb_df = out_count_comb_df.assign(**asgn_col)
            out_count_comb_df['run_start_time'] = np.array(outage_time_new)
            out_count_comb_df['outage_count'] = np.array(out_count_new)
            out_count_comb_df['index_comb_boo'] = np.array(index_comb_boo)
            
            # add the original outage count data of each utility company
            
            # first create a df with time and count of each utility company and merge into the df with combined count
            util_id_uniq = out_data_county_df['utility_id'].unique()
            for ii in np.arange(len(util_id_uniq)):
                out_data_util_df = out_data_county_df.loc[
                        out_data_county_df['utility_id']==util_id_uniq[ii],
                        ['run_start_time', 'outage_count']] 
                count_name_temp = 'outage_count_{}'.format(ii+1)
                out_data_util_df = out_data_util_df.rename(columns={'outage_count': count_name_temp})
                
                # merge into the combined dataset
                out_count_comb_df = out_count_comb_df.merge(
                        out_data_util_df, how='left', left_on='run_start_time', right_on='run_start_time')
                
            # add outage causes to be filled later
            out_count_comb_df['n_utility'] = n_util
            out_count_comb_df['event_type'] = ''
            print('# of utility: {}'.format(n_util))
            return out_count_comb_df

def extract_county_state(loc_str):
    # replace '.' with ';'
    loc_str = loc_str.replace('.',';')
    loc_str_split = re.split(';', loc_str)
    # remove empty string element in case the string ends with a ";"
    while("" in loc_str_split): 
        loc_str_split.remove("") 
    n_state = len(loc_str_split) 
    loc_dict={}
    for ii in np.arange(n_state):
        loc_str_sub = re.split(':', loc_str_split[ii])
        state_temp = loc_str_sub[0].strip()    
        # remove the "." following the last county
        county_list_temp = loc_str_sub[1:]
#        if ii==(n_state-1):
#            county_list_temp[-1] = county_list_temp[-1].replace('.','')
        # list to string
        county_str_temp = ' '.join(map(str,  county_list_temp))
        county_list_split = county_str_temp.split(',')
        county_list_split = [x.strip() for x in county_list_split]
        loc_dict[state_temp] = county_list_split
    return loc_dict


def plot_all_util_dist(county_outage_data, state, county, popul, event_start_time,
                       index_time, time_temp_datetime, dist_type, fig_num):
    global win_len
    win_len = 5 # moveing average window
    if ('Weather' in dist_type):
        x_loc = 0.65
    else:
        x_loc = 0.20
    global y_loc
    y_loc = 0.025
    
    global font_text
    font_text = 12
    
    dist_type = dist_type.replace('/', '-')
    save_fig_title = 'Restoration plots/other disturbance/{}_{}_{}_{}.png'.format(dist_type, state, county, event_start_time[0:10])        
    if 'n_utility' in county_outage_data.columns:
        n_utility = county_outage_data['n_utility'].iloc[0]
        utility_name = ['All combined']
        colors = iter(cm.rainbow(np.linspace(0, 1, n_utility)))
        plt.figure(fig_num)
        for jj in np.arange(n_utility):
            count_name_temp = 'outage_count_{}'.format(jj+1)
            outage_count_utility_i = county_outage_data[count_name_temp].iloc[index_time]                   
            outage_count_utility_i_arr = outage_count_utility_i.to_numpy(dtype='float32')
            restore_rate_utility_i = (popul - outage_count_utility_i_arr)/popul
            restore_rate_utility_i_ave = pd.Series(restore_rate_utility_i).rolling(window=win_len).mean()

            cl = next(colors)         
            plt.plot(time_temp_datetime, restore_rate_utility_i_ave,
                     color=cl)
            
            utility_name.append('Utility {}'.format(jj+1))
            
        utility_name = np.asarray(utility_name)    
        plt.legend(utility_name, bbox_to_anchor=(x_loc+0.02, y_loc+0.08), frameon=False,
                   loc="lower left", fontsize=font_text, fancybox=True, framealpha=0.5)
        plt.savefig(save_fig_title, dpi=600)

        
    else:
        plt.figure(fig_num)
        ax = plt.axes()
        plt.text(x_loc+0.04, y_loc+0.11, 'One utility only', fontsize=font_text,
                 color='black', fontweight='bold', transform=ax.transAxes) 
        plt.savefig(save_fig_title, dpi=600)


def plot_label(fig_num):
    
    plt.figure(fig_num)
    plt.ylabel('Restoration rate', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xlabel('Date and time', fontsize=14, fontname='Times New Roman', fontweight='bold')
    
    ax = plt.gca()
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(interval_multiples=True))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y \n %H:%M'))
    plt.gcf().autofmt_xdate()
    
    plt.yticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    plt.xticks(fontsize=12, fontname='Times New Roman', fontweight='bold')

#plot_all_county_dist(outage_data_df_dist, loc_dict, select_start_time, dist_type, select_end_time, plot_all = 1)

def plot_all_county_dist(outage_data_df, loc_dict, select_start_time, dist_type,
                         select_end_time=None, plot_all = 1, delay_end_boo = 0):
      
    if delay_end_boo == 1:
        if 'Weather' in dist_type:
            delay_end_day = 20
        else:
            delay_end_day = 5
    else:
        delay_end_day = 0
    select_end_time = pd.to_datetime(select_end_time) + timedelta(days=delay_end_day)
    select_end_time = select_end_time.strftime("%Y-%m-%d %H:%M:%S")  
    
    if select_end_time==None:
        delay_end_day = 45
        select_end_time = pd.to_datetime(select_start_time) + timedelta(days=delay_end_day)
        select_end_time = select_end_time.strftime("%Y-%m-%d %H:%M:%S")
 

    if ('Weather' in dist_type):
        x_loc = 0.65
    else:
        x_loc = 0.20
    
    state_list = list(loc_dict.keys())
    
    # to be used in legend
    county_all_list = []
    loc_dict_copy = copy.deepcopy(loc_dict)
    for jj in np.arange(len(state_list)):
        state_temp = state_list[jj]
        county_list_with_state = loc_dict_copy[state_list[jj]]

        for ii in np.arange(len(county_list_with_state)):
            county_list_with_state[ii] = county_list_with_state[ii] + ', ' + state_temp       
        county_all_list = county_all_list + county_list_with_state          
    county_all_arr = np.asarray(county_all_list)

    
    for jj in np.arange(len(state_list)):
        state = state_list[jj]
        county_list = loc_dict[state_list[jj]]
#        print('loc dict ', loc_dict)
               
        y_lim_LB = 1
          
        if len(county_list)>=2:
            
        
            for ii in np.arange(len(county_list)):
                
                # extract data for each county
                county = county_list[ii].strip()
                print('County: {}, {}'.format(county, state))
                
                
                county_outage_data = select_county_data_dist(
                        outage_data_df, county, state, start_time=select_start_time, end_time=select_end_time)
                
#                if min(county_outage_data.loc['', 'outage_count']<=
#                
#                select_start_time = pd.to_datetime(select_start_time) - timedelta(hours=20)
#                select_start_time = select_start_time.strftime("%Y-%m-%d %H:%M:%S")                 
                
                # get time data
                index_time = range(len(county_outage_data['run_start_time']))
                time_temp = county_outage_data['run_start_time'].iloc[index_time]
                time_temp_datetime = pd.to_datetime(np.asarray(time_temp))
                
                # calculate restoration rate
                fips_code =  county_outage_data['fips_code'].unique()[0]
                popul = int(county_popul_df.loc[county_popul_df['fips_code']==fips_code, 'population'].iloc[0])
                outage_count_comb = county_outage_data['outage_count'].iloc[index_time]                   
                outage_count_comb_arr = outage_count_comb.to_numpy(dtype = 'float32')
                restore_rate_comb = (popul-outage_count_comb_arr)/popul
                
                # smooth the restoration rate
                restore_rate_comb_ave = pd.Series(restore_rate_comb).rolling(window=win_len).mean()
                restore_rate_comb_ave[0:(win_len-1)] = restore_rate_comb[0:(win_len-1)]
                restore_rate_comb_ave[-(win_len-1):] = restore_rate_comb[-(win_len-1):]
                
                # find the start point and restoration point
                restore_LB = 0.99995
                norm_LB = restore_LB
                out_UB = 0.99995
                id_first_min_duration = 4*24*3  # reach minimum in 3 days
                id_restore_rate_min = restore_rate_comb_ave[:id_first_min_duration].idxmin()
                # the first point that is < LB
                # the first point that is > LB after the minimum
                id_start = 0 # default value
                id_restore = restore_rate_comb_ave.size # default value                
                if delay_end_boo == 0:
                    id_start = id_start
                    id_restore = id_restore
                else:
                    id_mid_point = max(id_restore_rate_min - win_len, win_len)
                    print('mid point',id_mid_point)
                    for kk in np.arange(id_mid_point):
                        if (restore_rate_comb_ave[kk] >= norm_LB) & (restore_rate_comb_ave[kk+win_len] <= out_UB):
                            id_start = kk
                            break
                    for kk in np.arange(id_mid_point, restore_rate_comb_ave.size):
                        if (restore_rate_comb_ave[kk-win_len] <= restore_LB) & (restore_rate_comb_ave[kk] >= restore_LB):
                            id_restore = kk
                            print('id_restore is updated')
                            break
                print('id_start', id_start)
                # the first point that is > LB after the minimum
                print('id_restore', id_restore)
#                restore_rate_comb_ave_seletc =  restore_rate_comb_ave[id_start:id_restore]   
                        

                df_temp = pd.DataFrame(columns=['Date','Restoration rate'], index=range(len(time_temp)))
                df_temp['Date'] = pd.to_datetime(np.asarray(time_temp))
                df_temp['Restoration rate'] = restore_rate_comb_ave                
                        
                # plot
                if np.min(restore_rate_comb)<=0:
                    
                    print('Error: population, {}, < maximum combined outage counts, {}'.format(
                            popul, np.max(outage_count_comb_arr)))
                    print('{}, {}'.format(county, state))
                    print('Fips code: {}'.format(fips_code))
                    time.sleep(5)
                    
                elif plot_all == 1:
                       
                    # restoration plots on the same figure
                    fig_num = 0
                    fig = plt.figure(fig_num, figsize=(10, 8))                         
                    # plt.figure()
                    plt.plot(df_temp['Date'][id_start:id_restore], df_temp['Restoration rate'][id_start:id_restore])
#                    plt.xlim(right = df_temp['Date'][id_restore])
    #                plt.ylim(bottom = 0.70)
                    if y_lim_LB >= np.min(restore_rate_comb_ave):
                        y_lim_LB = restore_rate_comb_ave[id_restore_rate_min]
    #                plot_label(fig_num)
    #                plt.show()
                        print('y lim LB', y_lim_LB)
                
                elif plot_all == 0:
                    # restoration plot on an individual figure
                    fig = plt.figure(ii, figsize=(10, 8))                         
                    plt.plot(df_temp['Date'][id_start:id_restore], df_temp['Restoration rate'][id_start:id_restore])
#                    plt.xlim(right = df_temp['Date'][id_restore])
                    ax = plt.axes()
                    plt.text(x_loc,y_loc+0.07,'County: {}, {}'.format(county, state),
                         fontsize=font_text, color='black', fontweight='bold', transform=ax.transAxes)                                
                    plot_label(ii)
                    
                    # add outages of each utility if there are more than one
                    plot_all_util_dist(county_outage_data, state, county, popul,select_start_time,
                                       index_time, time_temp_datetime, dist_type, fig_num = ii)
                    plt.show()
                    
            if plot_all == 1:
                dist_type_lgd = dist_type.replace('/', '/\n')
                fig_num = 0
                plt.figure(fig_num)
                
                plt.legend(county_all_arr, bbox_to_anchor=(x_loc-0.02, y_loc+0.08), frameon=False,
                           loc="lower left", fontsize=font_text, fancybox=True, framealpha=0.5)
                ax = plt.axes()
                if 'Weather' in dist_type:
                    y_loc_lgd = y_loc+0.03
                else:
                    y_loc_lgd = y_loc+0.06
                plt.text(x_loc,y_loc_lgd,'Disturbance: {}'.format(dist_type_lgd),
                         fontsize=font_text, color='black', fontweight='bold', transform=ax.transAxes)                 
                plt.ylim(bottom=y_lim_LB-0.0005)
                plot_label(fig_num)
                
                # plots of counties in different state on the same figure
                if jj == (len(state_list)-1):
                    dist_type_save = dist_type.replace('/', '-')
                    save_fig_title = 'Restoration plots/other disturbance/{}_{}_{}_{}.png'.format(dist_type_save, state, county, select_start_time[0:10]) 
                    plt.savefig(save_fig_title, dpi=600)
                    plt.show()

#for i_fips in np.arange(len(fips_two_utilities)):
#    outage_data_county_df = select_county_data(fips_code = fips_two_utilities[i_fips],
#                                               LB_count = 50)
#    # save pandas data frame as a csv file
#    outage_data_county_df.to_csv(
#        r'C:\GitHub\Power grid resilience\Power outage data of {county} County in {state}.csv'.format(
#        county=outage_data_county_df['county'].iloc[0], state=outage_data_county_df['state'].iloc[0]),
#        index=None, header=True)         


#UB_normal_outage_count = 2000
# detect the start and end time of all outages
def detect_outage_start_and_end_time(outage_data, time_data=None):
    ''' 
    Detect the start time and end time of an outage.
    args: 
        1. time series; 2. outage count array.
    return: 
        1. start time; 2. end time; 3. maximum outage count outage count is 
        increasing continuously for 4 hours, spanning over about 16 time points.
     ''' 
    if time_data is None:
        time_data = outage_data['run_start_time']
        outage_count_data = outage_data['outage_count']
    else:
        outage_count_data = outage_data.loc[outage_data['run_start_time']==time_data,'outage_count']
        
    n_time_in_outage = 5
    n_time_out_outage = 5
    n_day_before_max_outage = 1
    n_day_after_max_outage = 1
    if type(outage_count_data) != np.ndarray:
        outage_count_data = outage_count_data.to_numpy(dtype='float32')
    county_population = county_popul_df.loc[county_popul_df['fips_code']==
                                                 outage_data['fips_code'].iloc[1], 'population']       
    UB_normal_outage_count = min(county_population.to_numpy()*0.10, 200)
    LB_massive_outage_count = UB_normal_outage_count
    outage_start_time = []
    outage_end_time = []
    step_size = 1
    for ii in range(n_time_in_outage-1, len(time_data)-n_time_in_outage-1, 1):
        # start time satisfies 2 conditions: 
        #   1. 16 proceeding outage counts < median outage count while the follwoing 16 outage counts 
        #      > meadian outage count. (1 record every 15 mins)
        #   2. 16 following outage counts are monotonically increasing.
        #   3. 
        index_before_outage_start = np.arange(ii-n_time_out_outage,ii,step_size)
        index_after_outage_start = np.arange(ii,ii+n_time_in_outage,step_size)
        index_max_after_start = np.arange(ii,ii+n_day_before_max_outage*4*24,step_size)
        index_before_outage_end = np.arange(ii-n_time_in_outage,ii,step_size)
        index_after_outage_end = np.arange(ii,ii+n_time_out_outage,step_size)
        index_max_before_end = np.arange(ii-n_day_after_max_outage*4*24,ii,step_size)
        is_outage_start = (
                np.all(outage_count_data[index_before_outage_start] <= UB_normal_outage_count) and
                np.all(outage_count_data[index_after_outage_start] >= UB_normal_outage_count) and
                np.any(outage_count_data[n_day_before_max_outage] >= LB_massive_outage_count))
#                          (np.all(np.diff(outage_count_data[index_second_half])>=0)
#                              ==1)                               
        is_outage_end = (
                np.all(outage_count_data[index_before_outage_end] >= UB_normal_outage_count) and
                np.all(outage_count_data[index_after_outage_end] <= UB_normal_outage_count) and
                np.any(outage_count_data[n_day_after_max_outage] >= LB_massive_outage_count))
#                        (np.all(np.diff(outage_count_data[index_first_half])<=0)
#                              ==1)
#                        all([a>0, b==0, c<0])        
        if is_outage_start==1:
            outage_start_time.append(time_data.iloc[ii-1])
        if is_outage_end==1:
            outage_end_time.append(time_data.iloc[ii])  
    return outage_start_time, outage_end_time

# detect start and end point in the sample outage data
outage_start_time_list, outage_end_time_list = detect_outage_start_and_end_time(Fulton_GA_outage_data)
outage_start_time, outage_end_time = pd.to_datetime(outage_start_time_list), pd.to_datetime(outage_end_time_list)
print('outage start time: \n', outage_start_time.sort_values(ascending=True))
print('outage end time: \n', outage_end_time.sort_values(ascending=True))


#pd.to_datetime(outage_start_time_list)

# plot restoration curves

outage_start_time = np.asarray(outage_start_time_list)
outage_end_time = np.asarray(outage_end_time_list)

outage_start_time_selected = outage_start_time[[0,1,2]]
outage_end_time_selcted = outage_end_time[[0,1,2]]
outage_start_time_selected[0] = '2019-06-09 12:00:00'
#outage_start_time_selected = outage_start_time[[0,2,4,5]]
#outage_end_time_selcted = outage_end_time[[0,1,2,4]]
#outage_start_time_selected[0] = '2019-07-27 12:00:00'


#fig, ax = plt.subplots(figsize=(10,8))
#colors = mpl.cm.rainbow(np.linspace(0, 1, n_hazard))

def datetime_to_hour(time_datetime):
    return 24*time_datetime.days+time_datetime.seconds/3600

def strtime_to_datetime(time_str):
    return datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')


plt.figure(figsize=(10,8))
colors = mpl.cm.rainbow(np.linspace(0, 1, n_hazard))

for i_time in range(n_hazard):
    # plot
    start_time = outage_start_time_selected[i_time]
    end_time = outage_end_time_selcted[i_time]
#    start_time_datetime = strtime_to_datetime(start_time)
#    end_time_datetime = strtime_to_datetime(end_time)
#    outage_duration = end_time_datetime - start_time_datetime
#    outage_duration_hour = datetime_to_hour(outage_duration)
    outage_time = outage_data_sample['run_start_time'][
                      (outage_data_sample['run_start_time']>=start_time) &
                      (outage_data_sample['run_start_time']<=end_time)]
    
    
    outage_time_datetime = pd.to_datetime(np.asarray(outage_time))
    time_after_outage_datetime = outage_time_datetime - outage_time_datetime[0]
    time_after_outage_hour = datetime_to_hour(time_after_outage_datetime)
#    np.apply_along_axis(datetime_to_hour, 1, time_after_outage_datetime)???
#    np.array([datetime_to_hour(t_i) for t_i in time_after_outage_datetime])
    outage_count = outage_data_sample['outage_count'][
                       (outage_data_sample['run_start_time']>=start_time) &
                       (outage_data_sample['run_start_time']<=end_time)]
    outage_count_arr = outage_count.to_numpy(dtype = 'float32')
    restore_rate = (popul-(outage_count_arr-0))/popul
    plt.scatter(time_after_outage_hour, restore_rate, color=colors[i_time],
                label='%s' % outage_time.iloc[i_time][0:10])
    #UB_normal_outage_temp = 2000
    #plt.hlines(UB_normal_outage_temp,0,200)
    #plt.title('')
plt.ylabel('Restoration rate', fontsize=18,
           fontname='Times New Roman', fontweight='bold')
plt.xlabel('Time after a disaster (hours)', fontsize=18,
           fontname='Times New Roman', fontweight='bold')
plt.xlim(left=0, right=60)
plt.ylim(0.85, 1.02)
plt.legend(loc='best')       
#ax.yaxis.grid() 
plt.show()

print('outage start time: \n', outage_start_time.sort_values(ascending=True))
print('outage end time: \n', outage_end_time.sort_values(ascending=True))



## hurricane
#hurr_data_df_US.loc[hurr_data_df_US['fips_code']==13121,'ISO_time']
#print(hurr_data_df_US.loc[hurr_data_df_US['fips_code']==13121])
#plot_one_outage_data(Fulton_GA_outage_data, start_time='2019-07-08 00:00:00', 
#                     end_time='2019-07-15 16:15:00', LB_count=100)


print('outage start time: \n', outage_start_time.sort_values(ascending=True))
print('outage end time: \n', outage_end_time.sort_values(ascending=True))

outage_start_time = np.asarray(outage_start_time_list)
outage_end_time = np.asarray(outage_end_time_list)

outage_start_time_selected = outage_start_time[[0,1,2]]
outage_end_time_selcted = outage_end_time[[0,1,2]]
outage_start_time_selected[0] = '2019-06-09 12:00:00'

def plot_one_outage_data(outage_data_df, start_time=None, end_time=None, LB_count=1000):
    '''
    Plot outage count data
    args:
        1. start time and end time of outages in a county
        2. county-level power outage data with all attributes in the original data set
    returns:
                
    '''
    if start_time is None:
        start_time = min(outage_data_df['run_start_time'])
    if end_time is None:
        end_time = max(outage_data_df['run_start_time'])    
#    fips_code = outage_data_df['fips_code'][0]
    outage_data_temp = outage_data_df[
                            (outage_data_df['run_start_time']>=start_time) &
                            (outage_data_df['run_start_time']<=end_time) &
#                            (outage_data_df['state']=='District of Columbia') &
#                            (outage_data_df['utility_id'] == 119) &
#                            (outage_data_df['fips_code'] == fips_code) &
                            (outage_data_df['outage_count']>LB_count)]
    
    # plot
    plt.figure(figsize=(10,8))
    index_time = range(len(outage_data_temp['run_start_time']))
    time_temp = outage_data_temp['run_start_time'].iloc[index_time]
    time_temp_datetime = pd.to_datetime(np.asarray(time_temp))
    time_temp_datetime = time_temp_datetime - time_temp_datetime[0]
    time_after_outage_temp_hour = datetime_to_hour(time_temp_datetime)
    
    outage_count_temp = outage_data_temp['outage_count'].iloc[index_time]
    plt.scatter(time_after_outage_temp_hour, outage_count_temp)
    #UB_normal_outage_temp = 2000
    #plt.hlines(UB_normal_outage_temp,0,200)
#    plt.title('Power outage started at %s' % start_time[0:10], fontsize=16,
#               fontname='Times New Roman', fontweight='bold')
    plt.ylabel('Outage count', fontsize=14,
               fontname='Times New Roman', fontweight='bold')
    plt.xlabel('Time after an event (hours)', fontsize=14,
               fontname='Times New Roman', fontweight='bold')
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    ax = plt.axes()        
    # horizontal grid lines
    ax.yaxis.grid() 
#    plt.show()

#for i_time in range(len(outage_start_time_selected)):
#    print('Outage start time: ',outage_start_time_selected[i_time])
#    plot_one_outage_data(outage_start_time_selected[i_time], outage_end_time_selcted[i_time])
#
#start_time = outage_start_time_selected[i_time]
#end_time = outage_end_time_selcted[i_time]
#outage_start_time_selected: 
#array(['2019-07-27 19:00:00', '2015-02-15 03:45:00',
#       '2016-07-15 21:30:00', '2018-03-02 08:15:00'], dtype='<U19')
#
#outage_end_time_selcted: 
#array(['2019-07-29 01:45:00', '2015-02-16 00:45:00',
#    '2016-07-16 08:45:00', '2018-03-04 10:30:00'], dtype='<U19')    
# outage start time:  ['2019-07-26 12:45:00', '2015-02-15 03:30:00', 
# '2016-07-15 21:15:00', '2018-03-02 08:00:00']
# outage end time:  ['2019-07-29 02:00:00', '2015-02-16 01:00:00', 
#'2016-07-16 09:00:00', '2018-03-04 10:45:00']

# given time and fips, find the event type

def find_disruption(outage_data_county_df, fips_code=None, outage_time=None): # to do: use end time or simply time?
# select the hurricane or storm that occured on the same day of a county    
# input:
    # outage_datetime: the time of outage, a datatime variable
# output:
    # hurricane or not and hurricane name; other event or not and event name
    if fips_code is None:
        fips_code = outage_data_county_df['fips_code'][0]
    if outage_time is None:
        outage_time = outage_data_county_df['run_start_time']
    outage_datetime_array = pd.to_datetime(outage_time)
    for i_time in np.arange(0, len(outage_datetime_array)):
        outage_datetime = outage_datetime_array[i_time]
#        hurr_name_series = hurr_data_df_US.loc[
#            hurr_data_df_US['ISO_time'].dt.date==outage_datetime.date(),'Name']
#        if hurr_name_series.shape[0] >= 1:
#            hurr_name = hurr_name_series.iloc[0]
#            hurr_data_select = hurr_data_df_US[hurr_data_df_US['Name']==hurr_name] # could select coordinates only
#            for i_traj in np.arange(0, hurr_data_select.shape[1]):
#                dist_hurricane_to_county = cal_geo_distance(
#                    hurr_data_select.Latitude, hurr_data_select.Longitude,
#                    county_coor_fips_df.loc[county_coor_fips_df['FIPS']==fips_code, 'Latitude'].to_numpy(),
#                    county_coor_fips_df.loc[county_coor_fips_df['FIPS']==fips_code, 'Longitude'].to_numpy())
#                if np.amin(dist_hurricane_to_county) < 200:
#                # assign it to the time period in the county outage data
#                    outage_data_county_df[outage_data_county_df['run_start_time']==outage_datetime,
#                                          'hurr_boo'] = 1
#                    outage_data_county_df[outage_data_county_df['run_start_time']==outage_datetime,
#                                          'hurr_name'] = hurr_name
        
        other_event_name = weather_data_df.loc[
            (weather_data_df['BEGIN_DATE_TIME'].dt.date<=outage_datetime.date())&
            (weather_data_df['END_DATE_TIME'].dt.date>=outage_datetime.date())&
            (weather_data_df['fips_code']==fips_code),'EVENT_TYPE']
        
        if other_event_name.shape[0] >= 1:
#            outage_data_county_df.loc[outage_data_county_df['run_start_time']==outage_datetime, 
#                                      'other_event_boo'] = 1
            outage_data_county_df.loc[outage_data_county_df['run_start_time']==outage_datetime, 
                                      'event_type'] = other_event_name.iloc[0]
    return 


def plot_outage_weather(fips_code, t_landfall, event_type=None, LB_count=100):

#county_hurr = county_weather_data.loc[county_weather_data['EVENT_TYPE']=='Hurricane'] 

# match outage and weather using date and time
    if county_weather_data.shape[0]>0:
        for ii in np.arange(0, county_weather_data.shape[0]):
            end_to_start_hr= datetime_to_hour(county_weather_data['END_DATE_TIME'].iloc[ii]-
                                              county_weather_data['BEGIN_DATE_TIME'].iloc[ii])
            if end_to_start_hr >=1:
                event_start_time = county_weather_data['BEGIN_DATE_TIME'].astype(str).iloc[ii]
                event_end_time = county_weather_data['END_DATE_TIME'].astype(str).iloc[ii]
                event_type = county_weather_data['EVENT_TYPE'].iloc[ii].lower()
                
#                event_start_time = '2016-10-07 23:00:00'
                
                # detect outage start time and end time
                # outage start time is later than event start time; outage end time is later than event end time
                delay_end_day = 5
                select_start_time = event_start_time
                select_end_time = pd.to_datetime(event_end_time) + timedelta(days=delay_end_day)
                select_end_time = select_end_time.strftime("%Y-%m-%d %H:%M:%S") 
#                select_start_time = '2016-10-07 23:00:00'
#                select_end_time = '2017-09-01-18:00:00'
                outage_data_select = county_outage_data[
                            (county_outage_data['run_start_time']>=select_start_time) &
                            (county_outage_data['run_start_time']<=select_end_time) & #'2017-09-15 09:00:00'
                            (county_outage_data['outage_count']>LB_count)]           
#                outage_start_time_list, outage_end_time_list = detect_outage_start_and_end_time(outage_data_select)
#                outage_start_time, outage_end_time = pd.to_datetime(outage_start_time_list), pd.to_datetime(outage_end_time_list)
#                print('outage start time: \n', outage_start_time.sort_values(ascending=True))
#                print('outage end time: \n', outage_end_time.sort_values(ascending=True) 
                
                if outage_data_select.shape[0]>=1:
                    print(ii)
#                    print('Event type: ', county_weather_data['EVENT_TYPE'].iloc[ii])
#                    print(outage_data_select)
#                    print('outage start time: \n', outage_start_time.sort_values(ascending=True))
#                    print('outage end time: \n', outage_end_time.sort_values(ascending=True))
#                    print('Outages may occur due to weather events that takes places a few days earlier')

                    index_time = range(len(outage_data_select['run_start_time']))
                    time_temp = outage_data_select['run_start_time'].iloc[index_time]
                    time_temp_datetime = pd.to_datetime(np.asarray(time_temp))
                    time_temp_datetime = time_temp_datetime - time_temp_datetime[0]
                    time_after_outage_temp_hour = datetime_to_hour(time_temp_datetime)
                    
                    outage_count_temp = outage_data_select['outage_count'].iloc[index_time]                   
                    outage_count_arr = outage_count_temp.to_numpy(dtype = 'float32')
                    
                    popul = county_weather_data['population'].iloc[1]
                    print(popul)
                    UB_outage_count_flood = max(popul*0.05, 500)
                    if ((event_type=='flood')) & (np.max(outage_count_arr) >= UB_outage_count_flood):
                        event_type='hurricane'
                    event_type = 'thunderstorm wind'                
                    restore_rate = (popul-outage_count_arr)/popul
#                    print('restoration rate',restore_rate)

                    # plot restoration rate
                    plt.figure(figsize=(10, 8))    
#                    plt.scatter(time_after_outage_temp_hour, outage_count_temp)
                    plt.scatter(time_after_outage_temp_hour, restore_rate)
                    #UB_normal_outage_temp = 2000
                    #plt.hlines(UB_normal_outage_temp,0,200)
                #    plt.title('Power outage started at %s' % start_time[0:10], fontsize=16,
                #               fontname='Times New Roman', fontweight='bold')
                    plt.ylabel('Outage count', fontsize=14,
                               fontname='Times New Roman', fontweight='bold')
                    plt.xlabel('Time after an event (hours)', fontsize=14,
                               fontname='Times New Roman', fontweight='bold')
                    plt.ylim(bottom=np.min(restore_rate),top=1)
                    plt.xlim(left=0)
                    ax = plt.axes()        
                    # horizontal grid lines
#                    ax.yaxis.grid() 
    #                plt.axvline(x=end_to_start_hr, color='darkred')
                    plt.text(0.53,0.13,'Event type: {}'.format(event_type), 
                             fontsize=14, color='darkblue', transform=ax.transAxes)
                    plt.text(0.53,0.09, 'Event started at: {}'.format(event_start_time),
                             fontsize=14, color='black', transform=ax.transAxes)
    #                plt.text(0.48,0.85, 'Event ended after: {} hrs'.format(end_to_start_hr),
    #                         fontsize=14, color='darkred', transform=ax.transAxes)
                    plt.show()
                                        
                    # plot outage counts
                    plt.figure(figsize=(10, 8))    
                    plt.scatter(time_after_outage_temp_hour, outage_count_temp)
#                    plt.scatter(time_after_outage_temp_hour, restore_rate)
                    #UB_normal_outage_temp = 2000
                    #plt.hlines(UB_normal_outage_temp,0,200)
                #    plt.title('Power outage started at %s' % start_time[0:10], fontsize=16,
                #               fontname='Times New Roman', fontweight='bold')
                    plt.ylabel('Restoration rate', fontsize=14,
                               fontname='Times New Roman', fontweight='bold')
                    plt.xlabel('Time after an event (hours)', fontsize=14,
                               fontname='Times New Roman', fontweight='bold')
                    plt.ylim(bottom=0)
                    plt.xlim(left=0)
                    ax = plt.axes()        
                    # horizontal grid lines
#                    ax.yaxis.grid() 
    #                plt.axvline(x=end_to_start_hr, color='darkred')
                    plt.text(0.53,0.93,'Event type: {}'.format(event_type), 
                             fontsize=14, color='darkblue', transform=ax.transAxes)
                    plt.text(0.53,0.89, 'Event started at: {}'.format(event_start_time),
                             fontsize=14, color='black', transform=ax.transAxes)
    #                plt.text(0.48,0.85, 'Event ended after: {} hrs'.format(end_to_start_hr),
    #                         fontsize=14, color='darkred', transform=ax.transAxes)
                    plt.show()
        #        savefig()
    else:
        print('No extreme weather data about {}'.format())


###############################################################################
# Irma August 30, 2017 – September 14, 2017
###############################################################################
# hurricane Irma: hurricane made a second landfall in 
# Florida on Marco Island at 19:35 UTC with winds of 115 mph (185 km/h)
# Affected all the counties in florida     
# Irma: 
# t_landfall = '2017-09-10 14:35:00'

# impoart affected counties data
county_affected_hurr_df = pd.read_csv('county_impacted_by_hurricanes.csv') 

county_affected_storm_df = pd.read_csv('counties impacted by storms and flooding.csv')
# remove empty space at two ends
county_affected_hurr_df.county = county_affected_hurr_df.county.str.strip()

county_affected_storm_df.county = county_affected_storm_df.county.str.strip()
county_affected_storm_df.state = county_affected_storm_df.state.str.strip()
# change time format
county_affected_hurr_df['t_landfall'] = pd.to_datetime(county_affected_hurr_df['t_landfall']).dt.strftime("%Y-%m-%d %H:%M:%S")

county_affected_storm_df.t_start = pd.to_datetime(county_affected_storm_df.t_start).dt.strftime("%Y-%m-%d %H:%M:%S")
county_affected_storm_df.t_end = pd.to_datetime(county_affected_storm_df.t_end).dt.strftime("%Y-%m-%d %H:%M:%S")


# fill nan values in storm data
i_state = 0
while i_state < county_affected_storm_df.shape[0]:
    if pd.isnull(county_affected_storm_df.state.iloc[i_state]):
        county_affected_storm_df.state.iloc[i_state] = county_affected_storm_df.state.iloc[i_state-1]
        county_affected_storm_df.t_start.iloc[i_state] = county_affected_storm_df.t_start.iloc[i_state-1]
        county_affected_storm_df.t_end.iloc[i_state] = county_affected_storm_df.t_end.iloc[i_state-1]
        county_affected_storm_df.disaster_id.iloc[i_state] = county_affected_storm_df.disaster_id.iloc[i_state-1]
    else:
        i_state += 1

# match fips code
        
def mathch_fips_code(county_affected_df):
    county_affected_df['fips_code'] = np.nan
    for ii in np.arange(county_affected_df.shape[0]):
#        print(ii)
        county_temp = county_affected_df.loc[ii,'county']
        state_temp = county_affected_df.loc[ii,'state']
        county_affected_df['fips_code'].iloc[ii] = county_coor_fips_df.loc[
                (county_coor_fips_df['state']==state_temp)&
                (county_coor_fips_df['county']==county_temp), 'fips_code'].values
    return county_affected_df

county_affected_storm_df = mathch_fips_code(county_affected_storm_df)
    
county_affected_hurr_df = mathch_fips_code(county_affected_hurr_df)


# plot outage restoratio after hurricanes      
hurr = county_affected_hurr_df['hurricane'].unique()
for i_hurr in np.array([3]):
    county_affected_hurr_df_hurr = county_affected_hurr_df.loc[county_affected_hurr_df['hurricane']==hurr[i_hurr]]            
    for i_county in np.arange(county_affected_hurr_df_hurr.shape[0]):
        fips_code = int(county_affected_hurr_df_hurr['fips_code'].iloc[i_county])
        t_landfall = county_affected_hurr_df_hurr['t_landfall'].astype(str).iloc[i_county]
        county_outage_data_curr = select_county_data(fips_code)
        plot_outage_hurr(fips_code, county_outage_data=county_outage_data_curr,
                         event_name=hurr[i_hurr], event_start_time = t_landfall)
        time.sleep(0.01)
        print('Next iter #: {}'.format(i_county+1))

# plot outage restoration after storms       
state_affected = county_affected_storm_df['state'].unique()
for i_state in np.arange(len(state_affected)):
    county_affected_each_state = county_affected_storm_df.loc[county_affected_storm_df['state']==state_affected[i_state]]            
    for i_county in np.arange(county_affected_each_state.shape[0]):
        fips_code = int(county_affected_each_state['fips_code'].iloc[i_county])
        t_start = county_affected_each_state['t_start'].astype(str).iloc[i_county]
        county_outage_data_curr = select_county_data(fips_code)
        if county_outage_data_curr.empty == False:
            plot_outage_hurr(fips_code, county_outage_data=county_outage_data_curr,
                             event_type='storm', event_start_time = t_start, LB_count=1)
#        time.sleep(0.01)
        print('Next iter #: {}'.format(i_county+1))

              
def get_fips_code(county, state):
    fips_code = county_affected_hurr_df.loc[(county_affected_hurr_df['state']==state) & 
                                       (county_affected_hurr_df['county']==county), 'fips_code']
    return int(fips_code)

fips_code = get_fips_code('Assumption', 'Louisiana')


# function used to check odd outage plots
def plot_outage_hurr_ext_time(fips_code, county_outage_data=county_data, event_start_time=None,
                              event_end_time=None, event_name = None, hurr_curr=None):
    days_before_start = 15
    days_after_start = 25
    event_start_time_ext = pd.to_datetime(event_start_time) - timedelta(days=days_before_start)
    event_start_time_ext = event_start_time_ext.strftime("%Y-%m-%d %H:%M:%S")
    event_end_time_ext = pd.to_datetime(event_start_time) + timedelta(days=days_after_start)
    event_end_time_ext = event_end_time_ext.strftime("%Y-%m-%d %H:%M:%S")
    
    county_data = select_county_data(fips_code, start_time=event_start_time_ext, 
                                     end_time=event_end_time_ext)
    plot_outage_hurr(fips_code, county_outage_data=county_data, event_start_time=event_start_time_ext, 
                     event_end_time=event_end_time_ext, event_name = hurr_curr)       
# select county outage data and weather data
fips_code = 47157
county_outage_data = select_county_data(fips_code)
county_weather_data = weather_data_df.loc[
        (weather_data_df['fips_code'] == fips_code) &
        (weather_data_df['BEGIN_DATE_TIME'] >= min(county_outage_data['run_start_time'])) &
        (weather_data_df['END_DATE_TIME'] <= max(county_outage_data['run_start_time']))]


def plot_outage_hurr(fips_code, county_outage_data, event_start_time, event_type=None,
                     event_name=None, event_end_time=None, y_LB=None, LB_count=0):

# match outage and weather using date and time
#    if county_weather_data.shape[0]>0:
#        for ii in np.arange(0, county_weather_data.shape[0]):
#            end_to_start_hr= datetime_to_hour(county_weather_data['END_DATE_TIME'].iloc[ii]-
#                                              county_weather_data['BEGIN_DATE_TIME'].iloc[ii])
#            if end_to_start_hr >=1:
#                event_start_time = county_weather_data['BEGIN_DATE_TIME'].astype(str).iloc[ii]
#                event_end_time = county_weather_data['END_DATE_TIME'].astype(str).iloc[ii]
#                event_type = county_weather_data['EVENT_TYPE'].iloc[ii].lower()
                
#                event_start_time = '2016-10-07 23:00:00'
                
    # detect outage start time and end time
    # outage start time is later than event start time; outage end time is later than event end time
    if event_type == 'hurricane':
        event_name_full = 'Hurricane {}'.format(event_name)
        delay_end_day = 10
    elif event_type == 'storm':
        event_name_full = 'Storms'
        delay_end_day = 20
    elif event_type == 'dist':
        event_name_full = '{}'.format(event_name)
        delay_end_day = 3
        
    
    select_start_time = event_start_time
    if event_end_time==None:
        select_end_time = pd.to_datetime(select_start_time) + timedelta(days=delay_end_day)
        select_end_time = select_end_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        select_end_time = event_end_time
#    select_end_time = select_end_time.strftime("%Y-%m-%d %H:%M:%S") 
#                select_start_time = '2016-10-07 23:00:00'
#                select_end_time = '2017-09-01-18:00:00'
    outage_data_select = county_outage_data[
                (county_outage_data['run_start_time'] >= select_start_time) &
                (county_outage_data['run_start_time'] <= select_end_time) & 
                (county_outage_data['outage_count'] >= LB_count)]           
#                outage_start_time_list, outage_end_time_list = detect_outage_start_and_end_time(outage_data_select)
#                outage_start_time, outage_end_time = pd.to_datetime(outage_start_time_list), pd.to_datetime(outage_end_time_list)
#                print('outage start time: \n', outage_start_time.sort_values(ascending=True))
#                print('outage end time: \n', outage_end_time.sort_values(ascending=True) 
    
    if outage_data_select.shape[0] >= 1:
#                    print('Event type: ', county_weather_data['EVENT_TYPE'].iloc[ii])
#                    print(outage_data_select)
#                    print('outage start time: \n', outage_start_time.sort_values(ascending=True))
#                    print('outage end time: \n', outage_end_time.sort_values(ascending=True))
#                    print('Outages may occur due to weather events that takes places a few days earlier')

        index_time = range(len(outage_data_select['run_start_time']))
        time_temp = outage_data_select['run_start_time'].iloc[index_time]
#        date_datetime = [datetime.strptime(d,"%Y-%m-%d %H:%M:%S") for d in time_temp]
#        time_temp_datetime = pd.to_datetime(time_temp)
#        date_temp = time_temp_datetime.dt.date
        # usethe follwoing code if the x-axis is the hours after the start of the event
#        time_temp_datetime = pd.to_datetime(np.asarray(time_temp))
#        time_diff_datetime = time_temp_datetime - time_temp_datetime[0]
#        time_after_outage_temp_hour = datetime_to_hour(time_diff_datetime)
        
        # !!! to do: change the variable name in county_population.csv to fips_code and population later       
        popul = int(county_popul_df.loc[county_popul_df['fips_code']==fips_code, 'population'].iloc[0])
        # population is from 2014
        county = outage_data_select['county'].iloc[0]
        state = outage_data_select['state'].iloc[0]
#        coor = county_coor_fips_df.loc[county_coor_fips_df['fips_code']==fips_code] , 'county'].iloc[0]
#        print(popul)
#        LB_outage_count_hurr = min(popul*0.03, 5000)
        
        # plot
        # combine outage
        outage_count_comb = outage_data_select['outage_count'].iloc[index_time]                   
        outage_count_comb_arr = outage_count_comb.to_numpy(dtype = 'float32')
        restore_rate_comb = (popul-outage_count_comb_arr)/popul
        
        if (np.min(restore_rate_comb) <= 0.975) | (np.max(outage_count_comb_arr)>2000):
        
            if np.min(restore_rate_comb)<=0:
                print('Population, {}, < max outage count, {}'.format(popul, np.min(outage_count_comb_arr)))
                print('{}, {}'.format(county, state))
                print('Fips code: {}'.format(fips_code))
                time.sleep(10)
            else:
                fig = plt.figure(1, figsize=(10, 8))    
        #                    plt.scatter(time_after_outage_temp_hour, outage_count_temp)
#                date_temp = drange(date1, date2, delta)
#                plt.plot_date(date_datetime, restore_rate_comb)              
#                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
#                plt.gca().xaxis.set_major_locator(mdates.DayLocator())
                
                restore_rate_comb_ave = pd.Series(restore_rate_comb).rolling(window=win_len).mean()
                restore_rate_comb_ave[0:(win_len-1)] = restore_rate_comb[0:(win_len-1)]
                
                # find the start point and restoration point
                restore_LB = 0.99998
                norm_LB = restore_LB
                out_UB = 0.99995
                id_first_min_duration = 4*24*3  # reach minimum in 3 days
                id_restore_rate_min = restore_rate_comb_ave[:id_first_min_duration].idxmin()
                # the first point that is < LB
                # the first point that is > LB after the minimum
                id_start = 0
                for ii in np.arange(id_restore_rate_min - win_len):
                    if (restore_rate_comb_ave[ii] >= norm_LB) & (restore_rate_comb_ave[ii+win_len] <= out_UB):
                        id_start = ii
                        break                
                # the first point that is > LB after the minimum
                id_restore = restore_rate_comb_ave.size
                for ii in np.arange(id_restore_rate_min - win_len, restore_rate_comb_ave.size):
                    if (restore_rate_comb_ave[ii-win_len] <= restore_LB) & (restore_rate_comb_ave[ii] >= restore_LB):
                        id_restore = ii
                        break
#                restore_rate_comb_ave_seletc =  restore_rate_comb_ave[id_start:id_restore]                 

                df_temp = pd.DataFrame(columns=['Date','Restoration rate'], index=range(len(time_temp)))
                df_temp['Date'] = pd.to_datetime(np.asarray(time_temp))
                df_temp['Restoration rate'] = restore_rate_comb_ave
#                plt.figure()
                plt.plot(df_temp['Date'][id_start:id_restore], df_temp['Restoration rate'][id_start:id_restore])
                plt.xlim(right = df_temp['Date'][id_restore])
#                plt.ylim(bottom = 0.70)
                if y_LB == None:
                    y_LB = restore_rate_comb_ave[id_restore_rate_min]
                plt.ylim(bottom=y_LB)
                ax = plt.gca()
                ax.get_yaxis().get_major_formatter().set_useOffset(False)
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(interval_multiples=True))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y \n %H:%M'))
                plt.gcf().autofmt_xdate() # Rotation
#                plt.grid(True)
#                plt.show()
                #UB_normal_outage_temp = 2000
                #plt.hlines(UB_normal_outage_temp,0,200)
            #    plt.title('Power outage started at %s' % start_time[0:10], fontsize=16,
            #               fontname='Times New Roman', fontweight='bold')
                plt.ylabel('Restoration rate', fontsize=14,
                           fontname='Times New Roman', fontweight='bold')
                plt.xlabel('Date', fontsize=14, #Time after an event (hours)
                           fontname='Times New Roman', fontweight='bold')
#                plt.xlim(left=0)
                ax = plt.axes()        
                # horizontal grid lines
        #                    ax.yaxis.grid() 
        #                plt.axvline(x=end_to_start_hr, color='darkred')
                x_loc = 0.65
                
                plt.text(x_loc,y_loc+0.07,'County: {}, {}'.format(county,state),
                         fontsize=font_text, color='black', transform=ax.transAxes)
                plt.text(x_loc, y_loc+0.035,'Event: {}'.format(event_name_full), 
                         fontsize=font_text, color='black', transform=ax.transAxes) 
                plt.text(x_loc,y_loc, 'Event start time: {}'.format(event_start_time),
                         fontsize=font_text, color='black', transform=ax.transAxes) 
        #                plt.text(0.48,0.85, 'Event ended after: {} hrs'.format(end_to_start_hr),
        #                         fontsize=14, color='darkred', transform=ax.transAxes)               
                # add outages of each utility if there are more than one
                if event_type == 'hurricane':
                    hurr_curr = county_affected_hurr_df_hurr['hurricane'].iloc[0]
                    save_fig_title = 'Restoration plots/{}_{}_{}.png'.format(hurr_curr, state, county)                
                elif event_type == 'storm':
                    save_fig_title = 'Restoration plots/{}_{}_{}.png'.format(state, county, event_start_time[0:11])
                elif event_type == 'dist':
                    save_fig_title = 'Restoration plots/{}_{}_{}_{}.png'.format('dist', state, county,
                                                                                event_start_time[0:10])        
                #        Output_path =  os.path.join(dir_path, save_fig_title) 
                if 'n_utility' in outage_data_select.columns:
                    n_utility = outage_data_select['n_utility'].iloc[0]
                    utility_name = ['All combined']
                    colors = iter(cm.rainbow(np.linspace(0, 1, n_utility)))
                    plt.figure(1)
                    for ii in np.arange(n_utility):
                        count_name_temp = 'outage_count_{}'.format(ii+1)
                        outage_count_utility_i = outage_data_select[count_name_temp].iloc[index_time]                   
                        outage_count_utility_i_arr = outage_count_utility_i.to_numpy(dtype='float32')
    #                    cond_LB = np.max(outage_count_utility_i_arr) >= 0.001*np.max(outage_count_comb_arr)
    #                    cond_UB = np.max(outage_count_utility_i_arr) <= 0.999*np.max(outage_count_comb_arr)
    #                    if cond_LB and cond_UB:
                        utility_name.append('Utility {}'.format(ii+1))
                        restore_rate_utility_i = (popul - outage_count_utility_i_arr)/popul
                        restore_rate_utility_i_ave = pd.Series(restore_rate_utility_i).rolling(window=win_len).mean()
                        
                        df_temp_i = pd.DataFrame(columns=['Date','Restoration rate'], index=range(len(time_temp)))
                        df_temp_i['Date'] = pd.to_datetime(np.asarray(time_temp))
                        df_temp_i['Restoration rate'] = restore_rate_utility_i_ave                                            
                        plt.plot(df_temp_i['Date'][id_start:id_restore], df_temp_i['Restoration rate'][id_start:id_restore],
                                 color=next(colors))
                    utility_name = np.asarray(utility_name)    
                    plt.legend(utility_name, bbox_to_anchor=(x_loc-0.024, y_loc+0.08), frameon=False,
                               loc="lower left", fontsize=font_text, fancybox=True, framealpha=0.5)
                    plt.savefig(save_fig_title, dpi=1000)
                    plt.show()
                else:
                    plt.text(x_loc, y_loc+0.12, 'One utility only', fontsize=font_text,
                             color='black', transform=ax.transAxes) 
                    plt.savefig(save_fig_title, dpi=1000)
                    plt.show()
        else:
            print('No weather-induced outages occured in this county with fips code {} \nOr the data were missing'.
                  format(fips_code))
    #                plt.show()
    #            #UB_normal_outage_temp = 2000
    #            #plt.hlines(UB_normal_outage_temp,0,200)
    #        #    plt.title('Power outage started at %s' % start_time[0:10], fontsize=16,
    #        #               fontname='Times New Roman', fontweight='bold')
    #            plt.ylabel('Restoration rate', fontsize=14,
    #                       fontname='Times New Roman', fontweight='bold')
    #            plt.xlabel('Time after an event (hours)', fontsize=14,
    #                       fontname='Times New Roman', fontweight='bold')
    #            plt.ylim(bottom=np.min(restore_rate),top=1)
    #            plt.xlim(left=0)
    #            ax = plt.axes()        
    #            # horizontal grid lines
    #    #                    ax.yaxis.grid() 
    #    #                plt.axvline(x=end_to_start_hr, color='darkred')
    #            plt.text(0.53,0.17,'Event type: {}'.format(event_name_full), 
    #                     fontsize=14, color='darkblue', transform=ax.transAxes)
    #            plt.text(0.53,0.14, 'Event started at: {}'.format(event_start_time),
    #                     fontsize=14, color='black', transform=ax.transAxes)
    #            plt.text(0.53,0.11, 'County: {}'.format(county),
    #                     fontsize=14, color='black', transform=ax.transAxes)
    #    #                plt.text(0.48,0.85, 'Event ended after: {} hrs'.format(end_to_start_hr),
    #    #                         fontsize=14, color='darkred', transform=ax.transAxes)
                                
    #        # plot outage counts
    #        plt.figure(figsize=(10, 8))    
    #        plt.scatter(time_after_outage_temp_hour, outage_count_temp)
    ##                    plt.scatter(time_after_outage_temp_hour, restore_rate_comb)
    #        #UB_normal_outage_temp = 2000
    #        #plt.hlines(UB_normal_outage_temp,0,200)
    #    #    plt.title('Power outage started at %s' % start_time[0:10], fontsize=16,
    #    #               fontname='Times New Roman', fontweight='bold')
    #        plt.ylabel('Restoration rate', fontsize=14,
    #                   fontname='Times New Roman', fontweight='bold')
    #        plt.xlabel('Time after an event (hours)', fontsize=14,
    #                   fontname='Times New Roman', fontweight='bold')
    #        plt.ylim(bottom=0)
    #        plt.xlim(left=0)
    #        ax = plt.axes()        
    #        # horizontal grid lines
    ##                    ax.yaxis.grid() 
    ##                plt.axvline(x=end_to_start_hr, color='darkred')
    #        plt.text(0.53,0.93,'Event type: {}'.format(event_name_full), 
    #                 fontsize=14, color='darkblue', transform=ax.transAxes)
    #        plt.text(0.53,0.89, 'Event started at: {}'.format(event_start_time),
    #                 fontsize=14, color='black', transform=ax.transAxes)
    ##                plt.text(0.48,0.85, 'Event ended after: {} hrs'.format(end_to_start_hr),
    ##                         fontsize=14, color='darkred', transform=ax.transAxes)
    #        plt.show()
    #        dir_path = os.getcwd()


###############################################################################
### winter storms






# hurricane data: time, coordinates of hurricane trajectory, wind speed
# 1. coordinates mapped to county and county fips

# counties that each hurricane passed through in its course

## hurricane data
#hurr_data_df = pd.read_csv('hurricane data.csv', header=0)
#
## remove points that are located further south than key west, FL (24.5551, -81.7800)
## or further east than Yarmouth, Canada (43.8375, -66.1174)
#
#hurr_data_df_US = hurr_data_df.loc[(hurr_data_df.Latitude > 24) & (hurr_data_df.Longitude < -66)]
## create columns to county info
#hurr_data_df_US = hurr_data_df_US.reindex(columns=[*hurr_data_df_US.columns.tolist(), 
#                                          'county', 'state', 'fips_code','distance_to_county'], fill_value='')
#hurr_data_df_US.index = range(len(hurr_data_df_US.index))
#
## fill 'ISO_time' entries that do not have 'month/day/year'
#index_miss_boo = (hurr_data_df_US['ISO_time'].str.len()==7) | (hurr_data_df_US['ISO_time'].str.len()==8)
#hurr_date_temp = ''
#for ii in np.arange(0,hurr_data_df_US.shape[0]):
#    if index_miss_boo.iloc[ii] == False: # date is not missing, but ':00' is missing
#        hurr_data_df_US.loc[ii,'ISO_time'] = hurr_data_df_US.loc[ii,'ISO_time'] + ':00'
#        hurr_date_time = hurr_data_df_US.loc[ii,'ISO_time']
#        hurr_date_temp = hurr_date_time.split()[0]
#    else:
#        hurr_time = hurr_data_df_US.loc[ii,'ISO_time']
#        hurr_data_time = " ".join((hurr_date_temp, hurr_time))
#        hurr_data_df_US.loc[ii,'ISO_time'] = hurr_data_time

## unify time format to datetime format %Y%m%d %H:%M:%S, e.g. '2019-07-27 12:00:00'
#hurr_data_df_US['ISO_time'] = pd.to_datetime(hurr_data_df_US['ISO_time']) 

## county data
#
## find the county that is less than 50 km from the trajectory point of each hurricane
## county data
## '[]' in the county name have been removed
#county_coor_fips_df = pd.read_csv('county fips and coordinates.csv')
#
#county_coor_fips_df.Latitude = county_coor_fips_df.Latitude.replace('\°','', regex=True).astype(float)
## the orginal Longitude has "–" instead of "-", making it hard to convert string to float
#county_coor_fips_df.Longitude = county_coor_fips_df.Longitude_abs.replace('\°','', regex=True).astype(float)
#county_coor_fips_df.Longitude = -1*county_coor_fips_df.Longitude
#
## calculate distance between trajectory point in hurricane data and counties
#
#def cal_geo_distance(lat1,lon1,lat2,long2):
#    # approximate radius of earth in km
#    R = 6373.0
#    lat1_r = np.radians(lat1)
#    lon1_r = np.radians(lon1)
#    lat2_r = np.radians(lat2)
#    lon2_r = np.radians(long2)
#    dlon = lon2_r - lon1_r
#    dlat = lat2_r - lat1_r 
#    a = np.sin(dlat / 2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2)**2
#    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#    geo_distance = R * c
#    return geo_distance
#
#
#for i_traj in np.arange(0,hurr_data_df_US.shape[0]):
#    # calculate the distance between a trajectory point and center of counties
#    dist_hurricane_to_county = cal_geo_distance(hurr_data_df_US.Latitude.iloc[i_traj],
#                                                hurr_data_df_US.Longitude.iloc[i_traj],
#                                                county_coor_fips_df.Latitude.to_numpy(),
#                                                county_coor_fips_df.Longitude.to_numpy())
#    if np.amin(dist_hurricane_to_county) < 150:
#        i_county_in_traj = np.argmin(dist_hurricane_to_county) 
#        # get the county name, state name, and fips code
#        # assign it to the trajectory
#        hurr_data_df_US.loc[i_traj,'county'] = county_coor_fips_df.loc[i_county_in_traj,'County']
#        hurr_data_df_US.loc[i_traj,'state'] = county_coor_fips_df.loc[i_county_in_traj,'State']
#        hurr_data_df_US.loc[i_traj,'fips_code'] = county_coor_fips_df.loc[i_county_in_traj,'FIPS']
#        hurr_data_df_US.loc[i_traj,'distance_to_county'] = np.amin(dist_hurricane_to_county)


#find_disruption(Fulton_GA_outage_data)
#
## next step:
#       # 1.2 counties subject to hurricane only or storm only. Compare the patterns of two counties
#       
#       hurr_only_fips = pd.Series(list(set(hurr_fips).difference(set(storm_fips)))) 
#       
#       
#       storm_only_fips = pd.Series(list(set(storm_fips).difference(set(hurr_fips))))


#def determine_hazard_subject(fips_code): 
#    if any(hurr_and_storm_fips  == fips_code) == True:
#        print('Subject to both winter storms and hurricanes')
#    elif any(hurr_only_fips  == fips_code) == True:
#        print('Subject to hurricanes only')
#    elif any(storm_only_fips  == fips_code) == True:
#        print('Subject to winter storms only')
#    else:
#        print('Subject to other hazards')     
#        
        
   # 2. how to better detect the start and end time point of outages
       # add the maximum value of power outages, say 20000 outage count between start and end time.