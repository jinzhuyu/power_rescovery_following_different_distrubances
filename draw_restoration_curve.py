# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:36:08 2020
@author: Jinzhu Yu, Vanderbilt University

"""

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import (
        YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange)
from matplotlib import cm # rainbow color scheme
from datetime import datetime, timedelta
import time
import json 
import copy
#import dask.dataframe
import tkinter as tk
from tkinter import *
import os
os.chdir('C:\GitHub\Power_grid_resilience')


#%%
# select power outage data using state and county
def select_county_data(outage_data_df, county, state, start_time, end_time):
    '''
    args: fips code is unique for each county and can be found in original data: outage_data_df  
    out:  county outage data DataFrame with utility_name (named as comb_utility), 
          fips_code, county, state, outage_count, run_start_time.
    '''
    
    out_data_county_df = outage_data_df[
        (outage_data_df['run_start_time']>=start_time) & (outage_data_df['run_start_time']<=end_time) &
        (outage_data_df['county']==county) & (outage_data_df['state']==state)]
    out_data_county_df = out_data_county_df.sort_values(by='run_start_time') 
    
    len_time = len(out_data_county_df['run_start_time'])
    num_nan = out_data_county_df['outage_count'].isna().sum()
    
    if (out_data_county_df.empty==True):
        print('\n{}County, {}: data are missing (data selection stage)'.format(county, state))
        return pd.DataFrame()
    elif num_nan>=round(len_time*0.75,0):
        print('\{}County, {}: over 75% of data are missing (data selection stage)'.format(county, state))
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
#            print('# of utility: {}'.format(n_util))
            return out_count_comb_df


# add outage count of each utility to the plot if there are more than one
def plot_all_util(county_outage_data, state, county, popul, event_start_time,
                  index_time, time_temp_datetime, fig_num):
    
    save_fig_title = 'Restoration plots/GUI_plots/{}_{}_{}.png'.format(state, county, event_start_time[0:10])        
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

# label the x and y axis and format the date and time
def plot_label(fig_num):
    
    plt.figure(fig_num)
    plt.ylabel('Restoration rate', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xlabel('Date and time', fontsize=14, fontname='Times New Roman', fontweight='bold')
    
    ax = plt.gca()
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(interval_multiples=True))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y \n %H:%M'))
    plt.gcf().autofmt_xdate()
    
    plt.yticks(fontsize=font_text, fontname='Times New Roman', fontweight='bold')
    plt.xticks(fontsize=font_text, fontname='Times New Roman', fontweight='bold')


#loc_str = 'Florida: ; Georgia: Camden, Charlton;'
#loc_dict = extract_county_state(loc_str)

def extract_county_state(loc_str):
    # replace '.' with ';'
    loc_str = loc_str.replace('.',';')
    
    # in case the input is like: State A: State B:
    if (',' in loc_str)==False:
        loc_str = loc_str.replace(':',';')        
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
        county_str_temp = ', '.join(county_list_temp)
        county_list_split = county_str_temp.split(',')
        county_list_split = [x.strip() for x in county_list_split]
        loc_dict[state_temp] = county_list_split
    return loc_dict


def check_county_affected(outage_data_df, state, county_list, select_start_time, select_end_time, out_UB = 0.95):
    # case: no counties are input under a state        
    county_affected = []
    county_all_list = county_list        
    for ii in np.arange(len(county_all_list)):
        county = county_all_list[ii].strip()
        print('\nChecking if the {}-th, {} County {}, is affected'.format(ii, county, state))
        county_outage_data = select_county_data(
                    outage_data_df, county, state, start_time=select_start_time, end_time=select_end_time)                 
        
        if county_outage_data.shape[0]==0:
            continue
        else:
            # get time data
            len_time = len(county_outage_data['run_start_time'])
            index_time = range(len_time)
            
            # calculate restoration rate
            fips_code =  county_outage_data['fips_code'].unique()[0]
            popul = int(county_popul_df.loc[county_popul_df['fips_code']==fips_code, 'population'].iloc[0])
            outage_count_comb = county_outage_data['outage_count'].iloc[index_time]                   
            outage_count_comb_arr = outage_count_comb.to_numpy(dtype = 'float32')
            
            # interplolate when nan values occur
            num_nan = np.isnan(outage_count_comb_arr).sum()
            if 1<= num_nan <= round(len_time*0.75, 0):
                if np.isnan(outage_count_comb_arr[0]) == True:
                    print('\n{} County, {}: the first outage count is missing'.format(county, state))
                    outage_count_comb_arr[0] = 1
                if np.isnan(outage_count_comb_arr[len_time-1]) == True:
                    print('\n{} County, {}: the last outage count is missing'.format(county, state))
                    outage_count_comb_arr[len_time-1] = 1            
                outage_count_comb_list = pd.Series(outage_count_comb_arr).interpolate().values.ravel().tolist()
                outage_count_comb_arr = np.asarray(outage_count_comb_list, dtype = 'float32')
            
            restore_rate_comb = (popul-outage_count_comb_arr)/popul
            
            if (0 < np.min(restore_rate_comb) <= out_UB) & (np.max(outage_count_comb_arr) >= 500):
                print('{} County in {} is affected'.format(county, state))
                county_affected.append(county)
#            else:
#                print('{} County in {} is NOT affected'.format(county, state))
#                print('min rate', np.min(restore_rate_comb))
#                print('max outage count', np.max(outage_count_comb_arr))
#            print(county_affected)
                
    return county_affected



def find_county_affected(outage_data_df, state, select_start_time, select_end_time, out_UB = 0.975):
    # case: no counties are input under a state        

    county_affected = []
    county_all_list = county_coor_fips_df.loc[county_coor_fips_df['state']==state, 'county'].tolist()        
    for ii in np.arange(len(county_all_list)):
        county = county_all_list[ii].strip()
#        print('\nChecking if the {}-th , {} County, {}, is affected'.format(ii, county, state))
        county_outage_data = select_county_data(
                    outage_data_df, county, state, start_time=select_start_time, end_time=select_end_time)                 
        
        if county_outage_data.shape[0]==0:
            continue
        else:
            # get time data
            len_time = len(county_outage_data['run_start_time'])
            index_time = range(len_time)
            
            # calculate restoration rate
            fips_code =  county_outage_data['fips_code'].unique()[0]
            popul = int(county_popul_df.loc[county_popul_df['fips_code']==fips_code, 'population'].iloc[0])
            outage_count_comb = county_outage_data['outage_count'].iloc[index_time]                   
            outage_count_comb_arr = outage_count_comb.to_numpy(dtype = 'float32')
            
            # interplolate when nan values occur
            num_nan = np.isnan(outage_count_comb_arr).sum()
            if 1<= num_nan <= round(len_time*0.75, 0):
                if np.isnan(outage_count_comb_arr[0]) == True:
                    print('\nThe first outage count in {} County, {} State is missing'.format(county, state))
                    outage_count_comb_arr[0] = 1
                if np.isnan(outage_count_comb_arr[len_time-1]) == True:
                    print('\nThe last outage count in {} County, {} State is missing'.format(county, state))
                    outage_count_comb_arr[len_time-1] = 1            
                outage_count_comb_list = pd.Series(outage_count_comb_arr).interpolate().values.ravel().tolist()
                outage_count_comb_arr = np.asarray(outage_count_comb_list, dtype = 'float32')
            
            restore_rate_comb = (popul-outage_count_comb_arr)/popul
            
            if (0 < np.min(restore_rate_comb) <= out_UB) & (np.max(outage_count_comb_arr) >= 500):
#                print('{} County in {} is affected'.format(county, state))
                county_affected.append(county)
#            else:
#                print('{} County in {} is NOT affected'.format(county, state))
#                print('min rate', np.min(restore_rate_comb))
#                print('max outage count', np.max(outage_count_comb_arr))
#            print(county_affected)
                
    return county_affected



def cal_resil(time_temp_datetime, restore_rate_comb_ave):
    
    n_time_point = len(time_temp_datetime)
    n_restore_rate = len(restore_rate_comb_ave)
#    if n_time_point == n_restore_rate:
    
    duration_sec = (time_temp_datetime[-1] - time_temp_datetime[0]).total_seconds()/3600
    time_next_datetime = np.delete(time_temp_datetime, 0)
    time_curr_datetime = np.delete(time_temp_datetime, n_time_point-1)
    time_diff_sec = (time_next_datetime - time_curr_datetime).total_seconds()/3600
    
    restore_rate_comb_ave_arr = restore_rate_comb_ave.array
    np.delete(restore_rate_comb_ave_arr, 0)
    restore_rate_next = np.delete(restore_rate_comb_ave_arr, 0)
    restore_rate_curr = np.delete(restore_rate_comb_ave_arr, n_restore_rate-1)
    
    area_under = np.sum(np.dot((restore_rate_curr + restore_rate_next), time_diff_sec))/2
    resil = area_under/duration_sec
        
    return resil
            
def extract_str_from_dict(loc_dict_copy):
    loc_str = ''
    for key in loc_dict_copy.keys():
        loc_str +=  key + ': '
        county_list = loc_dict_copy[key]
        n_county  = len(county_list)
        for i in np.arange(n_county-1):
            loc_str += county_list[i] + ', '
        loc_str += county_list[n_county-1] + ';'
    return loc_str
 
def plot_all_county(outage_data_df, loc_dict, select_start_time, select_end_time=None, plot_all = 1, check_affected=1):
      
    if select_end_time==None:
        select_end_time = pd.to_datetime(select_start_time) + timedelta(days=25)
        select_end_time = select_end_time.strftime("%Y-%m-%d %H:%M:%S")     
    
    state_list = list(loc_dict.keys())
    
    # to be used in legend
    loc_dict_copy = copy.deepcopy(loc_dict)
    county_list_for_legend = []
    for i_state in np.arange(len(state_list)):
        state_temp = state_list[i_state]
        county_list = loc_dict_copy[state_temp]
        find_county_boo = 0
        check_county_boo = 0
        if len(county_list[0])==0:
            find_county_boo += 1
            county_list = find_county_affected(
                    outage_data_df, state_temp, select_start_time, select_end_time, out_UB = 0.95)
            print('counties affected in {} after finding'.format(state_temp), county_list)
            if county_list:
                loc_dict_copy[state_temp] = county_list
            else:
                print('Counties with less than 75% of missing data are not affected.\nOther counites may be affected')
                continue    # continue to next state
#        print('\nFinding counties affected finished for State of {}'.format(state_temp))
        elif check_affected ==1:
            check_county_boo += 1
            county_list = check_county_affected(
                    outage_data_df, state_temp, county_list, select_start_time, select_end_time, out_UB = 0.95)
            print('county_list after checking in {}'.format(state_temp), county_list)
            if county_list:    # county_list not empty
                loc_dict_copy[state_temp] = county_list
            else:    # county_list is emtpy
                print('None of the input counties in the State of {} are affected after checking'.format(state_temp))
                continue
        else:
            county_list = loc_dict_copy[state_temp]
        
        n_county = len(loc_dict_copy[state_temp])
        county_list_with_state = [None]*n_county
        for ii in np.arange(n_county):
            county_list_with_state[ii] = county_list[ii] + ', ' + state_temp       
        county_list_for_legend = county_list_for_legend + county_list_with_state        
    
    if not county_list_for_legend:
        print('None of the input counties in input state/states are affected after checking')
        return
    # save fig title
    loc_json_title = '{}_{}.json'.format(state_list[0], select_start_time[0:10]) 
    loc_dic_json = json.dumps(loc_dict_copy)
    f = open(loc_json_title,"w")
    f.write(loc_dic_json)
    f.close()
    
    loc_str_find = extract_str_from_dict(loc_dict_copy)

    # remove the state with no counites affected
    for i_state in np.arange(len(state_list)):
        state_temp = state_list[i_state]
        county_list = loc_dict_copy[state_temp]
        if len(county_list[0])==0:
            del loc_dict_copy[state_temp]
    
    if len(loc_dict_copy.keys())==0:
        print('No counties are affected. Please choose other date and time or other states')
        return

    county_arr_for_legend = np.asarray(county_list_for_legend)
#    print('Counties included in legend', county_arr_for_legend)
    if find_county_boo>=1:
        print('counties affected are found')
        print('loc_dict_copy', loc_dict_copy)
       
    # df for resilience from the dict: loc_dict_copy
    resil_df = pd.DataFrame(index=np.arange(len(county_arr_for_legend)), columns = ['state','county','resilience'])
    resil_df_index = 0
    for i_state in np.arange(len(state_list)):
        state_temp = state_list[i_state]
        county_list = loc_dict_copy[state_temp]
        for i_county in np.arange(len(county_list)):
            resil_df['state'].iloc[resil_df_index] = state_temp
            resil_df['county'].iloc[resil_df_index] = county_list[i_county]
            resil_df_index += 1
            
       
    resil_df_index = 0
    y_lim_LB = 1
    colors = iter(cm.rainbow(np.linspace(0, 1, len(county_arr_for_legend))))
     
    for i_state in np.arange(len(state_list)):
        state = state_list[i_state]
        print('\nCurrent state: {} (in plotting stage)'.format(state))
        county_list = loc_dict_copy[state_list[i_state]]
               
          
#        if len(county_list)>=2:            

        for ii in np.arange(len(county_list)):
            
            # extract data for each county
            county = county_list[ii].strip()
#                print('county: ', county)
#                print('state: ', state)
            county_outage_data = select_county_data(
                    outage_data_df, county, state, start_time=select_start_time, end_time=select_end_time)     
            
             
            
            # get time data
            len_time = len(county_outage_data['run_start_time'])  
            index_time = range(len_time)
            time_temp = county_outage_data['run_start_time'].iloc[index_time]
            time_temp_datetime = pd.to_datetime(np.asarray(time_temp))
            
            # calculate restoration rate
            fips_code =  county_outage_data['fips_code'].unique()[0]
            popul = int(county_popul_df.loc[county_popul_df['fips_code']==fips_code, 'population'].iloc[0])
            outage_count_comb = county_outage_data['outage_count'].iloc[index_time]                   
            outage_count_comb_arr = outage_count_comb.to_numpy(dtype = 'float32')
                  
            # interplolate when nan values occur
            num_nan = np.isnan(outage_count_comb_arr).sum()
            if 1<= num_nan <= round(len_time*0.75, 0):
                if np.isnan(outage_count_comb_arr[0]) == True:
                    print('\n{} County, State: the first outage count is missing'.format(county, state))
                    outage_count_comb_arr[0] = 1
                if np.isnan(outage_count_comb_arr[len_time-1]) == True:
                    print('\n{} County, {}: the last outage count is missing'.format(county, state))
                    outage_count_comb_arr[len_time-1] = 1                 
                outage_count_comb_list = pd.Series(outage_count_comb_arr).interpolate().values.ravel().tolist()
                outage_count_comb_arr = np.asarray(outage_count_comb_list, dtype = 'float32')
            
            
            restore_rate_comb = (popul-outage_count_comb_arr)/popul
            
            # smooth the restoration rate
            restore_rate_comb_ave = pd.Series(restore_rate_comb).rolling(window=win_len).mean()
            restore_rate_comb_ave[0:(win_len-1)] = restore_rate_comb[0:(win_len-1)]     
                    
            # plot
            if np.min(restore_rate_comb)<=0:
                
                print('Error: population, {}, < maximum combined outage counts, {}'.format(
                        popul, np.max(outage_count_comb_arr)))
                print('{}, {}'.format(county, state))
                print('Fips code: {}'.format(fips_code))
                time.sleep(1)
                
            elif plot_all == 1:
                
                if y_lim_LB >= np.min(restore_rate_comb_ave):
                    y_lim_LB = np.min(restore_rate_comb_ave)

                # restoration plots on the same figure
                fig_num = 0
                fig = plt.figure(fig_num, figsize=(10, 8))                         
                plt.plot(time_temp_datetime, restore_rate_comb_ave, color = next(colors))
#                plot_label(fig_num)

            
            elif plot_all == 0:
                # restoration plot on an individual figure
                fig = plt.figure(ii, figsize=(10, 8))                         
                plt.plot(time_temp_datetime, restore_rate_comb_ave)
                ax = plt.axes()
                plt.text(x_loc,y_loc+0.07,'County: {}, {}'.format(county, state),
                     fontsize=font_text, color='black', fontweight='bold', transform=ax.transAxes)                                
                plot_label(ii)
                
                # add outages of each utility if there are more than one
                plot_all_util(county_outage_data, state, county, popul,select_start_time,
                              index_time, time_temp_datetime, fig_num = ii)
                plt.show()
            
            # calculate resilience, i.e. area under curve, for each county
            resil_df['resilience'].iloc[resil_df_index] = cal_resil(time_temp_datetime, restore_rate_comb_ave)
            resil_df_index += 1


    # plot all restoration curves in the same figure         
    if plot_all == 1:
        fig_num = 0
        plt.figure(fig_num)
        ax = plt.axes()
        if len(county_arr_for_legend)<=12:
            plt.legend(county_arr_for_legend, bbox_to_anchor=(x_loc-0.02, y_loc+0.03),
                       frameon=False, loc="lower left", fontsize=font_text,
                       borderaxespad=0, fancybox=True, framealpha=0.5)
        elif 13<len(county_arr_for_legend)<=30:
            n_col = 6
            ax.legend(county_arr_for_legend, bbox_to_anchor=(0, 1.0), ncol = n_col, frameon=False,
                      loc="lower left", fontsize=font_text-5, fancybox=True, framealpha=0.5)                    
#            else:
#                plt.text(x_loc,y_loc+0.07,'',
#                     fontsize=font_text, color='black', fontweight='bold', transform=ax.transAxes)                
#            
#                plt.text(x_loc,y_loc+0.06,'State: {}'.format(state),
#                     fontsize=font_text, color='black', fontweight='bold', transform=ax.transAxes)                 
        plt.ylim(bottom=y_lim_LB-0.0005)
        plot_label(fig_num)
        
        # plots of counties in different states on the same figure
#        if i_state == (len(state_list)-1):
        save_fig_title = 'Restoration plots/GUI_plots/multi_state_{}_{}.png'.format(state, select_start_time[0:10]) 
        plt.savefig(save_fig_title, dpi=600)
        plt.show()
    
    # show the counties afffected in each state
    for i_state in np.arange(len(state_list)):
        state = state_list[i_state]
        county_list = loc_dict_copy[state]
        print('\n\nCounties affected in {}:\n{}'.format(state, county_list))
    
    
    # bar plot of resilience
    n_county_total = len(county_arr_for_legend)
    if n_county_total>=2:
        resil_df_sort = resil_df.sort_values(by=['resilience'], ascending=False)
       
        if n_county_total<=5:
            width_bar_plt = 6
        elif 6<=n_county_total<=10:
            width_bar_plt = 8
        else:
            width_bar_plt = round(2+0.5*n_county_total, 0)
        plt.figure(fig_num, figsize=(width_bar_plt, 6))    
        bar_list = plt.bar(resil_df_sort['county'], resil_df_sort['resilience'])
        
        color_arr_sort = cm.rainbow(np.linspace(0, 1, n_county_total))[resil_df_sort.index]
        colors_iter = iter(color_arr_sort)
        for i_bar in np.arange(resil_df_sort.shape[0]):
            bar_list[i_bar].set_color(next(colors_iter))
        plt.ylim(bottom=min(resil_df_sort['resilience'])-0.008)
        plt.ylabel('Resilience',  fontweight='bold', fontsize=font_text+2)
        plt.xticks(rotation=45)
        plt.show()
    return loc_str_find 
#        elif len(county_list) == 1 & (plot_all==0): 
#            
#            # extract data for the county
#            county = county_list[0].strip()
#            county_outage_data = select_county_data(
#                    outage_data_df, county, state, start_time=select_start_time, end_time=select_end_time)     
#            
#            # get time data
#            index_time = range(len(county_outage_data['run_start_time']))
#            time_temp = county_outage_data['run_start_time'].iloc[index_time]
#    #        date_datetime = [datetime.strptime(d_stamp,"%Y-%m-%d %H:%M:%S") for d_stamp in time_temp]
#            time_temp_datetime = pd.to_datetime(np.asarray(time_temp))
#            
#            # calculate restoration rate
#            fips_code =  county_outage_data['fips_code'].unique()[0]
#            popul = int(county_popul_df.loc[county_popul_df['fips_code']==fips_code, 'population'].iloc[0])
#            
#            # combine outage
#            outage_count_comb = county_outage_data['outage_count'].iloc[index_time]                   
#            outage_count_comb_arr = outage_count_comb.to_numpy(dtype = 'float32')
#            restore_rate_comb = (popul-outage_count_comb_arr)/popul
#            restore_rate_comb_ave = pd.Series(restore_rate_comb).rolling(window=win_len).mean()
#            restore_rate_comb_ave[0:(win_len-1)] = restore_rate_comb[0:(win_len-1)]  
#            
#            if np.min(restore_rate_comb) <= 0:
#                
#                print('Error: population, {}, < max outage count, {}'.format(popul, np.min(outage_count_comb_arr)))
#                print('{}, {}'.format(county, state))
#                print('Fips code: {}'.format(fips_code))
#                time.sleep(5)
#                
#            else:
#                fig_num = 0
#                fig = plt.figure(fig_num, figsize=(10, 8))    
#                
#                
#                plt.plot(time_temp_datetime, restore_rate_comb_ave)
#                plot_label(fig_num)
#    
#                ax = plt.axes()
#                plt.text(x_loc,y_loc+0.07,'County: {}, {}'.format(county, state),
#                         fontsize=font_text, color='black', fontweight='bold', transform=ax.transAxes)
#                            
#                # add outages of each utility if there are more than one
#                plt.figure(fig_num)
#                plot_all_util(county_outage_data, state, county, popul,
#                              select_start_time, index_time, time_temp_datetime, fig_num)
#                plt.show()


# function to be executed when the button is clicked           
def draw_each_cnty():
#    county_list = txt_cnty.get().split(",")
#    print(county_list)
#    state = txt_loc.get()
    loc_str = txt_loc.get()
    loc_dict = extract_county_state(loc_str)
    select_start_time = txt_start_date.get()  
    select_end_time = txt_end_date.get() 
    plot_all_county(outage_data_sample_df, loc_dict, select_start_time, select_end_time,
                    plot_all = 0, check_affected=1)

   
    
def draw_all_cnty():
#    county_list = txt_cnty.get().split(",")
#    print(county_list)
#    state = txt_loc.get()
    loc_str = txt_loc.get()
    loc_dict = extract_county_state(loc_str)
    select_start_time = txt_start_date.get()  
    select_end_time = txt_end_date.get() 
    plot_all_county(outage_data_sample_df, loc_dict, select_start_time, select_end_time,
                    plot_all = 1, check_affected=1)



#%%
    
# load data    
# county list in a state and fips code
county_coor_fips_df = pd.read_csv('county fips and coordinates.csv')

# population
county_popul_df = pd.read_csv('county_population.csv', header=0)
county_popul_df = county_popul_df.dropna()

# outage data

# outage_attr = ['utility_id','utility_name','fips_code','county','state','outage_count','run_start_time']
# outage_data_all_df = pd.read_csv('outage_summary.csv', header=None, names=outage_attr)
# out_data_sample = outage_data_all_df[
#         (outage_data_all_df['run_start_time']>='2017-09-05') & (outage_data_all_df['run_start_time']<='2017-10-15') &
#         ((outage_data_all_df['state']=='Florida') | (outage_data_all_df['state']=='Georgia'))]
# out_data_sample.to_csv (r'C:\GitHub\Power_grid_resilience\out_data_sample.csv', index = False, header=True)

#outage_data_sample_df = pd.read_csv('out_data_sample.csv', header=0)

#%%

# define parameters for plot and calculation
win_len = 5 # moveing average window
x_loc = 0.65
y_loc = 0.02
font_text = 12
 
#%%

# layout design

root = tk.Tk()
root.geometry('600x160')
root.title("Power restoration plots")

wth_txt = 65
id_row = 1
pad_x = 8

lbl = tk.Label(root, text=" ")
lbl.grid(column=0, row=0)

# location and date

lbl = tk.Label(root, text="State and county")
lbl.grid(column=0, row=id_row, sticky=(W), padx=(pad_x,0))
loc_str = StringVar(value='Florida: Broward, Clay, Palm Beach; Georgia: Camden, Charlton.')
txt_loc = Entry(root, width=wth_txt, textvariable=loc_str)
txt_loc.grid(column=1, row=id_row, sticky=(W), padx=(pad_x,0))

#lbl = tk.Label(root, text="County/Counties")
#lbl.grid(column=0, row=id_row+1, sticky=(W), padx=(pad_x,0), pady=(0,15))
#cnty_str=StringVar(value="Seminole, Palm Beach")
#txt_cnty = Entry(root, width=wth_txt, textvariable=cnty_str)
#txt_cnty.grid(column=1, row=id_row+1, sticky=(W), padx=(pad_x,0), pady=(0,15))

lbl = tk.Label(root, text="Start date and time")
lbl.grid(column=0, row=id_row+1, sticky=(W), padx=(pad_x,0))
start_date_str = StringVar(value='2017-09-08 22:00:00 ')
txt_start_date = Entry(root, width=wth_txt, textvariable=start_date_str)
txt_start_date.grid(column=1, row=id_row+1, sticky=(W), padx=(pad_x,0))

lbl = tk.Label(root, text="End date and time")
lbl.grid(column=0, row=id_row+2, sticky=(W), padx=(pad_x,0), pady=(0,15))
end_date_str = StringVar(value='2017-09-20 20:00:00')
txt_end_date = Entry(root, width=wth_txt, textvariable=end_date_str)
txt_end_date.grid(column=1, row=id_row+2, sticky=(W), padx=(pad_x,0), pady=(0,15))


#loc_str = 'Florida: Alachua, Broward; Georgia: Camden, Charlton.'
# loc_str = 'Florida: Broward, Clay, Palm Beach;  Georgia: Charlton;'
#loc_dict = extract_county_state(loc_str)
#select_start_time = '2017-09-09'
#select_end_time = '2017-09-26'
    
# options for plots with clicks and exit with click
btn_each_cnty = Button(root, text="  Draw a plot for each county  ", command=draw_each_cnty)
btn_each_cnty.grid(column=0, row=id_row+5, sticky=(W), padx=(pad_x,0), pady=(0,15))

frm = Frame(root)
frm.grid(column=1, row=id_row+5, sticky=(W), padx=(pad_x,0), pady=(0,15))
btn_all_cnty = Button(frm, text="  Draw a plot for all counties  ", command=draw_all_cnty)
btn_all_cnty.grid(column=0, row=0, sticky=(W), padx=(2,0), pady=(0,0))

# exit function
btn_exit = Button(frm, text="  Close  ", width = 10, command=root.destroy)
btn_exit.grid(column=1, row=0, sticky=(W), padx=(12,0), pady=(0,0))


root.mainloop()

#%%


# draw figure

# https://datatofish.com/matplotlib-charts-tkinter-gui/
# https://pythonprogramming.net/embedding-live-matplotlib-graph-tkinter-gui/

#from tkinter import*
#import tkinter as tk
#
#
#class Packbox(tk.Frame):
#    def __init__(self, root):
#        tk.Frame.__init__(self, root)
#
#        bottomframe = Frame(root)
#        bottomframe.pack( side = BOTTOM )
#
#        # Initialize buttons redbutton, whitebutton and bluebutton
#        whitebutton = Button(self, text="Red", fg="red", command=self.white_button)
#        whitebutton.pack( side = LEFT)
#
#        redbutton = Button(self, text="white", fg="white",  command=self.red_button)
#        redbutton.pack( side = LEFT )
#
#        self.white_button()
#        self.red_button()
#
#
#        # Define each buttons method, for example,  white_button() is whitebutton's method, which
#        # is called by command=self.white_button
#
#    def white_button(self):
#
#        self.top = tk.Toplevel(self)
#
#        # Creates new button that closes the new root that is opened when one of the color buttons
#        # are pressed. 
#        button = tk.Button(self.top, text="Close window", command=self.top.destroy)
#
#        # prints the text in the new window that's opened with the whitebutton is pressed
#        label = tk.Label(self.top, wraplength=200, text="This prints white button txt")
#
#
#        label.pack(fill="x")
#        button.pack()
#
#    def red_button(self):
#
#        self.top = tk.Toplevel(self)
#        button = tk.Button(self.top, text="Close window", command=self.top.destroy)
#
#        label = tk.Label(self.top, wraplength=200, text="This prints red button txt")
#
#        label.pack(fill="x")
#        button.pack()
#
#
#
#if __name__ == "__main__":
#    root = tk.Tk()
#    Packbox(root).pack(side="top", fill="both", expand=True)
#    root.mainloop()


#https://www.python-course.eu/tkinter_entry_widgets.php