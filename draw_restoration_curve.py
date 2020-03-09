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

import tkinter as tk
from tkinter import *

import os
os.chdir('C:\GitHub\Power grid resilience')


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


# add outages of each utility if there are more than one
def plot_all_util(county_outage_data, state, county, popul, event_start_time,
                  index_time, time_temp_datetime, fig_num):
    win_len = 5 # moveing average window
    x_loc = 0.61
    y_loc = 0.03
    font_text = 12  
    save_fig_title = 'Restoration plots/{}_{}_{}.png'.format(state, county, event_start_time[0:10])        
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
    
    plt.yticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    plt.xticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    
    
def plot_all_county(outage_data_df, county_list, state, select_start_time, select_end_time=None, plot_all = 1):
      
    if select_end_time==None:
        select_end_time = pd.to_datetime(select_start_time) + timedelta(days=45)
        select_end_time = select_end_time.strftime("%Y-%m-%d %H:%M:%S")      

    win_len = 5 # moveing average window
    x_loc = 0.65
    y_loc = 0.03
    font_text = 12  
       
    
    if len(county_list)>=2:
        
        y_lim_LB = 1
    
        for ii in np.arange(len(county_list)):
            
            # extract data for each county
            county = county_list[ii].strip()
            county_outage_data = select_county_data(
                    outage_data_df, county, state, start_time=select_start_time, end_time=select_end_time)     
            
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
                    
            # plot
            if np.min(restore_rate_comb)<=0:
                
                print('Error: population, {}, < maximum combined outage counts, {}'.format(
                        popul, np.max(outage_count_comb_arr)))
                print('{}, {}'.format(county, state))
                print('Fips code: {}'.format(fips_code))
                time.sleep(5)
                
            elif plot_all == 1:
                
                if y_lim_LB >= np.min(restore_rate_comb_ave):
                    y_lim_LB = np.min(restore_rate_comb_ave)

                # restoration plots on the same figure
                fig_num = 0
                fig = plt.figure(fig_num, figsize=(10, 8))                         
                plt.plot(time_temp_datetime, restore_rate_comb_ave)
#                plot_label(fig_num)
#                plt.show()
            
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
                
        if plot_all == 1:
            fig_num = 0
            plt.figure(fig_num)
            county_arr = np.asarray(county_list)
            plt.legend(county_arr, bbox_to_anchor=(x_loc-0.02, y_loc+0.08), frameon=False,
                       loc="lower left", fontsize=font_text, fancybox=True, framealpha=0.5)
            ax = plt.axes()
            plt.text(x_loc,y_loc+0.06,'State: {}'.format(state),
                 fontsize=font_text, color='black', fontweight='bold', transform=ax.transAxes)                 
            plt.ylim(bottom=y_lim_LB-0.0005)
            plot_label(fig_num)
            plt.show()
        
    elif len(county_list) == 1: 
        
        # extract data for each county
        county = county_list[0].strip()
        county_outage_data = select_county_data(
                outage_data_df, county, state, start_time=select_start_time, end_time=select_end_time)     
        
        # get time data
        index_time = range(len(county_outage_data['run_start_time']))
        time_temp = county_outage_data['run_start_time'].iloc[index_time]
        date_datetime = [datetime.strptime(d_stamp,"%Y-%m-%d %H:%M:%S") for d_stamp in time_temp]
        time_temp_datetime = pd.to_datetime(np.asarray(time_temp))
        
        # calculate restoration rate
        fips_code =  county_outage_data['fips_code'].unique()[0]
        popul = int(county_popul_df.loc[county_popul_df['fips_code']==fips_code, 'population'].iloc[0])
        
        # combine outage
        outage_count_comb = county_outage_data['outage_count'].iloc[index_time]                   
        outage_count_comb_arr = outage_count_comb.to_numpy(dtype = 'float32')
        restore_rate_comb = (popul-outage_count_comb_arr)/popul
        restore_rate_comb_ave = pd.Series(restore_rate_comb).rolling(window=win_len).mean()
        restore_rate_comb_ave[0:(win_len-1)] = restore_rate_comb[0:(win_len-1)]  
        
        if np.min(restore_rate_comb)<=0:
            
            print('Error: population, {}, < max outage count, {}'.format(popul, np.min(outage_count_comb_arr)))
            print('{}, {}'.format(county, state))
            print('Fips code: {}'.format(fips_code))
            time.sleep(5)
            
        else:
            fig_num = 0
            fig = plt.figure(fig_num, figsize=(10, 8))    
            
            
            plt.plot(time_temp_datetime, restore_rate_comb_ave)
            plot_label(fig_num)

            ax = plt.axes()
            plt.text(x_loc,y_loc+0.07,'County: {}, {}'.format(county, state),
                     fontsize=font_text, color='black', fontweight='bold', transform=ax.transAxes)
                        
            # add outages of each utility if there are more than one
            plt.figure(fig_num)
            plot_all_util(county_outage_data, state, county, popul,
                          select_start_time, index_time, time_temp_datetime, fig_num)
            plt.show()


# function to be executed when the button is clicked           
def draw_each_cnty():
    county_list = txt_cnty.get().split(",")
    print(county_list)
    state = txt_state.get()
    select_start_time = txt_start_date.get()  
    select_end_time = txt_end_date.get() 
    plot_all_county(outage_data_sample_df, county_list, state, select_start_time, select_end_time, plot_all = 0)


    
def draw_all_cnty():
    county_list = txt_cnty.get().split(",")
    print(county_list)
    state = txt_state.get()
    select_start_time = txt_start_date.get()  
    select_end_time = txt_end_date.get() 
    plot_all_county(outage_data_sample_df, county_list, state, select_start_time, select_end_time, plot_all = 1)





# import data
# set direc
# import os
# os.chdir('C:\GitHub\Power grid resilience')

# population
county_popul_df = pd.read_csv('county_population.csv', header=0)
county_popul_df = county_popul_df.dropna()

# outage data

# outage_attr = ['utility_id','utility_name','fips_code','county','state','outage_count','run_start_time']
# outage_data_all_df = pd.read_csv('outage_summary.csv', header=None, names=outage_attr)
# out_data_sample = outage_data_all_df[
#         (outage_data_all_df['run_start_time']>='2017-09-05') & (outage_data_all_df['run_start_time']<='2017-10-15') &
#         (outage_data_all_df['state']=='Florida')]
# out_data_sample.to_csv (r'C:\GitHub\Power grid resilience\out_data_sample.csv', index = False, header=True)

outage_data_sample_df = pd.read_csv('out_data_sample.csv', header=0)



# layout design

root = tk.Tk()
root.geometry('460x180')
root.title("Power restoration plots")

wth_txt = 42
id_row = 1
pad_x = 8

lbl = tk.Label(root, text=" ")
lbl.grid(column=0, row=0)

# location and date

lbl = tk.Label(root, text="State")
lbl.grid(column=0, row=id_row, sticky=(W), padx=(pad_x,0))
state_str=StringVar(value='Florida')
txt_state = Entry(root, width=wth_txt, textvariable=state_str)
txt_state.grid(column=1, row=id_row, sticky=(W), padx=(pad_x,0))

lbl = tk.Label(root, text="County/Counties")
lbl.grid(column=0, row=id_row+1, sticky=(W), padx=(pad_x,0), pady=(0,15))
cnty_str=StringVar(value="Seminole, Palm Beach")
txt_cnty = Entry(root, width=wth_txt, textvariable=cnty_str)
txt_cnty.grid(column=1, row=id_row+1, sticky=(W), padx=(pad_x,0), pady=(0,15))

lbl = tk.Label(root, text="Start date and time")
lbl.grid(column=0, row=id_row+2, sticky=(W), padx=(pad_x,0))
start_date_str=StringVar(value='2017-09-08 22:00:00 ')
txt_start_date = Entry(root, width=wth_txt, textvariable=start_date_str)
txt_start_date.grid(column=1, row=id_row+2, sticky=(W), padx=(pad_x,0))

lbl = tk.Label(root, text="End date and time")
lbl.grid(column=0, row=id_row+3, sticky=(W), padx=(pad_x,0), pady=(0,15))
end_date_str=StringVar(value='2017-09-25 20:00:00')
txt_end_date = Entry(root, width=wth_txt, textvariable=end_date_str)
txt_end_date.grid(column=1, row=id_row+3, sticky=(W), padx=(pad_x,0), pady=(0,15))


#county_list = ['Seminole ', '  Palm Beach ']
#state = 'Florida'
#select_start_time = '2017-09-09'
#select_end_time = '2017-09-24'
    
# options for plots with clicks and exit with click
btn_each_cnty = Button(root, text="  Draw a plot for each county  ", command=draw_each_cnty)
btn_each_cnty.grid(column=0, row=id_row+6, sticky=(W), padx=(pad_x,0), pady=(0,15))

frm = Frame(root)
frm.grid(column=1, row=id_row+6, sticky=(W), padx=(pad_x,0), pady=(0,15))
btn_all_cnty = Button(frm, text="  Draw a plot for all counties  ", command=draw_all_cnty)
btn_all_cnty.grid(column=0, row=0, sticky=(W), padx=(2,0), pady=(0,0))

# exit function
btn_exit = Button(frm, text="  Close  ", width = 10, command=root.destroy)
btn_exit.grid(column=1, row=0, sticky=(W), padx=(12,0), pady=(0,0))


root.mainloop()
