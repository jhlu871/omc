# -*- coding: utf-8 -*-

''' 
OMC Coding Assignment
Jason Lu

Assumptions:
    - Python3 and Anaconda
    - CSV file in same folder as this file
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import datetime

# Only six column headers are given, but some rows imply 8 fields
# We need to clean the file through a temporary file to read it with Pandas
with open('takehome.csv') as f, TemporaryFile('w+') as t:
    # write all lines to a temp file while finding maximum number
    # of fields
    max_fields = 0
    for line in f:
        t.write(line)
        max_fields = np.max([max_fields,len(line.split(','))])
    # send cursor back to first line of temp file
    t.seek(0)
    header = t.readline().strip().split(',')
    # add dummy column headers
    i = 1
    while len(header) < max_fields:
        header += ['x' + str(i)]
        i += 1
    # read temp file as pandas DataFrame
    df = pd.read_csv(t,names=header)

#confirm that the eighth column is all empty and drop it
if (sum(~df.x2.isnull())==0):
    df.drop('x2',1,inplace=True)

''' Resolve some points of uncertainty'''
# A first look of the csv makes it look like UNDEFs are the only
# non blank values in column 7. Let's try to confirm that only 
# UNDEFs show up in column 7
col7_vals = ~df.x1.isnull()
assert(df[col7_vals].x1.unique()==['UNDEF'])

#determine what types of messages lead to UNDEFs
print(df[col7_vals].message_type.unique())
# UNDEFs only show up on Cancel, Replace, and Rejected
# This seems to indicate that it's only bad cancel and replace messages
# and the matching exchange reject message

# Let's confirm our hypothesis on the UNDEFs
# get all of the undef ids of which there are 387
undef_orig_ids = df[col7_vals].orig_clordid.dropna().tolist()

# confirm that all the undefs are trying to cancel or replace
# nonexistent orders. I check if the last filled or canceled message from
# the exchange occurs before the UNDEF. 
for id1 in undef_orig_ids:
    msgs = df[(df.clordid ==id1) | (df.orig_clordid==id1) | (df.chain_clordid==id1)]
    undef_idx = msgs[msgs.x1=='UNDEF'].index[0]
    fills_and_cancels = msgs[(msgs.message_type=='Filled') |
                             (msgs.message_type=='Canceled')]
    
    if fills_and_cancels.index[-1] > undef_idx:
        print('Uh oh, our hypothesis needs further fleshing out')
        break
    
    # Interestingly, theres one case where we receive a canceled message
    # from the exchange before we send the cancel. We should probably
    # remove it, but there's one and the negative latency is not large
    # so it's probably ok to leave it since I'm not sure if this is some
    # rare case of expected behavior
    if fills_and_cancels.message_type.iloc[-1] == 'Canceled':
        print('Unclean data? exchange canceled before we cancel')
        print(msgs)
    

# One of the confusing things in the description of the message_type 
# field in the pdf states there are six types, but eight are actually 
# given above it (3 from us, and 5 from the exchnage). Let's see 
# which it is
print(df.message_type.unique())
# Output confirms that it is the eight listed

''' Ok, now that we've got all the ambiguities out of the way, we can
start working on the given problems'''
# Create a column of python datetime objects based on timestamp
df['dt'] = df.timestamp.apply(lambda x: datetime.datetime.fromtimestamp(x/1e9))

# main function that matches messages to get latencies
def get_latencies(df,sent_msg_type,ids=None):
    ''' Parameters
        ------------
        df: Pandas DataFrame
            DataFrame containing the log
            
        sent_msg_type: str
            Message type that we send to exchange. Must be one of 
            'New','Cancel','Replace'
            
        received_msg_type: str
            Message type that we received from the exchange that
            corresponds to the sent message type
            
        ids: Pandas Series of clordid ids (Optional)
            Series of ids that we use to filter for periods of medium
            or high load
            
        Returns
        --------
        df1: Pandas DataFrame
            Dataframe containing all relevant messages with an additional
            column of latencies   
        '''
    
    #Detemine matching message type that we expect from the exchange
    if sent_msg_type == 'New':
        received_msg_type = 'Acknowledged'
    elif sent_msg_type == 'Cancel':
        received_msg_type = 'Canceled'
    elif sent_msg_type == 'Replace':
        received_msg_type = 'Replaced'
    
    # remove all message types that are not of the type we expect
    df1 = df[(df.message_type == sent_msg_type) | 
             (df.message_type == received_msg_type)].copy()
             
    # if we have passed in the optional ids, filter down to just those
    if ids is not None:
        df1 = df1[df1.clordid.isin(ids)].copy()         
    # group by clordid, diff the two timestamps and drop the NaNs
    # There is always one NaN per group because the first timestamp has
    # nothing to diff against, which is fine since we are only interested
    # in the diff between the second and first timestamp. Some clordids only
    # have one message which is an anomoly that will be analyzed later
    
    # NOTE: may take some time depending on hardware
    # takes about 3 minutes on my old laptop
    df1['latencies'] = df1.groupby('clordid').timestamp.transform(lambda x:x.diff())    
    #latencies = df1.groupby('clordid').timestamp.diff().dropna()
    
    # randomly test a few of the calculations to make sure behavior is 
    # correct
    test_ids = np.random.choice(df1[~df1.latencies.isnull()].clordid,5)
    for test in test_ids:
        msgs = df1[df1.clordid==test]
        time_delta = msgs.iloc[1].timestamp - msgs.iloc[0].timestamp
        assert(msgs.iloc[1].latencies == time_delta)  
        

    
    # Ideally there should be an equal number of sent and received messages
    # but that's not the case. Since we are dropping any unmatched messages
    # we should see what percentage of the log is dropped
    num_sent = len(df1[df1.message_type==sent_msg_type])
    num_received = len(df1[(df1.message_type == received_msg_type)])
    # max group size should be 2, and luckily it is for all our cases
    max_group_size = max(df1.groupby('clordid').timestamp.count())
    if max_group_size > 2:
        print('Uh oh, more than 2 in a group')
        print(max_group_size)
    print('%i msgs dropped' % abs(num_sent-num_received))
    print('%0.2f Percent dropped' % 
          (100*abs(num_sent-num_received)/max(num_sent,num_received)))
    return df1

# calculate latencies for different order types
alldata_new = get_latencies(df,'New')
alldata_replace = get_latencies(df,'Replace')
alldata_cancel = get_latencies(df,'Cancel')

# helper function to get descriptive statistics
def display_stats(data,typ):
    ''' Parameters
        -------------        
        data: Pandas Series
        The data for which we generate the statistics and charts
        
        typ: str
        Message type that we sent to the exchange
        
        Returns
        --------
        stats: list
        List containing the relevant statistics
        '''
    percentiles = [25,50,75,90,95]    

    stats = [data.count(),data.mean(),data.median(),data.std(),
             data.max(),data.min()]
    print('Statistics for %s Messages' % typ)
    print('Count: %i' % data.count())
    print('Mean: %.2f' % data.mean())
    print('Median: %i' % data.median())
    print('Std: %.2f' % data.std())
    print('Max: %i' % data.max())
    print('Min: %i' % data.min())
    for p in percentiles:
        print('%i percentile: %i' % (p,np.percentile(data,p)))
        stats.append(np.percentile(data,p))
    two_std = data.mean() + 2 * data.std()
    tail = sum(data > two_std)/len(data)*100
    print('Percentage of latencies more than 2 std',
          'larger than the mean: %.2f%%' % tail)
    print('# of stds max latency is away from mean: %.2f' % 
          ((data.max()-data.mean())/data.std()))
    stats += [tail,(data.max()-data.mean())/data.std()]
    return stats
   
# get the stats
alldata_new_stats = display_stats(alldata_new.latencies.dropna(),'New')
alldata_replace_stats = display_stats(alldata_replace.latencies.dropna(),'Replace')
alldata_cancel_stats = display_stats(alldata_cancel.latencies.dropna(),'Cancel')

''' Questions 3 and 4''' 

# use Pandas time aware rolling window to aggregate over rolling 1s
df['msg_counts']=df.rolling('1s',on='dt')['timestamp'].count()
display_stats(df.msg_counts, 'Rolling One Second')

# plot load vs time (method 1)
fig = plt.figure()
ax = fig.gca()
ax.plot(df['dt'],df['msg_counts'])
ax.set_xlim(df['dt'].iloc[0],
            df['dt'].iloc[-1]+datetime.timedelta(minutes=10))
plt.title('Method 1 (Rolling One Second Window)')
plt.show()
# aggregate over one second buckets starting on the second
load = df.resample('1s',on='dt').timestamp.count()
display_stats(load,'Discrete One Second Buckets')
#plot the load over time as well as the boundaries for medium and large
fig = plt.figure()
ax = fig.gca()
ax.plot(load)
ax.axhline(y=1000,c='red')
ax.axhline(y=10000,c='red')
ax.set_xlim(df['dt'].iloc[0],
            df['dt'].iloc[-1]+datetime.timedelta(minutes=10))
plt.title('Method 2 (Discrete One Second Buckets)')
plt.show()

# plots indicate that throughout the day there were numerous spikes of activity
# ranging between 1000 and 5000 msgs per second and one massive
# spike at the close of around 20000 msgs per second


# combine load and log data
load_df = pd.DataFrame(load)
load_df.rename(columns={'timestamp':'load'},inplace=True)
joint_df = pd.concat([load_df,df.set_index('dt')])
joint_df.sort_index(inplace=True)
# forward fill the load values for times in between the seconds
joint_df['load'] = joint_df['load'].fillna(method='ffill')
#save a copy to use for cutoff analysis
df_copy = joint_df.copy()
# keep only the columns we need to use
joint_df = joint_df[['message_type','clordid','timestamp','load']].dropna()

# identify medium and large spike messages
medium_spike = joint_df[(joint_df.load > 1000) & (joint_df.load < 10000)]
large_spike = joint_df[joint_df.load >= 10000]
# save the ids of the messages we send
medium_spike_ids = medium_spike[medium_spike.message_type.isin(['New','Replace','Cancel'])].clordid
large_spike_ids = large_spike[large_spike.message_type.isin(['New','Replace','Cancel'])].clordid
# get the latencies of the mssages of the ids we saved
medium_new = get_latencies(joint_df,'New',medium_spike_ids)   
medium_replace = get_latencies(joint_df,'Replace',medium_spike_ids)   
medium_cancel = get_latencies(joint_df,'Cancel',medium_spike_ids)   
large_new = get_latencies(joint_df,'New',large_spike_ids)
large_replace = get_latencies(joint_df,'Replace',large_spike_ids)
large_cancel = get_latencies(joint_df,'Cancel',large_spike_ids)

# get the stats
medium_new_stats = display_stats(medium_new.latencies.dropna(),'Medium New')  
medium_replace_stats = display_stats(medium_replace.latencies.dropna(),'Medium Replace')  
medium_cancel_stats = display_stats(medium_cancel.latencies.dropna(),'Medium Cancel')                
large_new_stats = display_stats(large_new.latencies.dropna(),'Large New')
large_replace_stats = display_stats(large_replace.latencies.dropna(),'Large Replace') 
large_cancel_stats = display_stats(large_cancel.latencies.dropna(),'Large Cancel')             
       
# Create plots of latencies vs time
alldata_new = alldata_new.set_index('dt')
alldata_replace = alldata_replace.set_index('dt')
alldata_cancel = alldata_cancel.set_index('dt')
f,axes = plt.subplots(3,1,sharex=True)
axes[0].plot(alldata_new.latencies.dropna())
axes[1].plot(alldata_replace.latencies.dropna())
axes[2].plot(alldata_cancel.latencies.dropna())
axes[0].set_xlim(alldata_new.index[0],
            alldata_new.index[-1]+datetime.timedelta(minutes=10))
axes[0].set_title('Latency(ns) for New Messages vs Time')
axes[1].set_title('Latency(ns) for Replace Messages vs Time')
axes[2].set_title('Latency(ns) for Cancel Messages vs Time')
plt.show()

# Identify the cutoff region where latencies blow up
cutoffs = [500,1000,2000,3000,4000,5000,10000]
stats = []
for cutoff in cutoffs:
    cutoff_df = df_copy[df_copy.load < cutoff]
    cutoff_ids = cutoff_df[cutoff_df.message_type.isin(['New','Replace','Cancel'])].clordid
    cutoff_new = get_latencies(df_copy,'New',cutoff_ids)
    cutoff_new_stats = display_stats(cutoff_new.latencies.dropna(),'Cutoff New')
    stats.append([cutoff,cutoff_new_stats[1]])
    
# PLot latancey vs load
loads = [x[0] for x in stats]
means = [x[1] for x in stats]
fig = plt.figure()
ax = fig.gca()
ax.plot(loads,means,label='Mean')
plt.title('Latency vs Message Load')
plt.xlabel('Message Load (Messages per second bucket)')
plt.ylabel('Mean Latency (ns)')
plt.show()

''' Utility work for creating report '''
# helper function that aggregates stats into a table and saves as csv
def save_stats_table(fname,*args):
    '''Parameters
       -----------
       fname: str
       filename of csv to save data to
       
       *args: variable length lists of stats
       lists of stats to combine together into the table
       
       Returns
       -------
       stats: Pandas DataFrame
       Dataframe of combined stats
    '''
    
    stats = pd.DataFrame([*args]).T
    # add row labels
    stats.index = ['Count','Mean','Median','Standard Deviation (std)',
                   'Max','Min','25th Percentile',
                   '50th Percentile','75th Percentile','90th Percentile',
                   '95th Percentile','% of latencies more than 2 std larger than mean',
                   '# of stds away from mean for max latency']
    # add column labels
    if fname == 'alldata':
        stats.columns = ['New','Replace','Cancel']
    elif fname == 'by_load':
        index = pd.MultiIndex.from_product([['All Data','Medium Load','High Load'],
                                            ['New','Replace','Cancel']])
        stats.columns = index
    stats.to_csv(fname + '.csv',float_format='%.2f')
    return stats

# save first table in report    
save_stats_table('alldata',alldata_new_stats,alldata_replace_stats,
                     alldata_cancel_stats)
# save second table in report
save_stats_table('by_load',alldata_new_stats,alldata_replace_stats,
                     alldata_cancel_stats,medium_new_stats,medium_replace_stats,
                     medium_cancel_stats,large_new_stats,large_replace_stats,
                     large_cancel_stats)