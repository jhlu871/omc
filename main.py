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
import datetime

''' Tried to load data into pandas dataframe using load_csv function,
This fails initially because the csv file only contains six column headers,
but some rows contain an 'UNDEF' value in the seventh column followed
by an additional comma (essentially eight columns, last one being blank).
I resolved this by manually appending dummy headers of x1 and x2 to the 
first row of the csv file for columns 7 and 8 (consider it 
preprocessing or cleaning) so that read_csv works. This problem could 
also be resolved programatically, but it's much simpler to just do it 
once manually (Occam's Razor)'''


''' PLEASE READ LONG COMMENT ABOVE AND TAKE SAME ACTION
 OR CODE WILL NOT WORK!!!!!'''
df = pd.read_csv('takehome.csv')

#confirm that the eighth column is all empty and drop it
if (sum(~df.x2.isnull())==0):
    df.drop('x2',1,inplace=True)

''' Resolve some points of uncertainty'''
# A first look of the csv makes it look like UNDEFs are the only
# non blank values in column 7. Let's try to confirm that only 
# UNDEFs show up in column 7
col7_vals = ~df.x1.isnull()
print(df[col7_vals].x1.unique())
# Output only contains 'UNDEF' so that is confirmed

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


def get_latencies(df,sent_msg_type):
    ''' Parameters
        ------------
        df: Pandas DataFrame
            DataFrame containing the log
            
        sent_order_type: str
            Message type that we send to exchange. Must be one of 
            'New','Cancel','Replace'
            
        received_msg_type: str
            Message type that we received from the exchange that cor
            
        Returns
        --------
        latencies: Pandas Series
            Series containing all of the relevant latencies
    '''
    
    #Detemine matching message type that we expect from the exchange
    if sent_msg_type == 'New':
        received_msg_type = 'Acknowledged'
    elif sent_msg_type == 'Cancel':
        received_msg_type = 'Canceled'
    elif sent_msg_type == 'Replace':
        received_msg_type = 'Replaced'
    
    # remove all message types that are not 'New' or "Acknowledged'
    # or 'Rejected'
    df1 = df[(df.message_type == sent_msg_type) | 
             (df.message_type == received_msg_type)]
             
             
    # group by clordid, diff the two timestamps and drop the NaNs
    # There is always one NaN per group because the first timestamp has
    # nothing to diff against, which is fine since we are only interested
    # in the diff between the second and first timestamp. Some clordids only
    # have one message which is an anomoly that will be analyzed later
    
    # NOTE: may take some time depending on hardware
    # takes about 3 minutes on my old laptop
    latencies = df1.groupby('clordid').timestamp.diff().dropna()
    
    # Ideally there should be an equal number of sent and received messages
    # but that's not the case. Since we are dropping any unmatched messages
    # we should see what percentage of the log is dropped
    num_sent = len(df1[df1.message_type==sent_msg_type])
    num_received = len(df1[ (df.message_type == received_msg_type)])
    # max group size should be 2, and luckily it is for all our cases
    max_group_size = max(df1.groupby('clordid').timestamp.count())
    if max_group_size > 2:
        print('Uh oh, more than 2 in a group')
        print(max_group_size)
    print('%i msgs dropped' % abs(num_sent-num_received))
    print('%0.2f Percent dropped' % 
          (100*abs(num_sent-num_received)/max(num_sent,num_received)))
    return latencies

order_latencies = get_latencies(df,'New')
replace_latencies = get_latencies(df,'Replace')
cancel_latencies = get_latencies(df,'Cancel')


def display_stats(data,typ):
    ''' Parameters
        -------------        
        data: Pandas Series
        The data for which we generate the statistics and charts
        
        typ: str
        Message type that we sent to the exchange
        '''
        
    #Change to see other percentiles
    percentiles = [25,50,75]
    print('Statistics for %s Messages' % typ)
    print('Mean: %.2f' % data.mean())
    print('Median: %i' % data.median())
    print('Std: %.2f' % data.std())
    print('Max: %i' % data.max())
    print('Min: %i' % data.min())
    for p in percentiles:
        print('%i percentile: %i' % (p,np.percentile(data,p)))
    two_std = data.mean() + 2 * data.std()
    tail = sum(data > two_std)/len(data)*100
    print('Percentage of latencies more than 2 std',
          'larger than the mean: %.2f%%' % tail)
    print('# of stds max latency is away from mean: %.2f' % 
          ((data.max()-data.mean())/data.std()))
display_stats(order_latencies,'New')
display_stats(replace_latencies,'Replace')
display_stats(cancel_latencies,'Cancel')

# Create a column of python datetime objects based on timestamp
df['dt'] = df.timestamp.apply(lambda x: datetime.datetime.fromtimestamp(x/1e9))
# use Pandas time aware rolling window to aggregate over rolling 1s
df['msg_counts']=df.rolling('1s',on='datetime')['timestamp'].count()