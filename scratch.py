# -*- coding: utf-8 -*-

count = 0
for i in df1.clordid:
    msgs = a.get_group(i)

    try:
        msgs.timestamp.diff()
    except Exception as e:
        print(e)
        print(msgs)
        break
    count +=1
    if count > 100000:
        print(count)
        break
    
    
    
medium_spike = df[(df.msg_counts > 1000) & (df.msg_counts < 10000)]
large_spike = df[df.msg_counts >= 10000]

# get latencies for medium spikes
medium_new = get_latencies(medium_spike,'New')
medium_replace = get_latencies(medium_spike,'Replace')
medium_cancel = get_latencies(medium_spike,'Cancel')

# get latencies for large spikes
large_new = get_latencies(large_spike,'New')
large_replace = get_latencies(large_spike,'Replace')
large_cancel = get_latencies(large_spike,'Cancel')

# get stats for medium spikes
medium_new_stats = display_stats(medium_new.latencies.dropna(),'New')
medium_replace_stats = display_stats(medium_replace.latencies.dropna(),'Replace')
medium_cancel_stats = display_stats(medium_cancel.latencies.dropna(),'Cancel')

# get stats for large spikes
large_new_stats = display_stats(large_new.latencies.dropna(),'New')
large_replace_stats = display_stats(large_replace.latencies.dropna(),'Replace')
large_cancel_stats = display_stats(large_cancel.latencies.dropna(),'Cancel')

