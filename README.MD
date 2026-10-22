--csv --metrics gpu__time_duration.avg,gpu__cycles_active.avg,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__sass_average_branch_targets_threads_uniform.pct,dram__throughput.max.pct_of_peak_sustained_elapsed .\b 256


Python-csv:
python get_nsight.py
