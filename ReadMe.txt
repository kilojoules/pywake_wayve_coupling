TLM - framework

example_case.py: 
solve TLM equations and do some post processing

example_case_HPC.py: 
run TLM on supercomputer (read comment on top of script for code adaptation)
you can run it by using exampleCase.pbs

example_case_memory_profile.py: 
run TLM and gather statistics about memory usage

example_case_time_profile.py: 
run TLM and gather statistics about computational time
the output file (stats.prof) has to be read with Snakeviz

run_MPI_locally.py:
run example_case_HPC.py locally on multiple processors