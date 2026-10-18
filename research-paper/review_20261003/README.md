# Supporting analyses

Read-only analysis scripts written while the paper was revised, and their stored outputs. They compute numbers cited
in the paper (key retention for Table S3, the common-valid difference, the loss-decomposition, forced-bound and
fixed-denominator checks, judge timing); docs/OUTPUT_MAP.md says which. Each script reads the data root given by
`HMB_ROOT` and writes its JSON next to itself, except `r1_common_valid.py` and `r1_fixed_den.py`, which write to the
current directory.
