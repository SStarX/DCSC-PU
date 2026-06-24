# Datasets

All files use MATLAB MAT format. Feature matrices are stored under `data` (or
occasionally `X`), while labels are stored under `label` or `labels`. The public
loader handles these variants and applies feature-wise min-max normalization at
runtime. 

## Real-world datasets

| Dataset | Instances | Attributes | Clusters |
| --- | ---: | ---: | ---: |
| Spam | 4601 | 57 | 2 |
| Wdbc | 569 | 30 | 2 |
| BreastEW | 569 | 30 | 2 |
| Mfeat-fac | 2000 | 216 | 10 |
| Pima | 768 | 8 | 2 |
| Segmentation | 2100 | 19 | 7 |
| Dermatology | 366 | 34 | 6 |
| Iris | 150 | 4 | 3 |
| HeartEW | 270 | 13 | 2 |
| Mfeat-zer | 2000 | 47 | 10 |
| Mfeat-kar | 2000 | 64 | 10 |
| PenglungEW | 73 | 325 | 7 |
| Heart | 303 | 13 | 2 |
| SonarEW | 208 | 60 | 2 |
| Iris2 | 150 | 3 | 3 |
| Amber | 880 | 892 | 3 |
| Border | 840 | 892 | 3 |

## Synthetic datasets

`2-cluster`, `2circles`, `2circles_noise`, `2d-10c`, `2d-20c-no0`,
`2d-3c-no123`, `2d-4c-no4`, `2d-4c-no9`, `2d-4c`, `2dnormals`,
`2sp2glob`, `2spiral`, `3MC`, `Aggregation`, `aml28`, `atom`, `banana`,
`blobs`, `chainlink`, `circle`, `clusterincluster`, `cluto-t4-8k`,
`cluto-t5-8k`, `cluto-t7-10k`, `cluto-t8-8k`, `complex8`, `complex9`,
`Compound`, `corners`, `crescentfullmoon`, `crossline`, `cure-t2-4k`,
`cure-t2-4k1`, `curves1`, `curves2`, `D1`, `D13`, `D2`, `D31`,
`dartboard1`, `dartboard2`, `diamond9`, `disk-1000n`, `donut3`,
`donutcurves`, `dpb`, `DS5`, `DS6`, `even`, `Flame`, `halfkernel`,
`hybrid`, `jain`, `outlier`, `Pathbased`, `pearl`, `R15`, `rings`, `S1`,
`S2`, `S3`, `S4`, `smile2`, `Spiral`, `spiralsquare`, `target`,
`three_linear_planes`, `twenty`, `twomoon`, `zelink3`, and `zelink6`.

