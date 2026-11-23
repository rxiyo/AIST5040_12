# v0.1 / v0.2 / 联合测试结果

每行对应一个提交及一个评测域。不同训练种子单独列出；95% CI 仅反映独立测试结构采样，不包含训练种子方差。
联合成绩按两个 ID 域等权，旧 OOD/stress 见 conditions.csv。训练版本未填写时标为 unspecified，不据此推断训练流程。

| 原型 / 赛道 | 方法 | 训练版本 / 种子 | 测试版本 | 平均相对误差 %（95% CI） | 0.5% 成功率 % | Energy ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| rutile / blind | PriorSampler（无谱对照） | unspecified / None | v0.1 | 2.5453 [2.4595, 2.6287] | 1.07 | 0.26255 |
| rutile / blind | PriorSampler（无谱对照） | unspecified / None | v0.2 | 2.5449 [2.4582, 2.6268] | 1.30 | 0.26236 |
| rutile / blind | PriorSampler（无谱对照） | unspecified / None | v0.1+v0.2 | 2.5451 [2.4590, 2.6269] | 1.19 | 0.26245 |
| rutile / blind | AIST frozen MDN projected | external_acu / 42 | v0.1 | 1.9484 [1.8419, 2.0559] | 13.64 | 0.28201 |
| rutile / blind | AIST frozen MDN projected | external_acu / 42 | v0.2 | 2.2037 [2.1136, 2.3003] | 4.69 | 0.31817 |
| rutile / blind | AIST frozen MDN projected | external_acu / 42 | v0.1+v0.2 | 2.0761 [1.9813, 2.1747] | 9.16 | 0.30009 |
| rutile / blind | AIST frozen CVAE fixed KL projected | external_acu / 42 | v0.1 | 1.9116 [1.8000, 2.0219] | 12.66 | 0.27318 |
| rutile / blind | AIST frozen CVAE fixed KL projected | external_acu / 42 | v0.2 | 2.0403 [1.9444, 2.1392] | 6.45 | 0.28663 |
| rutile / blind | AIST frozen CVAE fixed KL projected | external_acu / 42 | v0.1+v0.2 | 1.9759 [1.8786, 2.0772] | 9.55 | 0.27990 |
| rutile / blind | AIST frozen CVAE KL warmup 30 projected | external_acu / 42 | v0.1 | 1.9231 [1.8127, 2.0331] | 12.92 | 0.27671 |
| rutile / blind | AIST frozen CVAE KL warmup 30 projected | external_acu / 42 | v0.2 | 1.9978 [1.9092, 2.0942] | 5.83 | 0.28195 |
| rutile / blind | AIST frozen CVAE KL warmup 30 projected | external_acu / 42 | v0.1+v0.2 | 1.9604 [1.8648, 2.0620] | 9.38 | 0.27933 |
| rutile / blind | AIST MDN | v0.1+v0.2 / 0 | v0.1 | 0.5364 [0.5166, 0.5578] | 54.65 | 0.06115 |
| rutile / blind | AIST MDN | v0.1+v0.2 / 0 | v0.2 | 0.8575 [0.8146, 0.9014] | 27.44 | 0.09609 |
| rutile / blind | AIST MDN | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.6970 [0.6702, 0.7239] | 41.05 | 0.07862 |
| rutile / blind | AIST CVAE fixed KL | v0.1+v0.2 / 0 | v0.1 | 0.3867 [0.3659, 0.4077] | 69.95 | 0.04545 |
| rutile / blind | AIST CVAE fixed KL | v0.1+v0.2 / 0 | v0.2 | 0.8452 [0.8032, 0.8888] | 33.11 | 0.09438 |
| rutile / blind | AIST CVAE fixed KL | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.6160 [0.5907, 0.6420] | 51.53 | 0.06991 |
| rutile / blind | AIST CVAE KL warmup 30 | v0.1+v0.2 / 0 | v0.1 | 0.3527 [0.3343, 0.3719] | 72.95 | 0.04167 |
| rutile / blind | AIST CVAE KL warmup 30 | v0.1+v0.2 / 0 | v0.2 | 0.7412 [0.7014, 0.7834] | 44.40 | 0.08150 |
| rutile / blind | AIST CVAE KL warmup 30 | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.5469 [0.5234, 0.5719] | 58.68 | 0.06158 |
