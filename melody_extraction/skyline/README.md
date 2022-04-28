# Skyline

Get the accuracy on pop909 using skyline algorithm
```
cd melody_extraction/skyline
python3 cal_acc.py
```

Since Pop909 contains *melody*, *bridge*, *accompaniment*, yet skyline cannot distinguish  between melody and bridge.

There are 2 ways to report its accuracy:

1. Consider *Bridge* as *Accompaniment*, attains 78.54% accuracy
2. Consider *Bridge* as *Melody*, attains 79.51%

Special thanks to Wen-Yi Hsiao for providing the code for skyline algorithm.
