# DependEX.
Dependency-aware Form Understanding

![Dependency-aware Form Understanding](https://github.com/skzhangPKU/DependEX/blob/master/frame/frame.png)

# Dataset
The raw dataset contains a total of 51,695 samples, each of which involves a UI state and a view hierarchy. After screening and removal of UIs without forms, the remaining samples are selected for analysis. Furthermore, we adopt a combination of automated and manual methods to label relations between form elements, which results in 25,140 annotated element dependency pairs (label-element: 8558, input-action: 5811, others: 10771). Notably, there are a large number of pairs in the others category, where input elements are randomly mapped to incorrect descriptions or actions. This aims to keep a balanced sample  distribution.

**UI_Dependency**: https://disk.pku.edu.cn:443/link/616DC67DEF24A4DAE8FD83F787270AAF

# Requirements for Reproducibility

## System Requirements:
- System: Ubuntu 18.04
- Language: Python 3.6.8
- Devices: GeForce RTX 2080 Ti GPU

## Library Requirements:

- scipy == 1.2.0
- numpy == 1.19.1
- pandas == 1.1.1
- torch == 1.2.0
- torchvision == 0.4.0
- scikit-learn == 0.23.2
- tables == 3.6.1

# Contact
For questions, please feel free to reach out via email at skzhang@pku.edu.cn.
