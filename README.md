# Investigating the Suitability of Concept Drift Detection and Model-based Drift Explanations for the Detection and Localization of Leakages in Water Distribution Networks

Experimental code of papers 

* [[1]](#paper_loc_of_small) [_"Localizing of Anomalies in Critical Infrastructure using Model-Based Drift Explanations"_](https://ieeexplore.ieee.org/iel8/10649807/10649898/10651472.pdf?casa_token=Qho8wSvDDpsAAAAA:lHHUqZ7Tp7BqGKgj155GbCT4GNCPazjbrlfOEp4pMxv_kTkqeb82eNy2vUVrRXGF3MCN3U_bMjDM) [(ArXiv version)](https://arxiv.org/abs/2310.15830) 
* [[2]](#paper_suitability) [_"Investigating the Suitability of Concept Drift Detection for Detecting Leakages in Water Distribution Networks"_](https://www.scitepress.org/Link.aspx?doi=10.5220/0012361200003654) [(ArXiv version)](https://arxiv.org/abs/2401.01733) 

by V. Vaquet et al.

## Abstract

Facing climate change the already limited availability of drinking water will decrease in the future rendering drinking water an increasingly scarce resource. Leakages are a major risk in water distribution networks. On the one hand, they cause water loss, on the other hand, they increase the risks of contamination. 
Leakage detection and localization are challenging problems due to the complex interactions and changing demands in water distribution networks. Especially small leakages are hard to pinpoint yet their localization is vital to avoid water loss over long periods of time. While there exist different approaches to solving the tasks of leakage detection and localization, they are relying on various information about the system, e.g. real-time demand measurements and the precise network topology, which is an unrealistic assumption in many real-world scenarios. 
From a machine learning perspective, leakages can be modeled as concept drift. Thus, a wide variety of fully data driven drift detection schemes seems to be a suitable choice for detecting leakages. 
In contrast, our works [[1](#paper_loc_of_small),[2](#paper_suitability)] we attempt leakage detection and localization using pressure measurements only. For this purpose, 
we explore the potential of both model-loss-based and distribution-based drift detection to tackle leakages detection. We additionally discuss the issue of temporal dependencies in the data and propose a way to cope with it when applying distribution-based detection. We evaluate different methods systematically for leakages of different sizes and detection times.
Furthermore, for leakage localization, we first model leakages in the water distribution network employing Bayesian networks, and the system dynamics are analyzed. We then show how the problem is connected to and can be considered through the lens of concept drift. In particular, we argue that model-based explanations of concept drift are a promising tool for localizing leakages given limited information about the network. The methodology is experimentally evaluated using realistic benchmark scenarios.

**Keywords:** Water Distribution Networks, Anomaly Detection, Virtual Sensors, Leakage Localization, Automated Data-Analysis, Concept Drift, Explainable AI, Model-Based Drift Explanations.

## Requirements

* Python 
* Numpy, SciPy, Pandas, Matplotlib
* scikit-learn

## Data and Usage

The file `pressures.tar.xz` contains a number of sample simulation data used in our experiments. It does not contain all the data used in our experiments due to the large file size. We are happy to provide the data on request. 

The code needs the uncompressed pressure value files. By default, it will assume to find them in a local `pressures` directory each in a subdirectory indicating the experimental setup and throw an exception otherwise (just unpack the `pressures.tar.xz` in the same directory should work). The default base-directory can be set in the `util.py` file, stored in the `base_path` variable. 

To run the code, there are two stages 1. preparing the experiments (`train_model`) which trains potential models and splits the experimental setups in several chunks (default is 25) for parallel processing on different devices and 2. running the experiments (`run_exp #n`) which runs the chunk as indicated by the command line attribute. To run the experiments all files created in the preparation phase and the pressure values need to be present at all devices. The results will be stored in several files that have to be merged before analysis. All 5 experiment files (`model_forecast.py`, `model_interploate.py`, `usup_dd.py`, `localization.py`, `drift_explain.py`) implement the same interface.

## Cite

Cite our papers:

<a name="paper_loc_of_small">[1]</a> V. Vaquet, F. Hinder, K. Lammers, J. Vaquet, and B. Hammer [_"Localizing of Anomalies in Critical Infrastructure using Model-Based Drift Explanations"_](https://ieeexplore.ieee.org/iel8/10649807/10649898/10651472.pdf?casa_token=Qho8wSvDDpsAAAAA:lHHUqZ7Tp7BqGKgj155GbCT4GNCPazjbrlfOEp4pMxv_kTkqeb82eNy2vUVrRXGF3MCN3U_bMjDM) [(ArXiv version)](https://arxiv.org/abs/2310.15830) 
```
@inproceedings{vaquet2024localizing,
  title={Localizing of Anomalies in Critical Infrastructure using Model-Based Drift Explanations},
  author={Vaquet, Valerie and Hinder, Fabian and Vaquet, Jonas and Lammers, Kathrin and Quakernack, Lars and Hammer, Barbara},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2024},
  organization={IEEE}
}
```

<a name="paper_suitability">[2]</a> V. Vaquet, F. Hinder, and B. Hammer [_"Investigating the Suitability of Concept Drift Detection for Detecting Leakages in Water Distribution Networks"_](https://www.scitepress.org/Link.aspx?doi=10.5220/0012361200003654) [(ArXiv version)](https://arxiv.org/abs/2401.01733) 
```
@conference{icpram24,
author={Valerie Vaquet. and Fabian Hinder. and Barbara Hammer.},
title={Investigating the Suitability of Concept Drift Detection for Detecting Leakages in Water Distribution Networks},
booktitle={Proceedings of the 13th International Conference on Pattern Recognition Applications and Methods - ICPRAM},
year={2024},
pages={296-303},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0012361200003654},
isbn={978-989-758-684-2},
issn={2184-4313},
}
```

## License

This code has a MIT license.

