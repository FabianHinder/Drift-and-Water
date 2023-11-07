# Investigating the Suitability of Concept Drift Detection and Model-based Drift Explanations for the Detection and Localization of Leakages in Water Distribution Networks

Experimental code of papers 

* [[1]](#paper_loc_of_small) [_"Localization of Small Leakages in Water Distribution Networks using Concept Drift Explanation Methods"_](https://arxiv.org/abs/2310.15830) 
* [[2]](#paper_suitability) _"Investigating the Suitability of Concept Drift Detection for Detecting Leakages in Water Distribution Networks"_

by V. Vaquet et al.

## Abstract

Facing climate change the already limited availability of drinking water will decrease in the future rendering drinking water an increasingly scarce resource. Leakages are a major risk in water distribution networks. On the one hand, they cause water loss, on the other hand, they increase the risks of contamination. 
Leakage detection and localization are challenging problems due to the complex interactions and changing demands in water distribution networks. Especially small leakages are hard to pinpoint yet their localization is vital to avoid water loss over long periods of time. While there exist different approaches to solving the tasks of leakage detection and localization, they are relying on various information about the system, e.g. real-time demand measurements and the precise network topology, which is an unrealistic assumption in many real-world scenarios. 
From a machine learning perspective, leakages can be modeled as concept drift. Thus, a wide variety of fully data driven drift detection schemes seems to be a suitable choice for detecting leakages. 
In contrast, our works [[1](#paper_loc_of_small),[2](#paper_suitability)] we attempt leakage detection and localization using pressure measurements only. For this purpose, 
we explore the potential of both model-loss-based and distribution-based drift detection to tackle leakages detection. We additionally discuss the issue of temporal dependencies in the data and propose a way to cope with it when applying distribution-based detection. We evaluate different methods systematically for leakages of different sizes and detection times.
Furthermore, for leakage localization, we first model leakages in the water distribution network employing Bayesian networks, and the system dynamics are analyzed. We then show how the problem is connected to and can be considered through the lens of concept drift. In particular, we argue that model-based explanations of concept drift are a promising tool for localizing leakages given limited information about the network. The methodology is experimentally evaluated using realistic benchmark scenarios.

**Keywords:** Water Distribution Networks , Anomaly Detection, Virtual Sensors , Leakage Localization , Automated Data-Analysis , Concept Drift , Explainable AI , Model Based Drift Explanations.

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

<a name="paper_loc_of_small">[1]</a> V. Vaquet, F. Hinder, K. Lammers, J. Vaquet, and B. Hammer [_"Localization of Small Leakages in Water Distribution Networks using Concept Drift Explanation Methods"_](https://arxiv.org/abs/2310.15830)
```
@misc{vaquet2023localization,
      title={Localization of Small Leakages in Water Distribution Networks using Concept Drift Explanation Methods}, 
      author={Valerie Vaquet and Fabian Hinder and Kathrin Lammers and Jonas Vaquet and Barbara Hammer},
      year={2023},
      eprint={2310.15830},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<a name="paper_suitability">[2]</a> V. Vaquet, F. Hinder, and B. Hammer [_"Investigating the Suitability of Concept Drift Detection for Detecting Leakages in Water Distribution Networks"_](#)
```
Under review
```

## License

This code has a MIT license.

