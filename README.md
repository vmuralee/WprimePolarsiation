# WprimePolarsiation
  This an analysis tool used for study the W' decays to tau and neutrino. 
## Installation 
  * Install Anaconda (which helps to install all machine learning packages)
    * select python3.9 enviornment 
    * install ROOT by using https://root.cern/install/#conda 
    * install xgboost
    * install numpy,pandas and matplotlib
  * Installing HistFactory
    * ``` python -m pip3 install pyhf ```
  * Finally the Analysis tool installation.
    * ``` git clone https://github.com/vmuralee/WprimePolarsiation.git ```
 
 ## To run the code
 
 First go to the ```WprimePolarisation/training/``` folder and open the ```samplesAndVariables.py``` file. Where the sample name, cross-section and number of events for signal and background samples are enlisted. We search for heavy gauage boson for 300 fb-1 to 3000 fb-1 luminosity. The number of events produced for each sample is luminosity times the x-section. The first step you can do is to check whether all the root files are okay by looking at the control plots. The plots can obtained by,
  ``` 
  python3 SkimAnalyzer.py 500 -s control -d
  ```
  * ```500``` is the tau pT threshold, one can look for different pT threshold
  * ```-s``` is the key name for which signal samples are used in the ```samplesAndVariables.py``` file.
  * ```-d``` is the flag to draw the constrol plots, if you don't want to produce plots remove the flag. 

To produce skim ntuple where all the usefull vaiables are stored can obtained by,
  ``` 
  python3 SkimAnalyzer.py 500 -s Right -t 
  ```
  * ```500``` is the tau pT threshold, one can look for different pT threshold
  * ```-s``` is the key name for which signal samples are used in the ```samplesAndVariables.py``` file
  * ```-d``` is the flag to draw the constrol plots, if you don't want to produce plots remove the flag. 
  * ```-t``` is the flag for training 

if you want to tune your BDT please use the ```--tune``` option in the above line   
To store mva_score for the final root file
``` 
  python3 SkimAnalyzer.py 500 -s Right -p 
```
The final skimmed ntuple will produce in the ```DataCards/```. The skim ntuple will use to produce the datacard for the limit computing.
move to the **/Limits** folder. To Create the datacard 
 ```
 python3 CreateDatacard.py 500 -s Right -v mT --hbins 20 100 3500 
 ```
  * ```-v``` define the variable mT or mva_score
  * ```hbins``` define the number of bins,x axis low and x axis high value of the histogram
The output Datacard are in the *json* fromat which saved at ```DataCards/``` folder. One can easily combined the datacards for different W' masses and study the Confidence Limit calculation. 
 ```
 python3 CreateDatacard.py 500 -s Right -v mT --hbins 20 100 3500 -c --list DataCard1.json DataCard2.json ...
 ```
 To compute the expectation limit for different W' masses. 
 ```
  python3 ComputeCLs.py Left mT CombinedDataCard.json
 ```
 The argument **Left** is your signal type which decides your cross-section for different mass samples it can be either of ( Left, Right_N0, Right_N1 ). The second argument is the variable name which is either mT or mva_score. Finally you add the combined json file for the different mass samples.


 


 
