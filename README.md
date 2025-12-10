## Fraud Detect
Fraud Detect is a project that utilizes concepts from computational rationality in order to detect credit card fraud. It cumulates in the creation of a Q-deep learning algorithm coupled with LSTM to create a model that can maximize detection of fraudulent banking activities.

### Financial Fraud
Financial fraud is an umbrella term that encapsulates many acts that involve gaining financial benefits with illegal or fraudulent means. Financial fraud crosses strictly financial institutions and can be committed in corporate, academic, and healthcare sectors. Despite efforts to curb fraud, the increase in financialization of institutions, had led to an uptick in these types of crimes. When taking the extreme amount of computing power available for commercial use into account, the prevalence of these crimes also sees an increased sophistication. The persistence of this type of fraud adversely impacts the economy and society, as well as tarnishing the reputation of major financial institutions.

### Methodology
The classifiers used to create a baseline were Logistic Regression, Random Forest, XGBoost, LSTM.

LSTM is used for that baseline to accurately assess if Q-deep learning impacts the metrics. Building Q-LSTM incorporates the LSTM weights from the baseline LSTM model. Once the model is built, the reward function is created for further training. The metrics recorded include precision, recall, F1, ROC, PRC as well as training time and memory consumption.

### Usage
Running directly from the pipeline folder, 
```python
python3 pipeline/classifier.py
```

Q-deep learning can be tweaked in the *pipeline/dqn.py* file and the hyperparameters can be tweaked in the *tools.py* file.