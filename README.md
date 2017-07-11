# On-Line Hand Writing Recognition using BLSTM
## Requirement:
    tensorflow >= 1.2
    numpy
    scipy
    python 3.5
    xml
    
## Setup
1. Go to http://www.fki.inf.unibe.ch/databases/iam-handwriting-database download the IAM On-Line Handwriting DataBase.
    And store the dataset folders 'ascii' and 'lineStrokes' under air_writing/data/

2.run
```python
python air_writing/recognition/src UltraProcess.py
```
to generate dense tensor inputf files: data.npy label.npy.   
And run
```python
python air_writing/recognition/src read.py
```
to generate the dense representation of label(text line).  

## Traning on IAM data   
run   
```python
python air_writing/recognition/src air_writing/recognition/src train_blstm.py
```
hyper parameters:   

--data_dir  
--checkpoint_dir   
-- log_dir    
--restore_path   
--batch_size    
--total_epoches   
...(details please refer to air_writing/recognition/src/train_blstm.py)
