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

2.generate dense tensor inputf files: data.npy label.npy.  
```python
python air_writing/recognition/src UltraProcess.py
```
  
3.generate the dense representation of label(text line).
```python
python air_writing/recognition/src read.py
```
 

## Traning on IAM data   
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

## Testing on VR data
1. project and normalize the 3D coordinated VR writing trajectory data and get filename.json
```python
python air_writing/ui_labeling /preprocessing sphere_fitting.py
```
2. generate input data from filename.json and get VRdataValidation.npy and VRlabelValidation.npy
```python
python air_writing/recognition/src tagProcess.py
```
