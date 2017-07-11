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

2. Generate dense tensor input data: data.npy and label.npy.  
```python
python air_writing/recognition/src UltraProcess.py
```
  
3. Generate the dense representation of label(text line): dense.npy
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
1. Project and normalize the 3D coordinated VR writing trajectory data and get filename.json
```python
python air_writing/ui_labeling /preprocessing sphere_fitting.py
```
2. Generate input data from filename.json and get VRdataValidation.npy and VRlabelValidation.npy
```python
python air_writing/recognition/src tagProcess.py
```
3. Test  
```python
python air_writing/recognition/src test_blstm.py
```

## Reference
[ [LiBu05-03] Liwicki, M. and Bunke, H.: IAM-OnDB - an On-Line English Sentence Database Acquired from Handwritten Text on a Whiteboard. 8th Intl. Conf. on Document Analysis and Recognition, 2005, Volume 2, pp. 956 - 961 ](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/iam-on-line-handwriting-database#LiBu05-03)
