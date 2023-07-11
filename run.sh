# %env JOBLIB_TEMP_FOLDER=/tmp
# export TMPDIR=.
export PYTHONPATH=.:$PYTHONPATH                                                     
PYTHONPATH=. python models/framework/Climai_train.py 
