# iMet Collection 2019 - FGVC6
This is a PyTorch implementation of [iMet Collection 2019](https://www.kaggle.com/c/imet-2019-fgvc6/overview).

This code is based on the [baseline code](https://github.com/lopuhin/kaggle-imet-2019).

## Usage

### Train
Make folds
```
python make_folds.py --n-folds 40
```

Train se_resnext101 from fold 0 to 9:
```
python main.py train model_se101_{fold} --model se_resnext101_32x4d --fold {fold} --n-epochs 40 --batch-size 32 --workers 8
```
Train inceptionresnetv2 from fold 5 to 9:
```
python main.py train model_inres2_{fold} --model inceptionresnetv2 --fold {fold} --n-epochs 40 --batch-size 32 --workers 8
```
Train pnas models from fold 0 to 4:
```
python main.py train model_pnas_{fold} --model pnasnet5large --fold {fold} --n-epochs 40 --batch-size 24 --workers 8
```
### Test
The ensemble of these model is used to predict results in `imet-predict-final.ipynb`.
