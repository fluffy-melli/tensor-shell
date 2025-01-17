>shell
```sh
python ./cli.py --model "Conv2D=(32,(3x3),'relu',(150x150x3)) MaxPooling2D=(2x2) Conv2D=(64,(3x3),'relu') MaxPooling2D=(2x2) Conv2D=(128,(3x3),'relu') MaxPooling2D=(2x2) Flatten=() Dense=(512,'relu') Dense=(1,'sigmoid')" --compile "Optimizer='adam' Loss='binary_crossentropy' Metrics=('accuracy')" --summary true --output "./model.keras"

python ./cli.py --input "./model.keras" --train-img "C:\Users\yummy\Downloads\kagglecatsanddogs_5340\PetImages" --shape "(150x150)" --mode "binary" --batch 32 --epoch 2 --step 10 --output "./train_model.keras"

python ./cli.py --input "./train_model.keras" --predict-img "C:\Users\yummy\Downloads\kagglecatsanddogs_5340\PetImages\Dog\999.jpg" --shape "(150x150)"
```