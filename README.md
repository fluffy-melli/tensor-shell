>shell
```sh
python ./cli.py --model "Flatten=(28x28) Dense=(128,'relu') Dropout=(0.2) Dense=(10,'softmax')" --compile "Optimizer='adam' Loss='sparse_categorical_crossentropy' Metrics=('accuracy')" --output "./model.keras"
```

```sh
python ./cli.py --model "Conv2D=(32,(3x3),'relu',(150x150x3)) MaxPooling2D=(2x2) Conv2D=(64,(3x3),'relu') MaxPooling2D=(2x2) Conv2D=(128,(3x3),'relu') MaxPooling2D=(2x2) Flatten=() Dense=(512,'relu') Dropout=(0.5) Dense=(10,'softmax')" --compile "Optimizer='adam' Loss='sparse_categorical_crossentropy' Metrics=('accuracy')" --output "./model.keras"
```

```sh
python ./cli.py --model "Conv2D=(32,(3x3),'relu',(150x150x3)) MaxPooling2D=(2x2) Conv2D=(64,(3x3),'relu') MaxPooling2D=(2x2) Conv2D=(128,(3x3),'relu') MaxPooling2D=(2x2) Flatten=() Dense=(512,'relu') Dense=(1,'sigmoid')" --compile "Optimizer='adam' Loss='binary_crossentropy' Metrics=('accuracy')" --output "./model.keras"

python ./cli.py --input "./model.keras" --train-img "C:\Users\yummy\Downloads\kagglecatsanddogs_5340\PetImages" --train-setting "Shape=(150x150) Batch=(32) Mode='binary'" --val-img "C:\Users\yummy\Downloads\kagglecatsanddogs_5340\PetImages" --val-setting "Shape=(150x150) Batch=(32) Mode=('binary')" --output "./train_model.keras"
```