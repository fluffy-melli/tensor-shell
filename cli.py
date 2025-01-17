import warnings, argparse, os
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model import define, model, parse
from train import gpu
from predict import predict
parser = argparse.ArgumentParser()
parser.add_argument('--model',             type=str, help='모델 구성을 입력하세요')
parser.add_argument('--compile',           type=str, help='모델 컴파일 구성을 입력하세요')
parser.add_argument('--output',            type=str, help='모델 저장 위치를 입력하세요')
parser.add_argument('--input',             type=str, help='학습시킬 모델 위치를 입력하세요')
parser.add_argument('--train-img',         type=str, help='학습시킬 이미지 `폴더` 위치를 입력하세요')
parser.add_argument('--val-img',           type=str, help='검증 이미지 `폴더` 위치를 입력하세요')
parser.add_argument('--epoch',             type=int, help='학습 시킬때 돌릴 epoch 크기를 정해요')
parser.add_argument('--step',              type=int, help='학습 시킬때 돌릴 step의 크기를 정해요')
parser.add_argument('--mode',              type=str, help='학습 시킬때 돌릴 step의 크기를 정해요')
parser.add_argument('--batch',             type=int, help='학습 시킬때 돌릴 step의 크기를 정해요')
parser.add_argument('--predict-img',       type=str, help='예측할 데이터중 이미지를 불러와요')
parser.add_argument('--shape',             type=str, help='예측할 데이터를 설정해요')
parser.add_argument('--summary',           type=bool, help='예측할 데이터를 설정해요')
args = parser.parse_args()

if args.model and args.compile and args.output:
    md = define.Creaft(parse.Model(args.model))
    model.Compile(md, parse.Compile(args.compile))
    model.Save(md, args.output)
    if args.summary:
        md.summary()

if args.input:
    import tensorflow as tf
    md = tf.keras.models.load_model(args.input)
    if args.summary:
        md.summary()
    if args.train_img:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,  
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        train_generator = train_datagen.flow_from_directory(
            args.train_img,
            target_size=tuple(map(int, args.shape.strip('()').split('x'))),
            batch_size=args.batch,
            class_mode=args.mode
        )
        gpu.Device()
        if args.val_img:
            val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
            validation_generator = val_datagen.flow_from_directory(
                args.val_img,
                target_size=tuple(map(int, args.shape.strip('()').split('x'))),
                batch_size=args.batch,
                class_mode=args.mode
            )
            md.fit(
                train_generator,
                steps_per_epoch=args.step,
                epochs=args.epoch,
                validation_data=validation_generator,
                validation_steps=args.step
            )
        else:
            md.fit(
                train_generator,
                steps_per_epoch=args.step,
                epochs=args.epoch,
            )
        if not args.output:
            md.save("train_"+args.input)
        else:
            md.save(args.output)

if args.input:
    import tensorflow as tf
    md = tf.keras.models.load_model(args.input)
    if args.predict_img != None:
        img_array = predict.ReadImage(args.predict_img, args.shape)
        predictions = md.predict(img_array)
        print(predictions)