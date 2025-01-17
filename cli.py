import argparse, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model import define, model, parse
from train import parse as tr_parse
parser = argparse.ArgumentParser()
parser.add_argument('--model',         type=str, help='모델 구성을 입력하세요')
parser.add_argument('--compile',       type=str, help='모델 컴파일 구성을 입력하세요')
parser.add_argument('--output',        type=str, help='모델 저장 위치를 입력하세요')
parser.add_argument('--input',         type=str, help='학습시킬 모델 위치를 입력하세요')
parser.add_argument('--train-img',     type=str, help='학습시킬 이미지 `폴더` 위치를 입력하세요')
parser.add_argument('--train-setting', type=str, help='학습시킬 데이터의 설정을 해요')
parser.add_argument('--val-img',       type=str, help='검증 이미지 `폴더` 위치를 입력하세요')
parser.add_argument('--val-setting',   type=str, help='검증시킬 데이터의 설정을 해요')
args = parser.parse_args()

if args.model and args.compile and args.output:
    md = define.Creaft(parse.Model(args.model))
    model.Compile(md, parse.Compile(args.compile))
    model.Save(md, args.output)
    md.summary()

elif args.input:
    import tensorflow as tf
    md = tf.keras.models.load_model(args.input)
    if args.train_img != "" and args.val_img != "":
        val_setting = tr_parse.Setting(args.val_setting)
        train_setting = tr_parse.Setting(args.train_setting)
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
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            args.train_img,
            target_size=train_setting["Shape"],
            batch_size=train_setting["Batch"],
            class_mode=train_setting["Mode"]
        )

        validation_generator = val_datagen.flow_from_directory(
            args.val_img,
            target_size=train_setting["Shape"],
            batch_size=train_setting["Batch"],
            class_mode=train_setting["Mode"]
        )

        md.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size
        )

        if args.output == "":
            md.save("train_"+args.input)
        else:
            md.save(args.output)