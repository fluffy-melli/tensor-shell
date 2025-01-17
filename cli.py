import argparse, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model import define, model, parse

parser = argparse.ArgumentParser()
parser.add_argument('--model',   type=str, help='모델 구성을 입력하세요')
parser.add_argument('--compile', type=str, help='모델 컴파일 구성을 입력하세요')
parser.add_argument('--output',  type=str, help='모델 저장 위치를 입력하세요')
args = parser.parse_args()

md = define.Creaft(parse.Model(args.model))
model.Compile(md, parse.Compile(args.compile))
model.Save(md, args.output)
md.summary()