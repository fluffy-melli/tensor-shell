import argparse

parser = argparse.ArgumentParser(description="Process a map of key-value pairs.")

parser.add_argument('--map', type=str, help='Enter key-value pairs as key=value')

args = parser.parse_args()

if args.map:
    map_dict = {}
    for item in args.map.split(','):
        key, value = item.split('=')
        map_dict[key] = value
    print(map_dict)
