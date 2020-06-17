from libs.svmutil import *
import pickle
import os
import shutil

REF_DIRECTORY = './train_data/'
MODEL_DIRECTORY = './model/'

radiomap = {}
config = {'bssid_to_apid_map': {}, 'max': -1000, 'min': 1000, 'normalize': True}
svm_param = '-s 0 -c 1 -b 1 -t 0 -d 3 -h 1'

# input checking and cleaning output directories
if os.path.isdir(MODEL_DIRECTORY):
    shutil.rmtree(MODEL_DIRECTORY)
os.makedirs(MODEL_DIRECTORY)

# construct bssid to apid map and statistics of the radiomap
for filename in os.listdir(REF_DIRECTORY):
    if filename[0] == '.':
        continue
    with open(os.path.join(REF_DIRECTORY, filename), 'r+') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            columns = line.replace(':', ',').split(' ')
            for values in columns[1:]:
                values = values.split(',')
                bssid = values[0].upper()
                rssi = float(values[1])
                if bssid not in config['bssid_to_apid_map']:
                    config['bssid_to_apid_map'][bssid] = len(config['bssid_to_apid_map'])
                apid = config['bssid_to_apid_map'][bssid]
                if rssi > config['max']:
                    config['max'] = rssi
                if rssi < config['min']:
                    config['min'] = rssi

# construct sparse radiomap
for filename in os.listdir(REF_DIRECTORY):
    if filename[0] == '.':
        continue
    area_id = int(filename.split('.')[0])
    with open(os.path.join(REF_DIRECTORY, filename), 'r+') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            columns = line.replace(':', ',').split(' ')
            vector = {}
            for values in columns[1:]:
                values = values.split(',')
                bssid = values[0].upper()
                rssi = float(values[1])
                apid = config['bssid_to_apid_map'][bssid]
                vector[apid] = (rssi - config['min']) * 1.0 / (config['max'] - config['min']) if config[
                    'normalize'] else rssi
            if area_id not in radiomap:
                radiomap[area_id] = []
            radiomap[area_id].append(vector)

# save config
with open(os.path.join(MODEL_DIRECTORY, 'config.dat'), 'wb') as f:
    pickle.dump(config, f)

# training one-against-all svm models
for area_id in radiomap:
    positive, negative = [], []
    for cur_area_id in radiomap:
        if cur_area_id == area_id:
            positive = positive + radiomap[cur_area_id]
        else:
            negative = negative + radiomap[cur_area_id]
    print('------------------------------')
    print('Training area: ' + str(area_id))
    print('Positive samples: ' + str(len(positive)))
    print('Negative samples: ' + str(len(negative)))

    x = positive + negative
    y = [1] * len(positive) + [-1] * len(negative)

    prob = svm_problem(y, x)
    param = svm_parameter(svm_param)
    m = svm_train(prob, param)
    svm_save_model(os.path.join(MODEL_DIRECTORY, str(area_id) + '.model'), m)

print('------------------------------')
print('Training completed')
