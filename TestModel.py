from AreaClassifier import *
import time

TAR_DIRECTORY = './test_data/'
MODEL_DIRECTORY = './model/'

correct, incorrect, unclassifiable = 0, 0, 0
total_time = 0
confusion_matrix = {}
clf = AreaClassifier(MODEL_DIRECTORY)

for filename in os.listdir(TAR_DIRECTORY):
    if filename[0] == '.':
        continue
    true_area_id = int(filename.split('.')[0])
    if true_area_id not in confusion_matrix:
        confusion_matrix[true_area_id] = {}

    with open(os.path.join(TAR_DIRECTORY, filename), 'r+') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            columns = line.replace(':', ',').split(' ')
            vector = {}
            for values in columns[1:]:
                values = values.split(',')
                bssid = values[0]
                rssi = float(values[1])
                vector[bssid] = rssi
            startTime = int(round(time.time() * 1000))
            predicted_area_id, predicted_prob = clf.classify(vector)
            endTime = int(round(time.time() * 1000))
            total_time += (endTime - startTime)
            if predicted_area_id not in confusion_matrix[true_area_id]:
                confusion_matrix[true_area_id][predicted_area_id] = 0
            confusion_matrix[true_area_id][predicted_area_id] += 1
            if predicted_area_id == true_area_id:
                correct += 1
            elif predicted_area_id != -1:
                incorrect += 1
            else:
                unclassifiable += 1

print '====== Evaluation ======'
for true in confusion_matrix:
    print('True => '+str(true))
    for predicted in confusion_matrix:
        print('  ' + str(predicted) + ': ' +
              str((0 if predicted not in confusion_matrix[true] else confusion_matrix[true][predicted])))
print '========================'
print('Accuracy    : %.1f%%' % (correct*100.0/(correct+incorrect)))
print('Classifiable: %.1f%%' % ((correct+incorrect)*100.0/(correct+incorrect+unclassifiable)))
print('Average Time: %.1fms' % (total_time*1.0/(correct+incorrect+unclassifiable)))
