from libs.svmutil import *
import pickle
import os


class AreaClassifier:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, 'config.dat'), 'rb') as f:
            self.config = pickle.load(f)
        self.model = {}
        for filename in os.listdir(model_dir):
            if filename.startswith('.') or filename == 'config.dat':
                continue
            area_id = int(filename.split('.')[0])
            self.model[area_id] = svm_load_model(os.path.join(model_dir, filename))

    def norm(self, rssi):  # max-min normalization
        return (rssi - self.config['min'])*1.0/(self.config['max']-self.config['min'])

    def classify(self, vector):  # vector = {BSSID: RSSI}
        formatted_vector = {}    # formatted vector = {APID: RSSI}
        for bssid in vector:
            if bssid.upper() not in self.config['bssid_to_apid_map']:
                continue
            apid = self.config['bssid_to_apid_map'][bssid.upper()]
            formatted_vector[apid] = self.norm(vector[bssid]) if self.config['normalize'] else vector[bssid]

        if len(formatted_vector) <= 3:  # early termination if only a few APs are received
            return -1, 1.0

        # compute probability of each area and find the maximum one
        y, x = [0], [formatted_vector]
        max_prob, max_area_id = 0, -1
        for area_id in self.model:
            p_labels, p_acc, p_vals = svm_predict(y, x, self.model[area_id], options='-b 1 -q')
            prob = p_vals[0][0]
            if prob > max_prob and prob >= 0.6:
                max_prob = prob
                max_area_id = area_id
        return max_area_id, max_prob  # area_id = -1 => unclassifiable
