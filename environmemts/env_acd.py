import os
import json
import random
from collections import Counter

class Active_vision_env():
    def __init__(self):
        super(Active_vision_env).__init__()
        self.action_space = [0, 1, 2, 3, 4, 5, 6]
        self.n_actions = len(self.action_space)
        self.n_feature = 'image w*h'
        self.path = "/home/wei/active vision/active_vistion_RL/dataset"

    def reset(self, num): #get initial state
        """

        :param train_set: It means the path og the room
        :return: A random poistion with a image and label, this label is one of the top 3 mount labels in this room to
                 make sure there is a process from diff 5 to diff 1.
        """
        train_set, test_set = Active_vision_env.select_a_room(self, num)
        path = os.path.join(train_set, 'annotations.json')
        total = 0
        with open(path) as f:
            fields = []
            for line in f.readlines():
                line = line.strip().split()
                line = line[0].strip(',').rstrip('{').rstrip('[').rstrip(':')
                fields.append(line)

            record = []
            for line_i in range(len(fields)):
                # print(fields[line_i])
                if (fields[line_i] == '"bounding_boxes"'):
                    begin_i = line_i
                elif (fields[line_i] == ']'):
                    final_i = line_i
                    total += 1
                    if (final_i - begin_i > 1):
                        for k in range(begin_i + 1, final_i):
                            if (fields[k][-2] == '4' or fields[k][-2] == '5'):
                                record.append(fields[k])
            label = []
            for s in record:
                label.append(s[1:-1].split(",")[-2])

            result = Counter(label)

            top3_label = {} # find out top3 num label
            for s in range(3):
                max_label = max(result, key=result.get)
                top3_label[max_label] = result[max_label]
                del result[max_label]
            the_label, _ = random.choice(list(top3_label.items()))

            record_img = {}
            for line_i in range(len(fields)):
                record = []
                if (fields[line_i] == '"bounding_boxes"'):
                    begin_i = line_i
                elif (fields[line_i] == ']'):
                    final_i = line_i
                    total += 1
                    if (final_i - begin_i > 1):
                        filename = fields[begin_i - 1]
                        for k in range(begin_i + 1, final_i):
                            if (fields[k][-2] == '4' or fields[k][-2] == '5'):
                                record.append(fields[k])
                        if record:
                            label_to_img = []
                            for s in record:
                                label_to_img.append(s[1:-1].split(","))
                            for s in label_to_img:
                                if s[-2] == the_label and s[-1] == '5':
                                    record_img[filename] = s
            img, bbox = random.choice(list(record_img.items()))
            img = img[1:-1]
            thing_label = int(bbox[-2])
            diff = int(bbox[-1])
            bbox = bbox[0:4]
        return train_set, img, thing_label, diff, bbox

    def step(self, train_set, curr_img, thing_label, diff, action):

        # set up curr_img and json paths
        images_path = os.path.join(train_set, 'jpg_rgb', curr_img)
        annotations_path = os.path.join(train_set, 'annotations.json')

        # load json data
        ann_file = open(annotations_path)
        annotations = json.load(ann_file)

        # set up for first image
        cur_image_name = images_path
        next_image_name = ''

        if action:
            # get the next image name to display based on the
            # user input, and the annotation.
            if action == 0:
                next_image_name = annotations[curr_img]['forward']
            elif action == 1:
                next_image_name = annotations[curr_img]['rotate_ccw']
            elif action == 2:
                next_image_name = annotations[curr_img]['backward']
            elif action == 3:
                next_image_name = annotations[curr_img]['rotate_cw']
            elif action == 4:
                next_image_name = annotations[curr_img]['left']
            elif action == 5:
                next_image_name = annotations[curr_img]['right']
            elif action == 6:
                next_image_name = curr_img
        reward = 0
        next_bbox = []
        cur_diff = diff
        if next_image_name != '':
            if next_image_name != curr_img:
                next_diff, next_bbox = Active_vision_env.bbox_diff(self, train_set
                                                         , next_image_name, thing_label)

                if next_diff == -1: # Nothing in img
                    cur_diff = 6
                    reward = -1
                elif next_diff < cur_diff != -1: # Get better
                    cur_diff = next_diff
                    reward = 1
                elif next_diff > cur_diff != -1: # Worse
                    cur_diff = next_diff
                    reward = -1
                elif next_diff == cur_diff != -1 != 6: # No change
                    reward = 0

            elif next_image_name == curr_img:
                if cur_diff == 6:
                    reward = -1
                elif cur_diff == 1:
                    reward = 1
                else:
                    reward = 0

            curr_img = next_image_name

        elif next_image_name == '':
            if cur_diff == 6:
                reward = -1
            elif cur_diff == 1:
                reward = 1
            else:
                reward = 0

        return reward, curr_img, cur_diff, next_bbox

    def bbox_diff(self, train_set, img, thing_label):
        # set up curr_img and json paths
        annotations_path = os.path.join(train_set, 'annotations.json')

        # load json data
        ann_file = open(annotations_path)
        annotations = json.load(ann_file)

        # set up for first image
        diff = 0
        bbox = []
        boxes = annotations[img]['bounding_boxes']
        if boxes:
            for i in boxes:
                if i[-2] == thing_label:
                    diff = i[-1]
                    bbox = i[0:4]
                    break
            if diff == 0:
                diff = -1
                bbox = [0, 0, 0, 0]
        else:
            diff = -1#
            bbox = [0, 0, 0, 0]
        return diff, bbox

    def select_a_room(self, num): # each epison select a new room
        default_train_list = [
            'Home_002_1',
            'Home_003_1',
            'Home_003_2',
            'Home_004_1',
            'Home_004_2',
            'Home_005_1',
            'Home_005_2',
            'Home_006_1',
            'Home_014_1',
            'Home_014_2',
            'Office_001_1'

        ]
        default_test_list = [
            'Home_001_1',
            'Home_001_2',
            'Home_008_1'
        ]
        room_total_num = len(default_train_list)
        room_num = num%room_total_num
        train_set = default_train_list[room_num]
        train_set = os.path.join(self.path, train_set)

        test_set = default_test_list[random.randint(0, len(default_test_list)-1)]
        test_set = os.path.join(self.path, test_set)
        return train_set, test_set


if __name__ == '__main__':
    print(random.randint(0, 7))


