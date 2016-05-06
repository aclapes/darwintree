import numpy as np
from os.path import isdir, splitext, join
from os import listdir, makedirs

from scipy.io import loadmat
import xmltodict



def load_XML_config(filepath):
    """
    Read the configuration from a .xml file in disk.
    This sets some paths and the features to use.
    :param filepath:
    :return:
    """
    config_dict = dict()

    with open(filepath) as fd:
        xml = xmltodict.parse(fd.read())

    for path in xml['configuration']['path']:
        config_dict[path['@key']] = path['#text'].encode('utf-8')

    if 'option' in xml['configuration']:
        if not isinstance(xml['configuration']['option'], list):
            option = xml['configuration']['option']
            config_dict[option['@key']] = option['#text'].encode('utf-8')
        else:
            for option in xml['configuration']['option']:
                config_dict[option['@key']] = option['#text'].encode('utf-8')
        # data type conversions from str to target type
        if 'num_threads' in config_dict:  # num threads has to be an integer
            config_dict['num_threads'] = int(config_dict['num_threads'])

    features_list = xml['configuration']['features_list']

    if type(features_list['item']) is unicode:
        config_dict.setdefault('features_list',[]).append(features_list['item'].encode('utf-8'))
    else:
        for item in features_list['item']:
            feat = item.encode('utf-8')
            config_dict.setdefault('features_list',[]).append(feat)

    # methods_list = xml['configuration']['methods_list']
    # if type(methods_list['item']) is unicode:
    #     config_dict.setdefault('methods_list',[]).append(methods_list['item'].encode('utf-8'))
    # else:
    #     for item in methods_list['item']:
    #         method = item.encode('utf-8')
    #         config_dict.setdefault('methods_list',[]).append(method)

    return config_dict


def get_dataset_info(parent_path, dataset_name):
    """
    Get dataset-related information
    :param xml_config:
    :param dataset_name:
    :return:
    """
    parent_path = join(parent_path, dataset_name)

    if dataset_name == 'hollywood2':
        dataset_info = get_hollywood2_config(parent_path)
    elif dataset_name == 'highfive':
        dataset_info = get_highfive_config(parent_path)
    elif dataset_name == 'ucf_sports_actions':
        dataset_info = get_ucfsportsaction_dataset(parent_path)
    elif dataset_name == 'olympic_sports':
        dataset_info = get_olympicsports_dataset(parent_path)

    return dataset_info  # it is a tuple masking several other variables


def create_main_directories(parent_path):
    if not isdir(parent_path):
        makedirs(parent_path)

    tracklets_path = join(parent_path, 'tracklets')
    clusters_path = join(parent_path, 'clusters')
    intermediates_path = join(parent_path, 'intermediates')
    feats_path = join(parent_path, 'feats')
    kernels_path = join(parent_path, 'kernels')

    return tracklets_path, clusters_path, intermediates_path, feats_path, kernels_path


def get_hollywood2_config(parent_path):
    videos_dir = join(parent_path, 'AVIClips')
    split_file_path = join(parent_path, 'train_test_split.mat')

    tmp_dict = loadmat(split_file_path)
    fullvideonames = np.array([join(videos_dir, str(element[0][0])) for element in tmp_dict['fnames']])
    videonames = np.array([str(element[0][0]) for element in tmp_dict['fnames']])

    class_labels = tmp_dict['labels2']

    cur_train_indx = np.squeeze(tmp_dict['cur_train_indx'][0][0])
    cur_test_indx = np.squeeze(tmp_dict['cur_test_indx'][0][0])
    train_test_indx = (cur_train_indx-1, cur_test_indx-1)  # was MATLAB's indexing

    action_names = ['AnswerPhone','DriveCar','Eat','FightPerson','GetOutCar', \
                    'HandShake','HugPerson','Kiss','Run','SitDown','SitUp','StandUp']

    return fullvideonames, videonames, class_labels, action_names, train_test_indx, 'map'


def get_highfive_config(parent_path):
    '''
    Hard-codes some paths and configuration values.
    :return:
    '''

    videos_dir = parent_path + 'tv_human_interactions_videos/'
    # splits_dir = parent_path + 'tv_human_interaction_annotations/'  # TODO: think if i'll need to use for test

    # Half train/test partition got from:
    # http://www.robots.ox.ac.uk/~alonso/tv_human_interactions.html
    # ----------------------------------------------------
    train_test = dict(
        train_inds = dict(
            handShake = [2,14,15,16,18,19,20,21,24,25,26,27,28,32,40,41,42,43,44,45,46,47,48,49,50],
            highFive = [1,6,7,8,9,10,11,12,13,23,24,25,27,28,29,30,31,32,33,34,35,44,45,47,48],
            hug = [2,3,4,11,12,15,16,17,18,20,21,27,29,30,31,32,33,34,35,36,42,44,46,49,50],
            kiss = [1,7,8,9,10,11,12,13,14,16,17,18,22,23,24,26,29,31,35,36,38,39,40,41,42],
            negative = np.linspace(1,50,50).astype('int32')
        ),
        test_inds = dict(
            handShake = [1,3,4,5,6,7,8,9,10,11,12,13,17,22,23,29,30,31,33,34,35,36,37,38,39],
            highFive = [2,3,4,5,14,15,16,17,18,19,20,21,22,26,36,37,38,39,40,41,42,43,46,49,50],
            hug = [1,5,6,7,8,9,10,13,14,19,22,23,24,25,26,28,37,38,39,40,41,43,45,47,48],
            kiss = [2,3,4,5,6,15,19,20,21,25,27,28,30,32,33,34,37,43,44,45,46,47,48,49,50],
            negative = np.linspace(51,100,50).astype('int32')
        )
    )
    # ----------------------------------------------------

    action_names = sorted(train_test['train_inds'].keys(), key=lambda x: x)

    videonames = []
    int_inds = []
    train_test_inds = []
    for partition_name in train_test:
        n = len(videonames)
        for i, name in enumerate(action_names):
            videonames += [name + '_' + str(j).zfill(4) for j in train_test[partition_name][name]]
            int_inds += [i]*len(train_test[partition_name][name])

        train_test_inds.append(np.linspace(n, len(videonames)-1, len(videonames)-n).astype('int32'))

    traintest_parts = []
    for part_inds in train_test_inds:
        part = np.zeros((len(videonames),)).astype('int32')
        part[part_inds] = 1
        traintest_parts.append(part)

    fullvideonames = [join(parent_path,'tv_human_interactions_videos/', videoname+'.avi')  for videoname in videonames]

    # create a matrix #{instances}x#{classes}, where entries are all "-1" except for 1s in corresponding class columns
    class_labels = (-1) * np.ones((len(int_inds),len(action_names))).astype('int32')
    for i in xrange(len(action_names)):
        class_labels[np.array(int_inds)==i,i] = 1

    class_labels = class_labels[:,np.where(np.array(action_names) != 'negative')[0]]

    return fullvideonames, videonames, class_labels, action_names, traintest_parts, 'map'


def get_ucfsportsaction_dataset(parent_path):
    videos_dir = parent_path

    # From the publication webpage, some hard-coded metainfo.
    # http://crcv.ucf.edu/data/UCF_Sports_Action.php
    # ----------------------------------------------------
    action_classes = {
        'Diving' : 14,
        'Golf Swing' : 18,
        'Kicking' : 20,
        'Lifting' : 6,
        'Riding Horse' : 12,
        'Running' : 13,
        'SkateBoarding' : 12,
        'Swing-Bench' : 20,
        'Swing-Side' : 13,
        'Walking' : 22
    }

    train_inds = [5,6,7,8,9,10,11,12,13,14,21,22,23,24,25,26,27,28,29,30,31,32,39,40,41,42,43,44,45,46,47,48,49,50, \
                  51,52,55,56,57,58,63,64,65,66,67,68,69,70,75,76,77,78,79,80,81,82,83,88,89,90,91,92,93,94,95,102, \
                  103,104,105,106,107,108,109,110,111,112,113,114,115,120,121,122,123,124,125,126,127,128,136,137,  \
                  138,139,140,141,142,143,144,145,146,147,148,149,150]
    test_inds = [1,2,3,4,15,16,17,18,19,20,33,34,35,36,37,38,53,54,59,60,61,62,71,72,73,74,84,85,86,87,96,97,98,99, \
                 100,101,116,117,118,119,129,130,131,132,133,134,135]

    # ----------------------------------------------------

    # This dataset was quite a mess. So I used a re-organized version of it. READ carefully.
    #
    # These dataset provides both .avi videos and .jpg image files most of times. However, I FOUND PROBLEMS:
    # (1) Video were missing (Diving action, instances in folders: 008-014),
    # (2) Video was shorter than the sequence of JPGs (ex: Diving action, instance 006).
    #
    # Therefore, I generated new videos from JPGs when available, or copying the existing video otherwise. The
    # generation of videos was done with ffmpeg. For more info please refer to the provided "fix_ucf_sports_dataset.py".
    # It prepares the dataset to be parsed with the following code:

    action_names = sorted(action_classes.keys(), key = lambda x : x)

    videonames = []
    for i, element in enumerate(listdir(videos_dir)):  # important listdir lists in alphabetical order
        stem = splitext(element)
        if stem[1] == '.avi':
            videonames.append(stem[0])
    videonames = np.sort(videonames)

    fullvideonames = [join(videos_dir, videoname + '.avi') for videoname in videonames]

    traintest_parts = []
    part = np.zeros((len(videonames),)).astype('int32')
    part[np.array(test_inds)-1] = 1
    traintest_parts.append(part)

    int_inds = []
    for i, name in enumerate(action_names):
        int_inds += [i] * action_classes[name]

    # create a matrix #{instances}x#{classes}, where entries are all "-1" except for 1s in corresponding class columns
    class_labels = (-1) * np.ones((len(int_inds),len(action_names))).astype('int32')
    for i in xrange(len(action_names)):
        class_labels[np.array(int_inds)==i,i] = 1

    # train_test_indx = (np.array(train_inds) - 1, np.array(test_inds) - 1)

    return fullvideonames, videonames, class_labels, action_names, traintest_parts, 'acc'


def get_olympicsports_dataset(parent_path):
    # recall in train_test_split/ diving_platform_10m and diving_spring_3m has the "diving" part missing. Correct it.
    action_names = ['basketball_layup', 'bowling', 'clean_and_jerk', 'discus_throw', 'diving_platform_10m', \
                    'diving_springboard_3m', 'hammer_throw', 'high_jump', 'javelin_throw', 'long_jump', 'pole_vault', \
                    'shot_put', 'snatch', 'tennis_serve', 'triple_jump', 'vault']


    fullvideonames = []
    videonames = []
    int_inds = []
    traintest_parts = []
    for i, part in enumerate(['train', 'test']):
        c = 0
        for action in action_names:
            with open(join(parent_path, 'train_test_split', part, action.replace('_',' ')+'.txt'), 'r') as f:
                k = 1
                line = f.readline().rstrip()
                fullvideonames.append(join(parent_path, action, line+'.mp4'))
                videonames.append(line)
                traintest_parts.append(i)
                while 1:
                    line = f.readline().rstrip()
                    if line != '':
                        fullvideonames.append(join(parent_path, action, line + '.mp4'))
                        videonames.append(line)
                        traintest_parts.append(i)
                        k += 1
                    else:
                        break
                int_inds += [c] * k
            c += 1


    class_labels = (-1) * np.ones((len(int_inds),len(action_names))).astype('int32')
    for i in xrange(len(action_names)):
        class_labels[np.array(int_inds)==i,i] = 1

    return fullvideonames, videonames, class_labels, action_names, [traintest_parts], 'acc'
