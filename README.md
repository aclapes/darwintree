# darwintree
Framework for action/activity recognition on videos (computer vision research work)

LICENSE: BSD

Copyrights: Albert Clapés, 2015



DEPENDENCES
-----------

For proper functioning of these code you need to satisfy the dependences in form of
python libs (numpy, scipy, opencv, etc), plus some external libraries:

1) A "release/" directory containing the DenseTrackStab executable from:

https://lear.inrialpes.fr/people/wang/improved_trajectories

2) "yael/" directory containing the Yael library from INRIA.


DATASETS
--------

The code is prepared to parse the data and metainfo of some public datasets:
- Hollywood2
- High five
- UCF Sports action

For more info read the set_<dataset_name>_config() in main.py.