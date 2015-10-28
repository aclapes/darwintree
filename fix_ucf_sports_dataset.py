__author__ = 'aclapes'

from os.path import isfile, isdir, join, splitext
from os import listdir, system, remove
import shutil

# TODO: change MANUALLY these parameters
# ----------------------------------------------------
INTERNAL_PARAMETERS = dict(
    home_path = '/Volumes/MacintoshHD/Users/aclapes/',
    datasets_path = 'Datasets/',
    dataset_dirname = 'ucf_sports_actions',
)
# ----------------------------------------------------

def get_video_from_images(images_path, videofile_path, prefix='', n_leadzeros=4, image_format='jpg', fps=10):
    ''' Use external program (ffmpeg) to convert a set of images to video (loseless) '''
    parameters = ['-i ' + images_path + prefix + '%0' + str(n_leadzeros) + 'd.' + image_format,
                  '-vcodec libx264 -crf 20', # alt: '-codec copy',
                  # '-r ' + str(fps),  # not working!
                  videofile_path]
    cmd = 'ffmpeg ' + parameters[0] + ' ' + parameters[1] + ' ' + parameters[2]
    system(cmd)


if __name__ == '__main__':
    parent_path = INTERNAL_PARAMETERS['home_path'] + INTERNAL_PARAMETERS['datasets_path'] + INTERNAL_PARAMETERS['dataset_dirname'] + '/'
    videos_dir = parent_path

    for i, element in enumerate(listdir(videos_dir)):
        if isdir(join(videos_dir, element)):
            action_dir = element

            for instance_dir in listdir(join(videos_dir, action_dir)):
                if isdir(join(videos_dir, action_dir, element)):
                    instance_dir = element
                    print 'Processing', join(videos_dir, action_dir, instance_dir), '...'

                    # check if there are JPGs (uncropped) to generate a video
                    contains_jpegs = False
                    for element in listdir(join(videos_dir, action_dir, instance_dir)):
                        if isfile(join(videos_dir, action_dir, instance_dir, element)) and splitext(element)[1] == '.jpg':
                            contains_jpegs = True
                            break

                    if contains_jpegs:
                        dir = join(videos_dir, action_dir, instance_dir) + '/'  # this is from where we'll get JPGs
                    else:
                        # check, as a secondary option, if there's already a video
                        contains_video = False
                        videoname = ''
                        for element in listdir(join(videos_dir, action_dir, instance_dir)):
                            if isfile(join(videos_dir, action_dir, instance_dir, element)) and splitext(element)[1] == '.avi':
                                contains_video = True
                                videoname = element
                                break

                        if contains_video:
                            shutil.copyfile(join(videos_dir, action_dir, instance_dir, element), \
                                            join(videos_dir, action_dir + '_' + instance_dir + '.avi'))
                            continue
                        else:
                            # no JPGs (uncropped) and no video? use the cropped JPGs (in jpeg/ subfolder)
                            dir = join(videos_dir, action_dir, instance_dir, 'jpeg/')

                    videoname = ''
                    temporary_files = []
                    frame_ctr = 1

                    for element in listdir(dir):
                        if isfile(join(dir, element)) and splitext(element)[1] == '.jpg':
                            if videoname == '':
                                videoname = splitext(element)[0]
                            new_file = str(frame_ctr).zfill(4) + '.jpg'
                            try:
                                shutil.copyfile(join(dir, element), \
                                                join(dir, new_file))
                            except shutil.Error:
                                print "Already existing file:", new_file, "(no need to copy)"
                                pass

                            temporary_files.append(join(dir, new_file))
                            frame_ctr += 1

                    # create my own video
                    get_video_from_images(dir, join(videos_dir, action_dir + '_' + instance_dir + '.avi'), \
                                          image_format='jpg', prefix='', n_leadzeros=4, fps=10)

                    # remove image files that I copied with proper name to generate the video using ffmpeg
                    for filepath in temporary_files:
                        try:
                            remove(filepath)
                        except OSError:
                            pass

                    print 'DONE.'

    print 'ALL videos DONE.'