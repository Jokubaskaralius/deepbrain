import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
from skimage.transform import resize
import re
import nibabel as nib
from multiprocessing import Pool

PB_FILE = os.path.join(os.path.dirname(__file__), "models", "graph_v2.pb")#"extractor", "graph_v2.pb")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "models", "v2") #"extractor", "v2")


class Extractor:

    def __init__(self):
        self.SIZE = 128
        self.load_pb()

    def load_pb(self):
        graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=graph)   
        with tf.io.gfile.GFile(PB_FILE, 'rb') as f: #tf.compat.v1.gfile.FastGFile(PB_FILE, 'rb')
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            with self.sess.graph.as_default():
                tf.import_graph_def(graph_def)

        self.img = graph.get_tensor_by_name("import/img:0")
        self.training = graph.get_tensor_by_name("import/training:0")
        self.dim = graph.get_tensor_by_name("import/dim:0")
        self.prob = graph.get_tensor_by_name("import/prob:0")
        self.pred = graph.get_tensor_by_name("import/pred:0")

    def load_ckpt(self):
        self.sess = tf.Session()
        ckpt_path = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_path))
        saver.restore(self.sess, ckpt_path)

        g = tf.get_default_graph()

        self.img = g.get_tensor_by_name("img:0")
        self.training = g.get_tensor_by_name("training:0")
        self.dim = g.get_tensor_by_name("dim:0")
        self.prob = g.get_tensor_by_name("prob:0")
        self.pred = g.get_tensor_by_name("pred:0")

    def run(self, image):
        shape = image.shape
        img = resize(image, (self.SIZE, self.SIZE, self.SIZE), mode='constant', anti_aliasing=True)
        img = (img / np.max(img))
        img = np.reshape(img, [1, self.SIZE, self.SIZE, self.SIZE, 1])

        prob = self.sess.run(self.prob, feed_dict={self.training: False, self.img: img}).squeeze()
        prob = resize(prob, (shape), mode='constant', anti_aliasing=True)
        #tf.compat.v1.reset_default_graph()
        return prob

def subfolder_list(dir_name):
    return [f.path for f in os.scandir(dir_name) if f.is_dir()]

def _process(image):
    ext = Extractor()
    prob = ext.run(image)
    return prob

def process_image(image):
    with Pool(1) as p:
        temp = p.apply(_process, (image,))
        print(temp)
        return temp

#Test to see if the GPU resources
#Released fix
def worksForBatch():
    #ext = Extractor()
    path = "/home/jokubas/DevWork/3rdYearProject/data/grade3"
    folders = subfolder_list(path)

    cnt = 0
    for folder2 in folders:
        for folder in subfolder_list(folder2):
            if (re.search(".*T1-axial", folder) is not None):#folder1.split(sep='-')[0]) is not None):
                for item in os.walk(folder):
                    file_path = os.path.join(folder, item[2][0])
                    print(file_path)

                img = nib.load(file_path)
                data = img.get_fdata()
                #prob = ext.run(image)
                prob = process_image(data)
                mask = prob > 0.5
                print(mask)
                print("Next")
                cnt = cnt + 1
        if cnt == 5:
            break
    while(True):
        pass

