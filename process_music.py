import librosa
import librosa.display
import librosa.feature
import tensorflow.compat.v1 as tf
from Attention import *
from arguments import parse_arguments 
import torch
import numpy as np
from utils import *
from Attention import *
from data_utils import *
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from vggish_smoke_test import *

def CreateVGGishNetwork(hop_size=0.96):   # Hop size is in seconds.
    """Define VGGish model, load the checkpoint, and return a dictionary that points
    to the different tensors defined by the model.
    """
    vggish_slim.define_vggish_slim()
    checkpoint_path = 'vggish_model.ckpt'
    vggish_params.EXAMPLE_HOP_SECONDS = hop_size
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)
    layers = {'conv1': 'vggish/conv1/Relu',
            'pool1': 'vggish/pool1/MaxPool',
            'conv2': 'vggish/conv2/Relu',
            'pool2': 'vggish/pool2/MaxPool',
            'conv3': 'vggish/conv3/conv3_2/Relu',
            'pool3': 'vggish/pool3/MaxPool',
            'conv4': 'vggish/conv4/conv4_2/Relu',
            'pool4': 'vggish/pool4/MaxPool',
            'fc1': 'vggish/fc1/fc1_2/Relu',
            #'fc2': 'vggish/fc2/Relu',
            'embedding': 'vggish/embedding',
            'features': 'vggish/input_features',
            }
    g = tf.get_default_graph()
    for k in layers:
        layers[k] = g.get_tensor_by_name( layers[k] + ':0')
    return {'features': features_tensor,
            'embedding': embedding_tensor,
            'layers': layers,
            }

def _ProcessWithVGGish(vgg, x, sr):
    '''Run the VGGish model, starting with a sound (x) at sample rate
    (sr). Return a whitened version of the embeddings. Sound must be scaled to be
    floats between -1 and +1.'''
    # Produce a batch of log mel spectrogram examples.
    input_batch = vggish_input.waveform_to_examples(x, sr)
    # print('Log Mel Spectrogram example: ', input_batch[0])
    [embedding_batch] = sess.run([vgg['embedding']],
                                feed_dict={vgg['features']: input_batch})
    # Postprocess the results to produce whitened quantized embeddings.
    pca_params_path = 'vggish_pca_params.npz'
    pproc = vggish_postprocess.Postprocessor(pca_params_path)
    postprocessed_batch = pproc.postprocess(embedding_batch)
    # print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
    return postprocessed_batch

classMap= {'accordion': 0,
'banjo': 1,
'bass': 2,
'cello': 3,
'clarinet': 4,
'cymbals': 5,
'drums': 6,
'flute': 7,
'guitar': 8,
'mallet_percussion': 9,
'mandolin': 10,
'organ': 11,
'piano': 12,
'saxophone': 13,
'synthesizer': 14,
'trombone': 15,
'trumpet': 16,
'ukulele': 17,
'violin': 18,
'voice': 19}
inv_map = {v: k for k, v in classMap.items()}
tf.compat.v1.disable_eager_execution()
tf.reset_default_graph()
sess = tf.Session()
vgg = CreateVGGishNetwork(0.21)
device = torch.device('cpu')
modelPath = 'log/Attention/Attention/trainedModel_0.pth'
model = DecisionLevelSingleAttention(
                freq_bins=128,
                classes_num=20,
                emb_layers=3,
                hidden_units=128,
                drop_rate=0.6)
model.load_state_dict(torch.load(modelPath))
model = model.eval()
model = model.to(device)


def main(music):
    y_v, sr_v  = librosa.load(music)
    X_v = _ProcessWithVGGish(vgg, y_v, sr_v)
    k_v1, k_v2 = X_v.shape
    X_v = np.asarray(X_v).reshape(-1,k_v1,k_v2)
    X_tst_torch_v = (torch.from_numpy(X_v)).type(torch.float32)
    X_tst_torch_v =(X_tst_torch_v/255)
    

    with torch.no_grad():
        output_v = model(X_tst_torch_v)

    output_v = to_numpy(output_v).ravel()
    # print(output_v.shape)
    # print(output_v) 
    outputIndex = np.where(output_v >= 0.9)
    # print(outputIndex)
    answer = ""
    for j,i in enumerate(outputIndex[0]):
        answer +=str(j+1)+". "+inv_map[i]+"\n"
    return answer

# print(main('Demon-Slayer.mp3'))