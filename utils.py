import matplotlib.pyplot as plt
from keras.utils import to_categorical
import keras.backend as K
import numpy as np


def string_to_int(string, length, vocab):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab"
    
    Arguments:
    string -- input string, e.g. 'Wed 10 Jul 2007'
    length -- the number of time steps you'd like, determines if the output will be padded or cut
    vocab -- vocabulary, dictionary used to index every character of your "string"
    
    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """
    
    string = string.lower()
    string = string.replace(',','')
    
    if len(string) > length:
        string = string[:length]
        
    rep = list(map(lambda x: vocab.get(x, vocab['UNK']), string))
    if len(string) < length:
        rep += [vocab['#']] * (length - len(string))
    return rep

def int_to_string(ints, inv_vocab):
    """
    Output a machine readable list of characters based on a list of indexes in the machine's vocabulary
    
    Arguments:
    ints -- list of integers representing indexes in the machine's vocabulary
    inv_vocab -- dictionary mapping machine readable indexes to machine readable characters 
    
    Returns:
    l -- list of characters corresponding to the indexes of ints thanks to the inv_vocab mapping
    """
    
    l = [inv_vocab[i] for i in ints]
    return l

def plot_attention_map(model, input_vocabulary, inv_output_vocabulary, text, n_s, num_layer, Tx, Ty):
    """
    Plot the attention map.
  
    """
    attention_map = np.zeros((10, 28))
    Ty, Tx = attention_map.shape
    
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    layer = model.layers[num_layer]

    encoded = np.array(string_to_int(text, Tx, input_vocabulary)).reshape((1, 28))
    encoded = np.array(list(map(lambda x: to_categorical(x, num_classes=len(input_vocabulary)), encoded)))

    f = K.function(model.inputs, [layer.get_output_at(t) for t in range(Ty)])
    r = f([encoded, s0, c0])
    
    for t in range(Ty):
        for t_prime in range(Tx):
            attention_map[t][t_prime] = r[t][0,t_prime,0]

    prediction = model.predict([encoded, s0, c0])
    
    predicted_text = []
    for i in range(len(prediction)):
        predicted_text.append(int(np.argmax(prediction[i], axis=-1)))
        
    predicted_text = list(predicted_text)
    predicted_text = int_to_string(predicted_text, inv_output_vocabulary)
    text_ = list(text)
    
    # get the lengths of the string
    input_length = len(text)
    output_length = Ty
    
    # Plot the attention_map
    # plt.clf();
    f = plt.figure(figsize=(8, 8.5));
    ax = f.add_subplot(1, 1, 1);

    # add image
    i = ax.imshow(attention_map, interpolation='nearest', cmap="Greys");

    # add colorbar
    # cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbaxes = f.add_axes([0.2, 0.1, 0.6, 0.2]);
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal');
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2);

    # add labels
    ax.set_yticks(range(output_length));
    ax.set_yticklabels(predicted_text[:output_length]);

    ax.set_xticks(range(input_length));
    ax.set_xticklabels(text_[:input_length], rotation=45);

    ax.set_xlabel('Input Sequence');
    ax.set_ylabel('Output Sequence');

    # add grid and legend
    ax.grid();
    return attention_map