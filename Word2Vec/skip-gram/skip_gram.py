# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:16:24 2019

@author: wmy
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.animation as animation

with open('data.txt', 'r') as f:
    article = f.read()
    sequence = article[:]
    for note in [",", ".", "'", '"']:
        sequence = sequence.replace(note, "")
        pass
    sequence = sequence.lower()
    sequence = sequence.split()
    pass

word_set = set(sequence)
word_list = sorted(list(word_set))
word_dict = {w: i for i, w in enumerate(word_list)}

batch_size = 64
wr = 1
embedding_size = 3 
voc_size = len(word_list)

def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(voc_size)[data[i][0]]) 
        random_labels.append(np.eye(voc_size)[data[i][1]])

    return random_inputs, random_labels

skip_grams = []
for i in range(wr, len(sequence) - wr):
    target = word_dict[sequence[i]]
    context = []
    for r in range(0, wr):
        context.append(word_dict[sequence[i - wr + r]])
        context.append(word_dict[sequence[i + wr - r]])
        pass
    for w in context:
        skip_grams.append([target, w])
        pass

inputs = tf.placeholder(tf.float32, shape=[None, voc_size])
labels = tf.placeholder(tf.float32, shape=[None, voc_size])

W = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
WT = tf.Variable(tf.random_uniform([embedding_size, voc_size], -1.0, 1.0))

hidden_layer = tf.matmul(inputs, W)
output_layer = tf.matmul(hidden_layer, WT)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=labels))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(20000):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)
        _, loss = sess.run([optimizer, cost], feed_dict={inputs: batch_inputs, labels: batch_labels})

        if (epoch + 1)%500 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        trained_embeddings = W.eval()
        pass
    
    pass

plt.rcParams['figure.dpi'] = 300
fig = plt.figure()
axes3d = Axes3D(fig)

for i, label in enumerate(word_list):
    x, y, z = trained_embeddings[i]
    plt.scatter(x, y, z)
    axes3d.text(x, y, z, label, fontsize=4)
    pass

axes3d.set_xlim(np.min(trained_embeddings), np.max(trained_embeddings))
axes3d.set_ylim(np.min(trained_embeddings), np.max(trained_embeddings))
axes3d.set_zlim(np.min(trained_embeddings), np.max(trained_embeddings))
    
def update(angle):
    if angle % 10 == 0:
        print(angle)
        pass
    axes3d.view_init(angle%360, (angle*2)%360)
    return axes3d

ani = animation.FuncAnimation(fig, update, frames=range(0, 360, 2), interval=50)
ani.save('Word2Vec.gif', writer='imagemagick', fps=60)
