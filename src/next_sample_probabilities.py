#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='124M',
    seed=None,
    batch_size=1,
    top_k=0,
    top_p=1,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.next_sample(
            hparams=hparams,
            context=context,
            batch_size=batch_size,
            top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)


        #raw_text = input("Model prompt >>> ")
        raw_text = "Give me some "
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("Model prompt >>> ")
        context_tokens = enc.encode(raw_text)
        generated = 0

        out = sess.run(output, feed_dict={
            context: [context_tokens for _ in range(batch_size)]
        })[:, len(context_tokens):]
        # extract most likely words
        print(out.shape)
        ind = np.argsort(-out)
        sorted = out[:, ind]
        print(sorted[:, :10])
        print(ind[:, :10])

        for i in range(batch_size):
            generated += 1
            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)

            print("argmax", enc.decode([out.argmax()]))

            words = enc.decode(ind[i, :10])
            probabilities = sorted[i, :10]#np.exp(sorted[i, :10])
            #probabilities = probabilities / probabilities.sum()
            print(words)
            print("probabilities sum to ", probabilities.sum())
            for j in range(10):
                print(words[j], probabilities[i, j])
        print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)

