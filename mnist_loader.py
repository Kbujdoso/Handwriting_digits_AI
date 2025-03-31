import tensorflow as tf
import tensorflow_datasets as tfds

def load_digits():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return (ds_train, ds_test, ds_info)

def normalize_images(image, label): 
    return tf.cast(image, tf.float32) / 255., label


def training_pipeline(ds_train,ds_info):
    ds_train = ds_train.map(normalize_images, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

def evaluation_pipeline(ds_test): 
    ds_test = ds_test.map(normalize_images, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(128)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    