import os
import tensorflow as tf
from tensorboard.plugins import projector
from imageResizing import LoadData

LOG_DIR = "./logs/embedding"
NAME = 'embedding'

path_for_sprites = os.path.join(LOG_DIR, 'image.png')
path_for_metadata = os.path.join(LOG_DIR, 'metadata.tsv')

(X_train, y_train), (X_test, y_test) = LoadData(1)

embedding_var = tf.Variable(X_train, name = NAME)
summary_writer = tf.summary.create_file_writer(LOG_DIR)

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

embedding.metadata_path = path_for_metadata

embedding.sprite.image_path = path_for_sprites
embedding.sprite.single_image_dim.extend([28, 28])

projector.visualize_embeddings(summary_writer, config)

tf.saved_model()