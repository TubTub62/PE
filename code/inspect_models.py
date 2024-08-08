from code.model_search.model_search_core import model_summaries
import tensorflow as tf

model = tf.saved_model.load("testmodel")

print(model)