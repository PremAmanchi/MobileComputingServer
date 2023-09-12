import tensorflow as tf


l1_model = tf.keras.models.load_model('l1_model')
l2_model = tf.keras.models.load_model('l2_model')
r1_model = tf.keras.models.load_model('r1_model')
r2_model = tf.keras.models.load_model('r2_model')


# Convert the model.
saved_model_dir = 'l1_assets'
tf.saved_model.save(l1_model, saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the model.
with open('l1_model.tflite', 'wb') as f:
 f.write(tflite_model)



# Convert the model.
saved_model_dir = 'l2_assets'
tf.saved_model.save(l2_model, saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the model.
with open('l2_model.tflite', 'wb') as f:
 f.write(tflite_model)


# Convert the model.
saved_model_dir = 'r1_assets'
tf.saved_model.save(r1_model, saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the model.
with open('r1_model.tflite', 'wb') as f:
 f.write(tflite_model)


# Convert the model.
saved_model_dir = 'r2_assets'
tf.saved_model.save(r2_model, saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the model.
with open('r2_model.tflite', 'wb') as f:
 f.write(tflite_model)
