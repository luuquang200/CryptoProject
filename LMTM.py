import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('LSTM.keras')

# Save the model again to ensure the format is correct
model.save('LSTM_fixed.keras')
