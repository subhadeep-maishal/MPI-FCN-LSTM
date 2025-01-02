import xarray as xr
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpi4py import MPI

# Load the dataset
netcdf_file = r"/scratch/20cl91p02/ANN_BIO/FCN/ann_input_data.nc"
ds = xr.open_dataset(netcdf_file)

# Extract data variables
fe = ds['fe'].values  # (time, depth, lat, lon)
po4 = ds['po4'].values
si = ds['si'].values
no3 = ds['no3'].values  # Predictor
nppv = ds['nppv'].values  # Target variable

# Extract latitude and longitude
latitude = ds['latitude'].values  # Shape: (lat,)
longitude = ds['longitude'].values  # Shape: (lon,)

# Remove NaN values by replacing with the mean of each variable
fe = np.nan_to_num(fe, nan=np.nanmean(fe))
po4 = np.nan_to_num(po4, nan=np.nanmean(po4))
si = np.nan_to_num(si, nan=np.nanmean(si))
no3 = np.nan_to_num(no3, nan=np.nanmean(no3))
nppv = np.nan_to_num(nppv, nan=np.nanmean(nppv))

# Since depth is constant, discard the depth dimension and focus on (time, lat, lon)
fe = fe[:, 0, :, :]
po4 = po4[:, 0, :, :]
si = si[:, 0, :, :]
no3 = no3[:, 0, :, :]
nppv = nppv[:, 0, :, :]  # Ensure this matches the structure

# Stack the input variables along a new channel dimension (fe, po4, si, no3)
inputs = np.stack([fe, po4, si, no3], axis=-1)  # Shape: (time, lat, lon, channels)

# Prepare input for LSTM
time_steps = 5  # Number of time steps to consider in each sequence
samples = inputs.shape[0] - time_steps
X_lstm = np.array([inputs[i:i + time_steps] for i in range(samples)])

# Target: Predict NPPV
y_lstm = nppv[time_steps:]  # Shape: (samples, lat, lon)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Normalize the data
scaler_X = StandardScaler()
X_train_reshaped = X_train.reshape(-1, X_train.shape[2] * X_train.shape[3] * X_train.shape[4])
X_test_reshaped = X_test.reshape(-1, X_test.shape[2] * X_test.shape[3] * X_test.shape[4])
X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

scaler_y = StandardScaler()
y_train_reshaped = y_train.reshape(-1, y_train.shape[1] * y_train.shape[2])
y_test_reshaped = y_test.reshape(-1, y_test.shape[1] * y_test.shape[2])
y_train_scaled = scaler_y.fit_transform(y_train_reshaped).reshape(y_train.shape)
y_test_scaled = scaler_y.transform(y_test_reshaped).reshape(y_test.shape)

# Define the FCN + LSTM Model
model = tf.keras.models.Sequential([
    # Input Layer to handle shape (time_steps, lat, lon, channels)
    tf.keras.layers.InputLayer(input_shape=(5, 1, 121, 221, 4)),  # Adjust input shape
    
    # Apply FCN on each time step
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),  # Flatten spatial dimensions

    # LSTM to process the temporal sequence of spatial features
    tf.keras.layers.LSTM(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1] * y_train.shape[2])  # Output shape is flattened (lat * lon)
])

model.compile(optimizer='adam', loss='mse')

# Data generator for batching to save memory
def data_generator(X, y, batch_size):
    size = X.shape[0]
    while True:
        for i in range(0, size, batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]

batch_size = 32
train_generator = data_generator(X_train_scaled, y_train_scaled, batch_size)
val_generator = data_generator(X_test_scaled, y_test_scaled, batch_size)

# Setup MPI communication
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Only the root process (rank 0) runs the training
if rank == 0:
    history = model.fit(train_generator, steps_per_epoch=len(X_train_scaled)//batch_size, 
                        validation_data=val_generator, validation_steps=len(X_test_scaled)//batch_size,
                        epochs=50)

    # Save the model and history to disk if needed
    model.save("/scratch/20cl91p02/ANN_BIO/FCN/fcn_lstm_model.h5")
    print("Model trained and saved.")

# Synchronize processes and ensure proper handling in distributed setup
comm.Barrier()

# Optionally, the root process can perform further evaluation and predictions
if rank == 0:
    test_loss = model.evaluate(X_test_scaled, y_test_scaled.reshape(y_test_scaled.shape[0], -1))
    print(f"Test Loss: {test_loss}")

    # Make predictions
    predictions = model.predict(X_test_scaled)
    predicted_y = scaler_y.inverse_transform(predictions.reshape(-1, y_test.shape[1] * y_test.shape[2])).reshape(y_test.shape)

    # Compute average actual and predicted values across time steps
    average_actual_nppv = np.nanmean(y_test, axis=0)  # Average across time dimension
    average_predicted_nppv = np.nanmean(predicted_y, axis=0)

    # Define output file path
    output_file_path = r"/scratch/20cl91p02/ANN_BIO/FCN/average_output_fcn_lstm_nppv.nc"

    # Create a new NetCDF file
    with xr.Dataset() as ds_out:
        # Create dimensions
        ds_out.coords['latitude'] = ('latitude', latitude)
        ds_out.coords['longitude'] = ('longitude', longitude)

        # Add actual and predicted values as variables
        ds_out['average_actual_nppv'] = (('latitude', 'longitude'), average_actual_nppv)
        ds_out['average_predicted_nppv'] = (('latitude', 'longitude'), average_predicted_nppv)

        # Set attributes for the dataset and variables
        ds_out.attrs['title'] = 'Average NPPV Concentrations (FCN + LSTM)'
        ds_out.attrs['description'] = 'Contains average actual and predicted NPPV concentrations using predictors (fe, po4, si, no3)'

        ds_out['average_actual_nppv'].attrs['units'] = 'mg C m-2 d-1'  # Example unit for NPPV
        ds_out['average_predicted_nppv'].attrs['units'] = 'mg C m-2 d-1'

        # Save the dataset to a NetCDF file
        ds_out.to_netcdf(output_file_path)

    print(f"Output saved to: {output_file_path}")
