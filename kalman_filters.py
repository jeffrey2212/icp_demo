import numpy as np

class KalmanFilter:
    def __init__(self):
        # State vector [x, y, theta]
        self.x = np.zeros((3, 1))
        
        # State covariance matrix
        self.P = np.eye(3)
        
        # Process noise covariance
        self.Q = np.diag([0.1, 0.1, np.deg2rad(5)])  # Adjust these values based on your robot
        
        # Measurement matrix
        self.H = np.eye(3)  # Direct observation
        
        # Measurement noise covariance
        self.R = np.diag([0.5, 0.5, np.deg2rad(10)])  # Adjust based on LiDAR accuracy
        
    def predict(self, u):
        """
        Predict the state and state covariance.
        u: control input [v, omega] (velocity and angular velocity)
        """
        dt = 1.0  # Time step. Adjust as needed.
        
        # State transition model
        F = np.array([[1, 0, -u[0]*np.sin(self.x[2])*dt],
                      [0, 1, u[0]*np.cos(self.x[2])*dt],
                      [0, 0, 1]])
        
        # Update state
        self.x[0] += u[0] * np.cos(self.x[2]) * dt
        self.x[1] += u[0] * np.sin(self.x[2]) * dt
        self.x[2] += u[1] * dt
        
        # Update covariance
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, z):
        """
        Update the state estimate with a measurement z.
        z: measurement vector [x, y, theta]
        """
        y = z - (self.H @ self.x)  # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x += K @ y  # Update state estimate
        self.P = (np.eye(3) - K @ self.H) @ self.P  # Update covariance estimate
        
    def get_state(self):
        """
        Returns the current state estimate.
        """
        return self.x.flatten()

# Example usage
#kf = KalmanFilter()
#kf.predict([0.5, np.deg2rad(5)])  # Assuming a forward velocity of 0.5 m/s and a rotation of 5 degrees per second
#kf.update([1.0, 0.5, np.deg2rad(45)])  # Example LiDAR measurement
#print("Estimated State:", kf.get_state())