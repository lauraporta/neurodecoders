import numpy as np
from noise import pnoise2  # for Perlin noise


class STA:
    def __init__(self):
        pass

    def get_simulated_sta(self, type="gabor,100,100", length=1000):
        if "gabor" in type:
            #  example: "gabor,100,100"
            sta_shape = (int(type.split(",")[1]), int(type.split(",")[2]))
            return self.make_gabor_patterns(sta_shape, length)
        elif "perlin_noise_patterns" in type:
            sta_shape = (int(type.split(",")[1]), int(type.split(",")[2]))
            return self.make_perlin_noise_patterns(sta_shape, length)
        elif "periodic_patterns" in type:
            sta_shape = (int(type.split(",")[1]), int(type.split(",")[2]))
            return self.make_periodic_patterns(sta_shape, length)
        else:
            raise ValueError(f"Type {type} not found")

    def make_gabor_patterns(self, sta_shape, n_patterns):
        patterns = np.zeros((n_patterns, sta_shape[0], sta_shape[1]))

        for i in range(n_patterns):
            # Random Gabor parameters
            sigma = np.random.uniform(5, 20)  # Standard deviation
            theta = np.random.uniform(0, 2 * np.pi)  # Orientation
            lambda_param = np.random.uniform(10, 30)  # Wavelength
            psi = np.random.uniform(0, 2 * np.pi)  # Phase offset
            gamma = np.random.uniform(0.3, 1.0)  # Spatial aspect ratio

            # Create coordinate grids
            x = np.arange(sta_shape[1])
            y = np.arange(sta_shape[0])
            X, Y = np.meshgrid(x, y)

            # Center the coordinates
            X = X - sta_shape[1] // 2
            Y = Y - sta_shape[0] // 2

            # Rotate coordinates
            X_theta = X * np.cos(theta) + Y * np.sin(theta)
            Y_theta = -X * np.sin(theta) + Y * np.cos(theta)

            # Generate Gabor pattern
            pattern = np.exp(
                -(X_theta**2 + gamma**2 * Y_theta**2) / (2 * sigma**2)
            ) * np.cos(2 * np.pi * X_theta / lambda_param + psi)

            # Normalize the pattern to [-1, 1]
            patterns[i] = (
                2 * (pattern - pattern.min()) / (pattern.max() - pattern.min())
                - 1
            )

        return patterns

    def make_perlin_noise_patterns(self, sta_shape, n_patterns):
        patterns = np.zeros((n_patterns, sta_shape[0], sta_shape[1]))
        scale = 25.0  # Controls the scale of the noise
        octaves = 6  # Number of octaves for the noise
        persistence = (
            0.5  # How much each octave contributes to the overall shape
        )
        lacunarity = 2.0  # How much detail is added at each octave

        for i in range(n_patterns):
            # Generate different base coordinates for each pattern
            base_x = np.random.randint(0, 1000)
            base_y = np.random.randint(0, 1000)

            for y in range(sta_shape[0]):
                for x in range(sta_shape[1]):
                    patterns[i, y, x] = pnoise2(
                        x / scale + base_x,
                        y / scale + base_y,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                    )

            # Normalize the pattern to [-1, 1]
            patterns[i] = (
                2
                * (patterns[i] - patterns[i].min())
                / (patterns[i].max() - patterns[i].min())
                - 1
            )

        return patterns

    def make_periodic_patterns(self, sta_shape, n_patterns):
        patterns = np.zeros((n_patterns, sta_shape[0], sta_shape[1]))

        for i in range(n_patterns):
            # Random parameters for each pattern
            freq_x = np.random.uniform(0.1, 0.5)  # Frequency in x direction
            freq_y = np.random.uniform(0.1, 0.5)  # Frequency in y direction
            phase_x = np.random.uniform(0, 2 * np.pi)  # Phase in x direction
            phase_y = np.random.uniform(0, 2 * np.pi)  # Phase in y direction

            # Create coordinate grids
            x = np.linspace(0, 2 * np.pi, sta_shape[1])
            y = np.linspace(0, 2 * np.pi, sta_shape[0])
            X, Y = np.meshgrid(x, y)

            # Generate periodic pattern
            pattern = np.sin(freq_x * X + phase_x) * np.sin(
                freq_y * Y + phase_y
            )

            # Normalize the pattern to [-1, 1]
            patterns[i] = (
                2 * (pattern - pattern.min()) / (pattern.max() - pattern.min())
                - 1
            )

        return patterns
