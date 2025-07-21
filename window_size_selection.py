import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from ast import literal_eval


from scipy.signal import fftconvolve
from scipy.stats import linregress
from astropy.timeseries import LombScargle

class Autoperiod(object):
    def __init__(self, times, values, plotter=None, threshold_method='mc', mc_iterations=40, confidence_level=.9):
        self.times = times - times[0] if times[0] != 0 else times
        self.values = values
        self.acf = self.autocorrelation(values)
        self.acf /= np.max(self.acf)
        self.acf *= 100
        self.time_span = self.times[-1] - self.times[0]
        self.time_interval = self.times[1] - self.times[0]

        freqs, self.powers = LombScargle(self.times, self.values).autopower(
            minimum_frequency=1 / self.time_span,
            maximum_frequency=1 / (self.time_interval * 2),
            normalization='psd'
        )
        self.periods = 1 / freqs

        self._power_norm_factor = 1 / (2 * np.var(self.values - np.mean(self.values)))
        self.powers = 2 * self.powers * self._power_norm_factor
        self._power_threshold = self._mc_threshold()

        self._period_hints = self._get_period_hints()

        period = None
        is_valid = False
        for i, p in self._period_hints:
            is_valid, period = self.validate_hint(i)
            if is_valid:
                break

        self._period = period if is_valid else None

    @property
    def period(self):
        return self._period

    def _mc_threshold(self):
        max_powers = []
        shuf = np.copy(self.values)
        for _ in range(40):  # Monte Carlo iterations
            np.random.shuffle(shuf)
            _, powers = LombScargle(self.times, shuf).autopower(normalization='psd')
            max_powers.append(np.max(powers))
        max_powers.sort()
        return max_powers[int(len(max_powers) * 0.99)] * self._power_norm_factor

    def _get_period_hints(self):
        period_hints = []
        for i, period in enumerate(self.periods):
            if self.powers[i] > self._power_threshold and self.time_span / 2 > period > 2 * self.time_interval:
                period_hints.append((i, period))
        return sorted(period_hints, key=lambda p: self.powers[p[0]], reverse=True)

    @staticmethod
    def autocorrelation(values):
        acf = fftconvolve(values, values[::-1], mode='full')
        return acf[acf.size // 2:]

    def validate_hint(self, period_idx):
        peak_idx = np.argmax(self.acf)
        valid = self.acf[peak_idx] > 0
        return valid, self.periods[period_idx] if valid else None

# Step 1: Load the PCA results from the previous step
file_path = '/home/yuanhongxu/tmp/cpc/patientwise_first_principal_component_sequence.csv'
results_df = pd.read_csv(file_path)

# Ensure the PCA results are parsed as Python lists
results_df['FirstPrincipalComponent'] = results_df['FirstPrincipalComponent'].apply(literal_eval)

# Step 2: Define a function to calculate autoperiod for each sequence
def calculate_autoperiod(sequence):
    try:
        times = np.arange(len(sequence))  # Time indices for the sequence
        autoperiod_instance = Autoperiod(times, np.array(sequence))
        return autoperiod_instance.period
    except Exception as e:
        print(f"Error calculating autoperiod for sequence: {e}")
        return -1


results_df['AutoPeriod'] = results_df['FirstPrincipalComponent'].apply(calculate_autoperiod)


output_file = '/home/yuanhongxu/tmp/cpc/patientwise_autoperiods.csv'
results_df[['PatientID', 'AutoPeriod']].to_csv(output_file, index=False)

print(f"AutoPeriods have been calculated and saved to: {output_file}")
