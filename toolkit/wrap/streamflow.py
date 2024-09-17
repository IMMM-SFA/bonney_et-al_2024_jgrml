import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from pathlib import Path
import matplotlib.dates as mdates

from toolkit.wrap.io import flo_to_df


class WRAPStreamFlow:
    def __init__(self, flo_file, outflow_name=None, ignore_columns=None):
        self.flo_file = flo_file
        self.outflow_name = outflow_name
        self.ignore_columns = ignore_columns        
        
    def load_streamflows(self, start_year=None, end_year=None):
        self.monthly_flo_df = flo_to_df(self.flo_file)
        if self.ignore_columns is not None:
            self.monthly_flo_df = self.monthly_flo_df.loc[:, ~self.monthly_flo_df.columns.isin(self.ignore_columns)]
        if start_year is not None:
            self.monthly_flo_df = self.monthly_flo_df[self.monthly_flo_df.index.year >= start_year]
        if end_year is not None:
            self.monthly_flo_df = self.monthly_flo_df[self.monthly_flo_df.index.year >= end_year]
        self.annual_flo_df = self.monthly_flo_df.groupby(self.monthly_flo_df.index.year).sum()
        
        self.outflow_index = list(self.monthly_flo_df.columns).index(self.outflow_name)

    def _fit_HMM(self, random_seed, n_iter=1000):
        outflow_sf = self.annual_flo_df.loc[:, self.outflow_name].values
        # log_outflow_sf = np.log1p(outflow_sf)
        self.hmm = hmm.GaussianHMM(
            n_components=2, n_iter=n_iter, random_state=random_seed
        ).fit(outflow_sf.reshape(-1, 1))
        return self

    def _generate_synthetic_annual_sf(self, num_years, drought=None, random_state=None):
        if self.hmm is None:
            raise Exception("Model has not been fit.")
        
        # if drought parameter is provided, adjust transition parameters to increase drought.
        if drought is not None:
            if self.hmm.means_[0] < self.hmm.means_[1]: # identify which state is dry and which is wet
                dry_state = 0
                wet_state = 1
            else:
                dry_state = 1
                wet_state = 0

            # increase drought state likelihoods based on climate_drought_adjustment parameter
            self.hmm.transmat_[:,dry_state] = self.hmm.transmat_[:,dry_state] + self.hmm.transmat_[:,wet_state]* drought
            self.hmm.transmat_[:,wet_state] = self.hmm.transmat_[:,wet_state] - self.hmm.transmat_[:,wet_state]* drought

        log_annual_synthetic_outflow_sf = self.hmm.sample(num_years, random_state=random_state)
        # annual_synthetic_outflow_sf = np.exp(log_annual_synthetic_outflow_sf[0]) - 1
        annual_synthetic_outflow_sf = log_annual_synthetic_outflow_sf[0]
        
        return annual_synthetic_outflow_sf

    def _disaggregate_sf(self, annual_synthetic_outflow_sf, random_state=None):
        np.random.seed(random_state)
        outflow_hist_annual_sf = self.annual_flo_df.loc[:, self.outflow_name].values
        hist_years = outflow_hist_annual_sf.shape[0]
        synth_years = annual_synthetic_outflow_sf.shape[0]
        hist_monthly_sf = self.monthly_flo_df.values.reshape(hist_years, 12, -1)
        outflow_hist_monthly_sf = hist_monthly_sf[:, :, self.outflow_index]
        num_sites = len(self.monthly_flo_df.columns)

        # compute the similarities in outflow control point streamflow between synthetic and historical data
        annual_distances = abs(
            np.subtract.outer(annual_synthetic_outflow_sf, outflow_hist_annual_sf)
        )
        annual_distances = annual_distances.squeeze()

        # initialize the full synthetic data array
        synth_monthly_sf = np.zeros([synth_years, 12, num_sites])

        # compute ratios of flow between all control points and the outflow control points in historical data
        Vratios_mh = np.zeros(hist_monthly_sf.shape)
        for i in range(np.shape(hist_monthly_sf)[2]):
            Vratios_mh[:, :, i] = hist_monthly_sf[:, :, i] / outflow_hist_monthly_sf
            
        neighbor_probabilities = np.zeros([int(np.sqrt(hist_years))])
        # We sample from the square root of the number of years many neighbors
        for j in range(len(neighbor_probabilities)):
            neighbor_probabilities[j] = 1 / (j + 1)
        neighbor_probabilities = neighbor_probabilities / np.sum(neighbor_probabilities)

        outflow_temporal_breakdown = np.zeros([hist_years, 12])
        for i in range(outflow_hist_monthly_sf.shape[0]):
            outflow_temporal_breakdown[i, :] = (
                outflow_hist_monthly_sf[i, :] / outflow_hist_annual_sf[i]
            )
        total_zeros = 0
        for j in range(synth_years):
            # select one of k nearest neighbors for each simulated year
            indices = np.argsort(annual_distances[j, :])[
                0 : int(np.sqrt(hist_years))
            ]  # obtain nearest neighbor indices
            neighbor_index = np.random.choice(
                indices, 1, p=neighbor_probabilities
            )  # use probabilities to randomly choose a neighbor

            # use selected neighbor to disaggregate to monthly timescale
            synth_monthly_sf[j, :, -1] = (
                outflow_temporal_breakdown[neighbor_index, :]
                * annual_synthetic_outflow_sf[j]
            )

            # use selected neighbor to disagregate across gage sites
            for k in range(12):
                synth_monthly_sf[j, k, :] = (
                    Vratios_mh[neighbor_index, k, :] * synth_monthly_sf[j, k, -1]
                )
                zeros = (Vratios_mh[neighbor_index, k, :] == 0).sum()
                total_zeros += zeros

        synth_monthly_sf = np.reshape(synth_monthly_sf, (synth_years * 12, num_sites))
        return synth_monthly_sf

    def generate_synthetic_streamflow(self, start_year, num_years, drought=None, random_seed=None):
        self._fit_HMM(random_seed)
        annual_synthetic_outflow_sf = self._generate_synthetic_annual_sf(num_years, drought=drought, random_state=random_seed)
        synthetic_monthly_flow = self._disaggregate_sf(annual_synthetic_outflow_sf, random_state=random_seed)
        start = str(start_year) + "-01"
        end = str(int(start_year) + num_years-1) + "-12"
        time_range = pd.date_range(start=start, end=end, freq="MS")
        assert len(time_range) == synthetic_monthly_flow.shape[0]
        synthetic_streamflow = pd.DataFrame(
            synthetic_monthly_flow, columns=self.monthly_flo_df.columns, index=time_range
        )
        
        return synthetic_streamflow
