import numpy as np
from scipy.stats import poisson
from sklearn import cluster, metrics
from vlhmm_.base_hmm import _BaseHMM


class PoissonHMM(_BaseHMM):
    def _get_rates(self):
        """Emission rate for each state."""
        return self._rates

    def _set_rates(self, rates):
        rates = np.asarray(rates)
        self._rates = rates.copy()

    rates_ = property(_get_rates, _set_rates)

    def _compute_log_likelihood(self, obs):
        n_samples = len(obs)
        log_prob = np.empty((n_samples, self.n_components))
        for c, rate in enumerate(self._rates):
            log_prob[:, c] = poisson.logpmf(obs, rate)
        return log_prob

    def _generate_sample_from_state(self, state, random_state=None):
        return poisson.rvs(self._rates[state])

    def _init(self, obs, params='str'):
        super(PoissonHMM, self)._init(obs, params=params)

        concat_obs = np.concatenate(obs)
        if 'r' in params:
            kmeans = cluster.KMeans(n_clusters=self.n_components)
            self._rates = (kmeans.fit(np.atleast_2d(concat_obs).T)
                           .cluster_centers_.T[0])
            self._rates.sort()

    def _initialize_sufficient_statistics(self):
        stats = super(PoissonHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros(self.n_components)
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(PoissonHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        if 'r' in params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

    def _do_mstep(self, stats, params):
        super(PoissonHMM, self)._do_mstep(stats, params)

        if 'r' in params:
            self._rates = stats['obs'] / stats["post"]

