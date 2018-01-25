/* 
 * Median Relational Generalized Learning Vector Quantization
 * 
 * Copyright (C) 2017
 * Benjamin Paaßen
 * AG Machine Learning
 * Centre of Excellence Cognitive Interaction Technology (CITEC)
 * University of Bielefeld
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package de.citec.ml.mrglvq;

import de.citec.ml.rng.RNGModel;
import de.citec.ml.rng.RelationalNeuralGas;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

/**
 * This class contains a Java implementation of median relational generalized learning vector
 * quantization as proposed by Nebel, Hammer, Frohberg, and Villmann
 * (2015, doi:10.1016/j.neucom.2014.12.096). Given a matrix of pairwise distances D and a
 * vector of labels Y it identifies prototypical data points (i.e. rows of D) which help
 * to classify the data set using a simple nearest neighbor rule. In particular, the algorithm
 * optimizes the generalized learning vector quantization cost function (Sato and Yamada, 1995)
 * via an expectation maximization scheme where in each iteration one prototype 'jumps' to
 * another data point in order to improve the cost function. If the cost function can not be
 * improved anymore for any of the data points, the algorithm terminates.
 *
 * @author Benjamin Paassen - bpaassen(at)techfak.uni-bielefeld.de
 */
public final class MedianRelationalGLVQ {

	private MedianRelationalGLVQ() {

	}

	/**
	 * <p>
	 * Trains a median relational generalized learning vector quantization (MRGLVQ) model for the
	 * given distance matrix D, the given label vector Y. The prototypes are initialized via
	 * relational neural gas.
	 * </p>
	 *
	 * <p>
	 * In particular, let m be the number of data points and L be the number of classes. Then
	 * we expect
	 * </p>
	 *
	 * @param D a m x m matrix of pairwise distances. Note that we do not make only very weak
	 * assumptions regarding the internal structure of this matrix. Mostly, we expect the diagonal
	 * to be the minimum of each row. Other than that, the matrix may be arbitrary. In particular,
	 * the distances may be asymmetric, negative, and not conform to the triangular inequality.
	 *
	 * Note: For the initialization via relational neural gas this matrix has to be symmetric
	 * and non-negative as well. If you have a weaker type of distance, please initialize your
	 * prototypes manually and use the train function with pre-initialized prototypes.
	 * @param Y a m x 1 vector of labels for each data point.
	 *
	 * @return A MedianRelationalGLVQLikelihoodModel with L prototypes, one per class.
	 */
	public static MedianRelationalGLVQLikelihoodModel train(double[][] D, int[] Y) {
		return train(D, Y, 1);
	}

	/**
	 * <p>
	 * Trains a median relational generalized learning vector quantization (MRGLVQ) model for the
	 * given distance matrix D, the given label vector Y, and the given number of prototypes per
	 * class K. The prototypes are initialized via relational neural gas.
	 * </p>
	 *
	 * <p>
	 * In particular, let m be the number of data points and L be the number of classes. Then
	 * we expect:
	 * </p>
	 *
	 * @param D a m x m matrix of pairwise distances. Note that we do not make only very weak
	 * assumptions regarding the internal structure of this matrix. Mostly, we expect the diagonal
	 * to be the minimum of each row. Other than that, the matrix may be arbitrary. In particular,
	 * the distances may be asymmetric, negative, and not conform to the triangular inequality.
	 *
	 * Note: For the initialization via relational neural gas this matrix has to be symmetric
	 * and non-negative as well. If you have a weaker type of distance, please initialize your
	 * prototypes manually and use the train function with pre-initialized prototypes.
	 * @param Y a m x 1 vector of labels for each data point.
	 * @param K a positive integer specifying the number of prototypes per class
	 *
	 * @return A MedianRelationalGLVQLikelihoodModel with K * L prototypes, K per class.
	 */
	public static MedianRelationalGLVQLikelihoodModel train(double[][] D, int[] Y, int K) {
		// check the inputs
		final int m = D.length;
		for (int i = 0; i < m; i++) {
			if (D[i].length != m) {
				throw new IllegalArgumentException("Expected a square input distance matrix, but row "
						+ i + " had " + D[i].length + " columns instead of " + m + " ones!");
			}
		}
		if (Y.length != m) {
			throw new IllegalArgumentException("Expected " + m + " labels but got " + Y.length + " ones!");
		}
		if (K < 1) {
			throw new IllegalArgumentException("The number of prototypes per class must be at least 1");
		}

		// initialize the prototypes via relational neural gas. In particular, we train a
		// separate relational neural gas model for each class.
		final int[][] members = getClassMemberships(Y);
		final int L = members.length;
		final int[] W_init = new int[L * K];
		for (int l = 0; l < L; l++) {
			// subselect the distance matrix only for this class
			final int m_l = members[l].length;
			final double[][] D_l = new double[m_l][m_l];
			for (int i = 0; i < m_l; i++) {
				for (int j = 0; j < m_l; j++) {
					D_l[i][j] = D[members[l][i]][members[l][j]];
				}
			}
			// train a relational neural gas model for the current class
			final RNGModel rngModel_l = RelationalNeuralGas.train(D_l, K);
			// use the exemplars as initial prototypes
			final int[] exemplars = RelationalNeuralGas.getExamplars(rngModel_l);
			for (int k = 0; k < K; k++) {
				W_init[l * K + k] = members[l][exemplars[k]];
			}
		}
		// then use the actual median relational GLVQ training
		return train(D, Y, W_init);
	}

	/**
	 * <p>
	 * Trains a median relational generalized learning vector quantization (MRGLVQ) model for the
	 * given distance matrix D, the given label vector Y, and the given initial prototypes W_init.
	 * </p>
	 *
	 * <p>
	 * In particular, let m be the number of data points and K be the number of prototypes. Then
	 * we expect:
	 * </p>
	 *
	 * @param D a m x m matrix of pairwise distances. Note that we do not make only very weak
	 * assumptions regarding the internal structure of this matrix. Mostly, we expect the diagonal
	 * to be the minimum of each row. Other than that, the matrix may be arbitrary. In particular,
	 * the distances may be asymmetric, negative, and not conform to the triangular inequality.
	 * @param Y a m x 1 vector of labels for each data point.
	 * @param W_init a K x 1 vector of initial prototypes, given as data point indices. That means:
	 * We expect that this vector contains integers in the range [0, m-1] without repetitions.
	 *
	 * @return A MedianRelationalGLVQLikelihoodModel with K prototypes.
	 */
	public static MedianRelationalGLVQLikelihoodModel train(double[][] D, int[] Y, int[] W_init) {
		// check the inputs
		final int m = D.length;
		for (int i = 0; i < m; i++) {
			if (D[i].length != m) {
				throw new IllegalArgumentException("Expected a square input distance matrix, but row "
						+ i + " had " + D[i].length + " columns instead of " + m + " ones!");
			}
		}
		if (Y.length != m) {
			throw new IllegalArgumentException("Expected " + m + " labels but got " + Y.length + " ones!");
		}
		final int K = W_init.length;
		for (int k = 0; k < K; k++) {
			if (W_init[k] < 0 || W_init[k] >= m) {
				throw new IllegalArgumentException("Expected the third argument to be initial "
						+ "prototypes in terms of data point indices in the range[0," + (m - 1)
						+ "], but the " + k + "th prototype was " + W_init[k] + "!");
			}
		}
		// initialize the prototypes
		final int[] W = new int[K];
		System.arraycopy(W_init, 0, W, 0, K);
		// Compute the following helper variables:
		// the closest prototype with the same label for each data point
		final int[] k_plus = new int[m];
		// closest prototype with a different label for each data point
		final int[] k_minus = new int[m];
		// the expected probability that a data point is assigned to the closest
		// prototype of the correct class.
		final double[] gamma_plus = new double[m];
		// the expected likelihood per data point, which we need for the maximization step
		final double[] L = new double[m];
		// the overall likelihood over all optimization runs
		final ArrayList<Double> Likelihoods = new ArrayList<>();
		double Likelihood = 0;
		for (int i = 0; i < m; i++) {
			// compute the closest prototype from the same and from a different class for
			// the current data point
			k_plus[i] = -1;
			k_minus[i] = -1;
			double d_plus = Double.POSITIVE_INFINITY;
			double d_minus = Double.POSITIVE_INFINITY;
			for (int k = 0; k < K; k++) {
				if (Y[i] == Y[W[k]]) {
					if (D[i][W[k]] < d_plus) {
						d_plus = D[i][W[k]];
						k_plus[i] = k;
					}
				} else {
					if (D[i][W[k]] < d_minus) {
						d_minus = D[i][W[k]];
						k_minus[i] = k;
					}
				}
			}
			if (k_plus[i] < 0) {
				throw new IllegalArgumentException("There was no prototype with the label " + Y[i]);
			}
			if (k_minus[i] < 0) {
				throw new IllegalArgumentException("There was no prototype with a label different than " + Y[i]);
			}
			// compute the cost function contribution
			final double Z = d_plus + d_minus;
			final double g_plus = 2 - d_plus / Z;
			final double g_minus = 2 + d_minus / Z;
			gamma_plus[i] = g_plus / (g_plus + g_minus);
			L[i] = gamma_plus[i] * Math.log(g_plus) + (1. - gamma_plus[i]) * Math.log(g_minus);
			Likelihood += L[i] - gamma_plus[i] * Math.log(gamma_plus[i]) - (1. - gamma_plus[i]) * Math.log(1. - gamma_plus[i]);
		}
		Likelihoods.add(Likelihood);

		// then start the expectation maximization algorithm. In each iteration, we try to improve
		// one of the prototypes and we cycle through the prototypes
		int k = 0;
		while (true) {
			// look for a prototype that we can still improve. If we have cycled through all
			// prototypes, stop the algorithm.
			final int k_init = k;
			boolean init = true;
			while (init || k != k_init) {

				// MAXIMIZATION STEP
				// check whether we can improve prototype k by considering all data points in its
				// Voronoi cell/receptive field and considering the expected likelihood if we would
				// change the prototype to this data point
				int best_j = -1;
				double best_likelihood_gain = 0;
				for (int j = 0; j < m; j++) {
					// consider only data points j in the Voronoi cell of k that are not k
					if (k_plus[j] != k || j == W[k]) {
						continue;
					}
					// a change to the expected likelihood for data point i occurs in exactly
					// four cases:
					// 1) k was the closest prototype with the same label for i (i.e. i &in;
					// V_plus.get(k) ). In that case, d_plus changes.
					// 2) i &notin; V_plus.get(k), but k_new is the closest prototype with the same
					// label for i, i.e. D[i][k_new] < d_plus. In that case, d_plus changes.
					// 3) k was the closest prototype with a different label for i (i.e. i &in;
					// V_minus.get(k) ). In that case, d_minus changes.
					// 4) i &notin; V_minus.get(k), but k_new is the closest prototype with a
					// different label for i, i.e. D[i][k_new] < d_minus. In that case,
					// d_minus changes as well.

					// we check these four cases systematically for each data point i.
					double likelihood_gain = 0;
					for (int i = 0; i < m; i++) {
						// check if the data point has the same label as the prototype
						if (Y[i] == Y[j]) {
							// if the data point has the same label, check whether it was in the
							// Voronoi cell of this prototype before.
							if (k_plus[i] == k) {
								// if the data point was in the Voronoi cell of this prototype,
								// compute the new d_plus value
								double d_plus_new = D[i][j];
								for (int k2 = 0; k2 < K; k2++) {
									if (Y[W[k2]] != Y[i] || k2 == k) {
										continue;
									}
									if (D[i][W[k2]] < d_plus_new) {
										d_plus_new = D[i][W[k2]];
									}
								}
								// and we compute the likelihood gain for this data point
								final double d_minus = D[i][W[k_minus[i]]];
								final double Z_new = d_plus_new + d_minus;
								final double g_plus_new = 2 - d_plus_new / Z_new;
								final double g_minus_new = 2 + d_minus / Z_new;
								likelihood_gain += gamma_plus[i] * Math.log(g_plus_new)
										+ (1 - gamma_plus[i]) * Math.log(g_minus_new) - L[i];
							} else {
								// if the data point was not in the Voronoi cell of this prototype
								// before, check if it is now.
								if (D[i][j] < D[i][k_plus[i]]) {
									// if it is, compute the likelihood gain for this data point
									final double d_plus_new = D[i][j];
									final double d_minus = D[i][W[k_minus[i]]];
									final double Z_new = d_plus_new + d_minus;
									final double g_plus_new = 2 - d_plus_new / Z_new;
									final double g_minus_new = 2 + d_minus / Z_new;
									likelihood_gain += gamma_plus[i] * Math.log(g_plus_new)
											+ (1 - gamma_plus[i]) * Math.log(g_minus_new) - L[i];
								}
							}
						} else {
							// if the data pooint has not the same lebel, check if it was in the
							// Voronoi cell of this prototype before
							if (k_minus[i] == k) {
								// if the data point was in the Voronoi cell of this prototype,
								// compute the new d_minus value
								double d_minus_new = D[i][j];
								for (int k2 = 0; k2 < K; k2++) {
									if (Y[W[k2]] == Y[i] || k2 == k) {
										continue;
									}
									if (D[i][W[k2]] < d_minus_new) {
										d_minus_new = D[i][W[k2]];
									}
								}
								// and we compute the likelihood gain for this data point
								final double d_plus = D[i][W[k_plus[i]]];
								final double Z_new = d_plus + d_minus_new;
								final double g_plus_new = 2 - d_plus / Z_new;
								final double g_minus_new = 2 + d_minus_new / Z_new;
								likelihood_gain += gamma_plus[i] * Math.log(g_plus_new)
										+ (1 - gamma_plus[i]) * Math.log(g_minus_new) - L[i];
							} else {
								// if the data point was not in the Voronoi cell of this prototype
								// before, check if it is now.
								if (D[i][j] < D[i][k_minus[i]]) {
									// if it is, compute the likelihood gain for this data point
									final double d_plus = D[i][W[k_plus[i]]];
									final double d_minus_new = D[i][j];
									final double Z_new = d_plus + d_minus_new;
									final double g_plus_new = 2 - d_plus / Z_new;
									final double g_minus_new = 2 + d_minus_new / Z_new;
									likelihood_gain += gamma_plus[i] * Math.log(g_plus_new)
											+ (1 - gamma_plus[i]) * Math.log(g_minus_new) - L[i];
								}
							}
						}
					}
					// check if we would gain more likelihood than before if we change to this
					// data point
					if (likelihood_gain > best_likelihood_gain) {
						best_j = j;
						best_likelihood_gain = likelihood_gain;
					}
				}

				// EXPECTATION STEP
				// check if we have a likelihood gain greater 0
				if (best_likelihood_gain > 0) {
					Likelihood = 0;
					// if so, we change the prototype k to the new k
					W[k] = best_j;
					// and we perform the expectation step, meaning we update d_plus, d_minus,
					// V_plus, V_minus, g_plus and L
					for (int i = 0; i < m; i++) {
						// check if the data point has the same label as the prototype
						if (Y[i] == Y[best_j]) {
							// if the data point has the same label, check whether it was in the
							// Voronoi cell of this prototype before.
							if (k_plus[i] == k) {
								// if the data point was in the Voronoi cell of this prototype,
								// check whether it still is
								double d_plus_new = D[i][best_j];
								for (int k2 = 0; k2 < K; k2++) {
									if (Y[W[k2]] != Y[i]) {
										continue;
									}
									if (D[i][W[k2]] < d_plus_new) {
										d_plus_new = D[i][W[k2]];
										k_plus[i] = k2;
									}
								}
								// and we compute the new gamma_plus and the new likelihood for
								// this data point
								final double d_minus = D[i][W[k_minus[i]]];
								final double Z_new = d_plus_new + d_minus;
								final double g_plus_new = 2 - d_plus_new / Z_new;
								final double g_minus_new = 2 + d_minus / Z_new;
								gamma_plus[i] = g_plus_new / (g_plus_new + g_minus_new);
								L[i] = gamma_plus[i] * Math.log(g_plus_new) + (1. - gamma_plus[i]) * Math.log(g_minus_new);
							} else {
								// if the data point was not in the Voronoi cell of this prototype
								// before, check if it is now.
								if (D[i][best_j] < D[i][k_plus[i]]) {
									// if it is, re-set k_plus and re-compute gamma_plus and L
									k_plus[i] = k;
									final double d_plus_new = D[i][best_j];
									final double d_minus = D[i][W[k_minus[i]]];
									final double Z_new = d_plus_new + d_minus;
									final double g_plus_new = 2 - d_plus_new / Z_new;
									final double g_minus_new = 2 + d_minus / Z_new;
									gamma_plus[i] = g_plus_new / (g_plus_new + g_minus_new);
									L[i] = gamma_plus[i] * Math.log(g_plus_new) + (1 - gamma_plus[i]) * Math.log(g_minus_new);
								}
							}
						} else {
							// if the data pooint has not the same lebel, check if it was in the
							// Voronoi cell of this prototype before
							if (k_minus[i] == k) {
								// if the data point was in the Voronoi cell of this prototype,
								// check whether it still is
								double d_minus_new = D[i][best_j];
								for (int k2 = 0; k2 < K; k2++) {
									if (Y[W[k2]] == Y[i]) {
										continue;
									}
									if (D[i][W[k2]] < d_minus_new) {
										d_minus_new = D[i][W[k2]];
										k_minus[i] = k2;
									}
								}
								// and we compute the new gamma_plus and the new likelihood for
								// this data point
								final double d_plus = D[i][W[k_plus[i]]];
								final double Z_new = d_plus + d_minus_new;
								final double g_plus_new = 2 - d_plus / Z_new;
								final double g_minus_new = 2 + d_minus_new / Z_new;
								gamma_plus[i] = g_plus_new / (g_plus_new + g_minus_new);
								L[i] = gamma_plus[i] * Math.log(g_plus_new) + (1 - gamma_plus[i]) * Math.log(g_minus_new);
							} else {
								// if the data point was not in the Voronoi cell of this prototype
								// before, check if it is now.
								if (D[i][best_j] < D[i][k_minus[i]]) {
									// if it is, re-set k_minus and re-compute gamma_plus and L
									k_minus[i] = k;
									final double d_plus = D[i][W[k_plus[i]]];
									final double d_minus_new = D[i][best_j];
									final double Z_new = d_plus + d_minus_new;
									final double g_plus_new = 2 - d_plus / Z_new;
									final double g_minus_new = 2 + d_minus_new / Z_new;
									gamma_plus[i] = g_plus_new / (g_plus_new + g_minus_new);
									L[i] = gamma_plus[i] * Math.log(g_plus_new) + (1 - gamma_plus[i]) * Math.log(g_minus_new);
								}
							}
						}
						Likelihood += L[i] - gamma_plus[i] * Math.log(gamma_plus[i]) - (1. - gamma_plus[i]) * Math.log(1. - gamma_plus[i]);
					}
					Likelihoods.add(Likelihood);

					// after the expectation step, stop looking for prototypes to improve in this
					// iteration
					break;
				}

				// get the next prototype
				init = false;
				k++;
				if (k >= K) {
					k = 0;
				}
			}

			// if we did not find any prototypes to improve, end the optimization
			if (!init && k_init == k) {
				break;
			}
			// cycle to the next prototype for the next iteration
			k++;
			if (k >= K) {
				k = 0;
			}

		}
		// transform the likelihoods list to a primitive array
		final double[] Ls_arr = new double[Likelihoods.size()];
		for (int e = 0; e < Likelihoods.size(); e++) {
			Ls_arr[e] = Likelihoods.get(e);
		}
		// retrieve the prototype labels
		final int[] Y_W = new int[K];
		for (k = 0; k < K; k++) {
			Y_W[k] = Y[W[k]];
		}

		return new MedianRelationalGLVQModelImpl(W, Y_W, Ls_arr);
	}

	/**
	 * Computes the sets of members for each class. In particular, let m be the number of data
	 * points and L be the number of classes. Then we expect:
	 *
	 * @param Y a m x 1 vector of labels.
	 *
	 * @return an array with L elements, each of which is an array containing the indices of all
	 * data points with the same label.
	 */
	public static int[][] getClassMemberships(int[] Y) {
		final TreeMap<Integer, List<Integer>> memberMap = new TreeMap<>();
		for (int i = 0; i < Y.length; i++) {
			List<Integer> members = memberMap.get(Y[i]);
			if (members == null) {
				members = new ArrayList<>();
				memberMap.put(Y[i], members);
			}
			members.add(i);
		}
		final int[][] memberArr = new int[memberMap.size()][];
		int l = 0;
		for (final List<Integer> members : memberMap.values()) {
			memberArr[l] = new int[members.size()];
			for (int j = 0; j < members.size(); j++) {
				memberArr[l][j] = members.get(j);
			}
			l++;
		}
		return memberArr;
	}

	/**
	 * Classifies data using the given median relational GLVQ model. In particular, assume that
	 * the model contains K prototypes, the training data set contains m data points and the
	 * test data set contains n data points. Then we expect:
	 *
	 * @param D a n x m matrix or a n x K matrix of distances from the test to the training data
	 * points or the test data points to the prototypes.
	 * @param model a MedianRelationalGLVQModel.
	 *
	 * @return a vector of labels for each data point.
	 */
	public static int[] classify(double[][] D, MedianRelationalGLVQModel model) {
		final int n = D.length;
		final int[] Y_out = new int[n];
		for (int j = 0; j < n; j++) {
			Y_out[j] = classify(D[j], model);
		}
		return Y_out;
	}

	/**
	 * Classifies a data point using the given median relational GLVQ model. In particular, assume
	 * that the model contains K prototypes and the training data set contains m data points.
	 * Then, we expect:
	 *
	 * @param d a 1 x m vector or a 1 x K vector of distances from the test data point to the
	 * training data points or just to the prototypes.
	 * @param model a MedianRelationalGLVQModel.
	 *
	 * @return the label of the closest prototype to the data point, meaning Y[k] where k is
	 * argmin d.
	 */
	public static int classify(double[] d, MedianRelationalGLVQModel model) {
		// check the inputs
		final int K = model.getNumberOfPrototypes();

		if (d.length == K) {
			int k_plus = 0;
			for (int k = 1; k < K; k++) {
				if (d[k] < d[k_plus]) {
					k_plus = k;
				}
			}
			return model.getPrototypeLabels()[k_plus];
		} else {
			final int[] W = model.getPrototypeIndices();
			int k_plus = 0;
			for (int k = 1; k < K; k++) {
				if (d[W[k]] < d[W[k_plus]]) {
					k_plus = k;
				}
			}
			return model.getPrototypeLabels()[k_plus];
		}
	}

	/**
	 * Classifies data using the given median relational GLVQ model and returns the predicted labels
	 * as well as the confidence in the prediction. In particular, assume that
	 * the model contains K prototypes, the training data set contains m data points and the
	 * test data set contains n data points. Then we expect:
	 *
	 * @param D a n x m matrix or a n x K matrix of distances from the test to the training data
	 * points or the test data points to the prototypes.
	 * @param model a MedianRelationalGLVQModel.
	 *
	 * @return a n x 2 matrix containing the predicted label for each data point in the first column
	 * and the confidence in the prediction as the second column.
	 */
	public static double[][] confidence(double[][] D, final MedianRelationalGLVQModel model) {

		final int n = D.length;
		final double[][] pred = new double[n][];
		for (int j = 0; j < n; j++) {
			pred[j] = confidence(D[j], model);
		}
		return pred;
	}

	/**
	 * Classifies a data point using the given median relational GLVQ model and returns the label as
	 * well as the confidence. In particular, assume
	 * that the model contains K prototypes and the training data set contains m data points.
	 * Then, we expect:
	 *
	 * @param d a 1 x m vector or a 1 x K vector of distances from the test data point to the
	 * training data points or just to the prototypes.
	 * @param model a MedianRelationalGLVQModel.
	 *
	 * @return a 1 x 2 vector containing the predicted label as first entry and the confidence of
	 * the prediction as second entry. In particular, the predicted label is Y[k] where k is
	 * argmin d. The confidence is (d^- - d^+) / (d^+ + d^-) where d^+ is min d and d^- is the
	 * second-lowest distance to a prototype in d with a different label than argmin d.
	 */
	public static double[] confidence(double[] d, final MedianRelationalGLVQModel model) {
		// check the inputs
		final int K = model.getNumberOfPrototypes();

		if (d.length == K) {
			// do one run over the distances to retreive the closest prototype
			int k_plus = 0;
			for (int k = 1; k < K; k++) {
				if (d[k] < d[k_plus]) {
					k_plus = k;
				}
			}
			// compute the predicted label as the label of the closest prototype
			final int y = model.getPrototypeLabels()[k_plus];

			// do a second run to retreive the closest prototpe with a different label
			int k_minus = -1;
			for (int k = 0; k < K; k++) {
				if (model.getPrototypeLabels()[k] != y && (k_minus < 0 || d[k] < d[k_minus])) {
					k_minus = k;
				}
			}
			// compute the confidence
			final double confidence = (d[k_minus] - d[k_plus]) / (d[k_plus] + d[k_minus]);
			return new double[]{y, confidence};
		} else {
			// do one run over the distances to retreive the closest prototype
			final int[] W = model.getPrototypeIndices();
			int k_plus = 0;
			for (int k = 1; k < K; k++) {
				if (d[W[k]] < d[W[k_plus]]) {
					k_plus = k;
				}
			}
			// compute the predicted label as the label of the closest prototype
			final int y = model.getPrototypeLabels()[k_plus];

			// do a second run to retreive the closest prototpe with a different label
			int k_minus = -1;
			for (int k = 0; k < K; k++) {
				if (model.getPrototypeLabels()[k] != y && (k_minus < 0 || d[W[k]] < d[W[k_minus]])) {
					k_minus = k;
				}
			}
			// compute the confidence
			final double confidence = (d[W[k_minus]] - d[W[k_plus]]) / (d[W[k_plus]] + d[W[k_minus]]);
			return new double[]{y, confidence};
		}

	}

}
