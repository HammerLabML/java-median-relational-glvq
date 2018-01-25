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

import java.util.Arrays;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Benjamin Paassen - bpaassen(at)techfak.uni-bielefeld.de
 */
public class MedianRelationalGLVQTest {

	public MedianRelationalGLVQTest() {
	}

	@BeforeClass
	public static void setUpClass() {
	}

	@AfterClass
	public static void tearDownClass() {
	}

	@Before
	public void setUp() {
	}

	@After
	public void tearDown() {
	}

	/**
	 * Test of train method, of class MedianRelationalGLVQ.
	 */
	@Test
	public void testTrain_with_initialization() {
		final double[] X = {-2, -0.5, -1.5, 1, 1.25, 1.5, 2, 1000, 10000, 100000};

		final int m = X.length;
		// compute labels
		final int[] Y = new int[m];
		for (int i = 0; i < m; i++) {
			Y[i] = X[i] < 0 ? -1 : 1;
		}
		// compute distances
		final double[][] D = new double[m][m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				D[i][j] = Math.abs(X[i] - X[j]);
			}
		}
		// train median relational glvq
		final MedianRelationalGLVQLikelihoodModel model = MedianRelationalGLVQ.train(D, Y);

		// we expect that the prototypes are roughly the class medians (disregarding outliers)
		final int[] W_expected = {2, 5};
		assertArrayEquals(W_expected, model.getPrototypeIndices());
		// we expect that the labels are correct
		final int[] Y_W_expected = new int[W_expected.length];
		for (int k = 0; k < W_expected.length; k++) {
			Y_W_expected[k] = Y[W_expected[k]];
		}
		assertArrayEquals(Y_W_expected, model.getPrototypeLabels());
		// we expect that the likelihood rises monotonously
		double last_likelihood = Double.NEGATIVE_INFINITY;
		for (final double L : model.getLikelihoods()) {
			assertTrue(L > last_likelihood);
			last_likelihood = L;
		}
	}

	/**
	 * Test of train method, of class MedianRelationalGLVQ.
	 */
	@Test
	public void testTrain_without_initialization() {
		final double[] X = {-2, -0.5, -1.5, 1, 1.25, 1.5, 2, 1000, 10000, 100000};
		final int[] W_init = {7, 0};

		final int m = X.length;
		// compute labels
		final int[] Y = new int[m];
		for (int i = 0; i < m; i++) {
			Y[i] = X[i] < 0 ? -1 : 1;
		}
		// compute distances
		final double[][] D = new double[m][m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				D[i][j] = Math.abs(X[i] - X[j]);
			}
		}
		// train median relational glvq
		final MedianRelationalGLVQLikelihoodModel model = MedianRelationalGLVQ.train(D, Y, W_init);

		// we expect that the prototypes are roughly the class medians (disregarding outliers)
		final int[] W_expected = {5, 2};
		assertArrayEquals(W_expected, model.getPrototypeIndices());
		// we expect that the labels are correct
		final int[] Y_W_expected = new int[W_expected.length];
		for (int k = 0; k < W_expected.length; k++) {
			Y_W_expected[k] = Y[W_expected[k]];
		}
		assertArrayEquals(Y_W_expected, model.getPrototypeLabels());
		// we expect that the likelihood rises monotonously
		double last_likelihood = Double.NEGATIVE_INFINITY;
		for (final double L : model.getLikelihoods()) {
			assertTrue(L > last_likelihood);
			last_likelihood = L;
		}
	}

	/**
	 * Test of getClassMemberships method, of class MedianRelationalGLVQ.
	 */
	@Test
	public void testGetClassMemberships() {
		final int[] Y = {1, 2, -1, 1, -1, -1, 1};

		final int[][] expected = {
			{2, 4, 5},
			{0, 3, 6},
			{1}
		};

		final int[][] actual = MedianRelationalGLVQ.getClassMemberships(Y);
		assertTrue(Arrays.deepEquals(expected, actual));
	}

	/**
	 * Test of classify method, of class MedianRelationalGLVQ.
	 */
	@Test
	public void testClassify() {
		// check a simple one-dimensional case where we have one prototype at 1 and one prototype
		// at -1 and all data points smaller 0 are class -1 and all data points larger 0 are class
		// 1.
		final double[] X = {-1.5, -1, 0.25, -0.25, 1};
		final int[] W = {4, 1};

		final int m = X.length;
		// compute labels
		final int[] Y = new int[m];
		for (int i = 0; i < m; i++) {
			Y[i] = X[i] < 0 ? -1 : 1;
		}
		// compute distances
		final double[][] D = new double[m][m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				D[i][j] = Math.abs(X[i] - X[j]);
			}
		}
		// compute prototype labels
		final int[] Y_W = new int[W.length];
		for (int k = 0; k < W.length; k++) {
			Y_W[k] = Y[W[k]];
		}
		final MedianRelationalGLVQModelImpl model = new MedianRelationalGLVQModelImpl(W, Y_W, null);

		// check both variants of the classify function for whole data sets
		assertArrayEquals(Y, MedianRelationalGLVQ.classify(D, model));
		final int K = W.length;
		final double[][] D_p = new double[m][K];
		for (int i = 0; i < m; i++) {
			for (int k = 0; k < K; k++) {
				D_p[i][k] = D[i][W[k]];
			}
		}
		assertArrayEquals(Y, MedianRelationalGLVQ.classify(D_p, model));
	}

	/**
	 * Test of confidence method, of class MedianRelationalGLVQ.
	 */
	@Test
	public void testConfidence() {
		// check a simple one-dimensional case where we have one prototype at 1 and one prototype
		// at -1 and all data points smaller 0 are class -1 and all data points larger 0 are class
		// 1.
		final double[] X = {-1.5, -1, 0.25, -0.25, 1};
		final int[] W = {4, 1};

		final int m = X.length;
		// compute labels
		final int[] Y = new int[m];
		for (int i = 0; i < m; i++) {
			Y[i] = X[i] < 0 ? -1 : 1;
		}
		// compute distances
		final double[][] D = new double[m][m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				D[i][j] = Math.abs(X[i] - X[j]);
			}
		}
		// compute prototype labels
		final int[] Y_W = new int[W.length];
		for (int k = 0; k < W.length; k++) {
			Y_W[k] = Y[W[k]];
		}
		final MedianRelationalGLVQModelImpl model = new MedianRelationalGLVQModelImpl(W, Y_W, null);

		// check both variants of the confidence function for whole data sets
		final double[] expected_confidences = {(2.5 - 0.5) / (2.5 + 0.5),
			(2 - 0) / (2 + 0),
			(1.25 - 0.75) / (1.25 + 0.75),
			(1.25 - 0.75) / (1.25 + 0.75),
			(2 - 0) / (2 + 0)};

		double[][] actual = MedianRelationalGLVQ.confidence(D, model);
		assertEquals(m, actual.length);
		for (int i = 0; i < m; i++) {
			assertEquals(Y[i], actual[i][0], 1E-8);
			assertEquals(expected_confidences[i], actual[i][1], 1E-3);
		}

		final int K = W.length;
		final double[][] D_p = new double[m][K];
		for (int i = 0; i < m; i++) {
			for (int k = 0; k < K; k++) {
				D_p[i][k] = D[i][W[k]];
			}
		}

		actual = MedianRelationalGLVQ.confidence(D_p, model);
		assertEquals(m, actual.length);
		for (int i = 0; i < m; i++) {
			assertEquals(Y[i], actual[i][0], 1E-8);
			assertEquals(expected_confidences[i], actual[i][1], 1E-3);
		}
	}
}
