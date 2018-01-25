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

/**
 * A default implementation of the MedianRelationalGLVQLikelihoodModel interface.
 *
 * @author Benjamin Paassen - bpaassen@techfak.uni-bielefeld.de
 */
public class MedianRelationalGLVQModelImpl implements MedianRelationalGLVQLikelihoodModel {

	private final int[] W;
	private final int[] Y_W;
	private final double[] L;

	/**
	 * Initializes a MedianRelationalGLVQModelImpl
	 *
	 * @param W a K x 1 vector of data point indices, which form the prototypes.
	 * @param Y_W a K x 1 vector of labels for these prototypes.
	 * @param L an array of training errors.
	 */
	public MedianRelationalGLVQModelImpl(int[] W, int[] Y_W, double[] L) {
		final int K = W.length;
		if (K != Y_W.length) {
			throw new IllegalArgumentException("Expected one label per prototype, but got "
					+ K + " prototypes and " + Y_W.length + " labels!");
		}
		for (int k = 0; k < K; k++) {
			if (W[k] < 0) {
				throw new IllegalArgumentException("Prototype indices must be non-negative!");
			}
		}
		this.W = W;
		this.Y_W = Y_W;
		this.L = L;
	}

	@Override
	public int getNumberOfPrototypes() {
		return W.length;
	}

	@Override
	public int[] getPrototypeIndices() {
		return W;
	}

	@Override
	public int[] getPrototypeLabels() {
		return Y_W;
	}

	@Override
	public int getNumberOfEpochs() {
		return L.length - 1;
	}

	@Override
	public double[] getLikelihoods() {
		return L;
	}

}
