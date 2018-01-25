/* 
 * Median Relational Generalized Learning Vector Quantization
 * 
 * Copyright (C) 2017-2018
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
 * This interface extends the MedianRelationalGLVQModel interface by also storing the training
 * error.
 *
 * @author Benjamin Paassen - bpaassen@techfak.uni-bielefeld.de
 */
public interface MedianRelationalGLVQLikelihoodModel extends MedianRelationalGLVQModel {

	/**
	 * Returns the number of training epochs for this model.
	 *
	 * @return the number of training epochs for this model.
	 */
	public int getNumberOfEpochs();

	/**
	 * Returns the likelihood after each epoch, as well as the initial likelihood, that is an
	 * array with getNumberOfEpochs() + 1 entries where the 0th entry is the initial likelihood
	 * and all other entries correspond to the likelihood after the respective iteration.
	 *
	 * @return the likelihood after each epoch, as well as the initial likelihood.
	 */
	public double[] getLikelihoods();
}
