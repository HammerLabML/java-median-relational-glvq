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
 * This interface defines the key properties of a MedianRelationalGLVQ model, namely that it
 * contains prototypes in terms of data point indices (getPrototypeIndices()) and that these
 * prototypes have labels (getPrototypeLabels()).
 *
 * @author Benjamin Paassen - bpaassen(at)techfak.uni-bielefeld.de
 */
public interface MedianRelationalGLVQModel {

	/**
	 * Returns the number of prototypes K.
	 *
	 * @return the number of prototypes K.
	 */
	public int getNumberOfPrototypes();

	/**
	 * Returns a K x 1 vector W of training data point indices. These data points are the
	 * prototypes.
	 *
	 * @return a K x 1 vector W of training data point indices. These data points are the
	 * prototypes.
	 */
	public int[] getPrototypeIndices();

	/**
	 * Returns a K x 1 vector Y_W of labels, one per prototype.
	 *
	 * @return a K x 1 vector Y_W of labels, one per prototype.
	 */
	public int[] getPrototypeLabels();
}
