# Median Relational Generalized Learning Vector Quantization

Copyright (C) 2018
Benjamin Paaßen
AG Machine Learning
Centre of Excellence Cognitive Interaction Technology (CITEC)
University of Bielefeld

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

## Introduction

This is a Java 7, fully MATLAB (R)-compatible implementation of _median relational generalized learning vector quantization_ (MRGLVQ).
_Learning vector quantization_ (LVQ) is a classification algorithm which represents classes in terms
of _prototypes_ and classifies data by assigning each data point to the class of the closest
prototype ([Kohonen, 1995][1]). Median versions of LVQ use data points as prototypes, that is: Each
prototype corresponds exactly to a data point from the training data. This particular
implementataion of median LVQ is _relational_, which means that it is solely based on _distances_.

The input to this algorithm is a m x m distance matrix D, a number of prototypes per class K and a
m x 1 vector of training data labels Y, and the output is an array of prototypes W with K prototypes
per class, given as data point indices. The training is performed according to an expectation
maximization scheme suggested by ([Nebel, Hammer, Frohberg, and Villmann, 2015][3]).

The main advantages of median relational GLVQ compared to other relational classifiers are

1. Because the number of prototypes is small compared to the number of data points, only very few
distances need to be computed to classify new data points.
2. The prototypes used for classification are well-interpretable because they directly correspond
to data points and thus give the option to inspect and explain the model, as well as making sense
of the data.
3. Even atypical distance measures can be treated, e.g. distances which are assymmetric and do not
conform to the triangular inequality.

## Installation

This implementation is written for Java 7 and depends only on the de.cit-ec.ml.rng package for
initialization. It is also fully compatible with Matlab as it only interfaces with primitive data
types. You can access this package by either downloading the [distribution package][4]
or declaring a maven dependency to

<pre>
&lt;dependency&gt;
	&lt;groupId&gt;de.cit-ec.ml&lt;/groupId&gt;
	&lt;artifactId&gt;mrglvq&lt;/artifactId&gt;
	&lt;version&gt;0.1.0&lt;/version&gt;
	&lt;scope&gt;compile&lt;/scope&gt;
&lt;/dependency&gt;
</pre>

If you want to compile the package from source you can download the source code via
`git pull https://gitlab.ub.uni-bielefeld.de/bpaassen/median_relational_glvq.git` and use the
command `mvn package` to compile a .jar distribution.

You can download the javadoc either via maven or directly as part of the [distribution package][4]
or compile it yourself by downloading the source code via 
`git pull https://gitlab.ub.uni-bielefeld.de/bpaassen/median_relational_glvq.git` and using the
command `mvn generate-sources javadoc:javadoc`.

If you want to use the package from MATLAB, please download the [distribution package][4] and add
the line

<pre>javaaddpath mrglvq-0.1.0.jar;</pre>

to your MATLAB script.

## Quickstart

You can obtain a classification model by using the `MedianRelationalGLVQ.train()` method. In
particular, if you have a data set given in term of a distance matrix D and a m x 1 label vector
Y, you can call:

<pre>
final MedianRelationalGLVQModel model = MedianRelationalGLVQ.train(D, Y); // Java
model = de.citec.ml.mrglvq.MedianRelationalGLVQ.train(D, Y); % MATLAB
</pre>

This results in a `MedianRelationalGLVQModel` object which can be used for further queries. In particular:

<pre>final int[] Y2 = MedianRelationalGLVQ.classify(D2, model); // Java
Y2 = de.citec.ml.mrglvq.MedianRelationalGLVQ.classify(D2, model) %MATLAB</pre>

classifies new data points given the distances D2 from the new data points to the prototypes.

## Background

This is a short introduction regarding the background of median relational generalized learning
vector quantization. For a more comprehensive explanation, I recommend to consult
[Nebel et al. (2015)][3].

### Learning Vector Quantization

Assume that we have data points x<sub>1</sub>, ..., x<sub>m</sub> and labels for these data points
y<sub>1</sub>, ..., y<sub>m</sub>. Then, learning vector quantization ([Kohonen, 1995][1]) aims at
finding K prototypes w<sub>1</sub>, ..., w<sub>K</sub> with labels z<sub>1</sub>, ..., z<sub>K</sub>
such that the data points can be correctly classified by using a one-nearest neighbor rule on the
prototypes. That is: If we assign to each data point the label of the closest prototype, we
misclassify as few data points as possible.

This problem as such is NP-hard. However, we can approximate the problem via the generalized
learning vector quantization cost function as proposed by [Sato and Yamada (1995)][2]:

<center>E = &Sigma;<sub>i=1,...,m</sub> &Phi;(
	(d<sub>i</sub><sup>+</sup> - d<sub>i</sub><sup>-</sup>) /
	(d<sub>i</sub><sup>+</sup> + d<sub>i</sub><sup>-</sup>)
)</center>

where d<sub>i</sub><sup>+</sup> is the distance of the i-th data point to the closest prototype
with the same label, d<sub>i</sub><sup>-</sup> is the distance of the i-th data point to the
closest prototype with a different label, and &Phi; is a non-linear function, typically a sigmoid.
Note that this error function measures exactly the classification error if &Phi; is mapped to 0
for inputs smaller than 0 and to 1 for inputs bigger than 0, because the input to &Phi; is bigger
than zero if and only if the data point is further away from a any correct prototype than from
the closest wrong prototype, i.e. if and only if the data point is misclassified.

### Median Relational Generalized Learning Vector Quantization

Median LVQ imposes the additional restriction that prototypes may not be any points in the data
space but are restricted to be one of the training data points. This implies that we can not
optimize the GLVQ error function via continuous schemes like stochastic gradient descent but we
require a discrete optimization scheme. [Nebel et al. (2015)][3] propose an expectation maximization
scheme which re-writes the GLVQ error function to a log-likelihood function as follows:

<center>L = &Sigma;<sub>i=1,...,m</sub> log(4 -
	(d<sub>i</sub><sup>+</sup> - d<sub>i</sub><sup>-</sup>) /
	(d<sub>i</sub><sup>+</sup> + d<sub>i</sub><sup>-</sup>)
)</center>

Note that maximizing this likelihood is equivalent to minimizing E if &Phi; is the log function.
The expectation step consists essentially of re-computing the closest prototypes for each data
point as well as the respective distances, while the maximization step consists of replacing one
prototype with a different data point from the same Voronoi cell such that L is improved.
As an initialization for the prototypes we use [relational neural gas](http://doi.org/10.4119/unibi/2916980)
which already provides a good starting point where prototypes are representative for their respective class.

## License

This documentation is licensed under the terms of the [creative commons attribution-shareAlike 4.0 international (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) license. The code
contained alongside this documentation is licensed unter the
[GNU General Public License Version 3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
A copy of this license is contained in the `gpl-3.0.md` file alongside this README.

## Literature

* Kohonen, T. (1995). Learning Vector Quantization. In: Self-Organizing Maps, 175-189. doi: [10.1007/978-3-642-97610-0_6][1]
* Sato, A., and Yamada, K. (1995). _Generalized Learning Vector Quantization_. In: Tesauro, G., Touretzky, D., and Leen, T. (eds.) Proceedings of the 7th Conference on Advances in Neural Information Processing (NIPS 1995), 423-429. [Link][2]
* Nebel, D., Hammer, B., Frohberg, K., and Villmann, T. (2015). _Median variants of learning vector quantization for learning of dissimilarity data_. Neurocomputing, 169, pp. 295-305. doi: [10.1016/j.neucom.2014.12.096][3]
* Paaßen, B. (2018). _Median Relational Generalized Learning Vector Quantization_. Bielefeld University. doi: [TODO][4]

[1]: https://doi.org/10.1007/978-3-642-97610-0_6 "Kohonen, T. (1995). Learning Vector Quantization. In: Self-Organizing Maps, 175-189"
[2]: https://papers.nips.cc/paper/1113-generalized-learning-vector-quantization "Sato, A., and Yamada, K. (1995). Generalized Learning Vector Quantization. In: Tesauro, G., Touretzky, D., and Leen, T. (eds.) Proceedings of the 7th Conference on Advances in Neural Information Processing (NIPS 1995), 423-429."
[3]: https://doi.org/10.1016/j.neucom.2014.12.096 "Nebel, D., Hammer, B., Frohberg, K., and Villmann, T. (2015). Median variants of learning vector quantization for learning of dissimilarity data. Neurocomputing, 169, pp. 295-305."
[4]: TODO "Paaßen, B. (2018). Median Relational Generalized Learning Vector Quantization. Bielefeld University."
