//  Griewank.java
//
//  Author:
//       Antonio J. Nebro <antonio@lcc.uma.es>
//       Juan J. Durillo <durillo@lcc.uma.es>
//
//  Copyright (c) 2011 Antonio J. Nebro, Juan J. Durillo
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
// 
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

package org.uma.jmetal.problem.singleObjective;

import org.uma.jmetal.core.Problem;
import org.uma.jmetal.core.Solution;
import org.uma.jmetal.core.Variable;
import org.uma.jmetal.encoding.solution.BinaryRealSolution;
import org.uma.jmetal.encoding.solution.RealSolution;
import org.uma.jmetal.util.JMetalException;

/**
 * Class representing problem Griewank
 */

public class Griewank extends Problem {
  /**
   *
   */
  private static final long serialVersionUID = 1531198336732615275L;

  /**
   * Constructor
   * Creates a default instance of the Griewank problem
   *
   * @param numberOfVariables Number of variables of the problem
   * @param solutionType      The solution type must "Real" or "BinaryReal".
   */
  public Griewank(String solutionType, Integer numberOfVariables) throws JMetalException {
    numberOfVariables_ = numberOfVariables;
    numberOfObjectives_ = 1;
    numberOfConstraints_ = 0;
    problemName_ = "Sphere";

    upperLimit_ = new double[numberOfVariables_];
    lowerLimit_ = new double[numberOfVariables_];
    for (int var = 0; var < numberOfVariables_; var++) {
      lowerLimit_[var] = -600.0;
      upperLimit_[var] = 600.0;
    }

    if (solutionType.compareTo("BinaryReal") == 0) {
      solutionType_ = new BinaryRealSolution(this);
    } else if (solutionType.compareTo("Real") == 0) {
      solutionType_ = new RealSolution(this);
    } else {
      throw new JMetalException("Error: solution type " + solutionType + " invalid");
    }
  }

  /**
   * Evaluates a solution
   *
   * @param solution The solution to evaluate
   * @throws org.uma.jmetal.util.JMetalException
   */
  public void evaluate(Solution solution) throws JMetalException {
    Variable[] decisionVariables = solution.getDecisionVariables();

    double sum = 0.0;
    double mult = 1.0;
    double d = 4000.0;
    for (int var = 0; var < numberOfVariables_; var++) {
      sum += decisionVariables[var].getValue() *
        decisionVariables[var].getValue();
      mult *= Math.cos(decisionVariables[var].getValue() / Math.sqrt(var + 1));
    }

    solution.setObjective(0, 1.0 / d * sum - mult + 1.0);
  }
}
