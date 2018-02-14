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

package org.uma.jmetal.runner.multiobjective;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import org.uma.jmetal.algorithm.Algorithm;
import org.uma.jmetal.algorithm.InteractiveAlgorithm;
import org.uma.jmetal.algorithm.multiobjective.adm.ArtificialDM;
import org.uma.jmetal.algorithm.multiobjective.adm.ArtificialDMBuilder;
import org.uma.jmetal.algorithm.multiobjective.wasfga.WASFGA;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.SelectionOperator;
import org.uma.jmetal.operator.impl.crossover.SBXCrossover;
import org.uma.jmetal.operator.impl.mutation.PolynomialMutation;
import org.uma.jmetal.operator.impl.selection.BinaryTournamentSelection;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.problem.multiobjective.dtlz.DTLZ1;
import org.uma.jmetal.runner.AbstractAlgorithmRunner;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.util.AlgorithmRunner;
import org.uma.jmetal.util.JMetalException;
import org.uma.jmetal.util.JMetalLogger;
import org.uma.jmetal.util.comparator.RankingAndCrowdingDistanceComparator;
import org.uma.jmetal.util.evaluator.impl.SequentialSolutionListEvaluator;
import org.uma.jmetal.util.fileoutput.SolutionListOutput;
import org.uma.jmetal.util.fileoutput.impl.DefaultFileOutputContext;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;
import org.uma.jmetal.util.referencePoint.ReferencePoint;
import org.uma.jmetal.util.referencePoint.impl.IdealPoint;
import org.uma.jmetal.util.referencePoint.impl.NadirPoint;

/**
 * Class to configure and run the Artificial Decision Making algorithm
 *
 * @author Antonio J. Nebro <antonio@lcc.uma.es>
 * @author Cristobal Barba <cbarba@lcc.uma.es>
 */
public class ArtificialDMRunner extends AbstractAlgorithmRunner {
  /**
   * @param args Command line arguments.
   * @throws JMetalException
   * @throws FileNotFoundException
   * Invoking command:

   */
  public static void main(String[] args) throws JMetalException, FileNotFoundException {
    Problem<DoubleSolution> problem;
    Algorithm<List<DoubleSolution>> algorithm;
    InteractiveAlgorithm<DoubleSolution,List<DoubleSolution>> algorithmRun;
    CrossoverOperator<DoubleSolution> crossover;
    MutationOperator<DoubleSolution> mutation;
    SelectionOperator<List<DoubleSolution>, DoubleSolution> selection;
    int numberIterations =1;
    int numberObjectives = 3;
    int numberVariables = 7;
    String weightsName = "MOEAD_Weights/W3D_100.dat";
    int populationSize=100;

    problem =new DTLZ1(numberVariables,numberObjectives);


    double crossoverProbability = 0.9 ;
    double crossoverDistributionIndex = 20.0 ;
    crossover = new SBXCrossover(crossoverProbability, crossoverDistributionIndex) ;

    double mutationProbability = 1.0 / problem.getNumberOfVariables() ;
    double mutationDistributionIndex = 20.0 ;
    mutation = new PolynomialMutation(mutationProbability, mutationDistributionIndex) ;

    selection = new BinaryTournamentSelection<DoubleSolution>(
        new RankingAndCrowdingDistanceComparator<DoubleSolution>());

    IdealPoint idealPoint = new IdealPoint(problem.getNumberOfObjectives());
    idealPoint.update(problem.createSolution());
    NadirPoint nadirPoint = new NadirPoint(problem.getNumberOfObjectives());
    nadirPoint.update(problem.createSolution());
    double considerationProbability = 0.5;
    List<Double> rankingCoeficient = new ArrayList<>();
    for (int i = 0; i < problem.getNumberOfObjectives() ; i++) {
      rankingCoeficient.add(1.0/problem.getNumberOfObjectives());
    }
    double tolerance = 0.5;

    for (int cont = 0; cont < numberIterations ; cont++) {
      List<Double> referencePoint = new ArrayList<>();

      double epsilon = 0.0045;
      List<Double> asp = new ArrayList<>();
      for (int i = 0; i < problem.getNumberOfObjectives(); i++) {
        asp.add(0.0);//initialize asp to ideal
        referencePoint.add(0.0);//initialization
      }

      algorithmRun = new WASFGA<>(problem, populationSize, 200, crossover, mutation,
            selection, new SequentialSolutionListEvaluator<>(), referencePoint,weightsName);

      algorithm = new ArtificialDMBuilder<>(problem, algorithmRun)
          .setConsiderationProbability(considerationProbability)
          .setMaxEvaluations(11)
          .setTolerance(0.001)
          .setAsp(asp)
          .build();

      AlgorithmRunner algorithmRunner = new AlgorithmRunner.Executor(algorithm)
          .execute();

      List<DoubleSolution> population = algorithm.getResult();
      long computingTime = algorithmRunner.getComputingTime();

      JMetalLogger.logger.info("Total execution time: " + computingTime + "ms");

      new SolutionListOutput(population)
          .setSeparator("\t")
          .setVarFileOutputContext(new DefaultFileOutputContext("VAR.tsv"))
          .setFunFileOutputContext(new DefaultFileOutputContext("FUN.tsv"))
          .print();

      JMetalLogger.logger.info("Random seed: " + JMetalRandom.getInstance().getSeed());
      JMetalLogger.logger.info("Objectives values have been written to file FUN.tsv");
      JMetalLogger.logger.info("Variables values have been written to file VAR.tsv");

      List<ReferencePoint> referencePoints= ((ArtificialDM<DoubleSolution>) algorithm).getReferencePoints();
      if(referencePoints!=null){
        System.out.println("Reference Points");
        for (ReferencePoint rp:referencePoints) {
          for (int i = 0; i <rp.getNumberOfObjectives()-1 ; i++) {
            System.out.print(rp.getObjective(i) +",");
          }
          System.out.print(rp.getObjective(rp.getNumberOfObjectives()-1) );
          System.out.println();
        }
      }

    }
  }

}
