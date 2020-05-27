package com.thobho.sprintpredictor

import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.evaluation.NominalPrediction
import weka.classifiers.rules.DecisionTable
import weka.classifiers.rules.PART
import weka.classifiers.trees.DecisionStump
import weka.classifiers.trees.J48
import weka.classifiers.trees.REPTree
import weka.classifiers.trees.RandomTree
import weka.core.Instances
import weka.core.converters.ConverterUtils


class WekaTest {

    fun classify(model: Classifier,
                 trainingSet: Instances?, testingSet: Instances?): Evaluation {
        val evaluation = Evaluation(trainingSet)
        model.buildClassifier(trainingSet)
        evaluation.evaluateModel(model, testingSet)
        return evaluation
    }

    fun calculateAccuracy(predictions: List<Any>): Double {
        var correct = 0.0
        for (i in predictions.indices) {
            val np = predictions.elementAt(i) as NominalPrediction
            if (np.predicted() == np.actual()) {
                correct++
            }
        }
        return 100 * correct / predictions.size
    }

    fun crossValidationSplit(data: Instances, numberOfFolds: Int): Array<Array<Instances?>> {
        val split = Array(2) { arrayOfNulls<Instances>(numberOfFolds) }
        for (i in 0 until numberOfFolds) {
            split[0][i] = data.trainCV(numberOfFolds, i)
            split[1][i] = data.testCV(numberOfFolds, i)
        }
        return split
    }

    fun testClassifiers() {
        val data = ConverterUtils.DataSource("exampledata.arff").dataSet

        data.setClassIndex(data.numAttributes() - 1)

        // Do 10-split cross validation
        val split = crossValidationSplit(data, 10)

        // Separate split into training and testing arrays
        val trainingSplits = split[0]
        val testingSplits = split[1]

        // Use a set of classifiers
        val models: Array<Classifier> = arrayOf(
                RandomTree(),
                NaiveBayes(),
                REPTree(),
                J48(),  // a decision tree
                PART(),
                DecisionTable(),  //decision table majority classifier
                DecisionStump() //one-level decision tree
        )

        // Run for each model
        for (j in models.indices) {

            // Collect every group of predictions for current model in a FastVector
            val predictions = mutableListOf<Any>()

            // For each training-testing split pair, train and test the classifier
            for (i in trainingSplits.indices) {
                val validation: Evaluation = classify(models[j], trainingSplits[i], testingSplits[i])
                val predictions1 = validation.predictions()
                predictions.addAll(predictions1)

                // Uncomment to see the summary for each training-testing pair.
                //System.out.println(models[j].toString());
            }

            // Calculate overall accuracy of current classifier on all splits
            val accuracy = calculateAccuracy(predictions)

            // Print current classifier's name and accuracy in a complicated,
            // but nice-looking way.
            System.out.println("""
    Accuracy of ${models[j].javaClass}: ${String.format("%.2f%%", accuracy)}
    ---------------------------------
    """.trimIndent())
        }
    }
}