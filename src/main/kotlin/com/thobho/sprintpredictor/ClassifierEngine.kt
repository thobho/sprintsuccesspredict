package com.thobho.sprintpredictor

import weka.classifiers.trees.RandomForest
import weka.core.DenseInstance
import weka.core.Instance
import weka.core.Instances
import weka.core.SparseInstance
import weka.core.converters.ConverterUtils
import weka.core.converters.JSONLoader


class PredictorEngine {
    private val dataSource = ConverterUtils.DataSource("exampledata.arff").dataSet
    private val classifier = RandomForest()

    init {
        dataSource.setClassIndex(3)
        classifier.buildClassifier(dataSource)
    }

    fun classifyNewData(sprintDataUnit: SprintDataUnit): DoubleArray {
        val testInstance = SparseInstance(1.0,
                doubleArrayOf(
                        sprintDataUnit.issuesCount,
                        sprintDataUnit.storyPointsSum,
                        sprintDataUnit.averageDescriptionLength))

        testInstance.setDataset(dataSource)

        return classifier.distributionForInstance(testInstance)
    }


}

data class SprintDataUnit(val issuesCount: Double, val storyPointsSum: Double, val averageDescriptionLength: Double)



