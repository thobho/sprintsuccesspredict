package com.thobho.sprintpredictor

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
class SprintpredictorApplication

fun main(args: Array<String>) {

    //sample predition
    val predictorEngine = PredictorEngine()
    val classess = predictorEngine.classifyNewData(SprintDataUnit(12.0, 100.0, 120.0))
    classess.map {
        println(it)
    }

    //testing various classifiers
    val wekaTest = WekaTest()
    wekaTest.testClassifiers()





//    runApplication<SprintpredictorApplication>(*args)
}
