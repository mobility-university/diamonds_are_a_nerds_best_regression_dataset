#!/usr/bin/env python3
import pyreadr
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.evaluation import RegressionEvaluator


if __name__ == "__main__":
    data = pyreadr.read_r("diamonds.rda")["diamonds"]
    print(f"Data: {data}")
    with open("diamonds.csv", "w", encoding="utf-8") as file:
        file.write(data.to_csv())
    with SparkSession.builder.getOrCreate() as spark:
        print()
        print("From now on we just use (py)Spark.")
        print()
        print(f"Context: {spark._sc}")

        df = (
            spark.read.option("header", True)
            .option("inferSchema", "true")
            .csv("diamonds.csv")
        )
        print("DataFrame:")
        df.printSchema()
        df.show(10)

        print(f"number of diamonds: {df.count()}")

        print(
            f'three most expansive diamonds: {", ".join((str(a["price"]) + " USD") for a in df.sort(df.price.desc()).limit(3).collect()) }'
        )
        print(
            f'three cheapest diamonds: {", ".join((str(a["price"]) + " USD") for a in df.sort(df.price.asc()).limit(3).collect()) }'
        )

        print(
            f'available cuts: {", ".join(b["cut"] for b in df.select("cut").distinct().collect())}'
        )
        print(
            f'the following colors are available: {", ".join(b["color"] for b in df.select("color").distinct().collect())}'
        )
        print(
            f'the following clarities are available: {", ".join(b["clarity"] for b in df.select("clarity").distinct().collect())}'
        )

        # train, test = VectorAssembler(inputCols = ['cut', 'carat', 'color', 'clarity'], outputCol = 'features').transform(df).select(['features', 'price']).randomSplit([0.7, 0.3])
        train, test = (
            VectorAssembler(inputCols=["carat"], outputCol="features")
            .transform(df)
            .select(["features", "price"])
            .randomSplit([0.7, 0.3])
        )

        lr = LinearRegression(
            featuresCol="features",
            labelCol="price",
            maxIter=10,
            regParam=0.3,
            elasticNetParam=0.8,
        )
        model = lr.fit(train)
        print(dir(model))

        # lr_model = lr.fit(train_df)
        print("Coefficients: {model.coefficients} Intercept: {model.intercept}")
        print(
            f"Model Summary: RMSE: {model.summary.rootMeanSquaredError} r2: {model.summary.r2}"
        )

        predictions = model.transform(test)
        predictions.select("prediction", "price", "features").show(5)

        evaluator = RegressionEvaluator(
            predictionCol="prediction", labelCol="price", metricName="r2"
        )
        print(f"R Squared (R2) on test data = {evaluator.evaluate(predictions)}")

        (prediction,) = [
            b["prediction"]
            for b in model.evaluate(
                VectorAssembler(inputCols=["carat"], outputCol="features")
                .transform(spark.createDataFrame(data=[dict(carat=1.5, price=0)]))
                .select(["features", "price"])
            ).predictions.collect()
        ]

        print(
            f"Got a 1.5 Carat Diamond. Our model estimates it to {prediction:.2f} USD"
        )

        print("")
        print("now also consider color, cut and clarity")

        cut_idxed = StringIndexer(inputCol="cut", outputCol="cut_idx").fit(df)
        color_idxed = StringIndexer(inputCol="color", outputCol="color_idx").fit(df)
        clarity_idxed = StringIndexer(inputCol="clarity", outputCol="clarity_idx").fit(
            df
        )

        df2 = cut_idxed.transform(color_idxed.transform(clarity_idxed.transform(df)))

        cut_one = OneHotEncoder(inputCol="cut_idx", outputCol="cut_cat").fit(df2)
        color_one = OneHotEncoder(inputCol="color_idx", outputCol="color_cat").fit(df2)
        clarity_one = OneHotEncoder(
            inputCol="clarity_idx", outputCol="clarity_cat"
        ).fit(df2)

        df3 = cut_one.transform(color_one.transform(clarity_one.transform(df2)))
        train, test = (
            VectorAssembler(
                inputCols=["cut_cat", "carat", "color_cat", "clarity_cat"],
                outputCol="features",
            )
            .transform(df3)
            .select(["features", "price"])
            .randomSplit([0.7, 0.3])
        )

        lr = LinearRegression(
            featuresCol="features",
            labelCol="price",
            maxIter=10,
            regParam=0.3,
            elasticNetParam=0.8,
        )
        model = lr.fit(train)
        print(dir(model))

        # lr_model = lr.fit(train_df)
        print("Coefficients: {model.coefficients} Intercept: {model.intercept}")
        print(
            f"Model Summary: RMSE: {model.summary.rootMeanSquaredError} r2: {model.summary.r2}"
        )

        predictions = model.transform(test)
        predictions.select("prediction", "price", "features").show(5)

        evaluator = RegressionEvaluator(
            predictionCol="prediction", labelCol="price", metricName="r2"
        )
        print(f"R Squared (R2) on test data = {evaluator.evaluate(predictions)}")

        predict_df = spark.createDataFrame(
            data=[dict(carat=1.5, cut="Ideal", clarity="VS2", color="H", price=0)]
        )
        predict_df2 = cut_idxed.transform(
            color_idxed.transform(clarity_idxed.transform(predict_df))
        )

        predict_df3 = cut_one.transform(
            color_one.transform(clarity_one.transform(predict_df2))
        )

        (prediction,) = [
            b["prediction"]
            for b in model.evaluate(
                VectorAssembler(
                    inputCols=["cut_cat", "carat", "color_cat", "clarity_cat"],
                    outputCol="features",
                ).transform(predict_df3)
            )
            
            .predictions.collect()
        ]

        print(
            f"Got a 1.5 Carat Diamond with ideal cut VS2 clarity and H color. Our model estimates it to {prediction:.2f} USD"
        )
