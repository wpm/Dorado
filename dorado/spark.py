import dorado.cli

if __name__ == "__main__":
    from pyspark import SparkContext

    spark_context = SparkContext()
    dorado.cli.run(spark_context)
