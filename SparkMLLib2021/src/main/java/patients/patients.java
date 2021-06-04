package patients;

import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.feature.*;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import java.util.Arrays;
import static org.apache.spark.sql.types.DataTypes.*;


public class patients {
	public static Dataset<Row> loadData(SparkSession sc,String filePath)
	{
		StructType schema = new StructType(new StructField[]{
				createStructField("Id",IntegerType,true), 
				createStructField("IndicadorDemencia",IntegerType,true), 
				createStructField("IndicadorConstipacion",IntegerType,true), 
				createStructField("IndicadorSordera",IntegerType,true), 
				createStructField("IndicadorAltVisual",IntegerType,true), 
				createStructField("Barthel",FloatType,true), 
				createStructField("Pfeiffer",FloatType,true), 
				createStructField("DiferenciaBarthel",IntegerType,true),
				createStructField("DiferenciaPfeiffer",IntegerType,true), 
				createStructField("Hemoglobina",FloatType,true),
				createStructField("Creatinina",FloatType,true), 
				createStructField("Albumina",FloatType,true), 
				createStructField("ListaDiagnosticosPri",StringType,true),
				createStructField("ListaDiagnosticosSec",StringType,true),
				createStructField("ListaProcedimientosPri",StringType,true), 
				createStructField("ListaProcedimientosSec",StringType,true), 
				createStructField("ListaCausasExternas",StringType,true), 
				createStructField("Reingreso",IntegerType,true), 
				createStructField("DiasEstancia",IntegerType,true)
				});
		
	      Dataset<Row> dataset = sc.read().format("csv")
	    		  .option("sep",";")
	    		  .option("nullValue", "NA")
//	    		  .option("inferSchema","true")
	    		  .option("header","true")
	    		  .schema(schema)
	    		  .load(filePath);
	      return(dataset);
	}


	public static Dataset<Row> naImputation(Dataset<Row> source)
	{
		// We decide to fill all columns with NA values as -1000, so that we don't have to write all
		// their names in a vector, as well as because of the fact that later on all of them will be
		// tackled to deal with such Null values:
		String[] columns = source.columns();
		Dataset <Row >output = source.na().fill(-1000, columns);
		  
		 return(output);
	}


	public static Dataset<Row> bucketizeFeatures(Dataset<Row> source)
	{
		// We bucketize continuous variables in ranges using domain knowledge:
		double[] diasEstancia = {Double.NEGATIVE_INFINITY, -999.0,12.0,42.0,71.0, Double.POSITIVE_INFINITY};
		double[] hemoglobina = {Double.NEGATIVE_INFINITY, -999.0,12.0, Double.POSITIVE_INFINITY};
		double[] creatinina = {Double.NEGATIVE_INFINITY, -999.0,1.12, Double.POSITIVE_INFINITY};
		double[] albumina = {Double.NEGATIVE_INFINITY, -999.0,3.5,5.1, Double.POSITIVE_INFINITY};
		double[] barthel = {Double.NEGATIVE_INFINITY, -999.0,20.0,61.0,91.0,99.0, Double.POSITIVE_INFINITY};
		double[] pfeiffer = {Double.NEGATIVE_INFINITY, -999.0,2.0,4.0,8.0, Double.POSITIVE_INFINITY};
		double[] difBarthel = {Double.NEGATIVE_INFINITY, -999.0,-20.0,21.0, Double.POSITIVE_INFINITY};
		double[] difPfeiffer = {Double.NEGATIVE_INFINITY, -999.0,-2.0,3.0, Double.POSITIVE_INFINITY};

		double[][] splitsArray = {diasEstancia, hemoglobina, creatinina, albumina, barthel, pfeiffer, difBarthel,
		difPfeiffer};

		String[] discretizedColnames = {"DiasEstancia", "Hemoglobina", "Creatinina", "Albumina",
				"Barthel", "Pfeiffer", "DiferenciaBarthel", "DiferenciaPfeiffer"};

		String[] outputCols = {"BDiasEstancia", "BHemoglobina", "BCreatinina", "BAlbumina",
				"BBarthel", "BPfeiffer", "BDiferenciaBarthel", "BDiferenciaPfeiffer"};

		Bucketizer bucketizer = new Bucketizer()
				.setInputCols(discretizedColnames)
				.setOutputCols(outputCols)
				.setSplitsArray(splitsArray);

		Dataset<Row> output = bucketizer.transform(source).drop(discretizedColnames);

	  return(output);
	}


	public static Dataset<Row> oneHotEncoding(Dataset<Row> source)
	{
		OneHotEncoderEstimator vecVarEncoder = new OneHotEncoderEstimator();
		OneHotEncoderEstimator indicatorVarEncoder = new OneHotEncoderEstimator();

		String[] vecInputCols = {"BHemoglobina", "BCreatinina", "BAlbumina",
				"BBarthel", "BPfeiffer", "BDiferenciaBarthel", "BDiferenciaPfeiffer", "Reingreso"};

		String[] vecOutputCols = {"VecHemoglobina", "VecCreatinina", "VecAlbumina",
				"VecBarthel", "VecPfeiffer", "VecDiferenciaBarthel", "VecDiferenciaPfeiffer", "VecReingreso"};

		String[] indicatorInputCols = {"IndicadorDemencia3", "IndicadorConstipacion2", "IndicadorSordera2",
				"IndicadorAltVisual2"};

		String[] indicatorOutputCols = {"VecIndicadorDemencia", "VecIndicadorConstipacion", "VecIndicadorSordera",
				"VecIndicadorAltVisual"};

		// In order to apply OneHot Encoding to Indicator Columns, we need first to deal with '-1000' values. We
		// checked the minimum and maximum values of these columns and we realised that all of them don't use the
		// '0' index (which has been used for 'NA' when bucketing), except the column 'IndicadorDemencia'.

		// Consequently, we convert the '0' and '1' Binary values of the 'IndicadorDemencia' to '1' and '2',
		// respectively, so that they are equal to the rest of columns and then we can interpret the -1000 of the
		// NA values as '0' for all of the Indicator Columns as well:
		source.createOrReplaceTempView("temp");
		Dataset<Row> output = source.sqlContext().sql("SELECT *, " +
				"CASE WHEN IndicadorDemencia > -1000 THEN IndicadorDemencia+1 ELSE IndicadorDemencia END " +
				"AS IndicadorDemencia2 FROM temp").drop("IndicadorDemencia");

		output.createOrReplaceTempView("temp2");
		output = output.sqlContext().sql("SELECT *, " +
				"CASE WHEN IndicadorDemencia2 == -1000 THEN 0 ELSE IndicadorDemencia2 END " +
				"AS IndicadorDemencia3, " +
				"CASE WHEN IndicadorConstipacion == -1000 THEN 0 ELSE IndicadorConstipacion END " +
				"AS IndicadorConstipacion2, " +
				"CASE WHEN IndicadorSordera == -1000 THEN 0 ELSE IndicadorSordera END " +
				"AS IndicadorSordera2, " +
				"CASE WHEN IndicadorAltVisual == -1000 THEN 0 ELSE IndicadorAltVisual END " +
				"AS IndicadorAltVisual2 FROM temp2")
				.drop("IndicadorDemencia2", "IndicadorConstipacion", "IndicadorSordera", "IndicadorAltVisual");

		ParamMap paramMap = new ParamMap()
				.put(vecVarEncoder.inputCols(), vecInputCols)
				.put(vecVarEncoder.outputCols(), vecOutputCols)
				.put(indicatorVarEncoder.inputCols(), indicatorInputCols)
				.put(indicatorVarEncoder.outputCols(), indicatorOutputCols);

		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{vecVarEncoder, indicatorVarEncoder});
		PipelineModel model = pipeline.fit(output, paramMap);
		output = model.transform(output).drop(vecInputCols).drop(indicatorInputCols);

		return(output);
	}


	public static Dataset<Row> listsToVectors(Dataset<Row> source)
	{
		String[] inputCols = {"ListaDiagnosticosPri", "ListaDiagnosticosSec", "ListaProcedimientosPri",
		"ListaProcedimientosSec", "ListaCausasExternas"};

		// Using a for-loop to perform the removal of leading/trailing spaces, the tokenization and
		// the vectorization of the List variables:
		Dataset<Row> result = source;
		for (int i=0; i < inputCols.length; i++) {
			String newCol = inputCols[i]+"2";
			result = result.withColumn(newCol, functions.regexp_replace(result.col(inputCols[i]),
					 "\\s+", "")).drop(inputCols[i]);

			String newCol2 = newCol.substring(0,newCol.length()-1)+"3";

			// Replacing empty strings in List Columns by the '0' index as String, so that 'NULL' value
			// in these columns is represented by the same index. After each step, we remove the previous
			// columns, so that we always end up with the exact columns we need to keep performing the
			// required transformations:
			result = result.withColumn(newCol2, functions.when(functions.col(newCol).equalTo(""), "0")
					.otherwise(functions.col(newCol))).drop(newCol);

			String newCol3 = newCol2.substring(0,newCol2.length()-1)+"4";
			RegexTokenizer tokenizer = new RegexTokenizer()
					.setInputCol(newCol2)
					.setOutputCol(newCol3)
					.setPattern(",");

			result = tokenizer.transform(result);

			result = result.drop(newCol2);

			String newCol4 = newCol3.substring(0,newCol3.length()-1)+"5";

			// We choose a CountVectorizer, instead of other vectorization approaches, as it ends up being
			// more explanatory than a Word2Vec approach, for instance, since we are not losing information
			// because of a summary function (such as the 'Mean' that is performed in Word2Vec) that is
			// applied over the word embedding vector that each different character or number is mapped to.

			// Furthermore, the choice of a Linear SVM that will be later on justified entails no problem
			// whatsoever when dealing with such a sparse and huge vector that the CountVectorizer outputs,
			// which reassures our decision:
			CountVectorizer cv = new CountVectorizer()
					.setInputCol(newCol3)
					.setOutputCol(newCol4)
					// After several checks, '1000' as vector vocabulary size ends up being sufficient to
					// get high accuracies:
					.setVocabSize(1000);

			CountVectorizerModel cvm = cv.fit(result);
			result = cvm.transform(result);

			result = result.drop(newCol3);
		}
		return(result);
	}


	public static Dataset<Row> featureSelection(Dataset<Row> source)
	{
		// Since we need to include in our vector variable all variables but the Response Variable, as well
		// as the 'Id' column that is added to every DataFrame in Spark, we use all columns except the
		// aforementioned ones, thereby getting an Array of Strings, as the expected argument for the function
		// 'VectorAssembler()'. Then, such an array can be passed as argument to perform the assembling into
		// a vector variable of all 'features' or 'predictors':
		// of distributed methods we find in Spark,
		String[] inputCols = Arrays.stream(source.toDF().columns())
				.filter(col -> !col.equals("BDiasEstancia") && !col.equals("Id")).toArray(String[]::new);

		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(inputCols)
				.setOutputCol("features");

		Dataset<Row> output = assembler.transform(source).select("features", "BDiasEstancia");
		return(output);
	}


	public static CrossValidatorModel fitModel(Dataset<Row> train)
	{
		// We determine that a Linear SVM is the best choice for this kind of dataset, since it is a dataset
		// with very high Dimensionality and reasonable Cardinality. Therefore, although the Linear SVM does
		// not include a Kernel to map the features to the Feature Space with infinite dimensionality (if
		// using the RBF Kernel Function) and determine the Optimal Separating Hyperplane there; it is actually
		// dealing with a very high dimensionality (more than 1000 features, as we used the CountVectorizer
		// with such a vocabulary vector to map the numbers of the list indicator variables, plus the rest
		// of the variables).

		// Consequently, using the existing dimensionality of the dataset and by means of the Hinge Loss
		// Function that the SVM uses to compute the OSH with the largest margin becomes enough to yield
		// a reliable classification of the 4 ranges of 'DiasEstancia', our Response Variable:
		LinearSVC lsvc = new LinearSVC()
				.setMaxIter(10)
				.setLabelCol("BDiasEstancia")
				.setFeaturesCol("features");

		OneVsRest ovr = new OneVsRest().setClassifier(lsvc)
				.setLabelCol("BDiasEstancia")
				.setFeaturesCol("features");

		// We decide to standardize as well, since some variables are measured in different scales, which
		// means that when performing the Optimal Separating Hyperplane by the SVM, one dimension might
		// dominate over the other ones if its scale is much higher. By standardizing, we ensure that this
		// is not going to happen and all variables will get the same relevance when it comes to classify
		// the observations with regard to the Response Variable:
		ParamMap[] paramGrid = new ParamGridBuilder()
				.addGrid(lsvc.regParam(), new double[]{100.0, 10.0, 1.0, 0.1, 0.01})
				.addGrid(lsvc.maxIter(), new int[]{20})
				.addGrid(lsvc.standardization())
				.build();

		CrossValidator cv = new CrossValidator()
				.setEstimator(ovr)
				.setEvaluator(new MulticlassClassificationEvaluator()
				.setMetricName("accuracy")
				.setLabelCol("BDiasEstancia")
				.setPredictionCol("prediction"))
				.setEstimatorParamMaps(paramGrid)
				.setNumFolds(5)
				.setParallelism(4);

		CrossValidatorModel cvModel = cv.fit(train);

      	return(cvModel);
	}

	public static double[] getIntervals(double metric, long n) {
		double pct_metric = Math.round(metric*10000.00)/100.00;
		double StdError = 1.967*Math.sqrt((metric*(1-metric))/n);
		return new double[]{pct_metric, Math.round((metric - StdError)*10000.00)/100.00,
				Math.round((metric + StdError)*10000.00)/100.00};
	}
	
	public static void classifyPatients(SparkSession sc) {

		  Dataset<Row> dataset = loadData(sc,"src/main/resources/PacientesSim.csv") ;	      
    
	      // Some Filters
		  dataset=dataset.filter("DiasEstancia <= 300");
		  dataset=dataset.filter("DiasEstancia > 0");
		  dataset.show(100);
		  
	      // Assign know values to  NA's (so we can bucketize them with a specific level and we can assign new levels to binary indicators)
		  dataset = naImputation(dataset);

		  dataset.show(5);

		  // Bucketization
		  dataset = bucketizeFeatures(dataset);

		  dataset.show(5);

		  // Checking indicator columns to determine if we can assign '0' index to 'NA' values
  		  dataset.filter("IndicadorConstipacion > -1000")
				.filter("IndicadorDemencia > -1000")
				.filter("IndicadorSordera > -1000")
				.filter("IndicadorAltVisual > -1000")
				.agg(functions.min("IndicadorDemencia"), functions.max("IndicadorDemencia"),
						functions.min("IndicadorConstipacion"), functions.max("IndicadorConstipacion"),
						functions.min("IndicadorSordera"), functions.max("IndicadorSordera"),
						functions.min("IndicadorAltVisual"), functions.max("IndicadorAltVisual")).show();

		  // One Hot encoding
		  dataset = oneHotEncoding(dataset);

		  dataset.show(5);

		  // Encode diagnosis lists and procedures with vector models
		  dataset = listsToVectors(dataset);

		  dataset.show(5);

		  // Feature selection
		  dataset = featureSelection(dataset);

		  dataset.show(5);
		  
	      // Split into train and test:
	      Dataset<Row>[] splits= dataset.randomSplit(new double[] {0.3,0.7}, 42);
	      Dataset<Row> train = splits[1];
	      Dataset<Row> test = splits[0];

	      // force train in memory if possible
	      train.persist();
	      // Fit model by CV
	      CrossValidatorModel cvModel = fitModel(train);
	      
	      // Make predictions 
	      Dataset<Row> predictions = cvModel.transform(test).select("prediction","BDiasEstancia");

	      // Evaluate and print metrics (general, by class and the confusion matrix)
		  MulticlassMetrics mm = new MulticlassMetrics(predictions);

		  System.out.println("Training samples: " + train.count());
		  System.out.println("Test samples: " + test.count() + "\n");

		  System.out.println("Global Accuracy of the model = " + Math.round(mm.accuracy()*10000.00)/100.00 + "%\n");
		  String[] ranges = {"for less than 12 days = ", "for between 12 and 41 days = ",
		  "for between 42 to 70 days = ", "for more than 70 days = "};

		  long n = test.count();
		  for (double i=1.0; i<5; i++){
		  	int j = (int)i-1;
		  	double[] precision = getIntervals(mm.precision(i), n);
		  	double[] recall = getIntervals(mm.recall(i), n);
		  	double[] f1score = getIntervals(mm.fMeasure(i), n);
		  	System.out.println("Precision " + ranges[j] + precision[0] + "%, with Confidence Intervals (95%):  (" +
					precision[1] + "%, " + precision[2] + "%)");
			System.out.println("Recall " + ranges[j] + recall[0] + "%, with Confidence Intervals (95%):  (" +
					recall[1] + "%, " + recall[2] + "%)");
			System.out.println("F1-Score " + ranges[j] + f1score[0] + "%, with Confidence Intervals (95%):  (" +
					f1score[1] + "%, " + f1score[2] + "%)\n");
		  }
		  System.out.println("Confusion Matrix: \n" + mm.confusionMatrix().toString());
		  
	      sc.stop();
	}
}
