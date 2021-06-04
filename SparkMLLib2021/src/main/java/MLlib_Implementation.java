

import org.apache.spark.sql.SparkSession;
import patients.patients;



public class MLlib_Implementation {
	public static void main(String[] args) throws Exception {
		SparkSession spark = SparkSession.builder().master("local[*]")
				.appName("EstanciasPacientes")
				.getOrCreate();
		patients.classifyPatients (spark);
	}

}
