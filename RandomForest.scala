package E2.codes

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.log4j._
    
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql._
import org.apache.spark.sql.types._


object RandomForest {
  
    def main(args: Array[String]) {
    
    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    // Use new SparkSession interface in Spark 2.0
    val spark = SparkSession
      .builder
      .appName("SparkSQL")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "file:///C:/temp") // Necessary to work around a Windows bug in Spark 2.0.0; omit if you're not on Windows.
      .getOrCreate()
      
    
     val struct =StructType(
       StructField("duration" , IntegerType, true) ::
       StructField("b", LongType, false) ::
       StructField("c", BooleanType, false) :: Nil) 
      
      
      
      
    }
  
  
}